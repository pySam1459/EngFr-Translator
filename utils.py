import torch
import numpy as np
import h5py
import pickle
from tiktoken import get_encoding, Encoding
from threading import Event as Event_t
from torch.multiprocessing import Event, Queue, Process
from random import shuffle, seed as r_seed
from dotenv import dotenv_values
from dataclasses import dataclass
from os import listdir
from os.path import join
from typing import Iterator


__all__ = [
    "load_encoding",
    "LRScheduler",
    "DataLoader",
    "CKPTMetaData"
    ]


def load_encoding() -> Encoding:
    _gpt2_encoding = get_encoding("gpt2")
    _gpt2_n_vocab = _gpt2_encoding.n_vocab
    return Encoding(
        name="gpt2_en_fr",
        pat_str=_gpt2_encoding._pat_str,
        mergeable_ranks=_gpt2_encoding._mergeable_ranks,
        special_tokens={
            **_gpt2_encoding._special_tokens,
            "<|start|>": _gpt2_n_vocab,
            "<|pad|>": _gpt2_n_vocab+1,
        },
        explicit_n_vocab=_gpt2_n_vocab+2
    )


@dataclass
class CKPTMetaData:
    epoch: int
    partition: str
    lr: float

    def save(self, file_path: str) -> None:
        with open(file_path, "wb") as f_out:
            pickle.dump(self, f_out)
    
    @staticmethod
    def load(file_path: str) -> "CKPTMetaData":
        with open(file_path, "rb") as f_in:
            return pickle.load(f_in)


class LRScheduler:
    """Linear warmup and inverse square root decay."""
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_steps: int, peak_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.peak_lr = peak_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.calculate_learning_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def calculate_learning_rate(self):
        if self.current_step < self.warmup_steps:
            return self.peak_lr * self.current_step / self.warmup_steps
        else:
            return self.peak_lr / (self.current_step ** 0.5)


class DataLoader:
    def __init__(self, epochs: int = 1, batch_size: int = 64):
        self.epochs = epochs
        self.batch_size = batch_size
        

        base_path = dotenv_values(".env")["DATASET_PATH"]
        self.partition_path = join(base_path, "partitioned")
        self.partitions = listdir(self.partition_path)
        self.partition_lengths = self.get_partition_lengths()

        self.total = self.epochs * (sum(self.partition_lengths.values()) // self.batch_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.c_epoch = -1
        self.c_partition = ""

    def get_partition_lengths(self) -> dict[str, int]:
        """Get the lengths of all partitions."""
        cache = {}
        for file in self.partitions:
            file_path = join(self.partition_path, file)
            with h5py.File(file_path, "r") as f_in:
                cache[file_path] = f_in["en"].shape[0]
        return cache
    
    def load_partition(self, file_path: str,
                       offset: int = 0, chunksize: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Load a partition (chunk) from disk."""
        length = self.partition_lengths[file_path]
        with h5py.File(file_path, "r") as f_in:
            if chunksize is None: ## load entire partition
                return f_in["en"][:], f_in["fr"][:]
            else: ## load chunk
                end = length if offset + chunksize > length else offset + chunksize
                return f_in["en"][offset:end], f_in["fr"][offset:end]
    
    def _next_partition_chunk(self, chunksize: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Loads a partition from disk and yields chunks of size chunksize."""
        for file in self.partitions:
            self.c_partition = file
            file_path = join(self.partition_path, file)
            part_length = self.partition_lengths[file_path]
            if part_length <= chunksize:
                yield self.load_partition(file_path)
            else:
                for offset in range(0, part_length, chunksize): ## yield part_length // chunksize chunks
                    yield self.load_partition(file_path, offset, chunksize)

    def next_partition(self, max_length: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """This function iterates through all partitions and 
            yields en-fr data of size max_partition_length."""
        k = 0
        chunksize = max_length * 32
        en_data = np.empty((max_length,), dtype=object)
        fr_data = np.empty((max_length,), dtype=object)
        for en_chunk, fr_chunk in self._next_partition_chunk(chunksize):
            part_len = len(en_chunk)
            if k + part_len <= max_length:
                en_data[k:k+part_len] = en_chunk
                fr_data[k:k+part_len] = fr_chunk
                k += part_len
                if k == max_length:
                    k = 0
                    yield (en_data, fr_data)
            elif k + part_len > max_length:
                offset = max_length - k
                en_data[k:] = en_chunk[:offset]
                fr_data[k:] = fr_chunk[:offset]
                yield (en_data, fr_data)
                
                n_sub = (part_len - offset) // max_length
                for i in range(n_sub):
                    a = offset + i*max_length
                    b = offset + (i+1)*max_length
                    yield (en_chunk[a:b], fr_chunk[a:b])
                
                k = max_length - k
                en_data[:k] = en_chunk[-k:]
                fr_data[:k] = fr_chunk[-k:]

    def score(self, x: torch.Tensor, a: float = 1.0, b: float = 1.0) -> torch.Tensor:
        """Guassian score function, sentences further away from target_idx are less likely to be selected."""
        return a * torch.exp(-b  * torch.pow(x, 2))

    def get_ix(self, length: int, indices: torch.Tensor) -> torch.Tensor:
        """select a random sentence to use to create a batch
            each sentence pair in the partition is given a probability
            depending on the random target_idx"""
        target_idx = torch.randint(0, length, (1, 1))
        relatives = (indices.view(1, -1) - target_idx).float()
        relatives /= torch.max(relatives, dim=1).values

        scores = self.score(relatives)
        probs = scores / scores.sum(dim=1, keepdims=True)
        return torch.multinomial(probs, self.batch_size, replacement=True).squeeze()

    def load_worker(self, out: Queue, next_batch: Event_t) -> None:
        """Child process worker function which loads batches of data 
            from disk and puts them in the out queue, 
            while the model is running on the GPU."""
        encoding = load_encoding()
        start_token, pad_token, end_token = torch.tensor(
            encoding.encode("<|start|><|pad|><|endoftext|>", allowed_special="all"), 
            dtype=torch.long).split(1)
        
        vec_len = np.vectorize(len)
        for epoch in range(self.epochs):
            self.c_epoch = epoch
            
            ## iterates over pre-processed dataset
            partition_iter = self.next_partition(self.batch_size * 64)
            while (data := next(partition_iter)) is not None:
                ## data below used to compute random-ish batches 
                ##   with sentences of approximately the same length
                en_data, fr_data = data
                fr_lens = vec_len(fr_data)
                unique = np.unique(fr_lens)
                indices = torch.from_numpy(np.searchsorted(fr_lens, unique))

                for _ in range(en_data.shape[0] // self.batch_size):
                    ## ix contains the indicies of the sentence pairs to use in the batch
                    ix = self.get_ix(en_data.shape[0], indices)
                    ens = [torch.from_numpy(en_data[i]) for i in ix]
                    frs = [torch.from_numpy(fr_data[i]) for i in ix]
                    en_length = max(ens, key=lambda x: x.shape[0]).shape[0]
                    fr_length = max(frs, key=lambda x: x.shape[0]).shape[0]
                    length = max(en_length, fr_length)

                    ## three tensors are in a batch
                    ##  0. en:   goes into the decoder, sentences to be translated
                    ##  1. fr_0: goes into the encoder, used to predict the next token
                    ##  2. fr_1: contains the correct token, used to calculate loss
                    ## special tokens are added at this stage
                    en_b = [torch.cat([en_d, pad_token.repeat(length-en_d.shape[0]+1)]) for en_d in ens]
                    fr_0 = [torch.cat([fr_d, end_token, pad_token.repeat(length-fr_d.shape[0])]) for fr_d in frs]
                    fr_1 = [torch.cat([start_token, fr_d, pad_token.repeat(length-fr_d.shape[0])]) for fr_d in frs]

                    ## batch is sent to device and put into out queue
                    en_b = torch.stack(en_b).pin_memory().to(self.device)#, non_blocking=True)
                    fr_0 = torch.stack(fr_0).pin_memory().to(self.device)#, non_blocking=True)
                    fr_1 = torch.stack(fr_1).pin_memory().to(self.device)#, non_blocking=True)

                    out.put((en_b, fr_0, fr_1))
                    next_batch.wait() ## wait for iterator to request next batch

        out.put(None) ## notify iterator that we're done

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Creates a deamon process to load batches in the background."""
        out_queue = Queue()
        next_batch = Event()
        
        load_process = Process(target=DataLoader.load_worker,
                               args=(self, out_queue, next_batch),
                               daemon=True)
        load_process.start()
        
        next_batch.set() ## prepare first batch
        while (result := out_queue.get()) is not None:
            next_batch.set() ## notify load_worker to load next batch
            yield result     ## batch to be used for training

        load_process.join()
