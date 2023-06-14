import torch
import numpy as np
import h5py
import pickle
from tiktoken import get_encoding, Encoding
from threading import Event as Event_t
from multiprocessing import Event, Queue, Process
from random import shuffle, seed as r_seed
from dotenv import dotenv_values
from dataclasses import dataclass
from os import listdir
from os.path import join
from tqdm import tqdm
from typing import Iterator


__all__ = [
    "load_encoding",
    "LRScheduler",
    "DataLoader",
    "CKPTMetaData"]


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

        self.total = self.get_total()
        self.prog_bar = tqdm(total=self.total)
        self.encoding = load_encoding()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.c_epoch = -1
        self.c_partition = ""

    def get_total(self) -> int:
        total = 0
        for file in self.partitions:
            with h5py.File(join(self.partition_path, file), "r") as f_in:
                total += f_in["en"].shape[0]

        total = self.epochs * (total // self.batch_size)
        return total
    
    def load_partition(self, file_path: str) -> tuple[np.ndarray, np.ndarray]:
        with h5py.File(file_path, "r") as f_in:
            return f_in["en"][:], f_in["fr"][:]

    def score(self, x: torch.Tensor, a: float = 1.0, b: float = 1.0) -> torch.Tensor:
        return a * torch.exp(-b  * torch.pow(x / x[:, -2:-1], 2))

    def load_worker(self, out: Queue, next_batch: Event_t) -> None:
        print("Loading worker started")
        vec_len = np.vectorize(len)
        special_tokens = torch.tensor(
            self.encoding.encode("<|start|><|pad|><|endoftext|>", allowed_special="all"), 
            dtype=torch.long)
        start_token, pad_token, end_token = special_tokens.split(1)

        r_seed(1459) ## set seed for partition shuffling
        for epoch in range(self.epochs):
            self.c_epoch = epoch
            shuffle(self.partitions) ## shuffle order of files
            
            for file in self.partitions:
                self.c_partition = file

                en_data, fr_data = self.load_partition(join(self.partition_path, file))
                fr_lens = vec_len(fr_data)
                unique = np.unique(fr_lens)
                indices = torch.from_numpy(np.searchsorted(fr_lens, unique))

                l = en_data.shape[0]
                n_batches = l // self.batch_size
                
                for _ in range(n_batches):
                    ## select a random sentence to use to create a batch
                    ## each sentence pair in the partition is given a probability
                    ##   depending on the random target_idx
                    target_idx = torch.randint(0, l, (1, 1))
                    relatives = indices.view(1, -1) - target_idx

                    scores = self.score(relatives)
                    probs = scores / scores.sum(axis=1, keepdims=True)

                    ## ix contains the indicies of the sentence pairs to use in the batch
                    ix = torch.multinomial(probs, self.batch_size, replacement=True).squeeze()
                    en = torch.stack([torch.from_numpy(en_data[i]) for i in ix])
                    
                    frs = [torch.from_numpy(fr_data[i]) for i in ix]
                    max_len = max(frs, key=lambda x: x.shape[0])
                    fr_0 = torch.stack([torch.cat([fr_d, end_token] + [pad_token] * (max_len-fr_d.shape[0])) for fr_d in frs])
                    fr_1 = torch.stack([torch.cat([start_token, fr_d] + [pad_token] * (max_len-fr_d.shape[0])) for fr_d in frs])
    
                    out.put((en, fr_0, fr_1))
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
        
        self.prog_bar.reset()
        next_batch.set() ## prepare first batch
        while (result := out_queue.get()) is not None:
            next_batch.set() ## notify load_worker to load next batch
            yield result     ## batch to be used for training
            self.prog_bar.update(1)

        load_process.join()

    def set_desc(self, desc: str) -> None:
        self.prog_bar.set_description(desc)