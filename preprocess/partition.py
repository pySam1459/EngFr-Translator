import numpy as np
import h5py
from tqdm import tqdm
from os import mkdir, remove
from os.path import join, exists
from dotenv import dotenv_values


base_path = dotenv_values("../.env")["DATASET_PATH"]
total = 22520317
chunksize = 10000 ## reduces number of reads from h5 file
pre_calculated = True

print("start...")
if not pre_calculated:
    en_lens = np.empty((total,), dtype=np.int32)
    ## calculate the lengths of each english tokenized sentence
    with h5py.File(join(base_path, "en-fr_tokenized.h5"), "r") as f_in, open(join(base_path, "en_length.npy"), "wb") as f_out:
        data = f_in["en"]

        k = total // chunksize ## chunking helps with performance and mem usage
        for j in tqdm(range(k), desc="Calc Lengths"):
            en_lens[j*chunksize:(j+1)*chunksize] = np.vectorize(len)(data[j*chunksize:(j+1)*chunksize])

        en_lens[k*chunksize:] = np.vectorize(len)(data[k*chunksize:]) ## remaining
        np.save(f_out, en_lens) ## save to disk
    print("Lengths calculated..")

else:
    with open(join(base_path, "en_length.npy"), "rb") as f_en:
        en_lens = np.load(f_en)


en_unique, en_counts = np.unique(en_lens, return_counts=True)
## en_data, fr_data contains the tokenized english, french sentences grouped by length
en_data = {u: np.empty((l, ), dtype=object) for u, l in zip(en_unique, en_counts)}
fr_data = {u: np.empty((l, ), dtype=object) for u, l in zip(en_unique, en_counts)}
curr = {u: 0 for u in en_unique} ## keeps track of current index for each unique length

with h5py.File(join(base_path, "en-fr_tokenized.h5"), "r") as f_in:
    def __iter(start: int, stop: int) -> None:
        for en_arr, fr_arr in zip(f_in["en"][start:stop], f_in["fr"][start:stop]):
            l = len(en_arr)
            en_data[l][curr[l]] = en_arr
            fr_data[l][curr[l]] = fr_arr
            curr[l] += 1

    d, m = divmod(total, chunksize)
    for j in tqdm(range(d), desc="Grouping"):
        __iter(j*chunksize, (j+1)*chunksize)

    __iter(d*chunksize, d*chunksize + m) ## iter over remaining


if not exists(join(base_path, "partitioned")):
    mkdir(join(base_path, "partitioned"))

remove_list = []
prog_bar = tqdm(total=total, desc="Sorting and Saving")
for i, u in enumerate(en_unique):
    with h5py.File(join(base_path, "partitioned", f"{u}.h5"), "w") as f_out:
        ## sort by french sentence length, makes batching easier
        fr_lens = np.vectorize(len)(fr_data[u])
        sorted_idx = np.argsort(fr_lens)
        en_data[u] = en_data[u][sorted_idx]
        fr_data[u] = fr_data[u][sorted_idx]
        
        ## eye-balled fr thresholds for higher quality sentence pairs
        ## assumption: if a sentence is too short/too long, it's probably not a good translation
        min_len, max_len = int(u * 0.7), u * 5
        en_lens = np.vectorize(len)(en_data[u])[sorted_idx]
        fr_lens = fr_lens[sorted_idx]
        min_idx = np.searchsorted(fr_lens, min_len, side="left")
        max_idx = np.searchsorted(fr_lens, max_len, side="right")
        en_data[u] = en_data[u][min_idx:max_idx]
        fr_data[u] = fr_data[u][min_idx:max_idx]
        
        if fr_data[u].shape[0] > 0: ## check if there are any sentences left
            f_out.create_dataset(f"en",
                                shape=en_data[u].shape,
                                dtype=h5py.vlen_dtype(int),
                                data=en_data[u])
            f_out.create_dataset(f"fr",
                                shape=fr_data[u].shape,
                                dtype=h5py.vlen_dtype(int),
                                data=fr_data[u])
        else:
            remove_list.append(u)

    prog_bar.total = prog_bar.total - (en_counts[i] - en_data[u].shape[0])
    prog_bar.update(en_data[u].shape[0])
    prog_bar.refresh()
    del en_data[u]
    del fr_data[u]

## remove empty files
for u in remove_list:
    remove(join(base_path, "partitioned", f"{u}.h5"))
