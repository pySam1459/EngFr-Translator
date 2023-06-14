import h5py
import numpy as np
from os import listdir
from os.path import join
from dotenv import dotenv_values
from tqdm import tqdm


## Even though cleaning already occurs in partition.py
##   if cleaning parameters wants to be changed, this script can be run
## Assumes that the partition.py script has already been run
base_path = dotenv_values("../.env")["DATASET_PATH"]
partition_path = join(base_path, "partitioned")

prog_bar = tqdm(listdir(partition_path))
for file in prog_bar:
    prog_bar.desc = f"{file}"
    with h5py.File(join(partition_path, file), "r") as f_in:
        en_raw = f_in["en"][:] ## load data into memory
        fr_raw = f_in["fr"][:]

        l = en_raw[0].shape[0] ## skip if empty
        if l == 0: continue
        
        min_len, max_len = int(l * 0.7), l * 5 ## min and max length of french sentences
        fr_lens = np.vectorize(len)(fr_raw)
        min_idx = np.searchsorted(fr_lens, min_len, side="left")
        max_idx = np.searchsorted(fr_lens, max_len, side="right")
        
        ## sentences outside of the min,max are removed
        en_clean = en_raw[min_idx:max_idx]
        fr_clean = fr_raw[min_idx:max_idx]

    ## data is written back to disk
    with h5py.File(join(partition_path, file), "w") as f_out:
        f_out.create_dataset("en",
                             shape=en_clean.shape,
                             dtype=h5py.vlen_dtype(int),
                             data=en_clean)
        f_out.create_dataset("fr",
                             shape=fr_clean.shape,
                             dtype=h5py.vlen_dtype(int),
                             data=fr_clean)
