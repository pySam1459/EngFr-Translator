import pandas as pd
import h5py
from os.path import join
from tqdm import tqdm
from utils import load_encoding
from dotenv import dotenv_values


encoding = load_encoding()
base_path = dotenv_values("../.env")["DATASET_PATH"]
filename = "en-fr.csv"

k = 0
total = 22520317
train, val = int(total * 0.9), int(total * 0.1)

with h5py.File(join(base_path, "train.h5"), "w") as f:
    en_data = f.create_dataset("en", (total,), dtype=h5py.vlen_dtype(int))
    fr_data = f.create_dataset("fr", (total,), dtype=h5py.vlen_dtype(int))

    chunksize = 100000
    for chunk in tqdm(pd.read_csv(join(base_path, filename), chunksize=chunksize)):
        chunk: pd.DataFrame = chunk.dropna().astype(str)
        chunk["en"] = chunk["en"].apply(encoding.encode)
        chunk["fr"] = chunk["fr"].apply(encoding.encode)
        
        en_data[k:k+chunk.shape[0]] = chunk["en"].values
        fr_data[k:k+chunk.shape[0]] = chunk["fr"].values
        k += chunk.shape[0]
