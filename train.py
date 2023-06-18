import torch
from torch.nn import functional as F
from torch.amp import autocast
from model import Translator, Config
from utils import *
from datetime import datetime
from os import mkdir
from os.path import exists, join
from dotenv import dotenv_values
from tqdm import tqdm
from time import perf_counter


# torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
# torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_config = dotenv_values(".env")


def save_ckpt(model: Translator, metadata: CKPTMetaData) -> None:
    ckpt_path = env_config["CKPT_PATH"]
    if not exists(ckpt_path):
        mkdir(ckpt_path)

    tid = datetime.now().strftime("%Y-%m-%d_%H,%M,%S")
    model.save_ckpt(join(ckpt_path, f"ckpt_{tid}.pt"))
    metadata.save(join(ckpt_path, "ckpt_{tid}.pkl"))


def main():
    encoding = load_encoding()
    config = Config(
        vocab_size = encoding.n_vocab,
        context_length = 4096, # max context length
        d_model = 384,#*2,
        n_head = 6,
        n_layer = 6,
        dropout = 0.1
    )
    batch_size = 64
    chunksize = 4
    ckpt_period = 512
    grad_clip = 1.0
    assert batch_size % chunksize == 0
    
    model = Translator(config, encoding)
    model = model.to(device)
    print(f"# of parameters: {model.n_params/1e6:.3f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = LRScheduler(optimizer, warmup_steps=1000, peak_lr=1e-4)
    
    ctx = autocast(device_type="cuda", dtype=torch.float16)
    scaler = torch.cuda.amp.GradScaler()
    
    loader = DataLoader(batch_size=batch_size, epochs=1)
    prog_bar = tqdm(enumerate(loader, start=1), total=loader.total)
    s1 = perf_counter()
    for i, (en, fr_0, fr_1) in prog_bar:
        s2 = perf_counter()
        optimizer.zero_grad(set_to_none=True)
        
        for k in range(0, batch_size, chunksize):
            with ctx:
                _, loss = model(en[k:k+chunksize], fr_0[k:k+chunksize], fr_1[k:k+chunksize])
                loss /= batch_size/chunksize
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        
        e1 = perf_counter()
        prog_bar.set_description(f"{loss.item():.7f} {s2-s1:.3f} {e1-s2:.3f}")
        s1 = e1
        if i % ckpt_period == 0: ## save checkpoint
            ckpt_metadata = CKPTMetaData(loader.c_epoch, loader.c_partition, optimizer.param_groups[0]["lr"])
            save_ckpt(model, ckpt_metadata)
        

if __name__ == "__main__":
    main()