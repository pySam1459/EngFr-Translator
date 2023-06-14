import torch
from torch.nn import functional as F
from model import Translator, Config
from utils import *
from datetime import datetime


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    ckpt_period = 1000
    
    model = Translator(config, encoding)
    model = model.to(device)
    print(f"# of parameters: {model.n_params/1e6:.3f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    loader = DataLoader(batch_size=64, epochs=1)
    for i, (en, fr_0, fr_1) in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        
        logits = model(en, fr_0)
        loss = F.cross_entropy(logits, fr_1)
        
        loss.backward()
        optimizer.step()

        loader.set_desc(f"{loss.item():.4f}")
        
        if i % ckpt_period == 0: ## save checkpoit
            tid = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            model.save_ckpt(f"D:/models/Translators/ckpt_{tid}.pt")
            CKPTMetaData(
                epoch = loader.c_epoch,
                partition = loader.c_partition,
                lr = optimizer.param_groups[0]["lr"],
            ).save(f"D:/models/Translators/ckpt_{tid}.json")


if __name__ == "__main__":
    main()