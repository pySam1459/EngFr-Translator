import torch
from model import Translator, Config
from utils import  load_encoding
from dotenv import dotenv_values
from os.path import join


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    encoding = load_encoding()
    ckpt_name = "ckpt_2023-06-18_03,17,39.pt"
    save_path = dotenv_values(".env")["CKPT_PATH"]
    
    config = Config(
        vocab_size = encoding.n_vocab,
        context_length = 4096, # max context length
        d_model = 384,#*2,
        n_head = 6,
        n_layer = 6,
        dropout = 0.1
    )
    model = Translator(config, encoding)
    model.load_state_dict(torch.load(join(save_path, ckpt_name)))
    model = model.to(device)

    while True:
        inp_text = input(">")
        inp_tok = torch.tensor(encoding.encode(inp_text)).view(1, -1).to(device)
        out_tok = model.translate(inp_tok)
        out_text = encoding.decode(out_tok)
        print(out_text)


if __name__ == "__main__":
    main()
