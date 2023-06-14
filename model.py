import torch
import torch.nn as nn
import torch.nn.functional as F
from tiktoken import Encoding
from dataclasses import dataclass


device = "cuda" if torch.cuda.is_available() else "cpu"


def init_(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.05)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        torch.nn.init.normal_(m.weight, mean=0.0, std=0.05)


@dataclass
class Config:
    vocab_size: int
    context_length: int = 512
    d_model: int = 384
    n_head: int = 6
    n_layer: int = 6
    dropout: float = 0.1


class SelfAttention(nn.Module):
    def __init__(self, config: Config, is_causal: bool) -> None:
        super(SelfAttention, self).__init__()
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.dropout = config.dropout
        self.is_causal = is_causal
        
        ## head_size = n_embd // n_head
        ## single Head, q,k,v = nn.Linear(n_embd, head_size, bias=False)
        ## 3 * n_embd = |{q,k,v}| * (n_embd * head_size) * n_head
        self.attn = nn.Linear(self.d_model, 3 * self.d_model, bias=False)
        self.proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        self.resid_dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B,T,C = x.size()
        q, k, v = self.attn(x).split(self.d_model, dim=2) # nh: num head, hs: head size
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        y: torch.Tensor = F.scaled_dot_product_attention(q, k, v, 
            attn_mask=None, dropout_p=self.dropout, is_causal=self.is_causal)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y)) ## (B, T, C)
        return y


class CrossAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super(CrossAttention, self).__init__()
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.dropout = config.dropout
        
        self.c_attn = nn.Linear(self.d_model, self.d_model, bias=False)
        self.proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        B,T,C = x.size()
        v: torch.Tensor = self.c_attn(x)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y: torch.Tensor = F.scaled_dot_product_attention(q, k, v, 
            attn_mask=None, dropout_p=self.dropout, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.proj(y)) ## (B, T, C)
        return y
        

class FeedForward(nn.Module):
    """a simple Linear Layer followed by a non-linearity"""
    def __init__(self, config: Config) -> None:
        super(FeedForward, self).__init__()
        self.lin1 = nn.Linear(config.d_model, 4 * config.d_model) ## the 4* is from the paper
        self.lin2 = nn.Linear(4 * config.d_model, config.d_model) ## projection layer going back into residual pathway
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = F.gelu(x)
        x = self.lin2(x)
        return self.dropout(x)


class EncoderBlock(nn.Module):
    """Transformer Bock: comm followed by computation"""
    
    def __init__(self, config: Config) -> None:
        super(EncoderBlock, self).__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.sa = SelfAttention(config, is_causal=False)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffwd = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class Encoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super(Encoder, self).__init__()
        self.tke = nn.Embedding(config.vocab_size, config.d_model)
        self.pse = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model) ## std layer norm before final projection
        self.lin_q = nn.Linear(config.d_model, config.d_model)
        self.lin_k = nn.Linear(config.d_model, config.d_model)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B,T = x.size()
        tok_emb = self.tke(x) ## B,T,C
        pse_emb = self.pse(torch.arange(T, device=device))

        x = tok_emb + pse_emb # B,T,C (batch, context, channels)
        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        q = self.lin_q(x)
        k = self.lin_k(x)
        return q, k


class DecoderBlock(nn.Module):
    def __init__(self, config: Config) -> None:
        super(DecoderBlock, self).__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.causal_sa = SelfAttention(config, is_causal=True)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ca = CrossAttention(config)
        self.ln3 = nn.LayerNorm(config.d_model)
        self.ffwd = FeedForward(config)

    def forward(self, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        x = x + self.causal_sa(self.ln1(x))
        x = x + self.ca(self.ln2(x), q, k)
        x = x + self.ffwd(self.ln3(x))
        return x


class Decoder(nn.Module):
    def __init__(self, config: Config) -> None:
        super(Decoder, self).__init__()
        self.tke = nn.Embedding(config.vocab_size, config.d_model)
        self.pse = nn.Embedding(config.context_length, config.d_model)
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(self, x: torch.Tensor, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        B,T = x.size()
        tok_emb = self.tke(x)
        pse_emb = self.pse(torch.arange(T, device=x.device))
        x = tok_emb + pse_emb
        for block in self.blocks:
            x = block(x, q, k)

        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        return logits


class Translator(nn.Module):
    def __init__(self, config: Config, encoding: Encoding) -> None:
        super(Translator, self).__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        self.apply(init_) ## init weights
        
        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.encoding = encoding
    
    def forward(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        q, k = self.encoder(x)
        logits = self.decoder(idx, q, k)
        return logits

    def translate(self, x: torch.Tensor, max_len: int = 100, temperature: float = 1.0) -> torch.Tensor:
        ## x text to be translated
        ## batch dim = 1
        start_tok, end_tok = self.encoding.encode("<|start|><|endoftext|>", 
            allowed_special="all")
        idx = torch.tensor([start_tok], dtype=torch.long, device=device)
        for _ in range(max_len):
            logits = self.forward(x)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            
            if next_idx.item() == end_tok: break
            idx = torch.cat((idx, next_idx), dim=-1)
        
        return self.encoding.decode(idx[0].tolist())

    def save_ckpt(self, path: str) -> None:
        torch.save(self, path)