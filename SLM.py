# Realization of GPT-2: https://miro.medium.com/v2/resize:fit:1400/1*YZTqlV51QyhX6VL9AV31eQ.png
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import dataloader
from dataclasses import dataclass

torch.manual_seed(1024)

@dataclass
class GPTConfig:
    block_size: int = 512 # max seq_len
    batch_size: int = 12
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768 # also called hidden_dim or hidden_size; 
    hidden_dim: int = n_embd
    dropout: float = 0.1
    head_size: int = n_embd // n_head
    # vocab_size
    # gpt2 officical tokenizer
    vocab_size: int = 50257

# 1. single head attention
class SingleHeadAttention(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.key = nn.Linear(config.hidden_dim, config.head_size)
        self.value = nn.Linear(config.hidden_dim, config.head_size)
        self.query = nn.Linear(config.hidden_dim, config.head_size)

        # use register_buffer to register attention_mask
        # because no need to calculate **gradien**, save video memory and RAM

        self.register_buffer(
            "attention_mask",
            # tril: lower triangle
            # block_size is 512
            torch.tril(
                torch.ones(config.block_size,config.block_size)
            )
        ) # <--- Automatically device-aware 
        # if attention_mask is a nn.parameter, it unnecessarily requires graduents, wasting memory and computation. But attention mask is a constant matrix, there is no need to calculate its gradient.
        self.dropout = nn.Dropout(config.dropout)

    def forward(self,x):
        # x shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weight = torch.matmul(q, k.transpose(-2,-1)) # @ is a simplified version of torch.matmul
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0,
            float("-inf")
        )

        weight = F.softmax(weight, dim = -1)/math.sqrt(self.head_size)

        # dropout after weight
        weight = self.dropout(weight)
        output = weight @ v
        return output

# 2. multi-head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heads = nn.ModuleList(
            [
                SingleHeadAttention(config)
                for _ in range(config.n_heads)
            ]
        )
        # we can also use matrix rotation to realize concatenation

        # interview question: how many weight matrices are there in a MHA?
        self.proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        output = torch.cat(
            [h(x) for h in self.heads],
            dim = -1
        )
        output = self.proj(output)
        output = self.dropout(output)
        return output

# 3. feed forward (MLP)
class FeedForward(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4), # increasing dimension to approximate the infinite basis of smooth projection
            nn.GELU(), # introduce non-linearity
            nn.Linear(config.hidden_dim * 4, config.hidden_dim ), # our target is to return a hidden_dim function so we project it back by adding up values of high-dimensional basis functions
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)

# 4. block
class Block(nn.Module): 
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.att = MultiHeadAttention(config) # mha
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)

    def forward(self,x):
        x += self.att(self.ln1(x))
        x += self.ffn(self.ln2(x))
        return x

# 5. GPT
class GPT(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # (embedding, position, norm, mlp, block)
        # position embedding: from 0, 1, ... to rope
        # norm layer -> rms norm 
        # mlp -> swiglu
        # mha -> gqa
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.tok_emb = nn.Embedding(config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_final = nn.LayerNorm(config.n_emb)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias = False)

        # current slm models use tie_weight to reduce the number of parameters
        self.token_embedding_table.weight = self.lm_head.weight # very important # linear (4 -> 8), actual weight shape is 8 * 4

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            # initialize into normal distribution
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx: token ids
        # targets is the target token ids
        # shape of idx and targets should be the same
        batch, seq_len = idx.size() # (batch, seq_len)
        token_emb = self.token_embedding_table(idx) # (batch, seq_len, n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(seq_len, device = idx.device) # Number 1 ... seq_len words are put into GPU
        ) # make sure they are on the same device

        # Classic interview question: why can we add token_emb to pos_emb
        x = token_emb + pos_emb # shape: (batch, seq_len, n_embd)
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x) # shape is (batch, seq_len, vocab_size)

        if targets is None:
            loss = None
        else: 
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            targets = targets.view(batch * seq_len)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        pass # TODO

# Dataset
# write a dataset, for the preparation of dataloader
class MyDataset(Dataset):
    def __init__(self, path, block_size = 512) -> None:
        import tiktoken
        self.enc = tiktoken.get_encoding()
        self.block_size = block_size # max length of pos

        self.encoded_data = []
        # a special character to sperate text
        # GPT: <|endoftext|> # [50256]
        self.eos_token = self.enc.encode(
            "<|endoftext|>",
            allowed_special = {"<|endoftext|>"}
        )[0]

        self.max_lines = 1000
        import json

        raw_data = [] # pre-process text and save it as .pkl or .npy and directly put it into GPU to increase the training efficiency
        with path(path, 'r') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                else:
                    try:
                        text = json.loads(line.strip())["text"]
                        raw_data.append(text)
                    except Exception as e:
                        continue

        full_encoded = []
        for text in raw_data:
            encoded_text = self.enc.encode(text) # list
            full_encoded.extend(encoded_text + [self.eos_token])
        
        # block size is 512 (max seq_len)
        # long text needs to be divided into short text: (512)
        for i in range(0, len(full_encoded), self.block_size):
            chunk = full_encoded[i: i+ self.block_size + 1] # 512 # shift 1 position to the right for the target, so actual row length is 513
            if len(chunk) < self.block_size + 1:
                chunk += [self.eos_token] * (self.block_size + 1 - len(chunk)) # padding eos_token to fill 512
            
            self.encoded_data.append(chunk)

        pass

        
