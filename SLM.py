#%% Realization of GPT-2: https://miro.medium.com/v2/resize:fit:1400/1*YZTqlV51QyhX6VL9AV31eQ.png
import torch.nn as nn
import math
import torch
torch.autograd.set_detect_anomaly(True)
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

        weight = F.softmax(weight, dim = -1)/math.sqrt(hidden_dim)

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
                for _ in range(config.n_head)
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
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
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
        self.tok_emb = nn.Embedding(config.vocab_size,config.n_embd)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_final = nn.LayerNorm(config.n_embd)
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

    def forward(self, idx, target=None):
        # idx: token ids
        # target is the target token ids
        # shape of idx and target should be the same
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

        if target is None:
            loss = None
        else: 
            batch, seq_len, vocab_size = logits.size()
            logits = logits.view(batch * seq_len, vocab_size)
            target = target.view(batch * seq_len)
            loss = F.cross_entropy(logits, target)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx shape (batch, seq_len)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:] # only take the last block_size tokens if prompt is too long
            logits, _ = self(idx_cond)
            # shape (batch, seq_len, vocab_size)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim = -1)

            # random sampling
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim = -1)
        

# Dataset
# write a dataset, for the preparation of dataloader
from datasets import load_dataset
import tiktoken

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name: str, block_size: int = 512, max_lines: int = 1000) -> None:
        """
        Custom dataset class for tokenizing and structuring text data for LLM training.

        Args:
            dataset_name (str): Name of the Hugging Face dataset.
            block_size (int): Maximum sequence length (default: 512).
            max_lines (int): Maximum number of lines to process (default: 1000).
        """
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size  
        self.max_lines = max_lines
        self.eos_token = self.enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        
        self.encoded_data = []  # Stores processed prompt-response pairs
        self._load_and_process_dataset(dataset_name)

    def _load_and_process_dataset(self, dataset_name: str) -> None:
        """Loads, processes, and tokenizes dataset into fixed-size prompt-response pairs."""
        dataset = load_dataset(dataset_name, split='train')

        def format_example(example):
            """Formats each example into a structured prompt-response pair."""
            prompt = (
                f"Problem: {example['problem']}\n"
                f"Sub-domain: {example['sub_domain']}\n"
                f"Main domain: {example['main_domain']}\n"
                f"Model: {example['model_name']}\n\n"
                f"Provide a solution:\n"
            )
            response = example['solution']
            return {"prompt": prompt, "response": response}
        
        dataset = dataset.map(format_example)
        dataset = dataset.remove_columns(["problem", "sub_domain", "main_domain", "model_name", "solution_model_name"])

        for i, example in enumerate(dataset):
            if i >= self.max_lines:
                break
            if not example["response"]:
                continue  # Skip empty responses

            prompt_encoded = self.enc.encode(example["prompt"]) + [self.eos_token]
            response_encoded = self.enc.encode(example["response"]) + [self.eos_token]

            # Split into fixed-size chunks
            for i in range(0, len(prompt_encoded), self.block_size):
                prompt_chunk = prompt_encoded[i : i + self.block_size]
                response_chunk = response_encoded[i : i + self.block_size]

                if len(prompt_chunk) < self.block_size:
                    prompt_chunk = prompt_chunk + [self.eos_token] * (self.block_size - len(prompt_chunk))  # Pad with EOS
                if len(response_chunk) < self.block_size:
                    response_chunk = response_chunk [self.eos_token] * (self.block_size - len(response_chunk))  # Pad with EOS

                self.encoded_data.append({"prompt": prompt_chunk, "response": response_chunk})

    def __len__(self) -> int:
        return len(self.encoded_data)

    def __getitem__(self, idx: int):
        data_pair = self.encoded_data[idx]
        x = torch.tensor(data_pair["prompt"], dtype=torch.long)  # Input (prompt)
        y = torch.tensor(data_pair["response"], dtype=torch.long)  # Target (response)
        return x, y

    def encode(self, text: str):
        """Encodes raw text into token IDs."""
        return self.enc.encode(text)

    def decode(self, ids: list):
        """Decodes token IDs back into text."""
        return self.enc.decode(ids)



#%% Training

model = GPT(GPTConfig())
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# print the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params / 1e6} M")

optimizer = torch.optim.AdamW(model.parameters(), lr = 3e-4) # AdamW: Adam with weight decay regularization
# set the cosine learning rate
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 1000) # cos annealing to decrease the learning rate slowly in a cosine curve. Balance between exploration and exploitation. Avoid local Minima

# load data from hugging face
from torch.utils.data import DataLoader
from datasets import load_dataset
dataset_name = "BoltzmannEntropy/QuantumLLMInstruct"

my_dataset = MyDataset(dataset_name, block_size=512, max_lines=1000)

# Create a DataLoader
train_dataset, val_dataset = torch.utils.data.random_split(my_dataset,[0.9,0.1])
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=True)



def train(model, optimizer, scheduler, train_loader, val_loader, device):
    model.train()
    total_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        # move data to device
        x, y = x.to(device), y.to(device)

        # propagate forward
        logits, loss = model(x, target = y)

        # propagate backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # change the learning rate
        scheduler.step()

        total_loss = total_loss + loss.item()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
        return total_loss

def eval(model, val_loader, device):
    # validation
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, target = y)
            val_loss = val_loss + loss.item() 

    return val_loss

for epoch in range(1000):
    train_loss = train(model, optimizer, scheduler, train_loader, val_loader,device)
    val_loss = eval(model, val_loader, device)
    print(f'Epoch: {epoch}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

    # save model
    avg_val_loss = val_loss / len(val_loader)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': avg_val_loss,
    }

# save model at the end 
torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')






# %%
