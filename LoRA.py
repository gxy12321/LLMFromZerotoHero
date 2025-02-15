#%% split a large parameter matrix into two small matrices
# a large matrix that is not full-rank is redundant, which contains lots of unnecessary parameters
# Two small matrices are closer to full-rank so it is more efficient to optmized the parameters in two small matrices

import torch
import math
import torch.nn as nn

class LinearLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, lora_alpha, dropout, merge = False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs) # initialize nn.Module
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.dropout = dropout
        self.merge = merge

        self.linear = nn.Linear(in_features, out_features)

        # linear: weight is (out_features, in_features)
        # input x shape is (batch, seq_len, in_features)
        # calculation: x @ weight.T
        # so weight shape is (out_features, in_features)

        if rank>0:
            self.lora_a = nn.Parameter(torch.zeros(out_features, rank))
            # gaussian distribution
            nn.init.kaiming_normal_(self.lora_a, a = 0.01) # a is the negative slope of leaky relu usually 0.01-0.3

            self.lora_b = nn.Parameter(torch.randn(rank, in_features))
            self.scale = lora_alpha / rank # lora_alpha controls how much the low-rank updates influence the pre-trained weights
            self.linear.weight.requires_grad = False
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # merge is bool: if true, merge lora weights with linear weights
        if merge:
            self.merge_weight()


    def merge_weight(self):
        if self.merge and self.rank > 0:
            # (out_features, rank) @ (rank, in_features) = (out_features, in_features)
            self.linear.weight.data += self.scale * (self.lora_a @ self.lora_b)
        
    def unmerge_weight(self):
        if self.merge and self.rank > 0:
            self.linear.weight.data -= self.scale * (self.lora_a @ self.lora_b)

    def forward(self, X):
        # x shape is (batch_size, seq_len, in_features)
        if self.rank > 0:
            output_part1 = self.linear(X)
            output_part2 = self.scale * ( X @ (self.lora_a @ self.lora_b).T )
            output = output_part1 + output_part2 # this is very important. We realize LoRA by Y = X @ W_0^T + X @ (A@B).T instead of Y = X @ (W_0 + A @ B).T. If we merge weights before forwaring, LoRA needs more video memory.
            
        else:
            output = self.linear(X)
        
        output = self.dropout(output)
        return output
        
# test
batch_size = 32
seq_len = 128
in_features = 768
out_features = 512
rank = 8
lora_alpha = 16
dropout = 0.1

x = torch.randn(batch_size, seq_len, in_features)

lora_layer = LinearLoRALayer(
    in_features=in_features,
    out_features=out_features,
    rank=rank,
    lora_alpha=lora_alpha,
    dropout=dropout,
    merge=False
)

output = lora_layer(x)
print(output.shape)
lora_layer_merged = LinearLoRALayer(
    in_features=in_features,
    out_features=out_features,
    rank=rank,
    lora_alpha=lora_alpha,
    dropout=dropout,
    merge=True
)

output_merged = lora_layer_merged(x)
print(output_merged.shape)
print(f"output shape (merged): {output_merged.shape}")

lora_layer.merge_weight()
output_after_merge = lora_layer(x)
lora_layer.unmerge_weight()
output_after_unmerge = lora_layer(x)

print("Max difference agter merge/unmerge cycle:",
      torch.max(torch.abs(output_after_merge - output_after_unmerge)).item())
# %%
