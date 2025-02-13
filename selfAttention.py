#%%
# different versions for self-attention questions in interviews
# equation attention(Q,K,V) = softmax(QK^T/sqrt(d_k))

import math 
import torch 
import torch.nn as nn


#%%
class SelfAttentopnV1(nn.Module):
    def __init__(self, hidden_dim: int = 720) -> None:
        super().__init__() # initialize nn.module
        self.hidden_dim = hidden_dim

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,X):
        # X shape is: (batch_size, seq_len, hidden_dim)
        Q = self.query_proj(X) # (batch_size, seq_len, hidden_dim)
        K = self.key_proj(X) # (batch_size, seq_len, hidden_dim)
        V = self.value_proj(X) # (batch_size, seq_len, hidden_dim)
        
        # attention_value is: (batch_size, seq, seq)
        attention_value = torch.matmul(
                Q, K.transpose(-1,-2) # the last two dimensions
        )
        
        # (batch_size, seq, seq)
        attention_weight = torch.softmax(
            attention_value/math.sqrt(self.hidden_dim), # divding the attention value by sqrt(hidden_dim) to prevent gradient vanishing
            dim = -1
        )

        # (batch_size, seq, hidden)
        output = torch.matmul(attention_weight,V)
        return output

X = torch.rand(3,2,4)
self_att_net = SelfAttentopnV1(4)     
output = self_att_net(X)
print(output)


#%%
# Efficiency optimization
# When Q, K, V is small, 
class SelfAttentionV2(nn.Module):
    def __init__(self, dim, *args, **kwargs) -> None:
        super().__init__() # initialize nn.Module
        self.dim = dim
        self.proj = nn.Linear(dim,dim*3)

    def forward(self,X):
        # X shape is (batch, seq, dim)
        # Q,K,V shape is (batch, seq, dim * 3)

        QKV = self.proj(X)
        Q, K, V = torch.split(QKV, self.dim, dim = -1)
        att_weight = torch.softmax(
            torch.matmul(
                Q, K.transpose(-1, -2)
            )/math.sqrt(self.dim),
            dim = -1
        )
        output = att_weight @ V
        return output
    
X = torch.rand(3, 2, 4)
self_att_net2 = SelfAttentionV2(4)
output = self_att_net2.forward(X)
print(output)

#%%
# Add some details
# 1. the location of dropout 
# 2. attention mask
# 3. output matrix projection

class SelfAttentionV3(nn.Module):
    def __init__(self, dim, dropout_rate=0.1) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim*3)

        self.attention_dropout = nn.Dropout(dropout_rate)
    
        self.output_proj = nn.Linear(dim,dim)
    
    def forward(self, X, attention_mask = None):
        # X: (batch_size, seq, dim)
        QKV = self.proj(X)

        Q, K, V = torch.split(QKV,self.dim, dim =-1)

        # (batch_size, seq, seq)
        attention_weight = Q @ K.transpose(-1,-2) / math.sqrt(self.dim)
        if attention_mask is not None: # multiply the masked elememts by a very small value
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0,
                float("-1e20")
            )
        attention_weight = torch.softmax(
            attention_weight,
            dim = -1
        )
        print(attention_weight)
        
        attention_weight = self.attention_dropout(attention_weight) # prevent overfitting
        attention_result = attention_weight @ V

        output = self.output_proj(attention_result) # for multihead attention, project output back to proper dimension after concatenation
        return output 
    
X = torch.rand(3,4,2)
# (batch, seq, seq)
mask = torch.tensor(
    [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0]
    ]
)
mask = mask.unsqueeze(dim = 1).repeat(1, 4, 1)
self_att_net3 = SelfAttentionV3(2)
output = self_att_net3.forward(X,mask)
print(output)
#%% self-attention interview
class SelfAttentionInterview(nn.Module):
    def __init__(self, dim : int, dropout_rate:float = 0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dim = dim 
        self.dropoutrate = dropout_rate

        self.query = nn.Linear(dim, dim) # bias: 1. shift activations 2. expressity 3. Handling sparsity
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

        self.attention_dropout = nn.Dropout(dropout_rate)

    def forward(self, X, attention_mask = None):
        # X shape is (batch, seq, dim)

        Q = self.query(X)
        K = self.key(X)
        V = self.value(X)

        attention_weight = Q @ K.transpose(-1,-2)/math.sqrt(self.dim)

        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0,
                float("-inf")
            )

        attention_weight = torch.softmax(
            attention_weight,
            dim = -1
        )
        print(attention_weight)

        attention_weight = self.attention_dropout(attention_weight)
        output = attention_weight @ V

        return output
    
X = torch.rand(3,4,2)

# mask is (batch, seq)
mask = torch.tensor(
    [
        [1, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 0],
    ]
)
# mask should be (batch, seq, seq)
mask = mask.unsqueeze(dim = 1).repeat(1,4,1)

self_att_net_interview = SelfAttentionInterview(2)

output = self_att_net_interview.forward(X,mask)
print(output)
# %%
