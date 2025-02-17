#%% MHA: large KV cache => small batch_size => low QPS
# GPA: Small K, V cache => quick inference, small RAM
# MQA: Single KV => loss of performance

import math
import torch
import torch.nn as nn
# ignore attention_mask, attention_dropout
class GroupQueryAttention(nn.Module):
    def __init__(self, hidden_dim, nums_head,nums_key_value_head, dropout_rate=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert hidden_dim % nums_head == 0 
        assert nums_head % nums_key_value_head == 0
        self.hidden_dim = hidden_dim
        self.nums_head = nums_head
        self.nums_key_value_head = nums_key_value_head
        self.head_dim = hidden_dim // nums_head
        self.att_dropout = nn.Dropout(dropout_rate)

        self.q_proj = nn.Linear(hidden_dim, self.head_dim * nums_head)
        self.k_proj = nn.Linear(hidden_dim, self.head_dim * nums_key_value_head)
        self.v_proj = nn.Linear(hidden_dim, self.head_dim * nums_key_value_head)
        
        self.o_proj = nn.Linear(hidden_dim, hidden_dim) # input_size nums_head * head_dum
        
    
    def forward(self, X, attention_mask = None):
        # X: (batch_size, seq_len, hidden_dim)
        batch_size, seq, _ = X.size()

        # qkv projection
        q = self.q_proj(X)
        k = self.k_proj(X)
        v = self.v_proj(X)

        # attention_weight target shape is (batch_size, nums_head, seq, seq)
        q = q.view(batch_size, seq, self.nums_head, self.head_dim)
        k = k.view(batch_size, seq, self.nums_key_value_head, self.head_dim)
        v = v.view(batch_size, seq, self.nums_key_value_head, self.head_dim)

        # nums_head % nums_key_value_head == 0
        q = q.transpose(1, 2) # (batch_size, nums_head, seq, seq)
        k = k.transpose(1, 2) # (batch_size, nums_key_value_head, seq, seq)
        v = v.transpose(1, 2) # (batch_size, nums_key_value_head, seq, seq)

        # k v repeat: boardscast
        k = k.repeat_interleave(self.nums_head // self.nums_key_value_head, dim = 1)
        v = v.repeat_interleave(self.nums_head // self.nums_key_value_head, dim = 1)

        attention_score = (q @ k.transpose(2,3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attention_score = attention_score.masked_fill(
                attention_mask == 0,
                float("-inf")
            )

        attention_weight = torch.softmax(attention_score, dim=-1)
        attention_weight = self.att_dropout(attention_weight) # why is dropout happening after softmax?
        # softmax converts attention_weights into probabilities
        # dropout zeros out specific attention weights 
        # if dropout were applied before softmax, droppint some logits would cause the remaining ones to be amplified during re-normalization.
        # post-softmax dropout makes the gradient locally depends on dropout but globally if it is pre-softmax
        # pre-softmax dropout leads to non-zero softmax probabilities at masked positions
        
        output = attention_weight @ v 
        output = self.o_proj(output.view(batch_size, seq, -1))
        return output

X = torch.randn(3,2,128)
net = GroupQueryAttention(128, 8, 4)
net(X).shape
# %%
