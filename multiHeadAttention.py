import math
import torch
import torch.nn as nn

class MultiHeadSelfAttentionFormal(nn.Module):
    def __init__(self, hidden_dim, head_num, attention_dropout = 0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs) # initialize nn.Module
        self.hidden_dim = hidden_dim 
        self.head_num = head_num # head_num * head_dim = hidden_num
        self.head_dim = hidden_dim // head_num

        self.q_proj = nn.Linear(hidden_dim, hidden_dim) # (hidden_num, head_num * head_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attention_dropout = nn.Dropout(attention_dropout)

    def forward(self, X, attention_mask = None):
        # X: (b, s, h)
        batch_size, seq_len, _ = X.size()

        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X) # (b, s, h)

        # (b, s, h) => (b, head_num, s, head_dim) # h = head_num * head_dum
        q_state = Q.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1,2)
        k_state = K.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1,2)
        v_state = V.view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1,2)

        # (b, head_num, s, s)
        attention_weight = torch.matmul(
            q_state, k_state.transpose(-1,-2) # (b, head_num, s, head_dim) => (b, head_num, head_dim, s)
        ) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0,
                float("-inf")
            )
        print(attention_weight)
        attention_weight = torch.softmax(
            attention_weight,
            -1
        )
        attention_weight = self.attention_dropout(attention_weight) # prevent overfitting
        output_mid = torch.matmul(
            attention_weight, v_state
        ) # (b, head_num, s, head_dim) => (b, s, h)

        output_mid = output_mid.transpose(1,2).contiguous()
        output_mid = output_mid.view(batch_size, seq_len, self.hidden_dim)

        output = self.out_proj(output)
        return output
    




