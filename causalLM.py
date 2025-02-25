#%%
import torch
import math 
import torch.nn as nn

class SimpleDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, head_num, attention_dropout_rate=0.1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim//head_num

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # layer (mha, ffn)
        # mha
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)
        self.drop_att = nn.Dropout(attention_dropout_rate)
        self.att_ln = nn.LayerNorm(hidden_dim, eps = 0.0000001) # eps prevent divided by zero

        #ffn (increase dim -> decrease dim -> ln)
        self.up_proj = nn.Linear(hidden_dim, hidden_dim * 4) # (swishGLU, ) 8 / 3
        self.down_proj = nn.Linear(hidden_dim * 4, hidden_dim)
        self.act_fn = nn.GELU() # (ReLU)
        self.drop_ffn = nn.Dropout(0.1)
        self.ffn_ln = nn.LayerNorm(hidden_dim, eps = 0.0000001)

    def attention_layer(self, query, key, value, attention_mask = None):
        key = key.transpose(2,3) # (b, head_num, head_dim, seq)
        attention_weight = torch.matmul(query, key)/math.sqrt(self.head_dim) #(b, head_num, seq, seq)

        if attention_mask is not None:
            attention_mask = attention_mask.tril()
            attention_weight = attention_weight.masked_fill(
                attention_mask == 0,
                float("-inf")
            )
        else: 
            attention_mask = torch.ones_like(attention_weight).tril()
            attention_weight = attention_weight.masked_fill(
                attention_mask==0,
                float("-inf")
            )
        # why if-else:
        # some seq_len of input is less than block_size, so we need to use both padding mask and attention mask

        print(attention_weight)
        attention_weight = torch.softmax(attention_weight, dim=-1)

        attention_weight = self.drop_att(attention_weight)

        mid_out = torch.matmul(attention_weight, value) # (b, head_num, seq, head_dim)

        mid_out = mid_out.transpose(1,2).contiguous()   # (b, head_num, seq, head_dim)=> (b, seq, head_num, head_dim)
        batch, seq, _, _ = mid_out.size()
        mid_out = mid_out.view(batch, seq, self.hidden_dim)
        output = self.o_proj(mid_out)
        output = self.att_ln(output + self.drop_att) # residual
        # (b, s, h)
        return output

    def mha(self, X, mask = None):
        batch, seq, _ = X.size()
        # (b, s, h) => (b, s, head_num, head_dim)
        query = self.q_proj(X).view(batch, seq, self.head_num, -1).transpose(1,2)
        key = self.k_proj(X).view(batch, seq, self.head_num, -1).transpose(1,2)
        value = self.v_proj(X).view(batch, seq, self.head_num, -1).transpose(1,2)

        output = self.attention_layer(query, key, value,mask)
        return output

    def ffn(self, X):
        up = self.up_proj(X)
        up = self.act_fn(up)
        down = self.down_proj(up)
        down = self.drop_ffn(down)
        return self.ffn_ln(X + down)

    def forward(self, X, attention_mask = None):
        X = self.mha(X, attention_mask)
        X = self.ffn(X)
        return X
    
class Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer_list = nn.ModuleList(
            [
                SimpleDecoderLayer(64, 8) for i in range(5)
            ]
        ) 
        self.emb = nn.Embedding(12, 64)
        self.out = nn.Linear(64, 12)
    def forward(self, X, mask = None):
        # (b, s)
        X = self.emb(X)
        print(X.shape)
        for i, l in enumerate(self.layer_list):
            X = l(X, mask)
        print(X.shape)
        output = self.out(X)
        return torch.softmax(output, dim = -1)


X = torch.randint(low = 0, high = 12, size = (3,4))
net = Decoder()
mask = (
    torch.tensor(
        [
            [1,1,1,1],
            [1,1,0,0],
            [1,1,1,0],
        ]
    )
    .unsqueeze(1)
    .unsqueeze(2)
    .repeat(1, 8, 4, 1)
)
net(X, mask)

# %%
