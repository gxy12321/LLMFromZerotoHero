# Build a miniMoE-LLM from scratch 
#%% 1. Basic version, to understand MOE
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicExpert(nn.Module):
    def __init__(self, feature_in, feature_out, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fc = nn.Linear(feature_in, feature_out)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        return x

class BasicMOE(nn.Module):
    def __init__(self, feature_in, feature_out, num_experts, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gate = nn.Linear(feature_in, num_experts)

        # output shape (batch_size, num_experts)
        self.experts = nn.ModuleList(
            BasicExpert(
                feature_in,
                feature_out
            ) for _ in range(num_experts)
        )
    
    def forward(self,x):
        # x shape is (batch, feature_in)
        # feature_in, also called hidden_size, hidden_dim

        expert_weights = self.gate(x) # (batch, num_experts)
        expert_out_list = [
            expert(x).unsqueeze(1) for expert in self.experts
        ] # every expert outputs (batch, 1, feature_out)

        expert_out = torch.concat(
            expert_out_list,
            dim = 1
        )# (batch, num_experts, feature_out)
        
        expert_weights = F.softmax(expert_weights, dim = 1)
        expert_weights = expert_weights.unsqueeze(1) # (batch, 1, num_experts)

        output = expert_weights @ expert_out
        return output.squeeze(1)

def test_basic_moe():
    x = torch.rand(4, 512)
    basic_moe = BasicMOE(512, 128, 4)
    output = basic_moe(x)
    print(output.shape)

test_basic_moe()
# %% SparseExpertMoE, MoE LLM, for current MOE LLM
# Different from Basic, MoE chooses topK experts and output weighted sum of different experts
# input shape (batch, seq_len, hidden_dim)
class MOEconfig:
    def __init__(
            self,
            hidden_dim,
            expert_number,
            top_k,
            shared_experts_number = 2
        ) -> None:
        self.hidden_dim = hidden_dim
        self.expert_number = expert_number
        self.top_k = top_k
        self.shared_experts_number = shared_experts_number

class SparseMOE(nn.Module):
    def __init__(self,config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.top_k = config.top_k
        self.hidden_dim = config.hidden_dim
        self.expert_num = config.expert_num

        self.experts = nn.ModuleList(
            BasicExpert(
                config.hidden_dim,
                config.hidden_dim
            ) for _ in range(config.expert_number)
        )

        self.router = None # TODO
    
    def forward(self,x):
        # x shape (batch, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()

        # calcualtion in token dimension , x reshape (batch * seq_len, expert_number)

        router_probs = F.softmax(router_)

