# Build a miniMoE-LLM from scratch 
#%% 1. Basic version, to understand MOE
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicExpert(nn.Module):
    def __init__(self, feature_in, feature_out, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.input_size = feature_in
        self.hidden_size = feature_out
        self.output_size = feature_out

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)
    def forward(self, x):
         
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
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
        
        expert_weights = F.softmax(expert_weights, dim = -1) # (batch, num_experts)
        expert_weights = expert_weights.unsqueeze(1) # (batch, 1, num_experts)

        output = expert_weights @ expert_out # (batch, 1, num_experts) @ (batch, num_experts, feature_out) = (batch, 1, feature_out)
        return output.squeeze(1) # (batch, feature_out)

def test_basic_moe():
    x = torch.rand(4, 512)
    basic_moe = BasicMOE(512, 128, 4)
    output = basic_moe(x)
    print(output.shape)

test_basic_moe()
# %% SparseExpertMoE, MoE LLM, for current MOE LLM
# Different from Basic, MoE chooses topK experts and output weighted sum of different experts
# input shape (batch, seq_len, hidden_dim)

# reference to: mistral MOE code
class MOEConfig:
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

class MOERouter(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gate = nn.Linear(config.hidden_dim, config.expert_number)

        self.top_k = config.top_k
        self.expert_number  = config.expert_number
    def forward(self, x):
        # assume the number of experts is 8, top_k is 2
        router_logits = self.gate(x) # (batch * seq_len, expert_number)

        # calculate the probabilty of an expert
        router_probs = F.softmax(router_logits, dim = 1, dtype = torch.float32)

        # calculate output of the top_k experts
        # top_k can be propagated backwards
        router_weights, selected_experts_indices = torch.topk(
            router_probs,
            self.top_k,
            dim = -1
        ) # router_weights, selected_experts_indices: (batch * seq_len, top_k)

        # re-normalization
        router_weights = router_weights / router_weights.sum(
            dim=-1,
            keepdim=True
        )

        router_weights = router_weights.to(x.dtype)

        expert_mask = F.one_hot(
            selected_experts_indices,
            num_classes=self.expert_number
        ) # (batch * seq_len, top_k, expert_number)
        # one hot vector: 1. easy to control loading balance by summing over the dimension batch * seq_len
        # 2. combine expected outputs: 

        expert_mask = expert_mask.permute(2, 1, 0)
        # (expert_number, top_k, batch * seq_len)

        return router_logits, router_weights, selected_experts_indices, expert_mask
        # router_logits: (batch * seq_len, expert_number)
        # router_weights: (batch * seq_len, top_k)
        # selected_experts_indices: (batch * seq_len, top_k)
        # expert_mask: (expert_number, top_k, batch * seq_len)

class SparseMOE(nn.Module):
    def __init__(self,config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.top_k = config.top_k
        self.hidden_dim = config.hidden_dim
        self.expert_number = config.expert_number

        self.experts = nn.ModuleList(
            BasicExpert(
                config.hidden_dim,
                config.hidden_dim # assume feature_in == feature_out
            ) for _ in range(config.expert_number)
        )

        self.router = MOERouter(config) # TODO
    
    def forward(self, x):
        # x shape (batch, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()

        # calcualtion in token dimension, x reshape (batch * seq_len, expert_number)
        hidden_states = x.view(-1, hidden_dim) #batch_size * seq_len, hidden_dim)

        # related to 

        router_logits, router_weights, selected_experts_indices, expert_mask = self.router(
            hidden_states
        )

        # expert_mask shape (expert_number, top_k, batch * seq_len)
        # output shape: (batch * seq_len, hidden_dim)
        final_hidden_states = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype= hidden_states.dtype,
            device=hidden_states.device
        )

        # iterate through each expert
        # add the hidden states of tokens of chosen expert to the final hidden_states
        for expert_idx in range(self.expert_number):
            expert_layer = self.experts[expert_idx]

            # expert_mask: (expert_number, top_k ,batch * seq_len)
            current_expert_mask = expert_mask[expert_idx]
            # current_expert_mask shape (top_k, batch * seq_len)

            router_weights_idx, top_x = torch.where(current_expert_mask)
            # (idx, top_x): input **top_x** is assigned to Expert **expert_idx** as the **idx** expert
            # where records row_idx and col_idx where mask == 1

            # hidden_states shape is (batch * seq_len, hidden_dim) before
            current_state = hidden_states.unsqueeze(0)[:, top_x, :].reshape(-1, hidden_dim)
            # current_state shape (selected_token_number, hidden_dim)

            # router_weights shape is (batch * seq_len, top_k)
            current_token_router_weight = router_weights[top_x, router_weights_idx]
            # current_token_router_weight is (selected_token_number)
            current_token_router_weight = current_token_router_weight.unsqueeze(-1)
            # current_token_router_weight is (selected_token_number, 1)

            current_hidden_states = expert_layer(current_state) * current_token_router_weight
            # (selected_token_number, hidden_dim) # boardcast here

            # element wise multiplication to scale the first dimension of current_state with current_token_router_weight

            final_hidden_states.index_add_(
                0,
                top_x,
                current_hidden_states.to(hidden_states.dtype)
            )
            # final_hidden_states[top_x] += current_hidden_states.to(hidden_states.dtype) # this way is okay but multiple read-write RAM when deal with duplicate indices

        # final_hidden_states to original shape
        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hidden_dim)
        return final_hidden_states,router_logits

# SharedExpert SparseMoE
# sharedexperts: this model is shared by all tokens, which means all tokens need to to through this model. The output is the weighted sum of this model and the output of top_k experts.
class ShareExpertMOE(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.routed_experts_moe = SparseMOE(config)
        self.shared_experts = nn.ModuleList(
            [
                BasicExpert(
                    config.hidden_dim, config.hidden_dim
                ) for _ in range(config.shared_experts_number)
            ]
        )

    def forward(self, x):
        # x shape is (b, s, hidden_dim)
        batch_size, seq_len, hidden_dim = x.size()

        shared_experts_output_list = [
            expert(x) for expert in self.shared_experts
        ]

        shared_expert_output = torch.stack(
            shared_experts_output_list,
            dim = 0
        )
        # shape (shared_experts_number, batch_size, seq_len, hidden_dim)
        print(shared_expert_output.size())
        shared_expert_out = shared_expert_output.sum(
            dim=0, 
            keepdim=False
        )
        # shape (batch_size, seq_len, hidden_dim)

        sparse_moe_out, router_logits = self.routed_experts_moe(
            x
        )

        output = shared_expert_out + sparse_moe_out
        return output, router_logits
    
def test_share_expert_moe():
    x = torch.rand(2, 4, 16)
    config = MOEConfig(16, 2, 2)
    share_expert_moe = ShareExpertMOE(config)
    out = share_expert_moe(x)
    print(out[0].shape, out[1].shape)


test_share_expert_moe() 
# %%
