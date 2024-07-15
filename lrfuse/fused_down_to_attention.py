import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import *
from svd_linear import SVDLinear

config = LlamaConfig()
down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.attention_bias)
post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

input_tensor = torch.randn(1, config.intermediate_size)

# Original
down_proj_output = down_proj(input_tensor)
# hidden_states = post_attention_layernorm(down_proj_output)
weight = post_attention_layernorm.weight
variance_epsilon = post_attention_layernorm.variance_epsilon

variance = down_proj_output.pow(2).mean(-1, keepdim=True)
hidden_states = down_proj_output * torch.rsqrt(variance + variance_epsilon)
hidden_states = weight * hidden_states
query_states = q_proj(hidden_states)

# --------------------------------------------------------------------------------------------

# Fused
down_proj_svd = SVDLinear.from_linear_rank_ratio(down_proj, 0.9999)
# hidden_states_svd = down_proj_svd(input_tensor)
BLinear = down_proj_svd.BLinear
ALinear = down_proj_svd.ALinear

B_out = BLinear(input_tensor)
hidden_states_svd = ALinear(B_out)

# hidden_states_svd = post_attention_layernorm(down_proj_output)
weight = post_attention_layernorm.weight
variance_epsilon = post_attention_layernorm.variance_epsilon

variance = hidden_states_svd.pow(2).mean(-1, keepdim=True)
hidden_states_svd = hidden_states_svd * torch.rsqrt(variance + variance_epsilon)
hidden_states_svd = weight * hidden_states_svd

query_states_svd = q_proj(hidden_states_svd)

assert torch.allclose(query_states, query_states_svd, rtol=1e-5, atol=1e-8), "Down_proj outputs are not close enough."
# assert torch.allclose(hidden_states, hidden_states_svd, rtol=1e-5, atol=1e-8), "Hidden states are not close enough."
# assert torch.allclose(query_states, query_states_svd, rtol=1e-5, atol=1e-8), "Query states are not close enough."

