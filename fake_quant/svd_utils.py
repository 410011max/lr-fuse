import model_utils
import torch
import utils
import tqdm

import sys
sys.path.append('..')
from lrfuse.svd_linear import SVDLinear


# @torch.inference_mode()
def fuse_o_proj_U_into_up_gate(layer, model_type):
    if model_type == model_utils.LLAMA_MODEL:
        o_proj = layer.self_attn.o_proj
        up_proj = layer.mlp.up_proj
        gate_proj = layer.mlp.gate_proj
    else:
        raise ValueError(f'Unknown model type {model_type}')
    
    svd_layer = SVDLinear.from_linear(o_proj, ratio=0.9999, sigma_fuse='V')
    U_layer = svd_layer.ALinear

    # TODO: Implement fusion version with bias
    fused_up_proj = torch.nn.Linear(U_layer.in_features, up_proj.out_features, bias=False)
    fused_gate_proj = torch.nn.Linear(U_layer.in_features, gate_proj.out_features, bias=False)

    dtype = up_proj.weight.dtype
    W = U_layer.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W1 = up_proj.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W2 = gate_proj.weight.data.to(device=utils.DEV, dtype=torch.float64)
    fused_up_proj.weight.data = torch.matmul(W1, W).to(device="cpu", dtype=dtype)
    fused_gate_proj.weight.data = torch.matmul(W2, W).to(device="cpu", dtype=dtype)

    layer.self_attn.o_proj = svd_layer.BLinear
    layer.mlp.up_proj = fused_up_proj
    layer.mlp.gate_proj = fused_gate_proj

# @torch.inference_mode()
def decompose_and_fuse_model(model, args):
    model_type = model_utils.model_type_extractor(model)
    layers = model_utils.get_transformer_layers(model, model_type)
    print(layers)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Decomposing and fusing")):
        fuse_o_proj_U_into_up_gate(layer, model_type)
        # return
        