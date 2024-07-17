
import torch
import utils
import tqdm
from typing import Optional
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import model_utils

import sys
sys.path.append('..')
from lrfuse.svd_linear import SVDLinear
from lr_llama import LRFusedLlamaDecoderLayer


def fuse_o_proj_U_into_up_gate(layer: LRFusedLlamaDecoderLayer,
                               model_type):
    if model_type == model_utils.LLAMA_MODEL:
        o_proj = layer.self_attn.o_proj
        up_proj = layer.mlp.up_proj
        gate_proj = layer.mlp.gate_proj
    else:
        raise ValueError(f'Not supported model type {model_type}')
    
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

    # Replaced with new linear layer
    layer.self_attn.o_proj = svd_layer.BLinear
    layer.mlp.up_proj = fused_up_proj
    layer.mlp.gate_proj = fused_gate_proj

    # Add UT to make residual correct
    UT_layer = torch.nn.Linear(U_layer.in_features, U_layer.out_features, bias=False)
    with torch.no_grad():
        UT_layer.weight.copy_(torch.linalg.inv(U_layer.weight))
    layer.o_proj_UT = UT_layer

def fuse_down_proj_U_into_qkv(layer: LRFusedLlamaDecoderLayer, 
                              prev_down_proj_U: Optional[torch.nn.Linear],
                              model_type):
    if model_type == model_utils.LLAMA_MODEL:
        down_proj = layer.mlp.down_proj
        q_proj = layer.self_attn.q_proj
        k_proj = layer.self_attn.k_proj
        v_proj = layer.self_attn.v_proj
    else:
        raise ValueError(f'Not supported model type {model_type}')
    
    svd_layer = SVDLinear.from_linear(down_proj, ratio=0.9999, sigma_fuse='V')
    U_layer = svd_layer.ALinear

    # Deal with first decoder layer (only do SVD but no fusing)
    if prev_down_proj_U is None:
        layer.mlp.down_proj = svd_layer.BLinear
        return U_layer

    # Fuse prev block down_proj U_layer into qkv
    # TODO: Implement fusion version with bias
    fused_q_proj = torch.nn.Linear(prev_down_proj_U.in_features, q_proj.out_features, bias=False)
    fused_k_proj = torch.nn.Linear(prev_down_proj_U.in_features, k_proj.out_features, bias=False)
    fused_v_proj = torch.nn.Linear(prev_down_proj_U.in_features, v_proj.out_features, bias=False)

    dtype = q_proj.weight.dtype
    W = prev_down_proj_U.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W1 = q_proj.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W2 = k_proj.weight.data.to(device=utils.DEV, dtype=torch.float64)
    W3 = v_proj.weight.data.to(device=utils.DEV, dtype=torch.float64)
    fused_q_proj.weight.data = torch.matmul(W1, W).to(device="cpu", dtype=dtype)
    fused_k_proj.weight.data = torch.matmul(W2, W).to(device="cpu", dtype=dtype)
    fused_v_proj.weight.data = torch.matmul(W3, W).to(device="cpu", dtype=dtype)

    # Replaced with new linear layer
    layer.mlp.down_proj = svd_layer.BLinear
    layer.self_attn.q_proj = fused_q_proj
    layer.self_attn.k_proj = fused_k_proj
    layer.self_attn.v_proj = fused_v_proj

    # Add UT to make residual correct
    UT_layer = torch.nn.Linear(U_layer.in_features, U_layer.out_features, bias=False)
    with torch.no_grad():
        UT_layer.weight.copy_(torch.linalg.inv(U_layer.weight))
    layer.down_proj_UT = UT_layer

    return U_layer

def decompose_and_fuse_model(model, args):
    model_type = model_utils.model_type_extractor(model)

    if model_type == model_utils.LLAMA_MODEL:
        target_module = LlamaDecoderLayer
    else:
        raise ValueError(f'Not supported model type {model_type}')
    
    # FIXME: check it (repalce decoder layer)
    model_utils.replace_modules(
        model,
        target_module,
        lambda module: LRFusedLlamaDecoderLayer.from_llama_decoder_layer(module, model.config),
        replace_layers=False,
    )

    prev_down_proj_U = None
    layers = model_utils.get_transformer_layers(model, model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Decomposing and fusing")):
        fuse_o_proj_U_into_up_gate(layer, model_type)
        # prev_down_proj_U = fuse_down_proj_U_into_qkv(layer, prev_down_proj_U, model_type)
        return
    
    # FIXME: Fused last down_proj_U into lm_head
    lm_head = model_utils.get_lm_head(model, model_type)
