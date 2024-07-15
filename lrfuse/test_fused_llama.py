import torch
import pytest
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaAttention, LlamaMLP, LlamaDecoderLayer
from svd_linear import SVDLinear
from fused_llama import FusedLlamaAttention, FusedLlamaMLP, FusedLlamaDecoderLayer



@pytest.fixture
def config():
    return LlamaConfig()

@pytest.fixture
def bs():
    return 1

@pytest.fixture
def q_len():
    return 128

@pytest.fixture(autouse=True)
def random_seed():
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

def test_fused_llama_mlp_forward_pass(config, bs):
    mlp = FusedLlamaMLP(config)
    input_tensor = torch.randn(bs, config.hidden_size)
    output = mlp(input_tensor)
    assert output.shape == (bs, config.hidden_size)

def test_llama_mlp_fused_vs_original(config, bs):
    input_tensor = torch.randn(bs, config.hidden_size)
    mlp = FusedLlamaMLP(config)
    original_output = mlp(input_tensor)
    mlp.svd_decomposition(0.9)
    fused_output = mlp(input_tensor)

    assert original_output.shape == fused_output.shape, "Shape mismatch between original and fused MLP outputs"
    torch.testing.assert_close(original_output, fused_output, rtol=1e-5, atol=1e-8)

# 運行測試
if __name__ == "__main__":
    pytest.main()
