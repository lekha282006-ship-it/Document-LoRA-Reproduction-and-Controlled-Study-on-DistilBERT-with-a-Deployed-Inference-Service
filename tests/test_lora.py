"""Tests for custom LoRA implementation against specs and Hugging Face PEFT."""

import copy
import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from src.lora import LoRALinear, apply_lora_to_model, count_parameters


@pytest.fixture
def base_linear():
    torch.manual_seed(42)
    return nn.Linear(in_features=768, out_features=768, bias=True)


@pytest.fixture
def sample_input():
    torch.manual_seed(42)
    return torch.randn(4, 32, 768)


def test_zero_init(base_linear, sample_input):
    """Zero init test: a fresh LoRALinear gives the same output as original nn.Linear."""
    original_linear = copy.deepcopy(base_linear)
    lora_layer = LoRALinear(base_linear=copy.deepcopy(base_linear), r=8, lora_alpha=16.0)

    orig_out = original_linear(sample_input)
    lora_out = lora_layer(sample_input)

    diff = torch.max(torch.abs(orig_out - lora_out)).item()
    assert diff == 0.0 or diff < 1e-7, f"Fresh LoRA altered output: max abs diff = {diff}"


def test_shapes(base_linear, sample_input):
    """Shape test: output shape matches input, A and B have expected shapes."""
    in_features, out_features, r = 768, 768, 4
    lora_layer = LoRALinear(base_linear=base_linear, r=r, lora_alpha=8.0)

    assert lora_layer.lora_A.shape == (r, in_features), (
        f"Expected A shape {(r, in_features)}, got {lora_layer.lora_A.shape}"
    )
    assert lora_layer.lora_B.shape == (out_features, r), (
        f"Expected B shape {(out_features, r)}, got {lora_layer.lora_B.shape}"
    )

    out = lora_layer(sample_input)
    assert out.shape == sample_input.shape, (
        f"Expected output shape {sample_input.shape}, got {out.shape}"
    )

    # 2D input test
    x_2d = torch.randn(8, in_features)
    out_2d = lora_layer(x_2d)
    assert out_2d.shape == (8, out_features)


def test_merge(base_linear, sample_input):
    """Merge test: outputs before and after merge() match within 1e-5."""
    lora_layer = LoRALinear(base_linear=base_linear, r=8, lora_alpha=16.0)

    # Populate B with non-zero values so adapter has non-trivial effect
    torch.manual_seed(123)
    nn.init.normal_(lora_layer.lora_B, std=0.02)

    lora_layer.eval()
    with torch.no_grad():
        out_before = lora_layer(sample_input)
        lora_layer.merge()
        assert lora_layer.merged is True
        out_after = lora_layer(sample_input)

    max_diff = torch.max(torch.abs(out_before - out_after)).item()
    assert max_diff < 1e-5, f"Outputs before and after merge differed by {max_diff} (exceeds 1e-5)"

    # Test unmerge returns to separate state and matches
    with torch.no_grad():
        lora_layer.unmerge()
        assert lora_layer.merged is False
        out_unmerged = lora_layer(sample_input)

    unmerge_diff = torch.max(torch.abs(out_before - out_unmerged)).item()
    assert unmerge_diff < 1e-5, f"Outputs after unmerge differed by {unmerge_diff}"


def test_library_peft_equivalence(sample_input):
    """Equivalence check: scratch LoRALinear matches Hugging Face PEFT on identical weights."""
    in_dim, out_dim, r, alpha = 768, 768, 4, 8.0
    torch.manual_seed(999)

    base = nn.Linear(in_dim, out_dim, bias=True)
    custom_lora = LoRALinear(base_linear=copy.deepcopy(base), r=r, lora_alpha=alpha)

    # Populate non-zero A and B for non-trivial verification
    torch.manual_seed(777)
    nn.init.normal_(custom_lora.lora_A, std=0.02)
    nn.init.normal_(custom_lora.lora_B, std=0.02)

    class DummyContainer(nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.lin = lin

        def forward(self, x):
            return self.lin(x)

    dummy = DummyContainer(copy.deepcopy(base))
    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=["lin"],
    )
    peft_model = get_peft_model(dummy, peft_config)

    # Sync weights exactly between our layer and PEFT
    with torch.no_grad():
        peft_model.lin.lora_A["default"].weight.copy_(custom_lora.lora_A.data)
        peft_model.lin.lora_B["default"].weight.copy_(custom_lora.lora_B.data)

    custom_lora.eval()
    peft_model.eval()

    with torch.no_grad():
        custom_out = custom_lora(sample_input)
        peft_out = peft_model(sample_input)

    max_diff = torch.max(torch.abs(custom_out - peft_out)).item()
    assert max_diff < 1e-5, f"Scratch LoRA vs HF PEFT mismatch: max abs diff = {max_diff}"
