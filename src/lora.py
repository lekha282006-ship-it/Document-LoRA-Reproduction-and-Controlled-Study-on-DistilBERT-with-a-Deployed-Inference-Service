"""LoRA (Low-Rank Adaptation) layer implemented from scratch.

Reference:
    Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021).
    h = W_0 x + (alpha / r) * B A x
    where:
      - W_0 in R^{d_out x d_in} is frozen.
      - A in R^{r x d_in} is initialized with Kaiming / normal distribution.
      - B in R^{d_out x r} is initialized to zeros (adapter starts as a no-op).
      - alpha / r is a constant scaling factor.
"""

import math
from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Wraps an existing nn.Linear, freezes it, and injects low-rank A and B matrices."""

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
    ) -> None:
        """Initialize LoRALinear layer.

        Args:
            base_linear: Pre-existing nn.Linear to adapt.
            r: Rank of the low-rank decomposition (r > 0).
            lora_alpha: LoRA scaling factor alpha.
            lora_dropout: Dropout probability applied to inputs before matrix A.
        """
        super().__init__()
        if r <= 0:
            raise ValueError(f"LoRA rank r must be positive, got {r}")

        self.base_linear = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.lora_alpha = float(lora_alpha)
        self.scaling = self.lora_alpha / float(self.r)

        # Freeze base linear layer
        self.base_linear.weight.requires_grad = False
        if self.base_linear.bias is not None:
            self.base_linear.bias.requires_grad = False

        # LoRA trainable low-rank weights
        # A: (r, in_features), B: (out_features, r)
        self.lora_A = nn.Parameter(torch.empty((r, self.in_features)))
        self.lora_B = nn.Parameter(torch.empty((self.out_features, r)))

        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        self.merged = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize LoRA parameters.

        A is initialized with Kaiming uniform (matching PyTorch/PEFT standard).
        B is initialized to zeros so the initial forward pass is an exact no-op.
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def merge(self) -> None:
        """Merge low-rank weights (alpha/r * B @ A) into base_linear.weight for inference."""
        if self.merged:
            return
        # W_merged = W_0 + scaling * (B @ A)
        weight_update = self.scaling * (self.lora_B @ self.lora_A)
        self.base_linear.weight.data.add_(weight_update)
        self.merged = True

    def unmerge(self) -> None:
        """Unmerge low-rank weights from base_linear.weight back to separate representations."""
        if not self.merged:
            return
        weight_update = self.scaling * (self.lora_B @ self.lora_A)
        self.base_linear.weight.data.sub_(weight_update)
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        If merged, runs base_linear directly.
        Otherwise computes base_linear(x) + scaling * (dropout(x) @ A.T @ B.T).
        """
        if self.merged:
            return self.base_linear(x)

        base_out = self.base_linear(x)
        # Low rank path: (x @ A.T) @ B.T
        lora_in = self.lora_dropout(x)
        # F.linear(x, A) computes x @ A.T -> shape (*, r)
        # F.linear(..., B) computes (...) @ B.T -> shape (*, out_features)
        lora_out = F.linear(F.linear(lora_in, self.lora_A), self.lora_B)
        return base_out + self.scaling * lora_out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, lora_alpha={self.lora_alpha}, scaling={self.scaling}, merged={self.merged}"
        )


def apply_lora_to_model(
    model: nn.Module,
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    target_modules: Tuple[str, ...] = ("q_lin", "v_lin"),
) -> nn.Module:
    """Recursively replaces targeted nn.Linear layers in a model with LoRALinear.

    Args:
        model: PyTorch model (e.g. DistilBertForSequenceClassification).
        r: Rank.
        lora_alpha: LoRA alpha scaling factor.
        lora_dropout: Dropout probability.
        target_modules: Tuple of attribute names to replace (e.g. ('q_lin', 'v_lin')).

    Returns:
        The modified model with LoRALinear layers installed.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and name in target_modules:
            lora_layer = LoRALinear(
                base_linear=module,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            setattr(model, name, lora_layer)
        else:
            apply_lora_to_model(
                module,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
            )
    return model


def mark_only_lora_as_trainable(model: nn.Module, train_classifier: bool = True) -> None:
    """Freezes all base model parameters, leaving only LoRA parameters (and classifier) trainable.

    Args:
        model: PyTorch model with LoRALinear modules.
        train_classifier: If True, keeps classifier head parameters trainable.
    """
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        elif train_classifier and ("classifier" in name or "pre_classifier" in name):
            param.requires_grad = True
        else:
            param.requires_grad = False


def merge_lora_model(model: nn.Module) -> None:
    """Merges all LoRALinear layers in the model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora_model(model: nn.Module) -> None:
    """Unmerges all LoRALinear layers in the model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


def count_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """Counts total and trainable parameters in the model.

    Returns:
        Tuple of (trainable_params, total_params, percentage_trainable).
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    pct = (trainable_params / total_params * 100.0) if total_params > 0 else 0.0
    return trainable_params, total_params, pct
