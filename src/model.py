import os
import sys

# Ensure repository root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import logging
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification

from src.lora import (
    apply_lora_to_model,
    mark_only_lora_as_trainable,
    count_parameters,
    merge_lora_model,
    unmerge_lora_model,
)

logger = logging.getLogger(__name__)


def build_model(
    method: str = "lora",
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 2,
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    target_modules: Tuple[str, ...] = ("q_lin", "v_lin"),
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Builds a DistilBERT model for sequence classification configured for full FT or LoRA.

    Args:
        method: 'full' for full fine-tuning, or 'lora' for Low-Rank Adaptation.
        model_name: Hugging Face model identifier.
        num_labels: Number of output classification labels.
        r: LoRA rank (only used if method == 'lora').
        lora_alpha: LoRA scaling alpha (only used if method == 'lora').
        lora_dropout: LoRA dropout probability.
        target_modules: Layer projection names to adapt with LoRA.

    Returns:
        Tuple of (model, param_summary_dict).
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    if method.lower() == "full":
        # Ensure all parameters require gradient
        for param in model.parameters():
            param.requires_grad = True
        lora_params = 0
    elif method.lower() == "lora":
        # Wrap target modules with scratch LoRALinear
        apply_lora_to_model(
            model,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
        )
        # Freeze base parameters, keep LoRA and classification head trainable
        mark_only_lora_as_trainable(model, train_classifier=True)
        lora_params = sum(
            p.numel()
            for n, p in model.named_parameters()
            if ("lora_A" in n or "lora_B" in n) and p.requires_grad
        )
    else:
        raise ValueError(f"Unknown method '{method}', must be 'full' or 'lora'")

    trainable_params, total_params, pct = count_parameters(model)
    head_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if ("classifier" in n or "pre_classifier" in n) and p.requires_grad
    )

    summary = {
        "method": method.lower(),
        "model_name": model_name,
        "rank": r if method.lower() == "lora" else None,
        "lora_alpha": lora_alpha if method.lower() == "lora" else None,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "pct_trainable": pct,
        "lora_params": lora_params,
        "head_params": head_params,
    }

    logger.info(
        "Built %s model. Trainable: %d / %d (%.2f%%)",
        method.upper(),
        trainable_params,
        total_params,
        pct,
    )
    return model, summary


def save_adapter(
    model: nn.Module,
    save_directory: str,
    base_model_name: str = "distilbert-base-uncased",
    r: int = 8,
    lora_alpha: float = 16.0,
    target_modules: Tuple[str, ...] = ("q_lin", "v_lin"),
    num_labels: int = 2,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Saves only the trained LoRA adapter parameters and classifier head weights to disk.

    Args:
        model: Trained model with LoRA layers.
        save_directory: Path to output directory.
        base_model_name: HF identifier of the base model.
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        target_modules: LoRA adapted modules.
        num_labels: Number of classes.
        additional_metadata: Optional dictionary with metrics, training args, etc.
    """
    os.makedirs(save_directory, exist_ok=True)

    # Extract only trainable adapter and classification head weights
    adapter_state_dict = {}
    for name, param in model.named_parameters():
        if (
            "lora_A" in name
            or "lora_B" in name
            or "classifier" in name
            or "pre_classifier" in name
        ):
            adapter_state_dict[name] = param.data.cpu()

    # Save weights
    weights_path = os.path.join(save_directory, "adapter_model.pt")
    torch.save(adapter_state_dict, weights_path)

    # Save metadata configuration
    config = {
        "base_model_name": base_model_name,
        "r": r,
        "lora_alpha": lora_alpha,
        "target_modules": list(target_modules),
        "num_labels": num_labels,
    }
    if additional_metadata:
        config.update(additional_metadata)

    config_path = os.path.join(save_directory, "adapter_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info("Saved adapter weights and config to %s", save_directory)


def load_adapter(
    adapter_directory: str,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Loads a saved LoRA adapter onto its base DistilBERT model.

    Args:
        adapter_directory: Path to folder containing adapter_config.json and adapter_model.pt.
        device: PyTorch device to place model on.

    Returns:
        Tuple of (model_with_adapter_loaded, config_dict).
    """
    config_path = os.path.join(adapter_directory, "adapter_config.json")
    weights_path = os.path.join(adapter_directory, "adapter_model.pt")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config at {config_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weights at {weights_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Build base model and inject LoRA layers matching config
    model, _ = build_model(
        method="lora",
        model_name=config.get("base_model_name", "distilbert-base-uncased"),
        num_labels=config.get("num_labels", 2),
        r=config.get("r", 8),
        lora_alpha=config.get("lora_alpha", 16.0),
        target_modules=tuple(config.get("target_modules", ["q_lin", "v_lin"])),
    )

    # Load adapter weights
    state_dict = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    logger.info("Loaded adapter into model. Missing non-adapter keys: %d, unexpected: %d", len(missing), len(unexpected))

    if device:
        model.to(device)

    model.eval()
    return model, config
