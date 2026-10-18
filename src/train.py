import os
import sys

# Ensure repository root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import yaml
import random
import logging
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.data import get_dataloaders
from src.model import build_model, save_adapter
from src.evaluate import evaluate_model


def set_seed(seed: int) -> None:
    """Sets random seeds for deterministic execution across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logger(log_file: str) -> logging.Logger:
    """Sets up file and stream logger."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def train_single_run(
    run_id: str,
    method: str,
    seed: int,
    learning_rate: float,
    rank: Optional[int],
    lora_alpha: Optional[float],
    common_cfg: Dict[str, Any],
    results_csv_path: str = "results/results.csv",
    save_best_adapter: bool = True,
) -> Dict[str, Any]:
    """Executes a single training run with the given hyperparameter configuration.

    Args:
        run_id: Experiment identifier ('A', 'B', 'C', 'D').
        method: 'full' or 'lora'.
        seed: Random seed.
        learning_rate: Learning rate for AdamW.
        rank: LoRA rank (or None for full FT).
        lora_alpha: LoRA alpha (or None for full FT).
        common_cfg: Dictionary of common experiment settings.
        results_csv_path: File path to append run metrics.
        save_best_adapter: Whether to save adapter if it attains best val accuracy.

    Returns:
        Dictionary of final run metrics.
    """
    set_seed(seed)

    base_model_name = common_cfg.get("base_model", "distilbert-base-uncased")
    subset_size = common_cfg.get("subset_size", 8000)
    batch_size = common_cfg.get("batch_size", 32)
    epochs = common_cfg.get("epochs", 1)
    max_length = common_cfg.get("max_length", 128)
    weight_decay = common_cfg.get("weight_decay", 0.01)
    warmup_ratio = common_cfg.get("warmup_ratio", 0.06)
    target_modules = tuple(common_cfg.get("target_modules", ["q_lin", "v_lin"]))

    log_file = os.path.join("results", "logs", f"run_{run_id}_seed_{seed}.log")
    logger = setup_logger(log_file)

    logger.info("==================================================")
    logger.info("Starting Run %s | Method: %s | Seed: %d | Rank: %s | Alpha: %s",
                run_id, method.upper(), seed, rank, lora_alpha)
    logger.info("Learning Rate: %e | Epochs: %d | Batch Size: %d | Subset: %s",
                learning_rate, epochs, batch_size, subset_size)
    logger.info("==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Compute Device: %s", device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Load tokenizer and dataloaders
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    train_loader, val_loader, data_stats = get_dataloaders(
        tokenizer=tokenizer,
        batch_size=batch_size,
        subset_size=subset_size,
        seed=seed,
        max_length=max_length,
    )
    logger.info("Dataset statistics: %s", data_stats)

    # Build model
    model, param_summary = build_model(
        method=method,
        model_name=base_model_name,
        num_labels=2,
        r=rank if rank is not None else 8,
        lora_alpha=lora_alpha if lora_alpha is not None else 16.0,
        target_modules=target_modules,
    )
    model.to(device)

    trainable_params = param_summary["trainable_params"]
    total_params = param_summary["total_params"]
    pct_trainable = param_summary["pct_trainable"]

    logger.info("Model Parameters: %d trainable / %d total (%.2f%%)",
                trainable_params, total_params, pct_trainable)

    # Optimizer and Scheduler
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params_list, lr=learning_rate, weight_decay=weight_decay)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    criterion = nn.CrossEntropyLoss()

    # Initial baseline validation (before training)
    pre_val = evaluate_model(model, val_loader, device)
    logger.info("Initial Val Accuracy (Pre-training): %.4f (Loss: %.4f)",
                pre_val["val_accuracy"], pre_val["val_loss"])

    # Training
    model.train()
    start_time = time.perf_counter()
    step = 0
    running_loss = 0.0

    for epoch in range(1, epochs + 1):
        for batch_idx, batch in enumerate(train_loader, start=1):
            step += 1
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            loss = criterion(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params_list, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if step % 50 == 0 or step == total_steps:
                avg_loss = running_loss / (50 if step % 50 == 0 else step % 50)
                logger.info("Epoch %d/%d | Step %d/%d | Train Loss: %.4f | LR: %.2e",
                            epoch, epochs, step, total_steps, avg_loss, scheduler.get_last_lr()[0])
                running_loss = 0.0

    train_time_sec = time.perf_counter() - start_time

    # Memory Tracking
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)
    else:
        # Host RAM RSS usage in MB
        peak_memory_mb = psutil.Process().memory_info().rss / (1024**2)

    # Post-training Validation
    val_metrics = evaluate_model(model, val_loader, device)
    val_acc = val_metrics["val_accuracy"]
    val_loss = val_metrics["val_loss"]

    logger.info("Training Finished in %.2f seconds", train_time_sec)
    logger.info("Final Val Accuracy: %.4f (%.2f%%) | Val Loss: %.4f", val_acc, val_acc * 100.0, val_loss)
    logger.info("Peak Memory Usage: %.2f MB", peak_memory_mb)

    # Record result
    result_record = {
        "run_id": run_id,
        "method": method,
        "rank": rank,
        "lora_alpha": lora_alpha,
        "seed": seed,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "pct_trainable": round(pct_trainable, 4),
        "val_accuracy": round(val_acc, 6),
        "val_loss": round(val_loss, 6),
        "train_time_sec": round(train_time_sec, 2),
        "peak_memory_mb": round(peak_memory_mb, 2),
        "device": str(device),
        "timestamp": datetime.now().isoformat(),
    }

    # Append to results.csv
    os.makedirs(os.path.dirname(results_csv_path), exist_ok=True)
    df_new = pd.DataFrame([result_record])
    if os.path.exists(results_csv_path):
        df_existing = pd.read_csv(results_csv_path)
        # Replace duplicate run if re-executed
        mask = (
            (df_existing["run_id"] == run_id)
            & (df_existing["seed"] == seed)
            & (df_existing["method"] == method)
        )
        if mask.any():
            df_existing = df_existing[~mask]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(results_csv_path, index=False)
    else:
        df_new.to_csv(results_csv_path, index=False)

    # Check if this LoRA run qualifies as best to save to adapters/
    if method == "lora" and save_best_adapter:
        adapters_dir = "adapters"
        should_save = True
        best_marker_file = os.path.join(adapters_dir, "best_accuracy.txt")
        if os.path.exists(best_marker_file):
            try:
                with open(best_marker_file, "r") as f:
                    prev_best = float(f.read().strip())
                if val_acc <= prev_best:
                    should_save = False
            except Exception:
                should_save = True

        if should_save:
            logger.info("New best LoRA run! Saving adapter weights to %s (Val Acc: %.4f)",
                        adapters_dir, val_acc)
            save_adapter(
                model=model,
                save_directory=adapters_dir,
                base_model_name=base_model_name,
                r=rank,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                num_labels=2,
                additional_metadata={
                    "best_run_id": run_id,
                    "seed": seed,
                    "val_accuracy": val_acc,
                    "val_loss": val_loss,
                    "trainable_params": trainable_params,
                },
            )
            with open(best_marker_file, "w") as f:
                f.write(str(val_acc))

    logger.info("Completed Run %s (Seed %d). Metric saved to %s", run_id, seed, results_csv_path)
    return result_record


def load_sweep_config(config_path: str = "configs/sweep.yaml") -> Dict[str, Any]:
    """Loads sweep settings from yaml file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train LoRA or full fine-tuning on DistilBERT SST-2")
    parser.add_argument("--config", type=str, default="configs/sweep.yaml", help="Path to sweep config yaml")
    parser.add_argument("--run", type=str, default=None, choices=["A", "B", "C", "D"], help="Run ID to execute")
    parser.add_argument("--seed", type=int, default=None, help="Specific seed to run")
    parser.add_argument("--run-all", action="store_true", help="Execute all runs and seeds in sweep")
    parser.add_argument("--subset-size", type=int, default=None, help="Override subset size (for fast checks)")
    args = parser.parse_args()

    cfg = load_sweep_config(args.config)
    common_cfg = cfg["common"]
    if args.subset_size is not None:
        common_cfg["subset_size"] = args.subset_size

    runs_cfg = cfg["runs"]

    if args.run_all:
        for r_id, r_info in runs_cfg.items():
            for s in r_info["seeds"]:
                train_single_run(
                    run_id=r_id,
                    method=r_info["method"],
                    seed=s,
                    learning_rate=r_info["learning_rate"],
                    rank=r_info.get("rank"),
                    lora_alpha=r_info.get("lora_alpha"),
                    common_cfg=common_cfg,
                )
    elif args.run:
        r_info = runs_cfg[args.run]
        seeds = [args.seed] if args.seed is not None else r_info["seeds"]
        for s in seeds:
            train_single_run(
                run_id=args.run,
                method=r_info["method"],
                seed=s,
                learning_rate=r_info["learning_rate"],
                rank=r_info.get("rank"),
                lora_alpha=r_info.get("lora_alpha"),
                common_cfg=common_cfg,
            )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
