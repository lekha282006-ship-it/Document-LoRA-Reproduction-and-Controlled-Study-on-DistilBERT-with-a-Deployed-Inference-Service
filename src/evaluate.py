"""Evaluation and benchmarking utilities for accuracy and inference latency."""

import time
import platform
import psutil
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
import numpy as np
import pandas as pd


def get_hardware_info() -> Dict[str, str]:
    """Retrieves host hardware description."""
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": platform.processor() or "x86_64",
        "cpu_count_logical": str(psutil.cpu_count(logical=True)),
        "cpu_count_physical": str(psutil.cpu_count(logical=False)),
        "ram_gb": f"{psutil.virtual_memory().total / (1024**3):.1f} GB",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU only)",
    }
    return info


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluates accuracy and loss over a dataloader.

    Args:
        model: PyTorch model.
        dataloader: Validation DataLoader.
        device: Evaluation device.

    Returns:
        Dict containing accuracy, loss, and num_samples.
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)

            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    return {
        "val_loss": avg_loss,
        "val_accuracy": accuracy,
        "val_samples": total,
    }


def measure_cpu_latency(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    num_requests: int = 100,
    sample_text: str = "This was an absolutely brilliant and uplifting film.",
) -> Dict[str, Any]:
    """Measures single-request inference latency over num_requests on CPU.

    Args:
        model: PyTorch model with LoRA or merged.
        tokenizer: Model tokenizer.
        num_requests: Number of test requests (default: 100).
        sample_text: Input text sentence to classify.

    Returns:
        Dictionary of latency statistics in milliseconds and hardware info.
    """
    model.eval()
    model.to("cpu")

    inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(**inputs)

    latencies_ms = []
    with torch.no_grad():
        for _ in range(num_requests):
            start = time.perf_counter()
            _ = model(**inputs)
            duration_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(duration_ms)

    latencies_arr = np.array(latencies_ms)
    hw = get_hardware_info()

    return {
        "num_requests": num_requests,
        "median_ms": float(np.median(latencies_arr)),
        "mean_ms": float(np.mean(latencies_arr)),
        "std_ms": float(np.std(latencies_arr)),
        "p95_ms": float(np.percentile(latencies_arr, 95)),
        "p99_ms": float(np.percentile(latencies_arr, 99)),
        "min_ms": float(np.min(latencies_arr)),
        "max_ms": float(np.max(latencies_arr)),
        "hardware": hw,
    }


def format_summary_table(df: pd.DataFrame) -> str:
    """Formats sweep results into a clean markdown table with mean ± std."""
    if df.empty:
        return "No results recorded."

    # Group by run_id, method, rank
    grouped = df.groupby(["run_id", "method", "rank", "trainable_params"], dropna=False)

    rows = []
    for (run_id, method, rank, trainable_params), group in grouped:
        rank_str = "n/a" if pd.isna(rank) or str(rank) == "None" else str(int(rank))
        acc_mean = group["val_accuracy"].mean() * 100.0
        acc_std = group["val_accuracy"].std() * 100.0
        time_mean = group["train_time_sec"].mean()
        time_std = group["train_time_sec"].std()
        mem_mean = group["peak_memory_mb"].mean()

        seeds_str = ", ".join(str(s) for s in sorted(group["seed"].unique()))

        rows.append({
            "Run": run_id,
            "Method": method.upper(),
            "Rank r": rank_str,
            "Trainable Params": f"{int(trainable_params):,}",
            "Accuracy (Mean ± Std %)": f"{acc_mean:.2f} ± {acc_std:.2f}%",
            "Train Time (s)": f"{time_mean:.1f} ± {time_std:.1f}s",
            "Peak Mem (MB)": f"{mem_mean:.1f} MB",
            "Seeds": seeds_str,
        })

    summary_df = pd.DataFrame(rows)
    return summary_df.to_markdown(index=False)
