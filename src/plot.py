"""Generates publication-quality figures comparing LoRA and Full FT from results.csv."""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", font_scale=1.1)
os.makedirs("results/figures", exist_ok=True)

df = pd.read_csv("results/results.csv")
grouped = df.groupby(["run_id", "method", "rank", "trainable_params"], dropna=False).agg(
    acc_mean=("val_accuracy", lambda x: float(np.mean(x) * 100)),
    acc_std=("val_accuracy", lambda x: float(np.std(x) * 100)),
    time_mean=("train_time_sec", "mean"),
    time_std=("train_time_sec", "std"),
    mem_mean=("peak_memory_mb", "mean"),
    pct_trainable=("pct_trainable", "first"),
).reset_index()

grouped["rank_str"] = grouped["rank"].apply(lambda r: "n/a" if pd.isna(r) or str(r) == "None" else str(int(r)))

# Figure 1: Accuracy vs Trainable Parameters
fig, ax = plt.subplots(figsize=(10, 6))
for _, row in grouped.iterrows():
    m = row["method"].upper()
    rk = row["rank_str"]
    lbl = f"{m} (r={rk})"
    marker = "o" if row["method"] == "lora" else "s"
    color = "#1f77b4" if row["method"] == "lora" else "#d62728"

    ax.errorbar(
        x=row["trainable_params"],
        y=row["acc_mean"],
        yerr=row["acc_std"],
        fmt=marker,
        markersize=9,
        capsize=5,
        capthick=1.5,
        color=color,
        label=lbl,
    )
    ax.annotate(
        f"{row['acc_mean']:.2f}%\n({row['pct_trainable']:.2f}%)",
        (row["trainable_params"], row["acc_mean"]),
        textcoords="offset points",
        xytext=(10, -5),
        fontsize=10,
        weight="bold",
    )

ax.set_xscale("log")
ax.set_xlabel("Trainable Parameters (Log Scale)", fontsize=12, fontweight="bold")
ax.set_ylabel("SST-2 Validation Accuracy (%)", fontsize=12, fontweight="bold")
ax.set_title("DistilBERT on SST-2: Validation Accuracy vs. Trainable Parameters", fontsize=14, fontweight="bold", pad=15)
ax.set_ylim(85.0, 93.0)
ax.grid(True, which="both", linestyle="--", alpha=0.5)
ax.legend(loc="lower right", frameon=True)
plt.tight_layout()
plt.savefig("results/figures/accuracy_vs_parameters.png", dpi=300)
plt.close()

# Figure 2: Resource Efficiency
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
labels = [f"{r}\n({m.upper()}, r={rk})" for r, m, rk in zip(grouped["run_id"], grouped["method"], grouped["rank_str"])]
colors = ["#d62728" if m == "full" else "#2ca02c" for m in grouped["method"]]

ax1.bar(labels, grouped["mem_mean"], color=colors, alpha=0.85, edgecolor="black")
ax1.set_ylabel("Peak VRAM (MB)", fontsize=11, fontweight="bold")
ax1.set_title("Peak Training Memory (MB)", fontsize=13, fontweight="bold")
ax1.grid(True, axis="y", linestyle="--", alpha=0.6)
for i, v in enumerate(grouped["mem_mean"]):
    ax1.text(i, v + 25, f"{v:.0f} MB", ha="center", fontweight="bold", fontsize=10)

ax2.bar(labels, grouped["time_mean"], yerr=grouped["time_std"], capsize=4, color=colors, alpha=0.85, edgecolor="black")
ax2.set_ylabel("Training Time (seconds)", fontsize=11, fontweight="bold")
ax2.set_title("Wall-Clock Training Time (s)", fontsize=13, fontweight="bold")
ax2.grid(True, axis="y", linestyle="--", alpha=0.6)
for i, v in enumerate(grouped["time_mean"]):
    ax2.text(i, v + 1.2, f"{v:.1f}s", ha="center", fontweight="bold", fontsize=10)

plt.tight_layout()
plt.savefig("results/figures/resource_efficiency.png", dpi=300)
plt.close()

print("Successfully generated figures in results/figures/!")
