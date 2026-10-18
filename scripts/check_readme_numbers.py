"""
scripts/check_readme_numbers.py
Reads results/results.csv and prints mean ± std validation accuracy
per configuration so the README summary table can be verified.
Usage: python scripts/check_readme_numbers.py
"""

import os
import sys
import pandas as pd
import numpy as np

CSV_PATH = os.path.join("results", "results.csv")

if not os.path.exists(CSV_PATH):
    sys.exit(f"ERROR: {CSV_PATH} not found. Run the sweep first.")

df = pd.read_csv(CSV_PATH).dropna(how="all")

print(f"Total data rows in {CSV_PATH}: {len(df)}")
print()

expected_rows = 12
if len(df) != expected_rows:
    print(f"WARNING: Expected {expected_rows} rows but found {len(df)}.")
    print("The README notes the actual count.")
else:
    print(f"Row count check: {len(df)} rows (expected {expected_rows}) — OK")
print()

print("=" * 80)
print(f"{'Run':^5} {'Method':^8} {'Rank r':^8} "
      f"{'Trainable':>14} {'Total':>14} "
      f"{'Adapter-only %':>15} {'Total trainable %':>18} "
      f"{'Mean Acc %':>12} {'Std Acc %':>10} {'N seeds':>8}")
print("-" * 80)

for (run_id, method, rank), g in df.groupby(
    ["run_id", "method", "rank"], dropna=False, sort=True
):
    acc = g["val_accuracy"] * 100
    tp = int(g["trainable_params"].iloc[0])
    tot = int(g["total_params"].iloc[0])

    # Adapter-only: LoRA A and B parameters, no classifier head
    # Classifier head: pre_classifier (768*768+768) + classifier (768*2+2) = 591,618 + 1,538 = 592,130 (but we keep what the CSV says)
    # For full FT rank is NaN; adapter fraction only meaningful for LoRA
    if method == "lora" and not pd.isna(rank):
        rank_int = int(rank)
        # 6 layers × 2 matrices (q,v) × 2 factor (A+B sides) × 768 × r
        adapter_only = 6 * 2 * (2 * 768 * rank_int)
        head = tp - adapter_only
        adapter_pct = adapter_only / tot * 100
    else:
        adapter_only = None
        adapter_pct = None

    pct_total = tp / tot * 100
    rank_str = "n/a" if pd.isna(rank) else str(int(rank))
    adapter_pct_str = f"{adapter_pct:.4f}%" if adapter_pct is not None else "n/a"

    print(
        f"{run_id:^5} {method:^8} {rank_str:^8} "
        f"{tp:>14,} {tot:>14,} "
        f"{adapter_pct_str:>15} {pct_total:>17.4f}% "
        f"{acc.mean():>11.4f}% {acc.std():>9.4f}% {len(g):>8}"
    )

print("=" * 80)
print()
print("Notes:")
print("  'Adapter-only %' = LoRA A+B matrices only, as % of total base params.")
print("  'Total trainable %' = all trainable params (adapter + classifier head), as % of total.")
print("  These numbers should match the README summary table exactly.")
