"""
scripts/run_3epoch.py
Runs the 3-epoch experiment: full fine-tuning (Run A) and LoRA r=8 (Run D)
for seeds 42, 43, 44. Saves results to results/results_3epoch.csv.

Usage:
    python scripts/run_3epoch.py                  # runs all 6 runs
    python scripts/run_3epoch.py --run A --seed 42
    python scripts/run_3epoch.py --run D --seed 42
"""

import os
import sys
import argparse
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import train_single_run, load_sweep_config

RESULTS_CSV = "results/results_3epoch.csv"


def main():
    parser = argparse.ArgumentParser(description="3-epoch experiment runner")
    parser.add_argument("--run", choices=["A", "D"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_sweep_config("configs/sweep.yaml")
    common = cfg["common"].copy()
    common["epochs"] = 3           # Override to 3 epochs

    runs_cfg = cfg["runs"]

    targets = []
    if args.run:
        r_info = runs_cfg[args.run]
        seeds = [args.seed] if args.seed is not None else r_info["seeds"]
        for s in seeds:
            targets.append((args.run, r_info, s))
    else:
        for r_id in ["A", "D"]:
            r_info = runs_cfg[r_id]
            for s in r_info["seeds"]:
                targets.append((r_id, r_info, s))

    print(f"Running {len(targets)} training run(s) for 3 epochs each.")
    print(f"Results will be saved to: {RESULTS_CSV}")
    print()

    for run_id, r_info, seed in targets:
        print(f"Starting Run {run_id} | Method: {r_info['method']} | Seed: {seed}")
        train_single_run(
            run_id=run_id,
            method=r_info["method"],
            seed=seed,
            learning_rate=r_info["learning_rate"],
            rank=r_info.get("rank"),
            lora_alpha=r_info.get("lora_alpha"),
            common_cfg=common,
            results_csv_path=RESULTS_CSV,
            save_best_adapter=False,   # Don't overwrite the adapter from 1-epoch best run
        )

    print()
    print(f"Done. Results saved to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
