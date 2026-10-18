# CHANGES.md — Audit trail for repository strengthening

This document lists every change made to the repository, the commit hash after each change,
and any result that differed from what the original README claimed.

---

## Changes Made

### Task 1: 3-Epoch Experiment

| Item | Detail |
|------|--------|
| New file | `scripts/run_3epoch.py` — runs full FT (A) and LoRA r=8 (D) for 3 epochs |
| New file | `results/results_3epoch.csv` — 6 rows (2 configs × 3 seeds) |
| README | Added 1-epoch vs 3-epoch comparison table |
| README | Added paragraph on whether accuracy gap changes with more training |
| Commit | *(see `git log --oneline` after each commit)* |

### Task 2: Validation-Set Protocol

| Item | Detail |
|------|--------|
| New file | `scripts/check_split_overlap.py` — asserts no index overlap among train subset (8,000), tuning set (1,000), and validation set |
| README | Added **Evaluation protocol** section |
| README | Stated that the tuning set is used only for the sanity check, not for hyperparameter selection |
| README | Stated: "Learning rates were fixed before any run; this was not re-verified by a separate experiment." |
| Note | SST-2 contains 2 exact-duplicate sentences that appear at different positions in the training split; the overlap check uses index position, not raw text, which is the correct criterion |

### Task 3: Verifiability Fixes

| Item | Detail |
|------|--------|
| New file | `scripts/check_readme_numbers.py` — reads `results/results.csv` and prints mean ± std per configuration |
| `results/results.csv` | **12 rows confirmed** (4 configs × 3 seeds); no rows added or edited |
| README | Removed "production-grade" |
| README | Removed "Windows 10 / Linux compatible" |
| README | Removed the 1,821-example test set line (SST-2 test labels are not public) |
| README | Replaced single "1.10% trainable" with two distinct percentages: adapter-only % and total trainable % |
| README | Rewrote memory explanation: LoRA reduces optimizer state and gradient storage for frozen weights; backpropagation still passes through frozen layers |
| README | Replaced "machine precision (max abs difference = 0.0)" with the actual measured values from running `tests/test_lora.py`: zero-init diff = 0.00e+00, merge diff = 2.38e-06, PEFT-equivalence diff = 0.00e+00 |
| README | Each memory and accuracy number is attributed to the specific run and seed that produced it |
| README | Clone instructions use `lora-distilbert-study` as the local directory name; GitHub remote URL is unchanged |

**Rename note for GitHub repository itself:**
The task asks not to rename the GitHub repo itself. To change only the *displayed* clone directory name, update the `README.md` clone command to:
```bash
git clone https://github.com/lekha282006-ship-it/Document-LoRA-Reproduction-and-Controlled-Study-on-DistilBERT-with-a-Deployed-Inference-Service.git lora-distilbert-study
cd lora-distilbert-study
```
No change is required on GitHub.

### Task 4: Docker Deployment

| Item | Detail |
|------|--------|
| Status | Docker is not installed on this machine; the build step could not be completed locally |
| README | States "runs locally in Docker" and documents exact `docker build` and `docker run` commands |
| README | Removed any implicit claim of cloud deployment |
| API tests | `tests/test_api.py` — 4 tests all pass (verified with FastAPI `TestClient`, not a live container) |

---

## Discrepancies Versus Original README

| Claim | Status |
|-------|--------|
| "1-epoch full FT val accuracy: 90.29 ± 0.35%" | **Confirmed** from `results/results.csv` (runs A, seeds 42/43/44) |
| "LoRA r=8 val accuracy: 90.67 ± 0.37%" | **Confirmed** from `results/results.csv` (runs D, seeds 42/43/44) |
| "max abs difference = 0.0" (PEFT equivalence) | **Corrected** — zero-init: 0.00e+00 ✓, merge: 2.38e-06 (not zero), PEFT equiv: 0.00e+00 ✓ |
| "Windows 10 / Linux compatible" | **Removed** — no cross-platform test was run |
| "production-grade" | **Removed** — unverifiable marketing claim |
| SST-2 test set size (1,821 examples) | **Removed** — test labels are not public; reporting on this set is not possible |
| 1-epoch results device: `cuda:0` | **Unchanged** — original CSV records `device=cuda:0`; README now attributes each number to its specific run |
| 3-epoch comparison | **New** — results in `results/results_3epoch.csv`; table added to README |
| Tuning-set protocol | **New** — `scripts/check_split_overlap.py` confirms no overlap |

---

## Commit Hash Reference

Run `git log --oneline` to see the full commit history.

Key commits (in order):
1. `435cdea` Docs: update clone URL to match GitHub repository name
2. `2841061` Task 2: add evaluation protocol sanity check and non-overlap verification
3. `c45d599` Task 3: make repository verifiable with corrected README and metric audit scripts
4. `e23a168` Task 4: document Docker deployment and add 3-epoch execution runner
5. `2cf3e20` Task 1: add 3-epoch experiment with results_3epoch.csv and comparison analysis


