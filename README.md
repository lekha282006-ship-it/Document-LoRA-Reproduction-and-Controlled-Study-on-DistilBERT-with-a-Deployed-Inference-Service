# LoRA Reproduction and Controlled Study on DistilBERT with Deployed Inference Service

[![Tests](https://img.shields.io/badge/pytest-8%20passed-brightgreen.svg)](#test-suite)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Paper](https://img.shields.io/badge/arXiv-2106.09685-B31B1B.svg)](https://arxiv.org/abs/2106.09685)

This repository reproduces the core claim of Hu et al., ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) (ICLR 2022), on a sequence classification task using `distilbert-base-uncased` fine-tuned on GLUE SST-2. It features:
- A scratch LoRA layer implementation written directly in PyTorch (without importing `peft` in `src/lora.py`).
- Numerical parity tests against Hugging Face PEFT (see measured differences in [Test Suite](#test-suite)).
- A controlled hyperparameter sweep comparing full fine-tuning against LoRA ranks $r \in \{1, 4, 8\}$ across 3 random seeds (`42, 43, 44`).
- A Dockerized FastAPI inference microservice for CPU inference.

---

## 1. Experimental Results & Controlled Study

### 1-Epoch Sweep (Primary Results)

All runs were evaluated under identical conditions on a fixed 8,000-example subset of SST-2 with batch size 32, 1 epoch, and seed control (`42, 43, 44`). The learning rate was selected once per method and held constant ($2\times 10^{-5}$ for full fine-tuning, $2\times 10^{-4}$ for LoRA). The LoRA scaling factor used the canonical $\alpha = 2r$ rule.

**Summary Table (Mean ± Standard Deviation over 3 Seeds)**

Source: [`results/results.csv`](results/results.csv). Verify with `python scripts/check_readme_numbers.py`.

| Run | Method | Rank $r$ | LoRA $\alpha$ | Trainable Params | Adapter-only % | Total Trainable % | Val Accuracy (Mean ± Std %) | Train Time (s) | Peak VRAM (MB) | Seeds |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | **Full FT** | n/a | n/a | 66,955,010 | n/a | 100.00% | **90.29 ± 0.35%** | 38.5 ± 0.3s | 1480.5 | 42, 43, 44 |
| **B** | **LoRA** | 1 | 2.0 | 610,562 | 0.0275% | 0.91% | **87.12 ± 0.52%** | 24.2 ± 0.2s | 920.4 | 42, 43, 44 |
| **C** | **LoRA** | 4 | 8.0 | 665,858 | 0.1101% | 0.99% | **89.72 ± 0.46%** | 25.1 ± 0.2s | 945.8 | 42, 43, 44 |
| **D** | **LoRA** | 8 | 16.0 | 739,586 | 0.2202% | 1.10% | **90.67 ± 0.37%** | 26.4 ± 0.2s | 982.1 | 42, 43, 44 |

> **Parameter Breakdown**:
> DistilBERT has 6 transformer layers. Injecting LoRA into `q_lin` and `v_lin` adds $2 \times (2 \times 768 \times r)$ parameters per layer ($18{,}432 \times r$ LoRA parameters total). The classification head (`pre_classifier` + `classifier`) adds 592,130 parameters.
> - **Adapter-only %**: LoRA A and B matrices alone as a fraction of total model parameters (e.g., $r=8$: $147{,}456 / 66{,}955{,}010 = 0.2202\%$).
> - **Total trainable %**: Adapter parameters + classification head (e.g., $r=8$: $739{,}586 / 66{,}955{,}010 = 1.10\%$).
>
> Individual run values: Train time and peak VRAM are from Run A/seed 42 (38.5s, 1480.5 MB), Run B/seed 42 (24.2s, 920.4 MB), Run C/seed 42 (25.1s, 945.8 MB), Run D/seed 42 (26.4s, 982.1 MB); all runs within each config showed identical VRAM and ≤0.3s variation in train time. All 1-epoch runs executed on `cuda:0`.

---

## 2. Findings

1. **Reproduction Confirmed**: LoRA at rank $r=8$ matches and slightly exceeds full fine-tuning ($90.67\% \pm 0.37\%$ vs $90.29\% \pm 0.35\%$) while adapting only **0.22% adapter-only** (or **1.10% total trainable** including the classification head) of parameters; the performance gap is well within seed variance.
2. **Rank Bottleneck at $r=1$**: LoRA at rank $r=1$ reached only $87.12\% \pm 0.52\%$, demonstrating that a single rank subspace is too restrictive for DistilBERT query/value projections.
3. **Memory Savings**: LoRA reduced peak GPU memory by **37.8%** (920 MB vs 1480 MB for Run B vs Run A, seed 42). The savings come primarily from not storing optimizer state (first and second moments in AdamW) and gradients for the 66M frozen backbone parameters. Backpropagation still passes through the frozen layers to compute gradients for the LoRA adapter matrices.

---

## 3. Evaluation Protocol

- **Reporting set**: All accuracy numbers are computed on the SST-2 validation split (872 examples) from the GLUE benchmark.
- **Training subset**: 8,000 examples sampled deterministically from the SST-2 training split (67,349 examples) via a seed-controlled shuffle.
- **Tuning set**: A held-out set of 1,000 examples drawn from the SST-2 training split immediately after the 8,000-example training subset (indices 8,000–8,999 in the shuffled training split, seed 42). This set was created solely for the sanity check (`scripts/check_split_overlap.py`) and was **not** used for hyperparameter selection.
- **Hyperparameter selection**: Learning rates were fixed before any run ($2\times 10^{-5}$ for full fine-tuning, $2\times 10^{-4}$ for LoRA); this was not re-verified by a separate experiment.
- **No overlap guarantee**: `scripts/check_split_overlap.py` verifies by index that the training subset, tuning set, and validation set are mutually disjoint. SST-2 contains a small number of exact-duplicate sentences at different positions in the training split; index-based separation is the correct non-overlap criterion.

---

## 4. Dataset Specifications

The model is trained and evaluated on the Stanford Sentiment Treebank (SST-2) binary sentiment classification task from GLUE:
- **Full Training Set**: 67,349 examples
- **Validation Set**: 872 examples (used for reporting)
- **Controlled Study Training Subset**: 8,000 examples (sampled deterministically per seed)
- **Held-Out Tuning Set**: 1,000 examples (sanity check only, not used for selection)

---

## 5. Scratch LoRA Implementation (`src/lora.py`)

LoRA injects rank decomposition matrices into frozen linear layers:
$$h = W_0 x + \frac{\alpha}{r} B A x$$
- $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is completely frozen ($\text{requires\_grad}=\text{False}$).
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$ is initialized with Kaiming uniform initialization ($a = \sqrt{5}$).
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$ is initialized to all zeros, guaranteeing the adapter starts as an exact identity/no-op.
- A `merge()` method directly computes $W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA$ into the weight tensor for zero-overhead inference.
- An `unmerge()` method recovers separate matrices if continued training is desired.

### Hugging Face PEFT Equivalence
In `tests/test_lora.py`, our custom `LoRALinear` layer is compared against Hugging Face `peft`'s `LoraLayer` with identical weights. Measured maximum absolute differences (from running the test suite):
- **Zero-init test**: `max abs diff = 0.00e+00` (fresh LoRA output is identical to base linear)
- **Merge/unmerge test**: `max abs diff = 2.38e-06` (within the `1e-5` tolerance; non-zero due to floating-point order of operations)
- **PEFT equivalence test**: `max abs diff = 0.00e+00` (identical output to Hugging Face PEFT with synced weights)

---

## 6. Repository Structure

```
lora-distilbert-study/
├── README.md              # Project documentation, findings, and reproduction guide
├── CHANGES.md             # Audit trail of all repository changes
├── requirements.txt       # Pinned dependencies
├── Dockerfile             # Container definition
├── pytest.ini             # Pytest configuration
├── src/
│   ├── __init__.py
│   ├── lora.py            # LoRA layer written from scratch (no peft import)
│   ├── data.py            # SST-2 dataset loading, subsetting, and tokenization
│   ├── model.py           # DistilBERT model factory, LoRA injection, and checkpointing
│   ├── train.py           # Unified training entry point for all sweep configs
│   ├── evaluate.py        # Accuracy evaluation and parameter counts
│   ├── serve.py           # FastAPI inference microservice
│   ├── plot.py            # Generates visualization figures
│   └── tool_agent.py      # Local LLM tool-calling extension
├── tests/
│   ├── test_lora.py       # Zero-init, shape, merge, and HF PEFT equivalence tests
│   └── test_api.py        # /health and /predict endpoint tests
├── scripts/
│   ├── check_readme_numbers.py  # Verify README table numbers against results.csv
│   ├── check_split_overlap.py   # Assert no overlap between train/tuning/val sets
│   └── run_3epoch.py            # 3-epoch experiment runner (Task 1)
├── notebooks/
│   └── experiments.ipynb  # Interactive sweep visualization and analysis
├── configs/
│   └── sweep.yaml         # Configuration file holding all sweep hyperparameters
├── results/
│   ├── results.csv        # Individual metrics for every 1-epoch run and seed
│   ├── results_3epoch.csv # Individual metrics for 3-epoch experiment (Task 1)
│   ├── latency_benchmark.json # 100-request CPU latency benchmark metrics
│   ├── figures/           # Generated visualization plots
│   └── logs/              # Training run logs and verification outputs
└── adapters/              # Saved LoRA weights & metadata for best run (Run D, r=8)
    ├── adapter_config.json
    └── adapter_model.pt
```

---

## 7. Inference Service & Latency Benchmark

The inference service is exposed via FastAPI in `src/serve.py`. It loads `distilbert-base-uncased`, injects the trained LoRA adapter from `adapters/`, and merges the weights for zero-overhead inference.

### API Endpoints
- **`GET /health`**: Returns system status, model name, device, and adapter loading state.
- **`POST /predict`**: Accepts `{"text": "..."}` and returns sentiment classification label, confidence score, and class ID.

#### Example Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "An astonishing, poignant, and uplifting masterpiece."}'
```

#### Example Response
```json
{
  "label": "positive",
  "score": 0.98421,
  "class_id": 1
}
```

### CPU Latency Benchmark (100 Requests)
Source: [`results/latency_benchmark.json`](results/latency_benchmark.json). Inference latency was benchmarked on a single request stream across 100 consecutive queries on CPU:
- **Median Latency**: **54.37 ms**
- **Mean Latency**: **56.98 ± 16.15 ms**
- **95th Percentile**: **87.99 ms**
- **99th Percentile**: **103.44 ms**
- **Min / Max**: 37.56 ms / 117.64 ms

**Benchmarking Hardware**:
- **Processor**: Intel(R) Core(TM) i5 (4 physical cores, 8 logical threads)
- **RAM**: 11.8 GB
- **Inference Mode**: CPU single-batch forward pass with merged LoRA weights

---

## 8. Docker Deployment

The inference service runs locally in Docker. No cloud deployment is provided.

### Build the Image
```bash
docker build -t lora-distilbert:latest .
```

### Run the Container
```bash
docker run -d --name lora-service -p 8000:8000 lora-distilbert:latest
```

### Test Inside Container
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The pacing was agonizingly slow and boring."}'
```

> **Note**: Docker was not available on the development machine during this study. The API tests (`tests/test_api.py`) were verified using FastAPI's `TestClient` (see [Test Suite](#test-suite)). To verify the Docker build and endpoints, run the commands above on a machine with Docker installed.

---

## 9. Test Suite

All 8 tests pass. Run the full automated test suite:
```bash
python -m pytest tests/ -v
```

Output (recorded in `results/logs/pytest_output.txt`):
```
tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_predict_positive_sentiment PASSED
tests/test_api.py::test_predict_negative_sentiment PASSED
tests/test_api.py::test_predict_empty_text_validation PASSED
tests/test_lora.py::test_zero_init PASSED
tests/test_lora.py::test_shapes PASSED
tests/test_lora.py::test_merge PASSED
tests/test_lora.py::test_library_peft_equivalence PASSED

8 passed in 23.33s
```

Tests verify:
1. `test_zero_init`: Fresh LoRALinear outputs match base linear layer exactly (max abs diff = 0.00e+00).
2. `test_shapes`: Matrix dimensions match $(r, d_{\text{in}})$ and $(d_{\text{out}}, r)$.
3. `test_merge`: Output before and after `merge()` match within $1\times 10^{-5}$ (measured: 2.38e-06).
4. `test_library_peft_equivalence`: Custom layer matches Hugging Face PEFT on identical parameters (max abs diff = 0.00e+00).
5. `test_health_endpoint`: FastAPI `/health` endpoint responds with HTTP 200 and correct schema.
6. `test_predict_positive_sentiment`: `/predict` returns valid sentiment format.
7. `test_predict_negative_sentiment`: Negative sentiment classified with valid probability.
8. `test_predict_empty_text_validation`: Invalid empty inputs rejected with HTTP 400/422.

---

## 10. Quickstart: Reproduce from Scratch

### 1. Clone & Install
```bash
git clone https://github.com/lekha282006-ship-it/Document-LoRA-Reproduction-and-Controlled-Study-on-DistilBERT-with-a-Deployed-Inference-Service.git lora-distilbert-study
cd lora-distilbert-study
pip install -r requirements.txt
```

### 2. Run the Controlled Sweep
```bash
# Run a specific configuration and seed
python src/train.py --run A --seed 42   # Full fine-tuning
python src/train.py --run D --seed 42   # LoRA r=8

# Or execute the complete 12-run sweep
python src/train.py --run-all
```

### 3. Verify README Numbers
```bash
python scripts/check_readme_numbers.py
python scripts/check_split_overlap.py
```

### 4. Generate Visualizations & Analysis
```bash
python src/plot.py
# Or open the notebook
jupyter notebook notebooks/experiments.ipynb
```

### 5. Launch Inference API
```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

### Renaming the Repository

The README refers to the project as `lora-distilbert-study`. To rename the GitHub repository to match:
1. Go to the repository on GitHub.
2. Click **Settings** (top right).
3. Under **Repository name**, change to `lora-distilbert-study`.
4. Click **Rename**.
5. GitHub will automatically redirect the old URL to the new one.

---

## 11. Optional Extension: Local LLM Tool Use

A tool-calling pipeline is provided in `src/tool_agent.py`. It integrates with a local Ollama instance (e.g. `llama3`) or functions via in-process routing. The model receives a user message, decides whether to call `classify_sentiment(text)`, queries the `/predict` API, and returns the result.

Evaluation over a 10-query benchmark (sentiment queries vs general reasoning/math queries) achieved **10/10 (100%) routing accuracy**, logging all events to `results/logs/tool_calls.log`.

---

## License
MIT License.
