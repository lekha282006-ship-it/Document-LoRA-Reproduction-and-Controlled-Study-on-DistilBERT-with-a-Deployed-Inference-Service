# LoRA Reproduction and Controlled Study on DistilBERT with Deployed Inference Service

[![Tests](https://img.shields.io/badge/pytest-8%20passed-brightgreen.svg)](#test-suite)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Paper](https://img.shields.io/badge/arXiv-2106.09685-B31B1B.svg)](https://arxiv.org/abs/2106.09685)

This repository reproduces the core claim of Hu et al., [“LoRA: Low-Rank Adaptation of Large Language Models”](https://arxiv.org/abs/2106.09685) (ICLR 2022), on a sequence classification task using `distilbert-base-uncased` fine-tuned on GLUE SST-2. It features:
- A scratch LoRA layer implementation written directly in PyTorch (without importing `peft` in `src/lora.py`).
- Exact mathematical and numerical parity tests against Hugging Face PEFT.
- A controlled hyperparameter sweep comparing Full Fine-Tuning against LoRA ranks $r \in \{1, 4, 8\}$ across 3 random seeds (`42, 43, 44`).
- A production-grade Dockerized FastAPI inference microservice with low-latency CPU inference.

---

## 1. Experimental Results & Controlled Study

All runs were evaluated under identical conditions on a fixed 8,000-example subset of SST-2 with batch size 32, 1 epoch, and seed control (`42, 43, 44`). The learning rate was selected once per method and held constant ($2\times 10^{-5}$ for full fine-tuning, $2\times 10^{-4}$ for LoRA). The LoRA scaling factor used the canonical $\alpha = 2r$ rule.

### Summary Table (Mean ± Standard Deviation over 3 Seeds)

| Run | Method | Rank $r$ | LoRA $\alpha$ | Trainable Params | % Trainable | Val Accuracy (Mean ± Std %) | Train Time (s) | Peak VRAM (MB) | Seeds |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | **Full FT** | n/a | n/a | 66,955,010 | 100.00% | **90.29 ± 0.35%** | 38.5 ± 0.3s | 1480.5 MB | 42, 43, 44 |
| **B** | **LoRA** | 1 | 2.0 | 610,562 | 0.91% | **87.12 ± 0.52%** | 24.2 ± 0.2s | 920.4 MB | 42, 43, 44 |
| **C** | **LoRA** | 4 | 8.0 | 665,858 | 0.99% | **89.72 ± 0.46%** | 25.1 ± 0.2s | 945.8 MB | 42, 43, 44 |
| **D** | **LoRA** | 8 | 16.0 | 739,586 | 1.10% | **90.67 ± 0.37%** | 26.4 ± 0.2s | 982.1 MB | 42, 43, 44 |

> **Parameter Breakdown Note**:
> DistilBERT has 6 transformer layers. Injecting LoRA into `q_lin` and `v_lin` adds $2 \times (2 \times 768 \times r)$ parameters per layer ($18,432 \times r$ LoRA parameters total). The classification head (`pre_classifier` + `classifier`) adds 592,130 parameters:
> - $r=1$: 18,432 adapter params + 592,130 head params = **610,562 trainable** (0.91% of backbone)
> - $r=4$: 73,728 adapter params + 592,130 head params = **665,858 trainable** (0.99% of backbone)
> - $r=8$: 147,456 adapter params + 592,130 head params = **739,586 trainable** (1.10% of backbone)

---

## 2. Findings

1. **Reproduction Confirmed**: LoRA at rank $r=8$ matches and slightly exceeds full fine-tuning ($90.67\% \pm 0.37\%$ vs $90.29\% \pm 0.35\%$) while adapting only **1.10%** of total parameters; the performance gap is well within seed variance.
2. **Rank Bottleneck at $r=1$**: LoRA at rank $r=1$ failed to match full fine-tuning, reaching only $87.12\% \pm 0.52\%$, demonstrating that a single rank subspace is too restrictive for DistilBERT query/value projections.
3. **Memory & Efficiency Savings**: LoRA reduced peak GPU memory consumption by **37.8%** (920 MB vs 1480 MB) and accelerated per-epoch runtime by **31.4%**, as backpropagation bypasses the 66M frozen backbone parameters.

---

## 3. Dataset Specifications

The model is trained and evaluated on the Stanford Sentiment Treebank (SST-2) binary sentiment classification task from GLUE:
- **Full Training Set**: 67,349 examples
- **Validation Set**: 872 examples
- **Test Set**: 1,821 examples
- **Controlled Study Training Subset**: 8,000 examples (sampled deterministically per seed)

---

## 4. Scratch LoRA Implementation (`src/lora.py`)

LoRA injects rank decomposition matrices into frozen linear layers:
$$h = W_0 x + \frac{\alpha}{r} B A x$$
- $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is completely frozen ($\text{requires\_grad}=\text{False}$).
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$ is initialized with Kaiming uniform initialization ($a = \sqrt{5}$).
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$ is initialized to all zeros, guaranteeing the adapter starts as an exact identity/no-op.
- A `merge()` method directly computes $W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA$ into the weight tensor for zero-overhead inference.
- An `unmerge()` method recovers separate matrices if continued training is desired.

### Hugging Face PEFT Equivalence
In `tests/test_lora.py`, our custom `LoRALinear` layer is compared against Hugging Face `peft`'s `LoraLayer` with identical weights. Forward pass outputs match down to machine precision (`max abs difference = 0.0`), confirming mathematical and implementation parity.

---

## 5. Repository Structure

```
lora-distilbert-study/
├── README.md              # Project documentation, findings, and reproduction guide
├── requirements.txt       # Pinned dependencies
├── Dockerfile             # Production container definition
├── pytest.ini             # Pytest configuration
├── src/
│   ├── __init__.py
│   ├── lora.py            # LoRA layer written from scratch (no peft import)
│   ├── data.py            # SST-2 dataset loading, subsetting, and tokenization
│   ├── model.py           # DistilBERT model factory, LoRA injection, and checkpointing
│   ├── train.py           # Unified training entry point for all sweep configs
│   ├── evaluate.py        # Accuracy evaluation, parameter counts, and latency benchmarking
│   ├── serve.py           # FastAPI inference microservice
│   ├── plot.py            # Generates publication-ready figures
│   └── tool_agent.py      # Local LLM tool-calling extension
├── tests/
│   ├── test_lora.py       # Zero-init, shape, merge, and HF PEFT equivalence tests
│   └── test_api.py        # /health and /predict endpoint tests
├── notebooks/
│   └── experiments.ipynb  # Interactive sweep visualization and analysis
├── configs/
│   └── sweep.yaml         # Configuration file holding all sweep hyperparameters
├── results/
│   ├── results.csv        # Individual metrics recorded for every run and seed
│   ├── latency_benchmark.json # 100-request CPU latency benchmark metrics
│   ├── figures/           # Generated visualization plots
│   └── logs/              # Training run logs and tool call logs
└── adapters/              # Saved LoRA weights & metadata for best run (Run D, r=8)
    ├── adapter_config.json
    └── adapter_model.pt
```

---

## 6. Inference Service & Latency Benchmark

The inference service is exposed via FastAPI in `src/serve.py`. It loads `distilbert-base-uncased`, injects the trained LoRA adapter from `adapters/`, and merges the weights for zero-latency inference overhead.

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
Inference latency was benchmarked on a single request stream across 100 consecutive queries on CPU:
- **Median Latency**: **54.37 ms**
- **Mean Latency**: **56.98 ± 16.15 ms**
- **95th Percentile**: **87.99 ms**
- **99th Percentile**: **103.44 ms**
- **Min / Max**: 37.56 ms / 117.64 ms

**Benchmarking Hardware**:
- **Processor**: Intel(R) Core(TM) i5 (4 physical cores, 8 logical threads)
- **RAM**: 11.8 GB
- **Operating System**: Windows 10 / Linux compatible
- **Inference Mode**: CPU single-batch forward pass with merged LoRA weights

---

## 7. Docker Deployment

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

---

## 8. Test Suite

Run the full automated test suite (all 8 tests passing):
```bash
python -m pytest tests/ -v
```

Tests verify:
1. `test_zero_init`: Fresh LoRALinear outputs match base linear layer exactly.
2. `test_shapes`: Matrix dimensions and tensor shapes match $(r, d_{\text{in}})$ and $(d_{\text{out}}, r)$.
3. `test_merge`: Output before and after `merge()` match within $1\times 10^{-5}$.
4. `test_library_peft_equivalence`: Custom layer matches Hugging Face PEFT on identical parameters with machine precision ($0.0$ difference).
5. `test_health_endpoint`: FastAPI `/health` endpoint responds with HTTP 200 and schema.
6. `test_predict_positive_sentiment`: FastAPI `/predict` accurately outputs valid sentiment format.
7. `test_predict_negative_sentiment`: Negative sentiment classified with valid probability.
8. `test_predict_empty_text_validation`: Invalid empty inputs rejected with HTTP 400/422.

---

## 9. Quickstart: Reproduce from Scratch

### 1. Clone & Install
```bash
git clone https://github.com/your-username/lora-distilbert-study.git
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

### 3. Generate Visualizations & Analysis
```bash
python src/plot.py
# Or open the notebook
jupyter notebook notebooks/experiments.ipynb
```

### 4. Launch Inference API
```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```

---

## 10. Optional Extension: Local LLM Tool Use (Phase 7)

A tool-calling pipeline is provided in `src/tool_agent.py`. It integrates with a local Ollama instance (e.g. `llama3`) or functions via in-process routing. The model receives a user message, decides whether to call `classify_sentiment(text)`, queries the `/predict` API, and returns the result.

Evaluation over a 10-query benchmark (sentiment queries vs general reasoning/math queries) achieved **10/10 (100%) routing accuracy**, logging all events to `results/logs/tool_calls.log`.

---

## License
MIT License.
