# LoRA Reproduction and Controlled Study on DistilBERT with Deployed Inference Service

[![Tests](https://img.shields.io/badge/pytest-8%20passed-brightgreen.svg)](#test-suite)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-teal.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Paper](https://img.shields.io/badge/arXiv-2106.09685-B31B1B.svg)](https://arxiv.org/abs/2106.09685)

This repository implements LoRA (Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models”, ICLR 2022) from scratch in PyTorch and tests it on sequence classification with distilbert-base-uncased on GLUE SST-2. It includes:

A LoRA layer written without importing peft in src/lora.py, with tests against Hugging Face PEFT.
A controlled sweep comparing full fine-tuning with LoRA at ranks $r \in \{1, 4, 8\}$, using 3 seeds (42, 43, 44).
A 3-epoch follow-up for full fine-tuning and LoRA at $r=8$.
A FastAPI inference service with a Dockerfile. The container build has not yet been verified (see Section 8).
1. Results
1.1 One-Epoch Sweep

All runs used a fixed 8,000-example training subset of SST-2, batch size 32, 1 epoch, and seeds 42, 43, 44. Learning rates were fixed before any run: $2\times10^{-5}$ for full fine-tuning and $2\times10^{-4}$ for LoRA. LoRA used $\alpha = 2r$. Accuracy is on the SST-2 validation split (872 examples).

Source: results/results.csv. Verify with python scripts/check_readme_numbers.py.

Percentages use the base model’s 66,955,010 parameters as the denominator.

Run	Method	Rank $r$	Adapter params	Adapter-only %	Total trainable % (adapter + head)	Val accuracy (mean ± std)	Peak memory (MB)
A	Full FT	n/a	n/a	n/a	100.00%	90.29 ± 0.35%	1480.5
B	LoRA	1	18,432	0.0275%	0.91%	87.12 ± 0.52%	920.4
C	LoRA	4	73,728	0.1101%	0.99%	89.72 ± 0.46%	945.8
D	LoRA	8	147,456	0.2202%	1.10%	90.67 ± 0.37%	982.1

Parameter breakdown. LoRA is applied to q_lin and v_lin in all 6 DistilBERT layers, adding $18{,}432 \times r$ parameters. The classification head (pre_classifier and classifier) adds 592,130 parameters and is trained in every run.

Device. All 1-epoch runs recorded device cuda:0. GPU model: not recorded.

1.2 Three-Epoch Experiment

Run A (full fine-tuning) and Run D (LoRA, $r=8$) were retrained for 3 epochs with the same subset, batch size, learning rates, and seeds.

Source: results/results_3epoch.csv.

Method	1-epoch val acc	3-epoch val acc	Gap (3-epoch)
Full FT	90.29 ± 0.35%	88.15 ± 0.85%	—
LoRA r=8	90.67 ± 0.37%	87.27 ± 0.61%	—

The 3-epoch gap is full fine-tuning minus LoRA: $88.15 - 87.27 = +0.88$ points. Using the per-seed differences (1.38, 0.00, 1.26 points), a paired t-test gives $t \approx 2.0$ with 2 degrees of freedom and $p \approx 0.18$. Three seeds give little statistical power, so this difference cannot be distinguished from noise.

At 1 epoch, LoRA was $0.38$ points ahead, within seed variance. At 3 epochs, validation accuracy fell for both methods and full fine-tuning was ahead by 0.88 points.

Device. All six 3-epoch rows record device cuda. GPU model: [FILL: output of python -c "import torch; print(torch.cuda.get_device_name(0))" on the training session].

Time and memory are not compared across the two sweeps. The 1-epoch and 3-epoch sweeps have different device records, so time and memory are reported only within each sweep.

2. Findings
At 1 epoch, LoRA matched full fine-tuning. LoRA $r=8$ scored $90.67\% \pm 0.37\%$ against $90.29\% \pm 0.35\%$ for full fine-tuning. The difference is within seed variance, and the adapter is 0.22% of the base model.
At 3 epochs, full fine-tuning was ahead. The gap was 0.88 points, not significant with three seeds ($p \approx 0.18$).
Rank $r=1$ is too restrictive. LoRA at $r=1$ reached $87.12\% \pm 0.52\%$, below $r=4$ and $r=8$.
Peak memory was lower for LoRA at $r=1$. At 1 epoch, seed 42, LoRA $r=1$ used 920 MB against 1,480 MB for full fine-tuning, 37.8% lower. Part of this comes from gradient and optimizer state for frozen weights, which LoRA does not allocate. The GPU model for this sweep is not recorded, so treat the figure as indicative.
3. Evaluation Protocol
Reporting set: the SST-2 validation split (872 examples). All accuracy numbers come from this split.
Training subset: 8,000 examples drawn from the SST-2 training split (67,349 examples) with a seeded shuffle.
Tuning set: 1,000 examples at shuffled-training indices 8,000–8,999. This set exists for the overlap check only and was not used for any selection.
Learning-rate selection: fixed before any run. This was not re-verified by a separate experiment.
Overlap check: scripts/check_split_overlap.py confirms that the training subset and tuning set have no shared indices. Training and validation sentences were compared by exact string match, and 0 of the 8,000 training examples matched any of the 872 validation sentences ([FILL: script name used for the string-match check]). Exact matching does not rule out near-duplicates or substring overlap, since SST-2 training examples include phrases.
4. Dataset

SST-2 (Stanford Sentiment Treebank), binary sentiment, from GLUE:

Full training split: 67,349 examples
Validation split: 872 examples (used for reporting)
Training subset used: 8,000 examples
Tuning set: 1,000 examples (overlap check only)
5. LoRA Implementation (src/lora.py)
$$h = W_0 x + \frac{\alpha}{r} B A x$$
$W_0$ is frozen (requires_grad=False).
$A \in \mathbb{R}^{r \times d_{\text{in}}}$ is initialized with Kaiming uniform ($a=\sqrt{5}$).
$B \in \mathbb{R}^{d_{\text{out}} \times r}$ is initialized to zero, so the adapter is a no-op at the start of training.
merge() folds $\frac{\alpha}{r} BA$ into $W_0$ for inference. unmerge() reverses it.
Hugging Face PEFT comparison

tests/test_lora.py compares LoRALinear against Hugging Face PEFT with the same weights. Values measured by pytest -s:

Zero-init vs base linear: max abs diff 0.0
Merge: max abs diff 2.15e-06
Unmerge: max abs diff 4.77e-07
PEFT equivalence: max abs diff 0.0

[CONFIRM before publishing]: A PEFT difference of exactly 0.0 is only meaningful if the PEFT layer received an independent copy of the weights. If both layers share the same tensor, the test compares the computation with itself. Confirm in tests/test_lora.py that the weights are cloned, then keep or remove this sentence.

6. Repository Structure
lora-distilbert-study/
├── README.md
├── CHANGES.md               # Audit trail of changes
├── requirements.txt
├── Dockerfile
├── pytest.ini
├── src/
│   ├── lora.py              # LoRA layer (no peft import)
│   ├── data.py              # SST-2 loading, subsetting, tokenization
│   ├── model.py             # Model factory, LoRA injection, checkpointing
│   ├── train.py             # Training entry point for the sweep
│   ├── evaluate.py          # Accuracy and parameter counts
│   ├── serve.py             # FastAPI service
│   ├── plot.py              # Figures
│   └── tool_agent.py        # Optional local LLM tool-use extension
├── tests/
│   ├── test_lora.py
│   └── test_api.py
├── scripts/
│   ├── check_readme_numbers.py
│   ├── check_split_overlap.py
│   └── run_3epoch.py
├── configs/
│   └── sweep.yaml
├── results/
│   ├── results.csv          # 12 rows: 4 configs × 3 seeds
│   ├── results_3epoch.csv   # 6 rows: 2 configs × 3 seeds
│   ├── latency_benchmark.json
│   ├── figures/
│   └── logs/
├── notebooks/
│   └── experiments.ipynb
└── adapters/                # Best LoRA run (r=8)
7. Inference Service and Latency

src/serve.py loads distilbert-base-uncased, applies the trained adapter from adapters/, and merges it for inference.

GET /health: service status, model name, device, and adapter state.
POST /predict: takes {"text": "..."} and returns label, score, and class_id.

Example:

bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "An astonishing, poignant, and uplifting masterpiece."}'
json
{"label": "positive", "score": 0.98421, "class_id": 1}

CPU latency (100 sequential requests). Source: results/latency_benchmark.json.

Median: 54.37 ms
Mean: 56.98 ± 16.15 ms
95th percentile: 87.99 ms
99th percentile: 103.44 ms

Hardware: Intel Core i5 (4 physical cores, 8 logical threads), 11.8 GB RAM, CPU inference with merged LoRA weights.

8. Docker

The repository includes a Dockerfile. The container build and endpoint checks have not been run. The API tests run with FastAPI’s TestClient. To verify the container, run:

bash
docker build -t lora-distilbert:latest .
docker run -d --name lora-service -p 8000:8000 lora-distilbert:latest
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The pacing was agonizingly slow and boring."}'

Add the output of these commands here once they have run. Until then, describe the project as having a Dockerfile, not as a containerized deployment.

No cloud deployment is provided.

9. Tests
bash
python -m pytest tests/ -v -s

Run on Windows, Python 3.11.9, pytest 9.1.1:

tests/test_api.py::test_health_endpoint PASSED
tests/test_api.py::test_predict_positive_sentiment PASSED
tests/test_api.py::test_predict_negative_sentiment PASSED
tests/test_api.py::test_predict_empty_text_validation PASSED
tests/test_lora.py::test_zero_init PASSED          (max_abs_diff = 0.0)
tests/test_lora.py::test_shapes PASSED
tests/test_lora.py::test_merge PASSED              (merge_diff = 2.15e-06, unmerge_diff = 4.77e-07)
tests/test_lora.py::test_library_peft_equivalence PASSED  (max_abs_diff = 0.0)

8 passed in 27.95s

What the tests check:

Zero-init: a fresh LoRALinear matches the base layer.
Shapes: A is $(r, d_{\text{in}})$ and B is $(d_{\text{out}}, r)$.
Merge: output before and after merge() agrees within $10^{-5}$.
PEFT equivalence: matches Hugging Face PEFT with the same weights (see the [CONFIRM] note in Section 5).
/health returns 200 with the expected schema.
/predict returns the expected format for positive and negative sentences.
Empty input is rejected with 400 or 422.
10. Quickstart
bash
git clone https://github.com/lekha282006-ship-it/Document-LoRA-Reproduction-and-Controlled-Study-on-DistilBERT-with-a-Deployed-Inference-Service.git
cd Document-LoRA-Reproduction-and-Controlled-Study-on-DistilBERT-with-a-Deployed-Inference-Service
pip install -r requirements.txt

Run the sweep:

bash
python src/train.py --run A --seed 42     # full fine-tuning, 1 epoch
python src/train.py --run D --seed 42     # LoRA r=8, 1 epoch
python src/train.py --run-all             # full 1-epoch sweep
python scripts/run_3epoch.py              # 3-epoch follow-up (6 runs)

Verify the numbers:

bash
python scripts/check_readme_numbers.py
python scripts/check_split_overlap.py

Serve the model:

bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000

Renaming the repository. The project name in this README is lora-distilbert-study. To match, rename the repository in GitHub under Settings → Repository name. GitHub redirects the old URL. After renaming, update the clone URL above.

11. Optional: Local LLM Tool Use

src/tool_agent.py lets a local Ollama model decide whether to call classify_sentiment(text), which queries /predict. On a 10-query test (sentiment queries and general reasoning or math queries), routing was correct 10 times out of 10. Ten queries is a small sample, so treat this as a smoke test. Logs are in results/logs/tool_calls.log.

License

MIT
