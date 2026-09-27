# RunSleuth: Evidence-grounded ML Training Failure Diagnosis and Repair Verification

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Gemini-8E75B2)

RunSleuth uses an LLM agent to inspect training code, configuration, and metrics,
propose a small configuration fix, and verify it through bounded retraining.
A repair is accepted only when validation performance recovers.

## How It Works

- Record loss, accuracy, gradient norms, and parameter updates during training.
- Let the agent collect evidence through restricted, read-only diagnostic tools.
- Validate its structured diagnosis and evidence references before applying an
  allowed configuration change.
- Retrain from scratch under an epoch budget and check accuracy and loss.

Currently supports **high learning rate** and **missing optimizer step** faults
in a FashionMNIST CNN. A rule-based diagnostic baseline is also included.
Training uses 55,000 images from FashionMNIST's training set, with the remaining
5,000 held out for validation.

## Results

Using **Gemini 3.5 Flash-Lite**, the development suite passed **9/9 cases on the
first agent attempt** across seeds 7, 123, and 2026.

| Measure | Result |
|---|---:|
| Fault top-1 diagnosis | 6/6 |
| Repairs accepted after verification | 6/6 |
| Healthy self-comparisons with no action | 3/3 |

Repairs used at most **2 training epochs**. Acceptance required an accuracy gain
of at least 20 percentage points over the failed run, a shortfall of at most
3 points from the reference, and a validation-loss ratio of at most 1.25.

[Evaluation record](evaluations/results/20260926T205834269355Z.json)

These are controlled development cases, not a held-out benchmark. Diagnosis
uses a healthy reference; confidence is heuristic, and checking citation values
does not verify every explanation. The official FashionMNIST test set has not yet
been evaluated. Broader faults and datasets remain future work.

## Quick Start

Use Python 3.11 and a compatible PyTorch installation. The example configs use
CUDA. Commands below use **Windows CMD**.

```bat
python -m pip install -e ".[dev]"
python -m runsleuth --config configs/clean.json
python -m runsleuth --config configs/high_learning_rate.json
```

Set your API key and replace the run paths with the directories printed above:

```bat
set "GEMINI_API_KEY=YOUR_API_KEY"
set "GEMINI_MODEL=gemini-3.8-flash"
set "REFERENCE_RUN=artifacts/runs/clean-REPLACE_WITH_TIMESTAMP"
set "FAILED_RUN=artifacts/runs/high-lr-REPLACE_WITH_TIMESTAMP"
python -m runsleuth.agent_diagnose --reference-run "%REFERENCE_RUN%" --candidate-run "%FAILED_RUN%"
```

After diagnosis, use the saved report to run repair verification:

```bat
set "AGENT_REPORT=artifacts/agent_runs/REPLACE_WITH_TIMESTAMP/agent_report.json"
python -m runsleuth.agent_repair_and_verify --agent-report "%AGENT_REPORT%" --reference-run "%REFERENCE_RUN%" --failed-run "%FAILED_RUN%" --max-epochs 2
```

Generated configurations, metrics, checkpoints, and reports are saved under
`artifacts/` and excluded from Git. Keep API keys out of source control.

## Evaluation and Checks

Run a fresh three-seed evaluation, including training and LLM calls:

```bat
python -m runsleuth.batch_evaluate --seeds 7 123 2026 --device cuda
```

Run code checks:

```bat
python -m pytest
ruff check .
git diff --check
```
