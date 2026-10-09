# RunSleuth: Evidence-grounded ML Training Failure Diagnosis and Repair Verification

[![CI](https://github.com/Kaiden118/RunSleuth-Evidence-Grounded-ML-Training-Failure-Diagnosis-and-Repair-Verification/actions/workflows/ci.yml/badge.svg)](https://github.com/Kaiden118/RunSleuth-Evidence-Grounded-ML-Training-Failure-Diagnosis-and-Repair-Verification/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)
![LLM](https://img.shields.io/badge/LLM-Gemini%20%7C%20Ollama-8E75B2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

RunSleuth finds silent bugs in PyTorch training runs, the kind that never crash and
barely move accuracy, and accepts a fix only after retraining verifies it.

It records evidence that metrics lack (gradients, parameter updates, trainability,
optimizer membership, train/eval mode, learning rates, input statistics), matches it
against declarative failure signatures, and suggests a minimal repair. An LLM
explains the diagnosis, but must cite the evidence and cannot overrule it.

## How It Works

1. **Monitor** a training loop with `RunMonitor`, or the Hugging Face `Trainer` with
   `RunSleuthCallback`.
2. **Diagnose** the run against the signature library, optionally with a healthy
   reference run of the same setup.
3. **Explain** with an LLM (Gemini, or a local model through Ollama); every cited
   value is checked.
4. **Verify** the repair: re-diagnose the retrained run and require no regression
   against the healthy run.

## Results

Camelyon17-WILDS with ResNet18 and DeiT-small and nine author-injected faults, each
run on two development seeds and two or more held-out seeds. The held-out seeds were
drawn at random after the signatures, LLM prompt and repair gate were frozen. Cells
read development · held-out.

| Fault | Decisive evidence | Silent in metrics* | Repair accepted |
|---|---|---:|---:|
| Stale optimizer binding | Head gets gradients but never updates | 1/2 · 3/4 | 2/2 · 4/4 |
| Frozen classifier head | Head untrainable; the reference trains it | 1/2 · 2/2 | 2/2 · 2/2 |
| Learning rate 100x too high | Rate inferred from AdamW's first update | 0/2 · 0/2 | 2/2 · 2/2 |
| Missing `optimizer.step()` | Forwards without optimizer steps | 0/2 · 0/2 | 2/2 · 2/2 |
| Train/eval normalization mismatch | Training and evaluation inputs differ | 0/2 · 0/2 | 2/2 · 1/2† |
| ImageNet head kept for 2 labels | 1000 head outputs for 2 classes | 1/2 · 0/2 | 2/2 · 2/2 |
| Frozen patch embedding | Part of the backbone untrainable | 2/2 · 1/2 | 2/2 · 2/2 |
| Training in eval mode | Steps train on eval-mode forwards; BatchNorm statistics frozen | 0/2 · 0/2 | 2/2 · 2/2 |
| Per-epoch LR schedule stepped every batch | The rate falls and rises again within an epoch | 0/2 · 0/2 | 2/2 · 2/2 |

- **Diagnosis:** with a healthy reference, 56/56 development and 42/42 held-out runs
  correct. Without one, faults that only a reference can confirm wait for one instead
  of guessing. No false positives on 56 healthy runs.
- \*Within the no-regression thresholds of the healthy run, so validation metrics
  alone would miss it: 11 of 38 faulty runs.
- †The repair evaluates with the DeiT processor's 0.5 normalization (its model card
  says ImageNet's), which trained worse on that seed; across four seeds neither
  normalization is consistently better.
- The repair gate was revised once, after the development runs; the held-out seeds
  used it unchanged.

**LLM review** of the held-out runs, one reply per review:

| Reviewer | Valid reply on first try | Agrees with matcher | Latency |
|---|---:|---:|---:|
| Gemini 3.5 Flash-Lite (API) | 64/64 | 63/64 | 2 s |
| Qwen3 8B (local, Ollama) | 51/64 | 52/52 | 73 s |

The matcher stays final, so no disagreement or failed reply changed a diagnosis.
Qwen's 12 failures came from Ollama's constrained decoding cutting Python's `e-05`
exponents, since fixed.

All numbers come from the JSON [records](evaluations/results), with development
thresholds on four seeds; they are not a statistical benchmark.

## Getting Started

Python 3.11. Training needs an NVIDIA GPU with 8 GB (tested on an RTX 5060 Ti) and
about 20 GB of disk; diagnosis does not. Commands use Windows CMD; on Windows, keep
the clone path under about 120 characters or enable long paths.

```bat
git clone https://github.com/Kaiden118/RunSleuth-Evidence-Grounded-ML-Training-Failure-Diagnosis-and-Repair-Verification.git
cd RunSleuth-Evidence-Grounded-ML-Training-Failure-Diagnosis-and-Repair-Verification
conda create -n runsleuth python=3.11 -y
conda activate runsleuth
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[dev,llm,hf,camelyon]"
```

Pick the PyTorch index for your CUDA version on
[pytorch.org](https://pytorch.org/get-started/locally/).

### Try it

The [examples](examples) are a held-out run with a frozen classifier head whose
metrics look healthy, its clean reference, and the same run repaired:

```bat
runsleuth diagnose --run examples/frozen_head/run_report.json --reference examples/clean/run_report.json --no-llm
runsleuth verify --candidate examples/frozen_head_repaired/run_report.json --reference examples/clean/run_report.json
```

Or with Docker, without installing anything else:

```bat
docker build --target runtime -t runsleuth .
docker run --rm -v "%cd%/examples:/work/examples:ro" runsleuth diagnose --run examples/frozen_head/run_report.json --reference examples/clean/run_report.json --no-llm
```

### Use it on your training

```python
from runsleuth.run_monitor import RunMonitor

monitor = RunMonitor(model, optimizer, "runs/my-run", head="fc")
for epoch in range(epochs):
    with monitor.epoch():
        ...  # your training loop, unchanged
    monitor.log(id_validation_accuracy=accuracy, id_validation_loss=loss)
monitor.close()
```

With the Hugging Face `Trainer`, add
`callbacks=[RunSleuthCallback("runs/my-run", head="classifier")]` from
`runsleuth.hf_callback`. Then diagnose, repair, retrain and verify:

```bat
runsleuth diagnose --run runs/my-run/run_report.json --reference runs/healthy/run_report.json
runsleuth verify --candidate runs/my-run-fixed/run_report.json --reference runs/healthy/run_report.json
```

`verify` exits with status 1 when it rejects the repair. For the LLM explanation, set
`GEMINI_API_KEY` and `GEMINI_MODEL`, or pass `--provider ollama` with `OLLAMA_MODEL`
set (start `ollama serve` with `OLLAMA_CONTEXT_LENGTH=12288`); `--no-llm` skips it.

### Run the demo

Set `data_dir` in [`configs/camelyon17_clean.json`](configs/camelyon17_clean.json),
then download the data, train a baseline and a healthy reference, and let the demo
inject a hidden fault, diagnose it, repair it, verify the repair and reveal it:

```bat
python -m runsleuth.camelyon --download --prepare-only
python -m runsleuth.camelyon
set "BASELINE=artifacts/camelyon_runs/camelyon17-clean-<timestamp>"
runsleuth demo --baseline "%BASELINE%" --fault clean --no-llm
runsleuth demo --baseline "%BASELINE%" --reference artifacts/demo/demo-<timestamp>/faulty/run_report.json --fault random --blind
```

### Reproduce the experiments

```bat
python -m runsleuth.camelyon_optimizer_sweep --reference-run "%BASELINE%" --execute
python -c "from transformers import AutoModelForImageClassification as M; M.from_pretrained('facebook/deit-small-patch16-224')"
set "SEED=artifacts/optimizer_sweeps/<sweep>/s1/baseline/<baseline-run>"
python -m runsleuth.camelyon_frozen_head_training --reference-run "%SEED%"
python -m runsleuth.camelyon_config_faults --reference-run "%SEED%"
python -m runsleuth.camelyon_vit --reference-run "%SEED%"
python -m runsleuth.camelyon_loop_faults --reference-run "%SEED%"
python -m runsleuth.signature_matching evaluate <experiment-report.json> ...
python -m runsleuth.heldout draw --count 2
python -m runsleuth.heldout run --reference-run "%BASELINE%"
python -m runsleuth.llm_review_eval run --provider ollama --model qwen3:8b --from-record <signature-matching record>
```

The sweep creates the seed 7 and 2026 baselines; `heldout draw` records new seeds
with the commit and the hashes of everything frozen, and `heldout run` trains and
scores them, resuming after an interruption.

### Tests

```bat
python -m pytest
```

[GitHub Actions](.github/workflows/ci.yml) lints and runs every test on CPU for each
push, then builds the [Docker image](Dockerfile), runs the tests inside it and
diagnoses the example there. The tests use tiny models and fake data.

## License

[MIT](LICENSE). Datasets and pretrained weights keep their own licenses.
