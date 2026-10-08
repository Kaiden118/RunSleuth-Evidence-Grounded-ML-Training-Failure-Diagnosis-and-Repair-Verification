# RunSleuth: Evidence-grounded ML Training Failure Diagnosis and Repair Verification

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)
![LLM](https://img.shields.io/badge/LLM-Gemini-8E75B2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

RunSleuth diagnoses silent PyTorch training faults from structural telemetry and
accepts a repair only after bounded retraining verifies it.

Many faults never crash and barely change accuracy. A frozen classifier head and
a stale optimizer binding, for example, leave bitwise-identical backbones and
identical metrics, yet need different repairs. RunSleuth separates them with
evidence that metrics lack: gradients, updates, trainability, optimizer
membership and input statistics.

## How It Works

1. **Instrument** any training loop with `RunMonitor`, or the Hugging Face
   `Trainer` with `RunSleuthCallback`.
2. **Diagnose** by matching the run against declarative failure signatures, with
   or without a healthy reference run.
3. **Review** with an LLM (Gemini) that must cite the evidence; it cannot
   overrule the deterministic verdict.
4. **Verify** the minimal repair with bounded retraining and a no-regression gate.

## Results

Camelyon17-WILDS (20K training patches; in-distribution and unseen-hospital
validation), ResNet18 and DeiT-small, seeds 7 and 2026, author-injected faults.

| Fault | Decisive evidence | Repair accepted |
|---|---|---:|
| Stale optimizer binding | Head gets gradients but never updates | 2/2 |
| Frozen classifier head | Head untrainable; the reference trains it | 2/2 |
| Learning rate 100x too high | Effective rate inferred from AdamW's first update | 2/2 |
| Missing `optimizer.step()` | Forwards without optimizer steps | 2/2 |
| Train/eval normalization mismatch | Training and evaluation input statistics disagree | 2/2 |
| ImageNet head kept for 2 labels | 1000 head outputs for 2 classes | 0/2* |
| Frozen patch embedding | Part of the backbone untrainable | 0/2* |

- **Diagnosis:** 46/46 runs correct with a healthy reference and 40/46 without
  (the other 6 are correctly deferred to a reference), with no false positives
  on 26 healthy runs ([record](evaluations/results/signature-matching-20261008.json)).
- \*These repairs restored the clean model bitwise, but the faulty runs trained
  as well or better, so the no-regression gate rejected them.
- The DeiT checkpoint's Hugging Face image processor contradicts its model card
  on normalization; RunSleuth's input statistics catch the resulting mismatch.

Results cover two seeds and development thresholds, not a statistical benchmark.

## Getting Started

### Requirements

- Python 3.11 and an NVIDIA GPU with 8 GB of memory (tested on an RTX 5060 Ti)
- About 20 GB of disk space for the cached Camelyon17 data
- Commands below use Windows CMD

### Installation

```bat
git clone https://github.com/Kaiden118/RunSleuth-Evidence-Grounded-ML-Training-Failure-Diagnosis-and-Repair-Verification.git
cd RunSleuth-Evidence-Grounded-ML-Training-Failure-Diagnosis-and-Repair-Verification
conda create -n runsleuth python=3.11 -y
conda activate runsleuth
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[dev,llm,hf,camelyon]"
```

Choose the PyTorch index that matches your CUDA version on
[pytorch.org](https://pytorch.org/get-started/locally/).

### 1. Prepare the data and a baseline

Set `data_dir` in [`configs/camelyon17_clean.json`](configs/camelyon17_clean.json)
to a folder for the dataset cache, then:

```bat
python -m runsleuth.camelyon --download --prepare-only
python -m runsleuth.camelyon
```

The second command trains a healthy ResNet18 baseline and saves it as
`artifacts/camelyon_runs/camelyon17-clean-<timestamp>`.

### 2. Run the demo

```bat
set "BASELINE=artifacts/camelyon_runs/camelyon17-clean-<timestamp>"
python -m runsleuth.demo --baseline "%BASELINE%" --fault clean --no-llm
python -m runsleuth.demo --baseline "%BASELINE%" --reference artifacts/demo/demo-<timestamp>/faulty/run_report.json --fault random --blind --no-llm
```

The first command trains a healthy reference run. The second injects a randomly
chosen fault, hides it, diagnoses the run, applies the repair, retrains to verify
it, and finally reveals the fault (about 7 minutes). Reports are written to
`artifacts/demo/`.

To add the LLM review, set your key and drop `--no-llm`:

```bat
set "GEMINI_API_KEY=your-api-key"
set "GEMINI_MODEL=gemini-3.8-flash"
```

### 3. Diagnose your own training

Wrap each epoch of your loop with `RunMonitor`:

```python
from runsleuth.run_monitor import RunMonitor

monitor = RunMonitor(model, optimizer, "runs/my-run", head="fc")
for epoch in range(epochs):
    with monitor.epoch():
        ...  # your training loop, unchanged
    monitor.log(id_validation_accuracy=accuracy)
monitor.close()
```

With the Hugging Face `Trainer`, pass
`callbacks=[RunSleuthCallback("runs/my-run", head="classifier")]` from
`runsleuth.hf_callback` instead. Then diagnose the run, optionally against a
healthy run of the same setup:

```bat
python -m runsleuth.diagnose_run --run runs/my-run/run_report.json --reference runs/healthy/run_report.json --no-llm
```

### 4. Reproduce the experiments

Create the seed 7 and 2026 baselines, download the DeiT weights once, then run
each experiment on a seed baseline:

```bat
python -m runsleuth.camelyon_optimizer_sweep --reference-run "%BASELINE%" --execute
python -c "from transformers import AutoModelForImageClassification as M; M.from_pretrained('facebook/deit-small-patch16-224')"
set "SEED=artifacts/optimizer_sweeps/<sweep>/seed-7/baseline/<seed-7-baseline>"
python -m runsleuth.camelyon_frozen_head_training --reference-run "%SEED%"
python -m runsleuth.camelyon_config_faults --reference-run "%SEED%"
python -m runsleuth.camelyon_vit --reference-run "%SEED%"
```

Score the signature matcher on the experiment reports these commands print:

```bat
python -m runsleuth.signature_matching evaluate <experiment-report.json> ...
```

### Tests

```bat
python -m pytest
```

## License

[MIT](LICENSE). Datasets and pretrained weights keep their own licenses.
