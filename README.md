# RunSleuth: Evidence-grounded ML Training Failure Diagnosis and Repair Verification

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)
![LLM](https://img.shields.io/badge/LLM-Gemini%20%7C%20Ollama-8E75B2)
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
3. **Review** with an LLM (Gemini, or a local model through Ollama) that must
   cite the evidence; it cannot overrule the deterministic verdict.
4. **Verify** the minimal repair with bounded retraining and a no-regression gate
   against a healthy run.

## Results

Camelyon17-WILDS (20K training patches; in-distribution and unseen-hospital
validation), ResNet18 and DeiT-small, author-injected faults. Development used
seeds 7 and 2026. Two held-out seeds were then drawn at random after the
signatures, LLM prompt and repair gate were frozen
([draw](evaluations/heldout_seeds.json)). Cells read development · held-out.

| Fault | Decisive evidence | Silent in metrics* | Repair accepted |
|---|---|---:|---:|
| Stale optimizer binding | Head gets gradients but never updates | 1/2 · 2/2 | 2/2 · 2/2 |
| Frozen classifier head | Head untrainable; the reference trains it | 1/2 · 2/2 | 2/2 · 2/2 |
| Learning rate 100x too high | Effective rate inferred from AdamW's first update | 0/2 · 0/2 | 2/2 · 2/2 |
| Missing `optimizer.step()` | Forwards without optimizer steps | 0/2 · 0/2 | 2/2 · 2/2 |
| Train/eval normalization mismatch | Training and evaluation input statistics disagree | 0/2 · 0/2 | 2/2 · 1/2† |
| ImageNet head kept for 2 labels | 1000 head outputs for 2 classes | 1/2 · 0/2 | 2/2 · 2/2 |
| Frozen patch embedding | Part of the backbone untrainable | 2/2 · 1/2 | 2/2 · 2/2 |

- **Diagnosis:** with a healthy reference, 46/46 development and 32/32 held-out
  runs correct; without one, 40/46 and 28/32, the rest correctly deferred to a
  reference. No false positives on 26 + 18 healthy runs
  ([development](evaluations/results/signature-matching-20261008.json),
  [held-out](evaluations/results/signature-matching-heldout-seeds-85302029-1948666596.json)).
- \*The faulty run stayed within the no-regression thresholds (1 point accuracy,
  10% loss) of the healthy run, so a check on validation metrics alone would not
  flag it: 10 of 28 faulty runs.
- **Repair gate:** a repair must not regress against the healthy run. The first
  gate also compared against the faulty run and rejected the ImageNet-head and
  patch-embedding repairs on both development seeds, although they restored the
  clean model bitwise. It was revised after seeing this
  ([re-scored](evaluations/results/repair-gate-v2-rescore.json)), then applied
  unchanged to the held-out seeds
  ([held-out](evaluations/results/repairs-heldout-seeds-85302029-1948666596.json)).
- †The DeiT checkpoint's image processor normalizes with 0.5, contradicting the
  ImageNet values in its model card; RunSleuth's input statistics catch the
  resulting mismatch. The repair aligns evaluation with the processor, which
  trained worse than ImageNet normalization on one held-out seed (2.8 points lower
  out-of-distribution accuracy), so the gate rejected it. The development seeds had
  favored 0.5 (+2.9 and +2.6 points); across four seeds neither is consistently
  better.

**LLM review.** Each reviewer saw the matcher's evidence for every run, with and
without a reference, one sampled reply per review. Prompt 1 was evaluated on the
development runs. Prompt 2, written after its errors and frozen before the
held-out draw, was evaluated on the held-out runs
([development](evaluations/results/llm-review-eval.json),
[held-out](evaluations/results/llm-review-eval-heldout.json)).

| Reviewer | Runs, prompt | Valid reply on first try | Agrees with matcher | Mean latency |
|---|---|---:|---:|---:|
| Gemini 3.5 Flash-Lite (API) | development, 1 | 92/92 | 89/92 | 6 s |
| Gemini 3.5 Flash-Lite (API) | held-out, 2 | 64/64 | 63/64 | 2 s |
| Qwen3 8B (local, Ollama) | development, 1 | 86/92 | 91/91‡ | 72 s |
| Qwen3 8B (local, Ollama) | held-out, 2 | 51/64 | 52/52‡ | 73 s |

- Gemini's three development disagreements were all on faulty runs and all wrong:
  twice it held a high learning rate pending a reference the signature does not
  need, and once it called a frozen patch embedding normal partial fine-tuning.
  Prompt 2 states both rules. Neither error recurred on held-out runs; Gemini's one
  disagreement there named a frozen head that only a reference run can confirm.
- Citation checks caught Qwen citing evidence that does not exist (prompt 1) and,
  on held-out runs, 12 replies with a wrong learning rate. Ollama's
  schema-constrained decoding rejects the leading zero in the exponent Python
  writes (`e-05`), so Qwen's exact copy was cut to `e-0`.
- ‡Of completed reviews. Reviews that never passed the checks (1 and 12, all on
  healthy runs) left the matcher's diagnosis standing. The matcher stays final,
  so no disagreement or failed review changed a diagnosis.

Results cover four seeds, two of them held out, with development thresholds; they
are not a statistical benchmark.

## Getting Started

### Requirements

- Python 3.11 and an NVIDIA GPU with 8 GB of memory (tested on an RTX 5060 Ti)
- About 20 GB of disk space for the cached Camelyon17 data
- On Windows, a clone path under about 120 characters (run files must stay below
  the 260-character limit) or long paths enabled
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

Or review locally with [Ollama](https://ollama.com) by adding `--provider ollama`.
Quit the Ollama tray app, start the server with a longer context than its
default, and pull the model in a second terminal:

```bat
set "OLLAMA_CONTEXT_LENGTH=12288"
ollama serve
```

```bat
ollama pull qwen3:8b
set "OLLAMA_MODEL=qwen3:8b"
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
set "SEED=artifacts/optimizer_sweeps/<sweep>/s1/baseline/<baseline-run>"
python -m runsleuth.camelyon_frozen_head_training --reference-run "%SEED%"
python -m runsleuth.camelyon_config_faults --reference-run "%SEED%"
python -m runsleuth.camelyon_vit --reference-run "%SEED%"
```

Score the signature matcher on the experiment reports these commands print:

```bat
python -m runsleuth.signature_matching evaluate <experiment-report.json> ...
```

Compare LLM reviewers on the same cases. Runs resume where they stopped; add
`--pause 5 --error-wait 60` for the Gemini free tier:

```bat
python -m runsleuth.llm_review_eval run --provider ollama --model qwen3:8b --from-record evaluations/results/signature-matching-20261008.json
python -m runsleuth.llm_review_eval summarize artifacts/llm_review_eval/ollama-qwen3-8b-prompt-v2 ...
```

Draw held-out seeds once and commit the seeds file, then train and evaluate them
in one resumable command:

```bat
python -m runsleuth.heldout draw --count 2
python -m runsleuth.heldout run --reference-run "%BASELINE%"
```

### Tests

```bat
python -m pytest
```

## License

[MIT](LICENSE). Datasets and pretrained weights keep their own licenses.
