# RunSleuth: Evidence-grounded ML Training Failure Diagnosis and Repair Verification

[![CI](https://github.com/Kaiden118/RunSleuth-Evidence-Grounded-ML-Training-Failure-Diagnosis-and-Repair-Verification/actions/workflows/ci.yml/badge.svg)](https://github.com/Kaiden118/RunSleuth-Evidence-Grounded-ML-Training-Failure-Diagnosis-and-Repair-Verification/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python\&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface\&logoColor=black)
![LLM](https://img.shields.io/badge/LLM-Gemini%20%7C%20GPT%20%7C%20Claude%20%7C%20DeepSeek%20%7C%20Qwen%20%7C%20Muse-8E75B2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

RunSleuth checks the repairs LLM agents make to medical-imaging training code. A
training run can be wrong without crashing and with accuracy barely moved; RunSleuth
accepts an agent's "fixed" only when evidence from the run supports it, and otherwise
rejects the repair or sends it to review.

![How RunSleuth works](docs/pipeline.svg)

The repository has three parts:

1. **Diagnosis and repair verification** of PyTorch training runs: silent faults in
   ResNet18 and DeiT-small (Hugging Face) on Camelyon17-WILDS pathology patches.
2. **Agent study:** does RunSleuth's evidence help LLM agents repair training scripts,
   or only catch their mistakes? 22 tasks, 5 conditions, 2 models.
3. **Repair verification** (in progress): can acceptance catch wrong agent repairs
   without rejecting correct ones written differently, at bounded cost? Six run-time
   tasks on Camelyon17 and HAM10000, repairs by five LLM agents.

## Install

```bash
git clone https://github.com/Kaiden118/RunSleuth-Evidence-Grounded-ML-Training-Failure-Diagnosis-and-Repair-Verification.git
cd RunSleuth-Evidence-Grounded-ML-Training-Failure-Diagnosis-and-Repair-Verification
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
pip install -e ".[dev,llm,hf,camelyon]"
```

Python 3.11. Diagnosis runs on CPU; training needs an NVIDIA GPU with 8 GB. Or use
Docker: `docker build --target runtime -t runsleuth .`

## Quick Start

A ResNet18 whose classification head was frozen by mistake reaches 98.2% validation
accuracy, above the healthy run's 97.9%. Its metrics pass every check; its evidence
does not (output shortened):

```bash
runsleuth diagnose --run examples/frozen_head/run_report.json --reference examples/clean/run_report.json --no-llm
runsleuth verify --candidate examples/frozen_head/run_report.json --reference examples/clean/run_report.json
```

```text
Diagnosis: frozen_head  (decided by signature_matcher)
Evidence:
  + head_max_gradient_fraction = 0.0 (== 0)
  + head_max_update_norm = 0.0 (== 0)
  + reference_head_min_trainable_fraction = 1.0 (== 1)
Decision: rejected
No regression against the reference:
  + id accuracy_drop = -0.003 (<= 0.01)
  + ood accuracy_drop = 0.005 (<= 0.01)
Reason: re-diagnosis against the reference found frozen_head
```

To record your own run, wrap the training loop:

```python
from runsleuth.run_monitor import RunMonitor

monitor = RunMonitor(model, optimizer, "runs/my-run", head="fc")
for epoch in range(epochs):
    with monitor.epoch():
        ...  # your training loop, unchanged
    monitor.log(id_validation_accuracy=accuracy, id_validation_loss=loss)
monitor.close()
```

With the Hugging Face `Trainer`, pass `callbacks=[RunSleuthCallback("runs/my-run",
head="classifier")]` from `runsleuth.hf_callback`. An optional LLM explanation
(`GEMINI_API_KEY` and `GEMINI_MODEL`, or `--provider ollama`) must cite the evidence
and cannot change the diagnosis.

## Diagnosis Results

| Fault | Decisive evidence |
|---|---|
| Stale optimizer binding | The new head gets gradients but never updates |
| Frozen classifier head\* | The head is untrainable; the reference trains it |
| Learning rate far too high | Rate inferred from AdamW's first update |
| Missing `optimizer.step()` | Gradients exist, no parameter changes |
| Train/eval normalization mismatch | Training and evaluation inputs differ |
| Head larger than the label set | 1000 outputs for 2 classes |
| Partly frozen backbone\* | Some backbone tensors never get gradients |
| Training in eval mode\* | Steps train on eval-mode forwards |
| Per-epoch schedule stepped every batch\* | The rate falls and rises within an epoch |

\*Confirmed only with a healthy reference run; without one RunSleuth waits instead of
guessing. Two more signatures, training on a held-out group and an epoch at zero
learning rate, were added later and checked on one development seed only.

| Camelyon17-WILDS, ResNet18 and DeiT-small | Development | Held-out |
|---|---:|---:|
| Correct diagnosis, given a healthy reference | 56/56 | 42/42 |
| Without a reference: correct · deferred · wrong | 46 · 10 · 0 | 34 · 8 · 0 |
| False positives on healthy runs | 0/32 | 0/24 |
| Repairs accepted after retraining | 18/18 | 19/20 |

Held-out seeds were drawn after signatures, prompt and thresholds were frozen. 11 of
38 faulty runs stayed within the metric thresholds of their healthy runs. Faults were
injected by the author, so this is not a statistical benchmark; numbers come from the
[records](evaluations/results).

## Agent Study

| | |
|---|---|
| Question | When an agent says a training script is fixed, is it? Does RunSleuth's evidence help the agent, or catch it? |
| Tasks | 22 Camelyon17 scripts: 9 faults RunSleuth knows, 7 it does not, 5 correct scripts that look faulty, 1 healthy |
| Conditions | Code only, training runs, raw evidence, diagnosis, a verdict on the agent's own runs |
| Models | Gemini 3.5 Flash-Lite and GPT-6 Luna, 550 episodes each |
| Labels | Hidden contract tests written from the task statement |

Frozen on 2026-10-13 before the first episode; every episode has run, and the results
go into the report.

## Repair Verification

| | |
|---|---|
| Projects | Small pipelines (`config.py`, `data.py`, `train.py`) that own their data, checkpoints and delivered weights. The harness may stop and restart a run, and records which samples trained with which labels |
| Tasks | Camelyon17: training-data filter, restart after a stop, delivered weights. HAM10000: restart after a stop, image cache, label mapping. Each has a development case and a held-out test case |
| Candidates | 180 test repairs by Gemini 3.5 Flash-Lite, GPT-6.1 Sol, DeepSeek V4.1 Flash, Muse Spark 1.3 and Qwen3.8-27B, plus hand-written versions, partial repairs and runs with evidence removed, reported apart |
| Verifiers | A calibrated metric gate; LLM reviewers Gemini 4 Argon and Claude Sonnet 5.5, from other vendors than the agents; RunSleuth v1; RunSleuth v2, which judges every requirement as supported, violated or untested and runs short probes chosen by what is still untested, against the same probes in a fixed order |
| Labels | Hidden behavioural tests; blind review by two annotators audits them |
| Reported | Wrong repairs accepted, correct ones rejected, share sent to review and cost, with denominators and 95% intervals from a bootstrap over cases |

The verifiers, hidden tests and analysis were frozen on 2026-10-23, before the first
test candidate ([`freeze.json`](evaluations/freeze.json)); the test seeds were drawn
after. Agent-edited code runs in a subprocess under a Python-level guard, not in a
container.

Preliminary test results on the 180 agent repairs, 16 of them wrong (14 by
Flash-Lite; Sol, Muse and Qwen repaired every test case):

| Verifier | Wrong accepted | Correct rejected | Sent to review |
|---|---:|---:|---:|
| Metric gate | 12/16 | 7/164 | 0/180 |
| RunSleuth v1 | 3/16 | 7/164 | 46/180 |
| Gemini 4 Argon | 13/16 | 0/164 | 0/180 |
| Claude Sonnet 5.5 | 2/16 | 0/164 | 2/180 |
| RunSleuth v2 | 0/16 | 0/164 | 44/180 |

The 16 wrong repairs come from 7 starting versions, so 95% intervals are wide (0/16
reaches 0.41). v2 sends a correct repair to review when its batch order differs from
the reference. The fixed-order control and the blind review of labels are still
running.

```bash
python -m runsleuth.challenge_set plan --split test
python -m runsleuth.candidates generate --split test --model-name gemini-3.5-flash-lite --provider gemini
python -m runsleuth.verifier_v2 verify --split test --order gap
python -m runsleuth.verifier_evaluation score --split test --reviewers gemini-4-argon claude-sonnet-5.5
```

## Roadmap

| Stage                                                         | Status |
| ------------------------------------------------------------- | :----: |
| Training monitor, nine faults, ResNet18 and DeiT              |   ☑    |
| Repair verification, held-out seeds, LLM explanation          |   ☑    |
| Command line, Hugging Face callback, CI, Docker               |   ☑    |
| Agent study: 22 tasks, 5 conditions, 2 LLMs                   |   ☑    |
| Pipeline harness, hidden tests, six run-time tasks            |   ☑    |
| Baselines and RunSleuth v2 on development cases, then freeze  |   ☑    |
| Test repairs from five LLM agents                             |   ☑    |
| Verifier comparison and blind review                          |   ◐    |
| Report and release                                            |   ☐    |

☑ done · ◐ in progress · ☐ planned

## Reproduce

Set `data_dir` in [`configs/camelyon17_clean.json`](configs/camelyon17_clean.json)
(about 20 GB). Names in angle brackets are run directories that earlier commands print.

```bash
python -m runsleuth.camelyon --download --prepare-only
python -m runsleuth.camelyon
python -m runsleuth.camelyon_optimizer_sweep --reference-run <baseline-run> --execute
python -m runsleuth.camelyon_frozen_head_training --reference-run <seed-baseline-run>
python -m runsleuth.camelyon_config_faults --reference-run <seed-baseline-run>
python -c "from transformers import AutoModelForImageClassification as M; M.from_pretrained('facebook/deit-small-patch16-224')"
python -m runsleuth.camelyon_vit --reference-run <seed-baseline-run>
python -m runsleuth.camelyon_loop_faults --reference-run <seed-baseline-run>
python -m runsleuth.heldout draw --count 2
python -m runsleuth.heldout run --reference-run <baseline-run>
```

`python -m pytest` runs every test on CPU with tiny models and synthetic data;
[GitHub Actions](.github/workflows/ci.yml) runs it on every push and inside the
[Docker image](Dockerfile).

## License

[MIT](LICENSE). Datasets and pretrained weights keep their own licenses.
