# RunSleuth: Evidence-grounded ML Training Failure Diagnosis and Repair Verification

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?logo=huggingface&logoColor=black)
![LLM](https://img.shields.io/badge/LLM-Gemini-8E75B2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

RunSleuth debugs PyTorch training runs from evidence rather than intuition. It
records structural telemetry while a model trains, matches it against a library
of failure signatures, lets an LLM review the evidence without letting it
overrule it, and accepts a repair only after bounded retraining verifies it.

Many training faults are silent: the run finishes and accuracy looks normal. On
Camelyon17, a frozen classifier head and a stale optimizer binding produce
**bitwise-identical backbones and identical accuracy**, yet need different
repairs, and a pretrained 1000-class head kept for a 2-class task trains as well
as the correct model. Validation metrics cannot separate these cases; gradients,
updates, trainability, optimizer membership and input statistics can.

## How It Works

```
training loop ─► RunMonitor / RunSleuthCallback ─► run report (JSON)
                                                        │
          failure-signature library ─► signature matcher ─► verdict per fault
                                                        │
                       LLM review (validated citations; cannot overrule)
                                                        │
                 minimal repair ─► bounded retraining ─► accept or reject
```

1. **Instrument.** `RunMonitor` wraps any PyTorch training loop and
   `RunSleuthCallback` plugs into the Hugging Face `Trainer`. Per layer group
   (head and backbone) they record gradient and update norms, trainable tensors,
   tensors that received gradients, update-to-weight ratios and the first update;
   per run, optimizer-membership audits, training forwards versus optimizer
   steps, per-split input statistics, label classes and head size.
2. **Diagnose.** Seven declarative signatures
   ([`failure_signatures.json`](src/runsleuth/failure_signatures.json)) list
   required and contradicting evidence. Every run gets a verdict per signature:
   supported, contradicted, not supported, insufficient evidence, or pending a
   healthy reference when only a reference can establish intent. Thresholds are
   documented heuristics, not tuned on these runs.
3. **Review.** An LLM (Gemini through an OpenAI-compatible API) explains the
   evidence and must cite it by name and value; citations are validated. When it
   disagrees with the matcher, the conflict is flagged for human review and the
   matcher's verdict stands.
4. **Verify.** The minimal repair is applied and the model retrained under a
   fixed budget. A repair is accepted only if structural checks pass and a
   development performance gate shows no regression against both the healthy
   reference and the faulty run.

## Supported Faults

| Fault | Subsystem | Decisive evidence | Minimal repair | Model |
|---|---|---|---|---|
| Stale optimizer binding | optimizer | Head gets gradients but never updates; current head tensors are missing from the optimizer | Rebuild the optimizer after replacing the head | ResNet18 |
| Frozen classifier head | model | Head untrainable, without gradients or updates; the reference trains it | Restore `requires_grad` | ResNet18 |
| Learning rate 100x too high | configuration | AdamW's effective learning rate, inferred from its first update, far above the fine-tuning range | Restore the reference learning rate | ResNet18 |
| Missing `optimizer.step()` | training loop | Training forwards without optimizer steps; gradients but no parameter change | Call `optimizer.step()` | ResNet18 |
| Train/eval normalization mismatch | data | Input statistics of training and a same-distribution evaluation split disagree | One preprocessing pipeline for both | DeiT-small |
| ImageNet head kept for 2 labels | model | 1000 head outputs for 2 label classes | Load with `num_labels=2` | DeiT-small |
| Frozen patch embedding | model | Part of the backbone untrainable; the reference trains it | Restore `requires_grad` | DeiT-small |

## Results

Camelyon17-WILDS histopathology: 20,000 training patches, 5,000
in-distribution validation patches and 5,000 out-of-distribution validation
patches from unseen hospitals. Pretrained ResNet18 (torchvision) and DeiT-small
(Hugging Face `Trainer`), 3 epochs per run, seeds 7 and 2026. Faults are
injected by the author; thresholds are development settings, not statistical
tests.

**Diagnosis** across 46 runs and 7 fault types
([record](evaluations/results/signature-matching-20261008.json)):

| Mode | Correct | Healthy false positives |
|---|---:|---:|
| With a healthy reference | 46/46 | 0/26 |
| Without a reference | 40/46; the 6 frozen-module cases are deferred to a reference | 0/26 |

**Repair verification** over seeds 7 and 2026:

| Fault | Reproduced | Repair accepted |
|---|---:|---:|
| Stale optimizer binding | 2/2 | 2/2 |
| Frozen classifier head | 2/2 | 2/2 |
| Learning rate 100x too high | 2/2 | 2/2 |
| Missing `optimizer.step()` | 2/2 | 2/2 |
| Train/eval normalization mismatch | 2/2 | 2/2 |
| ImageNet head kept for 2 labels | 2/2 | 0/2 |
| Frozen patch embedding | 2/2 | 0/2 |

DeiT rows come from the deterministic rerun described under Findings. The last
two repairs restored the clean model bitwise, but the faulty runs had
generalized as well or better, so the gate rejected them. In this setting those
faults are structural rather than harmful; the decisions are kept as recorded.

**Findings**

- **Indistinguishable by metrics.** A frozen head and a stale optimizer binding
  left bitwise-identical backbones and identical validation metrics after three
  epochs on CUDA; only trainability, gradient presence and the optimizer audit
  separate them.
- **Learning rate from telemetry.** AdamW's first update moves each element by
  about the learning rate, so RunSleuth recovers the effective rate without the
  config (1e-4 and 1e-2, within 0.1%).
- **Conflicting normalization in a public checkpoint.** The DeiT checkpoint's
  Hugging Face image processor (mean = std = 0.5) contradicts its model card
  (ImageNet statistics). Fine-tuning with 0.5 normalization reached higher OOD
  accuracy in both seeds (+2.9 and +2.6 points); two seeds suggest this, they do
  not establish it.
- **Determinism matters for verification.** DeiT training in bf16 was not
  reproducible: identically configured runs differed by more than the gate's
  tolerance, and the first DeiT round rejected all six repairs. With
  deterministic algorithms and eager attention, identical configurations now
  match bitwise.
- **Non-intrusive instrumentation.** The end-to-end demo (ResNet18, seed 7,
  faults named, no LLM) diagnosed 5/5 cases and accepted 4/4 repairs. Its plain
  training loop with `RunMonitor` produced exactly the same metrics as the
  experiment runner.
- **The LLM is checked, not trusted.** In three real Gemini 3.5 Flash-Lite
  reviews, the model twice disagreed with a correct matcher verdict while citing
  only valid values: once because of a payload bug (fixed), once by assuming an
  intent that only a reference can establish. Both were flagged and the
  matcher's verdict stood.

**Earlier development suite (FashionMNIST CNN).** With Gemini 3.5 Flash-Lite,
a tool-calling agent diagnosed a high learning rate and a missing optimizer
step and passed 9/9 cases on its first attempt across seeds 7, 123 and 2026:
6/6 correct diagnoses, 6/6 repairs accepted after at most 2 retraining epochs
and 3/3 healthy self-comparisons left unchanged
([record](evaluations/results/20260926T205834269355Z.json)).

**Limitations.** Faults are author-injected and results cover two seeds;
thresholds are development settings. The LLM evaluation is a handful of reviews,
and the blind demo with an LLM review has not been run. The monitor clears
gradients before each training forward, so gradient accumulation is
unsupported. Repairs are applied automatically only inside the demo; for other
code RunSleuth suggests them. No test-set evaluation has been performed.

## Quick Start

Use Python 3.11 with a CUDA build of PyTorch. Commands use **Windows CMD**.

```bat
python -m pip install -e ".[dev,llm,hf,camelyon]"
```

Instrument your own training loop:

```python
from runsleuth.run_monitor import RunMonitor

monitor = RunMonitor(model, optimizer, "runs/my-run", head="fc")
for epoch in range(epochs):
    with monitor.epoch():
        for inputs, targets in loader:  # your loop, unchanged
            optimizer.zero_grad()
            loss_fn(model(inputs), targets).backward()
            optimizer.step()
    monitor.log(train_loss=train_loss, id_validation_accuracy=accuracy)
monitor.close()
```

Or the Hugging Face `Trainer`:

```python
from runsleuth.hf_callback import RunSleuthCallback

trainer = Trainer(model=model, args=args, train_dataset=train,
                  callbacks=[RunSleuthCallback("runs/vit", head="classifier")])
```

Diagnose a run, optionally against a healthy reference. The LLM review needs
`GEMINI_API_KEY` and `GEMINI_MODEL`; `--no-llm` runs offline:

```bat
python -m runsleuth.diagnose_run --run runs/my-run/run_report.json --no-llm
python -m runsleuth.diagnose_run --run runs/my-run/run_report.json --reference runs/healthy/run_report.json
```

Generated reports, metrics and checkpoints are written under `artifacts/` and
excluded from Git. Keep API keys out of source control.

## Reproducing the Camelyon17 Experiments

Set `data_dir` in [`configs/camelyon17_clean.json`](configs/camelyon17_clean.json),
then prepare the data, train a baseline and create the seed baselines:

```bat
python -m runsleuth.camelyon --download --prepare-only
python -m runsleuth.camelyon
python -m runsleuth.camelyon_optimizer_sweep --reference-run artifacts/camelyon_runs/REPLACE_WITH_BASELINE --execute
```

With a seed baseline from the sweep:

```bat
set "BASELINE=artifacts/optimizer_sweeps/REPLACE_WITH_SWEEP/seed-7/baseline/REPLACE_WITH_SEED_BASELINE"
python -m runsleuth.camelyon_frozen_head_training --reference-run "%BASELINE%"
python -m runsleuth.camelyon_config_faults --reference-run "%BASELINE%"
python -m runsleuth.camelyon_vit --reference-run "%BASELINE%"
```

Score the matcher on the experiment reports, and run the end-to-end demo with a
hidden, randomly chosen fault:

```bat
python -m runsleuth.signature_matching evaluate REPLACE_WITH_EXPERIMENT_REPORTS
python -m runsleuth.demo --baseline "%BASELINE%" --reference REPLACE_WITH_CLEAN_RUN_REPORT --fault random --blind
```

Design notes and earlier results are in [`docs/`](docs).

## Checks

```bat
python -m pytest
python -m ruff check .
git diff --check
```

The suite has 676 passing tests (3 skipped), including CPU integration tests
that run every runner end to end on a tiny model and synthetic data.

## Roadmap

- A verification policy for silent faults: compare the repair with the healthy
  reference and treat the faulty run as informational, fixed in advance and
  tested on new seeds.
- A multi-model LLM evaluation on held-out cases, with Gemini and local open models.
- An MCP server for the diagnostic tools and retrieval over past diagnosis reports.

## License

RunSleuth's original source code is licensed under the [MIT License](LICENSE).

Third-party dependencies, datasets, and pretrained weights remain subject
to their respective licenses and terms.
