# RunSleuth: Evidence Grounded ML Training Failure Diagnosis and Repair Verification

RunSleuth is an evidence-grounded system for diagnosing machine-learning
training failures and verifying proposed repairs through bounded retraining.

Instead of accepting a plausible explanation or code change, RunSleuth requires
a repair to restore measurable validation performance before it is accepted.

## Current MVP

The current MVP implements a complete deterministic loop:

1. Run a reproducible FashionMNIST CNN experiment.
2. Record configuration, loss, accuracy, gradient norm, and parameter-update norm.
3. Compare a failed run against a healthy reference.
4. Rank supported root causes using explicit evidence.
5. Propose a minimal configuration diff.
6. Retrain under a fixed epoch budget.
7. Accept the repair only when every recovery requirement passes.

The MVP intentionally does not use an LLM yet. It establishes an objective,
testable diagnosis-and-verification foundation before adding an agent layer.

## Reproduced Failure

The first supported failure is an excessively high learning rate:

```diff
-  "learning_rate": 0.001
+  "learning_rate": 0.1
```

Both runs use the same model, dataset split, seed, batch size, and device.

| Run | Epochs | Validation accuracy | Validation loss | First update norm | Final gradient norm |
|---|---:|---:|---:|---:|---:|
| Healthy reference | 3 | 0.8896 | 0.3119 | 0.0538 | 1.6822 |
| High-learning-rate failure | 3 | 0.1054 | 2.3040 | 0.5263 | 0.0884 |
| Bounded repair | 2 | 0.8672 | 0.3683 | 0.0538 | 1.9616 |

The failed run produced an initial parameter update approximately 9.78 times
larger than the reference. Its final validation accuracy fell by 78.42
percentage points, while its final gradient norm collapsed to approximately
5.25% of the reference value.

## Repair Verification

RunSleuth proposed restoring the learning rate from `0.1` to `0.001` and
performed only two verification epochs.

A repair is accepted only when all configured requirements pass:

| Requirement | Observed | Threshold | Result |
|---|---:|---:|---|
| Validation-accuracy gain over failed run | 0.7618 | >= 0.20 | Pass |
| Accuracy shortfall from reference | 0.0224 | <= 0.03 | Pass |
| Validation-loss ratio to reference | 1.1808 | <= 1.25 | Pass |

Final decision: `accepted`.

The original faulty configuration remains unchanged. The repaired run receives
its own configuration, telemetry, model checkpoint, and structured
`repair_report.json`.

## Artifacts

Each training run writes:

```text
artifacts/runs/<run-name>-<timestamp>/
├── config.json
├── metrics.jsonl
└── model_state_dict.pt
```

A repair attempt additionally writes:

```text
repair_report.json
```

The repair report contains:

- the ranked diagnosis and confidence;
- the telemetry and configuration evidence;
- the proposed minimal diff;
- the bounded repaired configuration;
- every verification check;
- the final accepted or rejected decision.

Generated datasets, model checkpoints, and run artifacts are excluded from Git.

## Usage

Install the project in a Python 3.11 environment:

```bash
python -m pip install -e ".[dev]"
```

Run the healthy baseline:

```bash
python -m runsleuth --config configs/clean.json
```

Run the reproducible high-learning-rate failure:

```bash
python -m runsleuth --config configs/high_learning_rate.json
```

Compare two runs:

```bash
python -m runsleuth.compare \
  --clean-run <healthy-run-directory> \
  --candidate-run <failed-run-directory>
```

Generate a structured diagnosis:

```bash
python -m runsleuth.diagnose \
  --reference-run <healthy-run-directory> \
  --candidate-run <failed-run-directory> \
  --reference-config configs/clean.json \
  --candidate-config configs/high_learning_rate.json
```

Execute bounded repair and verification:

```bash
python -m runsleuth.repair_and_verify \
  --reference-run <healthy-run-directory> \
  --failed-run <failed-run-directory> \
  --reference-config configs/clean.json \
  --failed-config configs/high_learning_rate.json \
  --max-epochs 2
```

## Quality Checks

```bash
pytest
ruff check .
git diff --check
```

## Current Scope

The MVP currently supports one objectively reproducible failure class:
`high_learning_rate`.

Planned extensions include additional failure classes, source-code inspection,
an evaluation benchmark, and an LLM agent restricted to typed diagnostic and
repair tools.