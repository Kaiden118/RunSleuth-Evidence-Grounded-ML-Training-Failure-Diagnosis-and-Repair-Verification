# RunSleuth: Evidence-Grounded ML Training Failure Diagnosis and Repair Verification

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![LLM](https://img.shields.io/badge/LLM-Gemini-8E75B2)

RunSleuth is a tool-using LLM agent that diagnoses supported PyTorch training
failures from source-code structure, recorded configuration, and training
telemetry. It proposes constrained configuration repairs and accepts them only
after bounded retraining satisfies explicit validation criteria.

The project includes a rule-based diagnostic baseline, a bounded LLM agent,
repair verification, and evaluation across multiple training seeds.

## What It Does

1. Train a small FashionMNIST CNN and record configuration, loss, accuracy,
   gradient norms, and parameter-update norms.
2. Compare a candidate run with a healthy reference using typed, read-only tools:
   `inspect_training_source`, `load_run_config`, and `compare_runs`.
3. Produce a structured diagnosis with ranked causes, heuristic confidence,
   uncertainty, and citations to specific tool-result paths and values.
4. Propose a minimal change to an allowlisted configuration field.
5. Validate the report and replay its evidence against current artifacts before
   executing the repair.
6. Retrain from scratch within an epoch budget and accept the repair only when
   every recovery check passes.

Supported injected faults:

- `high_learning_rate`: increase the learning rate from `0.001` to `0.1`.
- `missing_optimizer_step`: set `optimizer_step_enabled` to `false`, leaving
  gradients present while parameter updates remain zero.

The agent chooses the order of evidence collection. Tool schemas restrict path
arguments to the current task, and execution checks enforce the same scope.
Repeated or rejected requests still consume the tool-call budget. Identical
reference and candidate paths are valid inputs and still require evidence
collection; they do not automatically produce a healthy verdict.

## Development Evaluation

On **September 26, 2026**, using **Gemini 3.5 Flash-Lite**, RunSleuth passed all
**9 cases on the first Agent attempt**: two injected fault types and one clean
self-comparison at each of three seeds (**7, 123, and 2026**).

| Metric | Result |
|---|---:|
| Cases passed on the first attempt | **9/9** |
| Fault top-1 diagnosis correct | **6/6** |
| Fault repairs accepted after verification | **6/6** |
| Clean self-comparisons with no action | **3/3** |
| Reports with valid citation values | **6/6** |

This evaluation used **21 model calls**, **33 tool calls**, **40,195 input tokens**,
and **5,214 output tokens**. Usage accounting was complete for this evaluation;
these totals exclude earlier experiments and debugging sessions.

Each case received one Agent invocation, with budgets of six model calls and
five tool calls. An Agent invocation may contain several model requests.
Repairs were limited to two training epochs. Execution failures remain in the
end-to-end evaluation denominators, and optional API-recovery results are
reported separately from first-attempt results.

[Recorded evaluation summary](evaluations/results/20260926T205834269355Z.json)

These are **development and regression cases**, not a held-out benchmark: they
also informed implementation changes. The healthy controls compare a run with
itself. Citation validation checks referenced values, not the correctness of
every narrative claim or causal inference.

Within each seed, reference, faulty, and repaired experiments use the same data
split and seed. Across seeds, the split, model initialization, and training
shuffle can all change.

## Repair Verification

The current verification policy requires all three conditions:

| Check | Acceptance threshold |
|---|---:|
| Validation-accuracy gain over the failed run | At least 20 percentage points |
| Validation-accuracy shortfall from the reference | At most 3 percentage points |
| Validation loss divided by reference loss | At most 1.25 |

The current agent repair allowlist permits a positive learning-rate decrease or
an `optimizer_step_enabled` change from `false` to `true`. The recorded old value
must match the failed configuration. The original faulty configuration remains
unchanged; the repaired run receives its own artifacts.

Verification uses **from-scratch retraining**, capped at two epochs in the
reported evaluation, against a three-epoch reference. It does not resume the
failed checkpoint. A completed repair execution is not sufficient for
acceptance: the verification decision must be `accepted`.

### Single-Seed Worked Example: High Learning Rate

This earlier example uses **seed 42** and is separate from the three-seed
evaluation above.

| Run | Epochs | Validation accuracy | Validation loss | First update norm | Final gradient norm |
|---|---:|---:|---:|---:|---:|
| Healthy reference | 3 | 0.8896 | 0.3119 | 0.0538 | 1.6822 |
| High-learning-rate failure | 3 | 0.1054 | 2.3040 | 0.5263 | 0.0884 |
| Bounded repair | 2 | 0.8672 | 0.3683 | 0.0538 | 1.9616 |

Restoring the learning rate from `0.1` to `0.001` yielded an accuracy gain of
76.18 percentage points, a reference shortfall of 2.24 percentage points, and a
loss ratio of 1.1808. All three verification checks passed.

## Run Locally

Use Python 3.11. From the repository root, install the package and development
dependencies into your active environment:

```bat
python -m pip install -e ".[dev]"
```

The recorded experiments ran on native Windows with an RTX 5060 Ti 8 GB,
Python 3.11.16, PyTorch `2.13.0+cu130`, and torchvision `0.28.0+cu130`.
The checked-in experiment configurations use CUDA; CPU runs require a CPU
device setting. CUDA-enabled PyTorch installation depends on the platform.

The commands below use **Windows CMD** syntax. Replace example artifact paths
with the paths printed by your own runs. Generated artifacts are not included
when cloning the repository.

### Generate Training Evidence

```bat
python -m runsleuth --config configs/clean.json
python -m runsleuth --config configs/high_learning_rate.json
python -m runsleuth --config configs/missing_optimizer_step.json
```

For the following examples, select the resulting healthy and failed run paths:

```bat
set "REFERENCE_RUN=artifacts/runs/clean-REPLACE_WITH_TIMESTAMP"
set "FAILED_RUN=artifacts/runs/high-lr-REPLACE_WITH_TIMESTAMP"
```

Compare the runs or use the rule-based diagnostic baseline:

```bat
python -m runsleuth.compare --clean-run "%REFERENCE_RUN%" --candidate-run "%FAILED_RUN%"
python -m runsleuth.diagnose --reference-run "%REFERENCE_RUN%" --candidate-run "%FAILED_RUN%" --reference-config configs/clean.json --candidate-config configs/high_learning_rate.json
python -m runsleuth.repair_and_verify --reference-run "%REFERENCE_RUN%" --failed-run "%FAILED_RUN%" --reference-config configs/clean.json --failed-config configs/high_learning_rate.json --max-epochs 2
```

For the missing-step fault, use its run directory and
`configs/missing_optimizer_step.json` as the failed configuration.

### Run the LLM Agent

The Gemini integration uses the OpenAI-compatible API through the OpenAI Python
SDK. A Gemini API key and available model quota are required.

```bat
set "GEMINI_API_KEY=YOUR_GEMINI_API_KEY"
set "GEMINI_MODEL=gemini-3.5-flash-lite"
python -m runsleuth.agent_diagnose --reference-run "%REFERENCE_RUN%" --candidate-run "%FAILED_RUN%"
```

Keep API keys out of source control. These `set` commands apply to the current
terminal session.

After a diagnosis completes with a proposed patch, pass its saved report to
the separate repair gate:

```bat
set "AGENT_REPORT=artifacts/agent_runs/REPLACE_WITH_TIMESTAMP/agent_report.json"
python -m runsleuth.agent_repair_and_verify --agent-report "%AGENT_REPORT%" --reference-run "%REFERENCE_RUN%" --failed-run "%FAILED_RUN%" --max-epochs 2
```

This step reads the saved report and runs local training verification; it does
not make another LLM request.

### Evaluate Across Seeds

Preview the budget, then generate and evaluate a fresh batch:

```bat
python -m runsleuth.batch_evaluate --seeds 7 123 2026 --device cuda --dry-run
python -m runsleuth.batch_evaluate --seeds 7 123 2026 --device cuda
```

To evaluate another model or an updated agent using existing training records,
set `BATCH_SUMMARY` to a saved original batch summary. Every case is diagnosed
again. Eligible proposed repairs pass through the repair gate and bounded
verification:

```bat
set "BATCH_SUMMARY=artifacts/batches/REPLACE_WITH_TIMESTAMP/batch_summary.json"
set "GEMINI_MODEL=gemini-3.5-flash-lite"
python -m runsleuth.reevaluate_batch --batch-summary "%BATCH_SUMMARY%" --model gemini-3.5-flash-lite --dry-run
python -m runsleuth.reevaluate_batch --batch-summary "%BATCH_SUMMARY%" --model gemini-3.5-flash-lite --max-attempts 1 --case-delay 30
```

Reevaluation creates no new baseline or fault-injection training runs. It saves
new Agent reports and separate first-attempt and final evaluation summaries.
Repair runs inherit the output directory in the original recorded configuration.

For transient API failures in an **original batch**, the recovery command
preserves original results and records every additional attempt. First restore
the model recorded in that original batch summary:

```bat
set "GEMINI_MODEL=REPLACE_WITH_ORIGINAL_BATCH_MODEL"
python -m runsleuth.retry_batch --batch-summary "%BATCH_SUMMARY%" --max-attempts 2 --retry-delay 10
```

Recovery requires the same model as that original batch. A different model
uses `reevaluate_batch`. Completed wrong diagnoses and rejected repairs are not
retried to improve their outcome. Daily quota exhaustion requires available
quota rather than repeated requests.

Evaluate saved reports without new LLM calls or training:

```bat
python -m runsleuth.evaluate --manifest evaluations/smoke_cases.json
```

That manifest references local example artifacts. For a new batch or model
evaluation, use the `manifest.json` saved in its output directory instead.

## Artifacts and Checks

Training runs save `config.json`, `metrics.jsonl`, and `model_state_dict.pt`.
Rule-based repairs add `repair_report.json`; Agent repairs add
`agent_repair_report.json`. Agent diagnosis reports include tool traces,
structured output, execution status, and recorded token usage.

Batch and model-evaluation directories contain settings, manifests, progress,
and evaluation reports. Compact published summaries live under
`evaluations/results/`; datasets, checkpoints, and full runtime artifacts remain
excluded from Git. The summary retains local source-report paths for provenance,
but is not a substitute for the full replayable artifacts.

```bat
python -m pytest
ruff check .
git diff --check
```

## Current Scope and Limitations

- One small FashionMNIST CNN and two controlled configuration faults.
- Diagnosis and verification have access to a healthy reference run.
- The current evaluation is a small development set with self-comparison
  controls, not an estimate of performance on unseen training systems.
- Source inspection describes current code structure, not historical runtime
  execution. Citation-value checks do not establish complete causal faithfulness.
- Confidence is heuristic. Repairs modify allowlisted configuration fields;
  arbitrary Python code repair is outside the current scope.
- Broader faults, independent healthy controls, held-out evaluation, and a
  deployed service remain future work.
