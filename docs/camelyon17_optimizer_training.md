# Camelyon17: full training with a stale optimizer binding

This experiment follows the single-step probe. It asks whether the reproduced
binding defect affects validation performance over a fixed training budget.
It does not assume a performance collapse or perform repair acceptance.

## Protocol

- Load the completed reference run's **initial** checkpoint, not its trained checkpoint.
- Verify checkpoint and manifest hashes recorded by that baseline.
- Train `clean` and `stale_head` for three epochs each using fresh models and loaders.
- Require identical pinned data metadata, ordered selected sample IDs, loader seeds,
  initialization, hyperparameters, preprocessing and validation schedule.
- In `clean`, replace the head before constructing AdamW. In `stale_head`, construct
  AdamW first, then replace the head with an identical-weight copy. The new head's
  parameter objects are consequently absent from the optimizer.
- Clear **all model gradients** before every training forward in both variants.
  This prevents the missing head's gradients from accumulating because optimizer
  zeroing does not cover those parameters. This is a controlled experiment detail;
  it does not fix optimizer membership.
- Measure head and backbone gradients immediately before each optimizer step,
  and parameter changes immediately afterward. Average scalar L2 norms over steps.
- Record ID and OOD validation every epoch; retain the fixed final checkpoint.
  OOD results do not select checkpoints or hyperparameters. Do not evaluate test data.

The monitor runs only during training and removes its hooks even after exceptions.
It observes current model parameters, not the optimizer's stale head objects.
Reported peak CUDA memory is allocated tensor memory, not total GPU or reserved memory.
Epoch time includes instrumentation and validation; it is not raw training throughput.

## Run

Copy the six new files into the existing repository, preserving their directories.
The previous baseline and optimizer-probe modules are prerequisites and remain unchanged.
From the repository root in the active `runsleuth` environment:

```cmd
python scripts\check_optimizer_training.py --reference-run artifacts/camelyon_runs/camelyon17-clean-20260927T002604680036Z
```

The script formats the new Python files, runs tests and lint/whitespace checks, then
starts the paired training. It stops at the first failed command. No LLM calls,
new model download or dataset download are requested. Existing local cached data is required.

## Outputs

`artifacts/optimizer_experiments/pair-<timestamp>/optimizer_training_report.json`
contains initialization provenance, mechanism checks, both variant reports and
signed clean-minus-stale ID/OOD accuracy differences in **percentage points**.

Each `clean/` and `stale_head/` directory contains config and data manifest snapshots,
epoch-zero validation, global telemetry, ID/OOD metrics, grouped telemetry, the
fixed final checkpoint and a run report. The pair directory holds source snapshots
and environment information. Partial reports survive ordinary exceptions and Ctrl+C;
an abrupt process kill or power loss cannot be guaranteed to preserve the final report.

`status=completed` means the paired run completed and reproduced the optimizer
mechanism. It does **not** mean a repair was accepted or accuracy degraded.
If performance is similar, retain that result: a trainable backbone may compensate
for a fixed head. This is one seed and a development subset from a community mirror,
not a full WILDS benchmark result or a generalization guarantee.

## Observed result: seed 42

Both variants used the same saved initialization and selected samples,
with three training epochs each. All nine mechanism checks passed.

| Final metric | Clean | Stale head |
|---|---:|---:|
| ID validation accuracy | 0.9786 | 0.9806 |
| OOD validation accuracy | 0.9016 | 0.9156 |
| ID validation loss | 0.07480 | 0.07118 |
| OOD validation loss | 0.37869 | 0.32010 |

The stale head received nonzero gradients but had exactly zero parameter
updates in every epoch, while the backbone continued updating.

This confirms a silent optimizer-binding defect without observed performance
degradation in this run. The slightly higher stale-head accuracy does not
establish a general benefit or statistical significance. No repair or
performance-recovery claim is made.

Local report:
`artifacts/optimizer_experiments/pair-20260927T012822181931Z/optimizer_training_report.json`
