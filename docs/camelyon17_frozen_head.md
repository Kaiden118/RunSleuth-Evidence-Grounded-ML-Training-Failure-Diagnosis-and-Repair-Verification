# Camelyon17 frozen-head fault: design and predictions

Status: design fixed on 2026-10-06, before any frozen-head probe or training run.
Run commands and observed results will be appended below; the definitions and
predictions in this document will not be edited after results are seen.

## Why this fault matters

In full-model fine-tuning, a classification head whose parameters have
`requires_grad=False` receives no gradients, and AdamW skips it. The backbone keeps
training through the fixed head, so loss, global update norms and accuracy can
look normal.

The visible symptom matches the stale optimizer binding: the head update norm is
exactly zero while the backbone updates. The mechanism, subsystem and repair are
different. This is the first differential-diagnosis pair in RunSleuth: head-update
evidence alone cannot separate the two faults.

Realistic origins include an inverted freeze condition, a head deep-copied from a
frozen EMA or teacher module (`deepcopy` preserves `requires_grad=False`), and a
staged unfreezing step that never runs.

The fault lives in runtime state. `requires_grad` is not stored in `state_dict`, so
checkpoint and state hashes cannot detect it, and loading the checkpoint into a
freshly built model silently removes it.

## Fault definition (ground truth)

The clean case is unchanged from the optimizer-binding probe:

```python
new_head = deepcopy(model.fc)
model.fc = new_head
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
```

The faulty case adds one line:

```python
new_head = deepcopy(model.fc)
model.fc = new_head
model.fc.requires_grad_(False)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
```

- Subsystem: model parameter trainability. The stale binding is an optimizer fault.
- The only difference from clean is `requires_grad` on `fc.weight` and `fc.bias`.
  Values, optimizer membership and parameter order are identical.
- Intent: full-model fine-tuning, as declared by the clean reference (62 of 62
  parameter tensors trainable). Frozen parameters can be intentional, for example
  in linear probing. Without a reference or a declared intent, a diagnosis should
  report a frozen head as an observation, not as a fault.
- Oracle independent of any analyzer: one CPU optimizer step confirms that head
  gradients are `None` and head tensors are bitwise unchanged.

## Variants

| Variant | Construction | Stage |
|---|---|---|
| `clean` | Replace head, then build AdamW | Probe, training |
| `stale_head` | Build AdamW, then replace head (existing variant) | Probe, training |
| `frozen_head` | Fault definition above | Probe, training |
| `frozen_head_repaired` | `frozen_head`, then `model.fc.requires_grad_(True)` before the first step | Probe, training |
| `frozen_head_optimizer_rebuilt` | `frozen_head`, then AdamW rebuilt from `model.parameters()` before the first step | Probe only |

`frozen_head_optimizer_rebuilt` applies the stale-binding repair to the wrong
fault. It is a negative control for repair verification.

## Evidence

| ID | Evidence | Clean | Stale head | Frozen head | Role |
|---|---|---|---|---|---|
| E1 | Head tensors with `requires_grad` | 2/2 | 2/2 | 0/2 | Identifies a frozen head (new) |
| E2 | Head tensors with a gradient at each step | 2/2 | 2/2 | 0/2 | Separates absent from zero gradients (new) |
| E3 | Head gradient L2 norm | Positive | Positive | Zero | Insufficient alone: absent gradients are reported as 0.0 |
| E4 | Head update L2 norm | Positive | Zero | Zero | Shared symptom |
| E5 | Backbone update L2 norm | Positive | Positive | Positive | Rules out global faults such as a missing step |
| E6 | Optimizer audit: missing current / foreign tensors | None / 0 | `fc.weight`, `fc.bias` / 2 | None / 0 | Identifies the stale binding |
| E7 | Initial state SHA-256 | Reference | Reference | Reference | State hashes cannot see the fault |

The monitor clears model gradients before every training forward, so E2 reflects
only the current step.

The existing audit counts only tensors with `requires_grad` as trainable. A frozen
head inside the optimizer is neither missing nor foreign, so the existing
complete-coverage check passes for `frozen_head`. This is expected and is part of
what this case tests.

Signatures:

- `frozen_head` is supported when, in every epoch, E4 is zero, E2 is 0/2 at every
  step, E1 is 0/2, E5 is positive and E6 shows complete coverage, and the reference
  shows the head as trainable.
- Evidence against `frozen_head`: the head receives gradients (E2 above 0/2),
  head tensors are missing from the optimizer (E6), or the backbone does not
  update (E5).
- Evidence against `stale_head`: no head gradients (E2 is 0/2) or complete
  optimizer coverage (E6).

## Predictions (fixed before running)

- **P1.** In `frozen_head`, head tensors remain bitwise identical to their initial
  values after every epoch.
- **P2.** `frozen_head` and `stale_head` follow the same backbone trajectory. In
  both, the head keeps its initial values, gradients still flow through the head to
  the backbone, and no gradient clipping is applied. In the CPU probe, loss,
  backbone gradients and backbone updates are bitwise identical. In CUDA training,
  metrics are expected to match, but exact equality is not required:
  `cudnn.deterministic` is set, but deterministic algorithms are not enforced
  globally. Any difference will be reported.
- **P3.** `frozen_head_repaired` matches `clean`. After the repair, model values,
  optimizer parameter order and optimizer state equal those of `clean` before the
  first step. The CPU probe should be bitwise identical; training is expected to match.
- **P4.** `frozen_head_optimizer_rebuilt` is rejected. The rebuilt optimizer has the
  same contents, so the head update stays zero and structural verification fails.

If P2 holds, accuracy and loss cannot distinguish the two faults; only E1, E2 and
E6 can.

## Repair and verification

The repair action is `restore_head_requires_grad`, applied before the first
optimizer step. It requires empty optimizer state and current head tensors that
are already optimizer members. The optimizer is not rebuilt.

Structural checks, all required:

- Every variant completes the fixed budget from the reference initial state, with
  the same initial validation metrics and optimizer step counts.
- The frozen head is reproduced before the repair (E1 recorded as 0/2).
- The repair runs before the first step with empty optimizer state and leaves the
  model state hash unchanged.
- The repaired optimizer covers all current parameters, and all 62 tensors are
  trainable, both initially and finally.
- The repaired head and backbone have gradients (E2 2/2) and positive updates in
  every epoch.

Performance uses the existing development policy: `max_accuracy_drop = 0.01` and
`max_loss_ratio = 1.10`, against `clean` and `frozen_head`, for ID and OOD
validation. Scope: development policy; no statistical or performance-recovery claim.

Known risk, stated in advance: the gate also compares against the faulty run. A
frozen head may generalize as well or better on OOD data. For seed 42, `stale_head`
reached 91.56% OOD accuracy versus 90.16% for `clean`, and a structurally correct
repair was rejected. Such outcomes will be recorded as `rejected`; the policy will
not be changed after seeing results.

## Controls

The controls match the optimizer training experiment: verified reference initial
checkpoint and manifest hashes; identical pinned data, selected sample IDs, loader
seeds, hyperparameters, preprocessing and validation schedule; model gradients
cleared before every training forward; fixed final checkpoint; OOD data never
used for selection; no test-set evaluation.

## Stages and budget

1. **Single-step probe on CPU:** all five variants, one fixed training batch and
   one optimizer step each. Tests P1 for one step, P2 and P3 bitwise, and P4.
   No GPU training epochs and no model calls.
2. **Bounded training on CUDA:** seeds 7 and 2026, reusing the seed baselines from
   `artifacts/optimizer_sweeps/20261001T033140591688Z` with no new baseline runs.
   `clean`, `stale_head`, `frozen_head` and `frozen_head_repaired` train for three
   epochs each: 24 training epochs in total and no model calls. Stage 2 runs only
   if stage 1 confirms the mechanism.

## Required instrumentation (not yet implemented)

- E1: per-group counts and names of tensors with `requires_grad`, recorded with the
  optimizer audit and in every epoch.
- E2: per-step counts of tensors with gradients for each group, summarized per
  epoch as the minimum and maximum.

Existing fields keep their meaning; new fields are additive.

## Scope

- Author-constructed fault with known ground truth. This is a development
  experiment with two seeds, not a held-out benchmark.
- Covers only whole-head freezing with an optimizer built from `model.parameters()`.
  Deferred: an optimizer built from a `requires_grad` filter, where the repair needs
  both unfreezing and rebuilding; partial freezing; and other causes of a
  non-updating head, such as a zero learning rate in the head's parameter group or
  deliberate exclusion from all parameter groups.
- The PyTorch behavior this design relies on was checked on torch 2.13.0 with a CPU
  ResNet18: `deepcopy` keeps `requires_grad=False`; `state_dict` does not store it;
  AdamW with weight decay leaves a parameter without gradients unchanged while the
  backbone trains through it. None of the predictions has been run.

## Stage 1 run (Windows CMD)

From the repository root in the `runsleuth` environment, once per seed baseline:

```bat
python -m runsleuth.camelyon_frozen_head_probe --reference-run artifacts/optimizer_sweeps/20261001T033140591688Z/seed-7/baseline/seed-7-baseline-20261001T033141996710Z
python -m runsleuth.camelyon_frozen_head_probe --reference-run artifacts/optimizer_sweeps/20261001T033140591688Z/seed-2026/baseline/seed-2026-baseline-20261001T034655720221Z
```

The probe always runs on CPU, uses cached data only and exits nonzero unless every
mechanism check passes. Predictions and candidate verdicts are reported as observed.

## Observed result: stage 1 single-step probe

Both seeds completed on CPU with torch 2.13.0, one fixed 32-image training batch
and one optimizer step per variant. All seven mechanism checks passed for each seed.

Seed 7 (seed 2026 shows the same pattern):

| Variant | Head trainable | Head tensors with gradient | Head update L2 | Backbone update L2 | Missing / foreign |
|---|---:|---:|---:|---:|---:|
| `clean` | 2/2 | 2/2 | 0.003203 | 0.334142 | 0 / 0 |
| `stale_head` | 2/2 | 2/2 | 0 | 0.334142 | 2 / 2 |
| `frozen_head` | 0/2 | 0/2 | 0 | 0.334142 | 0 / 0 |
| `frozen_head_repaired` | 2/2 | 2/2 | 0.003203 | 0.334142 | 0 / 0 |
| `frozen_head_optimizer_rebuilt` | 0/2 | 0/2 | 0 | 0.334142 | 0 / 0 |

- P1 held: the frozen head was bitwise unchanged after the step.
- P2 held: `frozen_head` and `stale_head` had identical pre-update loss, backbone
  gradient hashes and post-step backbone hashes. For one step, the two faults are
  indistinguishable from the backbone's side; only E1, E2 and E6 separate them.
- P3 held: `frozen_head_repaired` matched `clean` bitwise for gradients and
  post-step parameters of both groups.
- P4 held: restoring `requires_grad` was accepted; rebuilding the optimizer was
  rejected because not all parameters were trainable and the head had neither
  gradients nor updates.
- As expected, the existing optimizer audit reported complete coverage for
  `frozen_head`.

On AdamW's first step, each element moves by about `learning_rate * sign(gradient)`
when gradients are well above epsilon, so the update norm is close to
`learning_rate * sqrt(elements)`: 0.003203 = 1e-4 * sqrt(1026) for the head. A
single-step update norm therefore shows whether tensors moved, not how strongly.
The evidence relies on zero versus nonzero updates and on bitwise hashes.

Scope: one optimizer step per variant; mechanism evidence only. No validation,
training or performance result is claimed. Stage 2 has not been run.

Reports:
`artifacts/frozen_head_probes/probe-20261007T051446706061Z/frozen_head_probe_report.json` (seed 7),
`artifacts/frozen_head_probes/probe-20261007T051729795038Z/frozen_head_probe_report.json` (seed 2026)

## Stage 2 run (Windows CMD)

```bat
python -m runsleuth.camelyon_frozen_head_training --reference-run artifacts/optimizer_sweeps/20261001T033140591688Z/seed-7/baseline/seed-7-baseline-20261001T033141996710Z
python -m runsleuth.camelyon_frozen_head_training --reference-run artifacts/optimizer_sweeps/20261001T033140591688Z/seed-2026/baseline/seed-2026-baseline-20261001T034655720221Z
```

The command exits nonzero unless the mechanism is reproduced and the repair is accepted.

Implementation note: `camelyon_optimizer_training.py` is a recorded fixture for the
source-localization experiments, which parse its optimizer dispatch, so it was not
modified. The stage 2 runner mirrors its variant loop and performance gate. Tests
require bitwise-identical checkpoints and identical metrics for the shared `clean`
and `stale_head` variants, and identical gate outputs.

P2 and P3 do not specify a tolerance above. Before stage 2 was run, the runner was
written to report each comparison as `bitwise_equal`, `metrics_equal_not_bitwise` or
`differs`, with signed final-metric differences, and to apply no pass threshold.

## Observed result: stage 2 bounded training

Both seeds completed on CUDA (RTX 5060 Ti) with three epochs per variant, 24
training epochs in total and no model calls. All ten mechanism checks and all five
repair checks passed for each seed.

| Seed | Variant | ID accuracy | OOD accuracy | ID loss | OOD loss |
|---|---|---:|---:|---:|---:|
| 7 | `clean` | 0.9784 | 0.8994 | 0.06668 | 0.41611 |
| 7 | `stale_head` | 0.9792 | 0.8938 | 0.06377 | 0.39132 |
| 7 | `frozen_head` | 0.9792 | 0.8938 | 0.06377 | 0.39132 |
| 7 | `frozen_head_repaired` | 0.9784 | 0.8994 | 0.06668 | 0.41611 |
| 2026 | `clean` | 0.9794 | 0.9038 | 0.06225 | 0.37101 |
| 2026 | `stale_head` | 0.9766 | 0.8928 | 0.07775 | 0.36016 |
| 2026 | `frozen_head` | 0.9766 | 0.8928 | 0.07775 | 0.36016 |
| 2026 | `frozen_head_repaired` | 0.9794 | 0.9038 | 0.06225 | 0.37101 |

- P1 held for both seeds: the frozen head had no gradients and zero updates in
  every epoch, and its final tensors were bitwise identical to initialization.
- P2 outcome for both seeds: `bitwise_equal`. On CUDA, every non-head parameter and
  buffer of `frozen_head` matched `stale_head` bitwise after three epochs.
- P3 outcome for both seeds: `bitwise_equal`. The repaired checkpoint matched
  `clean` bitwise.
- Repair decision for both seeds: `accepted`, with structure verified and all eight
  development performance checks passed.

Against `frozen_head`, the repaired run's OOD loss ratio was 1.0633 (seed 7) and
1.0301 (seed 2026), and its ID loss ratio was 1.0456 (seed 7) and 0.8007 (seed 2026),
all within the 1.10 limit. The frozen head had lower OOD loss in both seeds. With
two seeds and a development policy, this supports neither a benefit nor a harm of a
fixed head; it does show that validation metrics alone would not reveal this fault.

A first attempt (`run-20261007T061342891035Z`) was stopped during seed 7's fourth
variant and produced no final report. It was rerun from scratch. For the three
variants it completed, final metrics were exactly equal and checkpoints were bitwise
identical to the rerun, so these CUDA runs were reproducible across processes.

Scope: two seeds, author-constructed fault and development thresholds; no
statistical or performance-recovery claim. No test-set evaluation was performed.

Reports:
`artifacts/frozen_head_training/run-20261007T062923611734Z/frozen_head_training_report.json` (seed 7),
`artifacts/frozen_head_training/run-20261007T064319356166Z/frozen_head_training_report.json` (seed 2026)
