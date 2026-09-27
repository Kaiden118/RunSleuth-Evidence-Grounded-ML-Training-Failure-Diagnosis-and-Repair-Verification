# Camelyon17 optimizer-binding probe

This feature adds a read-only optimizer membership audit and a controlled,
single-batch code-fault experiment. Overlay these files onto the existing
RunSleuth repository after the Camelyon17 baseline has completed.

## Why this fault matters

An optimizer retains the parameter objects supplied during construction.
Replacing a classification head afterward creates new parameter objects; equal
names, shapes, or values do not add them to the existing optimizer.
The backbone can continue learning, so a nonzero global update norm can hide
this defect.

The experiment runs this order for the clean case:

```python
new_head = deepcopy(model.fc)
model.fc = new_head
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
```

The faulty case changes only the last two operations:

```python
new_head = deepcopy(model.fc)
optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
model.fc = new_head
```

The deep copy preserves the reference's head weights while creating new
parameter objects. It isolates optimizer binding from random initialization.
Both cases clear model gradients before backward, including the new head, to
avoid introducing gradient accumulation as a second fault.

## Controlled comparison

- Load the baseline's **initial** checkpoint, not its already-trained final model.
- Verify the initial checkpoint and data manifest against recorded hashes.
- Reconstruct the same pinned dataset and compare selected sample IDs/metadata.
- Select one training batch, record its sample IDs and tensor hash, reuse it.
- Use fresh models with identical parameters and buffers; run in training mode.
- Execute exactly **one optimizer step per case**, two steps in total.
- Compare the classification head and backbone separately.
- Do not evaluate ID/OOD validation or the official test set. The existing data
  builder validates metadata for the cached train and validation splits; only
  the selected training batch is decoded for this probe.

The setup uses existing cached data and `pretrained=False` before loading the
saved initial checkpoint, so no dataset or pretrained-weight download is needed.
It writes a new directory under `artifacts/optimizer_probes/` and retains the
reference run unchanged.

## Expected evidence (to be confirmed by the run)

| Observation | Clean | Stale-head optimizer |
|---|---|---|
| Initial state and pre-update loss | Same | Same |
| Missing current trainable parameters | None | `fc.weight`, `fc.bias` |
| Optimizer tensors absent from current model | 0 | 2 |
| Head gradient norm | Positive | Positive |
| Head update norm | Positive | Zero |
| Backbone update norm | Positive | Positive |

The audit compares object identities internally and reports names/counts.
It does not serialize Python object IDs. A missing trainable parameter can be
intentional in another workload; this experiment specifies full-model training.

## Run (Windows CMD)

From the repository root in the existing `runsleuth` environment, using the
completed Camelyon baseline:

```bat
python scripts\check_optimizer_probe.py --reference-run artifacts/camelyon_runs/camelyon17-clean-20260927T002604680036Z
```

The script formats the new files, runs pytest/Ruff/whitespace checks, and then
starts the probe only if every command succeeds. It neither commits nor pushes.
Report: `artifacts/optimizer_probes/probe-<timestamp>/optimizer_probe_report.json`.
The report includes source snapshots/hashes, batch identity, parameter audit,
gradients, updates, and explicit mechanism checks. Runtime failures produce a
failed report; unmet mechanism checks produce `inconclusive` and a nonzero exit.

## Scope

A reproduced parameter-binding defect does **not** yet prove a large accuracy
loss or successful training repair. A backbone may partly compensate for a
fixed head. Full training comparisons and bounded repair verification follow
only after these low-level signals are confirmed. No LLM is called in this probe;
the audit is a primitive that can later be exposed as a typed diagnostic tool.

PyTorch references:
- <https://docs.pytorch.org/docs/main/optim.html>
- <https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html>

Local preparation checks: 19 dependency-free tests passed; 10 Torch CPU tests
were skipped because Torch was unavailable. A mocked CLI check verified report
writing and rejection of changed sample IDs. Actual Torch execution and the
full repository suite must pass in the target environment.

## Optimizer Rebinding Verification

A three-way single-step experiment compared `clean`, `stale_head`, and
`stale_head_repaired` using the same initial model state and input batch.

The faulty optimizer omitted the two current classification-head parameter
tensors and retained two obsolete tensors. Rebuilding AdamW from the current
model parameters removed both binding errors without changing model weights.

| Measurement | Clean | Stale head | Repaired |
|---|---:|---:|---:|
| Missing current parameter tensors | 0 | 2 | 0 |
| Foreign parameter tensors | 0 | 2 | 0 |
| Head gradient norm | 3.901329 | 3.901329 | 3.901329 |
| Head update norm | 0.003203 | 0.000000 | 0.003203 |

All nine repair-mechanism checks passed. The repaired step matched the
clean step's measured gradient and update norms.

Scope: rebuilding occurred before the first optimizer step, with empty
optimizer state. This verifies the update mechanism; it does not establish
validation-performance improvement or recovery from a trained checkpoint.

Report: `artifacts/optimizer_probes/probe-20260927T052121051006Z/optimizer_probe_report.json`