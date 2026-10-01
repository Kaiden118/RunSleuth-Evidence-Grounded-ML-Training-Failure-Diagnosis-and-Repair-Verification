# Camelyon17 optimizer rebinding: two-seed development results

Results recorded on 2026-10-01 UTC from completed local training and Agent
command output. Full run artifacts are stored locally under `artifacts/`.

## Controlled training

- Dataset: Camelyon17, fixed subsets of 20,000 training, 5,000 ID validation,
  and 5,000 OOD validation examples; subset seed 2026.
- Model: ImageNet-pretrained ResNet18, full fine-tuning.
- New training seeds: 7 and 2026. Each seed has a baseline and three paired
  variants: `clean`, `stale_head`, and `stale_head_repaired`.
- Each run uses three epochs: eight runs and 24 training epochs in total.
- Paired variants start from the same seed-specific initial model state.
  Rebinding happens before the first optimizer step, rather than resuming a
  trained defective checkpoint.
- Acceptance policy remains fixed: accuracy drop at most 0.01 and loss ratio
  at most 1.10, against both clean and defective comparators in both domains.

| Seed | Variant | ID accuracy | ID loss | OOD accuracy | OOD loss |
|---|---|---:|---:|---:|---:|
| 7 | Clean / repaired | 0.9784 | 0.066680 | 0.8994 | 0.416106 |
| 7 | Stale head | 0.9792 | 0.063774 | 0.8938 | 0.391319 |
| 2026 | Clean / repaired | 0.9794 | 0.062254 | 0.9038 | 0.371010 |
| 2026 | Stale head | 0.9766 | 0.077751 | 0.8928 | 0.360160 |

Both seeds passed all 19 structural checks and all eight performance checks;
both repair decisions were `accepted`. Clean and repaired final metrics
matched. Acceptance does not mean every metric improved: repaired OOD loss
was higher than the defective comparator for both seeds, within the fixed
tolerance.

The earlier seed-42 repair was structurally verified but rejected for OOD
regression. It remains a separate historical result and is not included in
this two-seed sweep's denominator.

Training summary:
`artifacts/optimizer_sweeps/20261001T033140591688Z/sweep_summary.json`.

## Agent diagnosis on saved evidence

These diagnoses inspect the completed runs; they did not initiate the sweep's
training or repairs. The optimizer profile recommends review or no change
and returns `proposed_patch: null`.

| Seed | Candidate | Implementation finding | Performance finding | Action |
|---|---|---|---|---|
| 7 | Stale head | Binding defect | Mixed | Review binding |
| 7 | Repaired | No binding defect | No observed degradation | No change |
| 2026 | Stale head | Binding defect | Mixed | Review binding |
| 2026 | Repaired | No binding defect | No observed degradation | No change |

All four completed diagnoses matched the expected case labels and passed
the first diagnosis validation, with zero validation corrections. This is
a four-case development result, not a general diagnostic accuracy estimate.
The repaired controls are separate training runs with matched initialization
and training settings, not independently sampled healthy configurations.

## API attempts and recorded usage

The first attempt completed one of four cases. Three attempts failed with
HTTP 503. Retrying each failed case once produced three additional completed
diagnoses, bringing final case completion to four of four.

Report paths use the prefix `artifacts/agent_runs/` and suffix
`/agent_report.json`:

| Seed | Candidate | First report timestamp | First result | Retry report timestamp |
|---|---|---|---|---|
| 7 | Stale head | 20261001T040926352534Z | API error (503) | 20261001T041212394580Z |
| 7 | Repaired | 20261001T040939871691Z | API error (503) | 20261001T041220265664Z |
| 2026 | Stale head | 20261001T040950456196Z | Completed | None |
| 2026 | Repaired | 20261001T040959585759Z | API error (503) | 20261001T042854633469Z |

| Scope | Agent attempts | Model calls | Tool calls | Recorded input tokens | Recorded output tokens | Usage complete |
|---|---:|---:|---:|---:|---:|---|
| Four completed reports | 4 | 8 | 8 | 30,454 | 5,014 | Yes |
| All attempts, including API errors | 7 | 12 | 10 | 31,363 | 5,179 | No |

The failed reports remain part of the record. All-attempt token totals are
recorded usage only, because the three API-error reports have incomplete
usage information. First-pass diagnosis validity and first-attempt API
completion are separate measurements.

## Scope

This study covers one injected optimizer-binding defect on two new training
seeds, one architecture, and fixed data subsets. OOD validation contributes
to repair acceptance; it is not an untouched final test set. No official
Camelyon17 test result or clinical-use claim is established here.
