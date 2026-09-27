# Camelyon17 ResNet18 development baseline

Run: `artifacts/camelyon_runs/camelyon17-clean-20260927T002604680036Z`

Status: `completed`. The accompanying repository checks reported 250 tests
passed, 2 skipped, and 185 subtests passed; Ruff and `git diff --check` passed.

## Protocol

- ImageNet-pretrained ResNet18, two-class head, full fine-tuning.
- Native 96 x 96 patches; ImageNet normalization; no augmentation.
- Fixed subsets: 20,000 training, 5,000 ID validation, 5,000 OOD validation.
- Training seed 42; subset seed 2026; batch size 32.
- AdamW; learning rate 0.0001; weight decay 0.0001; three epochs.
- Checkpoint selection: fixed final epoch. Official test split remains unused.
- Data: `wltjr1007/Camelyon17-WILDS`, revision
  `d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f`.
  This is a community mirror; identity with the original WILDS archive is unverified.

## Recorded results

Epoch 0 evaluates the pretrained backbone with a newly initialized binary head.
ID validation uses held-out patches from training hospitals; OOD validation uses
hospital 1, which is absent from training.

| Epoch | Train loss | Train accuracy | ID loss | ID accuracy | OOD loss | OOD accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 0 | — | — | 0.657765 | 60.48% | 0.667782 | 61.06% |
| 1 | 0.132217 | 95.02% | 0.070936 | 97.48% | 0.245131 | 91.96% |
| 2 | 0.048243 | 98.25% | 0.069526 | 97.64% | 0.259432 | 92.04% |
| 3 | 0.025434 | 99.18% | 0.074803 | 97.86% | 0.378694 | 90.16% |

| Epoch | Mean gradient L2 norm | Mean parameter-update L2 norm | Training + validation seconds |
|---|---:|---:|---:|
| 1 | 4.226915 | 0.059095 | 43.46 |
| 2 | 2.284870 | 0.048098 | 53.26 |
| 3 | 1.656394 | 0.047263 | 64.86 |

The three epoch loops totaled 161.59 seconds, excluding data preparation, weight
downloads, epoch-0 evaluation, and other setup. PyTorch peak allocated CUDA
memory was 415.41 MiB in each epoch; this is not total GPU memory usage.

## Interpretation

Loss reduction and nonzero gradients/parameter updates show that learning and
optimization occurred. Final ID accuracy exceeds OOD accuracy by 7.70 percentage
points. OOD accuracy falls 1.88 points between epochs 2 and 3 while OOD loss rises;
this observation alone does not establish a training-code defect.

The epoch-3 checkpoint remains the recorded reference under the fixed-epoch
protocol. This single-seed subset run establishes a development baseline, not
a full WILDS benchmark result or validation of Agent repairs on Camelyon17.
Repair verification thresholds for this task have not yet been established.

Full-precision metrics, source snapshots, selected sample IDs, provenance, and
checkpoint hashes are stored in the run directory. Preserve those local artifacts
for the next controlled fault experiment; they are excluded from Git.
