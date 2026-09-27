# Camelyon17 clean baseline

This is an **overlay for the existing RunSleuth repository**, not a standalone
project. Merge its folders into the repository root. It adds four modules,
five test files, a config, this guide, and check scripts. The only
existing file replaced is `pyproject.toml`, which adds the `camelyon` extra.
Existing `train.py`, `telemetry.py`, and `README.md` remain required and unchanged.

## Purpose

Establish healthy training signals on tumor/non-tumor histopathology patches
before injecting faults. Separating same-hospital (ID) and new-hospital (OOD)
validation helps distinguish optimization failure from distribution shift.
An OOD accuracy drop alone does not prove a training bug.

This round is a clean baseline. The existing FashionMNIST diagnosis and repair
commands do not yet support the new `CamelyonConfig`. No LLM calls are made.

## Protocol

| Setting | Value |
|---|---|
| Model | ImageNet-pretrained ResNet18; binary classification head; all parameters trainable |
| Input | Native 96 x 96 RGB; ImageNet normalization; no resize/crop/augmentation |
| Training subset | 20,000 patches from hospitals 0, 3, 4 |
| ID validation subset | 5,000 patches from held-out validation rows of hospitals 0, 3, 4 |
| OOD validation subset | 5,000 patches from hospital 1 |
| Official test | Hospital 2; neither downloaded nor evaluated |
| Subset selection | Proportional label/hospital strata; fixed subset seed 2026 |
| Training | Seed 42, 3 epochs, batch 32, AdamW, learning rate 0.0001, weight decay 0.0001 |
| Checkpoint | Fixed final epoch; no selection on OOD validation |

`seed` controls initialization and loader order; `subset_seed` controls which
patches are selected. Keeping them separate supports later comparable fault runs.
Epoch 0 records the pretrained backbone with a random binary head; it is an
initialization check, not an established healthy performance target.
A majority-class baseline is also stored in `data_manifest.json`.

## Data provenance

The WILDS project's legacy download service has reported availability problems.
This implementation uses the public **community mirror**, not a claimed official
replacement:

- Repository: <https://huggingface.co/datasets/wltjr1007/Camelyon17-WILDS>
- Fixed revision: `d784d5344ba6c967f83f9f3d9b2f1e2a4d6eb78f`
- Upstream protocol: <https://github.com/p-lambda/wilds/blob/main/wilds/datasets/camelyon17_dataset.py>
- Download issue: <https://github.com/p-lambda/wilds/issues/175>

The mirror combines ID and OOD validation rows. The loader reconstructs these
using hospital metadata and checks counts (302,436 train; 33,560 ID validation;
34,904 OOD validation), categories, unique image IDs, and split separation.
It stops on mismatch rather than silently using a different protocol.

Image/metadata identity with the original archive has **not** been verified.
Reports explicitly record `community_mirror` and `equivalence_not_verified`.
This subset experiment is not a reproduction of WILDS leaderboard results.

The first run downloads **all 14 train and 3 validation Parquet shards** before
selecting the 30,000 used patches. A subset cap does not reduce this download.
Allow time for several GB of downloads and reserve approximately 30 GB for
Parquet files, Arrow caches, and checkpoints. This is a planning allowance,
not an exact disk requirement. No test shards are requested.

## Run on Windows CMD

Run from the existing repository root, in the `runsleuth` environment.

```bat
python -m pip install -e ".[dev,camelyon]" && python -m pip check
```

The check script organizes imports/formats only the Camelyon modules, tests, and runner,
then runs the full pytest suite, repository lint, and whitespace checks.
Python executes each command with `subprocess.run(check=True)`. The same process
starts training only after every check succeeds. It does not commit or push.

```bat
python scripts\check_camelyon.py --train --download
```

`--download` permits missing dataset shards to be fetched and cached. Later
runs can omit this flag. ResNet18 weights have a separate PyTorch cache and may
still download on the first training run. Neither step needs a Gemini API key.
To prepare/cache data without training, optionally add `--prepare-only`.

Do not rerun the full download because `phase=prepare_data` takes time: this
phase downloads shards, builds Arrow data, and validates metadata. If a command
fails, retain and share its error instead of deleting the previous run.

The supplied config stores data under `C:/Users/ADMIN/rs-data/c17` to avoid
Windows path-length limits from nesting Hugging Face caches inside the long
repository path. Hub download files retain their existing revision-pinned cache
location; the separate Arrow conversion cache is placed directly under
`data_dir/arrow` because Datasets embeds this path again in lock filenames.
After all data shards are cached, use `python scripts\check_camelyon.py --train`
to reuse them without dataset downloads. No repository move or Windows registry
change is required.

## Output and interpretation

Each run writes to a new `artifacts/camelyon_runs/<name>-<timestamp>/` directory:

| File | Meaning |
|---|---|
| `config.json`, `environment.json` | Exact settings, installed versions, device, source hashes |
| `source_snapshot/` | Copies of the four new modules and existing train/telemetry modules |
| `data_manifest.json` | Provenance, selected sample IDs/hashes, class/hospital counts |
| `baseline_metrics.json` | Epoch 0 ID/OOD initialization metrics |
| `metrics.jsonl` | Existing telemetry schema; `validation_*` means **ID validation** here |
| `domain_metrics.jsonl` | Separate ID/OOD metrics, training-plus-validation epoch time, peak allocated CUDA memory |
| `initial_state_dict.pt`, `model_state_dict.pt` | Initial/final model weights |
| `run_report.json` | Completion/failure, completed epochs, hashes, final metrics |

`peak_cuda_memory_mb` is PyTorch peak allocated memory, not total GPU usage.
The original loop snapshots parameters to measure updates, so this first run
also measures telemetry overhead. GPU memory fit and wall time must be measured
on the actual machine. An `accepted` repair decision is not produced here.

Inspect finite losses, nonzero gradients and updates, learning relative to
initialization/majority baselines, and ID/OOD behavior. Do not set acceptance
thresholds after viewing a failed run or tune against the sealed test set.

## Verification supplied with this patch

The package includes 32 new tests. In the preparation environment, 23 passed
and 9 Torch/torchvision integration tests were skipped because those packages
were unavailable. The included CPU integration tests use synthetic data and
never download the dataset or pretrained weights. Full local pytest, Ruff,
real dataset decoding, and GPU training must pass on the target environment.
