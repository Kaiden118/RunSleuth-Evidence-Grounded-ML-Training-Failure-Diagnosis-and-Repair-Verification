"""Hugging Face DeiT faults on Camelyon17, trained by the Trainer with RunSleuthCallback.

Variants share the saved Camelyon17 subset, seeds and Trainer settings (bf16 on CUDA,
gradient clipping 1.0, constant learning rate):

- clean: a new 2-class head; ImageNet normalization, as the model card documents,
  for training and evaluation.
- train_eval_normalization_mismatch: training uses the checkpoint's image processor
  normalization (mean = std = 0.5), which disagrees with its model card, while
  evaluation uses ImageNet normalization.
- imagenet_head_kept: from_pretrained without num_labels keeps the 1000-class
  ImageNet head; training on 0/1 labels runs without any error.
- frozen_patch_embedding: the patch embedding projection is accidentally frozen.

Each fault has a repaired variant, verified by bounded retraining under the
unchanged development performance policy. Input statistics are recorded on the
exact tensors the model receives. Training is made deterministic (deterministic
algorithms, eager attention, a fixed cuBLAS workspace set before CUDA starts) so
identical configurations reproduce bitwise; resizing and normalization run on the
training device.
"""

import argparse
import json
import os
import platform
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from runsleuth.camelyon_data import IMAGENET_MEAN, IMAGENET_STD
from runsleuth.camelyon_optimizer_probe import check_matching_data, load_reference, read_json
from runsleuth.camelyon_optimizer_training import (
    REPAIR_POLICY,
    _complete_coverage,
    _valid_domain_metrics,
    write_json,
)
from runsleuth.camelyon_variant_runner import (
    metric_differences,
    performance_checks,
    snapshot_sources_with,
)
from runsleuth.optimizer_probe import state_dict_sha256
from runsleuth.signature_matching import input_shift

MODEL_NAME = "facebook/deit-small-patch16-224"
IMAGE_SIZE = 224
NORMALIZATIONS = {
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "processor": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}
FAULTS = {
    "train_eval_normalization_mismatch": "train_eval_normalization_mismatch_repaired",
    "imagenet_head_kept": "imagenet_head_kept_repaired",
    "frozen_patch_embedding": "frozen_patch_embedding_repaired",
}
VARIANTS = ("clean", *(name for pair in FAULTS.items() for name in pair))
PATCH_EMBEDDING = "vit.embeddings.patch_embeddings"
METRIC_NAMES = {
    f"eval_{domain}_{metric}": f"{domain}_validation_{metric}"
    for domain in ("id", "ood")
    for metric in ("loss", "accuracy")
}
# Stated before any ViT run: splits from one distribution through one pipeline should
# agree closely. Heuristics, not calibrated on these runs.
AGREEING_INPUTS = {"standardized_mean_shift": 0.25, "std_ratio": 1.25}
SHIFTED_INPUTS = {"standardized_mean_shift": 0.5, "std_ratio": 1.5}
EXTRA_SOURCES = (
    "camelyon_vit.py",
    "hf_callback.py",
    "run_monitor.py",
    "signature_matching.py",
    "camelyon_variant_runner.py",
)


def variant_plan() -> dict[str, dict]:
    """Normalization per split, head size, frozen patch embedding and repair record."""

    def plan(train="imagenet", evaluation="imagenet", num_labels=2, freeze=False, repair=None):
        return {
            "train_normalization": train,
            "eval_normalization": evaluation,
            "num_labels": num_labels,
            "freeze_patch_embedding": freeze,
            "repair": repair,
        }

    return {
        "clean": plan(),
        "train_eval_normalization_mismatch": plan(train="processor"),
        "train_eval_normalization_mismatch_repaired": plan(
            train="processor",
            evaluation="processor",
            repair={
                "action": "align_eval_normalization_with_training",
                "normalization": "processor",
            },
        ),
        "imagenet_head_kept": plan(num_labels=None),
        "imagenet_head_kept_repaired": plan(
            repair={"action": "reload_with_num_labels", "num_labels": 2}
        ),
        "frozen_patch_embedding": plan(freeze=True),
        "frozen_patch_embedding_repaired": plan(
            repair={"action": "restore_requires_grad", "module": PATCH_EMBEDDING}
        ),
    }


@contextmanager
def deterministic_training():
    """Enable deterministic algorithms for the block and restore the previous settings.

    The cuBLAS workspace setting only takes effect if set before CUDA creates its
    handles, so enter this before any CUDA work in the process.
    """
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    try:
        yield {
            "deterministic_algorithms": True,
            "attention": "eager",
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        }
    finally:
        torch.use_deterministic_algorithms(previous[0])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous[1:]


def load_pretrained(plan: dict):
    """Load the cached DeiT checkpoint with eager attention; without num_labels the
    ImageNet head is kept."""
    from transformers import AutoModelForImageClassification

    options = {"attn_implementation": "eager"}
    if plan["num_labels"] is not None:
        options |= {"num_labels": plan["num_labels"], "ignore_mismatched_sizes": True}
    return AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, local_files_only=True, **options
    )


def model_commit() -> str | None:
    """The cached snapshot's commit, read from the local Hugging Face cache path."""
    try:
        from transformers.utils import cached_file

        return Path(cached_file(MODEL_NAME, "config.json", local_files_only=True)).parent.name
    except (ImportError, OSError):
        return None


def to_model_inputs(images, normalization: str, image_size: int):
    """Undo the cached ImageNet normalization, resize, then apply the chosen normalization."""
    import torch
    from torch.nn import functional

    def column(values):
        return torch.tensor(values, dtype=images.dtype, device=images.device).view(1, 3, 1, 1)

    raw = images * column(IMAGENET_STD) + column(IMAGENET_MEAN)
    if raw.shape[-1] != image_size or raw.shape[-2] != image_size:
        raw = functional.interpolate(
            raw, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
    mean, std = NORMALIZATIONS[normalization]
    return (raw - column(mean)) / column(std)


class SplitDataset:
    """Tag each Camelyon17 item with its split so one collator can normalize per split."""

    def __init__(self, dataset, split: str) -> None:
        self.dataset, self.split = dataset, split

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        image, label = self.dataset[index]
        return {"image": image, "label": int(label), "split": self.split}


def split_collate(items: list[dict]) -> dict:
    """Stack cached images and labels; the split travels with the batch."""
    import torch

    split = items[0]["split"]
    if any(item["split"] != split for item in items):
        raise ValueError("A batch mixes splits")
    return {
        "pixel_values": torch.stack([item["image"] for item in items]),
        "labels": torch.tensor([item["label"] for item in items]),
        "split": split,
    }


class SplitPreprocessor:
    """On the training device: apply the split's normalization, resize, record inputs."""

    def __init__(self, plan: dict, callback, image_size: int) -> None:
        self.plan, self.callback, self.image_size = plan, callback, image_size

    def __call__(self, inputs: dict) -> dict:
        inputs = dict(inputs)
        split = inputs.pop("split")
        key = "train_normalization" if split == "train" else "eval_normalization"
        pixel_values = to_model_inputs(inputs["pixel_values"], self.plan[key], self.image_size)
        self.callback.observe_batch(split, pixel_values, inputs["labels"])
        return {**inputs, "pixel_values": pixel_values}


def _trainer_class():
    from transformers import Trainer

    class PreprocessingTrainer(Trainer):
        """Run SplitPreprocessor after the Trainer has moved a batch to the device."""

        def __init__(self, *args, preprocess, **kwargs):
            super().__init__(*args, **kwargs)
            self.preprocess = preprocess

        def _prepare_inputs(self, inputs):
            return self.preprocess(super()._prepare_inputs(inputs))

    return PreprocessingTrainer


def _accuracy(prediction) -> dict:
    import numpy as np

    logits = (
        prediction.predictions[0]
        if isinstance(prediction.predictions, tuple)
        else prediction.predictions
    )
    return {"accuracy": float((np.argmax(logits, axis=-1) == prediction.label_ids).mean())}


def run_variant(
    name, plan, *, config, data, directory, device, model_factory, image_size, max_steps=None
):
    """Train one variant with the Trainer and return its RunSleuth run report."""
    from transformers import TrainingArguments, set_seed

    from runsleuth.hf_callback import RunSleuthCallback

    print(f"phase=train variant={name}", flush=True)
    set_seed(config.seed)
    model = model_factory(plan)
    if plan["freeze_patch_embedding"]:
        model.get_submodule(PATCH_EMBEDDING).requires_grad_(False)
    callback = RunSleuthCallback(directory / name, head="classifier", rename=METRIC_NAMES)
    cuda = device.type == "cuda"
    arguments = TrainingArguments(
        output_dir=str(directory / "trainer" / name),
        num_train_epochs=config.epochs,
        max_steps=-1 if max_steps is None else max_steps,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        lr_scheduler_type="constant",
        warmup_steps=0,
        max_grad_norm=1.0,
        bf16=cuda,
        use_cpu=not cuda,
        optim="adamw_torch",
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="no",
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
        dataloader_num_workers=0,
        disable_tqdm=True,
        remove_unused_columns=False,
    )
    trainer = _trainer_class()(
        model=model,
        args=arguments,
        train_dataset=SplitDataset(data.train_loader.dataset, "train"),
        eval_dataset={
            "id": SplitDataset(data.id_val_loader.dataset, "id_eval"),
            "ood": SplitDataset(data.ood_val_loader.dataset, "ood_eval"),
        },
        data_collator=split_collate,
        compute_metrics=_accuracy,
        callbacks=[callback],
        preprocess=SplitPreprocessor(plan, callback, image_size),
    )
    started = perf_counter()
    trainer.train()
    path = directory / name / "run_report.json"
    report = read_json(path)
    report.update(
        variant=name,
        plan=plan,
        final_state_sha256=state_dict_sha256(model.state_dict()),
        elapsed_seconds=perf_counter() - started,
    )
    if plan["repair"] is not None:
        report["repair"] = plan["repair"]
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def _rows(variant: dict) -> list[dict]:
    return variant["parameter_group_epochs"]


def _completed(variant: dict, epochs: int) -> bool:
    return (
        variant["status"] == "completed"
        and [row["epoch"] for row in _rows(variant)] == list(range(1, epochs + 1))
        and _valid_domain_metrics(variant.get("final_metrics", {}))
    )


def _learns(variant: dict) -> bool:
    return bool(_rows(variant)) and all(
        row["optimizer_steps"] == row["training_forwards"] > 0
        and all(row[group]["mean_parameter_update_l2_norm"] > 0 for group in ("head", "backbone"))
        for row in _rows(variant)
    )


def _agreeing(shift: dict | None) -> bool:
    return shift is not None and all(shift[key] < limit for key, limit in AGREEING_INPUTS.items())


def _shifted(shift: dict | None) -> bool:
    return shift is not None and any(shift[key] >= limit for key, limit in SHIFTED_INPUTS.items())


def _fully_trainable_every_epoch(variant: dict, group: str) -> bool:
    return bool(_rows(variant)) and all(
        row[group]["min_trainable_tensors"] == row[group]["parameter_tensors"]
        and row[group]["min_tensors_with_gradient"] == row[group]["parameter_tensors"]
        for row in _rows(variant)
    )


def assess_vit_faults(variants: dict, plan: dict, epochs: int) -> dict:
    """Mechanism checks, one repair verification per fault, and observed differences."""
    clean = variants["clean"]
    shifts = {name: input_shift(variant) for name, variant in variants.items()}
    clean_steps = [row["optimizer_steps"] for row in _rows(clean)]
    frozen = variants["frozen_patch_embedding"]
    frozen_names = sorted(f"{PATCH_EMBEDDING}.projection.{kind}" for kind in ("weight", "bias"))
    common = {
        "all_variants_completed_fixed_budget": all(
            _completed(variant, epochs) for variant in variants.values()
        ),
        "same_step_counts_with_a_step_per_forward": all(
            [row["optimizer_steps"] for row in _rows(variant)] == clean_steps
            and all(row["optimizer_steps"] == row["training_forwards"] for row in _rows(variant))
            for variant in variants.values()
        ),
        "all_optimizer_audits_complete": all(
            _complete_coverage(variant["optimizer_audit"])
            and _complete_coverage(variant.get("final_optimizer_audit", {}))
            for variant in variants.values()
        ),
        "clean_inputs_agree_between_train_and_eval": _agreeing(shifts["clean"]),
        "clean_learns_with_a_step_per_forward": _learns(clean),
    }
    fault_checks = {
        "train_eval_normalization_mismatch": {
            "mismatch_inputs_shift_between_train_and_eval": _shifted(
                shifts["train_eval_normalization_mismatch"]
            ),
        },
        "imagenet_head_kept": {
            "kept_head_has_1000_outputs_for_2_label_classes": (
                variants["imagenet_head_kept"]["head_output_units"] == 1000
                and variants["imagenet_head_kept"]["label_classes"].get("train") == [0, 1]
                and clean["head_output_units"] == 2
            ),
        },
        "frozen_patch_embedding": {
            "only_the_patch_embedding_is_frozen": (
                frozen["trainability"]["backbone"]["frozen_names"] == frozen_names
                and not frozen["trainability"]["head"]["frozen_names"]
            ),
            "frozen_tensors_get_no_gradients_every_epoch": bool(_rows(frozen))
            and all(
                row["backbone"]["max_trainable_tensors"]
                == row["backbone"]["max_tensors_with_gradient"]
                == row["backbone"]["parameter_tensors"] - len(frozen_names)
                for row in _rows(frozen)
            ),
            "frozen_variant_still_learns_elsewhere": _learns(frozen),
        },
    }
    repair_checks = {
        "train_eval_normalization_mismatch": {
            "repaired_eval_uses_training_normalization": plan[
                "train_eval_normalization_mismatch_repaired"
            ]["eval_normalization"]
            == plan["train_eval_normalization_mismatch_repaired"]["train_normalization"],
            "repaired_inputs_agree_between_train_and_eval": _agreeing(
                shifts["train_eval_normalization_mismatch_repaired"]
            ),
        },
        "imagenet_head_kept": {
            "repaired_head_has_2_outputs": variants["imagenet_head_kept_repaired"][
                "head_output_units"
            ]
            == 2,
        },
        "frozen_patch_embedding": {
            "repaired_backbone_fully_trainable_every_epoch": _fully_trainable_every_epoch(
                variants["frozen_patch_embedding_repaired"], "backbone"
            ),
        },
    }
    verification = {}
    for fault, repaired_name in FAULTS.items():
        repaired, faulty = variants[repaired_name], variants[fault]
        structural = {
            **common,
            **fault_checks[fault],
            **repair_checks[fault],
            "repaired_learns_with_a_step_per_forward": _learns(repaired),
        }
        comparators = ("clean", fault) if _completed(faulty, epochs) else ("clean",)
        performance = performance_checks(variants, repaired_name, comparators)
        structure_verified = all(structural.values())
        passed = all(check["passed"] for check in performance)
        verification[fault] = {
            "candidate": repaired_name,
            "decision": "accepted" if structure_verified and passed else "rejected",
            "structural_checks": structural,
            "structure_verified": structure_verified,
            "policy": dict(REPAIR_POLICY),
            "comparators": list(comparators),
            "performance_checks": performance,
            "performance_nonregression": passed,
            "scope": "development policy; no statistical or performance-recovery claim",
        }
    observations = {
        "input_shift_train_vs_id_eval": shifts,
        "final_metric_differences_vs_clean": {
            name: metric_differences(variant, clean)
            for name, variant in variants.items()
            if name != "clean"
        },
        # These repaired variants are configured exactly like clean, so with
        # deterministic training they should reproduce it bitwise.
        "clean_configured_repairs_bitwise_equal_to_clean": {
            name: variants[name].get("final_state_sha256") is not None
            and variants[name].get("final_state_sha256") == clean.get("final_state_sha256")
            for name in ("imagenet_head_kept_repaired", "frozen_patch_embedding_repaired")
        },
        "processor_vs_imagenet_normalization": {
            "description": "processor (0.5) normalization for training and evaluation minus "
            "ImageNet normalization for both; answers which suits these weights here",
            "differences": metric_differences(
                variants["train_eval_normalization_mismatch_repaired"], clean
            ),
        },
    }
    mechanism = all(common.values()) and all(
        all(checks.values()) for checks in fault_checks.values()
    )
    return {
        "checks": {"common": common, **fault_checks},
        "mechanism_reproduced": mechanism,
        "repair_verification": verification,
        "observations": observations,
    }


def run_vit_faults(
    reference_run: Path,
    output_dir: Path,
    epochs: int = 3,
    requested_device: str | None = None,
    *,
    model_factory=load_pretrained,
    image_size: int = IMAGE_SIZE,
) -> Path:
    """Train all seven variants on the reference run's data subset and write one report."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 3:
        raise ValueError("epochs must be an integer between 1 and 3")
    import torch

    from runsleuth.camelyon_data import build_camelyon_data
    from runsleuth.train import resolve_device

    reference_run = reference_run.expanduser().resolve()
    config, manifest, _ = load_reference(reference_run)
    device = resolve_device(requested_device or config.device)
    config = replace(config, epochs=epochs, device=str(device))
    plan = variant_plan()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output_dir / f"run-{timestamp}"
    directory.mkdir(parents=True)
    report_path = directory / "vit_fault_report.json"
    report = {
        "schema_version": 1,
        "status": "running",
        "error": None,
        "task": "camelyon17_deit_fault_training",
        "model": MODEL_NAME,
        "reference_run": str(reference_run),
        "seed": config.seed,
        "epochs_per_variant": epochs,
        "device": str(device),
        "image_size": image_size,
        "test_evaluated": False,
        "budget": {"training_epoch_upper_bound": len(VARIANTS) * epochs, "model_calls": 0},
        "thresholds": {"agreeing_inputs": AGREEING_INPUTS, "shifted_inputs": SHIFTED_INPUTS},
        "variant_plan": plan,
        "variants": {},
    }
    print(f"experiment_directory={directory}", flush=True)
    started = perf_counter()
    determinism = deterministic_training()
    try:
        report["determinism"] = determinism.__enter__()
        report["model_commit"] = model_commit() if model_factory is load_pretrained else None
        write_json(
            directory / "environment.json",
            {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "device_name": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else "CPU",
                "optimizer": "AdamW",
                "source_sha256": snapshot_sources_with(directory, EXTRA_SOURCES),
            },
        )
        print("phase=load_cached_data", flush=True)
        data = build_camelyon_data(config, device=device, download=False)
        check_matching_data(manifest, data.manifest)
        for variant in VARIANTS:
            report["variants"][variant] = run_variant(
                variant,
                plan[variant],
                config=config,
                data=data,
                directory=directory,
                device=device,
                model_factory=model_factory,
                image_size=image_size,
            )
        report.update(assess_vit_faults(report["variants"], plan, epochs))
        report["status"] = "completed" if report["mechanism_reproduced"] else "inconclusive"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        determinism.__exit__(None, None, None)
        report["elapsed_seconds"] = perf_counter() - started
        write_json(report_path, report)
        print(f"vit_fault_report={report_path}", flush=True)
    decisions = {fault: item["decision"] for fault, item in report["repair_verification"].items()}
    print(f"status={report['status']}", flush=True)
    print("decisions=" + json.dumps(decisions), flush=True)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/vit_fault_training"))
    parser.add_argument("--epochs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args()
    report = read_json(
        run_vit_faults(args.reference_run, args.output_dir, args.epochs, args.device)
    )
    decisions = [item["decision"] for item in report.get("repair_verification", {}).values()]
    if report["status"] != "completed" or "rejected" in decisions or not decisions:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
