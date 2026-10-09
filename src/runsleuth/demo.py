"""End-to-end RunSleuth demo on Camelyon17: hidden fault, diagnosis, repair, verification.

    python -m runsleuth.demo --baseline <seed baseline run> \\
        --reference <healthy run_report.json> --fault random --blind

Training is an ordinary PyTorch loop instrumented only by RunMonitor. The repair is
chosen from the diagnosis, never from the injected fault, so a wrong diagnosis
applies a wrong repair that verification rejects. A repair is accepted only when
re-diagnosis of the retrained run finds no known fault and development gate v2
passes: no regression against the healthy reference, or against the faulty run
when no reference is given. Repairs run automatically only inside this demo; for
a user's own code RunSleuth suggests them.
"""

import argparse
import json
import random
import secrets
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from runsleuth.camelyon_optimizer_probe import check_matching_data, load_reference
from runsleuth.camelyon_optimizer_training import _valid_domain_metrics
from runsleuth.camelyon_variant_runner import verify_repair
from runsleuth.diagnose_run import diagnose_run, summary_text

FAULTS = ("clean", "stale_head", "frozen_head", "high_learning_rate", "missing_optimizer_step")
EXPECTED_DIAGNOSIS = {
    "clean": "no_known_fault",
    "stale_head": "stale_optimizer_binding",
    "frozen_head": "frozen_head",
    "high_learning_rate": "high_learning_rate",
    "missing_optimizer_step": "missing_optimizer_step",
}
REPAIRS = {
    "stale_optimizer_binding": "rebuild_optimizer",
    "frozen_head": "restore_head_requires_grad",
    "high_learning_rate": "restore_learning_rate",
    "missing_optimizer_step": "enable_optimizer_step",
}
HIGH_LEARNING_RATE_FACTOR = 100
# Without a reference config, lower the rate one order below the library's
# fine-tuning ceiling (max_adamw_finetune_learning_rate = 1e-3).
FALLBACK_LEARNING_RATE = 1e-4


def build_optimizer(model, fault, repair, learning_rate, weight_decay, repaired_learning_rate=None):
    """User-style setup: replace the head and build AdamW; the fault and repair act here."""
    import torch

    if fault == "high_learning_rate":
        learning_rate *= HIGH_LEARNING_RATE_FACTOR
    if repair == "restore_learning_rate":
        learning_rate = repaired_learning_rate
    new_head = deepcopy(model.fc)
    if fault == "stale_head" and repair != "rebuild_optimizer":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        model.fc = new_head
    else:
        model.fc = new_head
        if fault == "frozen_head" and repair != "restore_head_requires_grad":
            model.fc.requires_grad_(False)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    step = not (fault == "missing_optimizer_step" and repair != "enable_optimizer_step")
    return optimizer, step


def train_run(name, *, fault, repair, config, manifest, state, directory, device, repaired_lr=None):
    """Train with a plain loop wrapped by RunMonitor; returns the run report."""
    from torch import nn

    from runsleuth.camelyon import _evaluate_domains, build_camelyon_model
    from runsleuth.camelyon_data import build_camelyon_data
    from runsleuth.run_monitor import RunMonitor, _non_finite
    from runsleuth.train import seed_everything

    seed_everything(config.seed)
    print(f"phase=load_cached_data run={name}", flush=True)
    data = build_camelyon_data(config, device=device, download=False)
    check_matching_data(manifest, data.manifest)
    model = build_camelyon_model(pretrained=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    optimizer, step = build_optimizer(
        model, fault, repair, config.learning_rate, config.weight_decay, repaired_lr
    )
    loss_function = nn.CrossEntropyLoss()
    monitor = RunMonitor(model, optimizer, directory / name, head="fc")
    config.save(monitor.run_dir / "config.json")
    try:
        with monitor:
            for epoch in range(1, config.epochs + 1):
                print(f"phase=train run={name} epoch={epoch}/{config.epochs}", flush=True)
                total, count = 0.0, 0
                with monitor.epoch():
                    model.train()
                    for inputs, targets in data.train_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        optimizer.zero_grad(set_to_none=True)
                        loss = loss_function(model(inputs), targets)
                        loss.backward()
                        if step:
                            optimizer.step()
                        total += loss.item() * len(targets)
                        count += len(targets)
                metrics = _evaluate_domains(model, data, loss_function, device)
                monitor.log(train_loss=total / count, **metrics)
    except ValueError:
        report = json.loads(monitor.report_path.read_text(encoding="utf-8"))
        if report["status"] == "diverged" or _non_finite(model):
            report["status"] = "diverged"
            monitor.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            return report
        raise
    return json.loads(monitor.report_path.read_text(encoding="utf-8"))


def _reference_learning_rate(reference_path: Path | None) -> tuple[float, str]:
    config_path = reference_path.parent / "config.json" if reference_path else None
    if config_path is not None and config_path.is_file():
        value = json.loads(config_path.read_text(encoding="utf-8")).get("learning_rate")
        if isinstance(value, (int, float)) and value > 0:
            return float(value), "reference_config"
    return FALLBACK_LEARNING_RATE, "heuristic_without_reference"


def run_demo(
    baseline: Path,
    reference_path: Path | None,
    output_dir: Path,
    *,
    fault: str = "random",
    blind: bool = False,
    epochs: int = 3,
    requested_device: str | None = None,
    client=None,
    model_name: str | None = None,
    choice_seed: int | None = None,
) -> Path:
    """Run one demo case and write demo_report.json; returns its path."""
    import torch

    from runsleuth.train import resolve_device

    if fault not in (*FAULTS, "random"):
        raise ValueError(f"Unknown fault {fault!r}; expected one of {FAULTS} or random")
    config, manifest, checkpoint = load_reference(baseline.expanduser().resolve())
    device = resolve_device(requested_device or config.device)
    config = replace(config, epochs=epochs, device=str(device))
    if fault == "random":
        choice_seed = secrets.randbits(32) if choice_seed is None else choice_seed
        fault = random.Random(choice_seed).choice(FAULTS)
    reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path else None
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output_dir / f"demo-{timestamp}"
    directory.mkdir(parents=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    shared = {"config": config, "manifest": manifest, "state": state, "directory": directory}
    report = {
        "schema_version": 1,
        "task": "runsleuth_demo",
        "blind": blind,
        "fault_choice_seed": choice_seed,
        "baseline": str(baseline),
        "reference": str(reference_path) if reference_path else None,
        "epochs": epochs,
        "device": str(device),
        "llm_requested": client is not None,
    }
    print(f"demo_directory={directory}", flush=True)
    print(f"injected_fault={'[hidden]' if blind else fault}", flush=True)

    faulty = train_run("faulty", fault=fault, repair=None, device=device, **shared)
    diagnosis = diagnose_run(
        faulty, reference, client=client, model=model_name, optimizer_name="AdamW"
    )
    (directory / "diagnosis.json").write_text(json.dumps(diagnosis, indent=2) + "\n", "utf-8")
    print(summary_text(diagnosis), flush=True)
    final = diagnosis["final_diagnosis"]
    report["diagnosis"] = {
        "final": final,
        "decided_by": diagnosis["decided_by"],
        "conflict": diagnosis["conflict"],
        "conflict_kind": diagnosis.get("conflict_kind"),
        "llm_status": diagnosis["llm"]["status"] if diagnosis["llm"] else None,
    }
    action = REPAIRS.get(final)
    report["repair"] = None
    if action is None:
        reason = "no_known_fault" if final == "no_known_fault" else f"no repair for {final}"
        report["verification"] = {"decision": "no_repair_applied", "reason": reason}
    else:
        repaired_lr, lr_source = _reference_learning_rate(reference_path)
        report["repair"] = {"action": action, "from_diagnosis": final}
        if action == "restore_learning_rate":
            report["repair"].update(learning_rate=repaired_lr, learning_rate_source=lr_source)
        print(f"phase=repair action={action}", flush=True)
        repaired = train_run(
            "repaired",
            fault=fault,
            repair=action,
            device=device,
            repaired_lr=repaired_lr,
            **shared,
        )
        recheck = diagnose_run(repaired, reference, optimizer_name="AdamW")
        variants = {"faulty": faulty, "repaired": repaired}
        usable_reference = (
            reference is not None
            and reference.get("final_metrics", {}).get("epoch") == epochs
            and _valid_domain_metrics(reference["final_metrics"])
        )
        if usable_reference:
            variants["reference"] = reference
        gate = verify_repair(
            variants,
            "repaired",
            reference="reference" if usable_reference else None,
            faulty="faulty" if _valid_domain_metrics(faulty.get("final_metrics", {})) else None,
        )
        structural = (
            repaired["status"] == "completed" and recheck["final_diagnosis"] == "no_known_fault"
        )
        report["verification"] = {
            "decision": (
                "accepted" if structural and gate["performance_nonregression"] else "rejected"
            ),
            "re_diagnosis": recheck["final_diagnosis"],
            "structure_verified": structural,
            **gate,
            "scope": "development policy; no statistical or performance-recovery claim",
        }
    report["injected_fault"] = fault
    report["expected_diagnosis"] = EXPECTED_DIAGNOSIS[fault]
    report["diagnosis_correct"] = final == EXPECTED_DIAGNOSIS[fault]
    path = directory / "demo_report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    verification = report["verification"]
    print(f"repair={report['repair']['action'] if report['repair'] else None}", flush=True)
    print(f"verification={verification['decision']}", flush=True)
    print(
        f"reveal: injected={fault} expected={EXPECTED_DIAGNOSIS[fault]} "
        f"diagnosed={final} correct={report['diagnosis_correct']}",
        flush=True,
    )
    print(f"demo_report={path}", flush=True)
    return path


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=Path, required=True, help="Completed seed baseline run")
    parser.add_argument("--reference", type=Path, help="Healthy run_report.json of the same seed")
    parser.add_argument("--fault", choices=(*FAULTS, "random"), default="random")
    parser.add_argument("--blind", action="store_true", help="Hide the fault until the end")
    parser.add_argument("--epochs", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument(
        "--provider",
        choices=("gemini", "ollama"),
        default="gemini",
        help="LLM for the review: Gemini, or a local Ollama model (OLLAMA_MODEL)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    args = parser.parse_args(argv)
    client = model_name = None
    if not args.no_llm:
        from runsleuth.llm_client import create_llm_client

        try:
            client, model_name = create_llm_client(args.provider)
        except RuntimeError as error:
            raise SystemExit(f"{error}; or rerun with --no-llm") from error
    run_demo(
        args.baseline,
        args.reference,
        args.output_dir,
        fault=args.fault,
        blind=args.blind,
        epochs=args.epochs,
        requested_device=args.device,
        client=client,
        model_name=model_name,
    )


if __name__ == "__main__":
    main()
