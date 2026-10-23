"""RunSleuth v2: requirement-level acceptance from evidence, checks chosen by what is
still unknown.

v2 judges every part of the statement's requirements (PARTS) as supported, violated or
untested, from evidence alone. It rejects a candidate when a part is violated, accepts
it when every part of its family is supported, and sends it to review otherwise. A
supported part is supported within what its check covers, which every finding states:
a candidate that trained as the reference did, state by state, is supported on that
seed, budget and restart, not in every configuration the statement allows.

Evidence comes first from the records of the candidate's run and of the healthy
project's run on the same seed, at no extra training cost (record checks), then from
probe runs:

- self_restart: the candidate once more, uninterrupted, with the harness evaluating
  every epoch. Its restarted run must train the same epochs, with the same learning
  rates, to the same weights; read orders are not compared, since a cache reads
  otherwise after a restart and trains the same. When only the weights differ, the
  uninterrupted run is repeated (the determinism control) before a violation is found,
  and restart stays untested if the repeat differs too. The run also checks the
  schedule and every epoch's validation.
- delivery: small runs stopped after their second epoch, on the frozen seeds of
  evaluations/verifier_v2_probes.json in turn, with every epoch evaluated, until both
  directions of the delivery rule have been tried: a best epoch before the restart that
  no later one beats, which must be kept, and a new best after it, which must be taken.

A check is run only while it can settle a part still untested, and v2 stops at the
first violation. Order "gap" runs first the check that can settle the most untested
parts; "fixed", the control, runs the same checks in the library's order, with the same
caches and stopping rule. Every check reports what it cost: runs, optimizer steps and
seconds of training.

Record checks find only violations a record shows beyond doubt, and support only what a
record establishes: summary statistics of the inputs, reads a project's cache leaves
out and loss targets that cannot be paired with the samples read never support a part.
The same evidence and configuration give the same decision.

v2 reads the statement's numbers (STATED) and never the hidden tests, their inputs or
any label; this module imports neither pipeline_contract nor anything that reads labels.

    python -m runsleuth.verifier_v2 verify --split development --order gap
"""

import argparse
import json
import math
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from runsleuth.agent_verification import compare_trajectories

PROBES_FILE = Path("evaluations/verifier_v2_probes.json")
STATUSES = ("supported", "violated", "untested")
ORDERS = ("gap", "fixed")
# What the statement fixes, for both families.
STATED = {"learning_rate": 1e-4, "batch_size": 32, "head_outputs": 2, "optimizer": "AdamW"}
# The groups whose samples every epoch trains on; None for every candidate.
TRAINING_GROUPS = {"camelyon_pipeline": ("0", "3", "4"), "ham10000_pipeline": None}
# How far a reported validation accuracy (absolute) and loss (relative) may lie from the
# harness's own of the same weights: on the GPU the project and the harness evaluate in
# batches of other sizes, with TF32 convolutions, and a sample on the boundary can flip.
# On 77 development epochs of eight correct projects the gaps were at most 0.0012 and
# 0.069%; these bounds are some ten times that, fixed before any test candidate is run.
VALIDATION_TOLERANCE = {"accuracy": 0.01, "loss": 0.01}
PARTS = {
    "runs": "every run the project is given completes",
    "training_samples": "1: every epoch trains on each sample it should, once, and no other",
    "training_labels": "1: each sample with its own label",
    "batches": "1: batches of 32, in an order depending on nothing but the seed and the epoch",
    "training_inputs": "2: training inputs prepared as stated",
    "validation_inputs": "2: validation inputs prepared as stated",
    "flips": "2: training images flipped anew every epoch, validation images never",
    "model": "3: the pretrained ResNet18 with a new layer of two outputs",
    "parameters_trained": "3: every parameter is trained",
    "training_mode": "3: the network is in training mode whenever it is trained",
    "optimizer": "4: AdamW, learning rate 1e-4 and weight decay 1e-2, for all parameters",
    "steps": "4: one optimizer step per batch",
    "schedule": "4: a cosine from 1e-4 to 0 over the run, updated once per epoch",
    "loss": "4: minimizing the cross-entropy loss",
    "restart": "5: a restarted run trains exactly as the uninterrupted one",
    "validation": "6: the mean cross-entropy and accuracy on all validation samples, reported",
    "delivery": "6: the weights of the epoch with the highest reported accuracy, earliest on a tie",
}
# The parts a training run decides: equal to the reference's, state by state, they are
# the reference's in that run.
TRAINING_PARTS = (
    "training_samples",
    "training_labels",
    "batches",
    "training_inputs",
    "flips",
    "model",
    "parameters_trained",
    "training_mode",
    "optimizer",
    "steps",
    "schedule",
    "loss",
    "restart",
)


def family_parts(family: str) -> tuple[str, ...]:
    return tuple(part for part in PARTS if part != "flips" or family == "ham10000_pipeline")


def _finding(part: str, status: str, check: str, evidence: str, covers: str = "") -> dict:
    return {"part": part, "status": status, "check": check, "evidence": evidence, "covers": covers}


def epochs_of(report: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Every epoch a run trained, in order, from the monitor's rows, the pipeline's
    entries and the hashes; None if the three do not line up."""
    rows = report.get("parameter_group_epochs") or []
    entries = (report.get("pipeline") or {}).get("epochs")
    states = (report.get("workspace") or {}).get("epoch_state_sha256") or []
    if entries is None or not (len(rows) == len(entries) == len(states)):
        return None
    found = []
    for row, entry, state in zip(rows, entries, states, strict=True):
        rate = row.get("learning_rate") or {}
        found.append(
            {
                "epoch": entry.get("epoch"),
                "attempt": entry.get("attempt"),
                "label": f"epoch {entry.get('epoch')} (attempt {entry.get('attempt')})",
                "state": state,
                "steps": row.get("optimizer_steps"),
                "forwards": row.get("training_forwards"),
                "eval_forwards": row.get("eval_mode_training_forwards"),
                "rate_first": rate.get("first"),
                "rate_last": rate.get("last"),
                "rate_changes": rate.get("changes"),
                "reported": row.get("project_metrics") or {},
                "evaluation": entry.get("evaluation"),
                "entry": entry,
                "row": row,
            }
        )
    return found


def _training_size(report: dict[str, Any], family: str) -> int | None:
    reads = (report.get("pipeline") or {}).get("training_reads") or {}
    groups = TRAINING_GROUPS[family]
    if groups is None:
        return reads.get("candidates")
    by_group = reads.get("candidates_by_group")
    return None if by_group is None else sum(by_group.get(group, 0) for group in groups)


def _measured_alike(accuracy: Any, loss: Any, measured_accuracy: Any, measured_loss: Any) -> bool:
    """A reported accuracy and loss within VALIDATION_TOLERANCE of the harness's; the loss
    is compared where the harness measured one."""
    try:
        near = abs(float(accuracy) - float(measured_accuracy)) <= VALIDATION_TOLERANCE["accuracy"]
        if measured_loss is None:
            return near and loss is not None
        return near and math.isclose(
            float(loss), float(measured_loss), rel_tol=VALIDATION_TOLERANCE["loss"]
        )
    except (TypeError, ValueError):
        return False


def _close(a: Any, b: Any, rel: float = 1e-9) -> bool:
    try:
        return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=1e-15)
    except (TypeError, ValueError):
        return False


def record_checks(run: dict[str, Any], reference: dict[str, Any], family: str) -> list[dict]:
    """What the candidate's run and the reference's say, at no extra training cost."""
    check = "records"
    if run.get("status") != "completed":
        return [
            _finding("runs", "violated", check, f"the candidate's run ended {run.get('status')}")
        ]
    found = [_finding("runs", "supported", check, "the candidate's run completed", "its run")]
    epochs = epochs_of(run)
    if epochs is None:
        return found
    trajectory = compare_trajectories(run, reference)
    if trajectory["outcome"] == "identical_full_run":
        covers = (
            f"seed {run['workspace'].get('seed')}, {run['workspace'].get('epochs')} epochs, "
            f"stopped after epoch {run['workspace'].get('interrupt_after_epochs')}: "
            "the reference's training state by state"
        )
        found += [
            _finding(
                part,
                "supported",
                "reference_trajectory",
                "every hashed state is the reference's",
                covers,
            )
            for part in TRAINING_PARTS
        ]
    workspace, theirs = run.get("workspace") or {}, reference.get("workspace") or {}
    if run.get("head_output_units") not in (None, STATED["head_outputs"]):
        found.append(
            _finding("model", "violated", check, f"the head has {run['head_output_units']} outputs")
        )
    elif workspace.get("initial_state_sha256") and workspace.get(
        "initial_state_sha256"
    ) == theirs.get("initial_state_sha256"):
        found.append(
            _finding(
                "model",
                "supported",
                check,
                "it starts from the reference's weights",
                "the initial weights",
            )
        )
    frozen = sorted(
        {
            name
            for key in ("trainability", "final_trainability")
            for group in (run.get(key) or {}).values()
            for name in group.get("frozen_names", [])
        }
    )
    audits = [run.get(key) or {} for key in ("optimizer_audit", "final_optimizer_audit")]
    missing = sorted(
        {name for audit in audits for name in audit.get("missing_trainable_names", [])}
    )
    if frozen:
        found.append(
            _finding("parameters_trained", "violated", check, f"frozen: {', '.join(frozen[:3])}")
        )
    if missing or any(
        audit.get("duplicate_parameter_occurrences") or audit.get("foreign_parameter_tensors")
        for audit in audits
    ):
        found.append(
            _finding(
                "optimizer",
                "violated",
                check,
                f"the optimizer does not hold every parameter once ({', '.join(missing[:3])})",
            )
        )
    gradients = all(
        (epoch["row"].get(group) or {}).get("min_tensors_with_gradient")
        == (epoch["row"].get(group) or {}).get("parameter_tensors")
        for epoch in epochs
        for group in ("head", "backbone")
    )
    if not frozen and not missing and gradients and epochs:
        found.append(
            _finding(
                "parameters_trained",
                "supported",
                check,
                "every tensor is held and had a gradient in every epoch",
                "every epoch of its run",
            )
        )
    if run.get("optimizer") not in (None, STATED["optimizer"], "Adam"):
        found.append(
            _finding("optimizer", "violated", check, f"the optimizer is {run['optimizer']}")
        )
    if (
        epochs
        and epochs[0]["rate_first"] is not None
        and not _close(epochs[0]["rate_first"], STATED["learning_rate"], 1e-6)
    ):
        found.append(
            _finding(
                "optimizer",
                "violated",
                check,
                f"the first learning rate is {epochs[0]['rate_first']:g}",
            )
        )
    evaluated = [epoch["label"] for epoch in epochs if epoch["eval_forwards"]]
    if evaluated:
        found.append(
            _finding(
                "training_mode",
                "violated",
                check,
                f"steps follow forwards in evaluation mode in {', '.join(evaluated)}",
            )
        )
    size = _training_size(run, family)
    if size:
        batches = math.ceil(size / STATED["batch_size"])
        wrong = [epoch for epoch in epochs if epoch["steps"] != batches]
        if wrong:
            found.append(
                _finding(
                    "steps",
                    "violated",
                    check,
                    f"{wrong[0]['label']} takes {wrong[0]['steps']} steps where batches of "
                    f"{STATED['batch_size']} make {batches}",
                )
            )
        elif all(epoch["forwards"] == batches for epoch in epochs):
            found.append(
                _finding(
                    "steps",
                    "supported",
                    check,
                    f"every epoch takes {batches} steps on as many training forwards",
                    "every epoch of its run",
                )
            )
    found += _paired_reads(run, epochs, family, size)
    found += _inputs(run, reference)
    found += _delivered(run, epochs)
    return found


def _paired_reads(run: dict, epochs: list[dict], family: str, size: int | None) -> list[dict]:
    """Reads and loss targets, where every target pairs with the sample read for it."""
    check, found, paired = "records", [], 0
    groups = TRAINING_GROUPS[family]
    for epoch in epochs:
        entry = epoch["entry"]
        matching = entry.get("loss_targets_matching_read_labels")
        if matching is None:
            continue
        paired += 1
        outside = {
            g: n
            for g, n in (entry.get("reads_by_group") or {}).items()
            if groups and g not in groups
        }
        problems = []
        if outside:
            problems.append(
                f"trains on {sum(outside.values())} samples of groups {', '.join(sorted(outside))}"
            )
        if entry.get("distinct") != entry.get("reads"):
            problems.append(
                f"trains on {entry['reads'] - entry['distinct']} samples more than once"
            )
        if size is not None and entry.get("distinct") != size and not outside:
            problems.append(f"trains on {entry.get('distinct')} of the {size} samples")
        if problems:
            found.append(
                _finding(
                    "training_samples",
                    "violated",
                    check,
                    f"{epoch['label']} " + "; ".join(problems),
                )
            )
        if matching < entry.get("loss_targets", 0):
            found.append(
                _finding(
                    "training_labels",
                    "violated",
                    check,
                    f"{epoch['label']}: {entry['loss_targets'] - matching} of {entry['loss_targets']} targets are not the labels of the samples read",
                )
            )
    if paired and paired == len(epochs) and not found:
        covers = "every epoch of its run, each loss target paired with the sample read for it"
        found += [
            _finding(
                "training_samples",
                "supported",
                check,
                "every sample it should, once, and no other",
                covers,
            ),
            _finding(
                "training_labels",
                "supported",
                check,
                "every target is the label of its sample",
                covers,
            ),
        ]
    return found


def _inputs(run: dict, reference: dict) -> list[dict]:
    """Inputs whose channel statistics lie far from the reference's: a clear violation."""
    found = []
    for split, part in (("train", "training_inputs"), ("id_eval", "validation_inputs")):
        ours = (run.get("input_statistics") or {}).get(split)
        theirs = (reference.get("input_statistics") or {}).get(split)
        if not ours or not theirs:
            continue
        for channel, (mean, std, mean_ref, std_ref) in enumerate(
            zip(
                ours["channel_mean"],
                ours["channel_std"],
                theirs["channel_mean"],
                theirs["channel_std"],
                strict=True,
            )
        ):
            if std_ref > 0 and (not 2 / 3 <= std / std_ref <= 3 / 2 or abs(mean - mean_ref) > 0.5):
                found.append(
                    _finding(
                        part,
                        "violated",
                        "records",
                        f"channel {channel} has mean {mean:.3g} and deviation {std:.3g}; the reference's inputs have {mean_ref:.3g} and {std_ref:.3g}",
                    )
                )
                break
    return found


def _best(epochs: list[dict]) -> dict | None:
    """The epoch with the highest reported accuracy, the earliest on a tie."""
    scored = [
        epoch
        for epoch in epochs
        if isinstance(epoch["reported"].get("validation_accuracy"), (int, float))
    ]
    if len(scored) != len(epochs) or not epochs:
        return None
    return max(scored, key=lambda epoch: epoch["reported"]["validation_accuracy"])


def _delivered_epoch(run: dict, epochs: list[dict]) -> dict | None:
    same_as = ((run.get("pipeline") or {}).get("delivery") or {}).get("same_as") or []
    return next((epoch for epoch in epochs if epoch["label"] in same_as), None)


def _delivered(run: dict, epochs: list[dict]) -> list[dict]:
    """The delivered epoch's reported accuracy against the harness's, and the choice."""
    check, found = "records", []
    if (run.get("pipeline") or {}).get("delivery") is None:
        return [_finding("delivery", "violated", check, "the run delivered no weights")]
    delivered, best = _delivered_epoch(run, epochs), _best(epochs)
    if delivered is None:
        return [_finding("delivery", "violated", check, "the delivered weights are no epoch's")]
    final = run.get("final_metrics") or {}
    harness, harness_loss = final.get("id_validation_accuracy"), final.get("id_validation_loss")
    reported = delivered["reported"].get("validation_accuracy")
    reported_loss = delivered["reported"].get("validation_loss")
    if harness is not None and not _measured_alike(reported, reported_loss, harness, harness_loss):
        found.append(
            _finding(
                "validation",
                "violated",
                check,
                f"{delivered['label']} reports {reported} and {reported_loss}; its weights score "
                f"{harness:.4g} and {harness_loss}",
            )
        )
    if best is None:
        found.append(
            _finding("validation", "violated", check, "an epoch reports no validation accuracy")
        )
    elif best["label"] != delivered["label"] and best["state"] != delivered["state"]:
        found.append(
            _finding(
                "delivery",
                "violated",
                check,
                f"it delivers {delivered['label']}, but {best['label']} reports the highest accuracy",
            )
        )
    return found


def validation_check(run: dict, name: str) -> list[dict]:
    """Every epoch's reported loss and accuracy against the harness's own."""
    epochs = epochs_of(run)
    if not epochs or any(epoch["evaluation"] is None for epoch in epochs):
        return []
    for epoch in epochs:
        reported, measured = epoch["reported"], epoch["evaluation"]
        accuracy, loss = reported.get("validation_accuracy"), reported.get("validation_loss")
        if accuracy is None or loss is None:
            return [
                _finding(
                    "validation",
                    "violated",
                    name,
                    f"{epoch['label']} reports no validation loss or accuracy",
                )
            ]
        if not _measured_alike(
            accuracy, loss, measured["id_validation_accuracy"], measured["id_validation_loss"]
        ):
            return [
                _finding(
                    "validation",
                    "violated",
                    name,
                    f"{epoch['label']} reports {accuracy:.4g} and {loss:.4g}; its weights score {measured['id_validation_accuracy']:.4g} and {measured['id_validation_loss']:.4g}",
                )
            ]
    covers = f"every epoch of the {name} run"
    return [
        _finding(
            "validation", "supported", name, "every epoch reports what its weights score", covers
        ),
        _finding(
            "validation_inputs",
            "supported",
            name,
            "every epoch's reported loss and accuracy are those of inputs prepared as stated",
            covers,
        ),
    ]


def restart_check(
    restarted: dict, uninterrupted: dict, repeat: dict | None = None
) -> list[dict] | None:
    """The restarted run against the uninterrupted one; None if the determinism control
    is needed (only the weights differ) and no repeat is given."""
    name = "self_restart"
    ours, theirs = epochs_of(restarted), epochs_of(uninterrupted)
    if ours is None or theirs is None:
        return []
    sequence = [epoch["epoch"] for epoch in ours]
    expected = [epoch["epoch"] for epoch in theirs]
    if sequence != expected:
        return [
            _finding(
                "restart",
                "violated",
                name,
                f"the restarted run trains epochs {sequence}, the uninterrupted {expected}",
            )
        ]
    for mine, yours in zip(ours, theirs, strict=True):
        for end in ("rate_first", "rate_last"):
            if (mine[end], yours[end]) != (None, None) and not _close(mine[end], yours[end]):
                return [
                    _finding(
                        "restart",
                        "violated",
                        name,
                        f"epoch {mine['epoch']} trains with learning rate {mine[end]:.4g} after the restart, {yours[end]:.4g} without",
                    )
                ]
    if [epoch["state"] for epoch in ours] == [epoch["state"] for epoch in theirs]:
        seed, stop = (
            restarted["workspace"].get("seed"),
            restarted["workspace"].get("interrupt_after_epochs"),
        )
        return [
            _finding(
                "restart",
                "supported",
                name,
                "every epoch ends with the uninterrupted run's weights",
                f"seed {seed}, stopped after epoch {stop}",
            )
        ]
    if repeat is None:
        return None
    if [epoch["state"] for epoch in epochs_of(repeat) or []] == [
        epoch["state"] for epoch in theirs
    ]:
        first = next(
            mine["epoch"]
            for mine, yours in zip(ours, theirs, strict=True)
            if mine["state"] != yours["state"]
        )
        return [
            _finding(
                "restart",
                "violated",
                name,
                f"epoch {first} ends with other weights after the restart; the uninterrupted run repeats",
            )
        ]
    return [
        _finding(
            "restart",
            "untested",
            name,
            "the uninterrupted run does not repeat, so a difference after the restart shows nothing",
        )
    ]


def schedule_check(uninterrupted: dict) -> list[dict]:
    """An uninterrupted run's learning rate: one per epoch, on a cosine over the run."""
    epochs, total = epochs_of(uninterrupted), (uninterrupted.get("workspace") or {}).get("epochs")
    if not epochs or not total:
        return []
    for epoch in epochs:
        wanted = (
            STATED["learning_rate"] * (1 + math.cos(math.pi * (epoch["epoch"] - 1) / total)) / 2
        )
        if epoch["rate_changes"] or not _close(epoch["rate_first"], wanted, 1e-6):
            return [
                _finding(
                    "schedule",
                    "violated",
                    "self_restart",
                    f"epoch {epoch['epoch']} trains with {epoch['rate_first']:.4g} ({epoch['rate_changes']} changes); a cosine gives {wanted:.4g}",
                )
            ]
    return [
        _finding(
            "schedule",
            "supported",
            "self_restart",
            "one learning rate per epoch, on a cosine over the run",
            f"{total} epochs, uninterrupted",
        )
    ]


def delivery_case(run: dict, name: str) -> tuple[str | None, list[dict]]:
    """Which direction of the delivery rule a restarted run tried, and whether it held:
    kept (a best before the restart no later epoch beats) or taken (a new best after)."""
    epochs = epochs_of(run)
    best = _best(epochs or [])
    if not epochs or best is None:
        return None, []
    delivered = _delivered_epoch(run, epochs)
    if delivered is None or (
        best["label"] != delivered["label"] and best["state"] != delivered["state"]
    ):
        what = delivered["label"] if delivered else "weights no epoch ended with"
        return None, [
            _finding(
                "delivery",
                "violated",
                name,
                f"it delivers {what}; {best['label']} reports the highest accuracy",
            )
        ]
    before = [e["reported"]["validation_accuracy"] for e in epochs if e["attempt"] == 1]
    after = [e["reported"]["validation_accuracy"] for e in epochs if e["attempt"] != 1]
    if not before or not after:
        return None, []
    return ("kept" if max(before) >= max(after) else "taken"), []


class Probes:
    """The probe runs of v2, through train(sources, family, seed, tag=..., **settings), which
    returns a run's outcome as candidates.Runner does; probe_train trains on the probes'
    data. Every outcome's report is read once."""

    def __init__(
        self, train: Callable[..., dict], probe_train: Callable[..., dict], settings: dict
    ) -> None:
        self.train, self.probe_train, self.settings = train, probe_train, settings

    @staticmethod
    def _cost(outcome: dict, report: dict | None) -> dict[str, Any]:
        steps = sum(
            row.get("optimizer_steps") or 0
            for row in (report or {}).get("parameter_group_epochs", [])
        )
        return {"runs": 1, "steps": steps, "seconds": outcome.get("seconds", 0.0)}

    def run(
        self, kind: str, sources: dict, family: str, seed: int, epochs: int, tag: str = ""
    ) -> tuple[dict | None, dict]:
        if kind == "uninterrupted":
            outcome = self.train(
                sources, family, seed, tag=tag, epochs=epochs, evaluate_epochs=True
            )
        else:
            delivery = self.settings["delivery"]
            outcome = self.probe_train(
                sources,
                family,
                seed,
                epochs=delivery["epochs"],
                interrupt_after_epochs=delivery["interrupt_after_epochs"],
                evaluate_epochs=True,
                max_eval_samples=delivery["max_eval_samples"],
            )
        report = (
            json.loads(Path(outcome["report"]).read_text(encoding="utf-8"))
            if outcome.get("report")
            else None
        )
        return report, self._cost(outcome, report)


def _status(findings: list[dict], part: str) -> str:
    statuses = {finding["status"] for finding in findings if finding["part"] == part}
    if "violated" in statuses:
        return "violated"
    return "supported" if "supported" in statuses else "untested"


def verify(
    sources: dict[str, str],
    family: str,
    seed: int,
    run: dict | None,
    reference: dict,
    probes: Probes,
    *,
    order: str = "gap",
) -> dict[str, Any]:
    """v2's decision on a candidate, every part's status and findings, and the checks run."""
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    parts = family_parts(family)
    findings = (
        record_checks(run, reference, family)
        if run
        else [_finding("runs", "violated", "records", "the candidate has no run")]
    )
    checks = [{"name": "records", "cost": {"runs": 0, "steps": 0, "seconds": 0.0}}]
    cases: dict[str, int] = {}  # each direction tried, with the probe seed that tried it
    library = [
        "self_restart",
        *(f"delivery_{probe}" for probe in probes.settings["delivery"]["seeds"]),
    ]
    done: set[str] = set()

    def untested() -> set[str]:
        return {part for part in parts if _status(findings, part) == "untested"}

    def settles(check: str) -> set[str]:
        open_parts = untested()
        if check == "self_restart":
            return open_parts & {"restart", "schedule", "validation", "validation_inputs"}
        return open_parts & {"delivery", "validation", "validation_inputs"}

    while run and not any(_status(findings, part) == "violated" for part in parts):
        candidates = [check for check in library if check not in done and settles(check)]
        if not candidates:
            break
        if order == "gap":
            check = max(candidates, key=lambda name: len(settles(name)))
        else:
            check = candidates[0]
        done.add(check)
        epochs = (run.get("workspace") or {}).get("epochs")
        if check == "self_restart":
            report, cost = probes.run("uninterrupted", sources, family, seed, epochs)
            new = _probe_ran(report, check)
            if report is not None and report.get("status") == "completed":
                result = restart_check(run, report)
                if result is None:
                    repeat, extra = probes.run(
                        "uninterrupted", sources, family, seed, epochs, tag="repeat"
                    )
                    cost = {key: cost[key] + extra[key] for key in cost}
                    new += _probe_ran(repeat, "determinism_control")
                    result = (
                        restart_check(run, report, repeat)
                        if repeat and repeat.get("status") == "completed"
                        else []
                    )
                new += result + schedule_check(report) + validation_check(report, check)
                # An uninterrupted run tries the rule without a restart: best, not last.
                best, delivered = (
                    _best(epochs_of(report) or []),
                    _delivered_epoch(report, epochs_of(report) or []),
                )
                if (
                    best
                    and delivered
                    and best["label"] != delivered["label"]
                    and best["state"] != delivered["state"]
                ):
                    new.append(
                        _finding(
                            "delivery",
                            "violated",
                            check,
                            f"it delivers {delivered['label']}; {best['label']} reports the highest accuracy",
                        )
                    )
        else:
            probe_seed = int(check.removeprefix("delivery_"))
            report, cost = probes.run("delivery", sources, family, probe_seed, epochs)
            new = _probe_ran(report, check)
            if report is not None and report.get("status") == "completed":
                new += validation_check(report, check)
                case, violated = delivery_case(report, check)
                new += violated
                if case and not violated and _status(new, "validation") != "violated":
                    cases.setdefault(case, probe_seed)
                    if (
                        set(cases) == {"kept", "taken"}
                        and _status(findings, "delivery") != "supported"
                    ):
                        new.append(
                            _finding(
                                "delivery",
                                "supported",
                                "delivery_probes",
                                "a best before the restart was kept and a new best after it taken",
                                f"kept on probe seed {cases['kept']}, taken on probe seed {cases['taken']}",
                            )
                        )
        checks.append({"name": check, "cost": cost, "findings": new})
        findings += new
    statuses = {part: _status(findings, part) for part in parts}
    decision = (
        "reject"
        if "violated" in statuses.values()
        else "accept"
        if all(status == "supported" for status in statuses.values())
        else "review"
    )
    total = Counter()
    for check in checks:
        total.update(check["cost"])
    return {
        "decision": decision,
        "order": order,
        "parts": {
            part: {"status": statuses[part], "findings": [f for f in findings if f["part"] == part]}
            for part in parts
        },
        "delivery_cases": cases,
        "checks": [check["name"] for check in checks],
        "cost": {
            "runs": total["runs"],
            "steps": total["steps"],
            "seconds": round(total["seconds"], 1),
        },
    }


def _probe_ran(report: dict | None, check: str) -> list[dict]:
    if report is None or report.get("status") != "completed":
        status = "no report" if report is None else report.get("status")
        return [_finding("runs", "violated", check, f"the {check} run ended {status}")]
    return []


def load_settings(path: Path = PROBES_FILE) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RunSleuth v2 on the candidates of a split.")
    commands = parser.add_subparsers(dest="command", required=True)
    verified = commands.add_parser("verify", help="Decide every candidate of a split.")
    verified.add_argument("--split", required=True, choices=("development", "test"))
    verified.add_argument("--order", choices=ORDERS, default="gap")
    verified.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # The candidates' folders and runs; candidates holds no label until label is run,
    # and v2 reads none.
    from runsleuth import candidates as candidate_set
    from runsleuth.challenge_set import remove_evidence

    settings = load_settings()
    root = candidate_set.ROOT / args.split
    runs = json.loads((root / "runs.json").read_text(encoding="utf-8"))
    data = {
        family: (source, Path(config)) for family, (source, config) in candidate_set.DATA.items()
    }
    probe_data = {
        family: (candidate_set.DATA[family][0], Path(config))
        for family, config in settings["delivery"]["data"].items()
    }
    probes = Probes(
        candidate_set.Runner(root / "v2_runs", data=data, device=args.device).train,
        candidate_set.Runner(root / "v2_probes", data=probe_data, device=args.device).train,
        settings,
    )

    def read(outcome: dict | None) -> dict | None:
        if outcome is None:
            return None
        if outcome.get("report"):
            return json.loads(Path(outcome["report"]).read_text(encoding="utf-8"))
        return {key: outcome.get(key) for key in ("status", "problems", "error")}

    decided = {}
    for candidate in candidate_set.candidates(args.split):
        sources = candidate_set.candidate_sources(candidate)
        # An evidence-removed candidate is its parent's run without its transform's
        # records; v2's probes gather their own.
        outcome = runs["runs"].get(candidate.parent or candidate.id)
        if sources is None or outcome is None:
            continue
        run = read(outcome)
        if candidate.kind == "evidence_removed" and run is not None:
            run = remove_evidence(run, candidate.transform)
        reference = read(runs["references"][f"{candidate.family}/{candidate.seed}"])
        result = verify(
            sources,
            candidate.family,
            candidate.seed,
            run,
            reference,
            probes,
            order=args.order,
        )
        decided[candidate.id] = result
        print(
            f"{candidate.id}: {result['decision']} ({', '.join(result['checks'])}; {result['cost']['seconds']} s)",
            flush=True,
        )
    path = root / f"v2-{args.order}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "task": "verifier_v2",
                "order": args.order,
                "candidates": decided,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(dict(Counter(result["decision"] for result in decided.values())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
