"""Repair tasks for agents on training projects that own their pipeline.

A project is a folder of modules (pipeline_templates/<family>) that pipeline_workspace
runs. Unlike the scripts of agent_tasks, it filters its own training data, resumes
from its own checkpoints and chooses the weights it delivers. A task is one place
where such a project can go wrong and still train. Every task has two faults, written
as exact edits of the healthy project: one is development material, used to build and
tune verifiers, and the other is held out to test them. A variant is a correct project
written differently from the healthy one, where the statement allows it; it too
belongs to one split.

The statement is the public specification of a project: pipeline_contract tests it,
it never names a fault, and it is the same for every version of a family.
materialize() writes a project with TASK.md, the only files an agent is given.

    python -m runsleuth.pipeline_tasks list
    python -m runsleuth.pipeline_tasks show positions_index_base
    python -m runsleuth.pipeline_tasks write labels_not_filtered projects/labels
"""

import argparse
import difflib
from dataclasses import dataclass
from pathlib import Path

TEMPLATES = Path(__file__).with_name("pipeline_templates")
STATEMENT_NAME = "TASK.md"
HEALTHY = "healthy"
SPLITS = ("development", "test")
NO_FAULT = "no_known_fault"
CAMELYON_STATEMENT = """\
# Task

This project fine-tunes an ImageNet-pretrained ResNet18 to tell tumor from normal tissue
in 96x96 histopathology patches (Camelyon17). Its modules are `config.py`, `data.py` and
`train.py`.

A harness calls `main(run)` in `train.py`. `run` gives the project:

- `run.data.train`: the training candidates, patches from hospitals 0, 1, 3 and 4.
  `run.data.train[i]` is `(image, label)` and `run.data.train.image(i)` the image alone,
  an RGB tensor of shape (3, 96, 96) with values in [0, 1]. `run.data.train.labels` and
  `run.data.train.hospitals` hold every patch's label (1 for tumor) and hospital.
- `run.data.validation`: in-distribution validation patches, given the same way.
- `run.device`, `run.seed` and `run.epochs`: where to train, the run's seed, and how
  many epochs the run trains in all.
- `run.checkpoints`: `write(name, state)`, `read(name)` and `name in run.checkpoints`,
  the project's only storage.
- `run.track(model, optimizer)`, once both exist and before the first epoch;
  `with run.epoch(number):` around the training of every epoch, numbered from 1;
  `run.log(**metrics)` after every epoch; `run.deliver(state_dict)` once, at the end.

The harness may stop the run as it starts an epoch and call `main` again from the
start, as after a restart, with the same checkpoints. After the run it evaluates the
delivered weights itself, on the validation patches and on patches from hospital 1.
Samples are read in the main process (`num_workers=0`).

## What the project must do

1. Training data: every epoch trains on each patch of `run.data.train` from hospitals
   0, 3 and 4 exactly once, with its own label, and on no other patch; hospital 1 is
   for evaluation only. Batches hold 32 patches (the last of an epoch may hold fewer),
   in an order that is shuffled and depends on nothing but the run's seed and the epoch.
2. Inputs: subtract the ImageNet mean (0.485, 0.456, 0.406) and divide by the ImageNet
   standard deviation (0.229, 0.224, 0.225) per channel, in training and validation
   alike, and change nothing else.
3. Model: the pretrained ResNet18 with its final layer replaced by a new layer of two
   outputs. Every parameter is trained, and the network is in training mode whenever
   it is trained.
4. Optimizer: AdamW with learning rate 1e-4 and weight decay 1e-2 for all parameters,
   one step per batch, minimizing the cross-entropy loss. The learning rate follows a
   cosine from 1e-4 at the start of the run to 0 at its end, updated once per epoch.
5. Restarts: after every epoch the project checkpoints what it needs, so that a run
   stopped as it starts an epoch and restarted trains exactly as the uninterrupted run
   would have: the same epochs, batches, weights, optimizer state and learning rates.
6. Validation and delivery: after every epoch the project measures its loss and
   accuracy on `run.data.validation` and reports them with `run.log`. At the end it
   delivers the weights of the epoch with the highest validation accuracy, the
   earliest such epoch on a tie.

## Your job

1. Decide whether the project does all of the above.
2. If it does not, fix it, changing only what is needed.
3. Say whether the project now does all of the above and what you changed.

The project may already be correct.
"""
TASKS = {
    "training_data_filter": "A dataset wrapper keeps the training hospitals' patches; "
    "which patches it hands to training, and with which labels.",
    "restart": "After a stop between two epochs the project resumes from its checkpoint; "
    "whether it restores everything the rest of the run depends on.",
}


@dataclass(frozen=True, slots=True)
class Version:
    """A version of the healthy project as edits, each (module, old text, new text).

    task is the task it belongs to, and split whether it is development or test
    material. correct says whether it meets the statement. signature is the failure
    signature its runs should be diagnosed with: None for a correct version and for a
    fault the signature library was not written for.
    """

    task: str
    split: str
    correct: bool
    summary: str
    edits: tuple[tuple[str, str, str], ...]
    signature: str | None = None


_IMPORT = "from torch.utils.data import DataLoader, Dataset\n"
_SHUFFLED = (
    "    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, "
    "generator=generator)\n"
)
_RESUMED = (
    "    if config.LAST_CHECKPOINT in run.checkpoints:\n"
    "        state = run.checkpoints.read(config.LAST_CHECKPOINT)\n"
    '        model.load_state_dict(state["model"])\n'
    '        optimizer.load_state_dict(state["optimizer"])\n'
    '        scheduler.load_state_dict(state["scheduler"])\n'
    '        start, best_accuracy = state["epoch"] + 1, state["best_accuracy"]\n'
)
_SAVED = (
    "        run.checkpoints.write(\n"
    "            config.LAST_CHECKPOINT,\n"
    "            {\n"
    '                "epoch": epoch,\n'
    '                "model": model.state_dict(),\n'
    '                "optimizer": optimizer.state_dict(),\n'
    '                "scheduler": scheduler.state_dict(),\n'
    '                "best_accuracy": best_accuracy,\n'
    "            },\n"
    "        )\n"
)
CAMELYON_VERSIONS = {
    "positions_index_base": Version(
        "training_data_filter",
        "development",
        False,
        "The wrapper reads the patches at its own positions instead of at the indices it "
        "kept, so training takes patches from hospital 1, leaves out as many of the last "
        "training hospital's, and pairs most patches with another patch's label.",
        (
            (
                "data",
                "self.patches.image(self.indices[position])",
                "self.patches.image(position)",
            ),
        ),
        "train_evaluation_group_overlap",
    ),
    "labels_not_filtered": Version(
        "training_data_filter",
        "test",
        False,
        "The wrapper keeps the training hospitals' patches but not their labels, so the "
        "right patches are trained on with the labels of other patches.",
        (
            (
                "data",
                "self.labels = [patches.labels[index] for index in self.indices]",
                "self.labels = list(patches.labels)",
            ),
        ),
    ),
    "subset_random_sampler": Version(
        "training_data_filter",
        "development",
        True,
        "The loader samples the kept indices from all the patches with a "
        "SubsetRandomSampler: the same patches and labels, in another order.",
        (
            ("data", _IMPORT, _IMPORT.replace("Dataset", "Dataset, SubsetRandomSampler")),
            (
                "data",
                _SHUFFLED,
                "    sampler = SubsetRandomSampler(dataset.indices, generator=generator)\n"
                "    return DataLoader(dataset.patches, batch_size=config.BATCH_SIZE, "
                "sampler=sampler)\n",
            ),
        ),
    ),
    "seeded_per_epoch": Version(
        "training_data_filter",
        "test",
        True,
        "Every epoch seeds torch's global generator and the loader shuffles with it: the "
        "same patches and labels, in another order than the healthy project's and "
        "subset_random_sampler's.",
        (
            (
                "data",
                "    generator = torch.Generator().manual_seed(1000 * seed + epoch)\n" + _SHUFFLED,
                "    torch.manual_seed(1000 * seed + epoch)\n"
                "    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)\n",
            ),
        ),
    ),
    "scheduler_not_restored": Version(
        "restart",
        "development",
        False,
        "A restart rebuilds the schedule instead of restoring it. The cosine is computed "
        "step by step from the current rate, so the epoch after the restart still has the "
        "right rate, and the later epochs follow a new cosine from it and decay too slowly.",
        (("train", '        scheduler.load_state_dict(state["scheduler"])\n', ""),),
    ),
    "optimizer_not_restored": Version(
        "restart",
        "test",
        False,
        "A restart rebuilds the optimizer instead of restoring it: AdamW's moments start "
        "again from zero and the learning rate from 1e-4.",
        (("train", '        optimizer.load_state_dict(state["optimizer"])\n', ""),),
    ),
    "checkpoint_in_parts": Version(
        "restart",
        "development",
        True,
        "The model, the optimizer, the schedule and the progress are four checkpoints; the "
        "progress is written last and resumed from only when it is there.",
        (
            (
                "train",
                _RESUMED,
                '    if "progress" in run.checkpoints:\n'
                '        model.load_state_dict(run.checkpoints.read("model"))\n'
                '        optimizer.load_state_dict(run.checkpoints.read("optimizer"))\n'
                '        scheduler.load_state_dict(run.checkpoints.read("scheduler"))\n'
                '        progress = run.checkpoints.read("progress")\n'
                '        start, best_accuracy = progress["epoch"] + 1, progress["best_accuracy"]\n',
            ),
            (
                "train",
                _SAVED,
                '        run.checkpoints.write("model", model.state_dict())\n'
                '        run.checkpoints.write("optimizer", optimizer.state_dict())\n'
                '        run.checkpoints.write("scheduler", scheduler.state_dict())\n'
                "        run.checkpoints.write(\n"
                '            "progress", {"epoch": epoch, "best_accuracy": best_accuracy}\n'
                "        )\n",
            ),
        ),
    ),
    "numbered_checkpoints": Version(
        "restart",
        "test",
        True,
        "Every epoch writes a checkpoint named after it, and a restart resumes from the "
        "latest one there is.",
        (
            (
                "train",
                "    if config.LAST_CHECKPOINT in run.checkpoints:\n"
                "        state = run.checkpoints.read(config.LAST_CHECKPOINT)\n",
                "    numbers = range(1, run.epochs + 1)\n"
                '    written = [number for number in numbers if f"epoch-{number}" in run.checkpoints]\n'
                "    if written:\n"
                '        state = run.checkpoints.read(f"epoch-{max(written)}")\n',
            ),
            ("train", "            config.LAST_CHECKPOINT,\n", '            f"epoch-{epoch}",\n'),
        ),
    ),
}


@dataclass(frozen=True, slots=True)
class Family:
    """A healthy project, its statement, and the versions made from it."""

    name: str
    template: Path
    statement: str
    versions: dict[str, Version]

    @property
    def names(self) -> tuple[str, ...]:
        return (HEALTHY, *self.versions)


FAMILIES = {
    family.name: family
    for family in (
        Family("camelyon_pipeline", TEMPLATES / "camelyon", CAMELYON_STATEMENT, CAMELYON_VERSIONS),
    )
}


def family(name: str) -> Family:
    if name not in FAMILIES:
        raise ValueError(f"Unknown project family {name!r}; choose from {', '.join(FAMILIES)}")
    return FAMILIES[name]


def version(name: str, family_name: str = "camelyon_pipeline") -> Version:
    versions = family(family_name).versions
    if name not in versions:
        choices = ", ".join(family(family_name).names)
        raise ValueError(f"Unknown version {name!r}; choose from {choices}")
    return versions[name]


def kind(name: str, family_name: str = "camelyon_pipeline") -> str:
    """healthy, variant (a correct version written differently) or fault."""
    if name == HEALTHY:
        return HEALTHY
    return "variant" if version(name, family_name).correct else "fault"


def expected_diagnosis(name: str, family_name: str = "camelyon_pipeline") -> str:
    """The signature a version's run should be diagnosed with, or no_known_fault."""
    if name == HEALTHY:
        return NO_FAULT
    return version(name, family_name).signature or NO_FAULT


def healthy_sources(family_name: str = "camelyon_pipeline") -> dict[str, str]:
    """The healthy project's modules by name."""
    template = family(family_name).template
    return {path.stem: path.read_text(encoding="utf-8") for path in sorted(template.glob("*.py"))}


def version_sources(name: str, family_name: str = "camelyon_pipeline") -> dict[str, str]:
    """A version's modules: the healthy project, with the version's edits applied."""
    sources = healthy_sources(family_name)
    if name == HEALTHY:
        return sources
    for module, old, new in version(name, family_name).edits:
        if sources[module].count(old) != 1:
            raise ValueError(f"{name}: every edit must match its module exactly once")
        sources[module] = sources[module].replace(old, new)
    return sources


def version_diff(name: str, family_name: str = "camelyon_pipeline") -> str:
    """Unified diff from the healthy project to a version, module by module."""
    healthy, changed = healthy_sources(family_name), version_sources(name, family_name)
    return "".join(
        "".join(
            difflib.unified_diff(
                healthy[module].splitlines(keepends=True),
                changed[module].splitlines(keepends=True),
                f"{HEALTHY}/{module}.py",
                f"{name}/{module}.py",
            )
        )
        for module in healthy
    )


def materialize(name: str, directory: Path, family_name: str = "camelyon_pipeline") -> Path:
    """Write a fresh project for a version and return it; the directory must not exist."""
    sources = version_sources(name, family_name)
    directory = Path(directory)
    directory.mkdir(parents=True)
    for module, source in sources.items():
        (directory / f"{module}.py").write_text(source, encoding="utf-8", newline="\n")
    (directory / STATEMENT_NAME).write_text(
        family(family_name).statement, encoding="utf-8", newline="\n"
    )
    return directory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair tasks on training projects.")
    parser.add_argument("--family", choices=tuple(FAMILIES), default="camelyon_pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="Name every version with its task, split and truth.")
    show = commands.add_parser("show", help="Print a version's diff from the healthy project.")
    show.add_argument("version", help="A version of the family; list names them.")
    write = commands.add_parser("write", help="Write a version's project to a new directory.")
    write.add_argument("version", help="A version of the family; list names them.")
    write.add_argument("directory", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    projects = family(args.family)
    if args.command == "list":
        print(f"{HEALTHY}: the healthy project, used as the reference")
        for name, item in projects.versions.items():
            truth = "correct" if item.correct else item.signature or "no signature"
            print(f"{name} [{item.task}, {item.split}, {truth}]: {item.summary}")
    elif args.command == "show":
        diff = version_diff(args.version, args.family)
        print(diff or "No difference: this is the healthy project.", end="")
    else:
        print(materialize(args.version, args.directory, args.family))


if __name__ == "__main__":
    main()
