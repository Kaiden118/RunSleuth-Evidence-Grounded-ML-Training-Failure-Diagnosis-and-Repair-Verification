from pathlib import Path

from runsleuth.source_inspection import inspect_training_source


def test_finds_calls_with_correct_locations_and_conditions(tmp_path: Path) -> None:
    source = "\n".join(
        [
            "def train_one_epoch():",
            "    optimizer.zero_grad(set_to_none=True)",
            "    loss.backward()",
            "    if optimizer_step_enabled:",
            "        optimizer.step()",
            "    # optimizer.step()",
            '    message = "optimizer.step()"',
        ]
    )
    source_path = tmp_path / "train.py"
    source_path.write_text(source, encoding="utf-8")

    report = inspect_training_source(source_path)

    assert report.source_path == str(source_path)
    assert [call.call for call in report.calls] == [
        "optimizer.zero_grad",
        "loss.backward",
        "optimizer.step",
    ]
    assert all(call.function == "train_one_epoch" for call in report.calls)
    assert [call.conditions for call in report.calls] == [
        (),
        (),
        ("optimizer_step_enabled",),
    ]

    step_call = report.calls[2]
    assert step_call.line == 5
    assert step_call.code == "optimizer.step()"


def test_tracks_nested_branches_and_resets_function_conditions(tmp_path: Path) -> None:
    source = "\n".join(
        [
            "if enable_helpers:",
            "    async def helper():",
            "        optimizer.step()",
            "",
            "def train_one_epoch():",
            "    if ready:",
            "        if optimizer_step_enabled:",
            "            optimizer.step()",
            "        else:",
            "            optimizer.zero_grad()",
            "    loss.backward()",
        ]
    )
    source_path = tmp_path / "train.py"
    source_path.write_text(source, encoding="utf-8")

    report = inspect_training_source(source_path)

    assert [(call.function, call.call, call.conditions) for call in report.calls] == [
        ("helper", "optimizer.step", ()),
        (
            "train_one_epoch",
            "optimizer.step",
            ("ready", "optimizer_step_enabled"),
        ),
        (
            "train_one_epoch",
            "optimizer.zero_grad",
            ("ready", "not (optimizer_step_enabled)"),
        ),
        ("train_one_epoch", "loss.backward", ()),
    ]


def test_inspection_does_not_execute_or_modify_source(tmp_path: Path) -> None:
    source_path = tmp_path / "train.py"
    source_path.write_text(
        "raise RuntimeError('source was executed')\noptimizer.step()\n",
        encoding="utf-8",
    )
    original_bytes = source_path.read_bytes()

    report = inspect_training_source(source_path)

    assert len(report.calls) == 1
    assert report.calls[0].call == "optimizer.step"
    assert source_path.read_bytes() == original_bytes
