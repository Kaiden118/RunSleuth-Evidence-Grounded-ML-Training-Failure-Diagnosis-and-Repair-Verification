"""New, versioned semantic challenge fixtures; separate from the development suite.

These are author-constructed cases, not an independently blinded benchmark.
Known labels are checked by optimizer membership and an actual CPU step; unknown
cases are deliberately not executable because a dependency or input is absent.
The runtime oracle does not consult the AST analyzer or any LLM prediction.
"""

import hashlib
import json
from textwrap import dedent
from types import SimpleNamespace

VERSION = "optimizer-semantic-challenge-v1"
ASSUMPTIONS = """Classify the optimizer returned by make_optimizer.
Assume ordinary PyTorch nn.Module, model.fc is a trainable classification head,
all model parameters require gradients, and full-model optimization is intended.
model.parameters() has standard lazy iterator semantics. deepcopy(model.fc)
creates new parameter objects with equal values. AdamW consumes its iterable.
learning_rate is positive and weight_decay is nonnegative. Standard torch and
copy imports behave normally; no hidden monkey-patching is assumed.
Definitions in the source are available; externally imported custom helpers and
unprovided model flags have unknown behavior/value. Do not guess those details.
stale: the returned optimizer still holds replaced head parameter objects and
omits the current head. current: it covers all current trainable parameters with
no obsolete head objects. inconclusive: the supplied source and assumptions do
not determine one of these findings. This is not a patchability classification.
"""


def _factory(body: str) -> str:
    return (
        "import torch\nfrom copy import deepcopy\n\ndef make_optimizer(model, *, learning_rate, weight_decay):\n"
        + "\n".join("    " + line if line else "" for line in dedent(body).strip().splitlines())
        + "\n"
    )


def challenge_cases() -> list[dict]:
    definitions = [
        (
            "stale",
            "tuple_alias",
            _factory("""
            replacement = deepcopy(model.fc)
            captured = tuple(model.parameters())
            selected = captured
            model.fc = replacement
            updater = torch.optim.AdamW(selected, lr=learning_rate, weight_decay=weight_decay)
            return updater
        """),
        ),
        (
            "current",
            "lazy_alias",
            _factory("""
            pending = model.parameters()
            selected = pending
            model.fc = deepcopy(model.fc)
            updater = torch.optim.AdamW(selected, lr=learning_rate, weight_decay=weight_decay)
            return updater
        """),
        ),
        (
            "inconclusive",
            "external_optimizer",
            dedent("""
            from copy import deepcopy
            from project_extension import make_update_rule

            def make_optimizer(model, *, learning_rate, weight_decay):
                model.fc = deepcopy(model.fc)
                updater = make_update_rule(model, learning_rate, weight_decay)
                return updater
        """).lstrip(),
        ),
        (
            "stale",
            "local_helper",
            dedent("""
            import torch
            from copy import deepcopy

            def collect(net, lr_value, decay_value):
                return torch.optim.AdamW(net.parameters(), lr=lr_value, weight_decay=decay_value)

            def make_optimizer(model, *, learning_rate, weight_decay):
                replacement = deepcopy(model.fc)
                updater = collect(model, learning_rate, weight_decay)
                model.fc = replacement
                return updater
        """).lstrip(),
        ),
        (
            "current",
            "local_helper",
            dedent("""
            import torch
            from copy import deepcopy

            def collect(net, lr_value, decay_value):
                return torch.optim.AdamW(net.parameters(), lr=lr_value, weight_decay=decay_value)

            def make_optimizer(model, *, learning_rate, weight_decay):
                replacement = deepcopy(model.fc)
                model.fc = replacement
                updater = collect(model, learning_rate, weight_decay)
                return updater
        """).lstrip(),
        ),
        (
            "inconclusive",
            "unknown_flag",
            _factory("""
            updater = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            if model.replace_head:
                model.fc = deepcopy(model.fc)
            return updater
        """),
        ),
        (
            "current",
            "eager_alias_after_replace",
            _factory("""
            model.fc = deepcopy(model.fc)
            captured = tuple(model.parameters())
            selected = list(captured)
            updater = torch.optim.AdamW(selected, lr=learning_rate, weight_decay=weight_decay)
            return updater
        """),
        ),
        (
            "stale",
            "nested_eager_capture",
            _factory("""
            replacement = deepcopy(model.fc)
            selected = list(tuple(model.parameters()))
            model.fc = replacement
            updater = torch.optim.AdamW(selected, lr=learning_rate, weight_decay=weight_decay)
            return updater
        """),
        ),
        (
            "inconclusive",
            "external_mutation",
            dedent("""
            import torch
            from project_extension import configure_classifier

            def make_optimizer(model, *, learning_rate, weight_decay):
                updater = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
                configure_classifier(model)
                return updater
        """).lstrip(),
        ),
        (
            "current",
            "direct_return_lazy_group",
            _factory("""
            group = [{"params": model.parameters()}]
            model.fc = deepcopy(model.fc)
            return torch.optim.AdamW(group, lr=learning_rate, weight_decay=weight_decay)
        """),
        ),
        (
            "stale",
            "constant_branch",
            _factory("""
            pending = model.parameters()
            updater = torch.optim.AdamW(pending, lr=learning_rate, weight_decay=weight_decay)
            if True:
                model.fc = deepcopy(model.fc)
            return updater
        """),
        ),
        (
            "inconclusive",
            "external_selection",
            dedent("""
            import torch
            from copy import deepcopy
            from project_extension import select_parameters

            def make_optimizer(model, *, learning_rate, weight_decay):
                selected = select_parameters(model)
                model.fc = deepcopy(model.fc)
                updater = torch.optim.AdamW(selected, lr=learning_rate, weight_decay=weight_decay)
                return updater
        """).lstrip(),
        ),
    ]
    return [
        {
            "id": f"case-{index:03d}",
            "expected": expected,
            "family": family,
            "source": source,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        }
        for index, (expected, family, source) in enumerate(definitions, 1)
    ]


def challenge_sha256() -> str:
    payload = {"version": VERSION, "assumptions": ASSUMPTIONS, "cases": challenge_cases()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def verify_known_case(case: dict, receipt: dict) -> None:
    """Execute only an exact built-in, known-label fixture, never an arbitrary input."""
    canonical = next((row for row in challenge_cases() if row == case), None)
    if canonical is None or case["expected"] == "inconclusive":
        raise ValueError("Only exact built-in, fully specified fixtures can be executed")
    import torch
    from torch import nn

    from runsleuth.optimizer_source_runtime import _probe_step

    # Imports are restricted even though the source must equal a built-in fixture.
    def trusted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name not in ("torch", "copy"):
            raise ValueError("Import outside the fixed oracle allowlist")
        if name == "torch":
            return torch
        import copy

        return copy

    namespace = {"__builtins__": {"__import__": trusted_import, "list": list, "tuple": tuple}}
    exec(compile(case["source"], "<built-in-semantic-fixture>", "exec"), namespace)
    factory = namespace["make_optimizer"]

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(nn.Linear(3, 5, device="cpu"), nn.Tanh())
            self.fc = nn.Linear(5, 2, device="cpu")

        def forward(self, inputs):
            return self.fc(self.backbone(inputs))

    def constructor(model, *, variant, learning_rate, weight_decay):
        return factory(model, learning_rate=learning_rate, weight_decay=weight_decay)

    receipt.update(status="running", optimizer_steps_recorded=0, step_accounting_complete=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        model = TinyModel()
        inputs = torch.tensor(
            [[0.2, -0.5, 1.0], [1.0, 0.3, -0.2], [-0.4, 0.7, 0.8], [0.1, 0.9, -0.3]], device="cpu"
        )
        targets = torch.tensor([0, 1, 0, 1], device="cpu")
        receipt["step_accounting_complete"] = False
        row = _probe_step(
            model,
            inputs,
            targets,
            constructor=constructor,
            variant="fixture",
            config=SimpleNamespace(learning_rate=0.001, weight_decay=0.0),
        )
        receipt.update(optimizer_steps_recorded=1, step_accounting_complete=True, evidence=row)
    audit = row["optimizer_audit"]
    stale = case["expected"] == "stale"
    checks = {
        "membership_matches_label": sorted(audit["missing_trainable_names"])
        == (["fc.bias", "fc.weight"] if stale else []),
        "foreign_count_matches_label": audit["foreign_parameter_tensors"] == (2 if stale else 0),
        "no_duplicate_members": audit["duplicate_parameter_occurrences"] == 0,
        "head_has_gradients": row["head"]["gradient_l2_norm"] > 0,
        "head_update_matches_label": row["head"]["parameter_update_l2_norm"] == 0
        if stale
        else row["head"]["parameter_update_l2_norm"] > 0,
        "backbone_updates": row["backbone"]["parameter_update_l2_norm"] > 0,
    }
    receipt.update(
        status="verified" if all(checks.values()) else "failed",
        checks=checks,
        label_verified=all(checks.values()),
        device="cpu",
        torch_version=str(torch.__version__),
    )
    if not all(checks.values()):
        raise ValueError("Independent CPU oracle disagrees with the fixed semantic label")
