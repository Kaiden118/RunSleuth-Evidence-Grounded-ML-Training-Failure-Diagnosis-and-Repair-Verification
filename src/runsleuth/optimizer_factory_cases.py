"""Versioned development fixtures for parameter-capture order, not a held-out benchmark."""

from textwrap import dedent

SUITE_VERSION = "optimizer-factory-development-v1"

_IMPORTS = "import torch\nfrom copy import deepcopy\n\n"


def _source(body: str) -> str:
    return (
        _IMPORTS
        + "def make_optimizer(model, *, learning_rate, weight_decay):\n"
        + "\n".join("    " + line if line else "" for line in dedent(body).strip().splitlines())
        + "\n"
    )


_CASES = (
    (
        "direct_constructor_before_head",
        "patch_proposed",
        _source("""
        next_head = deepcopy(model.fc)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model.fc = next_head
        return optimizer
    """),
    ),
    (
        "import_alias_and_renamed_variables",
        "patch_proposed",
        dedent("""
        from copy import deepcopy as clone
        from torch.optim import AdamW as Optimizer

        def build_optimizer(network, *, learning_rate, weight_decay):
            classifier = clone(network.fc)
            updater = Optimizer(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
            network.fc = classifier
            return updater
    """).lstrip(),
    ),
    (
        "eager_parameter_list_before_head",
        "patch_proposed",
        _source("""
        next_head = deepcopy(model.fc)
        saved_parameters = list(model.parameters())
        model.fc = next_head
        optimizer = torch.optim.AdamW(saved_parameters, lr=learning_rate, weight_decay=weight_decay)
        return optimizer
    """),
    ),
    (
        "eager_parameter_group_before_head",
        "patch_proposed",
        _source("""
        next_head = deepcopy(model.fc)
        groups = [{"params": tuple(model.parameters())}]
        model.fc = next_head
        optimizer = torch.optim.AdamW(groups, lr=learning_rate, weight_decay=weight_decay)
        return optimizer
    """),
    ),
    (
        "healthy_head_before_constructor",
        "no_change",
        _source("""
        model.fc = deepcopy(model.fc)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        return optimizer
    """),
    ),
    (
        "healthy_materialization_after_head",
        "no_change",
        _source("""
        model.fc = deepcopy(model.fc)
        saved_parameters = list(model.parameters())
        optimizer = torch.optim.AdamW(saved_parameters, lr=learning_rate, weight_decay=weight_decay)
        return optimizer
    """),
    ),
    (
        "healthy_unconsumed_parameter_iterator",
        "no_change",
        _source("""
        parameters = model.parameters()
        model.fc = deepcopy(model.fc)
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
        return optimizer
    """),
    ),
    (
        "healthy_unconsumed_iterator_in_group",
        "no_change",
        _source("""
        groups = [{"params": model.parameters()}]
        model.fc = deepcopy(model.fc)
        optimizer = torch.optim.AdamW(groups, lr=learning_rate, weight_decay=weight_decay)
        return optimizer
    """),
    ),
    (
        "unsupported_helper_factory",
        "unsupported",
        _source("""
        model.fc = deepcopy(model.fc)
        optimizer = choose_optimizer(model.parameters(), learning_rate, weight_decay)
        return optimizer
    """),
    ),
    (
        "unsupported_conditional_replacement",
        "unsupported",
        _source("""
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        if learning_rate > 0:
            model.fc = deepcopy(model.fc)
        return optimizer
    """),
    ),
    (
        "unsupported_intervening_side_effect",
        "unsupported",
        _source("""
        next_head = deepcopy(model.fc)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        record_optimizer(optimizer)
        model.fc = next_head
        return optimizer
    """),
    ),
    (
        "unsupported_move_dependency",
        "unsupported",
        _source("""
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        next_head = deepcopy(model.fc)
        model.fc = next_head
        return optimizer
    """),
    ),
)


def factory_cases() -> list[dict]:
    """Ground truth is used for scoring only; analysis receives the source alone."""
    return [
        {"id": name, "expected_decision": expected, "source": source}
        for name, expected, source in _CASES
    ]


ORACLE_SOURCE = _source("""
    model.fc = deepcopy(model.fc)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer
""")
