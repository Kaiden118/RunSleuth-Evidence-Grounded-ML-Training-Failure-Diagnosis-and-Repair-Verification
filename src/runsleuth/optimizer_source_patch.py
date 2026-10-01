"""Locate one known stale-binding fixture without importing or executing it.

The returned change is a proposal for an isolated source copy.  It does not
modify the deliberate fault fixture or prove which branch executed at runtime.
Unknown function shapes are rejected rather than rewritten heuristically.
"""

import ast
from copy import deepcopy
from difflib import unified_diff

_EXPECTED = """
def make_probe_optimizer(model, *, variant, learning_rate, weight_decay):
    if variant not in _VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; expected one of {_VARIANTS}")
    _parameter_groups(model)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and greater than zero")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    new_head = deepcopy(model.fc)
    if variant == "clean":
        model.fc = new_head
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        model.fc = new_head
    return optimizer
"""


def _dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _target(tree: ast.Module) -> ast.FunctionDef:
    candidates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "make_probe_optimizer"
    ]
    if len(candidates) != 1:
        raise ValueError("Exactly one make_probe_optimizer definition is required")
    target = candidates[0]
    if not isinstance(target, ast.FunctionDef) or target not in tree.body:
        raise ValueError("The fixture must be a synchronous top-level function")
    if target.decorator_list:
        raise ValueError("Decorated optimizer constructors are not supported")

    normalized = deepcopy(target)
    if (
        normalized.body
        and isinstance(normalized.body[0], ast.Expr)
        and isinstance(normalized.body[0].value, ast.Constant)
        and isinstance(normalized.body[0].value.value, str)
    ):
        normalized.body.pop(0)
    # Annotations are descriptive; arguments, defaults and executable statements
    # must match the recorded controlled fixture exactly.
    normalized.returns = None
    for argument in normalized.args.posonlyargs + normalized.args.args + normalized.args.kwonlyargs:
        argument.annotation = None
    expected = ast.parse(_EXPECTED).body[0]
    if _dump(normalized) != _dump(expected):
        raise ValueError("Unsupported optimizer construction shape; proposal refused")
    return target


def _line_span(source_lines: list[str], statement: ast.stmt) -> tuple[int, int]:
    """Require statements on their own physical lines, excluding semicolons."""
    start, end = statement.lineno - 1, statement.end_lineno
    if end is None:
        raise ValueError("Source positions are unavailable")
    prefix = source_lines[start].encode("utf-8")[: statement.col_offset]
    suffix = source_lines[end - 1].encode("utf-8")[statement.end_col_offset :]
    if prefix.strip() or (suffix.strip() and not suffix.lstrip().startswith(b"#")):
        raise ValueError("Inline or semicolon-separated statements are unsupported")
    return start, end


def _physical_lines(text: str) -> list[str]:
    parts = text.split("\n")
    return [part + "\n" for part in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def _diff(before: str, after: str) -> str:
    # difflib omits the conventional marker for an unterminated final line.
    # Add it so that a proposal remains a valid patch for such source files.
    pieces = []
    for line in unified_diff(
        _physical_lines(before),
        _physical_lines(after),
        fromfile="a/optimizer_probe.py",
        tofile="b/optimizer_probe.py",
    ):
        pieces.append(line)
        if not line.endswith("\n"):
            pieces.append("\n\\ No newline at end of file\n")
    return "".join(pieces)


def propose_optimizer_order_patch(source_text: str) -> dict:
    """Return exact source sites and a two-statement reorder for the known bug.

    All other source text is preserved.  This is static source localization,
    not a claim that the stale branch ran or that performance would improve.
    """
    if not isinstance(source_text, str) or not source_text:
        raise ValueError("Nonempty Python source text is required")
    if "\r" in source_text.replace("\r\n", ""):
        raise ValueError("Only LF and CRLF source line endings are supported")
    try:
        tree = ast.parse(source_text)
    except SyntaxError as error:
        raise ValueError("Optimizer source is not valid Python") from error
    target = _target(tree)
    branch = target.body[-2]
    constructor, replacement = branch.orelse

    # Split only at LF: Unicode separators inside comments/docstrings are not
    # physical Python source lines.  Keep each original LF or CRLF intact.
    lines = _physical_lines(source_text)
    first, first_end = _line_span(lines, constructor)
    second, second_end = _line_span(lines, replacement)
    if first_end > second:
        raise ValueError("Optimizer statements must occupy separate source lines")
    if lines[first][: constructor.col_offset] != lines[second][: replacement.col_offset]:
        raise ValueError("Optimizer statements must have identical indentation")
    proposed = "".join(
        lines[:first]
        + lines[second:second_end]
        + lines[first_end:second]
        + lines[first:first_end]
        + lines[second_end:]
    )
    expected_tree = deepcopy(tree)
    expected_branch = _target(expected_tree).body[-2]
    expected_branch.orelse.reverse()
    try:
        proposed_tree = ast.parse(proposed)
    except SyntaxError as error:
        raise ValueError("The isolated statement reorder is not valid Python") from error
    if _dump(proposed_tree) != _dump(expected_tree):
        raise ValueError("Reordering changed more than the two permitted statements")

    def site(node: ast.stmt) -> dict:
        return {
            "line": node.lineno,
            "end_line": node.end_lineno,
            "code": ast.get_source_segment(source_text, node),
        }

    return {
        "function": "make_probe_optimizer",
        "branch_condition": 'variant != "clean"',
        "constructor": site(constructor),
        "head_replacement": site(replacement),
        "unified_diff": _diff(source_text, proposed),
        "proposed_source": proposed,
    }
