"""Conservative data-flow checks for small, straight-line AdamW factories.

This development adapter distinguishes materialized parameter collections from
unconsumed model.parameters() iterators. Unknown Python is refused. A proposal
moves only the head replacement, preserving the other source bytes.
"""

import ast
from copy import deepcopy
from dataclasses import dataclass

from runsleuth.optimizer_source_patch import _diff, _dump, _line_span, _physical_lines

MAX_SOURCE_BYTES = 128 * 1024


@dataclass(frozen=True)
class _Value:
    kind: str
    generation: int | None = None
    binding_index: int | None = None
    iterator_id: int | None = None


def _name(node, expected: str) -> bool:
    return isinstance(node, ast.Name) and node.id == expected


def _attr(node, base: str, attribute: str) -> bool:
    return isinstance(node, ast.Attribute) and _name(node.value, base) and node.attr == attribute


def _parse_factory(source: str) -> dict:
    if not isinstance(source, str) or not source or len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ValueError("A nonempty source file of at most 128 KiB is required")
    if "\r" in source.replace("\r\n", ""):
        raise ValueError("Only LF and CRLF line endings are supported")
    tree = ast.parse(source, feature_version=(3, 11))
    aliases, functions = {}, []
    for node in tree.body:
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        if isinstance(node, ast.FunctionDef):
            functions.append(node)
            continue
        if isinstance(node, ast.Import):
            imports = [(item.asname or item.name, item.name) for item in node.names]
            if any(target != "torch" for _, target in imports):
                raise ValueError(
                    "Only torch, torch.optim.AdamW and copy.deepcopy imports are supported"
                )
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imports = [
                (item.asname or item.name, f"{node.module}.{item.name}") for item in node.names
            ]
            if any(target not in {"copy.deepcopy", "torch.optim.AdamW"} for _, target in imports):
                raise ValueError("Unsupported imported symbol")
        else:
            raise ValueError("Module-level executable statements are unsupported")
        for alias, target in imports:
            if alias in aliases or alias in {"list", "tuple", "learning_rate", "weight_decay"}:
                raise ValueError("Import aliases must be unique and must not shadow reserved names")
            aliases[alias] = target
    if len(functions) != 1:
        raise ValueError("Exactly one synchronous factory function is supported")
    function = functions[0]
    args = function.args
    if (
        function.decorator_list
        or args.posonlyargs
        or args.vararg
        or args.kwarg
        or args.defaults
        or any(value is not None for value in args.kw_defaults)
        or len(args.args) != 1
        or [arg.arg for arg in args.kwonlyargs] != ["learning_rate", "weight_decay"]
    ):
        raise ValueError(
            "Expected factory(model, *, learning_rate, weight_decay), without defaults or decorators"
        )
    model_name = args.args[0].arg
    reserved = {*aliases, "list", "tuple", "learning_rate", "weight_decay", model_name}
    if model_name in aliases or model_name in {"list", "tuple", "learning_rate", "weight_decay"}:
        raise ValueError("Model argument shadows a reserved name")
    if function.name in reserved:
        raise ValueError("Factory name shadows an imported symbol or argument")
    reserved.add(function.name)
    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    if (
        len(body) < 3
        or not isinstance(body[-1], ast.Return)
        or not isinstance(body[-1].value, ast.Name)
    ):
        raise ValueError("Factory must end by returning one optimizer variable")
    values, definitions, consumed_iterators = {}, {}, set()
    head_index = optimizer_index = None
    generation = 0

    def expression(node, index):
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if (
            isinstance(node, ast.List)
            and len(node.elts) == 1
            and isinstance(node.elts[0], ast.Dict)
        ):
            group = node.elts[0]
            if (
                len(group.keys) != 1
                or not isinstance(group.keys[0], ast.Constant)
                or group.keys[0].value != "params"
            ):
                raise ValueError("Only one parameter group with the params key is supported")
            value = expression(group.values[0], index)
            if value.kind not in {"lazy", "eager"}:
                raise ValueError("Parameter group must contain an iterable of model parameters")
            return _Value(
                f"group_{value.kind}", value.generation, value.binding_index, value.iterator_id
            )
        if not isinstance(node, ast.Call):
            raise ValueError("Unsupported expression in optimizer factory")
        if _attr(node.func, model_name, "parameters") and not node.args and not node.keywords:
            return _Value("lazy", iterator_id=id(node))
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in {"list", "tuple"}
            and len(node.args) == 1
            and not node.keywords
        ):
            value = expression(node.args[0], index)
            if value.kind == "lazy":
                if value.iterator_id in consumed_iterators:
                    raise ValueError("Reusing a consumed parameter iterator is unsupported")
                consumed_iterators.add(value.iterator_id)
                return _Value("eager", generation, index)
            if value.kind == "eager":
                return value
            raise ValueError("Only materializing model parameters is supported")
        if (
            isinstance(node.func, ast.Name)
            and aliases.get(node.func.id) == "copy.deepcopy"
            and len(node.args) == 1
            and _attr(node.args[0], model_name, "fc")
            and not node.keywords
        ):
            return _Value("head_copy", generation)
        optimizer_call = (
            isinstance(node.func, ast.Name) and aliases.get(node.func.id) == "torch.optim.AdamW"
        ) or (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "AdamW"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "optim"
            and isinstance(node.func.value.value, ast.Name)
            and aliases.get(node.func.value.value.id) == "torch"
        )
        if optimizer_call:
            if (
                len(node.args) != 1
                or len(node.keywords) != 2
                or {kw.arg for kw in node.keywords} != {"lr", "weight_decay"}
                or any(
                    not _name(kw.value, "learning_rate" if kw.arg == "lr" else "weight_decay")
                    for kw in node.keywords
                )
            ):
                raise ValueError(
                    "AdamW must receive parameters, lr=learning_rate and weight_decay=weight_decay"
                )
            value = expression(node.args[0], index)
            if value.kind in {"lazy", "group_lazy"}:
                if value.iterator_id in consumed_iterators:
                    raise ValueError("Reusing a consumed parameter iterator is unsupported")
                consumed_iterators.add(value.iterator_id)
                return _Value("optimizer", generation, index)
            if value.kind in {"eager", "group_eager"}:
                return _Value("optimizer", value.generation, value.binding_index)
            raise ValueError("Unsupported optimizer parameter source")
        raise ValueError("Unknown function call in optimizer factory")

    for index, statement in enumerate(body[:-1]):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            raise ValueError("Only single-target, straight-line assignments are supported")
        value = expression(statement.value, index)
        target = statement.targets[0]
        if _attr(target, model_name, "fc"):
            if (
                head_index is not None
                or value.kind != "head_copy"
                or value.generation != generation
            ):
                raise ValueError(
                    "Exactly one replacement with a copy of the current fc head is supported"
                )
            head_index, generation = index, generation + 1
        elif isinstance(target, ast.Name) and target.id not in reserved and target.id not in values:
            values[target.id], definitions[target.id] = value, index
            if value.kind == "optimizer":
                if optimizer_index is not None:
                    raise ValueError("Only one optimizer construction is supported")
                optimizer_index = index
        else:
            raise ValueError("Reassignment, alias shadowing or an unsupported attribute mutation")
    returned = values.get(body[-1].value.id)
    if (
        head_index is None
        or optimizer_index is None
        or returned is None
        or returned.kind != "optimizer"
    ):
        raise ValueError("One head replacement and one returned optimizer are required")
    # Reject unsupported uses of temporary values even if they do not reach the return.
    if sum(value.kind == "head_copy" for value in values.values()) > 1:
        raise ValueError("Multiple head-copy aliases are unsupported")
    return {
        "tree": tree,
        "function": function,
        "model_name": model_name,
        "aliases": aliases,
        "body": body,
        "definitions": definitions,
        "head_index": head_index,
        "optimizer_index": optimizer_index,
        "binding_index": returned.binding_index,
        "stale": returned.generation != generation,
    }


def analyze_optimizer_factory(source: str) -> dict:
    """Return patch_proposed, no_change, or unsupported; never execute the input."""
    result = {
        "decision": "unsupported",
        "reason": None,
        "function": None,
        "binding_site": None,
        "replacement_site": None,
        "constructor_site": None,
        "proposed_source": None,
        "unified_diff": None,
    }
    try:
        info = _parse_factory(source)
        body = info["body"]
        binding, replacement, constructor = (
            body[info[key]] for key in ("binding_index", "head_index", "optimizer_index")
        )

        def site(node):
            return {
                "line": node.lineno,
                "end_line": node.end_lineno,
                "code": ast.get_source_segment(source, node),
            }

        result.update(
            function=info["function"].name,
            binding_site=site(binding),
            replacement_site=site(replacement),
            constructor_site=site(constructor),
        )
        if not info["stale"]:
            result.update(
                decision="no_change",
                reason="Optimizer binds the current head in the supported factory",
            )
            return result
        if info["binding_index"] >= info["head_index"]:
            raise ValueError("Unsupported binding dependency")
        if (
            isinstance(replacement.value, ast.Name)
            and info["definitions"][replacement.value.id] >= info["binding_index"]
        ):
            raise ValueError(
                "Head copy is prepared after capture; a one-statement move is insufficient"
            )
        lines = _physical_lines(source)
        start, _ = _line_span(lines, binding)
        head_start, head_end = _line_span(lines, replacement)
        if binding.col_offset != replacement.col_offset:
            raise ValueError("Capture and replacement must share indentation")
        proposed = "".join(
            lines[:start] + lines[head_start:head_end] + lines[start:head_start] + lines[head_end:]
        )
        expected = deepcopy(info["tree"])
        function = next(node for node in expected.body if isinstance(node, ast.FunctionDef))
        offset = len(function.body) - len(body)
        moved = function.body.pop(info["head_index"] + offset)
        function.body.insert(info["binding_index"] + offset, moved)
        checked = _parse_factory(proposed)
        if checked["stale"] or _dump(checked["tree"]) != _dump(expected):
            raise ValueError(
                "Proposal must change only the head replacement's position and remove stale binding"
            )
        result.update(
            decision="patch_proposed",
            reason="Parameter identities were captured before replacing the head",
            proposed_source=proposed,
            unified_diff=_diff(source, proposed),
        )
    except (ValueError, SyntaxError, RecursionError) as error:
        result.update(
            decision="unsupported", reason=str(error), proposed_source=None, unified_diff=None
        )
    return result


def bind_validated_factory(source: str, torch_module):
    """Compile only a checked function; import aliases receive trusted bindings."""
    if analyze_optimizer_factory(source)["decision"] == "unsupported":
        raise ValueError("Unsupported factory cannot be bound for execution")
    info = _parse_factory(source)
    function = deepcopy(info["function"])
    function.returns = None
    for arg in function.args.args + function.args.kwonlyargs:
        arg.annotation = None
    trusted = {
        "torch": torch_module,
        "torch.optim.AdamW": torch_module.optim.AdamW,
        "copy.deepcopy": deepcopy,
    }
    namespace = {"__builtins__": {"list": list, "tuple": tuple}}
    namespace.update({name: trusted[target] for name, target in info["aliases"].items()})
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, "<validated-optimizer-factory>", "exec", dont_inherit=True), namespace)
    factory = namespace[function.name]

    def constructor(model, *, variant, learning_rate, weight_decay):
        return factory(model, learning_rate=learning_rate, weight_decay=weight_decay)

    return constructor
