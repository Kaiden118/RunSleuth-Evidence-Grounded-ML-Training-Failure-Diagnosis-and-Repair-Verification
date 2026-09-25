"""Read-only inspection of training calls and enclosing if branches."""

import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path

TRAINING_CALLS = frozenset({"loss.backward", "optimizer.zero_grad", "optimizer.step"})


@dataclass(frozen=True)
class SourceCall:
    function: str | None
    call: str
    line: int
    code: str
    conditions: tuple[str, ...]


@dataclass(frozen=True)
class SourceInspectionReport:
    source_path: str
    calls: tuple[SourceCall, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


class _TrainingCallVisitor(ast.NodeVisitor):
    def __init__(self, source: str) -> None:
        self.source = source
        self.calls: list[SourceCall] = []
        self.function: str | None = None
        self.conditions: list[str] = []

    def visit_FunctionDef(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        previous_function = self.function
        previous_conditions = self.conditions

        self.function = f"{previous_function}.{node.name}" if previous_function else node.name
        self.conditions = []

        for statement in node.body:
            self.visit(statement)

        self.function = previous_function
        self.conditions = previous_conditions

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_If(self, node: ast.If) -> None:
        condition = ast.get_source_segment(self.source, node.test) or ast.unparse(node.test)
        self.visit(node.test)

        for label, statements in (
            (condition, node.body),
            (f"not ({condition})", node.orelse),
        ):
            self.conditions.append(label)
            for statement in statements:
                self.visit(statement)
            self.conditions.pop()

    def visit_Call(self, node: ast.Call) -> None:
        call = ast.unparse(node.func)
        if call in TRAINING_CALLS:
            self.calls.append(
                SourceCall(
                    function=self.function,
                    call=call,
                    line=node.lineno,
                    code=ast.get_source_segment(self.source, node) or ast.unparse(node),
                    conditions=tuple(self.conditions),
                )
            )
        self.generic_visit(node)


def inspect_training_source(source_path: Path) -> SourceInspectionReport:
    """Parse source without importing or executing the inspected file."""
    source = source_path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(source_path))
    visitor = _TrainingCallVisitor(source)
    visitor.visit(tree)

    return SourceInspectionReport(
        source_path=str(source_path),
        calls=tuple(sorted(visitor.calls, key=lambda item: item.line)),
    )
