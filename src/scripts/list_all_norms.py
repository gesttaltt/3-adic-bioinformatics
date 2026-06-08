#!/usr/bin/env python3
"""List ALL norm() calls with full context for manual review."""

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass
class NormCall:
    file: str
    line: int
    code: str
    context: str
    norm_type: str


class NormFinder(ast.NodeVisitor):
    def __init__(self, source_lines: list[str], filepath: str):
        self.source_lines = source_lines
        self.filepath = filepath
        self.calls: list[NormCall] = []
        self.context_stack = ["module"]

    def visit_FunctionDef(self, node):
        self.context_stack.append(f"def {node.name}")
        self.generic_visit(node)
        self.context_stack.pop()

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        self.context_stack.append(f"class {node.name}")
        self.generic_visit(node)
        self.context_stack.pop()

    def visit_Call(self, node):
        norm_type = None

        # .norm() method
        if isinstance(node.func, ast.Attribute) and node.func.attr == "norm":
            norm_type = ".norm()"

        # torch.norm()
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch":
                if node.func.attr == "norm":
                    norm_type = "torch.norm"

        # np.linalg.norm
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and (
                isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "np"
                and node.func.value.attr == "linalg"
                and node.func.attr == "norm"
            )
        ):
            norm_type = "np.linalg.norm"

        # torch.linalg.norm
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Attribute)
            and (
                isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "torch"
                and node.func.value.attr == "linalg"
                and node.func.attr == "norm"
            )
        ):
            norm_type = "torch.linalg.norm"

        if norm_type:
            line = node.lineno
            code = self.source_lines[line - 1].strip() if line <= len(self.source_lines) else ""
            self.calls.append(
                NormCall(file=self.filepath, line=line, code=code, context=self.context_stack[-1], norm_type=norm_type)
            )

        self.generic_visit(node)


def main():
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"

    all_calls: list[NormCall] = []

    for filepath in sorted(src_dir.rglob("*.py")):
        try:
            with open(filepath, encoding="utf-8", errors="ignore") as f:
                source = f.read()
                lines = source.split("\n")

            tree = ast.parse(source)
            finder = NormFinder(lines, str(filepath.relative_to(project_root)))
            finder.visit(tree)
            all_calls.extend(finder.calls)
        except:
            pass

    # Output as markdown table
    print(f"# All {len(all_calls)} norm() calls\n")
    print("| # | File | Line | Type | Context | Code |")
    print("|---|------|------|------|---------|------|")

    for i, call in enumerate(all_calls, 1):
        code_escaped = call.code[:60].replace("|", "\\|")
        print(f"| {i} | `{call.file}` | {call.line} | `{call.norm_type}` | `{call.context}` | `{code_escaped}` |")


if __name__ == "__main__":
    main()
