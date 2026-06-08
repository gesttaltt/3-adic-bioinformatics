#!/usr/bin/env python3
"""
V5.12.2 Hyperbolic Audit - AST-based norm usage analyzer

Scans all Python files in src/ for .norm() and np.linalg.norm usages,
categorizes them, and outputs a detailed report.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NormUsage:
    """A single norm usage found in code."""

    file: str
    line: int
    col: int
    code_snippet: str
    context: str  # function/class name
    norm_type: str  # 'torch.norm', 'np.linalg.norm', '.norm()'
    likely_category: str  # 'needs_review', 'likely_correct', 'in_helper'


@dataclass
class FileAnalysis:
    """Analysis results for a single file."""

    filepath: str
    has_hyperbolic_helper: bool = False
    has_poincare_distance: bool = False
    norm_usages: list[NormUsage] = field(default_factory=list)
    total_norms: int = 0
    in_helper_count: int = 0
    needs_review_count: int = 0


class NormVisitor(ast.NodeVisitor):
    """AST visitor to find norm usages."""

    def __init__(self, source_lines: list[str], filepath: str):
        self.source_lines = source_lines
        self.filepath = filepath
        self.usages: list[NormUsage] = []
        self.current_context = "module"
        self.has_hyperbolic_helper = False
        self.has_poincare_distance = False

    def visit_FunctionDef(self, node):
        old_context = self.current_context
        self.current_context = f"def {node.name}"

        # Check if this is a hyperbolic helper function
        if node.name in (
            "hyperbolic_radius",
            "hyperbolic_radii",
            "poincare_distance",
            "poincare_distance_np",
            "compute_poincare_distance",
        ):
            if "hyperbolic" in node.name:
                self.has_hyperbolic_helper = True
            if "poincare" in node.name:
                self.has_poincare_distance = True

        self.generic_visit(node)
        self.current_context = old_context

    def visit_AsyncFunctionDef(self, node):
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        old_context = self.current_context
        self.current_context = f"class {node.name}"
        self.generic_visit(node)
        self.current_context = old_context

    def visit_Call(self, node):
        """Check for norm calls."""
        norm_type = None

        # Check for .norm() method call
        if isinstance(node.func, ast.Attribute) and node.func.attr == "norm":
            norm_type = ".norm()"

        # Check for torch.norm()
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch":
                if node.func.attr == "norm":
                    norm_type = "torch.norm"

        # Check for np.linalg.norm
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Attribute) and (
                isinstance(node.func.value.value, ast.Name)
                and node.func.value.value.id == "np"
                and node.func.value.attr == "linalg"
                and node.func.attr == "norm"
            ):
                norm_type = "np.linalg.norm"

        # Check for F.normalize or similar
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "normalize":
            norm_type = "F.normalize"

        if norm_type:
            line_no = node.lineno
            code_snippet = self.source_lines[line_no - 1].strip() if line_no <= len(self.source_lines) else ""

            # Determine category
            category = self._categorize_usage(code_snippet, self.current_context)

            usage = NormUsage(
                file=self.filepath,
                line=line_no,
                col=node.col_offset,
                code_snippet=code_snippet[:100],
                context=self.current_context,
                norm_type=norm_type,
                likely_category=category,
            )
            self.usages.append(usage)

        self.generic_visit(node)

    def _categorize_usage(self, code: str, context: str) -> str:
        """Categorize a norm usage."""
        code_lower = code.lower()
        context_lower = context.lower()

        # In helper function - likely correct
        if any(h in context_lower for h in ("hyperbolic_radi", "poincare_distance", "geodesic")):
            return "in_helper"

        # Direction normalization - correct
        if "/" in code and "norm" in code_lower:
            if any(x in code_lower for x in ("direction", "unit", "normalize")):
                return "likely_correct"

        # Clamping/boundary check - correct
        if any(x in code_lower for x in ("clamp", "clip", "< 0.9", "> 0.9", "max_radius")):
            return "likely_correct"

        # Convergence check - correct
        if any(x in code_lower for x in ("tol", "eps", "converge", "update")):
            return "likely_correct"

        # Euclidean explicitly mentioned - intentional
        if "euc" in code_lower or "euclidean" in context_lower:
            return "likely_correct"

        # Radii computation - NEEDS REVIEW
        if "radi" in code_lower:
            return "needs_review"

        # Distance computation without poincare - NEEDS REVIEW
        if "dist" in code_lower and "poincare" not in code_lower:
            return "needs_review"

        # Embedding norms - NEEDS REVIEW
        if any(x in code_lower for x in ("emb", "latent", "z_")):
            return "needs_review"

        return "needs_review"


def analyze_file(filepath: Path) -> FileAnalysis:
    """Analyze a single Python file."""
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            source = f.read()
            source_lines = source.split("\n")

        tree = ast.parse(source)
        visitor = NormVisitor(source_lines, str(filepath))
        visitor.visit(tree)

        analysis = FileAnalysis(
            filepath=str(filepath),
            has_hyperbolic_helper=visitor.has_hyperbolic_helper,
            has_poincare_distance=visitor.has_poincare_distance,
            norm_usages=visitor.usages,
            total_norms=len(visitor.usages),
            in_helper_count=sum(1 for u in visitor.usages if u.likely_category == "in_helper"),
            needs_review_count=sum(1 for u in visitor.usages if u.likely_category == "needs_review"),
        )
        return analysis

    except SyntaxError as e:
        print(f"  Syntax error in {filepath}: {e}")
        return FileAnalysis(filepath=str(filepath))
    except Exception as e:
        print(f"  Error analyzing {filepath}: {e}")
        return FileAnalysis(filepath=str(filepath))


def main():
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"

    print("=" * 80)
    print("V5.12.2 HYPERBOLIC AUDIT - AST ANALYSIS")
    print("=" * 80)

    # Collect all Python files
    all_files = list(src_dir.rglob("*.py"))
    print(f"\nFound {len(all_files)} Python files in src/")

    # Analyze each file
    results: list[FileAnalysis] = []
    files_with_norms = []

    print("\nAnalyzing files...")
    for filepath in sorted(all_files):
        analysis = analyze_file(filepath)
        results.append(analysis)
        if analysis.total_norms > 0:
            files_with_norms.append(analysis)

    # Summary statistics
    total_norms = sum(r.total_norms for r in results)
    total_needs_review = sum(r.needs_review_count for r in results)
    files_with_helpers = sum(1 for r in results if r.has_hyperbolic_helper)
    files_needing_review = [r for r in results if r.needs_review_count > 0 and not r.has_hyperbolic_helper]

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total Python files:           {len(all_files)}")
    print(f"Files with norm() usages:     {len(files_with_norms)}")
    print(f"Files with hyperbolic helper: {files_with_helpers}")
    print(f"Total norm() calls:           {total_norms}")
    print(f"Calls needing review:         {total_needs_review}")
    print(f"Files needing review:         {len(files_needing_review)}")

    # Output detailed report
    print(f"\n{'=' * 80}")
    print("FILES NEEDING REVIEW (no hyperbolic helper, has suspicious norms)")
    print(f"{'=' * 80}")

    for analysis in sorted(files_needing_review, key=lambda x: -x.needs_review_count):
        rel_path = Path(analysis.filepath).relative_to(project_root)
        print(f"\n### {rel_path}")
        print(f"    Total norms: {analysis.total_norms}, Needs review: {analysis.needs_review_count}")

        for usage in analysis.norm_usages:
            if usage.likely_category == "needs_review":
                print(f"    L{usage.line}: [{usage.norm_type}] {usage.code_snippet[:70]}")
                print(f"           Context: {usage.context}")

    # Output files already fixed
    print(f"\n{'=' * 80}")
    print("FILES WITH HYPERBOLIC HELPERS (likely already fixed)")
    print(f"{'=' * 80}")

    for analysis in results:
        if analysis.has_hyperbolic_helper or analysis.has_poincare_distance:
            rel_path = Path(analysis.filepath).relative_to(project_root)
            helpers = []
            if analysis.has_hyperbolic_helper:
                helpers.append("hyperbolic_radius")
            if analysis.has_poincare_distance:
                helpers.append("poincare_distance")
            print(f"  {rel_path} ({', '.join(helpers)})")

    # Generate markdown report
    report_path = project_root / "V5.12.2_DETAILED_AUDIT.md"
    with open(report_path, "w") as f:
        f.write("# V5.12.2 Detailed Hyperbolic Audit Report\n\n")
        f.write("**Generated:** 2025-12-29\n")
        f.write(f"**Total Files Scanned:** {len(all_files)}\n")
        f.write(f"**Files with norm():** {len(files_with_norms)}\n")
        f.write(f"**Files needing review:** {len(files_needing_review)}\n\n")

        f.write("---\n\n")
        f.write("## Files Needing Review\n\n")

        for analysis in sorted(files_needing_review, key=lambda x: -x.needs_review_count):
            rel_path = Path(analysis.filepath).relative_to(project_root)
            f.write(f"### `{rel_path}`\n\n")
            f.write(f"- Total norms: {analysis.total_norms}\n")
            f.write(f"- Needs review: {analysis.needs_review_count}\n\n")

            f.write("| Line | Type | Code | Context |\n")
            f.write("|------|------|------|--------|\n")

            for usage in analysis.norm_usages:
                code_escaped = usage.code_snippet[:50].replace("|", "\\|")
                f.write(f"| {usage.line} | `{usage.norm_type}` | `{code_escaped}` | {usage.context} |\n")

            f.write("\n")

        f.write("---\n\n")
        f.write("## Files Already Fixed (have hyperbolic helpers)\n\n")

        for analysis in results:
            if analysis.has_hyperbolic_helper or analysis.has_poincare_distance:
                rel_path = Path(analysis.filepath).relative_to(project_root)
                f.write(f"- `{rel_path}`\n")

    print(f"\n\nDetailed report written to: {report_path}")

    return len(files_needing_review), total_needs_review


if __name__ == "__main__":
    files_count, norms_count = main()
    print(f"\n\nFINAL: {files_count} files with {norms_count} norm usages need review")
