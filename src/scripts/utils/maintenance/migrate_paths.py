# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Migrate hardcoded paths to use centralized path configuration.

This script scans Python files for hardcoded path patterns and optionally
updates them to use the centralized path configuration from src.config.paths.

Usage:
    # Dry run - show what would be changed
    python src/scripts/maintenance/migrate_paths.py --dry-run

    # Apply changes to specific directory
    python src/scripts/maintenance/migrate_paths.py --apply scripts/

    # Show statistics only
    python src/scripts/maintenance/migrate_paths.py --stats
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

# Path patterns to detect and their replacements
PATH_PATTERNS = [
    # Results directory
    (r'default\s*=\s*["\']results/', 'default=str(RESULTS_DIR / "'),
    (r'Path\(\s*["\']results/', 'RESULTS_DIR / "'),
    (r'["\']results/', 'str(RESULTS_DIR / "'),
    # Outputs directory
    (r'default\s*=\s*["\']outputs/', 'default=str(OUTPUT_DIR / "'),
    (r'Path\(\s*["\']outputs/', 'OUTPUT_DIR / "'),
    (r'["\']outputs/', 'str(OUTPUT_DIR / "'),
    # Data raw directory
    (r'default\s*=\s*["\']data/raw/', 'default=str(RAW_DATA_DIR / "'),
    (r'Path\(\s*["\']data/raw/', 'RAW_DATA_DIR / "'),
    (r'["\']data/raw/', 'str(RAW_DATA_DIR / "'),
    # Data processed directory
    (r'default\s*=\s*["\']data/processed/', 'default=str(PROCESSED_DATA_DIR / "'),
    (r'Path\(\s*["\']data/processed/', 'PROCESSED_DATA_DIR / "'),
    (r'["\']data/processed/', 'str(PROCESSED_DATA_DIR / "'),
    # Sandbox training checkpoints
    (r'default\s*=\s*["\']checkpoints/', 'default=str(CHECKPOINTS_DIR / "'),
    (r'Path\(\s*["\']checkpoints/', 'CHECKPOINTS_DIR / "'),
    (r'["\']checkpoints/', 'str(CHECKPOINTS_DIR / "'),
    # Note: sandbox-training/ paths have been migrated to checkpoints/
    # Runs directory
    (r'default\s*=\s*["\']runs/', 'default=str(RUNS_DIR / "'),
    (r'Path\(\s*["\']runs/', 'RUNS_DIR / "'),
    (r'["\']runs/', 'str(RUNS_DIR / "'),
    # Reports directory
    (r'default\s*=\s*["\']reports/', 'default=str(REPORTS_DIR / "'),
    (r'Path\(\s*["\']reports/', 'REPORTS_DIR / "'),
    # Configs directory
    (r'default\s*=\s*["\']configs/', 'default=str(CONFIG_DIR / "'),
    (r'Path\(\s*["\']configs/', 'CONFIG_DIR / "'),
]

# Import line to add when paths are used
PATHS_IMPORT = "from src.config.paths import ("
PATHS_IMPORT_ITEMS = [
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "OUTPUT_DIR",
    "RESULTS_DIR",
    "CHECKPOINTS_DIR",
    "RUNS_DIR",
    "REPORTS_DIR",
    "VIZ_DIR",
]


@dataclass
class PathMatch:
    """Represents a detected hardcoded path."""

    file_path: Path
    line_number: int
    line_content: str
    pattern: str
    replacement: str


@dataclass
class MigrationStats:
    """Statistics for path migration."""

    files_scanned: int = 0
    files_with_matches: int = 0
    total_matches: int = 0
    matches_by_pattern: dict[str, int] = field(default_factory=dict)
    matches_by_file: dict[str, list[PathMatch]] = field(default_factory=dict)


def scan_file(file_path: Path) -> list[PathMatch]:
    """Scan a single file for hardcoded path patterns.

    Args:
        file_path: Path to Python file

    Returns:
        List of detected path matches
    """
    matches = []

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception:
        return matches

    lines = content.split("\n")

    for line_num, line in enumerate(lines, 1):
        # Skip comments and import lines
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("from ") or stripped.startswith("import "):
            continue

        for pattern, replacement in PATH_PATTERNS:
            if re.search(pattern, line):
                matches.append(
                    PathMatch(
                        file_path=file_path,
                        line_number=line_num,
                        line_content=line,
                        pattern=pattern,
                        replacement=replacement,
                    )
                )

    return matches


def scan_directory(
    directory: Path,
    exclude_patterns: list[str] | None = None,
) -> MigrationStats:
    """Scan a directory for hardcoded paths.

    Args:
        directory: Directory to scan
        exclude_patterns: Patterns to exclude (e.g., ["**/test_*"])

    Returns:
        Migration statistics
    """
    if exclude_patterns is None:
        exclude_patterns = [
            "**/test_*.py",
            "**/*_test.py",
            "**/conftest.py",
            "**/migrate_paths.py",
            "**/.git/**",
            "**/__pycache__/**",
        ]

    stats = MigrationStats()

    # Find all Python files
    py_files = list(directory.rglob("*.py"))

    for file_path in py_files:
        # Check exclusions
        file_str = str(file_path)
        excluded = False
        for pattern in exclude_patterns:
            if Path(file_str).match(pattern):
                excluded = True
                break

        if excluded:
            continue

        stats.files_scanned += 1
        matches = scan_file(file_path)

        if matches:
            stats.files_with_matches += 1
            stats.total_matches += len(matches)
            stats.matches_by_file[str(file_path)] = matches

            for match in matches:
                pattern_key = match.pattern[:30] + "..."
                stats.matches_by_pattern[pattern_key] = stats.matches_by_pattern.get(pattern_key, 0) + 1

    return stats


def needs_import(content: str) -> bool:
    """Check if file needs paths import added.

    Args:
        content: File content

    Returns:
        True if import is needed
    """
    return "from src.config.paths import" not in content


def add_import(content: str, needed_imports: list[str]) -> str:
    """Add paths import to file content.

    Args:
        content: File content
        needed_imports: List of path constants needed

    Returns:
        Updated content with import
    """
    if not needed_imports:
        return content

    # Build import statement
    if len(needed_imports) <= 3:
        import_line = f"from src.config.paths import {', '.join(needed_imports)}\n"
    else:
        items = ",\n    ".join(needed_imports)
        import_line = f"from src.config.paths import (\n    {items},\n)\n"

    # Find where to insert (after other imports)
    lines = content.split("\n")
    insert_idx = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("from ") or stripped.startswith("import "):
            insert_idx = i + 1
        elif stripped and not stripped.startswith("#") and insert_idx > 0:
            break

    # Insert import
    lines.insert(insert_idx, import_line)
    return "\n".join(lines)


def apply_migration(file_path: Path, matches: list[PathMatch], dry_run: bool = True) -> bool:
    """Apply path migration to a file.

    Args:
        file_path: Path to file
        matches: List of matches to apply
        dry_run: If True, only show what would change

    Returns:
        True if changes were made (or would be made in dry run)
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return False

    original_content = content
    needed_imports = set()

    # Apply replacements
    for match in matches:
        # Determine which imports are needed based on replacement
        for import_name in PATHS_IMPORT_ITEMS:
            if import_name in match.replacement:
                needed_imports.add(import_name)

        # Apply the replacement
        content = re.sub(match.pattern, match.replacement, content)

    # Add import if needed
    if needed_imports and needs_import(content):
        content = add_import(content, sorted(needed_imports))

    if content != original_content:
        if dry_run:
            print(f"  Would update: {file_path}")
            for match in matches:
                print(f"    Line {match.line_number}: {match.pattern[:40]}...")
        else:
            file_path.write_text(content, encoding="utf-8")
            print(f"  Updated: {file_path}")
        return True

    return False


def print_stats(stats: MigrationStats) -> None:
    """Print migration statistics.

    Args:
        stats: Migration statistics
    """
    print("\n" + "=" * 60)
    print("PATH MIGRATION STATISTICS")
    print("=" * 60)
    print(f"\nFiles scanned: {stats.files_scanned}")
    print(f"Files with hardcoded paths: {stats.files_with_matches}")
    print(f"Total hardcoded paths found: {stats.total_matches}")

    if stats.matches_by_pattern:
        print("\nMatches by pattern:")
        for pattern, count in sorted(stats.matches_by_pattern.items(), key=lambda x: -x[1]):
            print(f"  {count:4d}  {pattern}")

    if stats.matches_by_file:
        print("\nFiles with most matches:")
        sorted_files = sorted(stats.matches_by_file.items(), key=lambda x: -len(x[1]))[:20]
        for file_path, matches in sorted_files:
            # Shorten path for display
            short_path = Path(file_path).name
            print(f"  {len(matches):4d}  {short_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Migrate hardcoded paths to centralized configuration")
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to scan (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without applying",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes (use with caution!)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show statistics only",
    )
    parser.add_argument(
        "--include-tests",
        action="store_true",
        help="Include test files in scan",
    )

    args = parser.parse_args()

    # Resolve directory
    directory = Path(args.directory).resolve()
    if not directory.exists():
        print(f"Error: Directory not found: {directory}")
        return 1

    print(f"Scanning: {directory}")

    # Set up exclusions
    exclude_patterns = ["**/.git/**", "**/__pycache__/**", "**/migrate_paths.py"]
    if not args.include_tests:
        exclude_patterns.extend(["**/test_*.py", "**/*_test.py", "**/conftest.py"])

    # Scan
    stats = scan_directory(directory, exclude_patterns)

    # Print stats
    print_stats(stats)

    # Apply if requested
    if args.apply and not args.stats:
        print("\n" + "=" * 60)
        print("APPLYING MIGRATIONS")
        print("=" * 60)

        for file_path, matches in stats.matches_by_file.items():
            apply_migration(Path(file_path), matches, dry_run=False)

        print("\nMigration complete!")

    elif args.dry_run and not args.stats:
        print("\n" + "=" * 60)
        print("DRY RUN - Would make these changes:")
        print("=" * 60)

        for file_path, matches in stats.matches_by_file.items():
            apply_migration(Path(file_path), matches, dry_run=True)

        print("\nRun with --apply to make changes")

    return 0


if __name__ == "__main__":
    exit(main())
