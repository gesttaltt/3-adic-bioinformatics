#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""
Add copyright headers to all Python source files.

Usage:
    python src/scripts/legal/add_copyright_headers.py [--dry-run] [--check]

Options:
    --dry-run: Show what would be changed without modifying files
    --check: Exit with error if any files lack copyright headers
"""

import argparse
import sys
from pathlib import Path

# Copyright header to add
COPYRIGHT_HEADER = """# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""

# Files with third-party copyright (do not modify)
THIRD_PARTY_FILES = {
    "research/alphafold3/utils/atom_types.py",
    "research/alphafold3/utils/residue_names.py",
}

# Directories to scan
SOURCE_DIRS = ["src", "research", "scripts"]


def has_copyright_header(file_path: Path) -> bool:
    """Check if file already has a copyright header."""
    try:
        with open(file_path, encoding="utf-8") as f:
            first_lines = "".join(f.readlines()[:10])
            return "Copyright" in first_lines and "AI Whisperers" in first_lines
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False


def is_third_party(file_path: Path, repo_root: Path) -> bool:
    """Check if file is third-party code."""
    rel_path = str(file_path.relative_to(repo_root)).replace("\\", "/")
    return rel_path in THIRD_PARTY_FILES


def add_copyright_header(file_path: Path, dry_run: bool = False) -> bool:
    """Add copyright header to file. Returns True if modified."""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Check if it starts with shebang
        if content.startswith("#!"):
            lines = content.split("\n", 1)
            shebang = lines[0] + "\n"
            rest = lines[1] if len(lines) > 1 else ""
            new_content = shebang + COPYRIGHT_HEADER + rest
        else:
            new_content = COPYRIGHT_HEADER + content

        if not dry_run:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(new_content)

        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def find_python_files(repo_root: Path) -> list[Path]:
    """Find all Python files in source directories."""
    python_files = []
    for source_dir in SOURCE_DIRS:
        dir_path = repo_root / source_dir
        if dir_path.exists():
            python_files.extend(dir_path.rglob("*.py"))
    return python_files


def main():
    parser = argparse.ArgumentParser(description="Add copyright headers to Python source files")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without modifying files",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if all files have headers, exit 1 if any are missing",
    )
    args = parser.parse_args()

    # Find repository root
    repo_root = Path(__file__).parent.parent.parent
    print(f"Repository root: {repo_root}")

    # Find all Python files
    python_files = find_python_files(repo_root)
    print(f"\nFound {len(python_files)} Python files")

    # Process files
    missing_header = []
    third_party = []
    modified = []

    for file_path in sorted(python_files):
        rel_path = file_path.relative_to(repo_root)

        # Skip third-party files
        if is_third_party(file_path, repo_root):
            third_party.append(rel_path)
            continue

        # Check if header exists
        if has_copyright_header(file_path):
            continue

        missing_header.append(rel_path)

        # Add header
        if not args.check and add_copyright_header(file_path, dry_run=args.dry_run):
            modified.append(rel_path)
            action = "Would add" if args.dry_run else "Added"
            print(f"{action} header: {rel_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files:           {len(python_files)}")
    print(f"Third-party files:     {len(third_party)}")
    print(f"Missing header:        {len(missing_header)}")
    print(f"Modified:              {len(modified)}")

    if third_party:
        print("\nThird-party files (skipped):")
        for f in third_party:
            print(f"  - {f}")

    if missing_header and args.check:
        print("\nERROR: Files missing copyright headers:")
        for f in missing_header:
            print(f"  - {f}")
        print("\nRun without --check to add headers automatically")
        sys.exit(1)

    if args.dry_run and missing_header:
        print(f"\nDry run: {len(missing_header)} files would be modified")
        print("Run without --dry-run to apply changes")

    if not args.dry_run and modified:
        print(f"\nSuccess: Added headers to {len(modified)} files")

    sys.exit(0)


if __name__ == "__main__":
    main()
