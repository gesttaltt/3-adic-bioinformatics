#!/usr/bin/env python3
"""
Add SPDX headers and YAML front-matter to theory documentation files.

This script:
1. Adds SPDX-License-Identifier header as first line
2. Adds YAML front-matter with title, date, authors, version
3. Preserves existing content
4. Generates a report of changes made

Usage:
    python src/scripts/docs/add_spdx_frontmatter.py [--dry-run] [--verbose]
"""

import argparse
import os
import re
from datetime import datetime
from pathlib import Path

# Configuration
SPDX_HEADER = "<!-- SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0 -->"
DEFAULT_AUTHORS = ["AI Whisperers"]
DEFAULT_VERSION = "0.1"
TODAY = datetime.now().strftime("%Y-%m-%d")

# Directories to process
THEORY_DIR = Path("DOCUMENTATION/01_PROJECT_KNOWLEDGE_BASE/02_THEORY_AND_FOUNDATIONS")


def extract_title_from_content(content: str) -> str:
    """Extract title from first H1 heading or filename."""
    # Look for first # heading
    match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return "Untitled"


def has_yaml_frontmatter(content: str) -> bool:
    """Check if content already has YAML front-matter."""
    return content.strip().startswith("---")


def has_spdx_header(content: str) -> bool:
    """Check if content already has SPDX header."""
    return "SPDX-License-Identifier" in content[:500]


def create_frontmatter(title: str, date: str, authors: list, version: str) -> str:
    """Create YAML front-matter block."""
    authors_yaml = "\n".join(f"  - {author}" for author in authors)
    return f"""---
title: "{title}"
date: {date}
authors:
{authors_yaml}
version: "{version}"
license: PolyForm-Noncommercial-1.0.0
---
"""


def process_file(filepath: Path, dry_run: bool = False, verbose: bool = False) -> dict:
    """Process a single markdown file."""
    result = {
        "path": str(filepath),
        "spdx_added": False,
        "frontmatter_added": False,
        "skipped": False,
        "error": None,
    }

    try:
        content = filepath.read_text(encoding="utf-8")
        modified = False

        # Check for existing SPDX
        if has_spdx_header(content):
            if verbose:
                print(f"  SPDX already present: {filepath.name}")
        else:
            # Add SPDX header at the very beginning
            content = SPDX_HEADER + "\n\n" + content
            result["spdx_added"] = True
            modified = True

        # Check for existing front-matter
        if has_yaml_frontmatter(content.replace(SPDX_HEADER, "").strip()):
            if verbose:
                print(f"  Front-matter already present: {filepath.name}")
        else:
            # Extract title and create front-matter
            title = extract_title_from_content(content)
            frontmatter = create_frontmatter(
                title=title,
                date=TODAY,
                authors=DEFAULT_AUTHORS,
                version=DEFAULT_VERSION,
            )

            # Insert front-matter after SPDX header
            if result["spdx_added"]:
                # SPDX was just added, insert after it
                content = SPDX_HEADER + "\n\n" + frontmatter + "\n" + content.replace(SPDX_HEADER + "\n\n", "")
            else:
                # SPDX was already present, find it and insert after
                spdx_match = re.search(r"(<!--\s*SPDX-License-Identifier:[^>]+-->)\s*\n*", content)
                if spdx_match:
                    end_pos = spdx_match.end()
                    content = content[:end_pos] + "\n" + frontmatter + "\n" + content[end_pos:].lstrip()
                else:
                    # No SPDX found (shouldn't happen), prepend frontmatter
                    content = frontmatter + "\n" + content

            result["frontmatter_added"] = True
            modified = True

        if modified and not dry_run:
            filepath.write_text(content, encoding="utf-8")

        if verbose and modified:
            print(f"  Modified: {filepath.name}")

    except Exception as e:
        result["error"] = str(e)
        result["skipped"] = True

    return result


def find_markdown_files(base_dir: Path) -> list:
    """Find all markdown files recursively."""
    return list(base_dir.rglob("*.md"))


def main():
    parser = argparse.ArgumentParser(description="Add SPDX headers and YAML front-matter to documentation")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--dir",
        type=str,
        help="Specific directory to process (relative to repo root)",
    )
    args = parser.parse_args()

    # Find repo root
    repo_root = Path(__file__).parent.parent.parent
    os.chdir(repo_root)

    # Determine directory to process
    target_dir = Path(args.dir) if args.dir else THEORY_DIR

    if not target_dir.exists():
        print(f"Error: Directory not found: {target_dir}")
        return 1

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing markdown files in: {target_dir}")
    print("-" * 60)

    # Find all markdown files
    md_files = find_markdown_files(target_dir)
    print(f"Found {len(md_files)} markdown files")
    print()

    # Process files
    results = {
        "total": len(md_files),
        "spdx_added": 0,
        "frontmatter_added": 0,
        "skipped": 0,
        "errors": [],
    }

    for filepath in sorted(md_files):
        if args.verbose:
            print(f"Processing: {filepath.relative_to(repo_root)}")

        result = process_file(filepath, dry_run=args.dry_run, verbose=args.verbose)

        if result["spdx_added"]:
            results["spdx_added"] += 1
        if result["frontmatter_added"]:
            results["frontmatter_added"] += 1
        if result["skipped"]:
            results["skipped"] += 1
        if result["error"]:
            results["errors"].append(f"{filepath}: {result['error']}")

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files processed: {results['total']}")
    print(f"SPDX headers added:    {results['spdx_added']}")
    print(f"Front-matter added:    {results['frontmatter_added']}")
    print(f"Files skipped:         {results['skipped']}")

    if results["errors"]:
        print(f"\nErrors ({len(results['errors'])}):")
        for error in results["errors"]:
            print(f"  - {error}")

    if args.dry_run:
        print("\n[DRY RUN] No files were modified. Run without --dry-run to apply changes.")

    return 0


if __name__ == "__main__":
    exit(main())
