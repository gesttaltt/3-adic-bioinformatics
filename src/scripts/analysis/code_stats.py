# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

import hashlib
import os
from collections import defaultdict


def get_file_stats(root_dirs, extensions=None):
    if extensions is None:
        extensions = {".py", ".md", ".ts", ".tsx", ".js", ".json"}
    file_stats = []
    # Adjust root_dirs to be absolute or relative to where script is run. Assuming run from repo root.
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    for dir_name in root_dirs:
        full_dir_path = os.path.join(repo_root, dir_name)
        if not os.path.exists(full_dir_path):
            print(f"Warning: {full_dir_path} does not exist.")
            continue

        for root, _, files in os.walk(full_dir_path):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, repo_root)

                    try:
                        with open(file_path, encoding="utf-8", errors="ignore") as f:
                            lines = f.readlines()
                            file_stats.append(
                                {
                                    "path": rel_path,
                                    "lines": len(lines),
                                    "content": lines,
                                }
                            )
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")

    return file_stats


def find_duplications(file_stats, window_size=6):
    # Map hash of window -> list of (file_index, start_line)
    window_hashes = defaultdict(list)

    print("Hashing content windows...")
    for f_idx, stats in enumerate(file_stats):
        lines = [line.strip() for line in stats["content"]]
        if len(lines) < window_size:
            continue

        for i in range(len(lines) - window_size + 1):
            window = tuple(lines[i : i + window_size])
            # Skip windows that are just common boilerplate (empty lines, simple imports)
            if all(not line for line in window):  # Skip all empty
                continue

            # Simple heuristic to skip very common short lines like '}', 'return', etc if the window is small
            # but with 6 lines it implies some logic.

            h = hashlib.sha256(str(window).encode("utf-8")).hexdigest()
            window_hashes[h].append((f_idx, i))

    print("Analyzing collisions...")
    # Filter for collisions
    duplicates = {k: v for k, v in window_hashes.items() if len(v) > 1}

    # Merge contiguous blocks? this is complex.
    # Simplified approach: Report collision sets.

    # We want to group by "Code Block".
    # If we have matches at (F1, L1) and (F2, L2), and also (F1, L1+1) and (F2, L2+1), that is one block.

    # Let's just create a list of duplicate blocks.
    # Structure: { 'files': [path1, path2], 'lines': {path1: start1, path2: start2}, 'length': N, 'code': "..." }

    # Ideally checking purely for file-pair duplication is easier to report.
    # "File A and File B share X lines."

    pair_dups = defaultdict(int)  # (path1, path2) -> count of shared windows

    for _, occurrences in duplicates.items():
        # occurrences is a list of (file_index, line_index)
        # All pairs in this list share code.
        unique_files = sorted(list(set(o[0] for o in occurrences)))
        for i in range(len(unique_files)):
            for j in range(i + 1, len(unique_files)):
                f1 = file_stats[unique_files[i]]["path"]
                f2 = file_stats[unique_files[j]]["path"]
                pair_dups[(f1, f2)] += 1

    return pair_dups


def generate_report(file_stats, pair_dups, output_file="CODEBASE_ANALYSIS.md"):
    # Sort files by length
    sorted_files = sorted(file_stats, key=lambda x: x["lines"], reverse=True)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Codebase Analysis Report\n\n")
        f.write("## File Lengths (Lines of Code)\n\n")
        f.write("| File | Lines |\n")
        f.write("|------|-------|\n")
        total_lines = 0
        for stats in sorted_files:
            f.write(f"| `{stats['path']}` | {stats['lines']} |\n")
            total_lines += stats["lines"]

        f.write(f"\n**Total Lines:** {total_lines}\n\n")

        f.write("## Code Duplication (Shared 6-line blocks)\n\n")
        f.write(
            "This section shows pairs of files that share significant chunks of identical code (ignoring whitespace).\n\n"
        )

        # Sort duplicates by "score" (number of shared windows)
        sorted_dups = sorted(pair_dups.items(), key=lambda x: x[1], reverse=True)

        if not sorted_dups:
            f.write("No significant duplication found.\n")
        else:
            f.write("| File A | File B | Shared Blocks (approx lines) |\n")
            f.write("|--------|--------|------------------------------|\n")
            for (f1, f2), count in sorted_dups:
                # 6-line window means 1 count is 6 lines. 2 contiguous counts is 7 lines.
                # So approx lines is count + window_size - 1 if contiguous, but we don't know if contiguous.
                # Roughly 'count' is okay as a metric of "duplicated logic density".
                if count < 3:  # Filter trivial overlaps
                    continue
                f.write(f"| `{f1}` | `{f2}` | {count} blocks |\n")


if __name__ == "__main__":
    print("Starting analysis...")
    dirs_to_analyze = ["src", "scripts", "tests"]
    stats = get_file_stats(dirs_to_analyze)
    print(f"Scanned {len(stats)} files.")

    dups = find_duplications(stats)
    print(f"Found varying degrees of duplication in {len(dups)} pairs.")

    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_path = os.path.join(repo_root, "CODEBASE_ANALYSIS.md")

    generate_report(stats, dups, output_path)
    print(f"Report generated at {output_path}")
