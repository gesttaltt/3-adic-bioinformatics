# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

import os
import re

DOCS_ROOT = "DOCUMENTATION"
DIAGRAMS_ROOT = "DOCUMENTATION/06_DIAGRAMS"

# Regex to find embedding markers: <!-- embed: path/to/diagram.mmd -->
EMBED_REGEX = re.compile(r"<!--\s*embed:\s*(.*?)\s*-->")


def read_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: File not found {path}")
        return None


def inject_diagrams(content, file_path):
    """
    Scans content for embed tags and replaces the FOLLOWING block with the diagram content.
    It expects the structure:
    <!-- embed: path/to.mmd -->
    ```mermaid
    ... content ...
    ```
    OR just the tag if it's the first run.
    """

    lines = content.split("\n")
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        match = EMBED_REGEX.search(line)

        if match:
            # Found an embed tag
            rel_path = match.group(1)
            # Resolve path relative to project root
            # The paths in tags should be relative to project root or DIAGRAMS_ROOT
            # Let's assume they are relative to PROJECT ROOT for clarity, e.g. DOCUMENTATION/06_DIAGRAMS/...

            diagram_path = rel_path.replace("\\", "/")
            diagram_content = read_file(diagram_path)

            new_lines.append(line)  # Keep the tag

            if diagram_content:
                new_lines.append("```mermaid")
                new_lines.append(diagram_content.strip())
                new_lines.append("```")
                print(f"  Injected: {rel_path}")
            else:
                new_lines.append("<!-- Error: Could not load diagram -->")

            # Skip existing code block if present immediately after
            # We look ahead to check if the next lines are a mermaid block we previously generated
            next_idx = i + 1
            if next_idx < len(lines) and lines[next_idx].strip().startswith("```mermaid"):
                # Skip until end of block
                while next_idx < len(lines) and not lines[next_idx].strip().startswith("```"):
                    next_idx += 1
                if next_idx < len(lines) and lines[next_idx].strip().startswith("```"):
                    next_idx += 1  # Skip the closing ```
                i = next_idx - 1  # Adjust i (loop will increment)

        else:
            new_lines.append(line)

        i += 1

    return "\n".join(new_lines)


def process_docs():
    print("Starting Documentation hydration...")
    for root, _dirs, files in os.walk(DOCS_ROOT):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                content = read_file(file_path)
                if not content:
                    continue

                # Check if file needs injection
                if "<!-- embed:" in content:
                    print(f"Processing: {file_path}")
                    new_content = inject_diagrams(content, file_path)
                    if new_content != content:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        print(f"  Updated: {file_path}")


if __name__ == "__main__":
    process_docs()
