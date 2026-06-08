# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

import json
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

REPORT_FILE = "CODE_HEALTH_DASHBOARD.md"
SRC_DIRS = ["src", "scripts"]


def run_command(command):
    """Run a command and return output.

    Uses subprocess with shell=False for security.
    Command should be a string that will be parsed via shlex.
    """
    try:
        # Parse command string into list for secure execution
        if isinstance(command, str):
            # On Windows, shlex.split may not work perfectly, use simple split for known commands
            if sys.platform == "win32":
                cmd_list = command.split()
            else:
                cmd_list = shlex.split(command)
        else:
            cmd_list = command

        result = subprocess.run(
            cmd_list,
            capture_output=True,
            text=True,
            shell=False,  # Security: avoid shell injection
            encoding="utf-8",
            errors="replace",
        )
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1


def run_ruff():
    """Run ruff in json mode to get structured data."""
    print("Running Ruff analysis...")
    stdout, stderr, code = run_command("ruff check . --output-format json")
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        print(f"Ruff failed to output JSON. Stderr: {stderr}")
        return []


def run_mypy():
    """Run mypy on source directories."""
    print("Running Mypy type checking...")
    # --no-error-summary to easier parsing
    cmd = f"mypy {' '.join(SRC_DIRS)} --ignore-missing-imports --no-error-summary"
    stdout, _, _ = run_command(cmd)
    issues = []
    for line in stdout.splitlines():
        parts = line.split(":")
        if len(parts) >= 4:
            issues.append(
                {
                    "file": parts[0].strip(),
                    "line": parts[1].strip(),
                    "type": parts[2].strip(),
                    "message": ":".join(parts[3:]).strip(),
                    "source": "mypy",
                }
            )
    return issues


def generate_report(ruff_issues, mypy_issues):
    """Generate a Markdown report."""
    total_issues = len(ruff_issues) + len(mypy_issues)

    # Aggregation
    issues_by_file = defaultdict(list)

    for i in ruff_issues:
        issues_by_file[i["filename"]].append(
            {
                "line": i["location"]["row"],
                "code": i["code"],
                "message": i["message"],
                "severity": "Warning",  # Ruff defaults
                "source": "ruff",
            }
        )

    for i in mypy_issues:
        issues_by_file[i["file"]].append(
            {
                "line": i["line"],
                "code": "TYPE",
                "message": i["message"],
                "severity": "Error" if i["type"] == "error" else "Note",
                "source": "mypy",
            }
        )

    # Sort files by issue count (descending)
    sorted_files = sorted(issues_by_file.items(), key=lambda x: len(x[1]), reverse=True)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# Code Health Dashboard\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Total Issues Found:** {total_issues}\n")
        f.write(f"- **Ruff (Lint/Style):** {len(ruff_issues)}\n")
        f.write(f"- **Mypy (Type Safety):** {len(mypy_issues)}\n\n")

        f.write("## Top Offenders\n")
        f.write("| File | Issue Count | Primary Issues |\n")
        f.write("| :--- | :---: | :--- |\n")
        for filename, issues in sorted_files[:10]:
            primary = issues[0]["message"] if issues else "N/A"
            if len(primary) > 50:
                primary = primary[:47] + "..."
            f.write(f"| `{filename}` | {len(issues)} | {primary} |\n")
        f.write("\n")

        f.write("## Detailed Verification Audit\n")
        f.write("> Issues grouped by file. Fix priority: Type Errors > Syntax Errors > Style.\n\n")

        for filename, issues in sorted_files:
            f.write(f"### 📄 `{filename}` ({len(issues)} issues)\n")

            # Group by source for clarity
            errors = sorted(
                issues,
                key=lambda x: int(x["line"]) if str(x["line"]).isdigit() else 0,
            )

            f.write("| Line | Tool | Code | Message |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for err in errors:
                f.write(f"| {err['line']} | **{err['source']}** | `{err['code']}` | {err['message']} |\n")
            f.write("\n")

    print(f"Report generated: {os.path.abspath(REPORT_FILE)}")


if __name__ == "__main__":
    ruff_data = run_ruff()
    mypy_data = run_mypy()
    generate_report(ruff_data, mypy_data)
