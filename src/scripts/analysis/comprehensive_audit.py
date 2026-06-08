# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime

# --- Configuration ---
REPORT_FILE = "COMPREHENSIVE_CODE_HEALTH.md"
JSON_OUTPUT = "code_health_metrics.json"

# Tools to run if available
TOOLS_CONFIG = {
    "ruff": {
        "cmd": ["ruff", "check", ".", "--output-format", "json"],
        "type": "linter",
        "json": True,
    },
    "mypy": {
        "cmd": ["mypy", ".", "--ignore-missing-imports", "--no-error-summary", "--no-pretty"],
        "type": "type_checker",
        "json": False,
    },
    "radon_cc": {
        "cmd": ["radon", "cc", ".", "--json"],
        "type": "complexity",
        "json": True,
    },
    "radon_mi": {
        "cmd": ["radon", "mi", ".", "--json"],
        "type": "maintainability",
        "json": True,
    },
    "bandit": {"cmd": ["bandit", "-r", ".", "-f", "json"], "type": "security", "json": True},
    "vulture": {
        "cmd": ["vulture", ".", "--min-confidence", "80"],
        "type": "dead_code",
        "json": False,
    },
    "pygount": {
        "cmd": ["pygount", "--format=json", "."],
        "type": "metrics",
        "json": True,
    },
}


def run_command(command):
    """Run a command and return stdout, stderr, code.

    Uses subprocess with shell=False for security.
    Command should be a list of arguments.
    """
    try:
        # Convert string to list if needed
        if isinstance(command, str):
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


def check_tool_availability(tool_cmd):
    """Check if the executable exists."""
    bin_name = tool_cmd[0] if isinstance(tool_cmd, list) else tool_cmd.split()[0]
    return shutil.which(bin_name) is not None


def parse_mypy(stdout):
    issues = []
    for line in stdout.splitlines():
        if line.startswith("Success"):
            continue
        parts = line.split(":")
        if len(parts) >= 3:
            issues.append(
                {
                    "file": parts[0].strip(),
                    "line": parts[1].strip(),
                    "message": ":".join(parts[2:]).strip(),
                    "severity": "error",
                }
            )
    return issues


def parse_vulture(stdout):
    issues = []
    for line in stdout.splitlines():
        try:
            # format: script.py:1: unused import 'os' (90% confidence)
            parts = line.split(":")
            if len(parts) >= 3:
                issues.append(
                    {
                        "file": parts[0].strip(),
                        "line": parts[1].strip(),
                        "message": parts[2].strip(),
                    }
                )
        except:
            pass
    return issues


def run_audit():
    print("Starting Comprehensive Audit...")
    results = {}

    for tool_name, config in TOOLS_CONFIG.items():
        cmd_str = config["cmd"]
        bin_name = cmd_str.split()[0]

        print(f"[{tool_name}] Checking availability...", end=" ")
        if not check_tool_availability(bin_name):
            print("❌ Not found (pip install required)")
            results[tool_name] = {"status": "missing", "data": None}
            continue

        print("✅ Running...", end=" ")
        stdout, stderr, code = run_command(cmd_str)

        data = None
        # Parse logic
        try:
            if config["json"]:
                # Sometimes tools output extra text before json, try to find { or [
                json_start = stdout.find("[")
                json_start_brace = stdout.find("{")
                if json_start == -1 and json_start_brace == -1:
                    data = stdout  # fallback
                else:
                    start = min(x for x in [json_start, json_start_brace] if x != -1)
                    data = json.loads(stdout[start:])
            elif tool_name == "mypy":
                data = parse_mypy(stdout)
            elif tool_name == "vulture":
                data = parse_vulture(stdout)
            else:
                data = stdout

            print("Done.")
            results[tool_name] = {"status": "success", "data": data}

        except json.JSONDecodeError:
            print("⚠️ JSON Parse Error")
            results[tool_name] = {
                "status": "error",
                "error": "Failed to parse JSON",
                "raw": stdout[:500],
            }
        except Exception as e:
            print(f"⚠️ Error: {e}")
            results[tool_name] = {"status": "error", "error": str(e)}

    return results


def generate_markdown(results):
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# Comprehensive Code Health Report\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        # Summary Section
        f.write("## 1. Tool Status Summary\n")
        f.write("| Tool | Status | Findings |\n")
        f.write("| :--- | :--- | :--- |\n")

        total_issues = 0

        for name, res in results.items():
            status = res["status"]
            count = "N/A"
            if status == "success":
                data = res["data"]
                if isinstance(data, list):
                    count = len(data)
                elif isinstance(data, dict):
                    count = len(data)  # approximate

                if isinstance(count, int):
                    total_issues += count

            icon = "✅" if status == "success" else "❌" if status == "missing" else "⚠️"
            f.write(f"| {name} | {icon} {status} | {count} |\n")

        f.write("\n")

        # Detailed Sections
        if results.get("radon_cc", {}).get("status") == "success":
            f.write("## 2. Complexity Analysis (Radon)\n")
            data = results["radon_cc"]["data"]
            # data is usually {filename: [blocks]}
            high_complexity = []
            for filename, blocks in data.items():
                for b in blocks:
                    if b["complexity"] > 10:
                        high_complexity.append((filename, b["name"], b["complexity"]))

            high_complexity.sort(key=lambda x: x[2], reverse=True)

            if high_complexity:
                f.write("### High Complexity Functions (CC > 10)\n")
                f.write("| File | Function | Complexity |\n")
                f.write("| :--- | :--- | :---: |\n")
                for f_name, func, cc in high_complexity[:15]:
                    f.write(f"| `{f_name}` | `{func}` | {cc} |\n")
            else:
                f.write("No functions found with Cyclomatic Complexity > 10. 🎉\n")
            f.write("\n")

        if results.get("bandit", {}).get("status") == "success":
            f.write("## 3. Security Audit (Bandit)\n")
            data = results["bandit"]["data"]
            results_list = data.get("results", [])
            if results_list:
                f.write(f"Found {len(results_list)} security issues.\n\n")
                for issue in results_list:
                    f.write(
                        f"- **{issue['issue_severity']}**: {issue['issue_text']} in `{issue['filename']}:{issue['line_number']}`\n"
                    )
            else:
                f.write("No security issues found. 🔒\n")
            f.write("\n")

        # Linter Fallback (Ruff/Mypy)
        f.write("## 4. Linting & Types\n")
        if results.get("ruff", {}).get("status") == "success":
            f.write(f"- **Ruff**: {len(results['ruff']['data'])} issues found.\n")
        if results.get("mypy", {}).get("status") == "success":
            f.write(f"- **Mypy**: {len(results['mypy']['data'])} issues found.\n")

    print(f"Report written to {os.path.abspath(REPORT_FILE)}")

    # Save raw JSON
    with open(JSON_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    data = run_audit()
    generate_markdown(data)
