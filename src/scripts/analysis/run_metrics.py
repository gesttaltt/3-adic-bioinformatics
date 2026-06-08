# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

import json
import os
import subprocess
import sys
from datetime import datetime

REPORT_FILE = "COMPREHENSIVE_CODE_HEALTH.md"


def run_radon_cc():
    print("Running Radon CC...")
    try:
        # Run radon module directly
        result = subprocess.run(
            [sys.executable, "-m", "radon", "cc", ".", "--json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            print(f"Radon failed: {result.stderr}")
            return None
    except Exception as e:
        print(f"Radon error: {e}")
        return None


def run_bandit():
    print("Running Bandit...")
    try:
        # Bandit usually needs 'bandit' command, but can be run via -m bandit
        result = subprocess.run(
            [sys.executable, "-m", "bandit", "-r", ".", "-f", "json", "-q"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        # Bandit returns 1 if issues found, so we check stdout for valid JSON
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            print(f"Bandit output invalid JSON: {result.stdout[:200]}...")
            return None
    except Exception as e:
        print(f"Bandit error: {e}")
        return None


def generate_report(radon_data, bandit_data):
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("# Comprehensive Code Health Report\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("**Scope:** Full Repository Audit (Security & Complexity)\n\n")

        # Security Section
        f.write("## 1. Security Analysis (Bandit)\n")
        if bandit_data:
            bandit_data.get("metrics", {}).get("_totals", {})
            results = bandit_data.get("results", [])

            f.write(f"- **Total Issues:** {len(results)}\n")
            f.write(f"- **High Severity:** {sum(1 for r in results if r['issue_severity'] == 'HIGH')}\n")
            f.write(f"- **Medium Severity:** {sum(1 for r in results if r['issue_severity'] == 'MEDIUM')}\n\n")

            if results:
                f.write("| Severity | Confidence | Issue | File |\n")
                f.write("| :--- | :--- | :--- | :--- |\n")
                for r in results:
                    icon = "🔴" if r["issue_severity"] == "HIGH" else "🟠" if r["issue_severity"] == "MEDIUM" else "🟡"
                    fname = r["filename"].replace("\\", "/")
                    f.write(
                        f"| {icon} {r['issue_severity']} | {r['issue_confidence']} | {r['issue_text']} | `{fname}:{r['line_number']}` |\n"
                    )
            else:
                f.write("✅ No security issues found.\n")
        else:
            f.write("⚠️ Bandit analysis failed or produced no output.\n")
        f.write("\n")

        # Complexity Section
        f.write("## 2. Complexity Analysis (Radon)\n")
        if radon_data:
            complex_blocks = []
            for filename, blocks in radon_data.items():
                if isinstance(blocks, list):
                    for b in blocks:
                        if b["complexity"] > 10:
                            complex_blocks.append(
                                {
                                    "file": filename,
                                    "name": b["type"] + " " + b["name"],
                                    "cc": b["complexity"],
                                    "rank": b["rank"],
                                }
                            )

            complex_blocks.sort(key=lambda x: x["cc"], reverse=True)

            f.write(
                f"**Cyclomatic Complexity (CC) Violations:** {len(complex_blocks)} functions/methods with CC > 10.\n\n"
            )

            if complex_blocks:
                f.write("| Complexity | Rank | Location | Function |\n")
                f.write("| :---: | :---: | :--- | :--- |\n")
                for b in complex_blocks[:25]:
                    rank_icon = "☢️" if b["rank"] in ["D", "E", "F"] else "⚠️"
                    fname = b["file"].replace("\\", "/")
                    f.write(f"| {b['cc']} | {rank_icon} **{b['rank']}** | `{fname}` | `{b['name']}` |\n")

                if len(complex_blocks) > 25:
                    f.write(f"\n*...and {len(complex_blocks) - 25} more.*\n")
            else:
                f.write("✅ No high complexity checks failed.\n")
        else:
            f.write("⚠️ Radon analysis failed or produced no output.\n")

    print(f"Report generated: {os.path.abspath(REPORT_FILE)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    r_data = run_radon_cc()
    b_data = run_bandit()

    if r_data:
        with open("radon_results.json", "w", encoding="utf-8") as f:
            json.dump(r_data, f, indent=2)

    if b_data:
        with open("bandit_results.json", "w", encoding="utf-8") as f:
            json.dump(b_data, f, indent=2)

    generate_report(r_data, b_data)
