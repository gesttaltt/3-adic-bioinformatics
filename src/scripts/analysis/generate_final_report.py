# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

import json
import os
from datetime import datetime

RADON_FILE = "radon_results.json"
BANDIT_FILE = "bandit_results.json"
REPORT_FILE = "COMPREHENSIVE_CODE_HEALTH.md"


def load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def generate_report():
    radon_data = load_json(RADON_FILE)
    bandit_data = load_json(BANDIT_FILE)

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
                    f.write(
                        f"| {icon} {r['issue_severity']} | {r['issue_confidence']} | {r['issue_text']} | `{r['filename']}:{r['line_number']}` |\n"
                    )
            else:
                f.write("✅ No security issues found.\n")
        else:
            f.write("⚠️ Bandit analysis failed or produced no output.\n")
        f.write("\n")

        # Complexity Section
        f.write("## 2. Complexity Analysis (Radon)\n")
        if radon_data:
            # radon_data is {filename: [blocks]}
            complex_blocks = []
            for filename, blocks in radon_data.items():
                for b in blocks:
                    # Filter for complexity > 10 (B rank or worse)
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
                for b in complex_blocks[:25]:  # Top 25
                    rank_icon = "☢️" if b["rank"] in ["D", "E", "F"] else "⚠️"
                    f.write(f"| {b['cc']} | {rank_icon} **{b['rank']}** | `{b['file']}` | `{b['name']}` |\n")

                if len(complex_blocks) > 25:
                    f.write(f"\n*...and {len(complex_blocks) - 25} more.*\n")
            else:
                f.write("✅ No high complexity checks failed.\n")
        else:
            f.write("⚠️ Radon analysis failed or produced no output.\n")

    print(f"Report report generated: {os.path.abspath(REPORT_FILE)}")


if __name__ == "__main__":
    generate_report()
