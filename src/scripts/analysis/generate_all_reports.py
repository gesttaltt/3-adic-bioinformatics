# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

import os
import shutil
import subprocess
import sys
from datetime import datetime

# Define paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TARGET_DIR = os.path.join(
    PROJECT_ROOT,
    "DOCUMENTATION",
    "04_PROJECT_MANAGEMENT",
    "02_REPORTS",
    "audit_data",
)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts", "analysis")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def run_script(script_name, args=None):
    if args is None:
        args = []
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    print(f"Running {script_name}...")
    try:
        cmd = [sys.executable, script_path] + args
        subprocess.run(cmd, check=True, text=True, encoding="utf-8", cwd=PROJECT_ROOT)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return False


def move_file(filename, target_dir):
    src = os.path.join(PROJECT_ROOT, filename)
    if os.path.exists(src):
        dst = os.path.join(target_dir, filename)
        shutil.move(src, dst)
        print(f"Moved {filename} to {target_dir}")
    else:
        print(f"Warning: {filename} not found in root.")


def main():
    print("Starting consolidated audit run...")
    print(f"Target Directory: {TARGET_DIR}")
    ensure_dir(TARGET_DIR)

    # 1. Run Internal Audit (Ruff, Mypy) -> CODE_HEALTH_DASHBOARD.md
    if run_script("audit_repo.py"):
        move_file("CODE_HEALTH_DASHBOARD.md", TARGET_DIR)

    # 2. Run Metrics (Radon, Bandit) -> COMPREHENSIVE_CODE_HEALTH.md, *_results.json
    if run_script("run_metrics.py"):
        move_file("COMPREHENSIVE_CODE_HEALTH.md", TARGET_DIR)
        move_file("radon_results.json", TARGET_DIR)
        move_file(
            "bandit_results.json", TARGET_DIR
        )  # Bandit JSON might be generated with a different name if not configured, checking run_metrics.py logic

    # 3. Run Tool Analysis -> EXTERNAL_TOOLS_REPORT.md
    # We need to capture stdout for this one as it prints to file via redirection typically
    # But let's modify the call to write to file if possible or redirect here
    print("Running analyze_external_tools.py...")
    tool_report_path = os.path.join(TARGET_DIR, "EXTERNAL_TOOLS_REPORT.md")
    with open(tool_report_path, "w", encoding="utf-8") as f:
        subprocess.run(
            [
                sys.executable,
                os.path.join(SCRIPTS_DIR, "analyze_external_tools.py"),
            ],
            stdout=f,
            text=True,
            encoding="utf-8",
            cwd=PROJECT_ROOT,
        )
    print(f"Generated {tool_report_path}")

    # 4. Generate Index
    index_path = os.path.join(TARGET_DIR, "README.md")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("# Audit Data\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("- [Code Health Dashboard](CODE_HEALTH_DASHBOARD.md): Ruff & Mypy Issues\n")
        f.write("- [Comprehensive Health Report](COMPREHENSIVE_CODE_HEALTH.md): Security & Complexity\n")
        f.write("- [External Tools Analysis](EXTERNAL_TOOLS_REPORT.md): Tool Availability\n")
        f.write("- [Radon Raw Data](radon_results.json)\n")
        f.write("- [Bandit Raw Data](bandit_results.json)\n")

    print("\nAll reports generated and moved.")


if __name__ == "__main__":
    main()
