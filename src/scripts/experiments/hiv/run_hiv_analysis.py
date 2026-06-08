#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
"""
HIV Analysis Runner

Unified entry point for running HIV bioinformatics analyses.

Usage:
    python src/scripts/run_hiv_analysis.py                    # Run all analyses
    python src/scripts/run_hiv_analysis.py --escape           # CTL escape only
    python src/scripts/run_hiv_analysis.py --drug-resistance  # Drug resistance only
    python src/scripts/run_hiv_analysis.py --glycan           # Glycan shield only
    python src/scripts/run_hiv_analysis.py --integrase        # Integrase validation
    python src/scripts/run_hiv_analysis.py --all-validations  # All validation scripts

Prerequisites:
    Run `python src/scripts/setup/setup_hiv_analysis.py` first to generate codon encoder.
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HIV_SCRIPTS_DIR = PROJECT_ROOT / "research" / "bioinformatics" / "codon_encoder_research" / "hiv" / "scripts"
GLYCAN_SCRIPTS_DIR = PROJECT_ROOT / "research" / "bioinformatics" / "codon_encoder_research" / "hiv" / "glycan_shield"


def check_prerequisites():
    """Check if codon encoder exists."""
    encoder_paths = [
        PROJECT_ROOT / "research" / "bioinformatics" / "genetic_code" / "data" / "codon_encoder_3adic.pt",
        PROJECT_ROOT
        / "research"
        / "bioinformatics"
        / "codon_encoder_research"
        / "hiv"
        / "data"
        / "codon_encoder_3adic.pt",
    ]

    for path in encoder_paths:
        if path.exists():
            return True

    print("ERROR: Codon encoder not found!")
    print("\nPlease run setup first:")
    print("  python src/scripts/setup/setup_hiv_analysis.py")
    return False


def run_script(script_path, description):
    """Run a Python script and return success status."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Script: {script_path.name}")
    print("=" * 60)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=False,
        )
        if result.returncode == 0:
            print(f"\n{description}: SUCCESS")
            return True
        else:
            print(f"\n{description}: FAILED (exit code {result.returncode})")
            return False
    except Exception as e:
        print(f"\n{description}: ERROR - {e}")
        return False


def run_escape_analysis():
    """Run HIV CTL escape mutation analysis."""
    script = HIV_SCRIPTS_DIR / "01_hiv_escape_analysis.py"
    return run_script(script, "HIV-1 CTL Escape Analysis")


def run_drug_resistance():
    """Run HIV drug resistance analysis."""
    script = HIV_SCRIPTS_DIR / "02_hiv_drug_resistance.py"
    return run_script(script, "HIV-1 Drug Resistance Analysis")


def run_handshake_analysis():
    """Run HIV handshake analysis."""
    script = HIV_SCRIPTS_DIR / "03_hiv_handshake_analysis.py"
    if script.exists():
        return run_script(script, "HIV-1 Handshake Analysis")
    print("Handshake analysis script not found, skipping...")
    return True


def run_hiding_landscape():
    """Run HIV hiding landscape analysis."""
    script = HIV_SCRIPTS_DIR / "04_hiv_hiding_landscape.py"
    if script.exists():
        return run_script(script, "HIV Hiding Landscape Analysis")
    print("Hiding landscape script not found, skipping...")
    return True


def run_glycan_analysis():
    """Run glycan shield analysis."""
    script = GLYCAN_SCRIPTS_DIR / "01_glycan_sentinel_analysis.py"
    if script.exists():
        return run_script(script, "Glycan Shield Sentinel Analysis")
    print("Glycan analysis script not found, skipping...")
    return True


def run_integrase_validation():
    """Run integrase vulnerability validation."""
    script = HIV_SCRIPTS_DIR / "06_validate_integrase_vulnerability.py"
    if script.exists():
        return run_script(script, "HIV Integrase Vulnerability Validation")
    print("Integrase validation script not found, skipping...")
    return True


def run_all_validations():
    """Run all validation scripts."""
    results = []

    # Core validations
    script = HIV_SCRIPTS_DIR / "07_validate_all_conjectures.py"
    if script.exists():
        results.append(run_script(script, "Validate All Conjectures"))

    script = HIV_SCRIPTS_DIR / "08_hybrid_integrase_validation.py"
    if script.exists():
        results.append(run_script(script, "Hybrid Integrase Validation"))

    script = HIV_SCRIPTS_DIR / "09_pdb_crossvalidation.py"
    if script.exists():
        results.append(run_script(script, "PDB Cross-Validation"))

    return all(results) if results else True


def run_visualizations():
    """Run visualization scripts."""
    results = []

    script = HIV_SCRIPTS_DIR / "05_visualize_hiding_landscape.py"
    if script.exists():
        results.append(run_script(script, "Visualize Hiding Landscape"))

    script = HIV_SCRIPTS_DIR / "10_visualize_approach_clusters.py"
    if script.exists():
        results.append(run_script(script, "Visualize Approach Clusters"))

    return all(results) if results else True


def main():
    parser = argparse.ArgumentParser(
        description="HIV Analysis Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/scripts/run_hiv_analysis.py              # Run core analyses
  python src/scripts/run_hiv_analysis.py --escape     # CTL escape only
  python src/scripts/run_hiv_analysis.py --all        # Run everything
  python src/scripts/run_hiv_analysis.py --visualize  # Run visualizations
        """,
    )

    # Analysis options
    parser.add_argument("--escape", action="store_true", help="Run CTL escape analysis")
    parser.add_argument("--drug-resistance", action="store_true", help="Run drug resistance analysis")
    parser.add_argument("--handshake", action="store_true", help="Run handshake analysis")
    parser.add_argument("--hiding", action="store_true", help="Run hiding landscape analysis")
    parser.add_argument("--glycan", action="store_true", help="Run glycan shield analysis")
    parser.add_argument("--integrase", action="store_true", help="Run integrase validation")
    parser.add_argument("--validations", action="store_true", help="Run all validation scripts")
    parser.add_argument("--visualize", action="store_true", help="Run visualization scripts")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    parser.add_argument("--skip-check", action="store_true", help="Skip prerequisite check")

    args = parser.parse_args()

    print("=" * 70)
    print("HIV BIOINFORMATICS ANALYSIS")
    print("Using p-adic Hyperbolic Codon Embeddings")
    print("=" * 70)

    # Check prerequisites
    if not args.skip_check and not check_prerequisites():
        return 1

    # Determine what to run
    run_all = args.all or not any(
        [
            args.escape,
            args.drug_resistance,
            args.handshake,
            args.hiding,
            args.glycan,
            args.integrase,
            args.validations,
            args.visualize,
        ]
    )

    results = []

    # Core analyses
    if args.escape or run_all:
        results.append(("CTL Escape", run_escape_analysis()))

    if args.drug_resistance or run_all:
        results.append(("Drug Resistance", run_drug_resistance()))

    if args.handshake or args.all:
        results.append(("Handshake", run_handshake_analysis()))

    if args.hiding or args.all:
        results.append(("Hiding Landscape", run_hiding_landscape()))

    if args.glycan or run_all:
        results.append(("Glycan Shield", run_glycan_analysis()))

    # Validations
    if args.integrase or args.validations or args.all:
        results.append(("Integrase", run_integrase_validation()))

    if args.validations or args.all:
        results.append(("All Validations", run_all_validations()))

    # Visualizations
    if args.visualize or args.all:
        results.append(("Visualizations", run_visualizations()))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = 0
    failed = 0
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"  {name}: {status}")
        if success:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed")

    # Results location
    results_dir = PROJECT_ROOT / "research" / "bioinformatics" / "codon_encoder_research" / "hiv" / "results"
    print(f"\nResults saved to: {results_dir}")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
