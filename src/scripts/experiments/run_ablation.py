#!/usr/bin/env python
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Run Ablation Experiments - Easy entry point for feature testing.

This script provides a simple way to run ablation experiments to test
which features help and which hurt performance.

Usage:
    # Run all experiments (quick mode with 20 epochs)
    python src/scripts/experiments/run_ablation.py --quick

    # Run full experiments (100 epochs)
    python src/scripts/experiments/run_ablation.py --full

    # Run specific experiments
    python src/scripts/experiments/run_ablation.py --experiments baseline high_weight_decay dropout_0.1

    # Run regularization experiments only
    python src/scripts/experiments/run_ablation.py --group regularization

    # Run feature experiments only
    python src/scripts/experiments/run_ablation.py --group features

    # List all available experiments
    python src/scripts/experiments/run_ablation.py --list

Example workflow:
    1. Run quick baseline: python src/scripts/experiments/run_ablation.py --experiments baseline --quick
    2. Compare features: python src/scripts/experiments/run_ablation.py --group features --quick
    3. Run full test on promising features: python src/scripts/experiments/run_ablation.py --experiments baseline hyperbolic_prior high_weight_decay --full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.experiments.parallel_feature_ablation import (
    EXPERIMENTS,
    compare_results,
    run_experiments_parallel,
    run_experiments_sequential,
    save_results,
)

# Experiment groups for easy selection
EXPERIMENT_GROUPS = {
    "baseline": ["baseline"],
    "features": [
        "baseline",
        "hyperbolic_prior",
        "padic_ranking",
        "fisher_rao",
    ],
    "curriculum": [
        "baseline",
        "curriculum_gentle",
        "curriculum_aggressive",
    ],
    "beta_warmup": [
        "baseline",
        "beta_warmup_early",
        "beta_warmup_late",
    ],
    "radial": [
        "baseline",
        "radial_tight",
        "radial_relaxed",
    ],
    "regularization": [
        "baseline",
        "high_weight_decay",
        "very_high_weight_decay",
        "dropout_0.1",
        "dropout_0.3",
        "dropout_0.5",
        "early_stopping_10",
        "early_stopping_20",
    ],
    "capacity": [
        "baseline",
        "reduced_capacity",
        "minimal_capacity",
    ],
    "combinations": [
        "baseline",
        "hyperbolic_padic",
        "all_geometric",
        "regularized_light",
        "regularized_moderate",
        "regularized_strong",
        "anti_overfit_combo",
    ],
    "quick_compare": [
        "baseline",
        "high_weight_decay",
        "hyperbolic_prior",
        "early_stopping_10",
        "anti_overfit_combo",
    ],
    # Phase 2: Hyperparameter tuning for optimal config
    "optimal_tuning": [
        "hyperbolic_padic",
        "optimal_deep",
        "optimal_wide",
        "optimal_latent32",
        "optimal_regularized",
        "optimal_radial",
    ],
    "optimal_lr": [
        "hyperbolic_padic",
        "optimal_lr_high",
        "optimal_lr_low",
    ],
    "padic_tuning": [
        "padic_ranking",
        "padic_weight_0.05",
        "padic_weight_0.2",
    ],
    "hyperbolic_tuning": [
        "hyperbolic_prior",
        "hyperbolic_c0.5",
        "hyperbolic_c2.0",
    ],
    # Recommended next experiments
    "phase2_all": [
        "hyperbolic_padic",
        "optimal_deep",
        "optimal_wide",
        "optimal_latent32",
        "optimal_regularized",
        "optimal_radial",
        "optimal_lr_high",
        "optimal_lr_low",
    ],
}


def main():
    parser = argparse.ArgumentParser(
        description="Run Ablation Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 20 epochs (default)",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Full mode: 100 epochs",
    )
    mode_group.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Custom number of epochs",
    )

    # Experiment selection
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Specific experiments to run",
    )
    parser.add_argument(
        "--group",
        choices=list(EXPERIMENT_GROUPS.keys()),
        default=None,
        help="Run a predefined group of experiments",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run ALL experiments",
    )

    # Execution options
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run sequentially (for debugging)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/results/ablation"),
        help="Output directory for results",
    )

    # Info
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and groups",
    )
    parser.add_argument(
        "--list-groups",
        action="store_true",
        help="List experiment groups",
    )

    args = parser.parse_args()

    # List experiments
    if args.list:
        print("\n" + "=" * 70)
        print("AVAILABLE EXPERIMENTS")
        print("=" * 70)
        for name, config in EXPERIMENTS.items():
            print(f"  {name:<30} - {config.description}")
        print(f"\nTotal: {len(EXPERIMENTS)} experiments")
        return

    # List groups
    if args.list_groups:
        print("\n" + "=" * 70)
        print("EXPERIMENT GROUPS")
        print("=" * 70)
        for group_name, experiments in EXPERIMENT_GROUPS.items():
            print(f"\n  {group_name}:")
            for exp in experiments:
                print(f"    - {exp}")
        return

    # Determine epochs
    if args.full:
        epochs = 100
    elif args.epochs:
        epochs = args.epochs
    else:
        epochs = 20  # Default to quick mode

    # Determine which experiments to run
    if args.experiments:
        experiment_names = args.experiments
    elif args.group:
        experiment_names = EXPERIMENT_GROUPS[args.group]
    elif args.all:
        experiment_names = list(EXPERIMENTS.keys())
    else:
        # Default to quick compare group
        experiment_names = EXPERIMENT_GROUPS["quick_compare"]

    # Validate experiments
    for name in experiment_names:
        if name not in EXPERIMENTS:
            print(f"ERROR: Unknown experiment '{name}'")
            print("Use --list to see available experiments")
            return

    # Set epochs for all selected experiments
    for name in experiment_names:
        EXPERIMENTS[name].epochs = epochs

    # Print summary
    print("\n" + "=" * 70)
    print("ABLATION EXPERIMENT RUN")
    print("=" * 70)
    print(f"Experiments: {len(experiment_names)}")
    print(f"Epochs: {epochs}")
    print(f"Mode: {'Sequential' if args.sequential else 'Parallel'}")
    print(f"Experiments: {', '.join(experiment_names)}")
    print("=" * 70 + "\n")

    # Run experiments
    if args.sequential:
        results = run_experiments_sequential(experiment_names)
    else:
        results = run_experiments_parallel(experiment_names, args.workers)

    # Generate and print report
    report = compare_results(results)
    print(report)

    # Save results
    save_results(results, args.output_dir)

    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
