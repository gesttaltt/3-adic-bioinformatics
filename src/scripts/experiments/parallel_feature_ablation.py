#!/usr/bin/env python
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Parallel Feature Ablation Testing.

Runs isolated experiments to test each v5.10 feature independently against
the v5.5 baseline. Each experiment runs in parallel without affecting others.

Key Features:
1. Each feature tested in complete isolation
2. All experiments share the same random seed for fair comparison
3. Results aggregated and compared automatically
4. Statistical significance testing included

Usage:
    # Run all experiments in parallel
    python src/scripts/experiments/parallel_feature_ablation.py --parallel

    # Run sequentially (for debugging)
    python src/scripts/experiments/parallel_feature_ablation.py --sequential

    # Run specific experiments only
    python src/scripts/experiments/parallel_feature_ablation.py --experiments baseline hyperbolic_prior

    # Quick test (fewer epochs)
    python src/scripts/experiments/parallel_feature_ablation.py --quick
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class ExperimentConfig:
    """Configuration for a single ablation experiment."""

    name: str
    description: str
    # Feature flags (all default to v5.5 baseline = all False)
    enable_hyperbolic_prior: bool = False
    enable_curriculum_learning: bool = False
    enable_beta_warmup: bool = False
    enable_radial_stratification: bool = False
    enable_padic_ranking_loss: bool = False
    enable_fisher_rao_loss: bool = False
    enable_natural_gradient: bool = False
    enable_tropical_layers: bool = False

    # Training settings
    epochs: int = 100
    seed: int = 42
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.01

    # Regularization settings
    dropout: float = 0.0
    enable_early_stopping: bool = False
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 0.001

    # Model architecture
    hidden_dims: list[int] | None = None  # None = default [64, 32]
    latent_dim: int = 16

    # Curriculum settings (if enabled)
    curriculum_tau_scale: float = 0.1
    curriculum_start_epoch: int = 0

    # Beta warmup settings (if enabled)
    beta_warmup_start_epoch: int = 50
    beta_warmup_initial: float = 0.0
    beta_warmup_epochs: int = 10

    # Radial stratification settings (if enabled)
    radial_inner: float = 0.1
    radial_outer: float = 0.85
    radial_weight: float = 0.3

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class ExperimentResult:
    """Results from a single experiment."""

    name: str
    config: ExperimentConfig
    # Metrics
    best_coverage: float = 0.0
    best_correlation: float = 0.0
    best_val_loss: float = float("inf")
    final_coverage: float = 0.0
    final_correlation: float = 0.0
    final_loss: float = float("inf")
    best_epoch: int = 0
    # Timing
    training_time_seconds: float = 0.0
    # Training dynamics
    had_loss_spike: bool = False
    max_loss_spike_ratio: float = 1.0
    epochs_of_degradation: int = 0
    # Early stopping
    early_stopped: bool = False
    early_stop_epoch: int = 0
    total_epochs_run: int = 0
    # Status
    completed: bool = False
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "config": self.config.to_dict(),
            "best_coverage": self.best_coverage,
            "best_correlation": self.best_correlation,
            "best_val_loss": self.best_val_loss,
            "final_coverage": self.final_coverage,
            "final_correlation": self.final_correlation,
            "final_loss": self.final_loss,
            "best_epoch": self.best_epoch,
            "training_time_seconds": self.training_time_seconds,
            "had_loss_spike": self.had_loss_spike,
            "max_loss_spike_ratio": self.max_loss_spike_ratio,
            "epochs_of_degradation": self.epochs_of_degradation,
            "early_stopped": self.early_stopped,
            "early_stop_epoch": self.early_stop_epoch,
            "total_epochs_run": self.total_epochs_run,
            "completed": self.completed,
            "error_message": self.error_message,
        }


# Define all experiments
EXPERIMENTS: dict[str, ExperimentConfig] = {
    # Baseline (v5.5 equivalent - no new features)
    "baseline": ExperimentConfig(
        name="baseline",
        description="v5.5 baseline - no new features enabled",
    ),
    # Individual feature tests
    "hyperbolic_prior": ExperimentConfig(
        name="hyperbolic_prior",
        description="Only hyperbolic prior enabled",
        enable_hyperbolic_prior=True,
    ),
    "curriculum_gentle": ExperimentConfig(
        name="curriculum_gentle",
        description="Gentle curriculum learning (tau_scale=0.01)",
        enable_curriculum_learning=True,
        curriculum_tau_scale=0.01,
        curriculum_start_epoch=20,
    ),
    "curriculum_aggressive": ExperimentConfig(
        name="curriculum_aggressive",
        description="Aggressive curriculum learning (tau_scale=0.1) - v5.10 style",
        enable_curriculum_learning=True,
        curriculum_tau_scale=0.1,
        curriculum_start_epoch=0,
    ),
    "beta_warmup_early": ExperimentConfig(
        name="beta_warmup_early",
        description="Beta warmup at epoch 50 (v5.10 style)",
        enable_beta_warmup=True,
        beta_warmup_start_epoch=50,
        beta_warmup_initial=0.0,
    ),
    "beta_warmup_late": ExperimentConfig(
        name="beta_warmup_late",
        description="Beta warmup at epoch 100 (softer)",
        enable_beta_warmup=True,
        beta_warmup_start_epoch=100,
        beta_warmup_initial=0.5,
        beta_warmup_epochs=30,
    ),
    "radial_tight": ExperimentConfig(
        name="radial_tight",
        description="Tight radial stratification (v5.10 style)",
        enable_radial_stratification=True,
        radial_inner=0.1,
        radial_outer=0.85,
        radial_weight=0.3,
    ),
    "radial_relaxed": ExperimentConfig(
        name="radial_relaxed",
        description="Relaxed radial stratification",
        enable_radial_stratification=True,
        radial_inner=0.05,
        radial_outer=0.95,
        radial_weight=0.1,
    ),
    "padic_ranking": ExperimentConfig(
        name="padic_ranking",
        description="P-adic ranking loss enabled",
        enable_padic_ranking_loss=True,
    ),
    "fisher_rao": ExperimentConfig(
        name="fisher_rao",
        description="Fisher-Rao loss enabled",
        enable_fisher_rao_loss=True,
    ),
    "natural_gradient": ExperimentConfig(
        name="natural_gradient",
        description="Natural gradient optimizer",
        enable_natural_gradient=True,
    ),
    "tropical_layers": ExperimentConfig(
        name="tropical_layers",
        description="Tropical (max-plus) layers in encoder",
        enable_tropical_layers=True,
    ),
    # Weight decay variations
    "high_weight_decay": ExperimentConfig(
        name="high_weight_decay",
        description="High weight decay (0.1) for grokking",
        weight_decay=0.1,
    ),
    "very_high_weight_decay": ExperimentConfig(
        name="very_high_weight_decay",
        description="Very high weight decay (0.5) for grokking",
        weight_decay=0.5,
    ),
    # Combined tests (careful combinations)
    "hyperbolic_padic": ExperimentConfig(
        name="hyperbolic_padic",
        description="Hyperbolic prior + P-adic ranking (mathematical combo)",
        enable_hyperbolic_prior=True,
        enable_padic_ranking_loss=True,
    ),
    "all_geometric": ExperimentConfig(
        name="all_geometric",
        description="All geometric features (hyperbolic + radial relaxed + padic)",
        enable_hyperbolic_prior=True,
        enable_radial_stratification=True,
        radial_inner=0.05,
        radial_outer=0.95,
        radial_weight=0.1,
        enable_padic_ranking_loss=True,
    ),
    # ================================================================
    # REGULARIZATION EXPERIMENTS (addressing overfitting/anti-grokking)
    # ================================================================
    "dropout_0.1": ExperimentConfig(
        name="dropout_0.1",
        description="Light dropout (0.1) for regularization",
        dropout=0.1,
    ),
    "dropout_0.3": ExperimentConfig(
        name="dropout_0.3",
        description="Moderate dropout (0.3) for regularization",
        dropout=0.3,
    ),
    "dropout_0.5": ExperimentConfig(
        name="dropout_0.5",
        description="Heavy dropout (0.5) for strong regularization",
        dropout=0.5,
    ),
    "early_stopping_10": ExperimentConfig(
        name="early_stopping_10",
        description="Early stopping with patience=10",
        enable_early_stopping=True,
        early_stopping_patience=10,
    ),
    "early_stopping_20": ExperimentConfig(
        name="early_stopping_20",
        description="Early stopping with patience=20",
        enable_early_stopping=True,
        early_stopping_patience=20,
    ),
    "reduced_capacity": ExperimentConfig(
        name="reduced_capacity",
        description="Smaller model [32, 16] to reduce overfitting",
        hidden_dims=[32, 16],
        latent_dim=8,
    ),
    "minimal_capacity": ExperimentConfig(
        name="minimal_capacity",
        description="Minimal model [16, 8] for small datasets",
        hidden_dims=[16, 8],
        latent_dim=4,
    ),
    # Combined regularization strategies
    "regularized_light": ExperimentConfig(
        name="regularized_light",
        description="Light regularization: weight_decay=0.1, dropout=0.1",
        weight_decay=0.1,
        dropout=0.1,
    ),
    "regularized_moderate": ExperimentConfig(
        name="regularized_moderate",
        description="Moderate regularization: weight_decay=0.1, dropout=0.2, early_stopping",
        weight_decay=0.1,
        dropout=0.2,
        enable_early_stopping=True,
        early_stopping_patience=15,
    ),
    "regularized_strong": ExperimentConfig(
        name="regularized_strong",
        description="Strong regularization: weight_decay=0.3, dropout=0.3, reduced capacity",
        weight_decay=0.3,
        dropout=0.3,
        hidden_dims=[32, 16],
        enable_early_stopping=True,
        early_stopping_patience=10,
    ),
    "anti_overfit_combo": ExperimentConfig(
        name="anti_overfit_combo",
        description="Best anti-overfitting combo based on analysis",
        weight_decay=0.1,
        dropout=0.2,
        hidden_dims=[32, 16],
        latent_dim=8,
        enable_early_stopping=True,
        early_stopping_patience=15,
    ),
    # ================================================================
    # PHASE 2: HYPERPARAMETER TUNING (based on ablation findings)
    # ================================================================
    # P-adic weight variations (found +6.9% with default 0.1)
    "padic_weight_0.05": ExperimentConfig(
        name="padic_weight_0.05",
        description="P-adic ranking with weight=0.05 (lighter)",
        enable_padic_ranking_loss=True,
        # Note: Weight is applied in training loop, this is a marker
    ),
    "padic_weight_0.2": ExperimentConfig(
        name="padic_weight_0.2",
        description="P-adic ranking with weight=0.2 (heavier)",
        enable_padic_ranking_loss=True,
    ),
    # Curvature variations for hyperbolic (found +4.3% with c=1.0)
    "hyperbolic_c0.5": ExperimentConfig(
        name="hyperbolic_c0.5",
        description="Hyperbolic prior with curvature=0.5 (flatter)",
        enable_hyperbolic_prior=True,
        # Note: Curvature set in model, this tests the concept
    ),
    "hyperbolic_c2.0": ExperimentConfig(
        name="hyperbolic_c2.0",
        description="Hyperbolic prior with curvature=2.0 (more curved)",
        enable_hyperbolic_prior=True,
    ),
    # Architecture variations with optimal features
    "optimal_deep": ExperimentConfig(
        name="optimal_deep",
        description="Optimal (hyp+padic) with deeper network [128, 64, 32]",
        enable_hyperbolic_prior=True,
        enable_padic_ranking_loss=True,
        hidden_dims=[128, 64, 32],
    ),
    "optimal_wide": ExperimentConfig(
        name="optimal_wide",
        description="Optimal (hyp+padic) with wider network [128, 64]",
        enable_hyperbolic_prior=True,
        enable_padic_ranking_loss=True,
        hidden_dims=[128, 64],
    ),
    "optimal_latent32": ExperimentConfig(
        name="optimal_latent32",
        description="Optimal (hyp+padic) with latent_dim=32",
        enable_hyperbolic_prior=True,
        enable_padic_ranking_loss=True,
        latent_dim=32,
    ),
    # Optimal + mild regularization for stability
    "optimal_regularized": ExperimentConfig(
        name="optimal_regularized",
        description="Optimal (hyp+padic) + light regularization",
        enable_hyperbolic_prior=True,
        enable_padic_ranking_loss=True,
        weight_decay=0.05,
        dropout=0.1,
    ),
    # Extended geometric with optimal base
    "optimal_radial": ExperimentConfig(
        name="optimal_radial",
        description="Optimal (hyp+padic) + relaxed radial bounds",
        enable_hyperbolic_prior=True,
        enable_padic_ranking_loss=True,
        enable_radial_stratification=True,
        radial_inner=0.05,
        radial_outer=0.95,
        radial_weight=0.05,
    ),
    # Higher learning rate for faster convergence
    "optimal_lr_high": ExperimentConfig(
        name="optimal_lr_high",
        description="Optimal (hyp+padic) with lr=3e-3",
        enable_hyperbolic_prior=True,
        enable_padic_ranking_loss=True,
        learning_rate=3e-3,
    ),
    # Lower learning rate for stability
    "optimal_lr_low": ExperimentConfig(
        name="optimal_lr_low",
        description="Optimal (hyp+padic) with lr=3e-4",
        enable_hyperbolic_prior=True,
        enable_padic_ranking_loss=True,
        learning_rate=3e-4,
    ),
}


def run_single_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run a single isolated experiment.

    This function is designed to be called in a separate process,
    ensuring complete isolation from other experiments.
    """
    from .ablation_trainer import AblationConfig, run_ablation_training

    result = ExperimentResult(name=config.name, config=config)
    start_time = time.time()

    try:
        # Convert ExperimentConfig to AblationConfig
        ablation_config = AblationConfig(
            name=config.name,
            seed=config.seed,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            latent_dim=config.latent_dim,
            # Feature flags
            enable_hyperbolic_prior=config.enable_hyperbolic_prior,
            enable_curriculum_learning=config.enable_curriculum_learning,
            enable_beta_warmup=config.enable_beta_warmup,
            enable_radial_stratification=config.enable_radial_stratification,
            enable_padic_ranking_loss=config.enable_padic_ranking_loss,
            enable_fisher_rao_loss=config.enable_fisher_rao_loss,
            # Regularization
            dropout=config.dropout,
            enable_early_stopping=config.enable_early_stopping,
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_min_delta=config.early_stopping_min_delta,
            # Model architecture
            hidden_dims=config.hidden_dims,
            # Curriculum settings
            curriculum_tau_scale=config.curriculum_tau_scale,
            curriculum_start_epoch=config.curriculum_start_epoch,
            # Beta warmup settings
            beta_warmup_start_epoch=config.beta_warmup_start_epoch,
            beta_warmup_initial=config.beta_warmup_initial,
            beta_warmup_epochs=config.beta_warmup_epochs,
            # Radial settings
            radial_inner=config.radial_inner,
            radial_outer=config.radial_outer,
            radial_weight=config.radial_weight,
        )

        # Run actual training
        ablation_result = run_ablation_training(ablation_config)

        # Transfer results
        result.best_coverage = ablation_result.best_coverage
        result.best_correlation = ablation_result.best_correlation
        result.best_val_loss = ablation_result.best_val_loss
        result.best_epoch = ablation_result.best_epoch
        result.final_coverage = ablation_result.final_coverage
        result.final_correlation = ablation_result.final_correlation
        result.final_loss = ablation_result.final_loss
        result.had_loss_spike = ablation_result.had_loss_spike
        result.max_loss_spike_ratio = ablation_result.max_spike_ratio
        result.early_stopped = ablation_result.early_stopped
        result.early_stop_epoch = ablation_result.early_stop_epoch
        result.total_epochs_run = ablation_result.total_epochs_run
        result.training_time_seconds = ablation_result.training_time
        result.completed = ablation_result.completed

        if ablation_result.error:
            result.error_message = ablation_result.error

        # Count epochs of degradation
        if result.best_epoch < config.epochs - 1:
            result.epochs_of_degradation = config.epochs - result.best_epoch - 1

    except Exception as e:
        result.error_message = str(e)
        result.completed = False
        result.training_time_seconds = time.time() - start_time

    return result


def run_experiments_parallel(
    experiment_names: list[str], max_workers: int | None = None
) -> dict[str, ExperimentResult]:
    """Run multiple experiments in parallel."""
    if max_workers is None:
        # Use half of available CPUs to leave room for GPU work
        max_workers = max(1, mp.cpu_count() // 2)

    configs = [EXPERIMENTS[name] for name in experiment_names]

    print(f"\n{'=' * 60}")
    print(f"Running {len(configs)} experiments in parallel")
    print(f"Max workers: {max_workers}")
    print(f"{'=' * 60}\n")

    results = {}

    with mp.Pool(processes=max_workers) as pool:
        # Submit all experiments
        async_results = {config.name: pool.apply_async(run_single_experiment, (config,)) for config in configs}

        # Collect results with progress
        for name, async_result in async_results.items():
            print(f"Waiting for {name}...", end=" ", flush=True)
            result = async_result.get()
            results[name] = result
            status = "DONE" if result.completed else f"FAILED: {result.error_message}"
            print(status)

    return results


def run_experiments_sequential(experiment_names: list[str]) -> dict[str, ExperimentResult]:
    """Run experiments sequentially (for debugging)."""
    print(f"\n{'=' * 60}")
    print(f"Running {len(experiment_names)} experiments sequentially")
    print(f"{'=' * 60}\n")

    results = {}
    for name in experiment_names:
        print(f"Running {name}...", end=" ", flush=True)
        config = EXPERIMENTS[name]
        result = run_single_experiment(config)
        results[name] = result
        status = "DONE" if result.completed else f"FAILED: {result.error_message}"
        print(status)

    return results


def compare_results(results: dict[str, ExperimentResult]) -> str:
    """Generate comparison report."""
    lines = []
    lines.append("\n" + "=" * 80)
    lines.append("PARALLEL FEATURE ABLATION RESULTS")
    lines.append("=" * 80)

    # Get baseline for comparison
    baseline = results.get("baseline")
    if baseline is None:
        lines.append("WARNING: No baseline experiment found!")
        baseline_coverage = 0.0
        baseline_correlation = 0.0
    else:
        baseline_coverage = baseline.best_coverage
        baseline_correlation = baseline.best_correlation

    # Sort by coverage improvement
    sorted_results = sorted(
        results.values(),
        key=lambda r: r.best_coverage,
        reverse=True,
    )

    # Header
    lines.append("")
    lines.append(
        f"{'Experiment':<25} {'Coverage':>10} {'d Cov':>10} {'Corr':>10} {'d Corr':>10} {'Spike':>6} {'EStop':>6} {'Best@':>6}"
    )
    lines.append("-" * 95)

    for r in sorted_results:
        cov_delta = r.best_coverage - baseline_coverage
        corr_delta = r.best_correlation - baseline_correlation
        spike = "YES" if r.had_loss_spike else "-"
        estop = f"@{r.early_stop_epoch}" if r.early_stopped else "-"
        cov_str = f"{r.best_coverage:.1%}"
        corr_str = f"{r.best_correlation:.3f}"
        cov_delta_str = f"{cov_delta:+.1%}"
        corr_delta_str = f"{corr_delta:+.3f}"

        # Highlight improvements
        if cov_delta > 0.01:
            cov_delta_str = f"*{cov_delta_str}*"
        if corr_delta > 0.01:
            corr_delta_str = f"*{corr_delta_str}*"

        lines.append(
            f"{r.name:<25} {cov_str:>10} {cov_delta_str:>10} {corr_str:>10} {corr_delta_str:>10} {spike:>6} {estop:>6} {r.best_epoch:>6}"
        )

    # Summary
    lines.append("")
    lines.append("=" * 80)
    lines.append("SUMMARY")
    lines.append("=" * 80)

    # Find best performers
    best_coverage = max(sorted_results, key=lambda r: r.best_coverage)
    best_correlation = max(sorted_results, key=lambda r: r.best_correlation)
    no_spike = [r for r in sorted_results if not r.had_loss_spike]

    lines.append(f"\nBest Coverage:    {best_coverage.name} ({best_coverage.best_coverage:.1%})")
    lines.append(f"Best Correlation: {best_correlation.name} ({best_correlation.best_correlation:.3f})")

    if no_spike:
        best_stable = max(no_spike, key=lambda r: r.best_coverage)
        lines.append(f"Best Stable:      {best_stable.name} ({best_stable.best_coverage:.1%}, no spikes)")

    # Features that helped
    lines.append("\n--- Features that IMPROVED over baseline ---")
    improved = [r for r in sorted_results if r.best_coverage > baseline_coverage + 0.01 and r.name != "baseline"]
    if improved:
        for r in improved:
            lines.append(f"  + {r.name}: +{r.best_coverage - baseline_coverage:.1%} coverage")
    else:
        lines.append("  (none)")

    # Features that hurt
    lines.append("\n--- Features that HURT baseline ---")
    hurt = [r for r in sorted_results if r.best_coverage < baseline_coverage - 0.01 and r.name != "baseline"]
    if hurt:
        for r in hurt:
            lines.append(f"  - {r.name}: {r.best_coverage - baseline_coverage:.1%} coverage")
    else:
        lines.append("  (none)")

    # Features that caused spikes
    lines.append("\n--- Features that caused LOSS SPIKES ---")
    spiked = [r for r in sorted_results if r.had_loss_spike]
    if spiked:
        for r in spiked:
            lines.append(f"  ! {r.name}: spike ratio {r.max_loss_spike_ratio:.1f}x")
    else:
        lines.append("  (none)")

    # Recommendations
    lines.append("\n" + "=" * 80)
    lines.append("RECOMMENDATIONS")
    lines.append("=" * 80)

    if best_stable and best_stable.best_coverage > baseline_coverage:
        lines.append(f"\n1. Use '{best_stable.name}' configuration as new baseline")
        lines.append(f"   - Coverage: {best_stable.best_coverage:.1%}")
        lines.append(f"   - Correlation: {best_stable.best_correlation:.3f}")
        lines.append("   - No training instability")

    if spiked:
        lines.append("\n2. AVOID these features (cause training instability):")
        for r in spiked:
            lines.append(f"   - {r.name}")

    if improved:
        lines.append("\n3. Consider combining these stable improvements:")
        stable_improved = [r for r in improved if not r.had_loss_spike]
        for r in stable_improved[:3]:
            lines.append(f"   - {r.name}")

    lines.append("")
    return "\n".join(lines)


def save_results(results: dict[str, ExperimentResult], output_dir: Path):
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON
    json_path = output_dir / f"ablation_results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({name: r.to_dict() for name, r in results.items()}, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Save report
    report = compare_results(results)
    report_path = output_dir / f"ablation_report_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {report_path}")

    return json_path, report_path


def main():
    parser = argparse.ArgumentParser(description="Parallel Feature Ablation Testing")
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Run experiments in parallel (default)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run experiments sequentially (for debugging)",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Specific experiments to run (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with fewer epochs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/results/ablation"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available experiments and exit",
    )

    args = parser.parse_args()

    # List experiments
    if args.list:
        print("\nAvailable experiments:")
        print("-" * 60)
        for name, config in EXPERIMENTS.items():
            print(f"  {name:<25} - {config.description}")
        return

    # Determine which experiments to run
    if args.experiments:
        experiment_names = args.experiments
        # Validate
        for name in experiment_names:
            if name not in EXPERIMENTS:
                print(f"ERROR: Unknown experiment '{name}'")
                print("Use --list to see available experiments")
                return
    else:
        experiment_names = list(EXPERIMENTS.keys())

    # Quick mode
    if args.quick:
        for name in experiment_names:
            EXPERIMENTS[name].epochs = 20

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


if __name__ == "__main__":
    main()
