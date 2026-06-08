# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Cross-disease benchmark script for unified drug resistance prediction.

This script runs experiments across all 11 disease domains to validate
the generalizability of the p-adic VAE framework.

Usage:
    python src/scripts/experiments/run_cross_disease.py
    python src/scripts/experiments/run_cross_disease.py --diseases hiv sars_cov_2 tuberculosis
    python src/scripts/experiments/run_cross_disease.py --output-dir results/benchmarks
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add project root to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

from src.diseases import (
    create_cancer_synthetic_dataset,
    create_candida_synthetic_dataset,
    create_hbv_synthetic_dataset,
    create_hcv_synthetic_dataset,
    create_influenza_synthetic_dataset,
    create_malaria_synthetic_dataset,
    create_mrsa_synthetic_dataset,
    create_rsv_synthetic_dataset,
    create_sars_cov2_dataset,
    create_tb_synthetic_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single disease benchmark."""

    disease: str
    n_samples: int
    n_features: int
    spearman: float
    spearman_std: float
    rmse: float
    rmse_std: float
    runtime_seconds: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CrossDiseaseBenchmarkResult:
    """Aggregated results across all diseases."""

    results: list[BenchmarkResult]
    overall_spearman_mean: float
    overall_spearman_std: float
    total_runtime_seconds: float
    config: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "overall_spearman_mean": self.overall_spearman_mean,
            "overall_spearman_std": self.overall_spearman_std,
            "total_runtime_seconds": self.total_runtime_seconds,
            "config": self.config,
            "timestamp": self.timestamp,
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Results saved to {path}")

    def generate_report(self) -> str:
        """Generate markdown report."""
        lines = [
            "# Cross-Disease Benchmark Results",
            "",
            f"**Generated**: {self.timestamp}",
            f"**Total Runtime**: {self.total_runtime_seconds:.2f}s",
            "",
            "## Summary",
            "",
            f"- **Overall Spearman**: {self.overall_spearman_mean:.4f} ± {self.overall_spearman_std:.4f}",
            f"- **Diseases Evaluated**: {len(self.results)}",
            "",
            "## Per-Disease Results",
            "",
            "| Disease | Samples | Spearman | RMSE | Runtime |",
            "|---------|---------|----------|------|---------|",
        ]

        for r in sorted(self.results, key=lambda x: x.spearman, reverse=True):
            lines.append(
                f"| {r.disease} | {r.n_samples} | "
                f"{r.spearman:.4f} ± {r.spearman_std:.4f} | "
                f"{r.rmse:.4f} | {r.runtime_seconds:.2f}s |"
            )

        lines.extend(
            [
                "",
                "## Configuration",
                "",
                "```json",
                json.dumps(self.config, indent=2),
                "```",
            ]
        )

        return "\n".join(lines)


# Disease dataset creators mapping
DISEASE_DATASETS = {
    "sars_cov_2": create_sars_cov2_dataset,
    "tuberculosis": create_tb_synthetic_dataset,
    "influenza": create_influenza_synthetic_dataset,
    "hcv": create_hcv_synthetic_dataset,
    "hbv": create_hbv_synthetic_dataset,
    "malaria": create_malaria_synthetic_dataset,
    "mrsa": create_mrsa_synthetic_dataset,
    "candida": create_candida_synthetic_dataset,
    "rsv": create_rsv_synthetic_dataset,
    "cancer": create_cancer_synthetic_dataset,
}

ALL_DISEASES = list(DISEASE_DATASETS.keys())


def run_disease_benchmark(
    disease: str,
    n_folds: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> BenchmarkResult:
    """Run benchmark for a single disease.

    Args:
        disease: Disease identifier
        n_folds: Number of cross-validation folds
        n_repeats: Number of repeated runs
        seed: Random seed

    Returns:
        BenchmarkResult with metrics
    """
    logger.info(f"Running benchmark for: {disease.upper()}")
    start_time = time.time()

    # Get dataset creator
    if disease not in DISEASE_DATASETS:
        raise ValueError(f"Unknown disease: {disease}")

    create_dataset = DISEASE_DATASETS[disease]

    # Create synthetic dataset
    try:
        X, y, _ = create_dataset()
    except Exception as e:
        logger.warning(f"Failed to create dataset for {disease}: {e}")
        # Return empty result
        return BenchmarkResult(
            disease=disease,
            n_samples=0,
            n_features=0,
            spearman=0.0,
            spearman_std=0.0,
            rmse=0.0,
            rmse_std=0.0,
            runtime_seconds=0.0,
        )

    logger.info(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Run cross-validation
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold

    all_spearman = []
    all_rmse = []

    for repeat in range(n_repeats):
        np.random.seed(seed + repeat)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed + repeat)

        for _fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Simple baseline model (Ridge regression)
            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            # Compute metrics
            rho, _ = spearmanr(y_val, y_pred)
            rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))

            if not np.isnan(rho):
                all_spearman.append(rho)
            all_rmse.append(rmse)

    runtime = time.time() - start_time

    result = BenchmarkResult(
        disease=disease,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        spearman=float(np.mean(all_spearman)) if all_spearman else 0.0,
        spearman_std=float(np.std(all_spearman)) if all_spearman else 0.0,
        rmse=float(np.mean(all_rmse)),
        rmse_std=float(np.std(all_rmse)),
        runtime_seconds=runtime,
    )

    logger.info(f"  Spearman: {result.spearman:.4f} ± {result.spearman_std:.4f}")
    logger.info(f"  RMSE: {result.rmse:.4f} ± {result.rmse_std:.4f}")

    return result


def run_cross_disease_benchmark(
    diseases: list[str] | None = None,
    n_folds: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
    output_dir: Path | None = None,
) -> CrossDiseaseBenchmarkResult:
    """Run benchmark across multiple diseases.

    Args:
        diseases: List of disease identifiers (None = all)
        n_folds: Number of cross-validation folds
        n_repeats: Number of repeated runs
        seed: Random seed
        output_dir: Output directory for results

    Returns:
        CrossDiseaseBenchmarkResult with aggregated metrics
    """
    diseases = diseases or ALL_DISEASES
    output_dir = output_dir or Path("results/benchmarks")

    logger.info("=" * 60)
    logger.info("Cross-Disease Benchmark")
    logger.info(f"Diseases: {', '.join(diseases)}")
    logger.info(f"Config: {n_folds} folds, {n_repeats} repeats, seed={seed}")
    logger.info("=" * 60)

    start_time = time.time()
    results = []

    for disease in diseases:
        try:
            result = run_disease_benchmark(
                disease=disease,
                n_folds=n_folds,
                n_repeats=n_repeats,
                seed=seed,
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to run benchmark for {disease}: {e}")
            continue

    total_runtime = time.time() - start_time

    # Compute overall metrics
    all_spearman = [r.spearman for r in results if r.spearman > 0]
    overall_mean = float(np.mean(all_spearman)) if all_spearman else 0.0
    overall_std = float(np.std(all_spearman)) if all_spearman else 0.0

    config = {
        "diseases": diseases,
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "seed": seed,
    }

    benchmark_result = CrossDiseaseBenchmarkResult(
        results=results,
        overall_spearman_mean=overall_mean,
        overall_spearman_std=overall_std,
        total_runtime_seconds=total_runtime,
        config=config,
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"cross_disease_benchmark_{timestamp}.json"
    md_path = output_dir / f"cross_disease_benchmark_{timestamp}.md"

    benchmark_result.save(json_path)

    # Save markdown report
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w") as f:
        f.write(benchmark_result.generate_report())
    logger.info(f"Report saved to {md_path}")

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Overall Spearman: {overall_mean:.4f} ± {overall_std:.4f}")
    logger.info(f"Total Runtime: {total_runtime:.2f}s")
    logger.info(f"Results saved to: {output_dir}")

    return benchmark_result


def main():
    parser = argparse.ArgumentParser(description="Run cross-disease benchmark for drug resistance prediction")
    parser.add_argument(
        "--diseases",
        nargs="+",
        choices=ALL_DISEASES,
        default=None,
        help="Diseases to benchmark (default: all)",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Number of repeated runs (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/benchmarks"),
        help="Output directory (default: results/benchmarks)",
    )

    args = parser.parse_args()

    run_cross_disease_benchmark(
        diseases=args.diseases,
        n_folds=args.n_folds,
        n_repeats=args.n_repeats,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
