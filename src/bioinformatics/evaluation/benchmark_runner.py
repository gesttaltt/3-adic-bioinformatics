# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Benchmark runner for comprehensive DDG prediction evaluation.

This module provides utilities for running standardized benchmarks
on multiple datasets and comparing with literature methods.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.bioinformatics.evaluation.cross_validation import (
    CrossValidator,
)
from src.bioinformatics.evaluation.metrics import (
    DDGMetrics,
    compare_with_literature,
)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    dataset_name: str
    model_name: str
    cv_type: str
    metrics: DDGMetrics
    comparison: dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Additional info
    n_train: int | None = None
    n_test: int | None = None
    training_time: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "model_name": self.model_name,
            "cv_type": self.cv_type,
            "metrics": self.metrics.to_dict(),
            "comparison": self.comparison,
            "timestamp": self.timestamp,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "training_time": self.training_time,
        }

    def __str__(self) -> str:
        lines = [
            "=" * 60,
            f"Benchmark: {self.model_name} on {self.dataset_name}",
            f"CV: {self.cv_type}",
            "=" * 60,
            str(self.metrics),
            "",
            "Literature Comparison:",
        ]
        for method, info in self.comparison.get("comparison", {}).items():
            symbol = "✓" if info["outperforms"] else "✗"
            lines.append(f"  {symbol} vs {method}: {info['reference']:.3f} (diff: {info['difference']:+.3f})")
        return "\n".join(lines)


class BenchmarkRunner:
    """Runner for standardized DDG prediction benchmarks.

    Runs evaluations on:
    - S669 full benchmark (N=669)
    - ProTherm benchmark subset (N=52, NOT from S669)
    - ProTherm curated (N=176+)
    - Cross-dataset generalization
    """

    # Standard benchmark datasets
    BENCHMARKS = {
        "s669_full": {
            "description": "S669 full benchmark (literature comparison)",
            "n_expected": 669,
            "cv_type": "5-fold x 3",
        },
        "s669_curated": {
            "description": "ProTherm benchmark subset (N=52, NOT from S669 by Pancotti et al. 2022)",
            "n_expected": 52,
            "cv_type": "LOO",
        },
        "protherm": {
            "description": "ProTherm curated high-quality mutations",
            "n_expected": 176,
            "cv_type": "LOO",
        },
    }

    def __init__(
        self,
        output_dir: Path | None = None,
        device: str = "cpu",
    ):
        """Initialize benchmark runner.

        Args:
            output_dir: Directory for results
            device: Device for evaluation
        """
        if output_dir is None:
            output_dir = Path("outputs/benchmarks")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        self.results: list[BenchmarkResult] = []

    def run_benchmark(
        self,
        model: torch.nn.Module,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str,
        model_name: str,
        cv_type: str = "5-fold",
        n_folds: int = 5,
        n_repeats: int = 3,
    ) -> BenchmarkResult:
        """Run a single benchmark evaluation.

        Args:
            model: Model to evaluate
            X: Features
            y: Targets
            dataset_name: Name of dataset
            model_name: Name of model
            cv_type: Type of cross-validation
            n_folds: Number of folds (if k-fold)
            n_repeats: Number of repeats

        Returns:
            BenchmarkResult
        """
        from src.bioinformatics.evaluation.cross_validation import PyTorchModelWrapper

        # Determine model type
        if isinstance(model, torch.nn.Module):

            def model_factory():
                return PyTorchModelWrapper(
                    type(model),
                    model_kwargs={"config": model.config} if hasattr(model, "config") else {},
                    device=self.device,
                )
        else:
            # Assume sklearn-compatible
            def model_factory():
                return model

        cv = CrossValidator(model_factory, scaler=True, seed=42)

        # Run appropriate CV
        if cv_type == "LOO" or len(y) < 100:
            result = cv.loo_cv(X, y, bootstrap_ci=True)
        else:
            result = cv.kfold_cv(
                X,
                y,
                n_folds=n_folds,
                n_repeats=n_repeats,
                stratify=True,
                bootstrap_ci=True,
            )

        # Compare with literature
        comparison = compare_with_literature(result.metrics)

        benchmark = BenchmarkResult(
            dataset_name=dataset_name,
            model_name=model_name,
            cv_type=result.cv_type,
            metrics=result.metrics,
            comparison=comparison,
            n_train=len(y),
        )

        self.results.append(benchmark)
        return benchmark

    def run_model_on_sklearn_features(
        self,
        model_class: type,
        X: np.ndarray,
        y: np.ndarray,
        dataset_name: str,
        model_name: str,
        **model_kwargs,
    ) -> BenchmarkResult:
        """Run benchmark with sklearn model.

        Args:
            model_class: sklearn model class (e.g., Ridge)
            X: Features
            y: Targets
            dataset_name: Dataset name
            model_name: Model name
            **model_kwargs: Model constructor arguments

        Returns:
            BenchmarkResult
        """

        def model_factory():
            return model_class(**model_kwargs)

        cv = CrossValidator(model_factory, scaler=True, seed=42)

        # Use LOO for small datasets, k-fold for larger
        if len(y) < 100:
            result = cv.loo_cv(X, y, bootstrap_ci=True)
        else:
            result = cv.kfold_cv(X, y, n_folds=5, n_repeats=3, bootstrap_ci=True)

        comparison = compare_with_literature(result.metrics)

        benchmark = BenchmarkResult(
            dataset_name=dataset_name,
            model_name=model_name,
            cv_type=result.cv_type,
            metrics=result.metrics,
            comparison=comparison,
            n_train=len(y),
        )

        self.results.append(benchmark)
        return benchmark

    def run_full_suite(
        self,
        model: torch.nn.Module,
        data_loader_factory: Callable[[str], tuple[np.ndarray, np.ndarray]],
        model_name: str,
    ) -> list[BenchmarkResult]:
        """Run full benchmark suite on all standard datasets.

        Args:
            model: Model to evaluate
            data_loader_factory: Function that takes dataset name and returns (X, y)
            model_name: Name of model

        Returns:
            List of BenchmarkResults
        """
        results = []

        for bench_name, bench_info in self.BENCHMARKS.items():
            print(f"\nRunning {bench_name}...")
            try:
                X, y = data_loader_factory(bench_name)
                result = self.run_benchmark(
                    model=model,
                    X=X,
                    y=y,
                    dataset_name=bench_name,
                    model_name=model_name,
                    cv_type=bench_info["cv_type"],
                )
                results.append(result)
                print(result)
            except Exception as e:
                print(f"  Failed: {e}")

        return results

    def save_results(self, filename: str | None = None) -> Path:
        """Save all results to JSON file.

        Args:
            filename: Output filename (default: benchmark_results_{timestamp}.json)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"

        path = self.output_dir / filename

        data = {
            "timestamp": datetime.now().isoformat(),
            "n_benchmarks": len(self.results),
            "results": [r.to_dict() for r in self.results],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to {path}")
        return path

    def generate_report(self) -> str:
        """Generate markdown report of all results.

        Returns:
            Markdown formatted report
        """
        lines = [
            "# DDG Prediction Benchmark Report",
            "",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Summary",
            "",
            "| Dataset | Model | Spearman ρ | 95% CI | N |",
            "|---------|-------|:----------:|--------|--:|",
        ]

        for r in self.results:
            ci_str = ""
            if r.metrics.spearman_ci:
                ci_str = f"[{r.metrics.spearman_ci[0]:.3f}, {r.metrics.spearman_ci[1]:.3f}]"
            lines.append(
                f"| {r.dataset_name} | {r.model_name} | {r.metrics.spearman_r:.3f} | {ci_str} | {r.metrics.n_samples} |"
            )

        lines.extend(
            [
                "",
                "## Literature Comparison",
                "",
                "| Method | Reference | Our Best |",
                "|--------|:---------:|:--------:|",
            ]
        )

        # Find best result for comparison
        if self.results:
            best = max(self.results, key=lambda r: r.metrics.spearman_r)
            for method, info in best.comparison.get("comparison", {}).items():
                symbol = "✓" if info["outperforms"] else ""
                lines.append(f"| {method} | {info['reference']:.3f} | {best.metrics.spearman_r:.3f} {symbol} |")

        lines.extend(
            [
                "",
                "## Detailed Results",
                "",
            ]
        )

        for r in self.results:
            lines.extend(
                [
                    f"### {r.model_name} on {r.dataset_name}",
                    "",
                    f"- CV Type: {r.cv_type}",
                    f"- N samples: {r.metrics.n_samples}",
                    f"- Spearman ρ: {r.metrics.spearman_r:.4f} (p={r.metrics.spearman_p:.2e})",
                    f"- Pearson r: {r.metrics.pearson_r:.4f}",
                    f"- MAE: {r.metrics.mae:.4f} kcal/mol",
                    f"- RMSE: {r.metrics.rmse:.4f} kcal/mol",
                    f"- 3-class accuracy: {r.metrics.accuracy_3class:.1%}",
                    "",
                ]
            )

        return "\n".join(lines)

    def save_report(self, filename: str = "benchmark_report.md") -> Path:
        """Save markdown report.

        Args:
            filename: Output filename

        Returns:
            Path to saved file
        """
        path = self.output_dir / filename
        report = self.generate_report()
        with open(path, "w") as f:
            f.write(report)
        print(f"Report saved to {path}")
        return path


__all__ = [
    "BenchmarkResult",
    "BenchmarkRunner",
]
