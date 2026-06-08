# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Metrics for DDG prediction evaluation.

This module provides standard metrics used in the DDG prediction
literature for fair comparison with other methods.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr


@dataclass
class DDGMetrics:
    """Container for DDG prediction metrics."""

    # Correlation metrics
    spearman_r: float
    spearman_p: float
    pearson_r: float
    pearson_p: float

    # Error metrics
    mae: float
    rmse: float
    mse: float

    # Classification metrics (stability categories)
    accuracy_3class: float  # Destabilizing/Neutral/Stabilizing
    accuracy_2class: float  # Destabilizing/Non-destabilizing

    # Sample info
    n_samples: int

    # Confidence intervals (optional, from bootstrap)
    spearman_ci: tuple[float, float] | None = None
    pearson_ci: tuple[float, float] | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "spearman_r": self.spearman_r,
            "spearman_p": self.spearman_p,
            "pearson_r": self.pearson_r,
            "pearson_p": self.pearson_p,
            "mae": self.mae,
            "rmse": self.rmse,
            "mse": self.mse,
            "accuracy_3class": self.accuracy_3class,
            "accuracy_2class": self.accuracy_2class,
            "n_samples": self.n_samples,
            "spearman_ci": self.spearman_ci,
            "pearson_ci": self.pearson_ci,
        }

    def __str__(self) -> str:
        """String representation."""
        lines = [
            f"DDG Metrics (N={self.n_samples})",
            "-" * 40,
            f"Spearman ρ: {self.spearman_r:.4f} (p={self.spearman_p:.4e})",
            f"Pearson r:  {self.pearson_r:.4f} (p={self.pearson_p:.4e})",
            f"MAE:        {self.mae:.4f} kcal/mol",
            f"RMSE:       {self.rmse:.4f} kcal/mol",
            f"3-class:    {self.accuracy_3class:.1%}",
            f"2-class:    {self.accuracy_2class:.1%}",
        ]
        if self.spearman_ci is not None:
            lines.append(f"Spearman 95% CI: [{self.spearman_ci[0]:.3f}, {self.spearman_ci[1]:.3f}]")
        return "\n".join(lines)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bootstrap_ci: bool = False,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> DDGMetrics:
    """Compute all DDG prediction metrics.

    Args:
        y_true: True DDG values
        y_pred: Predicted DDG values
        bootstrap_ci: Compute bootstrap confidence intervals
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for bootstrap

    Returns:
        DDGMetrics object
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: {len(y_true)} vs {len(y_pred)}")

    n_samples = len(y_true)

    # Correlation metrics
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    pearson_r, pearson_p = pearsonr(y_true, y_pred)

    # Error metrics
    errors = y_pred - y_true
    mae = np.mean(np.abs(errors))
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)

    # Classification metrics
    # 3-class: Destabilizing (>1), Neutral ([-1, 1]), Stabilizing (<-1)
    true_class = np.zeros(n_samples, dtype=int)
    pred_class = np.zeros(n_samples, dtype=int)

    true_class[y_true > 1.0] = 1  # Destabilizing
    true_class[y_true < -1.0] = -1  # Stabilizing

    pred_class[y_pred > 1.0] = 1
    pred_class[y_pred < -1.0] = -1

    accuracy_3class = np.mean(true_class == pred_class)

    # 2-class: Destabilizing (>1) vs Non-destabilizing
    true_destab = y_true > 1.0
    pred_destab = y_pred > 1.0
    accuracy_2class = np.mean(true_destab == pred_destab)

    # Bootstrap confidence intervals
    spearman_ci = None
    pearson_ci = None

    if bootstrap_ci and n_samples > 10:
        rng = np.random.RandomState(seed)
        spearman_samples = []
        pearson_samples = []

        for _ in range(n_bootstrap):
            idx = rng.choice(n_samples, n_samples, replace=True)
            s_r, _ = spearmanr(y_true[idx], y_pred[idx])
            p_r, _ = pearsonr(y_true[idx], y_pred[idx])
            spearman_samples.append(s_r)
            pearson_samples.append(p_r)

        spearman_ci = (
            np.percentile(spearman_samples, 2.5),
            np.percentile(spearman_samples, 97.5),
        )
        pearson_ci = (
            np.percentile(pearson_samples, 2.5),
            np.percentile(pearson_samples, 97.5),
        )

    return DDGMetrics(
        spearman_r=float(spearman_r),
        spearman_p=float(spearman_p),
        pearson_r=float(pearson_r),
        pearson_p=float(pearson_p),
        mae=float(mae),
        rmse=float(rmse),
        mse=float(mse),
        accuracy_3class=float(accuracy_3class),
        accuracy_2class=float(accuracy_2class),
        n_samples=n_samples,
        spearman_ci=spearman_ci,
        pearson_ci=pearson_ci,
    )


def compare_with_literature(metrics: DDGMetrics) -> dict:
    """Compare metrics with literature benchmarks.

    Reference values from S669 benchmark:
    - Rosetta: 0.69
    - FoldX: 0.48
    - ESM-1v: 0.51
    - ELASPIC-2: 0.50

    Args:
        metrics: DDGMetrics to compare

    Returns:
        Dictionary with comparison results
    """
    literature = {
        "Rosetta ddg_monomer": 0.69,
        "ESM-1v": 0.51,
        "ELASPIC-2": 0.50,
        "FoldX": 0.48,
    }

    comparison = {}
    for method, ref_spearman in literature.items():
        diff = metrics.spearman_r - ref_spearman
        comparison[method] = {
            "reference": ref_spearman,
            "difference": diff,
            "outperforms": diff > 0,
        }

    return {
        "our_spearman": metrics.spearman_r,
        "comparison": comparison,
        "note": "Literature values benchmarked on S669 (N=669). Ensure fair comparison.",
    }


__all__ = [
    "DDGMetrics",
    "compute_all_metrics",
    "compare_with_literature",
]
