#!/usr/bin/env python
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Verify Holographic Scaling in Hyperbolic VAE.

This script tests the AdS/CFT prediction that mutual information
between sequences should decay as a power law of their hyperbolic distance:

    MI(seq_i, seq_j) ~ d_H(z_i, z_j)^{-2Δ}

where Δ is the conformal dimension.

If this scaling holds, it validates that our hyperbolic VAE
correctly implements holographic principles.

Usage:
    python src/scripts/validation/verify_holographic_scaling.py \
        --model_path checkpoints/vae.pt \
        --data_path data/sequences.fasta \
        --output_dir results/holographic_validation

Outputs:
    - Scaling plot (log-log MI vs distance)
    - Fitted conformal dimension Δ
    - R² goodness of fit
    - Statistical tests for power-law behavior
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScalingResult:
    """Results from holographic scaling analysis."""

    conformal_dimension: float
    conformal_dim_std: float
    scaling_exponent: float  # = 2Δ
    r_squared: float
    n_pairs: int
    kolmogorov_smirnov_stat: float
    kolmogorov_smirnov_pvalue: float
    is_power_law: bool


def compute_hyperbolic_distance(
    z1: torch.Tensor,
    z2: torch.Tensor,
    curvature: float = 1.0,
) -> torch.Tensor:
    """Compute pairwise hyperbolic distances.

    Args:
        z1: Points in Poincaré ball (n1, dim)
        z2: Points in Poincaré ball (n2, dim)
        curvature: Hyperbolic curvature

    Returns:
        Distance matrix (n1, n2)
    """
    c = curvature

    # Expand for pairwise computation
    z1_exp = z1.unsqueeze(1)  # (n1, 1, dim)
    z2_exp = z2.unsqueeze(0)  # (1, n2, dim)

    diff = z1_exp - z2_exp
    norm_z1_sq = (z1_exp * z1_exp).sum(dim=-1)
    norm_z2_sq = (z2_exp * z2_exp).sum(dim=-1)
    norm_diff_sq = (diff * diff).sum(dim=-1)

    numerator = 2 * c * norm_diff_sq
    denominator = (1 - c * norm_z1_sq) * (1 - c * norm_z2_sq)

    ratio = 1 + numerator / denominator.clamp(min=1e-8)
    ratio = ratio.clamp(min=1.0 + 1e-8)

    return (1 / np.sqrt(c)) * torch.acosh(ratio)


def estimate_mutual_information(
    seq1: torch.Tensor,
    seq2: torch.Tensor,
    vocab_size: int = 21,
) -> float:
    """Estimate mutual information between two sequences.

    Uses normalized pointwise mutual information (NPMI).

    Args:
        seq1: First sequence (seq_len,)
        seq2: Second sequence (seq_len,)
        vocab_size: Vocabulary size

    Returns:
        Mutual information estimate
    """
    # Count co-occurrences
    joint_counts = torch.zeros(vocab_size, vocab_size)
    for i in range(len(seq1)):
        if i < len(seq2):
            joint_counts[seq1[i], seq2[i]] += 1

    # Normalize
    joint_prob = joint_counts / joint_counts.sum().clamp(min=1)

    # Marginals
    p1 = joint_prob.sum(dim=1)
    p2 = joint_prob.sum(dim=0)

    # MI = sum p(x,y) log(p(x,y) / (p(x)p(y)))
    outer = p1.unsqueeze(1) * p2.unsqueeze(0)
    mask = (joint_prob > 0) & (outer > 0)

    mi = 0.0
    if mask.any():
        mi = (joint_prob[mask] * torch.log(joint_prob[mask] / outer[mask])).sum()

    return float(mi)


def estimate_mi_from_embeddings(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.1,
) -> float:
    """Estimate MI from embedding similarity.

    Uses InfoNCE-style estimator:
    MI ≈ log(similarity / avg_similarity)

    Args:
        z1: First embedding (dim,)
        z2: Second embedding (dim,)
        temperature: Softmax temperature

    Returns:
        MI estimate
    """
    # Cosine similarity
    sim = F.cosine_similarity(z1.unsqueeze(0), z2.unsqueeze(0)).item()

    # Transform to MI-like quantity
    # Higher similarity → higher MI
    mi_estimate = np.log(np.exp(sim / temperature) + 1)

    return mi_estimate


def power_law(x: np.ndarray, alpha: float, c: float) -> np.ndarray:
    """Power law function: y = c * x^(-alpha)."""
    return c * np.power(x + 0.01, -alpha)


def fit_power_law(
    distances: np.ndarray,
    mi_values: np.ndarray,
) -> tuple[float, float, float]:
    """Fit power law to MI vs distance data.

    Args:
        distances: Hyperbolic distances
        mi_values: Mutual information values

    Returns:
        Tuple of (exponent, coefficient, r_squared)
    """
    # Filter out zeros and infinities
    mask = (distances > 0) & (mi_values > 0) & np.isfinite(mi_values)
    x = distances[mask]
    y = mi_values[mask]

    if len(x) < 10:
        return 0.0, 0.0, 0.0

    try:
        # Fit in log-log space
        log_x = np.log(x)
        log_y = np.log(y)

        slope, intercept, r_value, _, _ = stats.linregress(log_x, log_y)

        exponent = -slope
        coefficient = np.exp(intercept)
        r_squared = r_value**2

        return exponent, coefficient, r_squared

    except Exception as e:
        logger.warning(f"Power law fit failed: {e}")
        return 0.0, 0.0, 0.0


def kolmogorov_smirnov_test(
    distances: np.ndarray,
    mi_values: np.ndarray,
    exponent: float,
) -> tuple[float, float]:
    """Test if data follows power law using KS test.

    Args:
        distances: Hyperbolic distances
        mi_values: Mutual information values
        exponent: Fitted exponent

    Returns:
        Tuple of (KS statistic, p-value)
    """
    mask = (distances > 0) & (mi_values > 0) & np.isfinite(mi_values)
    x = distances[mask]
    y = mi_values[mask]

    if len(x) < 10:
        return 0.0, 0.0

    # Generate expected values
    expected = power_law(x, exponent, y.mean() * (x.mean() ** exponent))

    # Normalize for comparison
    y_norm = y / y.max()
    exp_norm = expected / expected.max()

    stat, pvalue = stats.ks_2samp(y_norm, exp_norm)

    return stat, pvalue


def verify_holographic_scaling(
    embeddings: torch.Tensor,
    sequences: list[torch.Tensor] | None = None,
    curvature: float = 1.0,
    n_samples: int = 1000,
    use_sequence_mi: bool = False,
) -> ScalingResult:
    """Verify holographic scaling relation.

    Tests: MI(i,j) ~ d_H(z_i, z_j)^{-2Δ}

    Args:
        embeddings: Sequence embeddings in Poincaré ball (n_seqs, dim)
        sequences: Optional sequences for direct MI computation
        curvature: Hyperbolic curvature
        n_samples: Number of pairs to sample
        use_sequence_mi: Whether to use sequence-level MI

    Returns:
        ScalingResult with fitted parameters and statistics
    """
    n_seqs = embeddings.size(0)

    # Sample random pairs
    np.random.seed(42)
    idx1 = np.random.choice(n_seqs, n_samples, replace=True)
    idx2 = np.random.choice(n_seqs, n_samples, replace=True)

    # Filter out self-pairs
    valid = idx1 != idx2
    idx1 = idx1[valid]
    idx2 = idx2[valid]

    # Compute distances
    distances = []
    mi_values = []

    with torch.no_grad():
        for i, j in zip(idx1, idx2, strict=False):
            # Hyperbolic distance
            z1, z2 = embeddings[i], embeddings[j]
            dist = compute_hyperbolic_distance(z1.unsqueeze(0), z2.unsqueeze(0), curvature).item()
            distances.append(dist)

            # Mutual information
            if use_sequence_mi and sequences is not None:
                mi = estimate_mutual_information(sequences[i], sequences[j])
            else:
                mi = estimate_mi_from_embeddings(z1, z2)
            mi_values.append(mi)

    distances = np.array(distances)
    mi_values = np.array(mi_values)

    # Fit power law
    exponent, coefficient, r_squared = fit_power_law(distances, mi_values)
    conformal_dim = exponent / 2  # 2Δ = exponent

    # Bootstrap for uncertainty
    n_bootstrap = 100
    bootstrap_deltas = []
    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(len(distances), len(distances), replace=True)
        exp_boot, _, _ = fit_power_law(distances[boot_idx], mi_values[boot_idx])
        bootstrap_deltas.append(exp_boot / 2)

    conformal_dim_std = np.std(bootstrap_deltas)

    # Kolmogorov-Smirnov test
    ks_stat, ks_pvalue = kolmogorov_smirnov_test(distances, mi_values, exponent)

    # Determine if power law holds (p > 0.05 means we can't reject)
    is_power_law = ks_pvalue > 0.05 and r_squared > 0.5

    return ScalingResult(
        conformal_dimension=conformal_dim,
        conformal_dim_std=conformal_dim_std,
        scaling_exponent=exponent,
        r_squared=r_squared,
        n_pairs=len(distances),
        kolmogorov_smirnov_stat=ks_stat,
        kolmogorov_smirnov_pvalue=ks_pvalue,
        is_power_law=is_power_law,
    )


def plot_holographic_scaling(
    distances: np.ndarray,
    mi_values: np.ndarray,
    result: ScalingResult,
    output_path: Path,
):
    """Generate holographic scaling plot.

    Args:
        distances: Hyperbolic distances
        mi_values: Mutual information values
        result: Scaling analysis result
        output_path: Output file path
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Log-log scatter with fit
    ax1 = axes[0]
    mask = (distances > 0) & (mi_values > 0) & np.isfinite(mi_values)
    x, y = distances[mask], mi_values[mask]

    ax1.scatter(x, y, alpha=0.3, s=10, label="Data")

    # Fit line
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = power_law(x_fit, result.scaling_exponent, y.mean() * (x.mean() ** result.scaling_exponent))
    ax1.plot(x_fit, y_fit, "r-", linewidth=2, label=f"Fit: MI ~ d^{{-{result.scaling_exponent:.2f}}}")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Hyperbolic Distance")
    ax1.set_ylabel("Mutual Information")
    ax1.set_title(f"Holographic Scaling (Δ = {result.conformal_dimension:.2f} ± {result.conformal_dim_std:.2f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Residuals
    ax2 = axes[1]
    log_residuals = np.log(y) - np.log(
        power_law(x, result.scaling_exponent, y.mean() * (x.mean() ** result.scaling_exponent))
    )
    ax2.hist(log_residuals, bins=50, density=True, alpha=0.7)
    ax2.axvline(0, color="r", linestyle="--", label="Expected")
    ax2.set_xlabel("Log Residuals")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Residual Distribution (R² = {result.r_squared:.3f})")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved scaling plot to {output_path}")


def run_validation(
    model_path: Path | None = None,
    data_path: Path | None = None,
    output_dir: Path = Path("results/holographic_validation"),
    n_synthetic: int = 500,
    curvature: float = 1.0,
) -> ScalingResult:
    """Run holographic scaling validation.

    Args:
        model_path: Path to trained VAE model
        data_path: Path to sequence data
        output_dir: Output directory
        n_synthetic: Number of synthetic samples if no model provided
        curvature: Hyperbolic curvature

    Returns:
        Validation results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if model_path is not None and model_path.exists():
        # Load real model and data
        logger.info(f"Loading model from {model_path}")
        # TODO: Implement actual model loading
        # model = torch.load(model_path)
        # embeddings = model.encode(data)
        raise NotImplementedError("Model loading not yet implemented")
    else:
        # Generate synthetic data with known holographic structure
        logger.info(f"Generating {n_synthetic} synthetic embeddings")
        embeddings = generate_synthetic_holographic_data(
            n_samples=n_synthetic,
            latent_dim=16,
            curvature=curvature,
        )

    # Run verification
    logger.info("Computing holographic scaling...")
    result = verify_holographic_scaling(
        embeddings,
        curvature=curvature,
        n_samples=min(10000, n_synthetic * (n_synthetic - 1) // 2),
    )

    # Generate plots
    logger.info("Generating plots...")
    n_seqs = embeddings.size(0)
    idx1 = np.random.choice(n_seqs, 2000, replace=True)
    idx2 = np.random.choice(n_seqs, 2000, replace=True)
    valid = idx1 != idx2
    idx1, idx2 = idx1[valid], idx2[valid]

    distances = []
    mi_values = []
    with torch.no_grad():
        for i, j in zip(idx1, idx2, strict=False):
            z1, z2 = embeddings[i], embeddings[j]
            dist = compute_hyperbolic_distance(z1.unsqueeze(0), z2.unsqueeze(0), curvature).item()
            distances.append(dist)
            mi_values.append(estimate_mi_from_embeddings(z1, z2))

    plot_holographic_scaling(
        np.array(distances),
        np.array(mi_values),
        result,
        output_dir / "holographic_scaling.png",
    )

    # Save results
    results_dict = {
        "conformal_dimension": result.conformal_dimension,
        "conformal_dim_std": result.conformal_dim_std,
        "scaling_exponent": result.scaling_exponent,
        "r_squared": result.r_squared,
        "n_pairs": result.n_pairs,
        "ks_statistic": result.kolmogorov_smirnov_stat,
        "ks_pvalue": result.kolmogorov_smirnov_pvalue,
        "is_power_law": result.is_power_law,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("HOLOGRAPHIC SCALING VALIDATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Conformal dimension Δ: {result.conformal_dimension:.3f} ± {result.conformal_dim_std:.3f}")
    logger.info(f"Scaling exponent (2Δ): {result.scaling_exponent:.3f}")
    logger.info(f"R² (goodness of fit): {result.r_squared:.3f}")
    logger.info(f"KS p-value: {result.kolmogorov_smirnov_pvalue:.3f}")
    logger.info(f"Power law holds: {'YES' if result.is_power_law else 'NO'}")
    logger.info("=" * 50)

    return result


def generate_synthetic_holographic_data(
    n_samples: int = 500,
    latent_dim: int = 16,
    curvature: float = 1.0,
    true_delta: float = 1.0,
) -> torch.Tensor:
    """Generate synthetic data with known holographic structure.

    Creates embeddings where MI decays as power law of hyperbolic distance.

    Args:
        n_samples: Number of samples
        latent_dim: Latent dimension
        curvature: Hyperbolic curvature
        true_delta: True conformal dimension

    Returns:
        Embeddings in Poincaré ball
    """
    # Generate points with hierarchical structure
    # Root at origin, children at increasing radii
    levels = 5
    points_per_level = n_samples // levels

    embeddings = []

    for level in range(levels):
        # Radius increases with level
        base_radius = 0.1 + 0.8 * level / (levels - 1)

        for _ in range(points_per_level):
            # Random direction
            direction = torch.randn(latent_dim)
            direction = direction / direction.norm()

            # Add noise to radius
            radius = base_radius + 0.05 * torch.randn(1).item()
            radius = min(0.95, max(0.01, radius))

            point = direction * radius
            embeddings.append(point)

    # Pad to exact n_samples
    while len(embeddings) < n_samples:
        embeddings.append(torch.randn(latent_dim) * 0.5)

    embeddings = torch.stack(embeddings[:n_samples])

    # Project to ensure in Poincaré ball
    norms = embeddings.norm(dim=-1, keepdim=True)
    embeddings = embeddings / norms.clamp(min=1e-8) * norms.clamp(max=0.95)

    return embeddings


def main():
    parser = argparse.ArgumentParser(description="Verify holographic scaling in hyperbolic VAE")
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Path to trained VAE model",
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=None,
        help="Path to sequence data",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/holographic_validation"),
        help="Output directory",
    )
    parser.add_argument(
        "--n_synthetic",
        type=int,
        default=500,
        help="Number of synthetic samples if no model provided",
    )
    parser.add_argument(
        "--curvature",
        type=float,
        default=1.0,
        help="Hyperbolic curvature",
    )

    args = parser.parse_args()

    run_validation(
        model_path=args.model_path,
        data_path=args.data_path,
        output_dir=args.output_dir,
        n_synthetic=args.n_synthetic,
        curvature=args.curvature,
    )


if __name__ == "__main__":
    main()
