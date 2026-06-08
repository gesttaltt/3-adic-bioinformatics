# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Hyperbolic geometry metrics for 3-adic structure evaluation.

This module provides metrics for evaluating how well the learned latent space
preserves 3-adic (p-adic with p=3) hierarchical structure using hyperbolic
(Poincare ball) geometry.

Key metrics:
- Comprehensive metrics: Full evaluation matching training script outputs
  (coverage, hierarchy, richness, dist_corr, r_v0/r_v9, etc.)
- Triplet ranking correlation: Concordance between 3-adic and Poincare distance orderings
- Mean radius: Distribution of points in hyperbolic space

Single responsibility: Hyperbolic geometry evaluation metrics only.

Note: Uses geoopt backend when available for numerical stability.
"""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import spearmanr

# Use geoopt-backed geometry module
from src.geometry import poincare_distance, project_to_poincare


def compute_3adic_valuation(diff: torch.Tensor, max_depth: int = 10) -> torch.Tensor:
    """Compute 3-adic valuation of integer differences.

    The 3-adic valuation v_3(n) counts the highest power of 3 that divides n.
    Larger valuations indicate "closer" elements in 3-adic topology.

    Args:
        diff: Absolute differences between indices, shape (n_pairs,)
        max_depth: Maximum valuation depth to compute

    Returns:
        3-adic valuations, shape (n_pairs,)
    """
    val = torch.zeros_like(diff, dtype=torch.float32)
    remaining = diff.clone()

    for _ in range(max_depth):
        mask = (remaining % 3 == 0) & (remaining > 0)
        val[mask] += 1
        remaining[mask] = remaining[mask] // 3

    # Identical elements have "infinite" valuation (use max_depth as proxy)
    val[diff == 0] = float(max_depth)

    return val


@dataclass
class ComprehensiveMetrics:
    """Full metrics matching training script outputs.

    These metrics match what training scripts compute and store in checkpoints:
    - coverage: Percentage of operations perfectly reconstructed
    - hierarchy_A/B: Spearman(valuation, radius) - negative is correct ordering
    - richness: Average within-valuation-level variance (geometric diversity)
    - dist_corr_A/B: Spearman of pairwise distances (embedding vs valuation)
    - r_v0, r_v9: Mean radius for valuation 0 and 9 samples
    - mean_radius_A/B, std_radius_A/B: Radius statistics
    - Q: Composite metric = dist_corr + 1.5 * |hierarchy|
    """

    # Coverage (reconstruction accuracy)
    coverage: float

    # Hierarchy - Spearman correlation of valuation vs radius
    # Negative = correct (v0 at outer edge, v9 at center)
    hierarchy_A: float
    hierarchy_B: float

    # Richness - within-level variance (geometric diversity)
    richness_A: float
    richness_B: float

    # Distance correlation - Spearman of pairwise distances
    dist_corr_A: float
    dist_corr_B: float

    # Radius by valuation level
    r_v0_A: float
    r_v9_A: float
    r_v0_B: float
    r_v9_B: float

    # Radius statistics
    mean_radius_A: float
    mean_radius_B: float
    std_radius_A: float
    std_radius_B: float

    # Composite Q metric
    Q_A: float
    Q_B: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for checkpoint storage."""
        return {
            "coverage": self.coverage,
            "hierarchy_A": self.hierarchy_A,
            "hierarchy_B": self.hierarchy_B,
            "radial_corr_A": self.hierarchy_A,  # Alias for compatibility
            "radial_corr_B": self.hierarchy_B,  # Alias for compatibility
            "richness_A": self.richness_A,
            "richness_B": self.richness_B,
            "dist_corr_A": self.dist_corr_A,
            "dist_corr_B": self.dist_corr_B,
            "distance_corr_A": self.dist_corr_A,  # Alias for compatibility
            "distance_corr_B": self.dist_corr_B,  # Alias for compatibility
            "r_v0_A": self.r_v0_A,
            "r_v9_A": self.r_v9_A,
            "r_v0_B": self.r_v0_B,
            "r_v9_B": self.r_v9_B,
            "radius_v0": self.r_v0_A,  # Alias for compatibility
            "radius_v9": self.r_v9_A,  # Alias for compatibility
            "mean_radius_A": self.mean_radius_A,
            "mean_radius_B": self.mean_radius_B,
            "std_radius_A": self.std_radius_A,
            "std_radius_B": self.std_radius_B,
            "Q_A": self.Q_A,
            "Q_B": self.Q_B,
        }


def compute_comprehensive_metrics(
    model: torch.nn.Module,
    device: str | torch.device,
    batch_size: int = 4096,
    dist_corr_samples: int = 1000,
    curvature: float = 1.0,
) -> ComprehensiveMetrics:
    """Compute full metrics matching training script outputs.

    This function computes ALL metrics that training scripts store in checkpoints,
    ensuring consistency between training evaluation and post-hoc analysis.

    Args:
        model: VAE model with forward(x, compute_control) returning dict with
               z_A_hyp, z_B_hyp (hyperbolic), z_A_euc, z_B_euc (Euclidean),
               and mu_A, mu_B (Euclidean means)
        device: Device to run evaluation on
        batch_size: Batch size for processing all 19683 operations
        dist_corr_samples: Number of samples for pairwise distance correlation
        curvature: Hyperbolic curvature for poincare_distance (V5.12.2)

    Returns:
        ComprehensiveMetrics dataclass with all metrics
    """
    from src.core import TERNARY
    from src.data.generation import generate_all_ternary_operations

    was_training = model.training
    model.eval()

    # Generate all operations
    all_ops_np = generate_all_ternary_operations()
    all_ops = torch.tensor(all_ops_np, dtype=torch.float32)
    indices = torch.arange(len(all_ops))
    n_samples = len(all_ops)

    # Collect embeddings and predictions
    all_radii_A = []
    all_radii_B = []
    all_z_A = []
    all_z_B = []
    all_correct = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_ops = all_ops[i : i + batch_size].to(device)

            out = model(batch_ops, compute_control=False)

            # Get hyperbolic embeddings
            z_A = out["z_A_hyp"]
            z_B = out["z_B_hyp"]

            # V5.12.2: Compute radii using hyperbolic distance
            origin_A = torch.zeros_like(z_A)
            origin_B = torch.zeros_like(z_B)
            radii_A = poincare_distance(z_A, origin_A, c=curvature).cpu().numpy()
            radii_B = poincare_distance(z_B, origin_B, c=curvature).cpu().numpy()
            all_radii_A.append(radii_A)
            all_radii_B.append(radii_B)

            # Store embeddings for distance correlation
            all_z_A.append(z_A.cpu().numpy())
            all_z_B.append(z_B.cpu().numpy())

            # Compute coverage (reconstruction accuracy)
            mu_A = out["mu_A"]
            logits = model.decoder_A(mu_A)
            preds = torch.argmax(logits, dim=-1) - 1
            correct = (preds == batch_ops.long()).float().mean(dim=1).cpu().numpy()
            all_correct.append(correct)

    # Concatenate results
    all_radii_A = np.concatenate(all_radii_A)
    all_radii_B = np.concatenate(all_radii_B)
    all_z_A = np.concatenate(all_z_A)
    all_z_B = np.concatenate(all_z_B)
    all_correct = np.concatenate(all_correct)
    valuations = TERNARY.valuation(indices).numpy()

    # === Coverage ===
    coverage = (all_correct == 1.0).mean()

    # === Hierarchy (Spearman correlation) ===
    hierarchy_A = spearmanr(valuations, all_radii_A)[0]
    hierarchy_B = spearmanr(valuations, all_radii_B)[0]

    # === Richness (within-level variance) ===
    richness_A = 0.0
    richness_B = 0.0
    for v in range(10):
        mask = valuations == v
        if mask.sum() > 1:
            richness_A += all_radii_A[mask].var()
            richness_B += all_radii_B[mask].var()
    richness_A /= 10
    richness_B /= 10

    # === Distance correlation (sampled pairwise) ===
    sample_idx = np.random.choice(n_samples, min(dist_corr_samples, n_samples), replace=False)
    z_sample_A = all_z_A[sample_idx]
    z_sample_B = all_z_B[sample_idx]
    val_sample = valuations[sample_idx]

    # Compute pairwise distances
    z_dists_A = np.sqrt(((z_sample_A[:, None] - z_sample_A[None, :]) ** 2).sum(-1))
    z_dists_B = np.sqrt(((z_sample_B[:, None] - z_sample_B[None, :]) ** 2).sum(-1))
    val_dists = np.abs(val_sample[:, None] - val_sample[None, :]).astype(float)

    # Flatten upper triangle
    triu_idx = np.triu_indices(len(sample_idx), k=1)
    z_flat_A = z_dists_A[triu_idx]
    z_flat_B = z_dists_B[triu_idx]
    val_flat = val_dists[triu_idx]

    dist_corr_A = spearmanr(z_flat_A, val_flat)[0]
    dist_corr_B = spearmanr(z_flat_B, val_flat)[0]

    # Handle NaN (can occur with degenerate embeddings)
    if np.isnan(dist_corr_A):
        dist_corr_A = 0.0
    if np.isnan(dist_corr_B):
        dist_corr_B = 0.0

    # === Radius by valuation level ===
    r_v0_A = all_radii_A[valuations == 0].mean()
    r_v0_B = all_radii_B[valuations == 0].mean()
    r_v9_A = all_radii_A[valuations == 9].mean() if (valuations == 9).any() else np.nan
    r_v9_B = all_radii_B[valuations == 9].mean() if (valuations == 9).any() else np.nan

    # === Radius statistics ===
    mean_radius_A = all_radii_A.mean()
    mean_radius_B = all_radii_B.mean()
    std_radius_A = all_radii_A.std()
    std_radius_B = all_radii_B.std()

    # === Q metric (composite) ===
    # Q = dist_corr + 1.5 * |hierarchy|
    Q_A = dist_corr_A + 1.5 * abs(hierarchy_A)
    Q_B = dist_corr_B + 1.5 * abs(hierarchy_B)

    if was_training:
        model.train()

    return ComprehensiveMetrics(
        coverage=coverage,
        hierarchy_A=hierarchy_A,
        hierarchy_B=hierarchy_B,
        richness_A=richness_A,
        richness_B=richness_B,
        dist_corr_A=dist_corr_A,
        dist_corr_B=dist_corr_B,
        r_v0_A=r_v0_A,
        r_v9_A=r_v9_A,
        r_v0_B=r_v0_B,
        r_v9_B=r_v9_B,
        mean_radius_A=mean_radius_A,
        mean_radius_B=mean_radius_B,
        std_radius_A=std_radius_A,
        std_radius_B=std_radius_B,
        Q_A=Q_A,
        Q_B=Q_B,
    )


def compute_ranking_correlation_hyperbolic(
    model: torch.nn.Module,
    device: str,
    n_samples: int = 5000,
    max_norm: float = 0.95,
    curvature: float = 1.0,
    n_triplets: int = 1000,
) -> tuple[float, float, float, float, float, float]:
    """Compute 3-adic TRIPLET ranking correlation using Poincare distance.

    NOTE: This is a triplet concordance metric, NOT the same as Spearman hierarchy.
    For Spearman hierarchy (valuation vs radius correlation), use
    `compute_comprehensive_metrics()` instead.

    Evaluates how well the learned hyperbolic embedding preserves the
    3-adic ultrametric structure by comparing pairwise distance orderings
    within triplets.

    For triplets (i, j, k), checks if:
    - 3-adic says j is closer to i than k → v_3(|i-j|) > v_3(|i-k|)
    - Hyperbolic says j is closer to i than k → d_hyp(z_i, z_j) < d_hyp(z_i, z_k)

    Concordance rate measures how often these orderings agree.

    Args:
        model: VAE model with forward(x, compute_control) returning dict with
               z_A_hyp, z_B_hyp (hyperbolic) and z_A_euc, z_B_euc (Euclidean)
        device: Device to run evaluation on
        n_samples: Number of samples to generate
        max_norm: Maximum norm for Poincare projection (unused, model handles projection)
        curvature: Hyperbolic curvature parameter
        n_triplets: Number of triplets to evaluate

    Returns:
        Tuple of (corr_A_hyp, corr_B_hyp, corr_A_euc, corr_B_euc, mean_radius_A, mean_radius_B)
        - corr_*_hyp: Hyperbolic correlation for VAE-A/B
        - corr_*_euc: Euclidean correlation for VAE-A/B (for comparison)
        - mean_radius_*: Mean radius in Poincare ball for VAE-A/B
    """
    was_training = model.training
    model.eval()

    with torch.no_grad():
        # Generate random operation indices
        indices = torch.randint(0, 19683, (n_samples,), device=device)

        # Convert indices to ternary representation
        ternary_data = torch.zeros(n_samples, 9, device=device)
        for i in range(9):
            ternary_data[:, i] = ((indices // (3**i)) % 3) - 1

        # Forward pass through model (V5.11+ API)
        outputs = model(ternary_data.float(), compute_control=False)

        # Get hyperbolic embeddings (already projected by model)
        z_A_hyp = outputs["z_A_hyp"]
        z_B_hyp = outputs["z_B_hyp"]

        # Get Euclidean embeddings for comparison
        z_A_euc = outputs["z_A_euc"]
        z_B_euc = outputs["z_B_euc"]

        # V5.12.2: Compute mean radius using hyperbolic distance (for homeostatic monitoring)
        origin = torch.zeros_like(z_A_hyp)
        mean_radius_A = poincare_distance(z_A_hyp, origin, c=curvature).mean().item()
        mean_radius_B = poincare_distance(z_B_hyp, origin, c=curvature).mean().item()

        # Sample triplet indices
        i_idx = torch.randint(n_samples, (n_triplets,), device=device)
        j_idx = torch.randint(n_samples, (n_triplets,), device=device)
        k_idx = torch.randint(n_samples, (n_triplets,), device=device)

        # Filter to distinct triplets
        valid = (i_idx != j_idx) & (i_idx != k_idx) & (j_idx != k_idx)
        i_idx, j_idx, k_idx = i_idx[valid], j_idx[valid], k_idx[valid]

        if len(i_idx) < 100:
            # Not enough valid triplets
            if was_training:
                model.train()
            return 0.5, 0.5, 0.5, 0.5, mean_radius_A, mean_radius_B

        # Compute 3-adic valuations for pairs
        diff_ij = torch.abs(indices[i_idx] - indices[j_idx])
        diff_ik = torch.abs(indices[i_idx] - indices[k_idx])

        v_ij = compute_3adic_valuation(diff_ij)
        v_ik = compute_3adic_valuation(diff_ik)

        # 3-adic ordering: larger valuation = smaller distance
        padic_closer_ij = (v_ij > v_ik).float()

        # Poincare distances
        d_A_ij = poincare_distance(z_A_hyp[i_idx], z_A_hyp[j_idx], curvature)
        d_A_ik = poincare_distance(z_A_hyp[i_idx], z_A_hyp[k_idx], curvature)
        d_B_ij = poincare_distance(z_B_hyp[i_idx], z_B_hyp[j_idx], curvature)
        d_B_ik = poincare_distance(z_B_hyp[i_idx], z_B_hyp[k_idx], curvature)

        # Euclidean distances (for comparison)
        d_A_ij_euc = torch.norm(z_A_euc[i_idx] - z_A_euc[j_idx], dim=1)
        d_A_ik_euc = torch.norm(z_A_euc[i_idx] - z_A_euc[k_idx], dim=1)
        d_B_ij_euc = torch.norm(z_B_euc[i_idx] - z_B_euc[j_idx], dim=1)
        d_B_ik_euc = torch.norm(z_B_euc[i_idx] - z_B_euc[k_idx], dim=1)

        # Hyperbolic correlations
        latent_A_closer_hyp = (d_A_ij < d_A_ik).float()
        latent_B_closer_hyp = (d_B_ij < d_B_ik).float()
        corr_A_hyp = (padic_closer_ij == latent_A_closer_hyp).float().mean().item()
        corr_B_hyp = (padic_closer_ij == latent_B_closer_hyp).float().mean().item()

        # Euclidean correlations
        latent_A_closer_euc = (d_A_ij_euc < d_A_ik_euc).float()
        latent_B_closer_euc = (d_B_ij_euc < d_B_ik_euc).float()
        corr_A_euc = (padic_closer_ij == latent_A_closer_euc).float().mean().item()
        corr_B_euc = (padic_closer_ij == latent_B_closer_euc).float().mean().item()

    if was_training:
        model.train()

    return (
        corr_A_hyp,
        corr_B_hyp,
        corr_A_euc,
        corr_B_euc,
        mean_radius_A,
        mean_radius_B,
    )


__all__ = [
    # Geometry utilities (re-exported from src.geometry)
    "project_to_poincare",
    "poincare_distance",
    # 3-adic valuation
    "compute_3adic_valuation",
    # Comprehensive metrics (matches training script outputs)
    "ComprehensiveMetrics",
    "compute_comprehensive_metrics",
    # Triplet ranking correlation (different from Spearman hierarchy)
    "compute_ranking_correlation_hyperbolic",
]
