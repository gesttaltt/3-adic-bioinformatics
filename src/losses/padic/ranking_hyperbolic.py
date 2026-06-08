# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Hyperbolic p-Adic Ranking Loss using Poincaré distance (v5.9).

This module implements ranking loss in hyperbolic space:
- Uses Poincaré distance instead of Euclidean
- Includes radial hierarchy enforcement
- Tree-like structure with root at origin

Single responsibility: Hyperbolic ranking with radial hierarchy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.constants import (
    DEFAULT_HARD_NEGATIVE_RATIO,
    DEFAULT_MARGIN_BASE,
    DEFAULT_MARGIN_SCALE,
    DEFAULT_N_TRIPLETS,
    DEFAULT_RADIAL_WEIGHT,
    HYPERBOLIC_CURVATURE,
    HYPERBOLIC_MAX_NORM,
    MAX_VALUATION,
)
from src.core import TERNARY
from src.geometry import poincare_distance, project_to_poincare

from .triplet_mining import HyperbolicTripletMiner


class PAdicRankingLossHyperbolic(nn.Module):
    """Hyperbolic p-Adic Ranking Loss using Poincaré distance.

    Key insight: 3-adic distances are ultrametric, and ultrametric spaces
    embed isometrically into hyperbolic space. By computing ranking loss
    using Poincaré distance instead of Euclidean, the VAE learns to arrange
    points in a hyperbolic-like structure that naturally captures tree geometry.

    The loss projects Euclidean latent points onto the Poincaré ball and
    measures distances using:
    d_poincare(x,y) = arcosh(1 + 2||x-y||² / ((1-||x||²)(1-||y||²)))

    Additionally includes radial hierarchy: points with high 3-adic valuation
    (divisible by 3^k) should be near the origin (tree root), while points
    with low valuation should be near the boundary (tree leaves).
    """

    def __init__(
        self,
        base_margin: float = DEFAULT_MARGIN_BASE,
        margin_scale: float = DEFAULT_MARGIN_SCALE,
        n_triplets: int = DEFAULT_N_TRIPLETS,
        hard_negative_ratio: float = DEFAULT_HARD_NEGATIVE_RATIO,
        curvature: float = HYPERBOLIC_CURVATURE,
        radial_weight: float = DEFAULT_RADIAL_WEIGHT,
        max_norm: float = HYPERBOLIC_MAX_NORM,
    ):
        """Initialize Hyperbolic p-Adic Ranking Loss.

        Args:
            base_margin: Minimum margin for all triplets
            margin_scale: Scale factor for valuation-based margin adjustment
            n_triplets: Number of triplets to sample per batch
            hard_negative_ratio: Fraction of triplets that should be hard negatives
            curvature: Hyperbolic curvature (higher = more curved)
            radial_weight: Weight for radial hierarchy regularization
            max_norm: Maximum norm for Poincaré ball projection (< 1)
        """
        super().__init__()
        self.base_margin = base_margin
        self.margin_scale = margin_scale
        self.n_triplets = n_triplets
        self.hard_negative_ratio = hard_negative_ratio
        self.curvature = curvature
        self.radial_weight = radial_weight
        self.max_norm = max_norm

        # Reuse triplet mining logic from base class
        self.miner = HyperbolicTripletMiner(
            curvature=curvature,
            max_norm=max_norm,
            base_margin=base_margin,
            margin_scale=margin_scale,
            n_triplets=n_triplets,
            hard_negative_ratio=hard_negative_ratio,
            semi_hard=True,
        )

    def _project_to_poincare(self, z: torch.Tensor) -> torch.Tensor:
        """Project Euclidean points onto the Poincaré ball.

        Uses geometry module for single source of truth.
        """
        return project_to_poincare(z, max_norm=self.max_norm, c=self.curvature)

    def _poincare_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Poincaré distance between points.

        Uses geometry module for numerical stability.
        """
        return poincare_distance(x, y, c=self.curvature)

    def _compute_radial_loss(self, z_hyp: torch.Tensor, batch_indices: torch.Tensor) -> torch.Tensor:
        """Compute radial hierarchy loss.

        Points with high 3-adic valuation (divisible by large powers of 3)
        should be closer to origin. This creates tree-like hierarchy where:
        - Origin = root (indices divisible by 3^9)
        - Boundary = leaves (indices not divisible by 3)
        """
        # Compute 3-adic valuation for each index
        valuations = TERNARY.valuation(batch_indices).float()

        # Normalize valuations to [0, 1]
        normalized_val = valuations / float(MAX_VALUATION)

        # Target radius: high valuation -> small radius (near center)
        target_radius = (1 - normalized_val) * self.max_norm * 0.9

        # V5.12.2: Actual radius using hyperbolic distance
        origin = torch.zeros_like(z_hyp)
        actual_radius = poincare_distance(z_hyp, origin, c=self.curvature)

        # MSE loss on radii
        radial_loss = F.mse_loss(actual_radius, target_radius)

        return radial_loss

    def forward(self, z: torch.Tensor, batch_indices: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute hyperbolic p-Adic ranking loss.

        Args:
            z: Latent codes in Euclidean space (batch_size, latent_dim)
            batch_indices: Operation indices for each sample (batch_size,)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch_size = z.size(0)
        device = z.device

        if batch_size < 3:
            return torch.tensor(0.0, device=device), {
                "hard_ratio": 0.0,
                "violations": 0,
                "poincare_dist_mean": 0.0,
                "radial_loss": 0.0,
                "ranking_loss": 0.0,
            }

        # Project to Poincaré ball
        z_hyp = self._project_to_poincare(z)

        # Radial hierarchy loss
        radial_loss = self._compute_radial_loss(z_hyp, batch_indices)

        # Mine triplets using shared hyperbolic logic
        triplets = self.miner.mine_triplets(z_hyp, batch_indices)

        if triplets.is_empty():
            total_loss = self.radial_weight * radial_loss
            return total_loss, {
                "hard_ratio": 0.0,
                "violations": 0,
                "poincare_dist_mean": 0.0,
                "radial_loss": radial_loss.item(),
                "ranking_loss": 0.0,
            }

        # Compute hierarchical margin
        hierarchical_margin = self.miner.compute_hierarchical_margin(triplets.v_pos, triplets.v_neg)

        # Compute Poincaré distances
        d_anchor_pos = self._poincare_distance(z_hyp[triplets.anchor_idx], z_hyp[triplets.pos_idx])
        d_anchor_neg = self._poincare_distance(z_hyp[triplets.anchor_idx], z_hyp[triplets.neg_idx])

        # Triplet loss with hierarchical margin
        violations = d_anchor_pos - d_anchor_neg + hierarchical_margin
        ranking_loss = F.relu(violations).mean()

        # Total loss
        total_loss = ranking_loss + self.radial_weight * radial_loss

        # Compute metrics
        n_violations = (violations > 0).sum().item()
        n_hard = int(len(triplets) * self.hard_negative_ratio)
        actual_hard_ratio = n_hard / max(len(triplets), 1)
        mean_poincare_dist = (d_anchor_pos.mean() + d_anchor_neg.mean()).item() / 2

        metrics = {
            "hard_ratio": actual_hard_ratio,
            "violations": n_violations,
            "mean_margin": hierarchical_margin.mean().item(),
            "total_triplets": len(triplets),
            "poincare_dist_mean": mean_poincare_dist,
            "radial_loss": radial_loss.item(),
            "ranking_loss": ranking_loss.item(),
        }

        return total_loss, metrics


__all__ = ["PAdicRankingLossHyperbolic"]
