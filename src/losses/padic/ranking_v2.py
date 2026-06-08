# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Enhanced p-Adic Ranking Loss with Hard Negative Mining (v5.8).

This module implements the enhanced ranking loss with:
1. Hard negative mining: Focus on triplets that violate the ranking
2. Hierarchical margin: Scale margin by valuation difference
3. Semi-hard strategy: Select negatives that are close but wrong

Single responsibility: Enhanced Euclidean ranking with hard negatives.
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
)
from src.geometry import poincare_distance

from .triplet_mining import EuclideanTripletMiner


class PAdicRankingLossV2(nn.Module):
    """Enhanced p-Adic Ranking Loss with Hard Negative Mining and Hierarchical Margin.

    Key improvements over PAdicRankingLoss:
    1. Hard negative mining: Focus on triplets that violate the ranking
    2. Hierarchical margin: Scale margin by valuation difference
    3. Semi-hard strategy: Select negatives that are close but wrong

    This addresses two bottlenecks:
    - Random sampling misses the hardest (most informative) triplets
    - Fixed margin ignores the multi-scale nature of 3-adic distances
    """

    def __init__(
        self,
        base_margin: float = DEFAULT_MARGIN_BASE,
        margin_scale: float = DEFAULT_MARGIN_SCALE,
        n_triplets: int = DEFAULT_N_TRIPLETS,
        hard_negative_ratio: float = DEFAULT_HARD_NEGATIVE_RATIO,
        semi_hard: bool = True,
        use_hyperbolic: bool = True,
        curvature: float = 1.0,
    ):
        """Initialize Enhanced p-Adic Ranking Loss.

        Args:
            base_margin: Minimum margin for all triplets
            margin_scale: Scale factor for valuation-based margin adjustment
            n_triplets: Number of triplets to sample per batch
            hard_negative_ratio: Fraction of triplets that should be hard negatives
            semi_hard: If True, use semi-hard negatives (close but wrong ordering)
            use_hyperbolic: V5.12.2 - Use poincare_distance for hyperbolic embeddings (default True)
            curvature: Hyperbolic curvature for poincare_distance
        """
        super().__init__()
        self.base_margin = base_margin
        self.margin_scale = margin_scale
        self.n_triplets = n_triplets
        self.hard_negative_ratio = hard_negative_ratio
        self.semi_hard = semi_hard
        self.use_hyperbolic = use_hyperbolic
        self.curvature = curvature

        # Reuse triplet mining logic from base class
        self.miner = EuclideanTripletMiner(
            base_margin=base_margin,
            margin_scale=margin_scale,
            n_triplets=n_triplets,
            hard_negative_ratio=hard_negative_ratio,
            semi_hard=semi_hard,
        )

    def forward(self, z: torch.Tensor, batch_indices: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute enhanced p-Adic ranking loss.

        Args:
            z: Latent codes (batch_size, latent_dim)
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
                "mean_margin": 0.0,
                "total_triplets": 0,
            }

        # Mine triplets using shared logic
        triplets = self.miner.mine_triplets(z, batch_indices)

        if triplets.is_empty():
            return torch.tensor(0.0, device=device), {
                "hard_ratio": 0.0,
                "violations": 0,
                "mean_margin": 0.0,
                "total_triplets": 0,
            }

        # Compute hierarchical margin
        hierarchical_margin = self.miner.compute_hierarchical_margin(triplets.v_pos, triplets.v_neg)

        # V5.12.2: Compute latent distances (Euclidean or Hyperbolic)
        if self.use_hyperbolic:
            d_anchor_pos = poincare_distance(z[triplets.anchor_idx], z[triplets.pos_idx], c=self.curvature)
            d_anchor_neg = poincare_distance(z[triplets.anchor_idx], z[triplets.neg_idx], c=self.curvature)
        else:
            d_anchor_pos = torch.norm(z[triplets.anchor_idx] - z[triplets.pos_idx], dim=1)
            d_anchor_neg = torch.norm(z[triplets.anchor_idx] - z[triplets.neg_idx], dim=1)

        # Triplet loss with hierarchical margin
        # We want d_pos < d_neg (positive is 3-adically closer)
        violations = d_anchor_pos - d_anchor_neg + hierarchical_margin
        loss = F.relu(violations).mean()

        # Compute metrics
        n_violations = (violations > 0).sum().item()
        n_hard = int(len(triplets) * self.hard_negative_ratio)
        actual_hard_ratio = n_hard / max(len(triplets), 1)

        metrics = {
            "hard_ratio": actual_hard_ratio,
            "violations": n_violations,
            "mean_margin": hierarchical_margin.mean().item(),
            "total_triplets": len(triplets),
        }

        return loss, metrics


__all__ = ["PAdicRankingLossV2"]
