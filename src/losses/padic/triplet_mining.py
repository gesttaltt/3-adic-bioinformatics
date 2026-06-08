# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Triplet mining utilities for p-adic ranking losses.

This module provides shared triplet mining logic used by both Euclidean
and Hyperbolic ranking losses. It eliminates code duplication between
PAdicRankingLossV2 and PAdicRankingLossHyperbolic.

Single responsibility: Triplet sampling and hard negative mining.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from src.config.constants import (
    DEFAULT_HARD_NEGATIVE_RATIO,
    DEFAULT_MARGIN_BASE,
    DEFAULT_MARGIN_SCALE,
    DEFAULT_N_TRIPLETS,
)
from src.core import TERNARY


@dataclass
class TripletBatch:
    """Container for a batch of triplets with metadata.

    Attributes:
        anchor_idx: Indices of anchor samples
        pos_idx: Indices of positive samples (3-adically closer to anchor)
        neg_idx: Indices of negative samples (3-adically farther from anchor)
        v_pos: 3-adic valuations for anchor-positive pairs
        v_neg: 3-adic valuations for anchor-negative pairs
    """

    anchor_idx: torch.Tensor
    pos_idx: torch.Tensor
    neg_idx: torch.Tensor
    v_pos: torch.Tensor
    v_neg: torch.Tensor

    def __len__(self) -> int:
        return self.anchor_idx.size(0)

    def is_empty(self) -> bool:
        return len(self) == 0

    @classmethod
    def empty(cls, device: torch.device) -> TripletBatch:
        """Create an empty triplet batch."""
        return cls(
            anchor_idx=torch.tensor([], dtype=torch.long, device=device),
            pos_idx=torch.tensor([], dtype=torch.long, device=device),
            neg_idx=torch.tensor([], dtype=torch.long, device=device),
            v_pos=torch.tensor([], device=device),
            v_neg=torch.tensor([], device=device),
        )

    def to_tuple(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert to tuple for backward compatibility."""
        return self.anchor_idx, self.pos_idx, self.neg_idx, self.v_pos, self.v_neg

    @classmethod
    def concat(cls, batches: list[TripletBatch]) -> TripletBatch:
        """Concatenate multiple triplet batches."""
        if not batches:
            raise ValueError("Cannot concatenate empty list of batches")

        device = batches[0].anchor_idx.device
        non_empty = [b for b in batches if not b.is_empty()]

        if not non_empty:
            return cls.empty(device)

        return cls(
            anchor_idx=torch.cat([b.anchor_idx for b in non_empty]),
            pos_idx=torch.cat([b.pos_idx for b in non_empty]),
            neg_idx=torch.cat([b.neg_idx for b in non_empty]),
            v_pos=torch.cat([b.v_pos for b in non_empty]),
            v_neg=torch.cat([b.v_neg for b in non_empty]),
        )


def compute_3adic_valuation_batch(idx_i: torch.Tensor, idx_j: torch.Tensor) -> torch.Tensor:
    """Compute 3-adic valuations for batches of index pairs.

    The valuation v_3(|i-j|) = max k such that 3^k divides |i-j|.
    Delegates to TERNARY.valuation() for O(1) lookups.

    Args:
        idx_i: First indices (batch,)
        idx_j: Second indices (batch,)

    Returns:
        3-adic valuations (batch,) - floats from 0 to MAX_VALUATION
    """
    diff = torch.abs(idx_i.long() - idx_j.long())
    return TERNARY.valuation(diff).float()


class TripletMiner(ABC):
    """Abstract base class for triplet mining strategies.

    Subclasses must implement the distance computation method,
    allowing for Euclidean, Poincaré, or other distance metrics.
    """

    def __init__(
        self,
        base_margin: float = DEFAULT_MARGIN_BASE,
        margin_scale: float = DEFAULT_MARGIN_SCALE,
        n_triplets: int = DEFAULT_N_TRIPLETS,
        hard_negative_ratio: float = DEFAULT_HARD_NEGATIVE_RATIO,
        semi_hard: bool = True,
    ):
        """Initialize triplet miner.

        Args:
            base_margin: Minimum margin for all triplets
            margin_scale: Scale factor for valuation-based margin adjustment
            n_triplets: Number of triplets to sample per batch
            hard_negative_ratio: Fraction of triplets from hard negative mining
            semi_hard: If True, use semi-hard negatives (close but wrong)
        """
        self.base_margin = base_margin
        self.margin_scale = margin_scale
        self.n_triplets = n_triplets
        self.hard_negative_ratio = hard_negative_ratio
        self.semi_hard = semi_hard

    @abstractmethod
    def compute_distance_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distance matrix.

        Args:
            z: Latent representations (batch_size, latent_dim)

        Returns:
            Distance matrix (batch_size, batch_size)
        """
        pass

    def compute_hierarchical_margin(self, v_pos: torch.Tensor, v_neg: torch.Tensor) -> torch.Tensor:
        """Compute hierarchical margin based on valuation difference.

        Larger valuation difference = easier distinction = larger margin.

        Args:
            v_pos: Valuations for positive pairs
            v_neg: Valuations for negative pairs

        Returns:
            Margin tensor
        """
        v_diff = torch.abs(v_pos - v_neg)
        return self.base_margin + self.margin_scale * v_diff

    def mine_triplets(self, z: torch.Tensor, batch_indices: torch.Tensor) -> TripletBatch:
        """Mine triplets combining hard negatives and random sampling.

        Args:
            z: Latent representations (batch_size, latent_dim)
            batch_indices: Operation indices for each sample

        Returns:
            TripletBatch with all mined triplets
        """
        batch_size = z.size(0)
        device = z.device

        if batch_size < 3:
            return TripletBatch.empty(device)

        n_triplets = min(self.n_triplets, batch_size)
        n_hard = int(n_triplets * self.hard_negative_ratio)
        n_random = n_triplets - n_hard

        batches = []

        # Hard negative mining
        if n_hard > 0:
            hard_batch = self._mine_hard_negatives(z, batch_indices, n_hard)
            if not hard_batch.is_empty():
                batches.append(hard_batch)

        # Random triplets for diversity
        if n_random > 0:
            random_batch = self._sample_random_triplets(batch_indices, n_random, device)
            if not random_batch.is_empty():
                batches.append(random_batch)

        if not batches:
            return TripletBatch.empty(device)

        return TripletBatch.concat(batches)

    def _mine_hard_negatives(self, z: torch.Tensor, batch_indices: torch.Tensor, n_hard: int) -> TripletBatch:
        """Mine hard negative triplets.

        Hard negatives are pairs where:
        - pos is 3-adically closer than neg (v_pos > v_neg)
        - BUT distance violates this (d_pos >= d_neg)

        Semi-hard negatives additionally require:
        - d_neg < d_pos + margin (negative is within margin)
        """
        batch_size = z.size(0)
        device = z.device

        # Sample candidate anchors
        n_candidates = min(batch_size, n_hard * 4)
        anchor_candidates = torch.randint(0, batch_size, (n_candidates,), device=device)

        # Compute distance matrix (Euclidean or Poincaré depending on subclass)
        with torch.no_grad():
            d_matrix = self.compute_distance_matrix(z)

        hard_anchors = []
        hard_pos = []
        hard_neg = []
        hard_v_pos = []
        hard_v_neg = []

        for anchor in anchor_candidates:
            if len(hard_anchors) >= n_hard:
                break

            anchor_idx_val = batch_indices[anchor]

            # Compute 3-adic valuations from anchor to all others (vectorized)
            v_to_all = compute_3adic_valuation_batch(anchor_idx_val.expand(batch_size), batch_indices)

            # Sort by valuation (high valuation = 3-adically close)
            v_sorted_idx = torch.argsort(v_to_all, descending=True)
            v_sorted_idx = v_sorted_idx[v_sorted_idx != anchor]

            if len(v_sorted_idx) < 2:
                continue

            # Top half are potential positives, bottom half are negatives
            n_half = len(v_sorted_idx) // 2
            pos_candidates = v_sorted_idx[: max(n_half, 1)]
            neg_candidates = v_sorted_idx[max(n_half, 1) :]

            if len(neg_candidates) == 0:
                continue

            # Find hard negatives
            for pos in pos_candidates[:3]:  # Check top 3 positives
                d_ap = d_matrix[anchor, pos]
                v_pos_val = v_to_all[pos]

                for neg in neg_candidates:
                    d_an = d_matrix[anchor, neg]
                    v_neg_val = v_to_all[neg]

                    # Must have v_pos > v_neg (pos is 3-adically closer)
                    if v_pos_val <= v_neg_val:
                        continue

                    # Check for violation or semi-hard condition
                    margin = self.base_margin + self.margin_scale * (v_pos_val - v_neg_val)
                    is_hard = d_an < d_ap + margin if self.semi_hard else d_an <= d_ap

                    if is_hard:
                        hard_anchors.append(anchor)
                        hard_pos.append(pos)
                        hard_neg.append(neg)
                        hard_v_pos.append(v_pos_val)
                        hard_v_neg.append(v_neg_val)

                        if len(hard_anchors) >= n_hard:
                            break
                if len(hard_anchors) >= n_hard:
                    break

        if len(hard_anchors) == 0:
            return TripletBatch.empty(device)

        return TripletBatch(
            anchor_idx=torch.stack(hard_anchors),
            pos_idx=torch.stack(hard_pos),
            neg_idx=torch.stack(hard_neg),
            v_pos=torch.stack(hard_v_pos),
            v_neg=torch.stack(hard_v_neg),
        )

    def _sample_random_triplets(self, batch_indices: torch.Tensor, n_random: int, device: torch.device) -> TripletBatch:
        """Sample random valid triplets for diversity."""
        batch_size = batch_indices.size(0)

        # Over-sample to account for filtering
        anchor_idx = torch.randint(0, batch_size, (n_random * 2,), device=device)
        pos_idx = torch.randint(0, batch_size, (n_random * 2,), device=device)
        neg_idx = torch.randint(0, batch_size, (n_random * 2,), device=device)

        # Filter degenerate triplets
        valid = (anchor_idx != pos_idx) & (anchor_idx != neg_idx) & (pos_idx != neg_idx)
        anchor_idx = anchor_idx[valid][:n_random]
        pos_idx = pos_idx[valid][:n_random]
        neg_idx = neg_idx[valid][:n_random]

        if len(anchor_idx) == 0:
            return TripletBatch.empty(device)

        # Compute valuations
        v_pos = compute_3adic_valuation_batch(batch_indices[anchor_idx], batch_indices[pos_idx])
        v_neg = compute_3adic_valuation_batch(batch_indices[anchor_idx], batch_indices[neg_idx])

        # Filter to valid triplets (pos is 3-adically closer)
        valid_order = v_pos > v_neg
        anchor_idx = anchor_idx[valid_order]
        pos_idx = pos_idx[valid_order]
        neg_idx = neg_idx[valid_order]
        v_pos = v_pos[valid_order]
        v_neg = v_neg[valid_order]

        return TripletBatch(
            anchor_idx=anchor_idx,
            pos_idx=pos_idx,
            neg_idx=neg_idx,
            v_pos=v_pos,
            v_neg=v_neg,
        )


class EuclideanTripletMiner(TripletMiner):
    """Triplet miner using Euclidean distances."""

    def compute_distance_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean pairwise distance matrix."""
        return torch.cdist(z, z, p=2)


class HyperbolicTripletMiner(TripletMiner):
    """Triplet miner using Poincaré distances.

    Uses the geometry module for proper hyperbolic distance computation.
    """

    def __init__(
        self,
        curvature: float = 1.0,
        max_norm: float = 0.95,
        **kwargs,
    ):
        """Initialize hyperbolic triplet miner.

        Args:
            curvature: Hyperbolic curvature parameter
            max_norm: Maximum norm for Poincaré ball projection
            **kwargs: Arguments passed to TripletMiner base class
        """
        super().__init__(**kwargs)
        self.curvature = curvature
        self.max_norm = max_norm

    def compute_distance_matrix(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Poincaré pairwise distance matrix.

        Assumes z is already projected to Poincaré ball.
        """
        from src.geometry import poincare_distance_matrix

        return poincare_distance_matrix(z, c=self.curvature)


__all__ = [
    "TripletBatch",
    "TripletMiner",
    "EuclideanTripletMiner",
    "HyperbolicTripletMiner",
    "compute_3adic_valuation_batch",
]
