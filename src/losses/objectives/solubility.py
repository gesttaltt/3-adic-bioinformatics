# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Solubility and stability objectives for multi-objective optimization.

These objectives evaluate physicochemical properties relevant to
protein expression and therapeutic stability.
"""

from __future__ import annotations

from typing import Any

import torch

from src.geometry import poincare_distance

from .base import Objective, ObjectiveResult

# Amino acid hydrophobicity scale (Kyte-Doolittle)
# Higher values = more hydrophobic
HYDROPHOBICITY = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
    "*": 0.0,  # Stop codon
}

# Amino acid isoelectric points
ISOELECTRIC_POINTS = {
    "A": 6.00,
    "R": 10.76,
    "N": 5.41,
    "D": 2.77,
    "C": 5.07,
    "Q": 5.65,
    "E": 3.22,
    "G": 5.97,
    "H": 7.59,
    "I": 6.02,
    "L": 5.98,
    "K": 9.74,
    "M": 5.74,
    "F": 5.48,
    "P": 6.30,
    "S": 5.68,
    "T": 5.60,
    "W": 5.89,
    "Y": 5.66,
    "V": 5.96,
    "*": 7.00,
}


class SolubilityObjective(Objective):
    """Objective for maximizing protein solubility.

    Proteins with extreme hydrophobicity or charged regions tend to
    aggregate. This objective penalizes sequences that are likely
    to be insoluble or form aggregates.

    Uses latent space features as proxy for sequence properties.
    """

    def __init__(
        self,
        target_hydrophobicity: float = 0.0,
        aggregation_penalty: float = 2.0,
        weight: float = 1.0,
    ):
        """Initialize solubility objective.

        Args:
            target_hydrophobicity: Optimal hydrophobicity score
            aggregation_penalty: Penalty for aggregation-prone regions
            weight: Weight for multi-objective combination
        """
        super().__init__(name="solubility", weight=weight)
        self.target_hydrophobicity = target_hydrophobicity
        self.aggregation_penalty = aggregation_penalty

    def _estimate_hydrophobicity(self, latent: torch.Tensor) -> torch.Tensor:
        """Estimate hydrophobicity from latent representation.

        Uses variance and mean of latent dimensions as proxy.
        High variance regions often correspond to hydrophobic patches.

        Args:
            latent: Latent vectors, shape (Batch, Dim)

        Returns:
            Estimated hydrophobicity, shape (Batch,)
        """
        # Use mean as base hydrophobicity indicator
        mean_val = latent.mean(dim=-1)

        # Variance indicates disorder/aggregation potential
        variance = latent.var(dim=-1)

        # Combined hydrophobicity estimate
        hydrophobicity = mean_val + 0.5 * variance

        return hydrophobicity

    def _estimate_aggregation_propensity(self, latent: torch.Tensor) -> torch.Tensor:
        """Estimate aggregation propensity from latent space.

        Clusters of high-magnitude values suggest aggregation-prone regions.

        Args:
            latent: Latent vectors, shape (Batch, Dim)

        Returns:
            Aggregation propensity, shape (Batch,)
        """
        # High magnitude clusters indicate hydrophobic patches
        magnitude = torch.abs(latent)

        # Consecutive high values suggest aggregation motifs
        # Use local max pooling as proxy
        if latent.shape[-1] >= 3:
            # Reshape for 1D pooling
            pooled = torch.nn.functional.max_pool1d(
                magnitude.unsqueeze(1),
                kernel_size=3,
                stride=1,
                padding=1,
            ).squeeze(1)
            aggregation = pooled.mean(dim=-1)
        else:
            aggregation = magnitude.mean(dim=-1)

        return aggregation

    def evaluate(
        self,
        latent: torch.Tensor,
        decoded: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ObjectiveResult:
        """Evaluate solubility objective.

        Args:
            latent: Latent space vectors, shape (Batch, Dim)
            decoded: Decoded sequences (optional, for direct analysis)
            **kwargs: Additional arguments

        Returns:
            ObjectiveResult with solubility scores (lower = more soluble)
        """
        # Estimate properties from latent space
        hydrophobicity = self._estimate_hydrophobicity(latent)
        aggregation = self._estimate_aggregation_propensity(latent)

        # Deviation from target hydrophobicity
        hydro_deviation = torch.abs(hydrophobicity - self.target_hydrophobicity)

        # Combined insolubility score
        # Lower = better solubility
        scores = hydro_deviation + self.aggregation_penalty * aggregation

        return ObjectiveResult(
            score=scores,
            name=self.name,
            metadata={
                "mean_hydrophobicity": hydrophobicity.mean().item(),
                "mean_aggregation": aggregation.mean().item(),
            },
        )


class StabilityObjective(Objective):
    """Objective for maximizing thermodynamic stability.

    Stable proteins have:
    - Well-packed hydrophobic cores
    - Balanced secondary structure propensity
    - Optimal number of stabilizing interactions

    Uses latent space geometry as proxy for stability.
    """

    def __init__(
        self,
        target_compactness: float = 0.8,
        entropy_weight: float = 0.5,
        weight: float = 1.0,
        use_hyperbolic: bool = True,
        curvature: float = 1.0,
    ):
        """Initialize stability objective.

        Args:
            target_compactness: Target latent space compactness
            entropy_weight: Weight for structural entropy term
            weight: Weight for multi-objective combination
            use_hyperbolic: V5.12.2 - Use poincare_distance for hyperbolic embeddings (default True)
            curvature: Hyperbolic curvature for poincare_distance
        """
        super().__init__(name="stability", weight=weight)
        self.target_compactness = target_compactness
        self.entropy_weight = entropy_weight
        self.use_hyperbolic = use_hyperbolic
        self.curvature = curvature

    def _compute_compactness(self, latent: torch.Tensor) -> torch.Tensor:
        """Compute compactness of latent representation.

        Compact representations suggest well-folded structures.

        Args:
            latent: Latent vectors, shape (Batch, Dim)

        Returns:
            Compactness scores, shape (Batch,)
        """
        # V5.12.2: Use hyperbolic distance from origin for Poincaré ball embeddings
        if self.use_hyperbolic:
            origin = torch.zeros_like(latent)
            norms = poincare_distance(latent, origin, c=self.curvature)
        else:
            # Euclidean L2 norm as compactness measure
            norms = torch.norm(latent, dim=-1)

        # Normalize to [0, 1] range
        max_norm = norms.max() + 1e-8
        compactness = norms / max_norm

        return compactness

    def _compute_structural_entropy(self, latent: torch.Tensor) -> torch.Tensor:
        """Compute entropy of latent distribution.

        Low entropy suggests ordered/stable structure.
        High entropy suggests disordered/unstable.

        Args:
            latent: Latent vectors, shape (Batch, Dim)

        Returns:
            Entropy scores, shape (Batch,)
        """
        # Softmax over dimensions as probability distribution
        probs = torch.softmax(latent, dim=-1)

        # Shannon entropy
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)

        # Normalize by max entropy (uniform distribution)
        max_entropy = torch.log(torch.tensor(latent.shape[-1], dtype=torch.float, device=latent.device))
        normalized_entropy = entropy / max_entropy

        return normalized_entropy

    def evaluate(
        self,
        latent: torch.Tensor,
        decoded: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ObjectiveResult:
        """Evaluate stability objective.

        Args:
            latent: Latent space vectors, shape (Batch, Dim)
            decoded: Decoded sequences (optional)
            **kwargs: Additional arguments

        Returns:
            ObjectiveResult with stability scores (lower = more stable)
        """
        compactness = self._compute_compactness(latent)
        entropy = self._compute_structural_entropy(latent)

        # Deviation from target compactness
        compactness_deviation = torch.abs(compactness - self.target_compactness)

        # Combined instability score
        # Lower = more stable
        scores = compactness_deviation + self.entropy_weight * entropy

        return ObjectiveResult(
            score=scores,
            name=self.name,
            metadata={
                "mean_compactness": compactness.mean().item(),
                "mean_entropy": entropy.mean().item(),
            },
        )
