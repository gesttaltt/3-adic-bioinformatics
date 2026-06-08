# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Manufacturability objectives for multi-objective optimization.

These objectives evaluate practical considerations for therapeutic
production including expression yield, purification ease, and cost.
"""

from __future__ import annotations

from typing import Any

import torch

from .base import Objective, ObjectiveResult

# Codon adaptation index weights for E. coli expression
# Higher values = more efficiently expressed in E. coli
ECOLI_CODON_PREFERENCE = {
    # Frequently used codons in E. coli
    0: 0.8,  # TTT
    1: 0.9,  # TTC (preferred)
    19: 1.0,  # CTG (most used for Leu)
    # ... simplified for demonstration
}


class ManufacturabilityObjective(Objective):
    """Objective for maximizing manufacturability score.

    Evaluates practical considerations:
    - Expression system compatibility
    - Purification tag accessibility
    - Post-translational modification sites
    - Proteolytic stability
    """

    def __init__(
        self,
        expression_system: str = "ecoli",
        target_length_range: tuple = (100, 500),
        complexity_penalty: float = 1.0,
        weight: float = 1.0,
    ):
        """Initialize manufacturability objective.

        Args:
            expression_system: Target expression system ('ecoli', 'yeast', 'mammalian')
            target_length_range: Optimal sequence length (min, max)
            complexity_penalty: Penalty for complex sequences
            weight: Weight for multi-objective combination
        """
        super().__init__(name="manufacturability", weight=weight)
        self.expression_system = expression_system
        self.target_length_range = target_length_range
        self.complexity_penalty = complexity_penalty

    def _compute_expression_score(self, latent: torch.Tensor) -> torch.Tensor:
        """Estimate expression efficiency from latent space.

        Well-expressed proteins tend to have:
        - Moderate complexity
        - Balanced amino acid composition
        - No rare codon clusters

        Args:
            latent: Latent vectors, shape (Batch, Dim)

        Returns:
            Expression difficulty scores, shape (Batch,)
        """
        # Use standard deviation as complexity proxy
        complexity = latent.std(dim=-1)

        # Extreme values indicate expression problems
        extreme_values = (torch.abs(latent) > 2.0).float().mean(dim=-1)

        # Combined expression difficulty (lower = easier to express)
        difficulty = complexity + 2.0 * extreme_values

        return difficulty

    def _compute_purification_score(self, latent: torch.Tensor) -> torch.Tensor:
        """Estimate purification difficulty from latent space.

        Easy-to-purify proteins have:
        - Distinct from host proteins
        - Stable under purification conditions
        - Accessible N/C-termini

        Args:
            latent: Latent vectors, shape (Batch, Dim)

        Returns:
            Purification difficulty, shape (Batch,)
        """
        # Use first and last dimensions as terminus accessibility proxy
        if latent.shape[-1] >= 4:
            n_term = torch.abs(latent[:, :2]).mean(dim=-1)
            c_term = torch.abs(latent[:, -2:]).mean(dim=-1)
            terminus_accessibility = (n_term + c_term) / 2
        else:
            terminus_accessibility = torch.abs(latent).mean(dim=-1)

        # Lower accessibility = harder to purify
        difficulty = 1.0 - torch.clamp(terminus_accessibility, 0, 1)

        return difficulty

    def evaluate(
        self,
        latent: torch.Tensor,
        decoded: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ObjectiveResult:
        """Evaluate manufacturability objective.

        Args:
            latent: Latent space vectors, shape (Batch, Dim)
            decoded: Decoded sequences (optional)
            **kwargs: Additional arguments

        Returns:
            ObjectiveResult with manufacturability scores (lower = easier to manufacture)
        """
        expression_difficulty = self._compute_expression_score(latent)
        purification_difficulty = self._compute_purification_score(latent)

        # Sequence complexity penalty
        complexity = latent.std(dim=-1) * self.complexity_penalty

        # Combined manufacturing difficulty
        scores = expression_difficulty + purification_difficulty + complexity

        return ObjectiveResult(
            score=scores,
            name=self.name,
            metadata={
                "expression_system": self.expression_system,
                "mean_expression_difficulty": expression_difficulty.mean().item(),
                "mean_purification_difficulty": purification_difficulty.mean().item(),
            },
        )


class ProductionCostObjective(Objective):
    """Objective for minimizing production cost.

    Estimates relative cost based on:
    - Sequence length (longer = more expensive)
    - Rare amino acid content
    - Special handling requirements
    - Yield predictions
    """

    def __init__(
        self,
        base_cost_per_residue: float = 1.0,
        rare_aa_penalty: float = 2.0,
        ptm_cost: float = 5.0,
        weight: float = 1.0,
    ):
        """Initialize production cost objective.

        Args:
            base_cost_per_residue: Base cost per amino acid
            rare_aa_penalty: Multiplier for rare amino acids
            ptm_cost: Cost for post-translational modifications
            weight: Weight for multi-objective combination
        """
        super().__init__(name="production_cost", weight=weight)
        self.base_cost_per_residue = base_cost_per_residue
        self.rare_aa_penalty = rare_aa_penalty
        self.ptm_cost = ptm_cost

    def _estimate_length(self, latent: torch.Tensor) -> torch.Tensor:
        """Estimate sequence length from latent representation.

        Larger norm often correlates with longer sequences.

        Args:
            latent: Latent vectors, shape (Batch, Dim)

        Returns:
            Estimated relative length, shape (Batch,)
        """
        # Use L1 norm as length proxy
        return torch.norm(latent, p=1, dim=-1)

    def _estimate_rare_content(self, latent: torch.Tensor) -> torch.Tensor:
        """Estimate rare amino acid content.

        Extreme latent values may correspond to rare residues.

        Args:
            latent: Latent vectors, shape (Batch, Dim)

        Returns:
            Rare content fraction, shape (Batch,)
        """
        # Count dimensions with extreme values
        rare_mask = torch.abs(latent) > 1.5
        rare_fraction = rare_mask.float().mean(dim=-1)
        return rare_fraction

    def _estimate_ptm_sites(self, latent: torch.Tensor) -> torch.Tensor:
        """Estimate number of PTM sites.

        Specific patterns in latent space may indicate PTM-prone regions.

        Args:
            latent: Latent vectors, shape (Batch, Dim)

        Returns:
            Estimated PTM count, shape (Batch,)
        """
        # Use sign changes as PTM site proxy
        if latent.shape[-1] >= 2:
            signs = torch.sign(latent)
            sign_changes = (signs[:, 1:] != signs[:, :-1]).float().sum(dim=-1)
            ptm_estimate = sign_changes / latent.shape[-1]
        else:
            ptm_estimate = torch.zeros(latent.shape[0], device=latent.device)

        return ptm_estimate

    def evaluate(
        self,
        latent: torch.Tensor,
        decoded: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ObjectiveResult:
        """Evaluate production cost objective.

        Args:
            latent: Latent space vectors, shape (Batch, Dim)
            decoded: Decoded sequences (optional)
            **kwargs: Additional arguments

        Returns:
            ObjectiveResult with cost scores (lower = cheaper)
        """
        length_estimate = self._estimate_length(latent)
        rare_content = self._estimate_rare_content(latent)
        ptm_sites = self._estimate_ptm_sites(latent)

        # Normalize length estimate
        length_normalized = length_estimate / (latent.shape[-1] + 1e-8)

        # Combined cost estimate
        base_cost = self.base_cost_per_residue * length_normalized
        rare_cost = self.rare_aa_penalty * rare_content
        ptm_cost = self.ptm_cost * ptm_sites

        scores = base_cost + rare_cost + ptm_cost

        return ObjectiveResult(
            score=scores,
            name=self.name,
            metadata={
                "estimated_length": length_estimate.mean().item(),
                "rare_fraction": rare_content.mean().item(),
                "ptm_estimate": ptm_sites.mean().item(),
            },
        )
