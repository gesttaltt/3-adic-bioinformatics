# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Autoimmunity-aware Codon Regularization.

This module implements loss functions that penalize sequences with high
autoimmune risk, helping to design safer vaccines and therapeutics.

Key Features:
- RSCU (Relative Synonymous Codon Usage) alignment with host preferences
- Autoimmune epitope avoidance based on risk profiles
- CD4/CD8 ratio-aware regularization
- Integration with AutoimmunityLoader for risk scoring

Usage:
    from src.losses.autoimmunity import AutoimmuneCodonRegularizer

    regularizer = AutoimmuneCodonRegularizer(pathogen="hiv")
    loss = regularizer(codon_indices, latent_z)

References:
    - DOCUMENTATION/.../RESEARCH_PROPOSALS/Autoimmunity_Codon_Adaptation/proposal.md
    - COMPREHENSIVE_RESEARCH_REPORT.md - Section 2.8
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.autoimmunity import AutoimmunityLoader

# Human codon usage frequencies (normalized RSCU values)
# Source: Kazusa Codon Usage Database for Homo sapiens
HUMAN_CODON_RSCU = {
    # Phe (F)
    0: 0.46,  # TTT
    1: 0.54,  # TTC
    # Leu (L)
    2: 0.08,  # TTA
    3: 0.13,  # TTG
    16: 0.13,  # CTT
    17: 0.20,  # CTC
    18: 0.07,  # CTA
    19: 0.39,  # CTG
    # Ser (S)
    4: 0.19,  # TCT
    5: 0.22,  # TCC
    6: 0.15,  # TCA
    7: 0.05,  # TCG
    24: 0.15,  # AGT
    25: 0.24,  # AGC
    # Tyr (Y)
    8: 0.44,  # TAT
    9: 0.56,  # TAC
    # Stop (*)
    10: 0.30,  # TAA
    11: 0.24,  # TAG
    14: 0.46,  # TGA
    # Cys (C)
    12: 0.46,  # TGT
    13: 0.54,  # TGC
    # Trp (W)
    15: 1.00,  # TGG
    # Pro (P)
    20: 0.29,  # CCT
    21: 0.32,  # CCC
    22: 0.28,  # CCA
    23: 0.11,  # CCG
    # His (H)
    26: 0.42,  # CAT
    27: 0.58,  # CAC
    # Gln (Q)
    28: 0.27,  # CAA
    29: 0.73,  # CAG
    # Arg (R)
    30: 0.08,  # CGT
    31: 0.18,  # CGC
    32: 0.11,  # CGA
    33: 0.19,  # CGG
    26 + 32: 0.21,  # AGA (idx 58)
    27 + 32: 0.23,  # AGG (idx 59)
    # Ile (I)
    32 + 12: 0.36,  # ATT (idx 44)
    32 + 13: 0.47,  # ATC (idx 45)
    32 + 14: 0.17,  # ATA (idx 46)
    # Met (M)
    32 + 15: 1.00,  # ATG (idx 47)
    # Thr (T)
    32 + 16: 0.25,  # ACT (idx 48)
    32 + 17: 0.35,  # ACC (idx 49)
    32 + 18: 0.28,  # ACA (idx 50)
    32 + 19: 0.12,  # ACG (idx 51)
    # Asn (N)
    32 + 20: 0.47,  # AAT (idx 52)
    32 + 21: 0.53,  # AAC (idx 53)
    # Lys (K)
    32 + 22: 0.43,  # AAA (idx 54)
    32 + 23: 0.57,  # AAG (idx 55)
    # Val (V)
    48 + 8: 0.18,  # GTT (idx 56)
    48 + 9: 0.24,  # GTC (idx 57)
    48 + 10: 0.12,  # GTA (idx 58)
    48 + 11: 0.46,  # GTG (idx 59)
    # Ala (A)
    48 + 12: 0.27,  # GCT (idx 60)
    48 + 13: 0.40,  # GCC (idx 61)
    48 + 14: 0.23,  # GCA (idx 62)
    48 + 15: 0.10,  # GCG (idx 63)
}


def _get_human_rscu_tensor(n_codons: int = 64, device: torch.device | None = None) -> torch.Tensor:
    """Get human RSCU values as a tensor."""
    rscu = torch.zeros(n_codons, device=device)
    for idx, value in HUMAN_CODON_RSCU.items():
        if idx < n_codons:
            rscu[idx] = value
    # Fill missing values with neutral 0.5
    rscu[rscu == 0] = 0.5
    return rscu


class AutoimmuneCodonRegularizer(nn.Module):
    """Regularizer that penalizes sequences with high autoimmune risk.

    This loss function encourages the model to generate sequences that:
    1. Align with host (human) codon usage preferences
    2. Avoid known immunogenic patterns
    3. Maintain diversity to prevent low-complexity regions

    The regularizer uses the AutoimmunityLoader for risk scoring and
    adds a codon usage bias term to encourage host-compatible sequences.
    """

    def __init__(
        self,
        pathogen: str = "hiv",
        rscu_weight: float = 0.3,
        risk_weight: float = 0.7,
        diversity_weight: float = 0.2,
        target_diversity: float = 0.5,
    ):
        """Initialize the regularizer.

        Args:
            pathogen: Target pathogen for risk scoring
            rscu_weight: Weight for RSCU alignment loss
            risk_weight: Weight for autoimmune risk penalty
            diversity_weight: Weight for diversity regularization
            target_diversity: Target sequence diversity (0-1)
        """
        super().__init__()
        self.pathogen = pathogen
        self.rscu_weight = rscu_weight
        self.risk_weight = risk_weight
        self.diversity_weight = diversity_weight
        self.target_diversity = target_diversity

        self.risk_loader = AutoimmunityLoader(pathogen=pathogen)

        # Register buffers for RSCU values
        self.register_buffer("human_rscu", _get_human_rscu_tensor(64))

    def compute_rscu_loss(self, codon_logits: torch.Tensor) -> torch.Tensor:
        """Compute loss for deviating from human codon usage.

        Args:
            codon_logits: Logits for each codon position, shape (Batch, SeqLen, 64)

        Returns:
            Scalar loss encouraging human-like codon usage
        """
        # Convert logits to probabilities
        probs = torch.softmax(codon_logits, dim=-1)

        # Weight by human RSCU (higher RSCU = preferred codon)
        # We want to encourage selection of codons with high RSCU
        rscu = self.human_rscu.to(probs.device)

        # Compute expected RSCU for the predicted distribution
        expected_rscu = (probs * rscu).sum(dim=-1)

        # Loss is negative expected RSCU (higher RSCU = lower loss)
        # Normalized to [0, 1] range
        loss = 1.0 - expected_rscu.mean()

        return loss

    def compute_risk_loss(self, codon_indices: torch.Tensor) -> torch.Tensor:
        """Compute autoimmune risk penalty.

        Args:
            codon_indices: Tensor of codon indices, shape (Batch, SeqLen)

        Returns:
            Scalar loss based on autoimmune risk
        """
        risk_scores = self.risk_loader.get_batch_risk(codon_indices)

        # Convert to tensor if necessary
        if not isinstance(risk_scores, torch.Tensor):
            risk_scores = torch.tensor(risk_scores, device=codon_indices.device)

        return risk_scores.mean()

    def compute_diversity_loss(self, codon_indices: torch.Tensor) -> torch.Tensor:
        """Penalize low-complexity (repetitive) sequences.

        Args:
            codon_indices: Tensor of codon indices, shape (Batch, SeqLen)

        Returns:
            Scalar loss penalizing deviation from target diversity
        """
        batch_size = codon_indices.shape[0]
        diversities = torch.zeros(batch_size, device=codon_indices.device)

        for i in range(batch_size):
            seq = codon_indices[i]
            n_unique = len(torch.unique(seq))
            diversity = n_unique / len(seq)
            diversities[i] = diversity

        # Penalize deviation from target diversity
        deviation = torch.abs(diversities - self.target_diversity)
        return deviation.mean()

    def forward(
        self,
        codon_indices: torch.Tensor,
        codon_logits: torch.Tensor | None = None,
        return_components: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute combined autoimmune regularization loss.

        Args:
            codon_indices: Tensor of codon indices, shape (Batch, SeqLen)
            codon_logits: Optional logits for RSCU loss, shape (Batch, SeqLen, 64)
            return_components: If True, return individual loss components

        Returns:
            Total regularization loss (or dict of components if return_components=True)
        """
        # Always compute risk and diversity losses
        risk_loss = self.compute_risk_loss(codon_indices)
        diversity_loss = self.compute_diversity_loss(codon_indices)

        # RSCU loss only if logits provided
        if codon_logits is not None:
            rscu_loss = self.compute_rscu_loss(codon_logits)
        else:
            rscu_loss = torch.tensor(0.0, device=codon_indices.device)

        # Combine losses
        total_loss = (
            self.rscu_weight * rscu_loss + self.risk_weight * risk_loss + self.diversity_weight * diversity_loss
        )

        if return_components:
            return {
                "total": total_loss,
                "rscu": rscu_loss,
                "risk": risk_loss,
                "diversity": diversity_loss,
            }

        return total_loss

    def get_safe_codon_mask(self, threshold: float = 0.3) -> torch.Tensor:
        """Get mask of codons considered safe (low autoimmune risk).

        Args:
            threshold: RSCU threshold below which codons are masked

        Returns:
            Boolean mask, shape (64,)
        """
        return self.human_rscu >= threshold


class CD4CD8AwareRegularizer(nn.Module):
    """Regularizer that adapts to immune system state (CD4/CD8 ratio).

    Different CD4/CD8 ratios indicate different immune states:
    - High ratio (>1.5): Normal/strong cellular immunity
    - Normal ratio (1.0-1.5): Balanced immunity
    - Low ratio (<1.0): Immunocompromised/suppressed

    This regularizer adjusts the penalty based on the immune state,
    with higher penalties for immunocompromised states where
    autoimmune risk is more critical.
    """

    def __init__(
        self,
        base_regularizer: AutoimmuneCodonRegularizer | None = None,
        sensitivity_scale: float = 2.0,
    ):
        """Initialize CD4/CD8-aware regularizer.

        Args:
            base_regularizer: Base autoimmune regularizer to use
            sensitivity_scale: How much to scale penalty based on immune state
        """
        super().__init__()
        self.base = base_regularizer or AutoimmuneCodonRegularizer()
        self.sensitivity_scale = sensitivity_scale

    def compute_immune_sensitivity(self, cd4_cd8_ratio: float) -> float:
        """Compute sensitivity multiplier based on CD4/CD8 ratio.

        Args:
            cd4_cd8_ratio: Patient's CD4/CD8 ratio

        Returns:
            Sensitivity multiplier (higher = more strict penalties)
        """
        # Normal range is ~1.0-2.0
        if cd4_cd8_ratio < 0.5:
            # Severely immunocompromised - maximum sensitivity
            return self.sensitivity_scale * 2.0
        elif cd4_cd8_ratio < 1.0:
            # Moderately compromised
            return self.sensitivity_scale * 1.5
        elif cd4_cd8_ratio < 1.5:
            # Normal low end
            return self.sensitivity_scale * 1.0
        else:
            # Strong immunity - can tolerate more immunogenic sequences
            return self.sensitivity_scale * 0.5

    def forward(
        self,
        codon_indices: torch.Tensor,
        cd4_cd8_ratio: float = 1.5,
        codon_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute immune-state-aware regularization loss.

        Args:
            codon_indices: Tensor of codon indices, shape (Batch, SeqLen)
            cd4_cd8_ratio: Patient's CD4/CD8 ratio
            codon_logits: Optional logits for RSCU loss

        Returns:
            Scaled regularization loss
        """
        base_loss = self.base(codon_indices, codon_logits)
        sensitivity = self.compute_immune_sensitivity(cd4_cd8_ratio)

        return base_loss * sensitivity


__all__ = [
    "AutoimmuneCodonRegularizer",
    "CD4CD8AwareRegularizer",
    "HUMAN_CODON_RSCU",
]
