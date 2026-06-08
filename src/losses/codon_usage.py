# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Codon Usage Loss for Biological Constraint Enforcement.

This module implements loss functions that enforce biologically realistic
codon usage patterns in generated sequences. Based on research from:

- tRNA Adaptation Index (tAI): Measures codon-anticodon efficiency
- Codon Adaptation Index (CAI): Measures codon usage bias
- CpG content: Affects immune recognition and stability
- Rare codon penalties: Affects translation rate

References:
- RiboDecode (2024): Deep learning for codon optimization
- CodonTransformer (2025): Context-aware codon optimization
- Sharp & Li (1987): CAI original formulation
- dos Reis et al. (2004): tAI formulation
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.base import LossComponent, LossResult


class Organism(Enum):
    """Supported organisms for codon usage tables."""

    HUMAN = "human"
    MOUSE = "mouse"
    ECOLI = "ecoli"
    YEAST = "yeast"
    HIV = "hiv"


# Codon usage frequencies (per 1000 codons) for different organisms
# Simplified - in production, load from Kazusa or GenScript databases
CODON_USAGE_HUMAN = {
    # Phe (F)
    "TTT": 17.6,
    "TTC": 20.3,
    # Leu (L)
    "TTA": 7.7,
    "TTG": 12.9,
    "CTT": 13.2,
    "CTC": 19.6,
    "CTA": 7.2,
    "CTG": 39.6,
    # Ile (I)
    "ATT": 16.0,
    "ATC": 20.8,
    "ATA": 7.5,
    # Met (M) - Start
    "ATG": 22.0,
    # Val (V)
    "GTT": 11.0,
    "GTC": 14.5,
    "GTA": 7.1,
    "GTG": 28.1,
    # Ser (S)
    "TCT": 15.2,
    "TCC": 17.7,
    "TCA": 12.2,
    "TCG": 4.4,
    "AGT": 12.1,
    "AGC": 19.5,
    # Pro (P)
    "CCT": 17.5,
    "CCC": 19.8,
    "CCA": 16.9,
    "CCG": 6.9,
    # Thr (T)
    "ACT": 13.1,
    "ACC": 18.9,
    "ACA": 15.1,
    "ACG": 6.1,
    # Ala (A)
    "GCT": 18.4,
    "GCC": 27.7,
    "GCA": 15.8,
    "GCG": 7.4,
    # Tyr (Y)
    "TAT": 12.2,
    "TAC": 15.3,
    # Stop
    "TAA": 1.0,
    "TAG": 0.8,
    "TGA": 1.6,
    # His (H)
    "CAT": 10.9,
    "CAC": 15.1,
    # Gln (Q)
    "CAA": 12.3,
    "CAG": 34.2,
    # Asn (N)
    "AAT": 17.0,
    "AAC": 19.1,
    # Lys (K)
    "AAA": 24.4,
    "AAG": 31.9,
    # Asp (D)
    "GAT": 21.8,
    "GAC": 25.1,
    # Glu (E)
    "GAA": 29.0,
    "GAG": 39.6,
    # Cys (C)
    "TGT": 10.6,
    "TGC": 12.6,
    # Trp (W)
    "TGG": 13.2,
    # Arg (R)
    "CGT": 4.5,
    "CGC": 10.4,
    "CGA": 6.2,
    "CGG": 11.4,
    "AGA": 12.2,
    "AGG": 12.0,
    # Gly (G)
    "GGT": 10.8,
    "GGC": 22.2,
    "GGA": 16.5,
    "GGG": 16.5,
}

# tRNA gene copy numbers (proxy for tRNA abundance) - simplified for human
TRNA_COPY_NUMBERS_HUMAN = {
    # Higher copy number = more abundant tRNA = faster translation
    "TTT": 7,
    "TTC": 10,
    "TTA": 2,
    "TTG": 4,
    "CTT": 5,
    "CTC": 8,
    "CTA": 2,
    "CTG": 12,
    "ATT": 6,
    "ATC": 8,
    "ATA": 3,
    "ATG": 10,
    "GTT": 4,
    "GTC": 6,
    "GTA": 3,
    "GTG": 10,
    "TCT": 5,
    "TCC": 7,
    "TCA": 4,
    "TCG": 2,
    "AGT": 4,
    "AGC": 8,
    "CCT": 6,
    "CCC": 8,
    "CCA": 6,
    "CCG": 2,
    "ACT": 5,
    "ACC": 8,
    "ACA": 5,
    "ACG": 2,
    "GCT": 7,
    "GCC": 10,
    "GCA": 5,
    "GCG": 3,
    "TAT": 5,
    "TAC": 7,
    "TAA": 0,
    "TAG": 0,
    "TGA": 0,
    "CAT": 4,
    "CAC": 6,
    "CAA": 5,
    "CAG": 10,
    "AAT": 6,
    "AAC": 8,
    "AAA": 8,
    "AAG": 10,
    "GAT": 8,
    "GAC": 10,
    "GAA": 10,
    "GAG": 12,
    "TGT": 4,
    "TGC": 5,
    "TGG": 5,
    "CGT": 2,
    "CGC": 4,
    "CGA": 2,
    "CGG": 4,
    "AGA": 5,
    "AGG": 5,
    "GGT": 4,
    "GGC": 8,
    "GGA": 6,
    "GGG": 6,
}


def get_codon_triplet(idx: int) -> str:
    """Convert codon index (0-63) to triplet string."""
    bases = "TCAG"
    b1 = bases[(idx >> 4) & 3]
    b2 = bases[(idx >> 2) & 3]
    b3 = bases[idx & 3]
    return b1 + b2 + b3


def create_codon_usage_tensor(usage_dict: dict[str, float], device: torch.device) -> torch.Tensor:
    """Create tensor of codon usage frequencies."""
    freqs = torch.zeros(64, device=device)
    for idx in range(64):
        triplet = get_codon_triplet(idx)
        freqs[idx] = usage_dict.get(triplet, 1.0)
    return freqs


def create_tai_tensor(trna_dict: dict[str, int], device: torch.device) -> torch.Tensor:
    """Create tensor of tAI weights from tRNA copy numbers."""
    weights = torch.zeros(64, device=device)
    for idx in range(64):
        triplet = get_codon_triplet(idx)
        weights[idx] = trna_dict.get(triplet, 1)
    # Normalize to 0-1
    if weights.max() > 0:
        weights = weights / weights.max()
    return weights


@dataclass
class CodonUsageConfig:
    """Configuration for codon usage loss."""

    organism: Organism = Organism.HUMAN
    rare_codon_threshold: float = 0.1  # Below this relative frequency = rare
    cpg_penalty_weight: float = 0.1
    tai_weight: float = 0.3
    cai_weight: float = 0.3
    rare_penalty_weight: float = 0.2
    gc_target: float = 0.5  # Target GC content (0.4-0.6 optimal)
    gc_weight: float = 0.1


class CodonUsageLoss(LossComponent):
    """Loss component for biological codon usage constraints.

    Penalizes unrealistic codon choices and rewards optimal usage:
    1. tAI (tRNA Adaptation Index): Penalize codons with low tRNA availability
    2. CAI (Codon Adaptation Index): Reward highly expressed codon patterns
    3. Rare codon penalty: Penalize rarely used codons
    4. CpG penalty: Penalize excessive CpG dinucleotides (immune activation)
    5. GC content: Keep GC content in optimal 40-60% range
    """

    def __init__(
        self,
        weight: float = 0.1,
        config: CodonUsageConfig | None = None,
        name: str | None = None,
    ):
        """Initialize CodonUsageLoss.

        Args:
            weight: Weight for this loss component
            config: Configuration object
            name: Name for logging
        """
        super().__init__(weight=weight, name=name or "codon_usage")

        if config is None:
            config = CodonUsageConfig()
        self.config = config

        # Load organism-specific data
        self._init_organism_data(config.organism)

    def _init_organism_data(self, organism: Organism):
        """Initialize organism-specific codon usage data."""
        if organism == Organism.HUMAN:
            usage_dict = CODON_USAGE_HUMAN
            trna_dict = TRNA_COPY_NUMBERS_HUMAN
        else:
            # Default to human for now
            usage_dict = CODON_USAGE_HUMAN
            trna_dict = TRNA_COPY_NUMBERS_HUMAN

        # These will be moved to device on first forward
        self.register_buffer(
            "_usage_freqs_cpu", torch.tensor([usage_dict.get(get_codon_triplet(i), 1.0) for i in range(64)])
        )
        self.register_buffer(
            "_tai_weights_cpu",
            torch.tensor([trna_dict.get(get_codon_triplet(i), 1) for i in range(64)], dtype=torch.float32),
        )

        # Normalize
        self._tai_weights_cpu = self._tai_weights_cpu / self._tai_weights_cpu.max()

        # Mark rare codons
        max_freq = self._usage_freqs_cpu.max()
        self.register_buffer("_rare_mask_cpu", (self._usage_freqs_cpu / max_freq) < self.config.rare_codon_threshold)

        # Stop codon indices
        stop_indices = []
        for idx in range(64):
            triplet = get_codon_triplet(idx)
            if triplet in ["TAA", "TAG", "TGA"]:
                stop_indices.append(idx)
        self.register_buffer("_stop_indices", torch.tensor(stop_indices))

    def compute_tai_loss(self, codon_indices: torch.Tensor) -> torch.Tensor:
        """Compute tRNA Adaptation Index loss.

        Lower tAI = slower translation = higher loss.

        Args:
            codon_indices: (batch, seq_len) codon indices 0-63

        Returns:
            Scalar loss (1 - mean_tAI)
        """
        tai_weights = self._tai_weights_cpu.to(codon_indices.device)
        tai_scores = tai_weights[codon_indices]

        # Mask out padding (assuming padding uses index 64 or special value)
        valid_mask = codon_indices < 64
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=codon_indices.device)

        mean_tai = tai_scores[valid_mask].mean()
        return 1.0 - mean_tai

    def compute_cai_loss(self, codon_indices: torch.Tensor) -> torch.Tensor:
        """Compute Codon Adaptation Index loss.

        CAI measures how well codon usage matches highly expressed genes.
        Lower CAI = suboptimal codon usage = higher loss.

        Args:
            codon_indices: (batch, seq_len) codon indices

        Returns:
            Scalar loss (1 - normalized_CAI)
        """
        usage_freqs = self._usage_freqs_cpu.to(codon_indices.device)

        # Get frequency for each codon
        codon_freqs = usage_freqs[codon_indices.clamp(0, 63)]

        # CAI is geometric mean of relative adaptiveness
        # For simplicity, use normalized frequency directly
        valid_mask = codon_indices < 64
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=codon_indices.device)

        # Normalize frequencies
        max_freq = usage_freqs.max()
        rel_freq = codon_freqs / max_freq

        # Geometric mean via log
        log_rel = torch.log(rel_freq[valid_mask] + 1e-8)
        cai = torch.exp(log_rel.mean())

        return 1.0 - cai

    def compute_rare_codon_loss(self, codon_indices: torch.Tensor) -> torch.Tensor:
        """Compute rare codon penalty.

        Args:
            codon_indices: (batch, seq_len) codon indices

        Returns:
            Fraction of rare codons (0-1)
        """
        rare_mask = self._rare_mask_cpu.to(codon_indices.device)

        is_rare = rare_mask[codon_indices.clamp(0, 63)]
        valid_mask = codon_indices < 64

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=codon_indices.device)

        rare_fraction = is_rare[valid_mask].float().mean()
        return rare_fraction

    def compute_cpg_loss(self, codon_indices: torch.Tensor) -> torch.Tensor:
        """Compute CpG dinucleotide penalty.

        Excessive CpG can trigger immune responses.

        Args:
            codon_indices: (batch, seq_len) codon indices

        Returns:
            CpG density penalty
        """
        batch_size, seq_len = codon_indices.shape

        # Convert codons to nucleotides
        nucleotides = []
        for pos in range(3):
            shift = 4 - 2 * pos  # 4, 2, 0 for positions 0, 1, 2
            nt = (codon_indices >> shift) & 3
            nucleotides.append(nt)

        # Interleave: (batch, seq_len * 3)
        nt_seq = torch.stack(nucleotides, dim=2).reshape(batch_size, seq_len * 3)

        # Find CpG: C (index 1) followed by G (index 2)
        is_c = nt_seq[:, :-1] == 1
        is_g = nt_seq[:, 1:] == 2
        is_cpg = (is_c & is_g).float()

        # CpG density
        cpg_density = is_cpg.mean()

        # Penalize if above natural human CpG frequency (~1%)
        target_density = 0.01
        excess = F.relu(cpg_density - target_density)

        return excess * 10  # Scale up penalty

    def compute_gc_loss(self, codon_indices: torch.Tensor) -> torch.Tensor:
        """Compute GC content deviation loss.

        Optimal GC content is typically 40-60%.

        Args:
            codon_indices: (batch, seq_len) codon indices

        Returns:
            Deviation from target GC content
        """
        device = codon_indices.device
        batch_size, seq_len = codon_indices.shape

        # Count GC in each codon
        # G = 2, C = 1 in our encoding
        gc_counts = torch.zeros(64, device=device)
        for idx in range(64):
            triplet = get_codon_triplet(idx)
            gc_counts[idx] = triplet.count("G") + triplet.count("C")

        codon_gc = gc_counts[codon_indices.clamp(0, 63)]
        gc_fraction = codon_gc.sum() / (codon_indices.numel() * 3 + 1e-8)

        # Deviation from target
        deviation = (gc_fraction - self.config.gc_target).abs()

        return deviation

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        **kwargs,
    ) -> LossResult:
        """Compute codon usage loss.

        Args:
            outputs: Model outputs, expecting 'generated_codons' or 'codon_logits'
            targets: Target tensor (may not be used)
            **kwargs: Additional arguments

        Returns:
            LossResult with combined codon usage loss
        """
        # Get generated codons
        if "generated_codons" in outputs:
            codon_indices = outputs["generated_codons"]
        elif "codon_logits" in outputs:
            codon_indices = outputs["codon_logits"].argmax(dim=-1)
        elif "x_recon" in outputs:
            # Reconstruction case
            codon_indices = outputs["x_recon"].argmax(dim=-1) if outputs["x_recon"].dim() > 2 else outputs["x_recon"]
        else:
            # No codon data available
            return LossResult(
                loss=torch.tensor(0.0, device=targets.device),
                metrics={},
                weight=self.weight,
            )

        # Compute component losses
        tai_loss = self.compute_tai_loss(codon_indices)
        cai_loss = self.compute_cai_loss(codon_indices)
        rare_loss = self.compute_rare_codon_loss(codon_indices)
        cpg_loss = self.compute_cpg_loss(codon_indices)
        gc_loss = self.compute_gc_loss(codon_indices)

        # Combine with weights
        total_loss = (
            self.config.tai_weight * tai_loss
            + self.config.cai_weight * cai_loss
            + self.config.rare_penalty_weight * rare_loss
            + self.config.cpg_penalty_weight * cpg_loss
            + self.config.gc_weight * gc_loss
        )

        metrics = {
            "tai_loss": tai_loss.item(),
            "cai_loss": cai_loss.item(),
            "rare_codon_fraction": rare_loss.item(),
            "cpg_penalty": cpg_loss.item(),
            "gc_deviation": gc_loss.item(),
            "mean_tai": (1.0 - tai_loss).item(),
            "mean_cai": (1.0 - cai_loss).item(),
        }

        return LossResult(
            loss=total_loss,
            metrics=metrics,
            weight=self.weight,
        )


class CodonOptimalityScore(nn.Module):
    """Compute codon optimality scores for evaluation.

    This is not a loss function but a metric calculator for
    evaluating generated sequences.
    """

    def __init__(self, organism: Organism = Organism.HUMAN):
        super().__init__()
        self.organism = organism

        if organism == Organism.HUMAN:
            usage_dict = CODON_USAGE_HUMAN
            trna_dict = TRNA_COPY_NUMBERS_HUMAN
        else:
            usage_dict = CODON_USAGE_HUMAN
            trna_dict = TRNA_COPY_NUMBERS_HUMAN

        self.register_buffer(
            "usage_freqs",
            torch.tensor([usage_dict.get(get_codon_triplet(i), 1.0) for i in range(64)]),
        )
        self.register_buffer(
            "tai_weights",
            torch.tensor([trna_dict.get(get_codon_triplet(i), 1) for i in range(64)], dtype=torch.float32),
        )
        self.tai_weights = self.tai_weights / self.tai_weights.max()

    def compute_tai(self, codon_indices: torch.Tensor) -> torch.Tensor:
        """Compute tRNA Adaptation Index for sequences."""
        tai_scores = self.tai_weights.to(codon_indices.device)[codon_indices.clamp(0, 63)]
        return tai_scores.mean(dim=-1)

    def compute_cai(self, codon_indices: torch.Tensor) -> torch.Tensor:
        """Compute Codon Adaptation Index for sequences."""
        freqs = self.usage_freqs.to(codon_indices.device)[codon_indices.clamp(0, 63)]
        max_freq = self.usage_freqs.max()
        rel_freq = freqs / max_freq

        # Geometric mean
        log_rel = torch.log(rel_freq + 1e-8)
        cai = torch.exp(log_rel.mean(dim=-1))
        return cai

    def forward(self, codon_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute all optimality scores.

        Args:
            codon_indices: (batch, seq_len) codon indices

        Returns:
            Dictionary with TAI, CAI, and other scores per sequence
        """
        return {
            "tai": self.compute_tai(codon_indices),
            "cai": self.compute_cai(codon_indices),
        }


__all__ = [
    "CodonUsageLoss",
    "CodonUsageConfig",
    "CodonOptimalityScore",
    "Organism",
]
