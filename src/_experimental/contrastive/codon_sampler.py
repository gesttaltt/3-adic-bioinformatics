# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Codon-Aware Contrastive Sampling for BYOL/SimCLR.

This module extends the p-adic positive sampling with biologically meaningful
codon relationships for contrastive learning.

Key Features:
- Synonymous codon sampling: Codons encoding same amino acid
- Wobble position variation: 3rd position mutations
- Conservative substitution sampling: Similar amino acids
- Multi-scale hierarchy: P-adic + biochemical similarity

Based on research from:
- CCPL: Codon-aware contrastive protein learning
- ProteinAligner: Evolutionary-aware contrastive learning
- CPCProt: Contrastive predictive coding for proteins
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.biology.codons import GENETIC_CODE, codon_index_to_triplet
from src.encoders.codon_encoder import (
    compute_amino_acid_distance,
    compute_padic_distance_between_codons,
)

# Amino acid similarity groups based on biochemical properties
AA_SIMILARITY_GROUPS = {
    "hydrophobic": {"A", "V", "I", "L", "M", "F", "W", "P"},
    "polar": {"S", "T", "N", "Q", "C", "Y"},
    "positive": {"K", "R", "H"},
    "negative": {"D", "E"},
    "special": {"G", "P"},
}

# BLOSUM62-derived similarity (simplified)
AA_SIMILAR_PAIRS = {
    "A": ["A", "S", "G", "V"],
    "R": ["R", "K", "H", "Q"],
    "N": ["N", "D", "S", "H"],
    "D": ["D", "E", "N"],
    "C": ["C", "S"],
    "Q": ["Q", "E", "K", "R"],
    "E": ["E", "D", "Q", "K"],
    "G": ["G", "A", "S"],
    "H": ["H", "R", "Q", "N"],
    "I": ["I", "V", "L", "M"],
    "L": ["L", "I", "V", "M", "F"],
    "K": ["K", "R", "Q", "E"],
    "M": ["M", "L", "I", "V"],
    "F": ["F", "Y", "W", "L"],
    "P": ["P"],
    "S": ["S", "T", "A", "N"],
    "T": ["T", "S", "A"],
    "W": ["W", "Y", "F"],
    "Y": ["Y", "F", "W", "H"],
    "V": ["V", "I", "L", "M", "A"],
}


@dataclass
class CodonSamplerConfig:
    """Configuration for codon-aware sampling."""

    # Sampling strategies
    use_synonymous: bool = True  # Same amino acid
    use_wobble: bool = True  # 3rd position only different
    use_conservative: bool = True  # Similar amino acid
    use_padic: bool = True  # P-adic distance based

    # Thresholds
    padic_threshold: float = 0.33  # 1/p for same first two bases
    aa_distance_threshold: float = 1.5  # Max AA property distance

    # Weights for sampling probability
    synonymous_weight: float = 0.4
    wobble_weight: float = 0.3
    conservative_weight: float = 0.2
    padic_weight: float = 0.1

    # Negative mining
    hard_negative_ratio: float = 0.5  # Fraction of hard negatives


def build_synonymous_codon_groups() -> dict[str, list[int]]:
    """Build mapping from amino acid to codon indices."""
    groups: dict[str, list[int]] = {}

    for idx in range(64):
        triplet = codon_index_to_triplet(idx)
        aa = GENETIC_CODE.get(triplet, "*")
        if aa not in groups:
            groups[aa] = []
        groups[aa].append(idx)

    return groups


def build_wobble_variants(codon_idx: int) -> list[int]:
    """Get codon indices that differ only in wobble (3rd) position.

    Args:
        codon_idx: Original codon index (0-63)

    Returns:
        List of variant indices with different 3rd position
    """
    # Codon index encodes as: b1*16 + b2*4 + b3
    # Same first two bases = same b1*16 + b2*4 = same (idx // 4) * 4
    base_idx = (codon_idx // 4) * 4

    variants = []
    for offset in range(4):
        variant = base_idx + offset
        if variant != codon_idx:
            variants.append(variant)

    return variants


class CodonPositiveSampler:
    """Sample positive pairs based on biological codon relationships.

    Combines multiple criteria:
    1. Synonymous: Same amino acid (strongest positive signal)
    2. Wobble: Same first two bases (evolutionary close)
    3. Conservative: Similar amino acid properties
    4. P-adic: Close in p-adic metric
    """

    def __init__(
        self,
        config: CodonSamplerConfig | None = None,
    ):
        """Initialize sampler.

        Args:
            config: Sampling configuration
        """
        if config is None:
            config = CodonSamplerConfig()
        self.config = config

        # Build lookup tables
        self.synonymous_groups = build_synonymous_codon_groups()

        # Build amino acid to codon mapping
        self.aa_to_codons: dict[str, list[int]] = self.synonymous_groups.copy()

        # Build conservative substitution map
        self.conservative_codons = self._build_conservative_map()

        # Precompute p-adic distances
        self.padic_matrix = self._compute_padic_matrix()

    def _build_conservative_map(self) -> dict[int, set[int]]:
        """Build map of conservatively substitutable codons."""
        conservative: dict[int, set[int]] = {}

        for idx in range(64):
            triplet = codon_index_to_triplet(idx)
            aa = GENETIC_CODE.get(triplet, "*")

            if aa == "*":
                conservative[idx] = set()
                continue

            # Get similar amino acids
            similar_aas = set(AA_SIMILAR_PAIRS.get(aa, [aa]))

            # Get all codons for similar AAs
            similar_codons = set()
            for sim_aa in similar_aas:
                if sim_aa in self.aa_to_codons:
                    similar_codons.update(self.aa_to_codons[sim_aa])

            conservative[idx] = similar_codons

        return conservative

    def _compute_padic_matrix(self) -> torch.Tensor:
        """Precompute 64x64 p-adic distance matrix."""
        matrix = torch.zeros(64, 64)
        for i in range(64):
            for j in range(64):
                matrix[i, j] = compute_padic_distance_between_codons(i, j)
        return matrix

    def get_positive_candidates(
        self,
        anchor_idx: int,
        strategy: str = "all",
    ) -> set[int]:
        """Get positive candidate codons for an anchor.

        Args:
            anchor_idx: Anchor codon index (0-63)
            strategy: Sampling strategy ('synonymous', 'wobble', 'conservative', 'padic', 'all')

        Returns:
            Set of positive candidate indices
        """
        candidates: set[int] = set()

        if strategy == "all" or (strategy == "synonymous" and self.config.use_synonymous):
            # Get synonymous codons
            triplet = codon_index_to_triplet(anchor_idx)
            aa = GENETIC_CODE.get(triplet, "*")
            if aa in self.synonymous_groups:
                candidates.update(self.synonymous_groups[aa])

        if strategy == "all" or (strategy == "wobble" and self.config.use_wobble):
            # Get wobble variants
            wobble_vars = build_wobble_variants(anchor_idx)
            candidates.update(wobble_vars)

        if strategy == "all" or (strategy == "conservative" and self.config.use_conservative):
            # Get conservative substitutions
            candidates.update(self.conservative_codons.get(anchor_idx, set()))

        if strategy == "all" or (strategy == "padic" and self.config.use_padic):
            # Get p-adically close codons
            padic_close = torch.where(self.padic_matrix[anchor_idx] <= self.config.padic_threshold)[0]
            candidates.update(padic_close.tolist())

        # Remove anchor itself
        candidates.discard(anchor_idx)

        return candidates

    def get_positive_weights(
        self,
        anchor_idx: int,
        candidate_idx: int,
    ) -> float:
        """Compute sampling weight for a positive pair.

        Higher weight = more likely to be sampled.

        Args:
            anchor_idx: Anchor codon index
            candidate_idx: Candidate codon index

        Returns:
            Sampling weight (0-1)
        """
        weight = 0.0

        # Synonymous bonus
        triplet1 = codon_index_to_triplet(anchor_idx)
        triplet2 = codon_index_to_triplet(candidate_idx)
        aa1 = GENETIC_CODE.get(triplet1, "*")
        aa2 = GENETIC_CODE.get(triplet2, "*")

        if aa1 == aa2 and aa1 != "*":
            weight += self.config.synonymous_weight

        # Wobble bonus (same first two bases)
        if anchor_idx // 4 == candidate_idx // 4:
            weight += self.config.wobble_weight

        # Conservative bonus
        if candidate_idx in self.conservative_codons.get(anchor_idx, set()):
            weight += self.config.conservative_weight

        # P-adic bonus
        padic_dist = self.padic_matrix[anchor_idx, candidate_idx]
        if padic_dist <= self.config.padic_threshold:
            weight += self.config.padic_weight * (1 - padic_dist)

        return min(weight, 1.0)

    def sample_positive(
        self,
        anchor_idx: int,
        exclude: set[int] | None = None,
    ) -> int | None:
        """Sample a single positive for anchor.

        Args:
            anchor_idx: Anchor codon index
            exclude: Indices to exclude from sampling

        Returns:
            Sampled positive index, or None if none available
        """
        candidates = self.get_positive_candidates(anchor_idx)

        if exclude:
            candidates = candidates - exclude

        if not candidates:
            return None

        # Compute weights
        candidate_list = list(candidates)
        weights = torch.tensor([self.get_positive_weights(anchor_idx, c) for c in candidate_list])

        # Normalize to probabilities
        probs = torch.ones_like(weights) / len(weights) if weights.sum() == 0 else weights / weights.sum()

        # Sample
        idx = torch.multinomial(probs, 1).item()
        return candidate_list[idx]

    def sample_negatives(
        self,
        anchor_idx: int,
        n_negatives: int = 10,
        hard_negative_ratio: float | None = None,
    ) -> list[int]:
        """Sample negative codons for anchor.

        Args:
            anchor_idx: Anchor codon index
            n_negatives: Number of negatives to sample
            hard_negative_ratio: Fraction of hard negatives (close but not positive)

        Returns:
            List of negative indices
        """
        if hard_negative_ratio is None:
            hard_negative_ratio = self.config.hard_negative_ratio

        positives = self.get_positive_candidates(anchor_idx)
        all_negatives = set(range(64)) - positives - {anchor_idx}

        if not all_negatives:
            return []

        n_hard = int(n_negatives * hard_negative_ratio)
        n_easy = n_negatives - n_hard

        negatives = []

        # Hard negatives: high AA distance but low p-adic distance
        if n_hard > 0:
            hard_candidates = []
            for neg_idx in all_negatives:
                padic_dist = self.padic_matrix[anchor_idx, neg_idx]
                aa_dist = compute_amino_acid_distance(anchor_idx, neg_idx)

                # Hard: similar structurally but different AA
                if padic_dist < 0.5 and aa_dist > 1.0:
                    hard_candidates.append(neg_idx)

            if hard_candidates:
                n_hard = min(n_hard, len(hard_candidates))
                hard_indices = torch.randperm(len(hard_candidates))[:n_hard]
                negatives.extend([hard_candidates[i] for i in hard_indices.tolist()])

        # Easy negatives: random from remaining
        remaining = list(all_negatives - set(negatives))
        if remaining and n_easy > 0:
            n_easy = min(n_easy, len(remaining))
            easy_indices = torch.randperm(len(remaining))[:n_easy]
            negatives.extend([remaining[i] for i in easy_indices.tolist()])

        return negatives


class CodonContrastiveDataset(torch.utils.data.Dataset):
    """Dataset wrapper for codon-aware contrastive learning.

    Creates positive pairs using the CodonPositiveSampler.
    """

    def __init__(
        self,
        sequences: torch.Tensor,
        sampler: CodonPositiveSampler | None = None,
        augmentation_fn: callable | None = None,
    ):
        """Initialize dataset.

        Args:
            sequences: (n_sequences, seq_len) codon sequences
            sampler: Codon positive sampler
            augmentation_fn: Optional augmentation function
        """
        self.sequences = sequences
        self.sampler = sampler or CodonPositiveSampler()
        self.augmentation = augmentation_fn

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get anchor and positive pair.

        Args:
            idx: Sequence index

        Returns:
            Tuple of (anchor_seq, positive_seq, labels)
        """
        anchor_seq = self.sequences[idx].clone()

        # Create positive by mutating some codons to synonymous/similar
        positive_seq = anchor_seq.clone()

        # Mutate ~20% of positions
        n_mutations = max(1, int(0.2 * len(anchor_seq)))
        mutation_positions = torch.randperm(len(anchor_seq))[:n_mutations]

        for pos in mutation_positions:
            original_codon = anchor_seq[pos].item()
            new_codon = self.sampler.sample_positive(original_codon)
            if new_codon is not None:
                positive_seq[pos] = new_codon

        # Apply augmentation if provided
        if self.augmentation is not None:
            anchor_seq = self.augmentation(anchor_seq)
            positive_seq = self.augmentation(positive_seq)

        # Labels indicate positive pair
        labels = torch.tensor([1])

        return anchor_seq, positive_seq, labels


__all__ = [
    "CodonPositiveSampler",
    "CodonSamplerConfig",
    "CodonContrastiveDataset",
    "build_synonymous_codon_groups",
    "build_wobble_variants",
]
