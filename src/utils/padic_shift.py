# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""P-adic shift wrapper functions for biological sequence analysis.

This module provides biological sequence-specific p-adic operations,
building on the core p-adic math utilities.

Key functions:
- codon_padic_distance: Distance between codons in 3-adic space
- sequence_padic_encoding: Full sequence encoding
- PAdicSequenceEncoder: Encoder class for biological sequences

The 3-adic framework is particularly suited for codon analysis because:
- Codons have 3 positions
- Each position has 4 nucleotides (mapped to 0-3)
- The p-adic distance captures wobble position effects

Note:
    Core p-adic operations (valuation, norm, distance, digits) are
    imported from src.core.padic_math. Use that module directly for
    general p-adic computations.

References:
- 2006_Kozyrev_Padic_Analysis_Methods.md
- 1975_Wong_CoEvolution_Theory.md
"""

from collections.abc import Sequence

import torch

from src.biology.codons import codon_to_index, index_to_codon

# Import all core p-adic operations from centralized module
from src.core.padic_math import (
    PAdicShiftResult,
    padic_digits,
    padic_distance,
    padic_distance_matrix,
    padic_norm,
    padic_shift,
    padic_valuation,
)


def index_to_rna_codon(index: int) -> str:
    """Convert a codon index (0-63) to RNA codon (using U instead of T).

    Args:
        index: Codon index from 0 to 63

    Returns:
        3-letter RNA codon string (using U for uracil)
    """
    return index_to_codon(index).replace("T", "U")


def codon_padic_distance(codon1: str, codon2: str, p: int = 3) -> float:
    """Compute p-adic distance between two codons.

    Args:
        codon1: First codon (3 letters)
        codon2: Second codon (3 letters)
        p: Prime base (default 3)

    Returns:
        p-adic distance
    """
    idx1 = codon_to_index(codon1)
    idx2 = codon_to_index(codon2)
    return padic_distance(idx1, idx2, p)


def sequence_padic_encoding(
    sequence: str,
    p: int = 3,
    n_digits: int = 4,
) -> torch.Tensor:
    """Encode a nucleotide sequence into p-adic representation.

    Converts sequence to codon indices and then to p-adic digits.

    Args:
        sequence: Nucleotide sequence (length must be multiple of 3)
        p: Prime base (default 3)
        n_digits: Number of p-adic digits per codon

    Returns:
        Tensor of shape (n_codons, n_digits) containing p-adic digits
    """
    sequence = sequence.upper().replace("T", "U")

    if len(sequence) % 3 != 0:
        # Pad to multiple of 3
        padding = 3 - (len(sequence) % 3)
        sequence += "N" * padding

    n_codons = len(sequence) // 3
    encodings = []

    for i in range(n_codons):
        codon = sequence[i * 3 : (i + 1) * 3]
        if "N" in codon:
            # Unknown nucleotide - use zero encoding
            encodings.append([0] * n_digits)
        else:
            idx = codon_to_index(codon)
            digits = padic_digits(idx, p, n_digits)
            encodings.append(digits)

    return torch.tensor(encodings, dtype=torch.float32)


def batch_padic_distance(
    indices1: torch.Tensor,
    indices2: torch.Tensor,
    p: int = 3,
) -> torch.Tensor:
    """Compute p-adic distances for batches of index pairs.

    Note: This is a convenience wrapper around padic_distance_vectorized
    from src.core.padic_math.

    Args:
        indices1: First indices (batch,)
        indices2: Second indices (batch,)
        p: Prime base (default 3)

    Returns:
        p-adic distances (batch,)
    """
    from src.core.padic_math import padic_distance_vectorized

    return padic_distance_vectorized(indices1, indices2, p)


class PAdicSequenceEncoder:
    """Encoder for biological sequences using p-adic representation.

    Provides methods for encoding DNA, RNA, and protein sequences
    into p-adic space for analysis.
    """

    def __init__(self, p: int = 3, n_digits: int = 4):
        """Initialize encoder.

        Args:
            p: Prime base (default 3)
            n_digits: Number of p-adic digits to compute
        """
        self.p = p
        self.n_digits = n_digits

        # Precompute p-adic digits for all 64 codons
        self.codon_digits = torch.zeros(64, n_digits)
        for i in range(64):
            digits = padic_digits(i, p, n_digits)
            self.codon_digits[i] = torch.tensor(digits, dtype=torch.float)

        # Precompute 64x64 distance matrix using centralized function
        indices = torch.arange(64)
        self.distance_matrix = padic_distance_matrix(indices, p)

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode a nucleotide sequence.

        Args:
            sequence: DNA/RNA sequence

        Returns:
            p-adic encoding tensor
        """
        return sequence_padic_encoding(sequence, self.p, self.n_digits)

    def encode_codons(self, codons: Sequence[str]) -> torch.Tensor:
        """Encode a list of codons.

        Args:
            codons: List of 3-letter codon strings

        Returns:
            Tensor of codon indices (n_codons,)
        """
        indices = [codon_to_index(c) for c in codons]
        return torch.tensor(indices)

    def get_padic_digits(self, codon_indices: torch.Tensor) -> torch.Tensor:
        """Get p-adic digit representations for codon indices.

        Args:
            codon_indices: Tensor of codon indices (batch, seq_len)

        Returns:
            p-adic digits (batch, seq_len, n_digits)
        """
        return self.codon_digits[codon_indices.long()]

    def compute_distances(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute pairwise p-adic distances.

        Uses centralized tensor utilities for efficient computation.

        Args:
            indices: Codon indices (batch, seq_len)

        Returns:
            Distance matrix (batch, seq_len, seq_len)
        """
        from src.core.tensor_utils import batch_index_select, pairwise_broadcast

        batch_size, seq_len = indices.shape
        indices = indices.long()

        # Use centralized utilities
        i_idx, j_idx = pairwise_broadcast(indices)

        # Move distance_matrix to same device as indices if needed
        dist_matrix = self.distance_matrix
        if dist_matrix.device != indices.device:
            dist_matrix = dist_matrix.to(indices.device)

        distances = batch_index_select(dist_matrix, i_idx, j_idx, (batch_size, seq_len, seq_len))

        return distances


class PAdicCodonAnalyzer:
    """Analyzer for codon usage patterns using p-adic metrics.

    Provides tools for analyzing synonymous codon usage,
    codon bias, and evolutionary distances.
    """

    # Genetic code table (standard)
    CODON_TABLE = {
        "UUU": "F",
        "UUC": "F",
        "UUA": "L",
        "UUG": "L",
        "UCU": "S",
        "UCC": "S",
        "UCA": "S",
        "UCG": "S",
        "UAU": "Y",
        "UAC": "Y",
        "UAA": "*",
        "UAG": "*",
        "UGU": "C",
        "UGC": "C",
        "UGA": "*",
        "UGG": "W",
        "CUU": "L",
        "CUC": "L",
        "CUA": "L",
        "CUG": "L",
        "CCU": "P",
        "CCC": "P",
        "CCA": "P",
        "CCG": "P",
        "CAU": "H",
        "CAC": "H",
        "CAA": "Q",
        "CAG": "Q",
        "CGU": "R",
        "CGC": "R",
        "CGA": "R",
        "CGG": "R",
        "AUU": "I",
        "AUC": "I",
        "AUA": "I",
        "AUG": "M",
        "ACU": "T",
        "ACC": "T",
        "ACA": "T",
        "ACG": "T",
        "AAU": "N",
        "AAC": "N",
        "AAA": "K",
        "AAG": "K",
        "AGU": "S",
        "AGC": "S",
        "AGA": "R",
        "AGG": "R",
        "GUU": "V",
        "GUC": "V",
        "GUA": "V",
        "GUG": "V",
        "GCU": "A",
        "GCC": "A",
        "GCA": "A",
        "GCG": "A",
        "GAU": "D",
        "GAC": "D",
        "GAA": "E",
        "GAG": "E",
        "GGU": "G",
        "GGC": "G",
        "GGA": "G",
        "GGG": "G",
    }

    def __init__(self, p: int = 3):
        """Initialize analyzer.

        Args:
            p: Prime base (default 3)
        """
        self.p = p
        self.encoder = PAdicSequenceEncoder(p=p)

        # Build synonymous codon groups
        self.synonymous_groups = self._build_synonymous_groups()

    def _build_synonymous_groups(self) -> dict[str, list[str]]:
        """Build mapping from amino acid to synonymous codons."""
        groups: dict[str, list[str]] = {}
        for codon, aa in self.CODON_TABLE.items():
            if aa not in groups:
                groups[aa] = []
            groups[aa].append(codon)
        return groups

    def synonymous_padic_spread(self, amino_acid: str) -> float:
        """Compute p-adic spread of synonymous codons for an amino acid.

        A larger spread indicates synonymous codons are more "distant"
        in p-adic space, suggesting wobble position effects.

        Args:
            amino_acid: Single-letter amino acid code

        Returns:
            Average pairwise p-adic distance among synonymous codons
        """
        codons = self.synonymous_groups.get(amino_acid.upper(), [])
        if len(codons) <= 1:
            return 0.0

        distances = []
        for i, c1 in enumerate(codons):
            for c2 in codons[i + 1 :]:
                d = codon_padic_distance(c1, c2, self.p)
                distances.append(d)

        return sum(distances) / len(distances) if distances else 0.0

    def codon_bias_score(self, sequence: str) -> float:
        """Compute codon bias score based on p-adic distances.

        Higher scores indicate preference for p-adically "close"
        synonymous codons.

        Args:
            sequence: Nucleotide sequence

        Returns:
            Codon bias score
        """
        sequence = sequence.upper().replace("T", "U")
        if len(sequence) < 3:
            return 0.0

        n_codons = len(sequence) // 3
        total_distance = 0.0
        count = 0

        for i in range(n_codons):
            codon = sequence[i * 3 : (i + 1) * 3]
            if "N" in codon:
                continue

            aa = self.CODON_TABLE.get(codon)
            if aa is None or aa == "*":
                continue

            # Compare to other synonymous codons
            for syn_codon in self.synonymous_groups.get(aa, []):
                if syn_codon != codon:
                    d = codon_padic_distance(codon, syn_codon, self.p)
                    total_distance += d
                    count += 1

        return total_distance / count if count > 0 else 0.0


# Re-export for backward compatibility (prefer importing from src.core directly)
__all__ = [
    # Core functions (re-exported from src.core.padic_math)
    "padic_valuation",
    "padic_norm",
    "padic_distance",
    "padic_digits",
    "padic_shift",
    "PAdicShiftResult",
    "padic_distance_matrix",
    # Sequence-specific functions
    "index_to_rna_codon",
    "codon_padic_distance",
    "sequence_padic_encoding",
    "batch_padic_distance",
    # Classes
    "PAdicSequenceEncoder",
    "PAdicCodonAnalyzer",
]
