# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Genetic code constants and codon utilities.

This is the single source of truth for codon/genetic code constants used in:
- src/encoders/codon_encoder.py (GENETIC_CODE, BASE_TO_IDX)
- src/evolution/viral_evolution.py
- Research scripts (codon extraction, analysis)
"""

# Nucleotide bases
NUCLEOTIDES = "TCAG"

# Base to index mapping (for p-adic computation)
# T=0, C=1, A=2, G=3 (standard order)
BASE_TO_IDX = {"T": 0, "C": 1, "A": 2, "G": 3}
IDX_TO_BASE = {0: "T", 1: "C", 2: "A", 3: "G"}

# Standard genetic code: codon -> amino acid (single letter)
# Using "*" for stop codons
GENETIC_CODE = {
    # T-- codons
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",  # Stop (Ochre)
    "TAG": "*",  # Stop (Amber)
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",  # Stop (Opal)
    "TGG": "W",
    # C-- codons
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    # A-- codons
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",  # Start codon
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    # G-- codons
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",  # Alternative start
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

# Reverse mapping: amino acid -> list of codons (for synonymous mutations)
AMINO_ACID_TO_CODONS = {}
for codon, aa in GENETIC_CODE.items():
    if aa not in AMINO_ACID_TO_CODONS:
        AMINO_ACID_TO_CODONS[aa] = []
    AMINO_ACID_TO_CODONS[aa].append(codon)

# Codon to index (0-63)
# Encoding: idx = B1 * 16 + B2 * 4 + B3
# where B1, B2, B3 are base indices (T=0, C=1, A=2, G=3)
CODON_TO_INDEX = {}
INDEX_TO_CODON = {}
for i, b1 in enumerate(NUCLEOTIDES):
    for j, b2 in enumerate(NUCLEOTIDES):
        for k, b3 in enumerate(NUCLEOTIDES):
            codon = b1 + b2 + b3
            idx = i * 16 + j * 4 + k
            CODON_TO_INDEX[codon] = idx
            INDEX_TO_CODON[idx] = codon

# Start and stop codon indices
START_CODON_INDEX = CODON_TO_INDEX["ATG"]  # 34
STOP_CODON_INDICES = [
    CODON_TO_INDEX["TAA"],  # 8
    CODON_TO_INDEX["TAG"],  # 9
    CODON_TO_INDEX["TGA"],  # 14
]


def codon_to_amino_acid(codon: str) -> str:
    """Convert codon to amino acid.

    Args:
        codon: Three-letter codon string

    Returns:
        Single-letter amino acid code, or "X" if unknown
    """
    codon = codon.upper().replace("U", "T")  # Handle RNA
    return GENETIC_CODE.get(codon, "X")


def triplet_to_codon_index(triplet: str) -> int:
    """Convert nucleotide triplet to codon index (0-63).

    Args:
        triplet: Three-letter nucleotide string (DNA or RNA)

    Returns:
        Codon index (0-63), or -1 if invalid
    """
    triplet = triplet.upper().replace("U", "T")
    return CODON_TO_INDEX.get(triplet, -1)


# Alias for more intuitive naming
codon_to_index = triplet_to_codon_index
"""Alias for triplet_to_codon_index - converts codon string to index (0-63)."""


def codon_index_to_triplet(idx: int) -> str:
    """Convert codon index (0-63) to nucleotide triplet.

    Encoding: idx = B1 * 16 + B2 * 4 + B3
    where B1, B2, B3 are base indices (T=0, C=1, A=2, G=3)

    Args:
        idx: Codon index (0-63)

    Returns:
        Three-letter nucleotide string
    """
    if idx < 0 or idx > 63:
        return "NNN"  # Invalid

    b1 = idx // 16
    b2 = (idx % 16) // 4
    b3 = idx % 4

    return IDX_TO_BASE[b1] + IDX_TO_BASE[b2] + IDX_TO_BASE[b3]


# Alias for more intuitive naming
index_to_codon = codon_index_to_triplet
"""Alias for codon_index_to_triplet - converts index (0-63) to codon string."""


def get_synonymous_codons(aa: str) -> list[str]:
    """Get all codons that code for an amino acid.

    Args:
        aa: Single-letter amino acid code

    Returns:
        List of codons for this amino acid
    """
    return AMINO_ACID_TO_CODONS.get(aa.upper(), [])


def is_synonymous_mutation(codon1: str, codon2: str) -> bool:
    """Check if two codons code for the same amino acid.

    Args:
        codon1: First codon
        codon2: Second codon

    Returns:
        True if synonymous (same amino acid)
    """
    return codon_to_amino_acid(codon1) == codon_to_amino_acid(codon2)


def get_single_nucleotide_neighbors(codon: str) -> list[tuple[str, int, str, str]]:
    """Get all codons reachable by a single nucleotide change.

    Args:
        codon: Input codon

    Returns:
        List of (neighbor_codon, position, old_base, new_base) tuples
    """
    codon = codon.upper().replace("U", "T")
    neighbors = []

    for pos in range(3):
        for base in NUCLEOTIDES:
            if base != codon[pos]:
                new_codon = codon[:pos] + base + codon[pos + 1 :]
                neighbors.append((new_codon, pos, codon[pos], base))

    return neighbors


__all__ = [
    # Constants
    "NUCLEOTIDES",
    "BASE_TO_IDX",
    "IDX_TO_BASE",
    "GENETIC_CODE",
    "AMINO_ACID_TO_CODONS",
    "CODON_TO_INDEX",
    "INDEX_TO_CODON",
    "START_CODON_INDEX",
    "STOP_CODON_INDICES",
    # Codon conversion functions
    "codon_to_amino_acid",
    "triplet_to_codon_index",
    "codon_index_to_triplet",
    "codon_to_index",  # Alias for triplet_to_codon_index
    "index_to_codon",  # Alias for codon_index_to_triplet
    # Synonymous codon utilities
    "get_synonymous_codons",
    "is_synonymous_mutation",
    "get_single_nucleotide_neighbors",
]
