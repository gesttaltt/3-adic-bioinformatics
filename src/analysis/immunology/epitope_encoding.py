# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unified amino acid sequence encoding for immunology analysis.

Consolidates duplicated encode_sequence() and sequence_to_indices() functions from:
- src/diseases/rheumatoid_arthritis.py (CitrullinationPredictor, PAdicCitrullinationShift)
- src/diseases/multiple_sclerosis.py (MolecularMimicryDetector)

This module provides a single source of truth for:
- Amino acid to index mapping
- Amino acid biochemical properties
- Sequence encoding utilities
"""

import torch

# Standard amino acid order (consistent with UniProt/BLOSUM conventions)
STANDARD_AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"

# Amino acid to index mapping (0-19 for standard AAs, 20+ for special tokens)
AMINO_ACID_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(STANDARD_AMINO_ACIDS)}
AMINO_ACID_INDEX["X"] = 20  # Unknown amino acid
AMINO_ACID_INDEX["-"] = 21  # Gap
AMINO_ACID_INDEX["Cit"] = 22  # Citrulline (modified arginine)
AMINO_ACID_INDEX["pSer"] = 23  # Phosphoserine
AMINO_ACID_INDEX["pThr"] = 24  # Phosphothreonine
AMINO_ACID_INDEX["pTyr"] = 25  # Phosphotyrosine

# Vocabulary size for embedding layers
AMINO_ACID_VOCAB_SIZE = 26

# Comprehensive amino acid properties
# Sources: Kyte-Doolittle hydrophobicity, molecular volume, pKa values
AMINO_ACID_PROPERTIES: dict[str, dict[str, float]] = {
    "A": {"hydrophobicity": 1.8, "volume": 88.6, "charge": 0.0, "polarity": 0.0},
    "R": {"hydrophobicity": -4.5, "volume": 173.4, "charge": 1.0, "polarity": 1.0},
    "N": {"hydrophobicity": -3.5, "volume": 114.1, "charge": 0.0, "polarity": 1.0},
    "D": {"hydrophobicity": -3.5, "volume": 111.1, "charge": -1.0, "polarity": 1.0},
    "C": {"hydrophobicity": 2.5, "volume": 108.5, "charge": 0.0, "polarity": 0.0},
    "Q": {"hydrophobicity": -3.5, "volume": 143.8, "charge": 0.0, "polarity": 1.0},
    "E": {"hydrophobicity": -3.5, "volume": 138.4, "charge": -1.0, "polarity": 1.0},
    "G": {"hydrophobicity": -0.4, "volume": 60.1, "charge": 0.0, "polarity": 0.0},
    "H": {"hydrophobicity": -3.2, "volume": 153.2, "charge": 0.5, "polarity": 1.0},
    "I": {"hydrophobicity": 4.5, "volume": 166.7, "charge": 0.0, "polarity": 0.0},
    "L": {"hydrophobicity": 3.8, "volume": 166.7, "charge": 0.0, "polarity": 0.0},
    "K": {"hydrophobicity": -3.9, "volume": 168.6, "charge": 1.0, "polarity": 1.0},
    "M": {"hydrophobicity": 1.9, "volume": 162.9, "charge": 0.0, "polarity": 0.0},
    "F": {"hydrophobicity": 2.8, "volume": 189.9, "charge": 0.0, "polarity": 0.0},
    "P": {"hydrophobicity": -1.6, "volume": 112.7, "charge": 0.0, "polarity": 0.0},
    "S": {"hydrophobicity": -0.8, "volume": 89.0, "charge": 0.0, "polarity": 1.0},
    "T": {"hydrophobicity": -0.7, "volume": 116.1, "charge": 0.0, "polarity": 1.0},
    "W": {"hydrophobicity": -0.9, "volume": 227.8, "charge": 0.0, "polarity": 0.0},
    "Y": {"hydrophobicity": -1.3, "volume": 193.6, "charge": 0.0, "polarity": 1.0},
    "V": {"hydrophobicity": 4.2, "volume": 140.0, "charge": 0.0, "polarity": 0.0},
    # Modified amino acids
    "Cit": {"hydrophobicity": -3.0, "volume": 173.4, "charge": 0.0, "polarity": 1.0},
    "pSer": {"hydrophobicity": -0.8, "volume": 89.0, "charge": -2.0, "polarity": 1.0},
    "pThr": {"hydrophobicity": -0.7, "volume": 116.1, "charge": -2.0, "polarity": 1.0},
    "pTyr": {"hydrophobicity": -1.3, "volume": 193.6, "charge": -2.0, "polarity": 1.0},
    # Unknown/gap
    "X": {"hydrophobicity": 0.0, "volume": 0.0, "charge": 0.0, "polarity": 0.0},
    "-": {"hydrophobicity": 0.0, "volume": 0.0, "charge": 0.0, "polarity": 0.0},
}

# Charge-only dictionary (for compatibility with existing code)
AMINO_ACID_CHARGE: dict[str, float] = {aa: props["charge"] for aa, props in AMINO_ACID_PROPERTIES.items()}


def sequence_to_indices(
    sequence: str,
    max_length: int | None = None,
    pad_value: int = 21,
) -> list[int]:
    """Convert amino acid sequence to index list.

    Args:
        sequence: Amino acid sequence string
        max_length: Optional maximum length (will pad/truncate)
        pad_value: Index to use for padding (default: gap index)

    Returns:
        List of amino acid indices
    """
    indices = []
    i = 0
    while i < len(sequence):
        # Check for modified amino acids (3-letter codes)
        if i + 2 < len(sequence):
            three_letter = sequence[i : i + 3]
            if three_letter in AMINO_ACID_INDEX:
                indices.append(AMINO_ACID_INDEX[three_letter])
                i += 3
                continue

        # Single letter amino acid
        aa = sequence[i].upper()
        if aa in AMINO_ACID_INDEX:
            indices.append(AMINO_ACID_INDEX[aa])
        else:
            indices.append(AMINO_ACID_INDEX["X"])  # Unknown
        i += 1

    # Handle max_length
    if max_length is not None:
        if len(indices) > max_length:
            indices = indices[:max_length]
        elif len(indices) < max_length:
            indices.extend([pad_value] * (max_length - len(indices)))

    return indices


def encode_amino_acid_sequence(
    sequence: str,
    max_length: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Encode amino acid sequence as tensor of indices.

    This is the unified replacement for:
    - CitrullinationPredictor.encode_sequence()
    - PAdicCitrullinationShift.encode_sequence()
    - MolecularMimicryDetector.sequence_to_indices()

    Args:
        sequence: Amino acid sequence string
        max_length: Optional maximum length
        device: Optional torch device

    Returns:
        Tensor of shape (seq_len,) with amino acid indices
    """
    indices = sequence_to_indices(sequence, max_length)
    tensor = torch.tensor(indices, dtype=torch.long)

    if device is not None:
        tensor = tensor.to(device)

    return tensor


def get_sequence_properties(sequence: str) -> dict[str, list[float]]:
    """Get biochemical properties for each position in sequence.

    Args:
        sequence: Amino acid sequence string

    Returns:
        Dictionary with property lists for each position
    """
    properties = {
        "hydrophobicity": [],
        "volume": [],
        "charge": [],
        "polarity": [],
    }

    for aa in sequence.upper():
        props = AMINO_ACID_PROPERTIES[aa] if aa in AMINO_ACID_PROPERTIES else AMINO_ACID_PROPERTIES["X"]

        for key in properties:
            properties[key].append(props[key])

    return properties


__all__ = [
    "STANDARD_AMINO_ACIDS",
    "AMINO_ACID_INDEX",
    "AMINO_ACID_VOCAB_SIZE",
    "AMINO_ACID_PROPERTIES",
    "AMINO_ACID_CHARGE",
    "sequence_to_indices",
    "encode_amino_acid_sequence",
    "get_sequence_properties",
]
