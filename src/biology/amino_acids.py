# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Amino acid constants and properties.

This is the single source of truth for amino acid properties used in:
- src/encoders/codon_encoder.py (AA_PROPERTIES)
- src/evolution/viral_evolution.py (AMINO_ACID_PROPERTIES)
- src/diseases/multiple_sclerosis.py (AMINO_ACID_PROPERTIES)
- src/analysis/immunology/epitope_encoding.py (AMINO_ACID_PROPERTIES)
"""

# Standard 20 amino acids in alphabetical order (UniProt convention)
STANDARD_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Three-letter codes
AMINO_ACID_3LETTER = {
    "A": "Ala",
    "C": "Cys",
    "D": "Asp",
    "E": "Glu",
    "F": "Phe",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "K": "Lys",
    "L": "Leu",
    "M": "Met",
    "N": "Asn",
    "P": "Pro",
    "Q": "Gln",
    "R": "Arg",
    "S": "Ser",
    "T": "Thr",
    "V": "Val",
    "W": "Trp",
    "Y": "Tyr",
    "*": "Stop",
}

# Comprehensive amino acid properties
# Sources:
#   - Hydrophobicity: Kyte-Doolittle scale
#   - Volume: Molecular volume in Angstroms^3
#   - Charge: Net charge at pH 7.0
#   - Polarity: 0 = nonpolar, 1 = polar
#   - Normalized values for ML: hydrophobicity_norm, volume_norm, charge_norm
AMINO_ACID_PROPERTIES: dict[str, dict[str, float]] = {
    "A": {
        "hydrophobicity": 1.8,
        "volume": 88.6,
        "charge": 0.0,
        "polarity": 0.0,
        # Normalized for ML (approx [-1, 1])
        "hydrophobicity_norm": 0.62,
        "volume_norm": -0.77,
        "charge_norm": 0.0,
        "polarity_norm": -0.5,
    },
    "R": {
        "hydrophobicity": -4.5,
        "volume": 173.4,
        "charge": 1.0,
        "polarity": 1.0,
        "hydrophobicity_norm": -2.53,
        "volume_norm": 0.69,
        "charge_norm": 1.0,
        "polarity_norm": 1.0,
    },
    "N": {
        "hydrophobicity": -3.5,
        "volume": 114.1,
        "charge": 0.0,
        "polarity": 1.0,
        "hydrophobicity_norm": -0.78,
        "volume_norm": -0.09,
        "charge_norm": 0.0,
        "polarity_norm": 1.0,
    },
    "D": {
        "hydrophobicity": -3.5,
        "volume": 111.1,
        "charge": -1.0,
        "polarity": 1.0,
        "hydrophobicity_norm": -0.90,
        "volume_norm": -0.25,
        "charge_norm": -1.0,
        "polarity_norm": 1.0,
    },
    "C": {
        "hydrophobicity": 2.5,
        "volume": 108.5,
        "charge": 0.0,
        "polarity": 0.0,
        "hydrophobicity_norm": 0.29,
        "volume_norm": -0.41,
        "charge_norm": 0.0,
        "polarity_norm": 0.0,
    },
    "Q": {
        "hydrophobicity": -3.5,
        "volume": 143.8,
        "charge": 0.0,
        "polarity": 1.0,
        "hydrophobicity_norm": -0.85,
        "volume_norm": 0.24,
        "charge_norm": 0.0,
        "polarity_norm": 1.0,
    },
    "E": {
        "hydrophobicity": -3.5,
        "volume": 138.4,
        "charge": -1.0,
        "polarity": 1.0,
        "hydrophobicity_norm": -0.74,
        "volume_norm": 0.07,
        "charge_norm": -1.0,
        "polarity_norm": 1.0,
    },
    "G": {
        "hydrophobicity": -0.4,
        "volume": 60.1,
        "charge": 0.0,
        "polarity": 0.0,
        "hydrophobicity_norm": 0.48,
        "volume_norm": -1.00,
        "charge_norm": 0.0,
        "polarity_norm": -0.5,
    },
    "H": {
        "hydrophobicity": -3.2,
        "volume": 153.2,
        "charge": 0.5,  # Partially charged at pH 7
        "polarity": 1.0,
        "hydrophobicity_norm": -0.40,
        "volume_norm": 0.36,
        "charge_norm": 0.5,
        "polarity_norm": 0.5,
    },
    "I": {
        "hydrophobicity": 4.5,
        "volume": 166.7,
        "charge": 0.0,
        "polarity": 0.0,
        "hydrophobicity_norm": 1.38,
        "volume_norm": 0.30,
        "charge_norm": 0.0,
        "polarity_norm": -1.0,
    },
    "L": {
        "hydrophobicity": 3.8,
        "volume": 166.7,
        "charge": 0.0,
        "polarity": 0.0,
        "hydrophobicity_norm": 1.06,
        "volume_norm": 0.30,
        "charge_norm": 0.0,
        "polarity_norm": -1.0,
    },
    "K": {
        "hydrophobicity": -3.9,
        "volume": 168.6,
        "charge": 1.0,
        "polarity": 1.0,
        "hydrophobicity_norm": -1.50,
        "volume_norm": 0.53,
        "charge_norm": 1.0,
        "polarity_norm": 1.0,
    },
    "M": {
        "hydrophobicity": 1.9,
        "volume": 162.9,
        "charge": 0.0,
        "polarity": 0.0,
        "hydrophobicity_norm": 0.64,
        "volume_norm": 0.44,
        "charge_norm": 0.0,
        "polarity_norm": -0.5,
    },
    "F": {
        "hydrophobicity": 2.8,
        "volume": 189.9,
        "charge": 0.0,
        "polarity": 0.0,
        "hydrophobicity_norm": 1.19,
        "volume_norm": 0.77,
        "charge_norm": 0.0,
        "polarity_norm": -1.0,
    },
    "P": {
        "hydrophobicity": -1.6,
        "volume": 112.7,
        "charge": 0.0,
        "polarity": 0.0,
        "hydrophobicity_norm": 0.12,
        "volume_norm": -0.45,
        "charge_norm": 0.0,
        "polarity_norm": -0.5,
    },
    "S": {
        "hydrophobicity": -0.8,
        "volume": 89.0,
        "charge": 0.0,
        "polarity": 1.0,
        "hydrophobicity_norm": -0.18,
        "volume_norm": -0.60,
        "charge_norm": 0.0,
        "polarity_norm": 0.5,
    },
    "T": {
        "hydrophobicity": -0.7,
        "volume": 116.1,
        "charge": 0.0,
        "polarity": 1.0,
        "hydrophobicity_norm": -0.05,
        "volume_norm": -0.28,
        "charge_norm": 0.0,
        "polarity_norm": 0.5,
    },
    "W": {
        "hydrophobicity": -0.9,
        "volume": 227.8,
        "charge": 0.0,
        "polarity": 0.0,
        "hydrophobicity_norm": 0.81,
        "volume_norm": 1.00,
        "charge_norm": 0.0,
        "polarity_norm": -0.5,
    },
    "Y": {
        "hydrophobicity": -1.3,
        "volume": 193.6,
        "charge": 0.0,
        "polarity": 1.0,
        "hydrophobicity_norm": 0.26,
        "volume_norm": 0.77,
        "charge_norm": 0.0,
        "polarity_norm": 0.0,
    },
    "V": {
        "hydrophobicity": 4.2,
        "volume": 140.0,
        "charge": 0.0,
        "polarity": 0.0,
        "hydrophobicity_norm": 1.08,
        "volume_norm": 0.00,
        "charge_norm": 0.0,
        "polarity_norm": -1.0,
    },
    # Stop codon
    "*": {
        "hydrophobicity": 0.0,
        "volume": 0.0,
        "charge": 0.0,
        "polarity": 0.0,
        "hydrophobicity_norm": 0.0,
        "volume_norm": 0.0,
        "charge_norm": 0.0,
        "polarity_norm": 0.0,
    },
    # Unknown amino acid
    "X": {
        "hydrophobicity": 0.0,
        "volume": 0.0,
        "charge": 0.0,
        "polarity": 0.0,
        "hydrophobicity_norm": 0.0,
        "volume_norm": 0.0,
        "charge_norm": 0.0,
        "polarity_norm": 0.0,
    },
}

# Modified amino acids (for post-translational modifications)
MODIFIED_AMINO_ACIDS = {
    "Cit": {  # Citrulline (deiminated Arginine)
        "hydrophobicity": -3.0,
        "volume": 173.4,
        "charge": 0.0,  # Lost positive charge
        "polarity": 1.0,
    },
    "pSer": {  # Phosphoserine
        "hydrophobicity": -0.8,
        "volume": 89.0,
        "charge": -2.0,
        "polarity": 1.0,
    },
    "pThr": {  # Phosphothreonine
        "hydrophobicity": -0.7,
        "volume": 116.1,
        "charge": -2.0,
        "polarity": 1.0,
    },
    "pTyr": {  # Phosphotyrosine
        "hydrophobicity": -1.3,
        "volume": 193.6,
        "charge": -2.0,
        "polarity": 1.0,
    },
}


def get_amino_acid_property(
    aa: str,
    prop: str,
    normalized: bool = False,
    default: float | None = None,
) -> float:
    """Get a specific property for an amino acid.

    Args:
        aa: Single-letter amino acid code
        prop: Property name (hydrophobicity, volume, charge, polarity)
        normalized: If True, return normalized value for ML
        default: Default value if AA or property not found

    Returns:
        Property value
    """
    aa = aa.upper()

    if aa not in AMINO_ACID_PROPERTIES:
        if default is not None:
            return default
        aa = "X"  # Unknown

    props = AMINO_ACID_PROPERTIES[aa]
    key = f"{prop}_norm" if normalized else prop

    if key in props:
        return props[key]
    elif prop in props:
        return props[prop]
    elif default is not None:
        return default
    else:
        return 0.0


def get_amino_acid_charge(aa: str) -> float:
    """Get charge for amino acid.

    Args:
        aa: Single-letter amino acid code

    Returns:
        Charge at pH 7.0
    """
    return get_amino_acid_property(aa, "charge", default=0.0)


def get_normalized_properties(aa: str) -> tuple[float, float, float, float]:
    """Get normalized property tuple for amino acid.

    Returns tuple compatible with src/encoders/codon_encoder.py AA_PROPERTIES.

    Args:
        aa: Single-letter amino acid code

    Returns:
        (hydrophobicity_norm, charge_norm, volume_norm, polarity_norm)
    """
    return (
        get_amino_acid_property(aa, "hydrophobicity", normalized=True),
        get_amino_acid_property(aa, "charge", normalized=True),
        get_amino_acid_property(aa, "volume", normalized=True),
        get_amino_acid_property(aa, "polarity", normalized=True),
    )


# Legacy compatibility: AA_PROPERTIES format for codon_encoder.py
AA_PROPERTIES = {aa: get_normalized_properties(aa) for aa in STANDARD_AMINO_ACIDS + "*"}


__all__ = [
    "STANDARD_AMINO_ACIDS",
    "AMINO_ACID_3LETTER",
    "AMINO_ACID_PROPERTIES",
    "MODIFIED_AMINO_ACIDS",
    "AA_PROPERTIES",
    "get_amino_acid_property",
    "get_amino_acid_charge",
    "get_normalized_properties",
]
