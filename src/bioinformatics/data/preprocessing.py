# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Feature extraction and preprocessing for DDG prediction.

This module provides utilities for computing features from mutations,
including physicochemical properties and hyperbolic embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# Amino acid properties (canonical values from literature)
AA_HYDROPHOBICITY = {
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
}

AA_CHARGES = {
    "A": 0,
    "R": 1,
    "N": 0,
    "D": -1,
    "C": 0,
    "Q": 0,
    "E": -1,
    "G": 0,
    "H": 0.1,
    "I": 0,
    "L": 0,
    "K": 1,
    "M": 0,
    "F": 0,
    "P": 0,
    "S": 0,
    "T": 0,
    "W": 0,
    "Y": 0,
    "V": 0,
}

AA_VOLUMES = {
    "A": 88.6,
    "R": 173.4,
    "N": 114.1,
    "D": 111.1,
    "C": 108.5,
    "Q": 143.8,
    "E": 138.4,
    "G": 60.1,
    "H": 153.2,
    "I": 166.7,
    "L": 166.7,
    "K": 168.6,
    "M": 162.9,
    "F": 189.9,
    "P": 112.7,
    "S": 89.0,
    "T": 116.1,
    "W": 227.8,
    "Y": 193.6,
    "V": 140.0,
}

AA_FLEXIBILITY = {
    "A": 0.36,
    "R": 0.53,
    "N": 0.46,
    "D": 0.51,
    "C": 0.35,
    "Q": 0.49,
    "E": 0.50,
    "G": 0.54,
    "H": 0.32,
    "I": 0.46,
    "L": 0.37,
    "K": 0.47,
    "M": 0.30,
    "F": 0.31,
    "P": 0.51,
    "S": 0.51,
    "T": 0.44,
    "W": 0.31,
    "Y": 0.42,
    "V": 0.39,
}

AA_MASS = {
    "A": 89.1,
    "R": 174.2,
    "N": 132.1,
    "D": 133.1,
    "C": 121.2,
    "Q": 146.2,
    "E": 147.1,
    "G": 75.1,
    "H": 155.2,
    "I": 131.2,
    "L": 131.2,
    "K": 146.2,
    "M": 149.2,
    "F": 165.2,
    "P": 115.1,
    "S": 105.1,
    "T": 119.1,
    "W": 204.2,
    "Y": 181.2,
    "V": 117.1,
}


@dataclass
class MutationFeatures:
    """Container for mutation features."""

    # Physicochemical changes
    hydrophobicity_change: float
    charge_change: float
    volume_change: float
    flexibility_change: float
    mass_change: float

    # Categorical features
    is_charged_wt: bool
    is_charged_mut: bool
    is_aromatic_wt: bool
    is_aromatic_mut: bool
    is_hydrophobic_wt: bool
    is_hydrophobic_mut: bool
    is_proline: bool
    is_glycine: bool
    to_alanine: bool

    # Structural features (optional)
    secondary_structure: str | None = None
    solvent_accessibility: float | None = None

    # Hyperbolic features (optional)
    hyperbolic_distance: float | None = None
    wt_radius: float | None = None
    mut_radius: float | None = None
    delta_radius: float | None = None

    def to_array(self, include_hyperbolic: bool = True) -> np.ndarray:
        """Convert to numpy array for ML models."""
        features = [
            self.hydrophobicity_change,
            self.charge_change,
            self.volume_change,
            self.flexibility_change,
            self.mass_change,
            int(self.is_charged_wt),
            int(self.is_charged_mut),
            int(self.is_aromatic_wt),
            int(self.is_aromatic_mut),
            int(self.is_hydrophobic_wt),
            int(self.is_hydrophobic_mut),
            int(self.is_proline),
            int(self.is_glycine),
            int(self.to_alanine),
        ]

        # Add structural features if available
        if self.secondary_structure is not None:
            features.extend(
                [
                    1 if self.secondary_structure == "H" else 0,
                    1 if self.secondary_structure == "E" else 0,
                    1 if self.secondary_structure == "C" else 0,
                ]
            )
        if self.solvent_accessibility is not None:
            features.append(self.solvent_accessibility)
            features.append(1 if self.solvent_accessibility < 0.25 else 0)
            features.append(1 if self.solvent_accessibility > 0.5 else 0)

        # Add hyperbolic features if available and requested
        if include_hyperbolic and self.hyperbolic_distance is not None:
            features.extend(
                [
                    self.hyperbolic_distance,
                    self.wt_radius or 0,
                    self.mut_radius or 0,
                    self.delta_radius or 0,
                ]
            )

        return np.array(features, dtype=np.float32)

    @property
    def feature_names(self) -> list[str]:
        """Get feature names in order."""
        names = [
            "hydrophobicity_change",
            "charge_change",
            "volume_change",
            "flexibility_change",
            "mass_change",
            "is_charged_wt",
            "is_charged_mut",
            "is_aromatic_wt",
            "is_aromatic_mut",
            "is_hydrophobic_wt",
            "is_hydrophobic_mut",
            "is_proline",
            "is_glycine",
            "to_alanine",
        ]

        if self.secondary_structure is not None:
            names.extend(["ss_helix", "ss_sheet", "ss_coil"])
        if self.solvent_accessibility is not None:
            names.extend(["rsa", "is_buried", "is_surface"])
        if self.hyperbolic_distance is not None:
            names.extend(["hyp_distance", "wt_radius", "mut_radius", "delta_radius"])

        return names


def compute_features(
    wild_type: str,
    mutant: str,
    secondary_structure: str | None = None,
    solvent_accessibility: float | None = None,
) -> MutationFeatures:
    """Compute physicochemical features for a mutation.

    Args:
        wild_type: Single-letter wild-type amino acid
        mutant: Single-letter mutant amino acid
        secondary_structure: H/E/C (optional)
        solvent_accessibility: RSA 0-1 (optional)

    Returns:
        MutationFeatures object
    """
    wt = wild_type.upper()
    mut = mutant.upper()

    return MutationFeatures(
        hydrophobicity_change=AA_HYDROPHOBICITY.get(mut, 0) - AA_HYDROPHOBICITY.get(wt, 0),
        charge_change=AA_CHARGES.get(mut, 0) - AA_CHARGES.get(wt, 0),
        volume_change=AA_VOLUMES.get(mut, 100) - AA_VOLUMES.get(wt, 100),
        flexibility_change=AA_FLEXIBILITY.get(mut, 0.4) - AA_FLEXIBILITY.get(wt, 0.4),
        mass_change=AA_MASS.get(mut, 100) - AA_MASS.get(wt, 100),
        is_charged_wt=wt in "KRHDE",
        is_charged_mut=mut in "KRHDE",
        is_aromatic_wt=wt in "FWY",
        is_aromatic_mut=mut in "FWY",
        is_hydrophobic_wt=wt in "AILMFVW",
        is_hydrophobic_mut=mut in "AILMFVW",
        is_proline=wt == "P" or mut == "P",
        is_glycine=wt == "G" or mut == "G",
        to_alanine=mut == "A",
        secondary_structure=secondary_structure,
        solvent_accessibility=solvent_accessibility,
    )


def compute_hyperbolic_features(
    wild_type: str,
    mutant: str,
    aa_embeddings: dict[str, torch.Tensor],
    curvature: float = 1.0,
) -> tuple[float, float, float, float]:
    """Compute hyperbolic embedding features for a mutation.

    Uses the Poincaré ball model with proper hyperbolic distance.

    Args:
        wild_type: Single-letter wild-type amino acid
        mutant: Single-letter mutant amino acid
        aa_embeddings: Dictionary mapping AA to hyperbolic embeddings
        curvature: Poincaré ball curvature

    Returns:
        Tuple of (hyperbolic_distance, wt_radius, mut_radius, delta_radius)
    """
    from src.geometry import poincare_distance

    wt = wild_type.upper()
    mut = mutant.upper()

    if wt not in aa_embeddings or mut not in aa_embeddings:
        return 0.0, 0.0, 0.0, 0.0

    wt_emb = aa_embeddings[wt]
    mut_emb = aa_embeddings[mut]

    # Ensure correct shape
    if wt_emb.dim() == 1:
        wt_emb = wt_emb.unsqueeze(0)
    if mut_emb.dim() == 1:
        mut_emb = mut_emb.unsqueeze(0)

    # Hyperbolic distance between embeddings
    hyp_dist = poincare_distance(wt_emb, mut_emb, c=curvature).item()

    # Distance from origin (hyperbolic radius)
    origin = torch.zeros_like(wt_emb)
    wt_radius = poincare_distance(wt_emb, origin, c=curvature).item()
    mut_radius = poincare_distance(mut_emb, origin, c=curvature).item()
    delta_radius = mut_radius - wt_radius

    return hyp_dist, wt_radius, mut_radius, delta_radius


def add_hyperbolic_features(
    features: MutationFeatures,
    wild_type: str,
    mutant: str,
    aa_embeddings: dict[str, torch.Tensor],
    curvature: float = 1.0,
) -> MutationFeatures:
    """Add hyperbolic features to existing MutationFeatures.

    Args:
        features: Existing MutationFeatures object
        wild_type: Wild-type amino acid
        mutant: Mutant amino acid
        aa_embeddings: Dictionary of AA embeddings
        curvature: Poincaré ball curvature

    Returns:
        MutationFeatures with hyperbolic features added
    """
    hyp_dist, wt_r, mut_r, delta_r = compute_hyperbolic_features(wild_type, mutant, aa_embeddings, curvature)

    features.hyperbolic_distance = hyp_dist
    features.wt_radius = wt_r
    features.mut_radius = mut_r
    features.delta_radius = delta_r

    return features


__all__ = [
    "AA_HYDROPHOBICITY",
    "AA_CHARGES",
    "AA_VOLUMES",
    "AA_FLEXIBILITY",
    "AA_MASS",
    "MutationFeatures",
    "compute_features",
    "compute_hyperbolic_features",
    "add_hyperbolic_features",
]
