# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""PTM-Goldilocks Encoder for Post-Translational Modification Analysis.

This module extends the CodonEncoder with PTM (Post-Translational Modification)
support and Goldilocks zone tracking for immunogenicity prediction.

Key Features:
- Embeds amino acids with PTM state awareness
- Tracks modifications in hyperbolic (Poincare ball) space
- Computes Goldilocks zone membership for immunogenicity
- Supports multiple PTM types (citrullination, phosphorylation, glycosylation, etc.)

Usage:
    from src.encoders.ptm_encoder import PTMGoldilocksEncoder

    encoder = PTMGoldilocksEncoder(embedding_dim=16)
    z = encoder(amino_acid_indices, ptm_states)
    in_goldilocks = encoder.compute_goldilocks_membership(z)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn as nn

from src.geometry.poincare import PoincareModule, poincare_distance


class PTMType(IntEnum):
    """Enumeration of supported Post-Translational Modification types."""

    NONE = 0
    CITRULLINATION = 1  # Arg -> Cit (key in RA)
    PHOSPHORYLATION = 2  # Ser/Thr/Tyr phosphorylation
    GLYCOSYLATION = 3  # N-linked or O-linked glycans
    ACETYLATION = 4  # Lysine acetylation
    METHYLATION = 5  # Arg/Lys methylation
    UBIQUITINATION = 6  # Ubiquitin attachment
    SUMOYLATION = 7  # SUMO attachment
    DEAMIDATION = 8  # Asn/Gln deamidation
    OXIDATION = 9  # Met/Cys oxidation


@dataclass(frozen=True)
class GoldilocksZone:
    """Goldilocks zone boundaries for immunogenicity.

    Based on RA citrullination research:
    - Entropy change (ΔH) range where PTMs become immunogenic
    - Too little change = "self" (not recognized)
    - Too much change = "foreign" (rapidly cleared)
    - Goldilocks zone = "modified-self" (immunogenic)

    Validated values from RA analysis:
        alpha = -0.1205 (lower bound)
        beta = +0.0495 (upper bound)
    """

    alpha: float = -0.1205  # Lower entropy bound
    beta: float = 0.0495  # Upper entropy bound
    centroid_threshold: float = 0.05  # Max centroid shift for Goldilocks


class PTMGoldilocksEncoder(PoincareModule):
    """Encoder for amino acids with PTM state awareness and Goldilocks tracking.

    This encoder maps (amino_acid, ptm_type) pairs to hyperbolic embeddings,
    enabling geometric analysis of how PTMs shift peptide representations
    relative to immunogenic zones.

    Architecture:
        1. Amino acid embedding (20 canonical + special tokens)
        2. PTM type embedding (modular, extensible)
        3. Fusion layer (combines AA + PTM)
        4. Hyperbolic projection (Poincare ball)
        5. Goldilocks zone computation

    Args:
        embedding_dim: Size of output embeddings
        num_amino_acids: Number of amino acids (20 + padding/unknown)
        num_ptm_types: Number of PTM types supported
        curvature: Hyperbolic curvature parameter
        max_norm: Maximum norm on Poincare ball (< 1.0)
        use_hierarchical_init: Initialize with amino acid property hierarchy
        goldilocks_zone: Custom Goldilocks zone bounds
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        num_amino_acids: int = 22,  # 20 + PAD + UNK
        num_ptm_types: int = 10,
        curvature: float = 1.0,
        max_norm: float = 0.95,
        use_hierarchical_init: bool = True,
        goldilocks_zone: GoldilocksZone | None = None,
    ):
        super().__init__(c=curvature, max_norm=max_norm)

        self.embedding_dim = embedding_dim
        self.num_amino_acids = num_amino_acids
        self.num_ptm_types = num_ptm_types
        self.goldilocks = goldilocks_zone or GoldilocksZone()

        # Amino acid embedding
        self.aa_embedding = nn.Embedding(
            num_embeddings=num_amino_acids,
            embedding_dim=embedding_dim,
            padding_idx=0,  # 0 = PAD
        )

        # PTM type embedding
        self.ptm_embedding = nn.Embedding(
            num_embeddings=num_ptm_types,
            embedding_dim=embedding_dim // 2,
        )

        # Fusion layer (AA + PTM -> final embedding)
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim + embedding_dim // 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Hyperbolic projection
        self.proj_layer = nn.Linear(embedding_dim, embedding_dim)

        if use_hierarchical_init:
            self._init_hierarchical_weights()

    def _init_hierarchical_weights(self) -> None:
        """Initialize amino acid embeddings with biochemical property hierarchy.

        Groups amino acids by:
        - Charge (positive, negative, neutral)
        - Hydrophobicity
        - Size/bulk
        - Aromaticity
        """
        # Amino acid groups based on biochemical properties
        # Indices: A=1, C=2, D=3, E=4, F=5, G=6, H=7, I=8, K=9, L=10,
        #          M=11, N=12, P=13, Q=14, R=15, S=16, T=17, V=18, W=19, Y=20
        # 0=PAD, 21=UNK

        aa_properties = {
            # (charge, hydrophobicity, size, aromaticity)
            1: [0, 0.5, -0.5, 0],  # A - Alanine
            2: [0, 0.3, -0.3, 0],  # C - Cysteine
            3: [-1, -1, 0, 0],  # D - Aspartic acid
            4: [-1, -1, 0.3, 0],  # E - Glutamic acid
            5: [0, 1, 0.8, 1],  # F - Phenylalanine
            6: [0, 0, -1, 0],  # G - Glycine
            7: [0.5, -0.5, 0.5, 0.5],  # H - Histidine
            8: [0, 1, 0.5, 0],  # I - Isoleucine
            9: [1, -1, 0.5, 0],  # K - Lysine
            10: [0, 1, 0.5, 0],  # L - Leucine
            11: [0, 0.7, 0.5, 0],  # M - Methionine
            12: [0, -0.5, 0, 0],  # N - Asparagine
            13: [0, 0.3, 0, 0],  # P - Proline
            14: [0, -0.5, 0.3, 0],  # Q - Glutamine
            15: [1, -1, 0.8, 0],  # R - Arginine (key for citrullination)
            16: [0, -0.5, -0.5, 0],  # S - Serine
            17: [0, -0.3, -0.3, 0],  # T - Threonine
            18: [0, 1, 0.3, 0],  # V - Valine
            19: [0, 1, 1, 1],  # W - Tryptophan
            20: [0, 0.7, 0.8, 0.8],  # Y - Tyrosine
        }

        with torch.no_grad():
            weights = self.aa_embedding.weight.data
            scale = 0.3

            for aa_idx, props in aa_properties.items():
                if aa_idx < self.num_amino_acids and self.embedding_dim >= 4:
                    # First 4 dims: biochemical properties
                    weights[aa_idx, 0] = props[0] * scale  # Charge
                    weights[aa_idx, 1] = props[1] * scale  # Hydrophobicity
                    weights[aa_idx, 2] = props[2] * scale  # Size
                    weights[aa_idx, 3] = props[3] * scale  # Aromaticity

    def forward(
        self,
        aa_indices: torch.Tensor,
        ptm_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed amino acids with optional PTM states.

        Args:
            aa_indices: Amino acid indices, shape (Batch, SeqLen)
            ptm_states: PTM type indices, shape (Batch, SeqLen)
                       If None, assumes no modifications (PTMType.NONE)

        Returns:
            Hyperbolic embeddings on Poincare ball, shape (Batch, SeqLen, Dim)
        """
        B, L = aa_indices.shape

        # Default to no PTM if not provided
        if ptm_states is None:
            ptm_states = torch.zeros_like(aa_indices)

        # Embed amino acids and PTM types
        aa_emb = self.aa_embedding(aa_indices)  # (B, L, D)
        ptm_emb = self.ptm_embedding(ptm_states)  # (B, L, D//2)

        # Fuse embeddings
        combined = torch.cat([aa_emb, ptm_emb], dim=-1)  # (B, L, D + D//2)
        fused = self.fusion(combined)  # (B, L, D)

        # Project to hyperbolic space
        projected = self.proj_layer(fused)
        z_hyp = self.proj(projected)  # Project to Poincare ball

        return z_hyp

    def compute_entropy_change(
        self,
        z_native: torch.Tensor,
        z_modified: torch.Tensor,
    ) -> torch.Tensor:
        """Compute entropy change (ΔH) between native and modified embeddings.

        The entropy change measures how much a PTM shifts the embedding
        in hyperbolic space, which correlates with immunogenicity.

        Args:
            z_native: Native (unmodified) embeddings, shape (..., Dim)
            z_modified: PTM-modified embeddings, shape (..., Dim)

        Returns:
            Entropy change values, shape (...)
        """
        # Compute Poincare distance as proxy for entropy change
        dist = self.dist(z_native, z_modified)

        # V5.12.2: Use hyperbolic distance from origin for radial comparison
        origin_native = torch.zeros_like(z_native)
        origin_modified = torch.zeros_like(z_modified)
        native_hyp_dist = poincare_distance(z_native, origin_native, c=self.c)
        modified_hyp_dist = poincare_distance(z_modified, origin_modified, c=self.c)

        # Direction of shift (negative = towards origin, positive = away)
        direction = torch.sign(modified_hyp_dist - native_hyp_dist)

        return direction * dist

    def compute_goldilocks_membership(
        self,
        entropy_change: torch.Tensor,
        return_distance: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Determine if entropy changes fall within the Goldilocks zone.

        Args:
            entropy_change: Entropy change values, shape (...)
            return_distance: If True, also return distance to zone center

        Returns:
            in_zone: Boolean mask of Goldilocks membership
            distance: Optional distance to zone center (if return_distance=True)
        """
        alpha = self.goldilocks.alpha
        beta = self.goldilocks.beta

        in_zone = (entropy_change >= alpha) & (entropy_change <= beta)

        if return_distance:
            # Distance to zone center
            zone_center = (alpha + beta) / 2
            zone_distance = torch.abs(entropy_change - zone_center)
            return in_zone, zone_distance

        return in_zone, None

    def compute_immunogenicity_score(
        self,
        z_native: torch.Tensor,
        z_modified: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Comprehensive immunogenicity analysis.

        Computes multiple metrics relevant to immunogenicity prediction:
        - Entropy change (ΔH)
        - Goldilocks zone membership
        - Distance to zone center
        - Centroid shift

        Args:
            z_native: Native embeddings, shape (B, L, D)
            z_modified: Modified embeddings, shape (B, L, D)

        Returns:
            Dict containing:
                - entropy_change: ΔH values
                - in_goldilocks: Boolean membership
                - zone_distance: Distance to zone center
                - centroid_shift: Shift in embedding centroid
                - immunogenicity_score: Combined score (0-1)
        """
        # Compute entropy change
        delta_h = self.compute_entropy_change(z_native, z_modified)

        # Goldilocks membership
        in_zone, zone_dist = self.compute_goldilocks_membership(delta_h, return_distance=True)

        # Centroid shift
        native_centroid = z_native.mean(dim=1)  # (B, D)
        modified_centroid = z_modified.mean(dim=1)  # (B, D)
        centroid_shift = self.dist(native_centroid, modified_centroid)

        # Combined immunogenicity score
        # Higher when in Goldilocks zone and close to center
        zone_width = self.goldilocks.beta - self.goldilocks.alpha
        normalized_dist = zone_dist / (zone_width / 2 + 1e-8)
        score = torch.where(
            in_zone,
            1.0 - normalized_dist.clamp(0, 1),  # In zone: higher = closer to center
            torch.zeros_like(normalized_dist),  # Out of zone: 0
        )

        return {
            "entropy_change": delta_h,
            "in_goldilocks": in_zone,
            "zone_distance": zone_dist,
            "centroid_shift": centroid_shift,
            "immunogenicity_score": score.mean(dim=-1),  # Average over sequence
        }

    def get_ptm_shift_vector(
        self,
        aa_index: int,
        ptm_type: int,
    ) -> torch.Tensor:
        """Get the embedding shift vector for a specific PTM.

        Useful for analyzing how different PTMs affect specific amino acids.

        Args:
            aa_index: Amino acid index (1-20)
            ptm_type: PTM type index (from PTMType enum)

        Returns:
            Shift vector in embedding space, shape (Dim,)
        """
        device = self.aa_embedding.weight.device

        # Create single-element inputs
        aa = torch.tensor([[aa_index]], device=device)
        no_ptm = torch.tensor([[PTMType.NONE]], device=device)
        with_ptm = torch.tensor([[ptm_type]], device=device)

        # Get embeddings
        z_native = self.forward(aa, no_ptm).squeeze()
        z_modified = self.forward(aa, with_ptm).squeeze()

        return z_modified - z_native


class PTMDataset(torch.utils.data.Dataset):
    """Dataset for PTM-Goldilocks encoder training.

    Handles paired (native, modified) sequences for contrastive learning.
    """

    def __init__(
        self,
        sequences: list[str],
        ptm_annotations: list[dict],
        max_length: int = 512,
    ):
        """Initialize PTM dataset.

        Args:
            sequences: List of amino acid sequences
            ptm_annotations: List of dicts with PTM positions and types
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.ptm_annotations = ptm_annotations
        self.max_length = max_length

        # Amino acid to index mapping
        self.aa_to_idx = {
            "A": 1,
            "C": 2,
            "D": 3,
            "E": 4,
            "F": 5,
            "G": 6,
            "H": 7,
            "I": 8,
            "K": 9,
            "L": 10,
            "M": 11,
            "N": 12,
            "P": 13,
            "Q": 14,
            "R": 15,
            "S": 16,
            "T": 17,
            "V": 18,
            "W": 19,
            "Y": 20,
            "X": 21,  # Unknown
        }

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        ptm_info = self.ptm_annotations[idx]

        # Convert sequence to indices
        aa_indices = [self.aa_to_idx.get(aa, 21) for aa in seq[: self.max_length]]
        aa_indices = aa_indices + [0] * (self.max_length - len(aa_indices))

        # Create PTM state tensor
        ptm_states = [PTMType.NONE] * self.max_length
        for pos, ptm_type in ptm_info.get("modifications", []):
            if pos < self.max_length:
                ptm_states[pos] = ptm_type

        return {
            "aa_indices": torch.tensor(aa_indices, dtype=torch.long),
            "ptm_states": torch.tensor(ptm_states, dtype=torch.long),
            "labels": torch.tensor(ptm_info.get("immunogenic", 0), dtype=torch.float),
        }


__all__ = [
    "PTMType",
    "GoldilocksZone",
    "PTMGoldilocksEncoder",
    "PTMDataset",
]
