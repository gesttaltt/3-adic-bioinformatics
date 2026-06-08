# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""P-adic Amino Acid Encoders.

Implements multiple prime-based encodings for amino acids that preserve
biochemical properties in an ultrametric space.

Key Encoders:
- FiveAdicAminoAcidEncoder: 5-adic encoding based on physicochemical groups
- SevenAdicSecondaryStructureEncoder: 7-adic encoding for secondary structure
- MultiPrimeAminoAcidEncoder: Combined multi-prime fusion encoder

Mathematical Foundation:
    The 5-adic encoding groups 20 amino acids + stop into 5 physicochemical
    classes (hydrophobic, polar, positive, negative, special), with distance
    based on class similarity and property differences.

References:
    - Roch, "Ultrametric theory of protein evolution" (2019)
    - Holly, "p-adic analysis and mathematical biology" (2001)
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# =============================================================================
# Amino Acid Classifications
# =============================================================================


class AminoAcidGroup(IntEnum):
    """Amino acid physicochemical groups for 5-adic encoding.

    Groups amino acids into 5 classes based on side chain properties:
    - HYDROPHOBIC: Nonpolar, hydrophobic side chains
    - POLAR: Polar, uncharged side chains
    - POSITIVE: Positively charged (basic) side chains
    - NEGATIVE: Negatively charged (acidic) side chains
    - SPECIAL: Special structure (Gly, Pro, Cys)
    """

    HYDROPHOBIC = 0  # A, V, L, I, M, F, W, Y
    POLAR = 1  # S, T, N, Q
    POSITIVE = 2  # K, R, H
    NEGATIVE = 3  # D, E
    SPECIAL = 4  # G, P, C, *


# Amino acid to group mapping
AA_TO_GROUP: dict[str, AminoAcidGroup] = {
    "A": AminoAcidGroup.HYDROPHOBIC,
    "V": AminoAcidGroup.HYDROPHOBIC,
    "L": AminoAcidGroup.HYDROPHOBIC,
    "I": AminoAcidGroup.HYDROPHOBIC,
    "M": AminoAcidGroup.HYDROPHOBIC,
    "F": AminoAcidGroup.HYDROPHOBIC,
    "W": AminoAcidGroup.HYDROPHOBIC,
    "Y": AminoAcidGroup.HYDROPHOBIC,
    "S": AminoAcidGroup.POLAR,
    "T": AminoAcidGroup.POLAR,
    "N": AminoAcidGroup.POLAR,
    "Q": AminoAcidGroup.POLAR,
    "K": AminoAcidGroup.POSITIVE,
    "R": AminoAcidGroup.POSITIVE,
    "H": AminoAcidGroup.POSITIVE,
    "D": AminoAcidGroup.NEGATIVE,
    "E": AminoAcidGroup.NEGATIVE,
    "G": AminoAcidGroup.SPECIAL,
    "P": AminoAcidGroup.SPECIAL,
    "C": AminoAcidGroup.SPECIAL,
    "*": AminoAcidGroup.SPECIAL,  # Stop codon
    "X": AminoAcidGroup.SPECIAL,  # Unknown
}

# Amino acid index mapping (0-20)
AA_TO_INDEX: dict[str, int] = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "Q": 5,
    "E": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
    "*": 20,
    "X": 21,
}

INDEX_TO_AA: dict[int, str] = {v: k for k, v in AA_TO_INDEX.items()}


# Physicochemical properties
# (hydrophobicity, molecular_weight, isoelectric_point, flexibility)
AA_PROPERTIES: dict[str, tuple[float, float, float, float]] = {
    "A": (1.8, 89, 6.01, 0.36),
    "R": (-4.5, 174, 10.76, 0.53),
    "N": (-3.5, 132, 5.41, 0.46),
    "D": (-3.5, 133, 2.77, 0.51),
    "C": (2.5, 121, 5.07, 0.35),
    "Q": (-3.5, 146, 5.65, 0.49),
    "E": (-3.5, 147, 3.22, 0.50),
    "G": (-0.4, 75, 5.97, 0.54),
    "H": (-3.2, 155, 7.59, 0.32),
    "I": (4.5, 131, 6.02, 0.46),
    "L": (3.8, 131, 5.98, 0.40),
    "K": (-3.9, 146, 9.74, 0.47),
    "M": (1.9, 149, 5.74, 0.30),
    "F": (2.8, 165, 5.48, 0.31),
    "P": (-1.6, 115, 6.30, 0.51),
    "S": (-0.8, 105, 5.68, 0.51),
    "T": (-0.7, 119, 5.60, 0.44),
    "W": (-0.9, 204, 5.89, 0.31),
    "Y": (-1.3, 181, 5.66, 0.42),
    "V": (4.2, 117, 5.97, 0.39),
    "*": (0.0, 0, 7.0, 0.5),
    "X": (0.0, 100, 7.0, 0.5),
}


# =============================================================================
# 5-adic Distance Functions
# =============================================================================


def compute_5adic_distance(aa1: str, aa2: str) -> float:
    """Compute 5-adic distance between two amino acids.

    The distance is based on the group hierarchy:
    - Same amino acid: 0
    - Same group: 1/5
    - Different groups: 1

    Args:
        aa1: First amino acid (single letter)
        aa2: Second amino acid (single letter)

    Returns:
        5-adic distance (0 to 1)
    """
    if aa1 == aa2:
        return 0.0

    group1 = AA_TO_GROUP.get(aa1, AminoAcidGroup.SPECIAL)
    group2 = AA_TO_GROUP.get(aa2, AminoAcidGroup.SPECIAL)

    if group1 == group2:
        # Same group, different AA: use property-based sub-distance
        props1 = np.array(AA_PROPERTIES.get(aa1, (0, 100, 7, 0.5)))
        props2 = np.array(AA_PROPERTIES.get(aa2, (0, 100, 7, 0.5)))

        # Normalize properties
        props1_norm = props1 / np.array([10, 250, 14, 1])
        props2_norm = props2 / np.array([10, 250, 14, 1])

        prop_dist = np.linalg.norm(props1_norm - props2_norm)
        # Scale to (0, 1/5)
        return min(prop_dist * 0.2, 0.2)
    else:
        # Different groups
        return 1.0


def compute_5adic_distance_matrix() -> np.ndarray:
    """Compute full 22x22 5-adic distance matrix.

    Returns:
        Distance matrix (22, 22)
    """
    n_aa = 22  # 20 AA + stop + unknown
    matrix = np.zeros((n_aa, n_aa))

    for i, aa1 in INDEX_TO_AA.items():
        for j, aa2 in INDEX_TO_AA.items():
            matrix[i, j] = compute_5adic_distance(aa1, aa2)

    return matrix


# =============================================================================
# 5-adic Amino Acid Encoder
# =============================================================================


class FiveAdicAminoAcidEncoder(nn.Module):
    """5-adic Amino Acid Encoder.

    Maps amino acids to a latent space preserving 5-adic ultrametric
    structure based on physicochemical groupings.

    Architecture:
        1. Group embedding (5 groups -> embedding_dim)
        2. Property embedding (4 properties -> embedding_dim)
        3. Position-aware modulation (optional)
        4. Fusion layer

    Example:
        >>> encoder = FiveAdicAminoAcidEncoder(embedding_dim=64)
        >>> indices = torch.tensor([0, 1, 2])  # A, R, N
        >>> embeddings = encoder(indices)
        >>> print(embeddings.shape)  # (3, 64)
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        n_amino_acids: int = 22,
        use_properties: bool = True,
        use_mds_init: bool = True,
        dropout: float = 0.1,
    ):
        """Initialize 5-adic encoder.

        Args:
            embedding_dim: Output embedding dimension
            n_amino_acids: Number of amino acids (22 = 20 + stop + unknown)
            use_properties: Include physicochemical property embedding
            use_mds_init: Initialize with MDS from distance matrix
            dropout: Dropout rate
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_amino_acids = n_amino_acids
        self.use_properties = use_properties

        # Group embedding (5 groups)
        self.group_embedding = nn.Embedding(5, embedding_dim // 2)

        # Amino acid embedding
        self.aa_embedding = nn.Embedding(n_amino_acids, embedding_dim // 2)

        # Property encoder (optional)
        if use_properties:
            self.property_encoder = nn.Sequential(
                nn.Linear(4, embedding_dim // 4),
                nn.GELU(),
                nn.Linear(embedding_dim // 4, embedding_dim // 2),
            )
            self.fusion = nn.Linear(embedding_dim + embedding_dim // 2, embedding_dim)
        else:
            self.fusion = nn.Linear(embedding_dim, embedding_dim)

        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Register property tensor
        self._register_properties()

        # Initialize with MDS if requested
        if use_mds_init:
            self._initialize_with_mds()

    def _register_properties(self) -> None:
        """Register amino acid properties as buffer."""
        props = torch.zeros(self.n_amino_acids, 4)
        for aa, idx in AA_TO_INDEX.items():
            if idx < self.n_amino_acids:
                p = AA_PROPERTIES.get(aa, (0, 100, 7, 0.5))
                # Normalize
                props[idx] = torch.tensor(
                    [
                        p[0] / 10,  # hydrophobicity
                        p[1] / 250,  # molecular weight
                        p[2] / 14,  # isoelectric point
                        p[3],  # flexibility
                    ]
                )
        self.register_buffer("aa_properties", props)

        # Register group mapping
        groups = torch.zeros(self.n_amino_acids, dtype=torch.long)
        for aa, idx in AA_TO_INDEX.items():
            if idx < self.n_amino_acids:
                groups[idx] = AA_TO_GROUP.get(aa, AminoAcidGroup.SPECIAL)
        self.register_buffer("aa_groups", groups)

    def _initialize_with_mds(self) -> None:
        """Initialize embeddings using MDS from 5-adic distance matrix."""
        try:
            from sklearn.manifold import MDS

            # Compute distance matrix
            dist_matrix = compute_5adic_distance_matrix()

            # Apply MDS
            mds = MDS(
                n_components=min(self.embedding_dim // 2, 10),
                dissimilarity="precomputed",
                random_state=42,
                normalized_stress="auto",
            )
            mds_coords = mds.fit_transform(dist_matrix)

            # Pad to embedding dimension
            if mds_coords.shape[1] < self.embedding_dim // 2:
                padding = np.zeros((mds_coords.shape[0], self.embedding_dim // 2 - mds_coords.shape[1]))
                mds_coords = np.concatenate([mds_coords, padding], axis=1)

            # Initialize embedding
            with torch.no_grad():
                self.aa_embedding.weight[:] = torch.from_numpy(mds_coords).float()

        except ImportError:
            pass  # sklearn not available, use random init

    def forward(
        self,
        indices: Tensor,
        return_components: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Encode amino acid indices.

        Args:
            indices: Amino acid indices (batch_size,) or (batch_size, seq_len)
            return_components: Return intermediate components

        Returns:
            Embeddings (batch_size, embedding_dim) or
            (batch_size, seq_len, embedding_dim) if sequence input
        """
        # Get group indices
        groups = self.aa_groups[indices]

        # Embeddings
        group_embed = self.group_embedding(groups)
        aa_embed = self.aa_embedding(indices)

        # Concatenate group and AA embeddings
        combined = torch.cat([group_embed, aa_embed], dim=-1)

        # Add property embedding if enabled
        if self.use_properties:
            props = self.aa_properties[indices]
            prop_embed = self.property_encoder(props)
            combined = torch.cat([combined, prop_embed], dim=-1)

        # Fusion
        output = self.fusion(combined)
        output = self.norm(output)
        output = self.dropout(output)

        if return_components:
            return output, {
                "group_embed": group_embed,
                "aa_embed": aa_embed,
                "groups": groups,
            }

        return output

    def get_distance_matrix(self) -> Tensor:
        """Get pairwise distance matrix in embedding space.

        Returns:
            Distance matrix (n_amino_acids, n_amino_acids)
        """
        indices = torch.arange(self.n_amino_acids, device=self.aa_embedding.weight.device)
        embeddings = self.forward(indices)

        # Compute pairwise distances
        dist = torch.cdist(embeddings.unsqueeze(0), embeddings.unsqueeze(0)).squeeze(0)
        return dist

    def get_5adic_distance_matrix(self) -> np.ndarray:
        """Get theoretical 5-adic distance matrix.

        Returns:
            Numpy distance matrix (22, 22)
        """
        return compute_5adic_distance_matrix()


# =============================================================================
# 7-adic Secondary Structure Encoder
# =============================================================================


class SecondaryStructure(IntEnum):
    """DSSP secondary structure types for 7-adic encoding."""

    HELIX_ALPHA = 0  # H - alpha helix
    HELIX_310 = 1  # G - 3-10 helix
    HELIX_PI = 2  # I - pi helix
    SHEET = 3  # E - extended strand (beta sheet)
    BRIDGE = 4  # B - isolated beta bridge
    TURN = 5  # T - turn
    COIL = 6  # C/S - coil/other


class SevenAdicSecondaryStructureEncoder(nn.Module):
    """7-adic Secondary Structure Encoder.

    Maps secondary structure elements (7 DSSP classes) to embeddings
    using a 7-adic metric that groups related structures.

    Hierarchy:
        - Helices (H, G, I): Close in 7-adic space
        - Extended (E, B): Close in 7-adic space
        - Other (T, C): Distant from regular structures
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        combine_with_sequence: bool = True,
    ):
        """Initialize 7-adic structure encoder.

        Args:
            embedding_dim: Output dimension
            combine_with_sequence: Whether to combine with sequence embeddings
        """
        super().__init__()

        self.embedding_dim = embedding_dim

        # Structure type embedding
        self.structure_embedding = nn.Embedding(7, embedding_dim)

        # Hierarchical group embedding (3 groups: helix, sheet, other)
        self.group_embedding = nn.Embedding(3, embedding_dim // 2)

        # Fusion
        self.fusion = nn.Linear(embedding_dim + embedding_dim // 2, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

        # Group mapping (structure -> group)
        self.register_buffer(
            "structure_to_group",
            torch.tensor([0, 0, 0, 1, 1, 2, 2], dtype=torch.long),
        )

        self._initialize_7adic()

    def _initialize_7adic(self) -> None:
        """Initialize with 7-adic structure."""
        # Helix types should be close
        # Sheet types should be close
        # Turns and coils should be distinct

        with torch.no_grad():
            # Initialize with structured pattern
            base = torch.randn(7, self.embedding_dim) * 0.1

            # Make helices similar
            helix_center = torch.randn(self.embedding_dim)
            for i in [0, 1, 2]:  # H, G, I
                base[i] = helix_center + torch.randn(self.embedding_dim) * 0.05

            # Make sheets similar
            sheet_center = torch.randn(self.embedding_dim)
            for i in [3, 4]:  # E, B
                base[i] = sheet_center + torch.randn(self.embedding_dim) * 0.05

            self.structure_embedding.weight.copy_(base)

    def forward(self, structure_indices: Tensor) -> Tensor:
        """Encode secondary structure indices.

        Args:
            structure_indices: Structure type indices (0-6)

        Returns:
            Embeddings
        """
        groups = self.structure_to_group[structure_indices]

        struct_embed = self.structure_embedding(structure_indices)
        group_embed = self.group_embedding(groups)

        combined = torch.cat([struct_embed, group_embed], dim=-1)
        output = self.fusion(combined)
        output = self.norm(output)

        return output


# =============================================================================
# Multi-Prime Amino Acid Encoder
# =============================================================================


class MultiPrimeAminoAcidEncoder(nn.Module):
    """Multi-Prime Fusion Encoder.

    Combines multiple p-adic encodings (2, 3, 5, 7) for comprehensive
    amino acid representation.

    Each prime captures different aspects:
        - 2-adic: Binary properties (polar/nonpolar)
        - 3-adic: Codon structure
        - 5-adic: Physicochemical groups
        - 7-adic: Secondary structure tendency
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        use_attention_fusion: bool = True,
    ):
        """Initialize multi-prime encoder.

        Args:
            embedding_dim: Output embedding dimension
            use_attention_fusion: Use attention-based fusion
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        component_dim = embedding_dim // 4

        # Individual encoders
        self.encoder_5adic = FiveAdicAminoAcidEncoder(
            embedding_dim=component_dim,
            use_properties=True,
        )

        # Simple embeddings for other primes
        self.encoder_2adic = nn.Embedding(22, component_dim)  # Polar/nonpolar
        self.encoder_3adic = nn.Embedding(22, component_dim)  # Codon-based
        self.encoder_7adic = nn.Embedding(22, component_dim)  # Structure tendency

        # Fusion
        if use_attention_fusion:
            self.fusion = nn.MultiheadAttention(
                embed_dim=component_dim,
                num_heads=4,
                batch_first=True,
            )
            self.fusion_proj = nn.Linear(component_dim, embedding_dim)
        else:
            self.fusion = None
            self.fusion_proj = nn.Linear(component_dim * 4, embedding_dim)

        self.norm = nn.LayerNorm(embedding_dim)

        self._initialize_primes()

    def _initialize_primes(self) -> None:
        """Initialize prime-specific embeddings."""
        # 2-adic: polar (0) vs nonpolar (1)
        polar_aa = {"S", "T", "N", "Q", "K", "R", "H", "D", "E", "Y"}
        with torch.no_grad():
            for aa, idx in AA_TO_INDEX.items():
                if idx < 22:
                    base = torch.randn(self.encoder_2adic.embedding_dim)
                    if aa in polar_aa:
                        base = base * 0.5  # Polar cluster
                    else:
                        base = base * 0.5 + 1.0  # Nonpolar cluster
                    self.encoder_2adic.weight[idx] = base

    def forward(self, indices: Tensor) -> Tensor:
        """Encode with multi-prime fusion.

        Args:
            indices: Amino acid indices

        Returns:
            Fused embeddings
        """
        # Get embeddings from each prime
        embed_2 = self.encoder_2adic(indices)
        embed_3 = self.encoder_3adic(indices)
        embed_5 = self.encoder_5adic(indices)
        embed_7 = self.encoder_7adic(indices)

        if self.fusion is not None:
            # Stack for attention
            stacked = torch.stack([embed_2, embed_3, embed_5, embed_7], dim=-2)

            # Self-attention fusion
            if stacked.dim() == 2:
                stacked = stacked.unsqueeze(0)
                fused, _ = self.fusion(stacked, stacked, stacked)
                fused = fused.squeeze(0).mean(dim=-2)
            else:
                batch_size = stacked.shape[0]
                if stacked.dim() == 3:
                    # (batch, 4, dim)
                    fused, _ = self.fusion(stacked, stacked, stacked)
                    fused = fused.mean(dim=-2)
                else:
                    # (batch, seq, 4, dim) -> (batch*seq, 4, dim)
                    seq_len = stacked.shape[1]
                    stacked_flat = stacked.view(-1, 4, stacked.shape[-1])
                    fused, _ = self.fusion(stacked_flat, stacked_flat, stacked_flat)
                    fused = fused.mean(dim=-2).view(batch_size, seq_len, -1)

            output = self.fusion_proj(fused)
        else:
            # Simple concatenation
            combined = torch.cat([embed_2, embed_3, embed_5, embed_7], dim=-1)
            output = self.fusion_proj(combined)

        return self.norm(output)


# =============================================================================
# Mutation-Type Embedding
# =============================================================================


class MutationType(IntEnum):
    """Types of nucleotide/amino acid mutations."""

    SYNONYMOUS = 0  # Same amino acid
    CONSERVATIVE = 1  # Similar amino acid (same group)
    MODERATE = 2  # Different group, similar properties
    RADICAL = 3  # Very different amino acid
    NONSENSE = 4  # Stop codon introduced
    FRAMESHIFT = 5  # Reading frame change


class MutationTypeEmbedding(nn.Module):
    """Mutation Type Embedding.

    Encodes the type and impact of mutations for resistance prediction.

    Features encoded:
        - Mutation type (synonymous, conservative, radical, etc.)
        - Transition vs transversion (for nucleotide)
        - BLOSUM62 score
        - Position in protein (N-term, C-term, core)
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        include_position: bool = True,
        include_blosum: bool = True,
    ):
        """Initialize mutation type embedding.

        Args:
            embedding_dim: Output dimension
            include_position: Include positional features
            include_blosum: Include BLOSUM62 similarity
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.include_position = include_position
        self.include_blosum = include_blosum

        # Mutation type embedding
        self.type_embedding = nn.Embedding(6, embedding_dim // 2)

        # Continuous features
        n_features = 1  # Base: transition/transversion
        if include_blosum:
            n_features += 1
        if include_position:
            n_features += 3  # Relative position, N-term distance, C-term distance

        self.feature_encoder = nn.Sequential(
            nn.Linear(n_features, embedding_dim // 2),
            nn.GELU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2),
        )

        self.fusion = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

        # Register BLOSUM62 matrix (simplified)
        self._register_blosum()

    def _register_blosum(self) -> None:
        """Register BLOSUM62 substitution matrix."""
        # Simplified BLOSUM62 scores (normalized to -1 to 1)
        # Only store diagonal and common pairs
        blosum = torch.zeros(22, 22)

        # Diagonal (same AA)
        for i in range(20):
            blosum[i, i] = 1.0

        # Similar amino acids (high BLOSUM scores)
        similar_pairs = [
            (AA_TO_INDEX["L"], AA_TO_INDEX["I"], 0.6),
            (AA_TO_INDEX["L"], AA_TO_INDEX["V"], 0.4),
            (AA_TO_INDEX["I"], AA_TO_INDEX["V"], 0.6),
            (AA_TO_INDEX["F"], AA_TO_INDEX["Y"], 0.6),
            (AA_TO_INDEX["K"], AA_TO_INDEX["R"], 0.5),
            (AA_TO_INDEX["D"], AA_TO_INDEX["E"], 0.5),
            (AA_TO_INDEX["N"], AA_TO_INDEX["D"], 0.4),
            (AA_TO_INDEX["Q"], AA_TO_INDEX["E"], 0.4),
            (AA_TO_INDEX["S"], AA_TO_INDEX["T"], 0.4),
        ]

        for i, j, score in similar_pairs:
            blosum[i, j] = score
            blosum[j, i] = score

        self.register_buffer("blosum", blosum)

    def classify_mutation(
        self,
        from_aa: str,
        to_aa: str,
    ) -> MutationType:
        """Classify mutation type.

        Args:
            from_aa: Original amino acid
            to_aa: Mutated amino acid

        Returns:
            MutationType
        """
        if from_aa == to_aa:
            return MutationType.SYNONYMOUS

        if to_aa == "*":
            return MutationType.NONSENSE

        from_group = AA_TO_GROUP.get(from_aa, AminoAcidGroup.SPECIAL)
        to_group = AA_TO_GROUP.get(to_aa, AminoAcidGroup.SPECIAL)

        if from_group == to_group:
            return MutationType.CONSERVATIVE

        # Check BLOSUM score
        from_idx = AA_TO_INDEX.get(from_aa, 21)
        to_idx = AA_TO_INDEX.get(to_aa, 21)
        blosum_score = self.blosum[from_idx, to_idx].item()

        if blosum_score >= 0.2:
            return MutationType.MODERATE
        else:
            return MutationType.RADICAL

    def forward(
        self,
        mutation_types: Tensor,
        features: Tensor | None = None,
    ) -> Tensor:
        """Encode mutations.

        Args:
            mutation_types: Mutation type indices (N,) or (N, M)
            features: Additional continuous features (N, n_features)

        Returns:
            Embeddings
        """
        type_embed = self.type_embedding(mutation_types)

        if features is not None:
            feat_embed = self.feature_encoder(features)
        else:
            # Default features
            feat_embed = torch.zeros(
                *mutation_types.shape,
                self.embedding_dim // 2,
                device=mutation_types.device,
            )

        combined = torch.cat([type_embed, feat_embed], dim=-1)
        output = self.fusion(combined)
        output = self.norm(output)

        return output

    def encode_mutation(
        self,
        from_aa: str,
        to_aa: str,
        position: int | None = None,
        seq_length: int | None = None,
    ) -> Tensor:
        """Encode a single mutation.

        Args:
            from_aa: Original amino acid
            to_aa: Mutated amino acid
            position: Position in sequence (optional)
            seq_length: Total sequence length (optional)

        Returns:
            Embedding tensor
        """
        mut_type = self.classify_mutation(from_aa, to_aa)
        mut_type_tensor = torch.tensor([mut_type], device=self.type_embedding.weight.device)

        # Build features
        features = []

        # Transition/transversion (0 for AA, use as placeholder)
        features.append(0.0)

        if self.include_blosum:
            from_idx = AA_TO_INDEX.get(from_aa, 21)
            to_idx = AA_TO_INDEX.get(to_aa, 21)
            blosum_score = self.blosum[from_idx, to_idx].item()
            features.append(blosum_score)

        if self.include_position and position is not None and seq_length is not None:
            rel_pos = position / seq_length
            n_term_dist = position / seq_length
            c_term_dist = (seq_length - position) / seq_length
            features.extend([rel_pos, n_term_dist, c_term_dist])

        feature_tensor = torch.tensor([features], device=self.type_embedding.weight.device).float()

        return self.forward(mut_type_tensor, feature_tensor)
