# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Codon symmetry layers for biological sequence processing.

This module provides neural network layers that respect biological symmetries
in the genetic code, including:

1. Synonymous codon groups: Multiple codons encoding the same amino acid
2. Wobble position flexibility: Third position mutations often preserve function
3. Amino acid physicochemical similarities

These symmetries are important for:
- Codon optimization
- Synonymous mutation analysis
- Evolutionary sequence modeling

References:
    - Crick, "Codon-Anticodon Pairing: The Wobble Hypothesis" (1966)
    - Sharp et al., "Codon Usage Bias" (2010)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Standard genetic code: codon -> amino acid
GENETIC_CODE = {
    "TTT": "F",
    "TTC": "F",  # Phenylalanine
    "TTA": "L",
    "TTG": "L",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",  # Leucine
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",  # Isoleucine
    "ATG": "M",  # Methionine (start)
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",  # Valine
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "AGT": "S",
    "AGC": "S",  # Serine
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",  # Proline
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",  # Threonine
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",  # Alanine
    "TAT": "Y",
    "TAC": "Y",  # Tyrosine
    "TAA": "*",
    "TAG": "*",
    "TGA": "*",  # Stop codons
    "CAT": "H",
    "CAC": "H",  # Histidine
    "CAA": "Q",
    "CAG": "Q",  # Glutamine
    "AAT": "N",
    "AAC": "N",  # Asparagine
    "AAA": "K",
    "AAG": "K",  # Lysine
    "GAT": "D",
    "GAC": "D",  # Aspartic acid
    "GAA": "E",
    "GAG": "E",  # Glutamic acid
    "TGT": "C",
    "TGC": "C",  # Cysteine
    "TGG": "W",  # Tryptophan
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGA": "R",
    "AGG": "R",  # Arginine
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",  # Glycine
}

# Nucleotide to index mapping
NUC_TO_IDX = {"T": 0, "C": 1, "A": 2, "G": 3}
IDX_TO_NUC = {v: k for k, v in NUC_TO_IDX.items()}

# Amino acid to index mapping
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY*"
AA_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def codon_to_index(codon: str) -> int:
    """Convert codon string to index (0-63)."""
    n1, n2, n3 = codon
    return NUC_TO_IDX[n1] * 16 + NUC_TO_IDX[n2] * 4 + NUC_TO_IDX[n3]


def index_to_codon(idx: int) -> str:
    """Convert index (0-63) to codon string."""
    n1 = idx // 16
    n2 = (idx % 16) // 4
    n3 = idx % 4
    return IDX_TO_NUC[n1] + IDX_TO_NUC[n2] + IDX_TO_NUC[n3]


def get_synonymous_groups() -> dict[str, list[int]]:
    """Get synonymous codon groups for each amino acid.

    Returns:
        Dictionary mapping amino acid to list of codon indices
    """
    groups: dict[str, list[int]] = {}
    for codon, aa in GENETIC_CODE.items():
        if aa not in groups:
            groups[aa] = []
        groups[aa].append(codon_to_index(codon))
    return groups


def get_wobble_equivalences() -> list[tuple[int, int]]:
    """Get pairs of codons that differ only at wobble position.

    The wobble position (3rd nucleotide) often allows flexibility
    in codon-anticodon pairing.

    Returns:
        List of (codon1_idx, codon2_idx) pairs
    """
    equivalences = []
    for i in range(64):
        codon1 = index_to_codon(i)
        # Get amino acid for this codon
        aa1 = GENETIC_CODE.get(codon1)
        if aa1 is None:
            continue

        # Check codons that differ only at position 3
        for j in range(i + 1, 64):
            codon2 = index_to_codon(j)
            aa2 = GENETIC_CODE.get(codon2)

            # Same amino acid and differ only at position 3
            if aa1 == aa2 and codon1[:2] == codon2[:2]:
                equivalences.append((i, j))

    return equivalences


class CodonEmbedding(nn.Module):
    """Embedding layer for codons with symmetry awareness.

    Creates embeddings that can respect synonymous codon relationships.

    Args:
        embedding_dim: Dimension of embeddings
        share_synonymous: Whether to share base embeddings for synonymous codons
        learn_deviation: Whether to learn deviations from shared embeddings
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        share_synonymous: bool = True,
        learn_deviation: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.share_synonymous = share_synonymous
        self.learn_deviation = learn_deviation

        if share_synonymous:
            # Shared embeddings per amino acid (21 including stop)
            self.aa_embedding = nn.Embedding(21, embedding_dim)

            # Codon-specific deviations
            if learn_deviation:
                self.codon_deviation = nn.Embedding(64, embedding_dim)
                # Initialize deviations to small values
                nn.init.normal_(self.codon_deviation.weight, std=0.1)

            # Build codon to amino acid mapping
            codon_to_aa = torch.zeros(64, dtype=torch.long)
            for codon, aa in GENETIC_CODE.items():
                idx = codon_to_index(codon)
                codon_to_aa[idx] = AA_TO_IDX[aa]
            self.register_buffer("codon_to_aa", codon_to_aa)
        else:
            # Direct codon embeddings
            self.embedding = nn.Embedding(64, embedding_dim)

    def forward(self, codons: Tensor) -> Tensor:
        """Embed codon indices.

        Args:
            codons: Codon indices of shape (...), values in [0, 63]

        Returns:
            Embeddings of shape (..., embedding_dim)
        """
        if self.share_synonymous:
            # Get amino acid for each codon
            aa_indices = self.codon_to_aa[codons]
            emb = self.aa_embedding(aa_indices)

            if self.learn_deviation:
                emb = emb + self.codon_deviation(codons)

            return emb
        else:
            return self.embedding(codons)


class SynonymousPooling(nn.Module):
    """Pooling layer that aggregates over synonymous codons.

    Projects 64-dimensional codon space to 21-dimensional amino acid space.

    Args:
        pool_type: Pooling method ("mean", "max", "attention")
        hidden_dim: Hidden dimension for attention pooling
    """

    def __init__(self, pool_type: str = "mean", hidden_dim: int = 64):
        super().__init__()
        self.pool_type = pool_type

        # Build synonymous groups
        syn_groups = get_synonymous_groups()

        # Create mapping matrix
        mapping = torch.zeros(64, 21)
        for aa, codons in syn_groups.items():
            aa_idx = AA_TO_IDX[aa]
            for codon_idx in codons:
                mapping[codon_idx, aa_idx] = 1.0

        # Normalize for mean pooling
        if pool_type == "mean":
            mapping = mapping / mapping.sum(dim=0, keepdim=True).clamp(min=1)

        self.register_buffer("mapping", mapping)

        # Attention for attention pooling
        if pool_type == "attention":
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, codon_features: Tensor) -> Tensor:
        """Pool codon features to amino acid features.

        Args:
            codon_features: Features of shape (..., 64, feature_dim)

        Returns:
            Amino acid features of shape (..., 21, feature_dim)
        """
        if self.pool_type in ["mean", "max"]:
            # Use mapping matrix
            # (..., 64, feat) @ (64, 21) -> (..., 21, feat)
            return torch.einsum("...cf,ca->...af", codon_features, self.mapping)
        else:
            # Attention pooling
            # Compute attention scores per codon
            scores = self.attention(codon_features).squeeze(-1)  # (..., 64)

            # Mask by synonymous groups and softmax
            # This is more complex - simplified version
            return torch.einsum("...cf,ca->...af", codon_features * scores.unsqueeze(-1), self.mapping)


class WobbleAwareConv(nn.Module):
    """Convolution layer aware of wobble position flexibility.

    Applies different processing to stable (positions 1-2) vs
    flexible (position 3) nucleotides.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        wobble_discount: Weight discount for wobble position
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        wobble_discount: float = 0.5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wobble_discount = wobble_discount

        # Separate convolutions for stable and wobble positions
        self.stable_conv = nn.Linear(2 * in_channels, out_channels)
        self.wobble_conv = nn.Linear(in_channels, out_channels)

        # Combine
        self.combine = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply wobble-aware convolution.

        Args:
            x: Nucleotide features of shape (batch, seq_len, 3, in_channels)
               where dimension 2 is the 3 codon positions

        Returns:
            Codon features of shape (batch, seq_len, out_channels)
        """
        # Split by position
        stable = x[..., :2, :].flatten(-2, -1)  # (batch, seq, 2*in)
        wobble = x[..., 2, :]  # (batch, seq, in)

        # Process separately
        stable_out = self.stable_conv(stable)  # (batch, seq, out)
        wobble_out = self.wobble_conv(wobble) * self.wobble_discount

        # Combine
        combined = torch.cat([stable_out, wobble_out], dim=-1)
        return self.combine(combined)


class CodonSymmetryLayer(nn.Module):
    """Neural network layer respecting codon symmetries.

    Encodes biological symmetries:
    - 64 codons -> 21 amino acids (many-to-one)
    - Wobble position (3rd position flexibility)
    - Synonymous codon groups

    Args:
        hidden_dim: Hidden dimension
        respect_wobble: Whether to apply wobble position weighting
        respect_synonymy: Whether to tie synonymous codon processing
        use_position_encoding: Whether to add position-specific encoding
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        respect_wobble: bool = True,
        respect_synonymy: bool = True,
        use_position_encoding: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.respect_wobble = respect_wobble
        self.respect_synonymy = respect_synonymy

        # Codon embedding with symmetry awareness
        self.embedding = CodonEmbedding(
            embedding_dim=hidden_dim,
            share_synonymous=respect_synonymy,
            learn_deviation=True,
        )

        # Position encoding for codon positions (1, 2, 3)
        if use_position_encoding:
            self.position_encoding = nn.Embedding(3, hidden_dim)
        else:
            self.position_encoding = None

        # Wobble-aware processing
        if respect_wobble:
            self.wobble_weights = nn.Parameter(torch.tensor([1.0, 1.0, 0.5]))
        else:
            self.register_buffer("wobble_weights", torch.ones(3))

        # Feature transformation
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Synonymous pooling for amino acid-level features
        if respect_synonymy:
            self.syn_pool = SynonymousPooling(pool_type="mean", hidden_dim=hidden_dim)

        # Build symmetry masks
        self._build_symmetry_structures()

    def _build_symmetry_structures(self):
        """Build internal structures for symmetry operations."""
        # Synonymous groups
        syn_groups = get_synonymous_groups()

        # Codon to synonymous group mapping
        codon_to_group = torch.zeros(64, dtype=torch.long)
        group_masks = []

        for group_idx, (_aa, codons) in enumerate(syn_groups.items()):
            mask = torch.zeros(64)
            for codon_idx in codons:
                codon_to_group[codon_idx] = group_idx
                mask[codon_idx] = 1.0
            group_masks.append(mask)

        self.register_buffer("codon_to_group", codon_to_group)

        # Stack group masks
        group_masks = torch.stack(group_masks)  # (21, 64)
        self.register_buffer("group_masks", group_masks)

        # Wobble equivalences
        wobble_pairs = get_wobble_equivalences()
        if wobble_pairs:
            wobble_tensor = torch.tensor(wobble_pairs)  # (n_pairs, 2)
            self.register_buffer("wobble_pairs", wobble_tensor)
        else:
            self.register_buffer("wobble_pairs", torch.zeros(0, 2, dtype=torch.long))

    def forward(self, codons: Tensor) -> Tensor:
        """Apply symmetry-aware transformation.

        Args:
            codons: Codon indices of shape (batch, seq_len), values in [0, 63]

        Returns:
            Features of shape (batch, seq_len, hidden_dim)
        """
        # Embed codons with synonymy awareness
        x = self.embedding(codons)  # (batch, seq, hidden)

        # Apply transformation
        x = self.transform(x)

        return x

    def get_amino_acid_features(self, codon_features: Tensor) -> Tensor:
        """Pool codon features to amino acid level.

        Args:
            codon_features: Features for each codon of shape (batch, 64, hidden)

        Returns:
            Amino acid features of shape (batch, 21, hidden)
        """
        if self.respect_synonymy:
            return self.syn_pool(codon_features)
        else:
            # Simple mean pooling using group masks
            # (batch, 64, hidden) -> (batch, 21, hidden)
            weights = self.group_masks / self.group_masks.sum(dim=1, keepdim=True).clamp(min=1)
            return torch.einsum("ac,bch->bah", weights, codon_features)

    def wobble_similarity(self, codon_features: Tensor) -> Tensor:
        """Compute similarity encouraging wobble equivalence.

        Args:
            codon_features: Features of shape (batch, 64, hidden)

        Returns:
            Similarity loss encouraging wobble-equivalent codons to be similar
        """
        if self.wobble_pairs.shape[0] == 0:
            return torch.tensor(0.0, device=codon_features.device)

        # Get features for wobble pairs
        idx1, idx2 = self.wobble_pairs[:, 0], self.wobble_pairs[:, 1]
        feat1 = codon_features[:, idx1]  # (batch, n_pairs, hidden)
        feat2 = codon_features[:, idx2]

        # Cosine similarity
        feat1_norm = F.normalize(feat1, dim=-1)
        feat2_norm = F.normalize(feat2, dim=-1)
        similarity = (feat1_norm * feat2_norm).sum(dim=-1).mean()

        return 1.0 - similarity  # Return as loss (minimize for high similarity)


class CodonAttention(nn.Module):
    """Attention mechanism aware of codon structure.

    Implements attention where synonymous codons attend to each other
    with higher weight.

    Args:
        hidden_dim: Hidden dimension
        n_heads: Number of attention heads
        synonymy_bias: Bias for synonymous codon attention
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_heads: int = 4,
        synonymy_bias: float = 1.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.synonymy_bias = synonymy_bias

        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Build synonymy bias matrix
        syn_groups = get_synonymous_groups()
        syn_bias = torch.zeros(64, 64)
        for _aa, codons in syn_groups.items():
            for i in codons:
                for j in codons:
                    syn_bias[i, j] = synonymy_bias
        self.register_buffer("syn_bias_matrix", syn_bias)

    def forward(
        self,
        x: Tensor,
        codons: Tensor | None = None,
    ) -> Tensor:
        """Apply codon-aware attention.

        Args:
            x: Features of shape (batch, seq_len, hidden_dim)
            codons: Optional codon indices for synonymy bias

        Returns:
            Attended features of shape (batch, seq_len, hidden_dim)
        """
        batch, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # Add synonymy bias if codons provided
        if codons is not None and self.synonymy_bias > 0:
            # Get synonymy bias for each pair
            syn_bias = self.syn_bias_matrix[codons[:, :, None], codons[:, None, :]]
            scores = scores + syn_bias.unsqueeze(1)

        # Softmax and apply to values
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        return self.out_proj(out)


class CodonTransformer(nn.Module):
    """Transformer for codon sequences with biological inductive biases.

    Args:
        vocab_size: Vocabulary size (64 for codons)
        hidden_dim: Hidden dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        respect_synonymy: Whether to use synonymy-aware attention
    """

    def __init__(
        self,
        vocab_size: int = 64,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        respect_synonymy: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        # Codon embedding with symmetry
        self.embedding = CodonEmbedding(
            embedding_dim=hidden_dim,
            share_synonymous=respect_synonymy,
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, hidden_dim) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                CodonTransformerLayer(
                    hidden_dim=hidden_dim,
                    n_heads=n_heads,
                    respect_synonymy=respect_synonymy,
                )
                for _ in range(n_layers)
            ]
        )

        # Output projection
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(
        self,
        codons: Tensor,
        return_hidden: bool = False,
    ) -> Tensor:
        """Forward pass.

        Args:
            codons: Codon indices of shape (batch, seq_len)
            return_hidden: Whether to return hidden states

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
            or hidden states if return_hidden=True
        """
        batch, seq_len = codons.shape

        # Embed
        x = self.embedding(codons)
        x = x + self.pos_encoding[:, :seq_len]

        # Apply layers
        for layer in self.layers:
            x = layer(x, codons)

        if return_hidden:
            return x

        return self.output(x)


class CodonTransformerLayer(nn.Module):
    """Single transformer layer with codon symmetry awareness."""

    def __init__(
        self,
        hidden_dim: int = 128,
        n_heads: int = 4,
        respect_synonymy: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Self-attention
        if respect_synonymy:
            self.attention = CodonAttention(hidden_dim, n_heads)
        else:
            self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.respect_synonymy = respect_synonymy

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, codons: Tensor | None = None) -> Tensor:
        """Forward pass."""
        # Self-attention with residual
        if self.respect_synonymy:
            x = x + self.attention(self.norm1(x), codons)
        else:
            attn_out, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
            x = x + attn_out

        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))

        return x
