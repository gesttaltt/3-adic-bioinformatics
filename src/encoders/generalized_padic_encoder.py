#!/usr/bin/env python3
"""Generalized P-adic Encoder with Segment-Based Support.

This module implements a generalized p-adic encoder that:
1. Supports any prime p (2, 3, 5, 7, ...) - not just 3-adic
2. Uses segment-based encoding for long sequences
3. Aggregates segment embeddings hierarchically

Architecture:
    Input: Sequence of elements (codons, amino acids, nucleotides)
    → Segment Splitter: Split into overlapping windows
    → P-adic Encoder: Encode each segment to hyperbolic space
    → Hierarchical Aggregator: Combine segments with attention
    → Output: Single embedding preserving p-adic structure

Key Insight from Regime Analysis:
    - Long sequences (>25 elements) fail with current approaches
    - Segment-based encoding allows local pattern capture
    - Hierarchical aggregation preserves global structure

Usage:
    from src.encoders.generalized_padic_encoder import GeneralizedPadicEncoder

    encoder = GeneralizedPadicEncoder(
        prime=5,  # 5-adic for amino acids
        segment_size=10,
        overlap=5,
        latent_dim=16,
    )
    z_hyp = encoder(sequence_indices)  # (batch, latent_dim)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.geometry import (
    exp_map_zero,
    poincare_distance,
    project_to_poincare,
)

# =============================================================================
# P-adic Mathematics (Generalized)
# =============================================================================


def compute_padic_valuation(n: int, p: int) -> int:
    """Compute p-adic valuation v_p(n).

    The p-adic valuation is the largest power of p dividing n.
    v_p(0) = infinity (we return p for practical purposes).

    Args:
        n: Integer to compute valuation for
        p: Prime number

    Returns:
        p-adic valuation v_p(n)
    """
    if n == 0:
        return p  # Convention: v_p(0) = infinity, use p as proxy
    if n < 0:
        n = abs(n)

    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v


def compute_padic_distance(a: int, b: int, p: int) -> float:
    """Compute p-adic distance d_p(a, b) = p^(-v_p(a-b)).

    Args:
        a, b: Integers
        p: Prime number

    Returns:
        p-adic distance in [0, 1]
    """
    if a == b:
        return 0.0
    v = compute_padic_valuation(a - b, p)
    return float(p ** (-v))


def to_padic_representation(n: int, p: int, n_digits: int) -> list[int]:
    """Convert integer to p-adic digit representation.

    Args:
        n: Integer in [0, p^n_digits - 1]
        p: Prime base
        n_digits: Number of digits

    Returns:
        List of digits in {0, 1, ..., p-1}
    """
    digits = []
    for _ in range(n_digits):
        digits.append(n % p)
        n //= p
    return digits


def from_padic_representation(digits: list[int], p: int) -> int:
    """Convert p-adic digits back to integer.

    Args:
        digits: List of digits in {0, 1, ..., p-1}
        p: Prime base

    Returns:
        Integer value
    """
    result = 0
    for i, d in enumerate(digits):
        result += d * (p**i)
    return result


# =============================================================================
# Segment-Based Input Processing
# =============================================================================


class SegmentSplitter(nn.Module):
    """Split sequences into overlapping segments.

    For a sequence of length L:
    - Creates segments of size `segment_size`
    - Overlaps by `overlap` elements
    - Pads if necessary
    """

    def __init__(
        self,
        segment_size: int = 10,
        overlap: int = 5,
        pad_value: int = 0,
    ):
        """Initialize segment splitter.

        Args:
            segment_size: Size of each segment
            overlap: Number of overlapping elements between segments
            pad_value: Value to use for padding
        """
        super().__init__()
        self.segment_size = segment_size
        self.overlap = overlap
        self.stride = segment_size - overlap
        self.pad_value = pad_value

        assert self.stride > 0, "Overlap must be less than segment size"

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Split sequence into segments.

        Args:
            x: Input tensor of shape (batch, seq_len) or (batch, seq_len, dim)
            lengths: Actual sequence lengths (batch,)

        Returns:
            segments: (batch, n_segments, segment_size) or (..., dim)
            segment_mask: (batch, n_segments) - True for valid segments
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        has_features = len(x.shape) > 2

        # Calculate number of segments
        n_segments = max(1, (seq_len - self.overlap) // self.stride)
        if (seq_len - self.overlap) % self.stride > 0:
            n_segments += 1

        # Pad sequence if needed
        required_len = (n_segments - 1) * self.stride + self.segment_size
        if seq_len < required_len:
            pad_size = required_len - seq_len
            if has_features:
                x = F.pad(x, (0, 0, 0, pad_size), value=self.pad_value)
            else:
                x = F.pad(x, (0, pad_size), value=self.pad_value)

        # Extract segments
        segments = []
        for i in range(n_segments):
            start = i * self.stride
            end = start + self.segment_size
            segments.append(x[:, start:end])

        if has_features:
            segments = torch.stack(segments, dim=1)  # (batch, n_seg, seg_size, dim)
        else:
            segments = torch.stack(segments, dim=1)  # (batch, n_seg, seg_size)

        # Create segment mask
        if lengths is not None:
            segment_mask = torch.zeros(batch_size, n_segments, dtype=torch.bool, device=x.device)
            for b in range(batch_size):
                # A segment is valid if its start is within the original sequence
                for i in range(n_segments):
                    start = i * self.stride
                    segment_mask[b, i] = start < lengths[b]
        else:
            segment_mask = torch.ones(batch_size, n_segments, dtype=torch.bool, device=x.device)

        return segments, segment_mask


# =============================================================================
# P-adic Segment Encoder
# =============================================================================


class PadicSegmentEncoder(nn.Module):
    """Encode a segment using p-adic structure.

    For a segment of `segment_size` elements:
    - Embed each element
    - Apply p-adic positional encoding based on valuation
    - Aggregate with attention
    """

    def __init__(
        self,
        prime: int = 5,
        vocab_size: int = 22,  # 20 AA + stop + pad
        segment_size: int = 10,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize segment encoder.

        Args:
            prime: Prime for p-adic structure
            vocab_size: Number of possible input tokens
            segment_size: Expected segment size
            embed_dim: Token embedding dimension
            hidden_dim: Transformer hidden dimension
            latent_dim: Output latent dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.prime = prime
        self.segment_size = segment_size
        self.latent_dim = latent_dim

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=vocab_size - 1)

        # P-adic positional encoding
        # Each position gets an embedding based on its p-adic valuation
        max_valuation = int(math.log(segment_size, prime)) + 1 if segment_size > 0 else 1
        self.valuation_embedding = nn.Embedding(max_valuation + 1, embed_dim)

        # Pre-compute valuations for positions
        valuations = [compute_padic_valuation(i, prime) for i in range(segment_size)]
        valuations = [min(v, max_valuation) for v in valuations]
        self.register_buffer("position_valuations", torch.tensor(valuations, dtype=torch.long))

        # Transformer layer for segment encoding
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Attention pooling
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.pool_attention = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)

        # Project to latent space
        self.output_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, segment: Tensor, mask: Tensor | None = None) -> Tensor:
        """Encode a segment.

        Args:
            segment: Token indices (batch, segment_size)
            mask: Valid positions mask (batch, segment_size)

        Returns:
            Segment embedding (batch, latent_dim)
        """
        batch_size = segment.shape[0]
        device = segment.device

        # Token embedding
        x = self.embedding(segment)  # (batch, seg_size, embed_dim)

        # Add p-adic positional encoding
        val_indices = self.position_valuations.to(device)
        val_emb = self.valuation_embedding(val_indices)  # (seg_size, embed_dim)
        x = x + val_emb.unsqueeze(0)

        # Project to hidden dim
        x = self.input_proj(x)

        # Transformer encoding
        src_key_padding_mask = ~mask if mask is not None else None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Attention pooling
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attention(query, x, x, key_padding_mask=src_key_padding_mask)
        pooled = pooled.squeeze(1)

        # Project to latent space
        z = self.output_proj(pooled)

        return z


# =============================================================================
# Hierarchical Aggregator
# =============================================================================


class HierarchicalAggregator(nn.Module):
    """Aggregate segment embeddings hierarchically.

    Uses multi-level attention to combine segment embeddings:
    Level 1: Adjacent segment pairs
    Level 2: Pairs of pairs
    ...
    Until single embedding
    """

    def __init__(
        self,
        latent_dim: int = 16,
        n_heads: int = 4,
        max_segments: int = 32,
        dropout: float = 0.1,
        curvature: float = 1.0,
    ):
        """Initialize hierarchical aggregator.

        Args:
            latent_dim: Segment embedding dimension
            n_heads: Attention heads
            max_segments: Maximum number of segments
            dropout: Dropout rate
            curvature: Poincare ball curvature
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.curvature = curvature
        self.max_segments = max_segments

        # Segment-level positional encoding
        self.segment_pos = nn.Embedding(max_segments, latent_dim)

        # Cross-segment attention
        self.cross_attention = nn.MultiheadAttention(latent_dim, n_heads, dropout=dropout, batch_first=True)

        # Aggregation MLP
        self.aggregator = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # Final projection to Poincare ball
        self.final_proj = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        segment_embeddings: Tensor,
        segment_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Aggregate segment embeddings.

        Args:
            segment_embeddings: (batch, n_segments, latent_dim) in tangent space
            segment_mask: (batch, n_segments) valid segment mask

        Returns:
            z_hyp: Final embedding on Poincare ball (batch, latent_dim)
            radius: Hyperbolic radius (batch,)
        """
        batch_size, n_segments, _ = segment_embeddings.shape
        device = segment_embeddings.device

        # Add segment position encoding
        pos_indices = torch.arange(n_segments, device=device)
        pos_emb = self.segment_pos(pos_indices)
        x = segment_embeddings + pos_emb.unsqueeze(0)

        # Self-attention across segments
        key_padding_mask = ~segment_mask if segment_mask is not None else None

        x, _ = self.cross_attention(x, x, x, key_padding_mask=key_padding_mask)

        # Masked mean pooling
        if segment_mask is not None:
            mask_expanded = segment_mask.unsqueeze(-1).float()
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        # Aggregation MLP
        z_tangent = self.aggregator(pooled)
        z_tangent = self.final_proj(z_tangent)

        # Map to Poincare ball
        z_hyp = exp_map_zero(z_tangent, c=self.curvature)
        z_hyp = project_to_poincare(z_hyp, max_norm=0.95, c=self.curvature)

        # Compute radius
        origin = torch.zeros(batch_size, self.latent_dim, device=device)
        radius = poincare_distance(z_hyp, origin, c=self.curvature)

        return z_hyp, radius


# =============================================================================
# Generalized P-adic Encoder
# =============================================================================


class GeneralizedPadicEncoder(nn.Module):
    """Full generalized p-adic encoder with segment-based support.

    Combines:
    1. Segment splitting for long sequences
    2. P-adic segment encoding
    3. Hierarchical aggregation
    4. Hyperbolic output
    """

    def __init__(
        self,
        prime: int = 5,
        vocab_size: int = 22,
        segment_size: int = 10,
        overlap: int = 5,
        embed_dim: int = 32,
        hidden_dim: int = 64,
        latent_dim: int = 16,
        n_heads: int = 4,
        dropout: float = 0.1,
        curvature: float = 1.0,
        max_segments: int = 32,
    ):
        """Initialize generalized p-adic encoder.

        Args:
            prime: Prime for p-adic structure (2, 3, 5, 7, ...)
            vocab_size: Number of input tokens
            segment_size: Size of each segment
            overlap: Overlap between segments
            embed_dim: Token embedding dimension
            hidden_dim: Transformer hidden dimension
            latent_dim: Output latent dimension
            n_heads: Attention heads
            dropout: Dropout rate
            curvature: Poincare ball curvature
            max_segments: Maximum segments to handle
        """
        super().__init__()

        self.prime = prime
        self.vocab_size = vocab_size
        self.segment_size = segment_size
        self.latent_dim = latent_dim
        self.curvature = curvature

        # Segment splitter
        self.splitter = SegmentSplitter(segment_size, overlap)

        # Segment encoder
        self.segment_encoder = PadicSegmentEncoder(
            prime=prime,
            vocab_size=vocab_size,
            segment_size=segment_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Hierarchical aggregator
        self.aggregator = HierarchicalAggregator(
            latent_dim=latent_dim,
            n_heads=n_heads,
            max_segments=max_segments,
            dropout=dropout,
            curvature=curvature,
        )

        # Short sequence bypass (for sequences <= segment_size)
        self.short_encoder = PadicSegmentEncoder(
            prime=prime,
            vocab_size=vocab_size,
            segment_size=segment_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_heads=n_heads,
            dropout=dropout,
        )

    def forward(
        self,
        x: Tensor,
        lengths: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Token indices (batch, seq_len)
            lengths: Actual sequence lengths (batch,)

        Returns:
            Dictionary with:
                z_hyp: Hyperbolic embedding (batch, latent_dim)
                radius: Hyperbolic radius (batch,)
                n_segments: Number of segments per sequence (batch,)
        """
        batch_size, seq_len = x.shape
        device = x.device

        # For short sequences, use direct encoding
        if seq_len <= self.segment_size:
            # Pad to segment size if needed
            if seq_len < self.segment_size:
                x_padded = F.pad(x, (0, self.segment_size - seq_len), value=self.vocab_size - 1)
            else:
                x_padded = x

            z_tangent = self.short_encoder(x_padded)
            z_hyp = exp_map_zero(z_tangent, c=self.curvature)
            z_hyp = project_to_poincare(z_hyp, max_norm=0.95, c=self.curvature)

            origin = torch.zeros(batch_size, self.latent_dim, device=device)
            radius = poincare_distance(z_hyp, origin, c=self.curvature)

            return {
                "z_hyp": z_hyp,
                "radius": radius,
                "n_segments": torch.ones(batch_size, device=device),
            }

        # Split into segments
        segments, segment_mask = self.splitter(x, lengths)
        # segments: (batch, n_segments, segment_size)
        # segment_mask: (batch, n_segments)

        n_segments = segments.shape[1]

        # Encode each segment
        # Reshape for batch processing
        segments_flat = segments.view(batch_size * n_segments, self.segment_size)
        segment_embeddings_flat = self.segment_encoder(segments_flat)
        segment_embeddings = segment_embeddings_flat.view(batch_size, n_segments, self.latent_dim)

        # Hierarchical aggregation
        z_hyp, radius = self.aggregator(segment_embeddings, segment_mask)

        return {
            "z_hyp": z_hyp,
            "radius": radius,
            "n_segments": torch.full((batch_size,), n_segments, device=device),
        }

    def encode(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Convenience method to get just the embedding."""
        return self.forward(x, lengths)["z_hyp"]


# =============================================================================
# Factory Functions
# =============================================================================


def create_codon_encoder(
    prime: int = 3,
    segment_size: int = 10,
    overlap: int = 5,
    latent_dim: int = 16,
    **kwargs,
) -> GeneralizedPadicEncoder:
    """Create encoder for codon sequences (64 codons + stop + pad)."""
    return GeneralizedPadicEncoder(
        prime=prime,
        vocab_size=66,  # 64 codons + stop + pad
        segment_size=segment_size,
        overlap=overlap,
        latent_dim=latent_dim,
        **kwargs,
    )


def create_aminoacid_encoder(
    prime: int = 5,
    segment_size: int = 10,
    overlap: int = 5,
    latent_dim: int = 16,
    **kwargs,
) -> GeneralizedPadicEncoder:
    """Create encoder for amino acid sequences (20 AA + stop + pad)."""
    return GeneralizedPadicEncoder(
        prime=prime,
        vocab_size=22,  # 20 AA + stop + pad
        segment_size=segment_size,
        overlap=overlap,
        latent_dim=latent_dim,
        **kwargs,
    )


def create_nucleotide_encoder(
    prime: int = 2,
    segment_size: int = 20,
    overlap: int = 10,
    latent_dim: int = 16,
    **kwargs,
) -> GeneralizedPadicEncoder:
    """Create encoder for nucleotide sequences (4 bases + N + pad)."""
    return GeneralizedPadicEncoder(
        prime=prime,
        vocab_size=6,  # A, C, G, T/U, N, pad
        segment_size=segment_size,
        overlap=overlap,
        latent_dim=latent_dim,
        **kwargs,
    )


__all__ = [
    "GeneralizedPadicEncoder",
    "SegmentSplitter",
    "PadicSegmentEncoder",
    "HierarchicalAggregator",
    "create_codon_encoder",
    "create_aminoacid_encoder",
    "create_nucleotide_encoder",
    "compute_padic_valuation",
    "compute_padic_distance",
]
