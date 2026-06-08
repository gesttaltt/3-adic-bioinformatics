#!/usr/bin/env python3
"""Segment-Based Codon Encoder for Long Sequences.

This module implements a segment-based encoder that:
1. Uses proper codon-to-embedding mapping (12-dim one-hot → hyperbolic)
2. Splits long sequences into overlapping segments
3. Aggregates segment embeddings hierarchically

Key insight from analysis:
- P-adic structure on codon indices (0-63) works well (Spearman 0.81)
- P-adic structure on amino acid indices (0-19) doesn't work (Spearman 0.35)
- Reason: Codons have natural 4^3 structure reflecting genetic code

Architecture:
    Input: Codon sequence (variable length)
    → Segment Splitter: Overlapping windows
    → Codon Encoder: 12-dim one-hot → hyperbolic embedding
    → Hierarchical Aggregator: Attention-based segment fusion
    → Output: Single hyperbolic embedding

Usage:
    from src.encoders.segment_codon_encoder import SegmentCodonEncoder

    encoder = SegmentCodonEncoder(segment_size=10, overlap=5)
    z_hyp = encoder(codon_sequences, lengths)  # (batch, latent_dim)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.biology.codons import codon_index_to_triplet
from src.geometry import exp_map_zero, poincare_distance, project_to_poincare

# Base encoding: A=0, C=1, G=2, T/U=3
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def codon_to_onehot_12dim(codon_idx: int) -> torch.Tensor:
    """Convert codon index to 12-dim one-hot encoding."""
    triplet = codon_index_to_triplet(codon_idx)
    onehot = torch.zeros(12)
    for pos, base in enumerate(triplet):
        base_idx = BASE_TO_IDX[base]
        onehot[pos * 4 + base_idx] = 1.0
    return onehot


class CodonSegmentEncoder(nn.Module):
    """Encode a segment of codons to hyperbolic space."""

    def __init__(
        self,
        segment_size: int = 10,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        curvature: float = 1.0,
    ):
        super().__init__()

        self.segment_size = segment_size
        self.latent_dim = latent_dim
        self.curvature = curvature

        # Codon embedding: 12-dim one-hot → hidden_dim
        self.codon_proj = nn.Linear(12, hidden_dim)

        # Positional encoding for codons within segment
        self.pos_embedding = nn.Embedding(segment_size, hidden_dim)

        # Pre-compute one-hot encodings for all 64 codons + padding (65)
        codon_onehots = torch.stack([codon_to_onehot_12dim(i) for i in range(64)])
        pad_onehot = torch.zeros(12)  # All zeros for padding
        codon_onehots = torch.cat([codon_onehots, pad_onehot.unsqueeze(0)])  # (65, 12)
        self.register_buffer("codon_onehots", codon_onehots)

        # Transformer for segment encoding
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

    def forward(
        self,
        segment: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Encode a segment of codon indices.

        Args:
            segment: Codon indices (batch, segment_size) in range [0, 64]
                     Use 64 for padding
            mask: Valid positions (batch, segment_size), True = valid

        Returns:
            Segment embedding in tangent space (batch, latent_dim)
        """
        batch_size = segment.shape[0]
        device = segment.device

        # Clamp indices to valid range
        segment = segment.clamp(0, 64)

        # Get one-hot encodings
        # segment: (batch, seg_size) -> (batch, seg_size, 12)
        one_hot = self.codon_onehots[segment]

        # Project to hidden dim
        x = self.codon_proj(one_hot)  # (batch, seg_size, hidden)

        # Add positional encoding
        pos_indices = torch.arange(self.segment_size, device=device)
        pos_emb = self.pos_embedding(pos_indices)
        x = x + pos_emb.unsqueeze(0)

        # Transformer encoding
        src_key_padding_mask = ~mask if mask is not None else None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Attention pooling
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, _ = self.pool_attention(query, x, x, key_padding_mask=src_key_padding_mask)
        pooled = pooled.squeeze(1)

        # Project to latent
        z_tangent = self.output_proj(pooled)

        return z_tangent


class HierarchicalSegmentAggregator(nn.Module):
    """Aggregate segment embeddings with attention."""

    def __init__(
        self,
        latent_dim: int = 16,
        n_heads: int = 4,
        max_segments: int = 32,
        dropout: float = 0.1,
        curvature: float = 1.0,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.curvature = curvature

        # Segment position encoding
        self.segment_pos = nn.Embedding(max_segments, latent_dim)

        # Cross-segment attention
        self.cross_attention = nn.MultiheadAttention(latent_dim, n_heads, dropout=dropout, batch_first=True)

        # Final aggregation MLP
        self.aggregator = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(
        self,
        segment_embeddings: Tensor,
        segment_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Aggregate segment embeddings.

        Args:
            segment_embeddings: (batch, n_segments, latent_dim) in tangent space
            segment_mask: (batch, n_segments), True = valid segment

        Returns:
            z_hyp: Final embedding on Poincaré ball (batch, latent_dim)
            radius: Hyperbolic radius (batch,)
        """
        batch_size, n_segments, _ = segment_embeddings.shape
        device = segment_embeddings.device

        # Add segment position
        pos_indices = torch.arange(n_segments, device=device)
        pos_emb = self.segment_pos(pos_indices)
        x = segment_embeddings + pos_emb.unsqueeze(0)

        # Self-attention across segments
        key_padding_mask = ~segment_mask if segment_mask is not None else None
        x, _ = self.cross_attention(x, x, x, key_padding_mask=key_padding_mask)

        # Masked mean pooling
        if segment_mask is not None:
            mask_f = segment_mask.unsqueeze(-1).float()
            pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        else:
            pooled = x.mean(dim=1)

        # Final aggregation
        z_tangent = self.aggregator(pooled)

        # Map to Poincaré ball
        z_hyp = exp_map_zero(z_tangent, c=self.curvature)
        z_hyp = project_to_poincare(z_hyp, max_norm=0.95, c=self.curvature)

        # Compute radius
        origin = torch.zeros_like(z_hyp)
        radius = poincare_distance(z_hyp, origin, c=self.curvature)

        return z_hyp, radius


class SegmentCodonEncoder(nn.Module):
    """Full segment-based codon encoder for long sequences.

    For short sequences (<= segment_size): direct encoding
    For long sequences: segment splitting + hierarchical aggregation
    """

    def __init__(
        self,
        segment_size: int = 10,
        overlap: int = 5,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
        curvature: float = 1.0,
        max_segments: int = 32,
    ):
        """Initialize SegmentCodonEncoder.

        Args:
            segment_size: Codons per segment
            overlap: Overlap between segments
            latent_dim: Output dimension
            hidden_dim: Hidden dimension
            n_heads: Attention heads
            dropout: Dropout rate
            curvature: Poincaré ball curvature
            max_segments: Maximum segments to handle
        """
        super().__init__()

        self.segment_size = segment_size
        self.overlap = overlap
        self.stride = segment_size - overlap
        self.latent_dim = latent_dim
        self.curvature = curvature

        assert self.stride > 0, "Overlap must be < segment_size"

        # Segment encoder
        self.segment_encoder = CodonSegmentEncoder(
            segment_size=segment_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            dropout=dropout,
            curvature=curvature,
        )

        # Hierarchical aggregator
        self.aggregator = HierarchicalSegmentAggregator(
            latent_dim=latent_dim,
            n_heads=n_heads,
            max_segments=max_segments,
            dropout=dropout,
            curvature=curvature,
        )

    def _split_into_segments(
        self,
        x: Tensor,
        lengths: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Split sequence into overlapping segments.

        Args:
            x: Codon indices (batch, seq_len)
            lengths: Actual sequence lengths (batch,)

        Returns:
            segments: (batch, n_segments, segment_size)
            segment_mask: (batch, n_segments), True = valid
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Number of segments
        n_segments = max(1, (seq_len - self.overlap) // self.stride)
        if (seq_len - self.overlap) % self.stride > 0:
            n_segments += 1

        # Pad if needed
        required_len = (n_segments - 1) * self.stride + self.segment_size
        if seq_len < required_len:
            x = F.pad(x, (0, required_len - seq_len), value=64)  # 64 = padding

        # Extract segments
        segments = []
        for i in range(n_segments):
            start = i * self.stride
            end = start + self.segment_size
            segments.append(x[:, start:end])

        segments = torch.stack(segments, dim=1)  # (batch, n_seg, seg_size)

        # Create segment mask
        if lengths is not None:
            segment_mask = torch.zeros(batch_size, n_segments, dtype=torch.bool, device=device)
            for b in range(batch_size):
                for i in range(n_segments):
                    start = i * self.stride
                    # Valid if segment start is within sequence
                    segment_mask[b, i] = start < lengths[b]
        else:
            segment_mask = torch.ones(batch_size, n_segments, dtype=torch.bool, device=device)

        return segments, segment_mask

    def forward(
        self,
        x: Tensor,
        lengths: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Forward pass.

        Args:
            x: Codon indices (batch, seq_len) in range [0, 63]
            lengths: Actual sequence lengths (batch,)

        Returns:
            Dictionary with z_hyp, radius, n_segments
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Short sequence: direct encoding
        if seq_len <= self.segment_size:
            # Pad to segment size
            if seq_len < self.segment_size:
                x = F.pad(x, (0, self.segment_size - seq_len), value=64)

            z_tangent = self.segment_encoder(x)
            z_hyp = exp_map_zero(z_tangent, c=self.curvature)
            z_hyp = project_to_poincare(z_hyp, max_norm=0.95, c=self.curvature)

            origin = torch.zeros_like(z_hyp)
            radius = poincare_distance(z_hyp, origin, c=self.curvature)

            return {
                "z_hyp": z_hyp,
                "radius": radius,
                "n_segments": torch.ones(batch_size, device=device),
            }

        # Long sequence: segment-based encoding
        segments, segment_mask = self._split_into_segments(x, lengths)
        n_segments = segments.shape[1]

        # Encode each segment
        segments_flat = segments.view(batch_size * n_segments, self.segment_size)
        segment_embs_flat = self.segment_encoder(segments_flat)
        segment_embs = segment_embs_flat.view(batch_size, n_segments, self.latent_dim)

        # Aggregate
        z_hyp, radius = self.aggregator(segment_embs, segment_mask)

        return {
            "z_hyp": z_hyp,
            "radius": radius,
            "n_segments": torch.full((batch_size,), n_segments, device=device, dtype=torch.float),
        }

    def encode(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Convenience method to get just the embedding."""
        return self.forward(x, lengths)["z_hyp"]


def create_segment_codon_encoder(**kwargs) -> SegmentCodonEncoder:
    """Factory function for SegmentCodonEncoder."""
    return SegmentCodonEncoder(**kwargs)


__all__ = [
    "SegmentCodonEncoder",
    "CodonSegmentEncoder",
    "HierarchicalSegmentAggregator",
    "create_segment_codon_encoder",
]
