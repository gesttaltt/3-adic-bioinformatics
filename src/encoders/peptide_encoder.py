# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""PeptideEncoder: Biologically-Grounded AMP Activity Predictor.

This module implements a learned peptide encoder for antimicrobial peptide (AMP)
activity prediction. Following the successful TrainableCodonEncoder pattern
(Spearman 0.60 for DDG), it uses multi-component embeddings and hyperbolic
projections to learn biologically meaningful representations.

Architecture:
    Input: Peptide Sequence (10-50 AA)
    → PeptideInputProcessor (tokenize, pad, position encode)
    → MultiComponentEmbedding (AA + 5-adic group + properties = 56D)
    → Transformer Encoder (2 layers, 4 heads)
    → Dual Pooling (mean + attention = 112D)
    → HyperbolicProjection (16D Poincaré ball)
    → MIC Prediction Head (16D → 1)

    Decoder Path:
    → Hyperbolic → Euclidean (inverse projection)
    → Transformer Decoder (2 layers, 4 heads, causal mask)
    → Sequence Output (vocab size 22)

Loss Components (6):
    1. Reconstruction (sequence cross-entropy)
    2. MIC Prediction (Smooth L1)
    3. Property Alignment (embed dist ~ property dist)
    4. Radial Hierarchy (low MIC → center)
    5. Cohesion (same pathogen clusters)
    6. Separation (different pathogens separate)

Usage:
    from src.encoders.peptide_encoder import PeptideVAE

    model = PeptideVAE(latent_dim=16)
    z_hyp = model.encode(sequences)  # (batch, 16) on Poincaré ball
    mic_pred = model.predict_mic(z_hyp)  # (batch, 1)
    decoded = model.decode(z_hyp)  # (batch, seq_len, vocab_size)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.encoders.padic_amino_acid_encoder import (
    AA_PROPERTIES,
    AA_TO_GROUP,
    AA_TO_INDEX,
    INDEX_TO_AA,
    AminoAcidGroup,
)
from src.geometry import (
    log_map_zero,
    poincare_distance,
)
from src.models.hyperbolic_projection import HyperbolicProjection

# =============================================================================
# Constants
# =============================================================================

MAX_SEQ_LEN = 50  # Maximum peptide length (padded)
VOCAB_SIZE = 22  # 20 AA + stop + unknown/pad
PAD_IDX = 21  # Index for padding token (X)


# =============================================================================
# Input Processing
# =============================================================================


class PeptideInputProcessor(nn.Module):
    """Process peptide sequences into model inputs.

    Handles:
    - Tokenization (AA → index 0-21)
    - Padding to MAX_SEQ_LEN
    - Positional encoding (sinusoidal)
    - N/C-terminal distance features
    """

    def __init__(
        self,
        max_seq_len: int = MAX_SEQ_LEN,
        embedding_dim: int = 56,
    ):
        """Initialize processor.

        Args:
            max_seq_len: Maximum sequence length
            embedding_dim: Position embedding dimension
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

        # Precompute sinusoidal positional encoding
        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if embedding_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", pe)

    def tokenize(self, sequence: str) -> Tensor:
        """Convert sequence string to token indices.

        Args:
            sequence: Amino acid sequence (uppercase)

        Returns:
            Token indices tensor (seq_len,)
        """
        indices = []
        for aa in sequence.upper():
            idx = AA_TO_INDEX.get(aa, PAD_IDX)
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long)

    def pad_sequence(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Pad sequence to max_seq_len.

        Args:
            tokens: Token indices (seq_len,)

        Returns:
            Tuple of (padded_tokens, attention_mask)
        """
        seq_len = tokens.shape[0]

        if seq_len > self.max_seq_len:
            # Truncate
            padded = tokens[: self.max_seq_len]
            mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        else:
            # Pad
            padded = F.pad(tokens, (0, self.max_seq_len - seq_len), value=PAD_IDX)
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
            mask[:seq_len] = True

        return padded, mask

    def get_position_embeddings(self, seq_len: int, device: torch.device) -> Tensor:
        """Get positional embeddings for sequence.

        Args:
            seq_len: Actual sequence length
            device: Target device

        Returns:
            Position embeddings (max_seq_len, embedding_dim)
        """
        return self.positional_encoding[: self.max_seq_len].to(device)

    def get_terminal_features(self, seq_len: int, device: torch.device) -> Tensor:
        """Get N/C-terminal distance features.

        Args:
            seq_len: Actual sequence length
            device: Target device

        Returns:
            Terminal features (max_seq_len, 2) - [n_term_dist, c_term_dist]
        """
        features = torch.zeros(self.max_seq_len, 2, device=device)
        if seq_len > 0:
            positions = torch.arange(self.max_seq_len, device=device).float()
            # N-terminal distance (0 at start)
            features[:, 0] = positions / max(seq_len - 1, 1)
            # C-terminal distance (0 at end)
            features[:, 1] = (seq_len - 1 - positions).clamp(min=0) / max(seq_len - 1, 1)
        return features

    def forward(
        self,
        sequences: list[str],
    ) -> dict[str, Tensor]:
        """Process batch of sequences.

        Args:
            sequences: List of AA sequences

        Returns:
            Dictionary with tokens, mask, positions, terminal_features
        """
        len(sequences)
        device = self.positional_encoding.device

        all_tokens = []
        all_masks = []
        all_lengths = []

        for seq in sequences:
            tokens = self.tokenize(seq)
            padded, mask = self.pad_sequence(tokens)
            all_tokens.append(padded)
            all_masks.append(mask)
            all_lengths.append(len(seq))

        tokens_batch = torch.stack(all_tokens).to(device)
        masks_batch = torch.stack(all_masks).to(device)

        # Position embeddings (shared across batch)
        positions = self.get_position_embeddings(self.max_seq_len, device)

        # Terminal features per sequence
        terminal_features = torch.stack([self.get_terminal_features(length, device) for length in all_lengths])

        return {
            "tokens": tokens_batch,
            "mask": masks_batch,
            "positions": positions,
            "terminal_features": terminal_features,
            "lengths": torch.tensor(all_lengths, device=device),
        }


# =============================================================================
# Multi-Component Embedding
# =============================================================================


class PropertyEncoder(nn.Module):
    """Encode amino acid physicochemical properties to learned embeddings."""

    def __init__(
        self,
        output_dim: int = 8,
        n_properties: int = 4,
    ):
        """Initialize property encoder.

        Args:
            output_dim: Output embedding dimension
            n_properties: Number of input properties (hydrophobicity, MW, pI, flexibility)
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_properties, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )

        # Register normalized AA properties as buffer
        props = torch.zeros(VOCAB_SIZE, n_properties)
        for aa, idx in AA_TO_INDEX.items():
            if idx < VOCAB_SIZE and aa in AA_PROPERTIES:
                p = AA_PROPERTIES[aa]
                # Normalize to ~[0, 1]
                props[idx] = torch.tensor(
                    [
                        (p[0] + 5) / 10,  # hydrophobicity: [-4.5, 4.5] → [0, 1]
                        p[1] / 250,  # molecular weight: [75, 204] → ~[0.3, 0.8]
                        p[2] / 14,  # isoelectric point: [2.77, 10.76] → ~[0.2, 0.8]
                        p[3],  # flexibility: already [0, 1]
                    ]
                )
        self.register_buffer("aa_properties", props)

    def forward(self, token_indices: Tensor) -> Tensor:
        """Encode token properties.

        Args:
            token_indices: Token indices (batch, seq_len)

        Returns:
            Property embeddings (batch, seq_len, output_dim)
        """
        props = self.aa_properties[token_indices]
        return self.encoder(props)


class MultiComponentEmbedding(nn.Module):
    """Multi-component embedding combining AA, group, and property information.

    Total dimension: aa_dim + group_dim + property_dim = 32 + 16 + 8 = 56
    """

    def __init__(
        self,
        aa_dim: int = 32,
        group_dim: int = 16,
        property_dim: int = 8,
        dropout: float = 0.1,
    ):
        """Initialize multi-component embedding.

        Args:
            aa_dim: AA embedding dimension
            group_dim: 5-adic group embedding dimension
            property_dim: Property encoding dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.total_dim = aa_dim + group_dim + property_dim

        # AA embedding (22 tokens)
        self.aa_embedding = nn.Embedding(VOCAB_SIZE, aa_dim, padding_idx=PAD_IDX)

        # 5-adic group embedding (5 groups)
        self.group_embedding = nn.Embedding(5, group_dim)

        # Property encoder
        self.property_encoder = PropertyEncoder(output_dim=property_dim)

        # Normalization and dropout
        self.norm = nn.LayerNorm(self.total_dim)
        self.dropout = nn.Dropout(dropout)

        # Register AA to group mapping
        groups = torch.zeros(VOCAB_SIZE, dtype=torch.long)
        for aa, idx in AA_TO_INDEX.items():
            if idx < VOCAB_SIZE:
                groups[idx] = AA_TO_GROUP.get(aa, AminoAcidGroup.SPECIAL)
        self.register_buffer("aa_to_group", groups)

    def forward(self, token_indices: Tensor) -> Tensor:
        """Get multi-component embeddings.

        Args:
            token_indices: Token indices (batch, seq_len)

        Returns:
            Combined embeddings (batch, seq_len, total_dim)
        """
        # AA embeddings
        aa_emb = self.aa_embedding(token_indices)

        # Group embeddings
        group_indices = self.aa_to_group[token_indices]
        group_emb = self.group_embedding(group_indices)

        # Property embeddings
        prop_emb = self.property_encoder(token_indices)

        # Concatenate
        combined = torch.cat([aa_emb, group_emb, prop_emb], dim=-1)
        combined = self.norm(combined)
        combined = self.dropout(combined)

        return combined


# =============================================================================
# Attention Pooling
# =============================================================================


class AttentionPooling(nn.Module):
    """Learned attention pooling for sequence aggregation."""

    def __init__(
        self,
        input_dim: int,
        n_heads: int = 4,
    ):
        """Initialize attention pooling.

        Args:
            input_dim: Input feature dimension
            n_heads: Number of attention heads
        """
        super().__init__()

        # Learned query for attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, input_dim) * 0.02)

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=n_heads,
            batch_first=True,
        )

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Apply attention pooling.

        Args:
            x: Sequence features (batch, seq_len, dim)
            mask: Attention mask (batch, seq_len), True for valid positions

        Returns:
            Pooled features (batch, dim)
        """
        batch_size = x.shape[0]

        # Expand query to batch
        query = self.query.expand(batch_size, -1, -1)

        # Create key padding mask (True = ignore)
        if mask is not None:
            key_padding_mask = ~mask  # Invert: True means padding
        else:
            key_padding_mask = None

        # Attention pooling
        pooled, _ = self.attention(
            query,
            x,
            x,
            key_padding_mask=key_padding_mask,
        )

        return pooled.squeeze(1)


# =============================================================================
# Peptide Encoder
# =============================================================================


class PeptideEncoderTransformer(nn.Module):
    """Transformer-based peptide encoder to hyperbolic space."""

    def __init__(
        self,
        embedding_dim: int = 56,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_radius: float = 0.95,
        curvature: float = 1.0,
    ):
        """Initialize peptide encoder.

        Args:
            embedding_dim: Input embedding dimension (from MultiComponentEmbedding)
            hidden_dim: Transformer hidden dimension
            latent_dim: Output latent dimension (Poincaré ball)
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            dropout: Dropout rate
            max_radius: Maximum radius in Poincaré ball
            curvature: Hyperbolic curvature
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.curvature = curvature
        self.max_radius = max_radius

        # Project embedding to hidden dim
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)

        # Positional encoding (in hidden_dim space)
        pe = torch.zeros(MAX_SEQ_LEN, hidden_dim)
        position = torch.arange(0, MAX_SEQ_LEN, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if hidden_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", pe)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Dual pooling
        self.mean_pool_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attention_pool = AttentionPooling(hidden_dim, n_heads=n_heads)

        # Fusion layer (mean + attention = 2 * hidden_dim → hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Hyperbolic projection
        self.hyperbolic_proj = HyperbolicProjection(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            max_radius=max_radius,
            curvature=curvature,
            n_layers=1,
            dropout=dropout,
        )

        # Pre-projection from fusion to latent
        self.pre_projection = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        embeddings: Tensor,
        mask: Tensor | None = None,
        positions: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Encode peptide embeddings to hyperbolic space.

        Args:
            embeddings: Multi-component embeddings (batch, seq_len, embedding_dim)
            mask: Attention mask (batch, seq_len), True for valid
            positions: Position embeddings (seq_len, embedding_dim)

        Returns:
            Dictionary with z_hyp, z_euclidean, direction, radius
        """
        embeddings.shape[0]

        # Project to hidden dim
        x = self.input_proj(embeddings)

        # Add positional encoding (use internal PE in hidden_dim space)
        x = x + self.positional_encoding[: x.shape[1]].unsqueeze(0)

        # Create transformer mask (True = ignore)
        src_key_padding_mask = ~mask if mask is not None else None

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Dual pooling
        # Mean pooling (masked)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            mean_pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            mean_pooled = x.mean(dim=1)
        mean_pooled = self.mean_pool_proj(mean_pooled)

        # Attention pooling
        attn_pooled = self.attention_pool(x, mask)

        # Fuse pooled representations
        fused = self.fusion(torch.cat([mean_pooled, attn_pooled], dim=-1))

        # Project to latent dimension
        z_euclidean = self.pre_projection(fused)

        # Project to Poincaré ball with components
        z_hyp, direction, radius = self.hyperbolic_proj.forward_with_components(z_euclidean)

        return {
            "z_hyp": z_hyp,
            "z_euclidean": z_euclidean,
            "direction": direction,
            "radius": radius,
            "transformer_output": x,
        }


# =============================================================================
# Peptide Decoder
# =============================================================================


class PeptideDecoder(nn.Module):
    """Transformer-based decoder for sequence reconstruction."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        embedding_dim: int = 56,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = MAX_SEQ_LEN,
        curvature: float = 1.0,
    ):
        """Initialize peptide decoder.

        Args:
            latent_dim: Input latent dimension
            hidden_dim: Transformer hidden dimension
            embedding_dim: Target embedding dimension
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            curvature: Hyperbolic curvature (for inverse projection)
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.curvature = curvature

        # Inverse hyperbolic projection: Poincaré → Euclidean
        self.inverse_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Initial sequence embedding (for autoregressive decoding start)
        self.start_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Target embedding (for teacher forcing)
        self.target_embedding = nn.Embedding(VOCAB_SIZE, hidden_dim, padding_idx=PAD_IDX)

        # Positional encoding
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        if hidden_dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", pe)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Output projection to vocabulary
        self.output_proj = nn.Linear(hidden_dim, VOCAB_SIZE)

        # Register causal mask
        causal_mask = torch.triu(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self,
        z_hyp: Tensor,
        target_tokens: Tensor | None = None,
        target_mask: Tensor | None = None,
    ) -> Tensor:
        """Decode from hyperbolic latent to sequence logits.

        Args:
            z_hyp: Hyperbolic latent (batch, latent_dim)
            target_tokens: Target tokens for teacher forcing (batch, seq_len)
            target_mask: Target mask (batch, seq_len)

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        batch_size = z_hyp.shape[0]
        device = z_hyp.device

        # Apply log map to get tangent space representation
        z_tangent = log_map_zero(z_hyp, c=self.curvature)

        # Inverse projection
        memory = self.inverse_proj(z_tangent)
        memory = memory.unsqueeze(1)  # (batch, 1, hidden_dim)

        if target_tokens is not None:
            # Teacher forcing mode
            seq_len = target_tokens.shape[1]

            # Embed targets
            tgt = self.target_embedding(target_tokens)
            tgt = tgt + self.positional_encoding[:seq_len].unsqueeze(0)

            # Create masks
            tgt_mask = self.causal_mask[:seq_len, :seq_len].to(device)
            tgt_key_padding_mask = ~target_mask if target_mask is not None else None

            # Decode
            output = self.transformer(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )
        else:
            # Autoregressive mode (for inference)
            # Start with start token
            tgt = self.start_token.expand(batch_size, -1, -1)
            outputs = []

            for _i in range(self.max_seq_len):
                # Add positional encoding
                tgt_pos = tgt + self.positional_encoding[: tgt.shape[1]].unsqueeze(0)

                # Create causal mask
                tgt_mask = self.causal_mask[: tgt.shape[1], : tgt.shape[1]].to(device)

                # Decode one step
                output = self.transformer(tgt_pos, memory, tgt_mask=tgt_mask)

                # Get last token prediction
                last_output = output[:, -1:, :]
                outputs.append(last_output)

                # Predict next token
                logits = self.output_proj(last_output)
                next_token = logits.argmax(dim=-1)

                # Embed and append
                next_emb = self.target_embedding(next_token)
                tgt = torch.cat([tgt, next_emb], dim=1)

            output = torch.cat(outputs, dim=1)

        # Project to vocabulary
        logits = self.output_proj(output)

        return logits


# =============================================================================
# MIC Prediction Head
# =============================================================================


class MICPredictionHead(nn.Module):
    """Prediction head for MIC (Minimum Inhibitory Concentration)."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ):
        """Initialize MIC prediction head.

        Args:
            latent_dim: Input dimension (from hyperbolic space)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_hyp: Tensor) -> Tensor:
        """Predict log10(MIC) from hyperbolic embedding.

        Args:
            z_hyp: Hyperbolic embeddings (batch, latent_dim)

        Returns:
            Predicted log10(MIC) (batch, 1)
        """
        return self.predictor(z_hyp)


# =============================================================================
# Full PeptideVAE Model
# =============================================================================


class PeptideVAE(nn.Module):
    """Full Peptide VAE with encoder, decoder, and MIC prediction.

    This is the main model class integrating all components for
    antimicrobial peptide activity prediction.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 128,
        embedding_dim: int = 56,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_radius: float = 0.95,
        curvature: float = 1.0,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        """Initialize PeptideVAE.

        Args:
            latent_dim: Latent dimension in Poincaré ball
            hidden_dim: Transformer hidden dimension
            embedding_dim: Multi-component embedding dimension
            n_layers: Transformer layers
            n_heads: Attention heads
            dropout: Dropout rate
            max_radius: Maximum Poincaré ball radius
            curvature: Hyperbolic curvature
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.curvature = curvature
        self.max_radius = max_radius
        self.max_seq_len = max_seq_len

        # Input processing
        self.input_processor = PeptideInputProcessor(
            max_seq_len=max_seq_len,
            embedding_dim=embedding_dim,
        )

        # Multi-component embedding
        self.embedding = MultiComponentEmbedding(
            aa_dim=32,
            group_dim=16,
            property_dim=8,
            dropout=dropout,
        )

        # Encoder
        self.encoder = PeptideEncoderTransformer(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_radius=max_radius,
            curvature=curvature,
        )

        # Decoder
        self.decoder = PeptideDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            curvature=curvature,
        )

        # MIC prediction head
        self.mic_head = MICPredictionHead(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim // 4,
            dropout=dropout,
        )

    def encode(
        self,
        sequences: list[str],
    ) -> dict[str, Tensor]:
        """Encode peptide sequences to hyperbolic space.

        Args:
            sequences: List of amino acid sequences

        Returns:
            Dictionary with z_hyp, z_euclidean, direction, radius, etc.
        """
        # Process inputs
        inputs = self.input_processor(sequences)

        # Get multi-component embeddings
        embeddings = self.embedding(inputs["tokens"])

        # Encode to hyperbolic space
        encoder_output = self.encoder(
            embeddings,
            mask=inputs["mask"],
            positions=inputs["positions"],
        )

        # Add input info to output
        encoder_output["tokens"] = inputs["tokens"]
        encoder_output["mask"] = inputs["mask"]
        encoder_output["lengths"] = inputs["lengths"]

        return encoder_output

    def decode(
        self,
        z_hyp: Tensor,
        target_tokens: Tensor | None = None,
        target_mask: Tensor | None = None,
    ) -> Tensor:
        """Decode from hyperbolic latent to sequence.

        Args:
            z_hyp: Hyperbolic latent (batch, latent_dim)
            target_tokens: Target for teacher forcing (batch, seq_len)
            target_mask: Target mask (batch, seq_len)

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        return self.decoder(z_hyp, target_tokens, target_mask)

    def predict_mic(self, z_hyp: Tensor) -> Tensor:
        """Predict MIC from hyperbolic embedding.

        Args:
            z_hyp: Hyperbolic embedding (batch, latent_dim)

        Returns:
            Predicted log10(MIC) (batch, 1)
        """
        return self.mic_head(z_hyp)

    def forward(
        self,
        sequences: list[str],
        teacher_forcing: bool = True,
    ) -> dict[str, Tensor]:
        """Full forward pass.

        Args:
            sequences: List of peptide sequences
            teacher_forcing: Use teacher forcing for decoder

        Returns:
            Dictionary with all model outputs
        """
        # Encode
        encoder_output = self.encode(sequences)

        # Decode with teacher forcing
        if teacher_forcing:
            logits = self.decode(
                encoder_output["z_hyp"],
                target_tokens=encoder_output["tokens"],
                target_mask=encoder_output["mask"],
            )
        else:
            logits = self.decode(encoder_output["z_hyp"])

        # Predict MIC
        mic_pred = self.predict_mic(encoder_output["z_hyp"])

        return {
            **encoder_output,
            "logits": logits,
            "mic_pred": mic_pred,
        }

    def get_hyperbolic_radii(self, z_hyp: Tensor) -> Tensor:
        """Get hyperbolic radii (distance from origin).

        Args:
            z_hyp: Hyperbolic embeddings (batch, latent_dim)

        Returns:
            Radii tensor (batch,)
        """
        origin = torch.zeros(1, self.latent_dim, device=z_hyp.device)
        radii = poincare_distance(z_hyp, origin.expand(z_hyp.shape[0], -1), c=self.curvature)
        return radii

    def generate(
        self,
        z_hyp: Tensor,
        temperature: float = 1.0,
        max_len: int | None = None,
    ) -> list[str]:
        """Generate sequences from latent codes.

        Args:
            z_hyp: Hyperbolic latent codes (batch, latent_dim)
            temperature: Sampling temperature
            max_len: Maximum generation length

        Returns:
            List of generated sequences
        """
        self.eval()
        max_len = max_len or self.max_seq_len

        with torch.no_grad():
            logits = self.decode(z_hyp)

            if temperature != 1.0:
                logits = logits / temperature

            # Get predicted tokens
            tokens = logits.argmax(dim=-1)

            # Convert to sequences
            sequences = []
            for token_seq in tokens:
                seq = []
                for idx in token_seq.cpu().numpy():
                    if idx == PAD_IDX:
                        break
                    aa = INDEX_TO_AA.get(idx, "X")
                    if aa == "*":
                        break
                    seq.append(aa)
                sequences.append("".join(seq))

            return sequences


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "PeptideInputProcessor",
    "PropertyEncoder",
    "MultiComponentEmbedding",
    "AttentionPooling",
    "PeptideEncoderTransformer",
    "PeptideDecoder",
    "MICPredictionHead",
    "PeptideVAE",
    "MAX_SEQ_LEN",
    "VOCAB_SIZE",
    "PAD_IDX",
]
