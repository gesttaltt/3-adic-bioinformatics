# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Transformer heads for precise DDG prediction.

This module implements transformer architectures that process full
protein sequences for precise DDG predictions:

1. DDGTransformer: Full-sequence transformer with mutation-aware attention
2. HierarchicalTransformer: Two-level attention (local + global)

Both are designed to work within 6GB VRAM constraints (RTX 3050).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Configuration for DDG transformers."""

    # Sequence parameters
    max_seq_len: int = 256  # Reduced for memory
    vocab_size: int = 22  # 20 AA + gap + mask

    # Architecture
    d_model: int = 128  # Reduced for memory
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 512
    dropout: float = 0.1

    # Memory optimization
    use_gradient_checkpointing: bool = True
    use_flash_attention: bool = False  # If available

    # Hierarchical transformer
    local_window: int = 21  # Local context around mutation
    stride: int = 4  # Stride for global context

    # Pre-LayerNorm (more stable)
    use_pre_layernorm: bool = True

    # Gated MLP
    use_gated_mlp: bool = True


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: Input tensor (batch, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class GatedMLP(nn.Module):
    """Gated MLP (SwiGLU variant)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class TransformerBlock(nn.Module):
    """Single transformer block with optional gradient checkpointing."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Feed-forward
        if config.use_gated_mlp:
            self.ff = GatedMLP(config.d_model, config.d_ff, config.dropout)
        else:
            self.ff = nn.Sequential(
                nn.Linear(config.d_model, config.d_ff),
                nn.SiLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_ff, config.d_model),
                nn.Dropout(config.dropout),
            )

        # Layer norms
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        self.use_pre_layernorm = config.use_pre_layernorm

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, seq_len, d_model)
            attn_mask: Attention mask
            key_padding_mask: Key padding mask

        Returns:
            Output tensor (batch, seq_len, d_model)
        """
        if self.use_pre_layernorm:
            # Pre-LN: norm -> attention -> residual
            normed = self.norm1(x)
            attn_out, _ = self.attn(
                normed,
                normed,
                normed,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
            x = x + attn_out

            normed = self.norm2(x)
            x = x + self.ff(normed)
        else:
            # Post-LN: attention -> residual -> norm
            attn_out, _ = self.attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
            x = self.norm1(x + attn_out)
            x = self.norm2(x + self.ff(x))

        return x


class DDGTransformer(nn.Module):
    """Full-sequence transformer for DDG prediction.

    Processes the full protein sequence with mutation-aware attention
    to capture long-range context effects on stability.

    Memory-optimized for 6GB VRAM:
    - Reduced d_model (128 vs 256)
    - Gradient checkpointing
    - Smaller max sequence length
    """

    def __init__(self, config: TransformerConfig | None = None):
        super().__init__()

        if config is None:
            config = TransformerConfig()

        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(config.d_model, config.max_seq_len, config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)

        # Mutation-aware attention for pooling
        self.mutation_query = nn.Parameter(torch.randn(1, 1, config.d_model))
        self.mutation_attn = nn.MultiheadAttention(config.d_model, config.n_heads, config.dropout, batch_first=True)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        mutation_pos: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            sequence: Token indices (batch, seq_len)
            mutation_pos: Mutation position indices (batch,)
            padding_mask: Padding mask (batch, seq_len)

        Returns:
            Dictionary with 'ddg_pred' and 'attention'
        """
        batch_size = sequence.size(0)

        # Embed
        x = self.embedding(sequence)
        x = self.pos_encoding(x)

        # Process through transformer blocks
        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x, key_padding_mask=padding_mask)

        x = self.final_norm(x)

        # Mutation-aware pooling
        query = self.mutation_query.expand(batch_size, -1, -1)
        context, attn_weights = self.mutation_attn(query, x, x, key_padding_mask=padding_mask)

        # Predict DDG
        ddg_pred = self.head(context.squeeze(1))

        return {
            "ddg_pred": ddg_pred,
            "attention": attn_weights,
            "sequence_repr": x,
        }

    def predict(
        self,
        sequence: torch.Tensor,
        mutation_pos: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Make DDG predictions."""
        self.eval()
        with torch.no_grad():
            return self.forward(sequence, mutation_pos, padding_mask)["ddg_pred"]


class HierarchicalTransformer(nn.Module):
    """Hierarchical transformer with local + global attention.

    Uses two-level attention:
    1. Local encoder: Processes mutation site ± window residues
    2. Global encoder: Processes full sequence with striding
    3. Cross-attention: Local queries attend to global context

    More memory-efficient than full-sequence transformer.
    """

    def __init__(self, config: TransformerConfig | None = None):
        super().__init__()

        if config is None:
            config = TransformerConfig()

        self.config = config

        # Shared embedding
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(config.d_model, config.max_seq_len, config.dropout)

        # Local encoder (mutation context)
        self.local_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.local_norm = nn.LayerNorm(config.d_model)

        # Global encoder (strided sequence)
        self.global_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.global_norm = nn.LayerNorm(config.d_model)

        # Cross-attention: local queries, global keys/values
        self.cross_attn = nn.MultiheadAttention(config.d_model, config.n_heads, config.dropout, batch_first=True)

        # Fusion and prediction
        self.fusion = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.SiLU(),
            nn.Dropout(config.dropout),
        )

        self.head = nn.Linear(config.d_model, 1)

    def _get_local_window(
        self,
        sequence: torch.Tensor,
        mutation_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Extract local window around mutation.

        Args:
            sequence: Token indices (batch, seq_len)
            mutation_pos: Mutation positions (batch,)

        Returns:
            Local window (batch, local_window)
        """
        batch_size, seq_len = sequence.shape
        half_window = self.config.local_window // 2

        # Pad sequence for edge cases
        padded = F.pad(sequence, (half_window, half_window), value=0)

        # Extract windows
        windows = []
        for i in range(batch_size):
            pos = mutation_pos[i].item() + half_window
            start = max(0, pos - half_window)
            end = start + self.config.local_window
            windows.append(padded[i, start:end])

        return torch.stack(windows)

    def _get_strided_global(self, sequence: torch.Tensor) -> torch.Tensor:
        """Get strided global context.

        Args:
            sequence: Token indices (batch, seq_len)

        Returns:
            Strided sequence (batch, seq_len // stride)
        """
        return sequence[:, :: self.config.stride]

    def forward(
        self,
        sequence: torch.Tensor,
        mutation_pos: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            sequence: Token indices (batch, seq_len)
            mutation_pos: Mutation position indices (batch,)
            padding_mask: Padding mask (batch, seq_len)

        Returns:
            Dictionary with 'ddg_pred'
        """
        # Get local window
        local_seq = self._get_local_window(sequence, mutation_pos)
        x_local = self.embedding(local_seq)
        x_local = self.pos_encoding(x_local)

        # Process local context
        for block in self.local_blocks:
            if self.config.use_gradient_checkpointing and self.training:
                x_local = torch.utils.checkpoint.checkpoint(block, x_local, use_reentrant=False)
            else:
                x_local = block(x_local)
        x_local = self.local_norm(x_local)
        local_repr = x_local.mean(dim=1)  # Pool

        # Get strided global context
        global_seq = self._get_strided_global(sequence)
        x_global = self.embedding(global_seq)
        x_global = self.pos_encoding(x_global)

        # Process global context
        global_padding_mask = None
        if padding_mask is not None:
            global_padding_mask = padding_mask[:, :: self.config.stride]

        for block in self.global_blocks:
            if self.config.use_gradient_checkpointing and self.training:
                x_global = torch.utils.checkpoint.checkpoint(block, x_global, use_reentrant=False)
            else:
                x_global = block(x_global, key_padding_mask=global_padding_mask)
        x_global = self.global_norm(x_global)

        # Cross-attention: local queries global context
        local_query = local_repr.unsqueeze(1)
        context, _ = self.cross_attn(
            local_query,
            x_global,
            x_global,
            key_padding_mask=global_padding_mask,
        )
        context = context.squeeze(1)

        # Fuse and predict
        fused = self.fusion(torch.cat([local_repr, context], dim=-1))
        ddg_pred = self.head(fused)

        return {
            "ddg_pred": ddg_pred,
            "local_repr": local_repr,
            "global_repr": x_global.mean(dim=1),
        }

    def predict(
        self,
        sequence: torch.Tensor,
        mutation_pos: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Make DDG predictions."""
        self.eval()
        with torch.no_grad():
            return self.forward(sequence, mutation_pos, padding_mask)["ddg_pred"]


__all__ = [
    "TransformerConfig",
    "SinusoidalPositionalEncoding",
    "GatedMLP",
    "TransformerBlock",
    "DDGTransformer",
    "HierarchicalTransformer",
]
