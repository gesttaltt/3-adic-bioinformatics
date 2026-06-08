# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Hyperbolic embedding for CRISPR off-target analysis.

This module provides neural network embedding of CRISPR target/off-target
pairs in hyperbolic space, capturing sequence similarity patterns.

Single responsibility: Hyperbolic sequence embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geometry import poincare_distance, project_to_poincare

from .types import NUCLEOTIDE_TO_IDX


def encode_sequence(sequence: str) -> torch.Tensor:
    """Encode DNA sequence to tensor.

    Args:
        sequence: DNA sequence string

    Returns:
        Tensor of nucleotide indices
    """
    indices = [NUCLEOTIDE_TO_IDX.get(nt.upper(), 4) for nt in sequence]
    return torch.tensor(indices)


def sequence_to_onehot(sequence: str) -> torch.Tensor:
    """Convert sequence to one-hot encoding.

    Args:
        sequence: DNA sequence string

    Returns:
        One-hot tensor of shape (seq_len, 4)
    """
    indices = encode_sequence(sequence)
    onehot = F.one_hot(indices.clamp(0, 3), num_classes=4).float()
    return onehot


class HyperbolicOfftargetEmbedder(nn.Module):
    """Embeds CRISPR target/off-target pairs in hyperbolic space.

    The hyperbolic embedding captures:
    - Distance from guide = off-target risk
    - Radial position = overall specificity
    - Angular position = mismatch pattern

    Attributes:
        seq_len: Length of guide sequence (typically 20)
        embedding_dim: Dimension of hyperbolic embedding
        curvature: Hyperbolic curvature parameter
        max_norm: Maximum norm for Poincaré ball
    """

    def __init__(
        self,
        seq_len: int = 20,
        embedding_dim: int = 64,
        curvature: float = 1.0,
        max_norm: float = 0.95,
    ):
        """Initialize embedder.

        Args:
            seq_len: Length of guide sequence (typically 20)
            embedding_dim: Dimension of hyperbolic embedding
            curvature: Hyperbolic curvature parameter
            max_norm: Maximum norm for Poincaré ball
        """
        super().__init__()
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.curvature = curvature
        self.max_norm = max_norm

        # Nucleotide embedding
        self.nt_embedding = nn.Embedding(5, 16)  # 4 nucleotides + N

        # Position encoding
        self.pos_embedding = nn.Embedding(seq_len, 16)

        # Sequence encoder
        self.seq_encoder = nn.Sequential(
            nn.Linear(32 * seq_len, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim),
        )

        # Mismatch-aware attention
        self.mismatch_attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            batch_first=True,
        )

    def project_to_poincare(self, x: torch.Tensor) -> torch.Tensor:
        """Project to Poincaré ball using geometry module.

        Args:
            x: Input tensor

        Returns:
            Projected tensor with norm < max_norm
        """
        return project_to_poincare(x, max_norm=self.max_norm, c=self.curvature)

    def encode_sequence(self, sequences: list[str]) -> torch.Tensor:
        """Encode multiple sequences.

        Args:
            sequences: List of DNA sequences

        Returns:
            Encoded tensor (batch, seq_len, 32)
        """
        batch_size = len(sequences)

        # Encode nucleotides
        nt_indices = torch.stack([encode_sequence(s) for s in sequences])
        nt_emb = self.nt_embedding(nt_indices)  # (batch, seq_len, 16)

        # Add position embeddings
        positions = torch.arange(self.seq_len, device=nt_emb.device)
        pos_emb = self.pos_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)

        # Combine
        combined = torch.cat([nt_emb, pos_emb], dim=-1)  # (batch, seq_len, 32)

        return combined

    def forward(
        self,
        target_sequences: list[str],
        offtarget_sequences: list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Embed sequences in hyperbolic space.

        Args:
            target_sequences: List of guide RNA targets
            offtarget_sequences: Optional list of off-target sequences

        Returns:
            Dictionary with embeddings and distances
        """
        # Encode targets
        target_encoded = self.encode_sequence(target_sequences)

        # Apply attention
        target_attended, _ = self.mismatch_attention(target_encoded, target_encoded, target_encoded)

        # Flatten and project
        target_flat = target_attended.flatten(start_dim=1)
        target_emb = self.seq_encoder(target_flat)
        target_emb = self.project_to_poincare(target_emb)

        result = {"target_embeddings": target_emb}

        # If off-targets provided, compute relative embeddings
        if offtarget_sequences is not None:
            offtarget_encoded = self.encode_sequence(offtarget_sequences)

            # Cross-attention with target
            offtarget_attended, attn_weights = self.mismatch_attention(
                offtarget_encoded, target_encoded, target_encoded
            )

            offtarget_flat = offtarget_attended.flatten(start_dim=1)
            offtarget_emb = self.seq_encoder(offtarget_flat)
            offtarget_emb = self.project_to_poincare(offtarget_emb)

            result["offtarget_embeddings"] = offtarget_emb
            result["attention_weights"] = attn_weights

            # V5.12.2: Use proper hyperbolic distance function
            hyperbolic_dist = poincare_distance(target_emb, offtarget_emb, c=self.curvature)
            result["hyperbolic_distances"] = hyperbolic_dist

        return result


__all__ = [
    "HyperbolicOfftargetEmbedder",
    "encode_sequence",
    "sequence_to_onehot",
]
