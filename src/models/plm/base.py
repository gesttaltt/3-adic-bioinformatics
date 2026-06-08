# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Base class for Protein Language Model encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class PLMEncoderBase(nn.Module, ABC):
    """Abstract base class for protein language model encoders.

    All PLM encoders should inherit from this class and implement
    the required methods for encoding protein sequences.

    Attributes:
        output_dim: Dimension of output embeddings
        device: Device for computation
    """

    def __init__(self, output_dim: int, device: str = "cuda"):
        super().__init__()
        self.output_dim = output_dim
        self._device = device

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return torch.device(self._device)

    @abstractmethod
    def encode(
        self,
        sequences: str | list[str],
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode protein sequence(s) to embeddings.

        Args:
            sequences: Single sequence or list of sequences
            return_attention: Whether to return attention weights

        Returns:
            Embeddings tensor of shape (batch, seq_len, output_dim)
            or (batch, output_dim) for pooled output.
            Optionally returns attention weights.
        """
        pass

    @abstractmethod
    def encode_batch(
        self,
        sequences: list[str],
        batch_size: int = 32,
    ) -> torch.Tensor:
        """Encode a large batch of sequences efficiently.

        Args:
            sequences: List of protein sequences
            batch_size: Batch size for processing

        Returns:
            Embeddings tensor
        """
        pass

    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension."""
        return self.output_dim

    def forward(
        self,
        sequences: str | list[str],
    ) -> torch.Tensor:
        """Forward pass - alias for encode."""
        return self.encode(sequences)

    @abstractmethod
    def get_layer_embeddings(
        self,
        sequences: str | list[str],
        layers: list[int],
    ) -> dict[int, torch.Tensor]:
        """Get embeddings from specific transformer layers.

        Args:
            sequences: Input sequences
            layers: List of layer indices to extract

        Returns:
            Dictionary mapping layer index to embeddings
        """
        pass

    def pool_embeddings(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pooling: str = "mean",
    ) -> torch.Tensor:
        """Pool sequence embeddings to fixed-size representation.

        Args:
            embeddings: (batch, seq_len, dim) embeddings
            attention_mask: (batch, seq_len) mask for valid positions
            pooling: Pooling strategy ('mean', 'max', 'cls')

        Returns:
            Pooled embeddings of shape (batch, dim)
        """
        if pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                return (embeddings * mask).sum(dim=1) / mask.sum(dim=1)
            return embeddings.mean(dim=1)

        elif pooling == "max":
            if attention_mask is not None:
                embeddings = embeddings.masked_fill(~attention_mask.unsqueeze(-1), float("-inf"))
            return embeddings.max(dim=1).values

        elif pooling == "cls":
            return embeddings[:, 0]

        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")
