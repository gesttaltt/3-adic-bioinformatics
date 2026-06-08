# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Discrete diffusion model for codon sequences.

This module provides a diffusion model specifically designed for
generating and optimizing codon sequences. Based on D3PM (Discrete
Denoising Diffusion Probabilistic Models).

References:
    - Austin et al., "Structured Denoising Diffusion Models" (2021)
    - Hoogeboom et al., "Argmax Flows and Multinomial Diffusion" (2021)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .noise_schedule import DiscreteNoiseScheduler


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, : x.size(1)]


class TimestepEmbedding(nn.Module):
    """Embedding for diffusion timesteps."""

    def __init__(self, d_model: int, max_timesteps: int = 1000):
        super().__init__()
        self.d_model = d_model

        # Sinusoidal embedding
        half_dim = d_model // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim) * -emb)
        self.register_buffer("emb", emb)

        # Projection
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Get timestep embedding.

        Args:
            t: Timesteps of shape (batch,)

        Returns:
            Embeddings of shape (batch, d_model)
        """
        t = t.float()
        emb = t[:, None] * self.emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return self.proj(emb)


class TransformerDenoiser(nn.Module):
    """Transformer-based denoiser for discrete diffusion.

    Args:
        vocab_size: Size of vocabulary (64 for codons)
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        max_len: Maximum sequence length
    """

    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.timestep_embedding = TimestepEmbedding(d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output projection
        self.output = nn.Linear(d_model, vocab_size)

        # Layer norms
        self.embed_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        context: Tensor | None = None,
    ) -> Tensor:
        """Predict denoised logits.

        Args:
            x: Noised token indices of shape (batch, seq_len)
            t: Timesteps of shape (batch,)
            context: Optional context embeddings (batch, context_len, d_model)

        Returns:
            Logits of shape (batch, seq_len, vocab_size)
        """
        # Embed tokens
        h = self.token_embedding(x)  # (batch, seq, d_model)
        h = self.pos_encoding(h)

        # Add timestep information
        t_emb = self.timestep_embedding(t)  # (batch, d_model)
        h = h + t_emb.unsqueeze(1)  # Broadcast to all positions

        h = self.embed_norm(h)

        # Concatenate context if provided
        if context is not None:
            h = torch.cat([context, h], dim=1)

        # Transformer
        h = self.transformer(h)

        # Remove context tokens if they were added
        if context is not None:
            h = h[:, context.shape[1] :]

        # Output logits
        return self.output(h)


class CodonDiffusion(nn.Module):
    """Discrete diffusion model for codon sequences.

    Uses absorbing state diffusion (D3PM) for discrete tokens.

    Args:
        n_steps: Number of diffusion steps
        vocab_size: Vocabulary size (64 for codons)
        hidden_dim: Hidden dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        schedule_type: Noise schedule type
        max_len: Maximum sequence length
    """

    def __init__(
        self,
        n_steps: int = 1000,
        vocab_size: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        schedule_type: str = "cosine",
        max_len: int = 512,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_len = max_len

        # Noise scheduler
        self.scheduler = DiscreteNoiseScheduler(
            n_steps=n_steps,
            vocab_size=vocab_size,
            schedule_type=schedule_type,
        )

        # Denoiser network
        self.denoiser = TransformerDenoiser(
            vocab_size=vocab_size,
            d_model=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            max_len=max_len,
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor | None = None,
        context: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Training forward pass.

        Args:
            x: Original codon indices of shape (batch, seq_len)
            t: Timesteps (if None, sampled randomly)
            context: Optional conditioning context

        Returns:
            Dictionary with loss and other metrics
        """
        batch_size, seq_len = x.shape
        device = x.device

        # Sample timesteps if not provided
        if t is None:
            t = torch.randint(0, self.n_steps, (batch_size,), device=device)

        # Add noise
        x_noisy = self.scheduler.add_noise(x, t)

        # Predict original tokens
        logits = self.denoiser(x_noisy, t, context)

        # Compute loss (cross-entropy)
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            x.view(-1),
            reduction="mean",
        )

        # Compute accuracy
        preds = logits.argmax(dim=-1)
        accuracy = (preds == x).float().mean()

        return {
            "loss": loss,
            "accuracy": accuracy,
            "logits": logits,
        }

    def training_step(
        self,
        x: Tensor,
        context: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Single training step.

        Args:
            x: Original codon indices
            context: Optional conditioning context

        Returns:
            Dictionary with loss
        """
        return self.forward(x, context=context)

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        seq_length: int,
        temperature: float = 1.0,
        context: Tensor | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        """Generate codon sequences via reverse diffusion.

        Args:
            n_samples: Number of sequences to generate
            seq_length: Length of sequences
            temperature: Sampling temperature
            context: Optional conditioning context
            device: Device to generate on

        Returns:
            Generated sequences of shape (n_samples, seq_length)
        """
        if device is None:
            device = next(self.parameters()).device

        # Start from absorbing state (all mask tokens)
        x = torch.full(
            (n_samples, seq_length),
            self.scheduler.absorbing_state,
            dtype=torch.long,
            device=device,
        )

        # Reverse diffusion
        for t in reversed(range(self.n_steps)):
            t_tensor = torch.full((n_samples,), t, dtype=torch.long, device=device)

            # Get model prediction
            logits = self.denoiser(x, t_tensor, context)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Sample from posterior
            if t > 0:
                probs = F.softmax(logits, dim=-1)
                x = torch.multinomial(probs.view(-1, self.vocab_size), num_samples=1).view(n_samples, seq_length)
            else:
                # At t=0, take argmax
                x = logits.argmax(dim=-1)

        return x

    @torch.no_grad()
    def sample_ddim(
        self,
        n_samples: int,
        seq_length: int,
        n_steps: int = 50,
        temperature: float = 1.0,
        context: Tensor | None = None,
        device: torch.device | None = None,
    ) -> Tensor:
        """Fast sampling using DDIM-like strategy.

        Args:
            n_samples: Number of sequences
            seq_length: Sequence length
            n_steps: Number of sampling steps (< n_steps for faster)
            temperature: Sampling temperature
            context: Optional context
            device: Device

        Returns:
            Generated sequences
        """
        if device is None:
            device = next(self.parameters()).device

        # Subsample timesteps
        step_size = self.n_steps // n_steps
        timesteps = list(range(0, self.n_steps, step_size))[::-1]

        # Start from absorbing state
        x = torch.full(
            (n_samples, seq_length),
            self.scheduler.absorbing_state,
            dtype=torch.long,
            device=device,
        )

        for i, t in enumerate(timesteps):
            t_tensor = torch.full((n_samples,), t, dtype=torch.long, device=device)

            logits = self.denoiser(x, t_tensor, context)

            if temperature != 1.0:
                logits = logits / temperature

            if i < len(timesteps) - 1:
                # Sample progressively
                probs = F.softmax(logits, dim=-1)

                # Compute how many tokens to unmask
                current_masked = (x == self.scheduler.absorbing_state).float().mean()
                next_t = timesteps[i + 1]
                target_masked = self.scheduler.stay_probs_cumprod[next_t]

                # Unmask some tokens based on confidence
                confidence = probs.max(dim=-1).values
                unmask_prob = (current_masked - target_masked) / (current_masked + 1e-8)
                unmask_mask = torch.rand_like(confidence) < unmask_prob.clamp(0, 1)

                # Sample new tokens
                new_tokens = torch.multinomial(probs.view(-1, self.vocab_size), num_samples=1).view(
                    n_samples, seq_length
                )

                # Only unmask selected positions
                x = torch.where(
                    unmask_mask & (x == self.scheduler.absorbing_state),
                    new_tokens,
                    x,
                )
            else:
                x = logits.argmax(dim=-1)

        return x


class ConditionalCodonDiffusion(CodonDiffusion):
    """Conditional codon diffusion model.

    Can be conditioned on:
    - Protein structure
    - Amino acid sequence
    - Expression level targets
    - Other biological features

    Args:
        context_dim: Dimension of context embeddings
        **kwargs: Arguments passed to CodonDiffusion
    """

    def __init__(
        self,
        context_dim: int = 128,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.context_dim = context_dim

        # Context projection
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def forward(
        self,
        x: Tensor,
        t: Tensor | None = None,
        context: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Training forward pass with context."""
        # Project context if provided
        if context is not None:
            context = self.context_proj(context)
            if context.dim() == 2:
                context = context.unsqueeze(1)

        return super().forward(x, t, context)

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        seq_length: int,
        context: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        """Generate conditioned on context."""
        if context is not None:
            context = self.context_proj(context)
            if context.dim() == 2:
                context = context.unsqueeze(1)

        return super().sample(n_samples, seq_length, context=context, **kwargs)
