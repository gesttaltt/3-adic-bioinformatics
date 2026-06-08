# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Simple VAE for ablation studies.

This module provides a basic, fully-trainable VAE for isolated feature testing.
Unlike TernaryVAEV5_11 which has frozen components, this VAE trains end-to-end.

Usage:
    from src.models.simple_vae import SimpleVAE

    model = SimpleVAE(input_dim=9, latent_dim=16, hidden_dims=[64, 32])
    outputs = model(x)  # Returns dict with logits, mu, logvar, z
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleEncoder(nn.Module):
    """Simple MLP encoder for VAE."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class SimpleDecoder(nn.Module):
    """Simple MLP decoder for VAE.

    Outputs logits for 3-class classification per position (for ternary values).
    """

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        in_dim = latent_dim

        # Reverse hidden dims for decoder
        for h_dim in reversed(hidden_dims):
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.decoder = nn.Sequential(*layers)
        # Output 3 logits per position for ternary classification
        self.fc_out = nn.Linear(hidden_dims[0], output_dim * 3)
        self.output_dim = output_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder(z)
        logits = self.fc_out(h)
        # Reshape to (batch, output_dim, 3) for cross-entropy
        return logits.view(-1, self.output_dim, 3)


class SimpleVAE(nn.Module):
    """Simple, fully-trainable VAE for ablation studies.

    This is a basic VAE architecture without frozen components,
    designed for testing individual features in isolation.

    Args:
        input_dim: Input dimension (9 for ternary operations)
        latent_dim: Latent space dimension
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate (0.0 = no dropout)
    """

    def __init__(
        self,
        input_dim: int = 9,
        latent_dim: int = 16,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [64, 32]

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        self.encoder = SimpleEncoder(input_dim, latent_dim, hidden_dims, dropout)
        self.decoder = SimpleDecoder(latent_dim, input_dim, hidden_dims, dropout)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass.

        Args:
            x: Input tensor (batch, input_dim) with values in {-1, 0, 1}

        Returns:
            Dict with:
                - logits: (batch, input_dim, 3) reconstruction logits
                - mu: (batch, latent_dim) mean
                - logvar: (batch, latent_dim) log variance
                - z: (batch, latent_dim) sampled latent
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)

        return {
            "logits": logits,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input (for evaluation).

        Returns:
            Reconstructed values in {-1, 0, 1}
        """
        outputs = self.forward(x)
        logits = outputs["logits"]
        # Convert logits to class predictions, then to {-1, 0, 1}
        classes = torch.argmax(logits, dim=-1)
        return classes.float() - 1.0

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent space (mean only)."""
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to input space."""
        return self.decoder(z)

    def count_parameters(self) -> dict:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
        }


class SimpleVAEWithHyperbolic(SimpleVAE):
    """SimpleVAE with hyperbolic latent space projection.

    Projects the Euclidean latent space to hyperbolic space using
    exponential map projection. The hyperbolic projection is used for
    geometric losses (p-adic ranking), while Euclidean z is used for decoding.
    """

    def __init__(
        self,
        input_dim: int = 9,
        latent_dim: int = 16,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.0,
        curvature: float = 1.0,
    ):
        super().__init__(input_dim, latent_dim, hidden_dims, dropout)
        self.curvature = curvature

    def exp_map(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map from tangent space to hyperbolic space (Poincare ball).

        Uses a softer projection that doesn't saturate to boundary.
        """
        c = self.curvature
        sqrt_c = c**0.5
        v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-8)

        # Softer projection: scale norm to stay well within ball
        # tanh(x/2) gives range [0, 0.76] for typical latent norms
        scale = torch.tanh(sqrt_c * v_norm / 2.0) / (sqrt_c * v_norm)
        return v * scale

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass with hyperbolic projection.

        Uses Euclidean z for decoding, hyperbolic z_hyp for geometric losses.
        """
        # Get Euclidean forward pass (uses z_euc for decoding)
        mu, logvar = self.encoder(x)
        z_euc = self.reparameterize(mu, logvar)
        logits = self.decoder(z_euc)  # Decode from Euclidean z

        # Project to hyperbolic space for geometric losses
        z_hyp = self.exp_map(z_euc)

        return {
            "logits": logits,
            "mu": mu,
            "logvar": logvar,
            "z": z_euc,  # Use Euclidean for main latent (backward compat)
            "z_euc": z_euc,
            "z_hyp": z_hyp,  # Hyperbolic for geometric losses
        }


__all__ = [
    "SimpleVAE",
    "SimpleVAEWithHyperbolic",
    "SimpleEncoder",
    "SimpleDecoder",
]
