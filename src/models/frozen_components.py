# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Frozen VAE components for checkpoint preservation.

This module provides frozen encoder and decoder components that preserve
learned representations from previous training (v5.5 checkpoint).

Single responsibility: Frozen model components with checkpoint loading.
"""

from pathlib import Path

import torch
import torch.nn as nn


class FrozenEncoder(nn.Module):
    """Frozen encoder from v5.5 checkpoint.

    This encoder has achieved 100% coverage and NEVER trains.
    We only use it to produce Euclidean latent codes for projection.

    Attributes:
        input_dim: Input dimension (default 9 for ternary operations)
        latent_dim: Latent space dimension
    """

    def __init__(self, input_dim: int = 9, latent_dim: int = 16):
        """Initialize frozen encoder.

        Args:
            input_dim: Input dimension
            latent_dim: Latent space dimension
        """
        super().__init__()

        # Architecture must match v5.5 exactly
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # FREEZE all parameters
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass - deterministic, no gradients.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Tuple of (mu, logvar) tensors
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    @classmethod
    def from_v5_5_checkpoint(
        cls,
        checkpoint_path: Path,
        encoder_prefix: str = "encoder_A",
        device: str = "cpu",
    ) -> "FrozenEncoder":
        """Load frozen encoder from v5.5 checkpoint.

        Args:
            checkpoint_path: Path to v5.5 checkpoint
            encoder_prefix: Which encoder to load ('encoder_A' or 'encoder_B')
            device: Device to load to

        Returns:
            FrozenEncoder with loaded weights

        Security Note:
            weights_only=False is required for full checkpoint loading.
            Only load checkpoints from trusted sources.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = checkpoint["model"]

        # Create encoder
        encoder = cls()

        # Extract and load weights
        encoder_state = {}
        prefix = f"{encoder_prefix}."
        for key, value in model_state.items():
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
                encoder_state[new_key] = value

        encoder.load_state_dict(encoder_state)
        encoder.to(device)

        # Ensure frozen
        for param in encoder.parameters():
            param.requires_grad = False

        return encoder


class FrozenDecoder(nn.Module):
    """Frozen decoder from v5.5 checkpoint.

    Used for reconstruction verification (not training).

    Attributes:
        latent_dim: Latent space dimension
        output_dim: Output dimension (default 9 for ternary operations)
    """

    def __init__(self, latent_dim: int = 16, output_dim: int = 9):
        """Initialize frozen decoder.

        Args:
            latent_dim: Latent space dimension
            output_dim: Output dimension
        """
        super().__init__()
        self.output_dim = output_dim

        # Architecture must match v5.5 exactly (decoder_A style)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim * 3),
        )

        # FREEZE
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass - produces logits.

        Args:
            z: Latent tensor (batch, latent_dim)

        Returns:
            Logits tensor (batch, output_dim, 3)
        """
        logits = self.decoder(z)
        return logits.view(-1, self.output_dim, 3)

    @classmethod
    def from_v5_5_checkpoint(
        cls,
        checkpoint_path: Path,
        decoder_prefix: str = "decoder_A",
        device: str = "cpu",
    ) -> "FrozenDecoder":
        """Load frozen decoder from v5.5 checkpoint.

        Args:
            checkpoint_path: Path to v5.5 checkpoint
            decoder_prefix: Which decoder to load
            device: Device to load to

        Returns:
            FrozenDecoder with loaded weights

        Security Note:
            weights_only=False is required for full checkpoint loading.
            Only load checkpoints from trusted sources.
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = checkpoint["model"]

        decoder = cls()

        decoder_state = {}
        prefix = f"{decoder_prefix}."
        for key, value in model_state.items():
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
                decoder_state[new_key] = value

        decoder.load_state_dict(decoder_state)
        decoder.to(device)

        for param in decoder.parameters():
            param.requires_grad = False

        return decoder


__all__ = ["FrozenEncoder", "FrozenDecoder"]
