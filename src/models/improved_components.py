# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Improved VAE components with modern architecture choices.

V5.12.3 improvements over FrozenEncoder/FrozenDecoder:
1. SiLU activation (smoother gradients, no dead neurons)
2. LayerNorm (stable training, better gradient flow)
3. Dropout (regularization)
4. Logvar clamping (numerical stability)

These components can load v5.5 checkpoint weights for the Linear layers,
with fresh initialization for the new LayerNorm layers.

Single responsibility: Improved encoder/decoder with modern architecture.
"""

from pathlib import Path

import torch
import torch.nn as nn


class ImprovedEncoder(nn.Module):
    """Improved encoder with SiLU, LayerNorm, and Dropout.

    Architecture improvements over FrozenEncoder:
    - SiLU activation: Smoother gradients, self-gated, no dead neurons
    - LayerNorm: Stabilizes training, enables higher learning rates
    - Dropout: Regularization to prevent overfitting
    - Logvar clamping: Prevents KL explosion/collapse

    Can load v5.5 checkpoint weights (Linear layers only).

    Attributes:
        input_dim: Input dimension (default 9 for ternary operations)
        latent_dim: Latent space dimension
        dropout: Dropout probability
        logvar_min: Minimum logvar value (prevents collapse)
        logvar_max: Maximum logvar value (prevents explosion)
    """

    def __init__(
        self,
        input_dim: int = 9,
        latent_dim: int = 16,
        dropout: float = 0.1,
        logvar_min: float = -10.0,
        logvar_max: float = 2.0,
    ):
        """Initialize improved encoder.

        Args:
            input_dim: Input dimension
            latent_dim: Latent space dimension
            dropout: Dropout probability (default 0.1)
            logvar_min: Minimum logvar clamp (default -10, var > 0.00005)
            logvar_max: Maximum logvar clamp (default 2, var < 7.4)
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        # Improved architecture with LayerNorm and SiLU
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Initialize LayerNorm to near-identity for compatibility
        self._init_layernorms()

    def _init_layernorms(self):
        """Initialize LayerNorm layers to near-identity.

        This ensures that when loading v5.5 weights, the model
        behaves similarly to the original initially.
        """
        for module in self.encoder.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with logvar clamping.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Tuple of (mu, logvar) tensors with logvar clamped
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Clamp logvar for numerical stability
        logvar = logvar.clamp(self.logvar_min, self.logvar_max)

        return mu, logvar

    def load_v5_5_weights(
        self,
        checkpoint_path: Path,
        encoder_prefix: str = "encoder_A",
        device: str = "cpu",
        strict: bool = False,
    ) -> "ImprovedEncoder":
        """Load Linear layer weights from v5.5 checkpoint.

        Only loads weights for Linear layers (encoder.0, encoder.2, encoder.4,
        fc_mu, fc_logvar). LayerNorm and Dropout are freshly initialized.

        Args:
            checkpoint_path: Path to v5.5 checkpoint
            encoder_prefix: Which encoder to load ('encoder_A' or 'encoder_B')
            device: Device to load to
            strict: If True, raise error on missing keys

        Returns:
            Self with loaded weights
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = checkpoint.get("model", checkpoint.get("model_state_dict", {}))

        # Map v5.5 keys to improved encoder keys
        # v5.5: encoder.0 (Linear), encoder.2 (Linear), encoder.4 (Linear)
        # improved: encoder.0 (Linear), encoder.4 (Linear), encoder.8 (Linear)
        key_mapping = {
            f"{encoder_prefix}.encoder.0.weight": "encoder.0.weight",
            f"{encoder_prefix}.encoder.0.bias": "encoder.0.bias",
            f"{encoder_prefix}.encoder.2.weight": "encoder.4.weight",  # Skip LayerNorm, SiLU, Dropout
            f"{encoder_prefix}.encoder.2.bias": "encoder.4.bias",
            f"{encoder_prefix}.encoder.4.weight": "encoder.8.weight",
            f"{encoder_prefix}.encoder.4.bias": "encoder.8.bias",
            f"{encoder_prefix}.fc_mu.weight": "fc_mu.weight",
            f"{encoder_prefix}.fc_mu.bias": "fc_mu.bias",
            f"{encoder_prefix}.fc_logvar.weight": "fc_logvar.weight",
            f"{encoder_prefix}.fc_logvar.bias": "fc_logvar.bias",
        }

        # Build state dict
        new_state = {}
        missing_keys = []
        for old_key, new_key in key_mapping.items():
            if old_key in model_state:
                new_state[new_key] = model_state[old_key]
            else:
                missing_keys.append(old_key)

        if missing_keys and strict:
            raise KeyError(f"Missing keys in checkpoint: {missing_keys}")

        # Load with strict=False to allow LayerNorm to keep initialization
        self.load_state_dict(new_state, strict=False)
        self.to(device)

        return self

    @classmethod
    def from_v5_5_checkpoint(
        cls,
        checkpoint_path: Path,
        encoder_prefix: str = "encoder_A",
        device: str = "cpu",
        dropout: float = 0.1,
    ) -> "ImprovedEncoder":
        """Create ImprovedEncoder and load v5.5 weights.

        Args:
            checkpoint_path: Path to v5.5 checkpoint
            encoder_prefix: Which encoder to load
            device: Device to load to
            dropout: Dropout probability

        Returns:
            ImprovedEncoder with v5.5 Linear weights loaded
        """
        encoder = cls(dropout=dropout)
        encoder.load_v5_5_weights(checkpoint_path, encoder_prefix, device)
        return encoder


class ImprovedDecoder(nn.Module):
    """Improved decoder with SiLU and LayerNorm.

    Architecture improvements over FrozenDecoder:
    - SiLU activation: Smoother gradients
    - LayerNorm: Stabilizes training

    Can load v5.5 checkpoint weights (Linear layers only).

    Attributes:
        latent_dim: Latent space dimension
        output_dim: Output dimension (default 9 for ternary operations)
    """

    def __init__(
        self,
        latent_dim: int = 16,
        output_dim: int = 9,
        dropout: float = 0.1,
    ):
        """Initialize improved decoder.

        Args:
            latent_dim: Latent space dimension
            output_dim: Output dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        # Improved architecture
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.LayerNorm(64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim * 3),
        )

        # Initialize LayerNorm to near-identity
        self._init_layernorms()

    def _init_layernorms(self):
        """Initialize LayerNorm layers to near-identity."""
        for module in self.decoder.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass - produces logits.

        Args:
            z: Latent tensor (batch, latent_dim)

        Returns:
            Logits tensor (batch, output_dim, 3)
        """
        logits = self.decoder(z)
        return logits.view(-1, self.output_dim, 3)

    def load_v5_5_weights(
        self,
        checkpoint_path: Path,
        decoder_prefix: str = "decoder_A",
        device: str = "cpu",
        strict: bool = False,
    ) -> "ImprovedDecoder":
        """Load Linear layer weights from v5.5 checkpoint.

        Args:
            checkpoint_path: Path to v5.5 checkpoint
            decoder_prefix: Which decoder to load
            device: Device to load to
            strict: If True, raise error on missing keys

        Returns:
            Self with loaded weights
        """
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_state = checkpoint.get("model", checkpoint.get("model_state_dict", {}))

        # Map v5.5 keys to improved decoder keys
        # v5.5: decoder.0 (Linear), decoder.2 (Linear), decoder.4 (Linear)
        # improved: decoder.0 (Linear), decoder.4 (Linear), decoder.8 (Linear)
        key_mapping = {
            f"{decoder_prefix}.decoder.0.weight": "decoder.0.weight",
            f"{decoder_prefix}.decoder.0.bias": "decoder.0.bias",
            f"{decoder_prefix}.decoder.2.weight": "decoder.4.weight",
            f"{decoder_prefix}.decoder.2.bias": "decoder.4.bias",
            f"{decoder_prefix}.decoder.4.weight": "decoder.8.weight",
            f"{decoder_prefix}.decoder.4.bias": "decoder.8.bias",
        }

        new_state = {}
        missing_keys = []
        for old_key, new_key in key_mapping.items():
            if old_key in model_state:
                new_state[new_key] = model_state[old_key]
            else:
                missing_keys.append(old_key)

        if missing_keys and strict:
            raise KeyError(f"Missing keys in checkpoint: {missing_keys}")

        self.load_state_dict(new_state, strict=False)
        self.to(device)

        return self

    @classmethod
    def from_v5_5_checkpoint(
        cls,
        checkpoint_path: Path,
        decoder_prefix: str = "decoder_A",
        device: str = "cpu",
        dropout: float = 0.1,
    ) -> "ImprovedDecoder":
        """Create ImprovedDecoder and load v5.5 weights.

        Args:
            checkpoint_path: Path to v5.5 checkpoint
            decoder_prefix: Which decoder to load
            device: Device to load to
            dropout: Dropout probability

        Returns:
            ImprovedDecoder with v5.5 Linear weights loaded
        """
        decoder = cls(dropout=dropout)
        decoder.load_v5_5_weights(checkpoint_path, decoder_prefix, device)
        return decoder


def create_encoder(
    encoder_type: str = "improved",
    checkpoint_path: Path | None = None,
    encoder_prefix: str = "encoder_A",
    device: str = "cpu",
    freeze: bool = False,
    **kwargs,
) -> nn.Module:
    """Factory function to create encoder.

    Args:
        encoder_type: "improved" or "frozen"
        checkpoint_path: Path to load weights from (optional)
        encoder_prefix: Which encoder to load
        device: Device to create on
        freeze: Whether to freeze parameters
        **kwargs: Additional arguments (dropout, etc.)

    Returns:
        Encoder module
    """
    from .frozen_components import FrozenEncoder

    if encoder_type == "frozen":
        if checkpoint_path:
            encoder = FrozenEncoder.from_v5_5_checkpoint(checkpoint_path, encoder_prefix, device)
        else:
            encoder = FrozenEncoder()
    elif encoder_type == "improved":
        dropout = kwargs.get("dropout", 0.1)
        encoder = ImprovedEncoder(dropout=dropout)
        if checkpoint_path:
            encoder.load_v5_5_weights(checkpoint_path, encoder_prefix, device)
        encoder.to(device)
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")

    if freeze:
        for param in encoder.parameters():
            param.requires_grad = False

    return encoder


def create_decoder(
    decoder_type: str = "improved",
    checkpoint_path: Path | None = None,
    decoder_prefix: str = "decoder_A",
    device: str = "cpu",
    freeze: bool = False,
    **kwargs,
) -> nn.Module:
    """Factory function to create decoder.

    Args:
        decoder_type: "improved" or "frozen"
        checkpoint_path: Path to load weights from (optional)
        decoder_prefix: Which decoder to load
        device: Device to create on
        freeze: Whether to freeze parameters
        **kwargs: Additional arguments (dropout, etc.)

    Returns:
        Decoder module
    """
    from .frozen_components import FrozenDecoder

    if decoder_type == "frozen":
        if checkpoint_path:
            decoder = FrozenDecoder.from_v5_5_checkpoint(checkpoint_path, decoder_prefix, device)
        else:
            decoder = FrozenDecoder()
    elif decoder_type == "improved":
        dropout = kwargs.get("dropout", 0.1)
        decoder = ImprovedDecoder(dropout=dropout)
        if checkpoint_path:
            decoder.load_v5_5_weights(checkpoint_path, decoder_prefix, device)
        decoder.to(device)
    else:
        raise ValueError(f"Unknown decoder_type: {decoder_type}")

    if freeze:
        for param in decoder.parameters():
            param.requires_grad = False

    return decoder


__all__ = [
    "ImprovedEncoder",
    "ImprovedDecoder",
    "create_encoder",
    "create_decoder",
]
