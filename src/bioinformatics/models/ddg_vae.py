# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Base DDG VAE architecture.

Implements a Variational Autoencoder for DDG prediction with
configurable input dimensions, latent space size, and architecture.

Three specialist variants are trained:
- VAE-S669: Trained on S669 benchmark (N=669)
- VAE-ProTherm: Trained on curated ProTherm (N=2000+)
- VAE-Wide: Trained on ProteinGym + other sources (N=500K+)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DDGVAEConfig:
    """Configuration for DDG VAE."""

    # Architecture
    input_dim: int = 14  # Base physicochemical features
    hidden_dim: int = 64
    latent_dim: int = 16
    n_layers: int = 2

    # Activation and regularization
    activation: str = "silu"
    dropout: float = 0.1
    use_layer_norm: bool = True

    # VAE parameters
    logvar_min: float = -10.0
    logvar_max: float = 2.0
    beta: float = 1.0  # KL weight

    # Output
    output_dim: int = 1  # DDG prediction

    # Hyperbolic features (optional)
    use_hyperbolic: bool = False
    hyperbolic_dim: int = 4  # Extra dims when using hyperbolic features

    @property
    def effective_input_dim(self) -> int:
        """Get effective input dimension including hyperbolic features."""
        if self.use_hyperbolic:
            return self.input_dim + self.hyperbolic_dim
        return self.input_dim


class DDGEncoder(nn.Module):
    """Encoder for DDG VAE.

    Uses modern architecture with SiLU activation, LayerNorm,
    and Dropout for stable training.
    """

    def __init__(self, config: DDGVAEConfig):
        super().__init__()
        self.config = config

        # Build encoder layers
        layers = []
        in_dim = config.effective_input_dim

        for i in range(config.n_layers):
            out_dim = config.hidden_dim // (2**i) if i > 0 else config.hidden_dim
            out_dim = max(out_dim, config.latent_dim * 2)

            layers.append(nn.Linear(in_dim, out_dim))
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(self._get_activation())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))

            in_dim = out_dim

        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, config.latent_dim)

    def _get_activation(self) -> nn.Module:
        """Get activation function."""
        if self.config.activation == "silu":
            return nn.SiLU()
        elif self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "gelu":
            return nn.GELU()
        else:
            return nn.SiLU()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution.

        Args:
            x: Input features (batch, input_dim)

        Returns:
            Tuple of (mu, logvar) for the latent distribution
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Clamp logvar for numerical stability
        logvar = logvar.clamp(self.config.logvar_min, self.config.logvar_max)

        return mu, logvar


class DDGDecoder(nn.Module):
    """Decoder for DDG VAE.

    Maps latent representation to DDG prediction.
    """

    def __init__(self, config: DDGVAEConfig):
        super().__init__()
        self.config = config

        # Build decoder layers
        layers = []
        in_dim = config.latent_dim

        for i in range(config.n_layers):
            out_dim = config.hidden_dim // (2 ** (config.n_layers - 1 - i))
            out_dim = max(out_dim, config.latent_dim)

            layers.append(nn.Linear(in_dim, out_dim))
            if config.use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(self._get_activation())
            if config.dropout > 0 and i < config.n_layers - 1:
                layers.append(nn.Dropout(config.dropout))

            in_dim = out_dim

        # Final prediction layer (no activation)
        layers.append(nn.Linear(in_dim, config.output_dim))

        self.decoder = nn.Sequential(*layers)

    def _get_activation(self) -> nn.Module:
        """Get activation function."""
        if self.config.activation == "silu":
            return nn.SiLU()
        elif self.config.activation == "relu":
            return nn.ReLU()
        elif self.config.activation == "gelu":
            return nn.GELU()
        else:
            return nn.SiLU()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to DDG prediction.

        Args:
            z: Latent representation (batch, latent_dim)

        Returns:
            DDG prediction (batch, output_dim)
        """
        return self.decoder(z)


class DDGVAE(nn.Module):
    """Variational Autoencoder for DDG prediction.

    This is the base architecture used for all three specialist VAEs:
    - VAE-S669: input_dim=14, hidden_dim=64, latent_dim=16
    - VAE-ProTherm: input_dim=18 (with structural), hidden_dim=128, latent_dim=32
    - VAE-Wide: input_dim=14, hidden_dim=256, latent_dim=64

    The VAE learns a latent representation of mutation features that
    captures the relationship between physicochemical properties and
    protein stability effects.
    """

    def __init__(self, config: DDGVAEConfig | None = None, **kwargs):
        """Initialize DDG VAE.

        Args:
            config: DDGVAEConfig object
            **kwargs: Override config parameters
        """
        super().__init__()

        if config is None:
            config = DDGVAEConfig(**kwargs)
        else:
            # Apply any overrides
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        self.config = config

        # Build encoder and decoder
        self.encoder = DDGEncoder(config)
        self.decoder = DDGDecoder(config)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution.

        Args:
            x: Input features (batch, input_dim)

        Returns:
            Tuple of (mu, logvar)
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to DDG prediction.

        Args:
            z: Latent representation (batch, latent_dim)

        Returns:
            DDG prediction (batch, 1)
        """
        return self.decoder(z)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """Reparameterization trick for VAE.

        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            training: Whether to sample (True) or use mean (False)

        Returns:
            Latent sample z
        """
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(
        self,
        x: torch.Tensor,
        return_latent: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input features (batch, input_dim)
            return_latent: Whether to return latent representations

        Returns:
            Dictionary with 'ddg_pred', 'mu', 'logvar', and optionally 'z'
        """
        # Encode
        mu, logvar = self.encode(x)

        # Sample latent
        z = self.reparameterize(mu, logvar, self.training)

        # Decode
        ddg_pred = self.decode(z)

        result = {
            "ddg_pred": ddg_pred,
            "mu": mu,
            "logvar": logvar,
        }

        if return_latent:
            result["z"] = z

        return result

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        reduction: str = "mean",
    ) -> dict[str, torch.Tensor]:
        """Compute VAE loss.

        Loss = Reconstruction Loss + beta * KL Divergence

        Args:
            x: Input features (batch, input_dim)
            y: Target DDG values (batch, 1) or (batch,)
            reduction: Loss reduction method

        Returns:
            Dictionary with 'loss', 'recon_loss', 'kl_loss'
        """
        # Forward pass
        output = self.forward(x)
        ddg_pred = output["ddg_pred"]
        mu = output["mu"]
        logvar = output["logvar"]

        # Ensure y has correct shape
        if y.dim() == 1:
            y = y.unsqueeze(-1)

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(ddg_pred, y, reduction=reduction)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        if reduction == "mean":
            kl_loss = kl_loss.mean()
        elif reduction == "sum":
            kl_loss = kl_loss.sum()

        # Total loss
        total_loss = recon_loss + self.config.beta * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make DDG predictions (no sampling).

        Args:
            x: Input features (batch, input_dim)

        Returns:
            DDG predictions (batch, 1)
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
            return self.decode(mu)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation (mean).

        Args:
            x: Input features (batch, input_dim)

        Returns:
            Latent mean (batch, latent_dim)
        """
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
            return mu

    @classmethod
    def create_s669_variant(cls, use_hyperbolic: bool = True) -> DDGVAE:
        """Create VAE-S669 variant for benchmark dataset.

        This variant is optimized for the S669 benchmark:
        - Moderate capacity (hidden_dim=64)
        - Standard latent size (16-dim)
        - Input: 14 physicochemical + 4 hyperbolic features
        """
        config = DDGVAEConfig(
            input_dim=14,
            hidden_dim=64,
            latent_dim=16,
            n_layers=2,
            use_hyperbolic=use_hyperbolic,
            hyperbolic_dim=4,
            dropout=0.1,
        )
        return cls(config)

    @classmethod
    def create_protherm_variant(cls, use_hyperbolic: bool = True) -> DDGVAE:
        """Create VAE-ProTherm variant for curated high-quality data.

        This variant has larger capacity for cleaner data:
        - Larger capacity (hidden_dim=128)
        - Larger latent (32-dim)
        - Input: 20 features (14 base + 6 structural hints)
        """
        config = DDGVAEConfig(
            input_dim=20,  # Include structural features (ss, rsa, etc.)
            hidden_dim=128,
            latent_dim=32,
            n_layers=2,
            use_hyperbolic=use_hyperbolic,
            hyperbolic_dim=4,
            dropout=0.05,  # Less dropout for cleaner data
        )
        return cls(config)

    @classmethod
    def create_wide_variant(cls, use_hyperbolic: bool = True) -> DDGVAE:
        """Create VAE-Wide variant for large diverse dataset.

        This variant has maximum capacity for diversity:
        - Large capacity (hidden_dim=256)
        - Large latent (64-dim)
        - 3 layers for complex patterns
        """
        config = DDGVAEConfig(
            input_dim=14,
            hidden_dim=256,
            latent_dim=64,
            n_layers=3,
            use_hyperbolic=use_hyperbolic,
            hyperbolic_dim=4,
            dropout=0.15,  # More dropout for diverse data
        )
        return cls(config)


__all__ = [
    "DDGVAEConfig",
    "DDGEncoder",
    "DDGDecoder",
    "DDGVAE",
]
