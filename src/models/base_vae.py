# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Base VAE abstraction for all VAE variants.

This module provides a unified base class that reduces code duplication
across the 19+ VAE variants in the codebase. All VAEs share common patterns:
- Reparameterization trick
- KL divergence computation
- Parameter counting
- Forward pass structure

By inheriting from BaseVAE, new VAE variants only need to implement
the encode() and decode() methods.

Usage:
    from src.models.base_vae import BaseVAE

    class MyCustomVAE(BaseVAE):
        def encode(self, x):
            # Custom encoder
            return mu, logvar

        def decode(self, z):
            # Custom decoder
            return reconstruction
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEOutput:
    """Standardized output container for all VAE models.

    Attributes:
        logits: Reconstruction logits (batch, seq_len, n_classes) or (batch, dim)
        mu: Latent mean (batch, latent_dim)
        logvar: Latent log variance (batch, latent_dim)
        z: Sampled latent vector (batch, latent_dim)
        z_hyp: Optional hyperbolic projection of z
        extras: Dictionary for model-specific additional outputs
    """

    logits: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor
    z: torch.Tensor
    z_hyp: torch.Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert to dictionary format for backward compatibility."""
        result = {
            "logits": self.logits,
            "mu": self.mu,
            "logvar": self.logvar,
            "z": self.z,
        }
        if self.z_hyp is not None:
            result["z_hyp"] = self.z_hyp
        result.update(self.extras)
        return result


@dataclass
class VAEConfig:
    """Configuration for BaseVAE and derived classes.

    Attributes:
        input_dim: Input dimension
        latent_dim: Latent space dimension
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout rate
        activation: Activation function name ('relu', 'gelu', 'silu')
        use_batch_norm: Whether to use batch normalization
        use_layer_norm: Whether to use layer normalization
        curvature: Hyperbolic curvature (for hyperbolic variants)
        beta_vae_weight: Beta weight for KL divergence
        output_classes: Number of output classes per position
    """

    input_dim: int = 9
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.0
    activation: str = "relu"
    use_batch_norm: bool = False
    use_layer_norm: bool = False
    curvature: float = 1.0
    beta_vae_weight: float = 1.0
    output_classes: int = 3  # Ternary classification


class BaseVAE(nn.Module, ABC):
    """Abstract base class for all VAE models.

    Provides standardized implementations for:
    - Reparameterization trick
    - KL divergence computation
    - Forward pass structure
    - Parameter counting
    - Reconstruction methods

    Subclasses must implement:
    - encode(x) -> (mu, logvar)
    - decode(z) -> reconstruction

    Example:
        class SimpleVAE(BaseVAE):
            def __init__(self, config):
                super().__init__(config)
                self.encoder = nn.Linear(config.input_dim, config.latent_dim * 2)
                self.decoder = nn.Linear(config.latent_dim, config.input_dim)

            def encode(self, x):
                h = self.encoder(x)
                mu, logvar = h.chunk(2, dim=-1)
                return mu, logvar

            def decode(self, z):
                return self.decoder(z)
    """

    def __init__(self, config: VAEConfig | None = None, **kwargs):
        """Initialize base VAE.

        Args:
            config: VAE configuration. If None, uses kwargs to build config.
            **kwargs: Override config parameters
        """
        super().__init__()

        if config is None:
            config = VAEConfig(**kwargs)
        else:
            # Allow kwargs to override config
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        self.config = config
        self.input_dim = config.input_dim
        self.latent_dim = config.latent_dim
        self.hidden_dims = config.hidden_dims
        self.dropout = config.dropout
        self.curvature = config.curvature
        self.beta = config.beta_vae_weight

        # Build activation function
        self.activation = self._get_activation(config.activation)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
        }
        return activations.get(name.lower(), nn.ReLU())

    @abstractmethod
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor (batch, input_dim) or (batch, seq_len, input_dim)

        Returns:
            Tuple of (mu, logvar) where:
            - mu: Mean of latent distribution (batch, latent_dim)
            - logvar: Log variance of latent distribution (batch, latent_dim)
        """
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction.

        Args:
            z: Latent vector (batch, latent_dim)

        Returns:
            Reconstruction tensor (shape depends on model)
        """
        pass

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for sampling from latent distribution.

        During training, samples z = mu + std * eps where eps ~ N(0, 1).
        During evaluation, returns mu directly (deterministic).

        Args:
            mu: Mean of latent distribution (batch, latent_dim)
            logvar: Log variance of latent distribution (batch, latent_dim)

        Returns:
            Sampled latent vector z (batch, latent_dim)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass through VAE.

        Args:
            x: Input tensor
            **kwargs: Additional arguments passed to encode/decode

        Returns:
            Dictionary with keys:
            - logits: Reconstruction logits
            - mu: Latent mean
            - logvar: Latent log variance
            - z: Sampled latent vector
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)

        return {
            "logits": logits,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute KL divergence from standard normal prior.

        KL(q(z|x) || p(z)) where:
        - q(z|x) = N(mu, exp(logvar))
        - p(z) = N(0, I)

        Args:
            mu: Latent mean (batch, latent_dim)
            logvar: Latent log variance (batch, latent_dim)
            reduction: 'mean', 'sum', or 'none'

        Returns:
            KL divergence (scalar if reduction='mean' or 'sum')
        """
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        if reduction == "mean":
            return kl.mean()
        elif reduction == "sum":
            return kl.sum()
        else:
            return kl

    def reconstruction_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_type: str = "cross_entropy",
    ) -> torch.Tensor:
        """Compute reconstruction loss.

        Args:
            logits: Model output logits
            targets: Target values
            loss_type: 'cross_entropy', 'mse', or 'bce'

        Returns:
            Reconstruction loss (scalar)
        """
        if loss_type == "cross_entropy":
            # Reshape for cross-entropy if needed
            if logits.dim() == 3:
                batch, seq_len, n_classes = logits.shape
                logits_flat = logits.view(-1, n_classes)
                # Convert targets from {-1, 0, 1} to {0, 1, 2}, clamp for safety
                targets_flat = (targets.view(-1) + 1).long().clamp(0, n_classes - 1)
                return F.cross_entropy(logits_flat, targets_flat)
            else:
                n_classes = logits.shape[-1]
                targets_clamped = targets.long().clamp(0, n_classes - 1)
                return F.cross_entropy(logits, targets_clamped)
        elif loss_type == "mse":
            return F.mse_loss(logits, targets)
        elif loss_type == "bce":
            return F.binary_cross_entropy_with_logits(logits, targets)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def compute_loss(
        self,
        x: torch.Tensor,
        outputs: dict[str, torch.Tensor] | None = None,
        beta: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute VAE loss (reconstruction + beta * KL).

        Args:
            x: Input tensor (also used as reconstruction target)
            outputs: Pre-computed model outputs (if None, runs forward pass)
            beta: KL weight (if None, uses self.beta)

        Returns:
            Dictionary with loss components:
            - total: Total loss
            - recon: Reconstruction loss
            - kl: KL divergence
        """
        if outputs is None:
            outputs = self.forward(x)

        if beta is None:
            beta = self.beta

        recon_loss = self.reconstruction_loss(outputs["logits"], x)
        kl_loss = self.kl_divergence(outputs["mu"], outputs["logvar"])

        total_loss = recon_loss + beta * kl_loss

        return {
            "total": total_loss,
            "recon": recon_loss,
            "kl": kl_loss,
        }

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent mean (deterministic encoding).

        Args:
            x: Input tensor

        Returns:
            Latent mean (batch, latent_dim)
        """
        mu, _ = self.encode(x)
        return mu

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input (for evaluation).

        Args:
            x: Input tensor

        Returns:
            Reconstructed values
        """
        outputs = self.forward(x)
        logits = outputs["logits"]

        if logits.dim() == 3:
            # Ternary classification: convert to {-1, 0, 1}
            classes = torch.argmax(logits, dim=-1)
            return classes.float() - 1.0
        else:
            return logits

    def sample(self, n_samples: int, device: torch.device | None = None) -> torch.Tensor:
        """Sample from prior and decode.

        Args:
            n_samples: Number of samples to generate
            device: Device to generate samples on

        Returns:
            Generated samples (n_samples, input_dim)
        """
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode(z)

    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        n_steps: int = 10,
    ) -> list[torch.Tensor]:
        """Interpolate between two inputs in latent space.

        Args:
            x1: First input
            x2: Second input
            n_steps: Number of interpolation steps

        Returns:
            List of decoded interpolations
        """
        z1 = self.encode_mean(x1)
        z2 = self.encode_mean(x2)

        interpolations = []
        for alpha in torch.linspace(0, 1, n_steps):
            z_interp = (1 - alpha) * z1 + alpha * z2
            decoded = self.decode(z_interp)
            interpolations.append(decoded)

        return interpolations

    def count_parameters(self) -> dict[str, int]:
        """Count model parameters.

        Returns:
            Dictionary with 'total' and 'trainable' parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable

        return {
            "total": total,
            "trainable": trainable,
            "frozen": frozen,
        }

    def get_latent_dim(self) -> int:
        """Get latent space dimension."""
        return self.latent_dim

    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder_params():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder_params():
            param.requires_grad = True

    def encoder_params(self):
        """Yield encoder parameters. Override in subclasses."""
        # Default: look for self.encoder
        if hasattr(self, "encoder"):
            return self.encoder.parameters()
        return iter([])

    def decoder_params(self):
        """Yield decoder parameters. Override in subclasses."""
        # Default: look for self.decoder
        if hasattr(self, "decoder"):
            return self.decoder.parameters()
        return iter([])


class HyperbolicBaseVAE(BaseVAE):
    """Base VAE with hyperbolic latent space projection.

    Extends BaseVAE with methods for projecting to/from hyperbolic space
    using the Poincare ball model. Useful for capturing hierarchical
    relationships in biological sequences.
    """

    def __init__(self, config: VAEConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

    def exp_map(self, v: torch.Tensor, c: float | None = None) -> torch.Tensor:
        """Exponential map from tangent space to Poincare ball.

        Args:
            v: Tangent vector (batch, dim)
            c: Curvature (if None, uses self.curvature)

        Returns:
            Point on Poincare ball (batch, dim)
        """
        if c is None:
            c = self.curvature

        sqrt_c = c**0.5
        v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-8)

        # Softer projection: tanh gives range [0, ~0.76]
        scale = torch.tanh(sqrt_c * v_norm / 2.0) / (sqrt_c * v_norm)
        return v * scale

    def log_map(self, y: torch.Tensor, c: float | None = None) -> torch.Tensor:
        """Logarithmic map from Poincare ball to tangent space.

        Args:
            y: Point on Poincare ball (batch, dim)
            c: Curvature (if None, uses self.curvature)

        Returns:
            Tangent vector (batch, dim)
        """
        if c is None:
            c = self.curvature

        sqrt_c = c**0.5
        y_norm = torch.clamp(torch.norm(y, dim=-1, keepdim=True), min=1e-8, max=1 - 1e-5)

        scale = 2.0 * torch.atanh(sqrt_c * y_norm) / (sqrt_c * y_norm)
        return y * scale

    def hyperbolic_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        c: float | None = None,
    ) -> torch.Tensor:
        """Compute hyperbolic distance between points.

        Args:
            x: First point (batch, dim)
            y: Second point (batch, dim)
            c: Curvature

        Returns:
            Hyperbolic distances (batch,)
        """
        if c is None:
            c = self.curvature

        sqrt_c = c**0.5

        # Mobius addition: -x + y
        diff = self._mobius_add(-x, y, c)
        diff_norm = torch.clamp(torch.norm(diff, dim=-1), min=1e-8, max=1 - 1e-5)

        return 2.0 / sqrt_c * torch.atanh(sqrt_c * diff_norm)

    def _mobius_add(self, x: torch.Tensor, y: torch.Tensor, c: float) -> torch.Tensor:
        """Mobius addition in Poincare ball."""
        x_sq = (x * x).sum(dim=-1, keepdim=True)
        y_sq = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)

        num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
        denom = 1 + 2 * c * xy + c**2 * x_sq * y_sq

        return num / torch.clamp(denom, min=1e-8)

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass with hyperbolic projection.

        Returns dict with additional 'z_hyp' key for hyperbolic latent.
        """
        mu, logvar = self.encode(x)
        z_euc = self.reparameterize(mu, logvar)
        z_hyp = self.exp_map(z_euc)
        logits = self.decode(z_euc)  # Decode from Euclidean

        return {
            "logits": logits,
            "mu": mu,
            "logvar": logvar,
            "z": z_euc,
            "z_euc": z_euc,
            "z_hyp": z_hyp,
        }


class ConditionalBaseVAE(BaseVAE):
    """Base VAE with conditioning support.

    Extends BaseVAE to support conditional generation, where the
    latent space is conditioned on additional information (e.g.,
    disease type, drug class, genotype).
    """

    def __init__(
        self,
        config: VAEConfig | None = None,
        condition_dim: int = 0,
        n_conditions: int = 0,
        **kwargs,
    ):
        """Initialize conditional VAE.

        Args:
            config: VAE configuration
            condition_dim: Dimension of condition embedding
            n_conditions: Number of discrete conditions (for embedding)
        """
        super().__init__(config, **kwargs)

        self.condition_dim = condition_dim
        self.n_conditions = n_conditions

        if n_conditions > 0:
            self.condition_embedding = nn.Embedding(n_conditions, condition_dim)
        else:
            self.condition_embedding = None

    def get_condition_embedding(
        self,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Get condition embedding.

        Args:
            condition: Condition indices (batch,) or continuous (batch, dim)

        Returns:
            Condition embedding (batch, condition_dim)
        """
        if self.condition_embedding is not None and condition.dim() == 1:
            return self.condition_embedding(condition.long())
        return condition

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with optional conditioning.

        Args:
            x: Input tensor
            condition: Optional condition tensor
        """
        if condition is not None:
            cond_emb = self.get_condition_embedding(condition)
            # Subclasses should override to use cond_emb appropriately
            kwargs["condition"] = cond_emb

        return super().forward(x, **kwargs)


__all__ = [
    "BaseVAE",
    "HyperbolicBaseVAE",
    "ConditionalBaseVAE",
    "VAEConfig",
    "VAEOutput",
]
