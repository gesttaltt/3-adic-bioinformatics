# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tropical Variational Autoencoder.

A VAE where the latent space uses tropical (max-plus) geometry instead
of Euclidean geometry. This provides:

1. Native tree learning: Tropical space is naturally tree-like
2. Piecewise linear latent space: Better for discrete structures
3. Ultrametric distances: Matches phylogenetic tree metrics

The key insight is that tropical linear operations produce piecewise
linear functions, and the space of tropical linear functions has
inherent tree-like structure.

Mathematical Background:
- Tropical projective space TP^n is a metric space
- Tropical convex sets are tree-like (ultrametric)
- Tropical geodesics are piecewise linear
- Tropical mean = tropical barycenter ≈ Steiner point
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.tropical.tropical_layers import (
    TropicalActivation,
    TropicalLayerNorm,
    TropicalLinear,
)


@dataclass
class TropicalVAEConfig:
    """Configuration for Tropical VAE.

    Attributes:
        input_dim: Input dimension (e.g., sequence length × vocab)
        latent_dim: Latent space dimension
        hidden_dim: Hidden layer dimension
        n_encoder_layers: Number of encoder layers
        n_decoder_layers: Number of decoder layers
        temperature: Temperature for soft tropical operations
        soft_tropical: Whether to use differentiable soft tropical
        beta: KL divergence weight
        vocab_size: Output vocabulary size
        max_seq_len: Maximum sequence length
        use_tropical_prior: Whether to use tropical prior (vs Gaussian)
    """

    input_dim: int = 512
    latent_dim: int = 16
    hidden_dim: int = 128
    n_encoder_layers: int = 3
    n_decoder_layers: int = 3
    temperature: float = 1.0
    soft_tropical: bool = True
    beta: float = 1.0
    vocab_size: int = 21
    max_seq_len: int = 100
    use_tropical_prior: bool = True


class TropicalEncoder(nn.Module):
    """Encoder using tropical layers.

    Maps input sequences to tropical latent space.
    """

    def __init__(self, config: TropicalVAEConfig):
        """Initialize tropical encoder.

        Args:
            config: VAE configuration
        """
        super().__init__()
        self.config = config

        # Input embedding
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Tropical encoder layers
        self.layers = nn.ModuleList()
        for i in range(config.n_encoder_layers):
            in_dim = config.hidden_dim if i == 0 else config.hidden_dim
            self.layers.append(
                nn.Sequential(
                    TropicalLinear(
                        in_dim,
                        config.hidden_dim,
                        temperature=config.temperature,
                        soft_tropical=config.soft_tropical,
                    ),
                    TropicalLayerNorm(config.hidden_dim),
                    TropicalActivation("relu"),
                )
            )

        # Pooling layer
        self.pool = TropicalPooling(config.hidden_dim)

        # Output to latent parameters
        # For tropical prior: single location parameter
        # For Gaussian prior: mean and log_var
        if config.use_tropical_prior:
            self.latent_proj = TropicalLinear(
                config.hidden_dim,
                config.latent_dim,
                temperature=config.temperature,
                soft_tropical=config.soft_tropical,
            )
            self.scale_proj = TropicalLinear(
                config.hidden_dim,
                config.latent_dim,
                temperature=config.temperature,
                soft_tropical=config.soft_tropical,
            )
        else:
            self.mu_proj = nn.Linear(config.hidden_dim, config.latent_dim)
            self.logvar_proj = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters.

        Args:
            x: Input sequence (batch, seq_len)
            return_hidden: Whether to return hidden states

        Returns:
            Tuple of (location, scale) for tropical prior
            or (mu, logvar) for Gaussian prior
            Optionally also hidden states
        """
        # Embed input
        h = self.embedding(x)  # (batch, seq_len, hidden_dim)

        # Apply tropical layers
        for layer in self.layers:
            h = layer(h)

        # Pool over sequence
        h_pooled = self.pool(h)  # (batch, hidden_dim)

        # Project to latent
        if self.config.use_tropical_prior:
            location = self.latent_proj(h_pooled)
            scale = F.softplus(self.scale_proj(h_pooled)) + 0.01
            if return_hidden:
                return location, scale, h
            return location, scale
        else:
            mu = self.mu_proj(h_pooled)
            logvar = self.logvar_proj(h_pooled)
            if return_hidden:
                return mu, logvar, h
            return mu, logvar


class TropicalDecoder(nn.Module):
    """Decoder using tropical layers.

    Maps tropical latent codes to output sequences.
    """

    def __init__(self, config: TropicalVAEConfig):
        """Initialize tropical decoder.

        Args:
            config: VAE configuration
        """
        super().__init__()
        self.config = config

        # Project latent to hidden
        self.latent_proj = TropicalLinear(
            config.latent_dim,
            config.hidden_dim,
            temperature=config.temperature,
            soft_tropical=config.soft_tropical,
        )

        # Learned position embeddings
        self.position_embed = nn.Embedding(config.max_seq_len, config.hidden_dim)

        # Tropical decoder layers
        self.layers = nn.ModuleList()
        for _ in range(config.n_decoder_layers):
            self.layers.append(
                nn.Sequential(
                    TropicalLinear(
                        config.hidden_dim,
                        config.hidden_dim,
                        temperature=config.temperature,
                        soft_tropical=config.soft_tropical,
                    ),
                    TropicalLayerNorm(config.hidden_dim),
                    TropicalActivation("relu"),
                )
            )

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.vocab_size)

    def forward(
        self,
        z: torch.Tensor,
        seq_len: int | None = None,
    ) -> torch.Tensor:
        """Decode latent to output logits.

        Args:
            z: Latent codes (batch, latent_dim)
            seq_len: Target sequence length

        Returns:
            Output logits (batch, seq_len, vocab_size)
        """
        z.size(0)
        seq_len = seq_len or self.config.max_seq_len

        # Project and expand latent
        h = self.latent_proj(z)  # (batch, hidden_dim)
        h = h.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq_len, hidden_dim)

        # Add position embeddings
        positions = torch.arange(seq_len, device=z.device)
        pos_emb = self.position_embed(positions)
        h = h + pos_emb.unsqueeze(0)

        # Apply tropical layers
        for layer in self.layers:
            h = layer(h)

        # Project to output
        logits = self.output_proj(h)

        return logits


class TropicalPooling(nn.Module):
    """Pooling for tropical features.

    Uses tropical max-pooling (which is just regular max).
    """

    def __init__(self, hidden_dim: int, method: str = "tropical_mean"):
        """Initialize pooling.

        Args:
            hidden_dim: Hidden dimension
            method: Pooling method ('max', 'tropical_mean', 'attention')
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.method = method

        if method == "attention":
            self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pooling.

        Args:
            x: Input (batch, seq_len, hidden_dim)

        Returns:
            Pooled output (batch, hidden_dim)
        """
        if self.method == "max":
            # Tropical sum = max
            return x.max(dim=1)[0]

        elif self.method == "tropical_mean":
            # Tropical mean approximation via logsumexp
            return torch.logsumexp(x, dim=1) - torch.log(torch.tensor(x.size(1), device=x.device))

        elif self.method == "attention":
            # Attention-weighted pooling
            weights = F.softmax(self.attention(x), dim=1)
            return (x * weights).sum(dim=1)

        else:
            raise ValueError(f"Unknown pooling method: {self.method}")


class TropicalVAE(nn.Module):
    """Tropical Variational Autoencoder.

    A VAE with tropical geometry in the latent space. Key features:
    1. Tropical encoder/decoder using max-plus algebra
    2. Tropical prior (Laplace-like) or Gaussian prior
    3. Tropical distance metrics for regularization
    4. Native tree structure in latent space

    The tropical latent space is well-suited for:
    - Phylogenetic data (sequences with tree relationships)
    - Hierarchical data (nested categories)
    - Discrete structures (graphs, trees)
    """

    def __init__(self, config: TropicalVAEConfig | None = None):
        """Initialize Tropical VAE.

        Args:
            config: VAE configuration
        """
        super().__init__()
        self.config = config or TropicalVAEConfig()

        # Encoder and decoder
        self.encoder = TropicalEncoder(self.config)
        self.decoder = TropicalDecoder(self.config)

        # Prior parameters (learnable for tropical prior)
        if self.config.use_tropical_prior:
            self.prior_location = nn.Parameter(torch.zeros(self.config.latent_dim))
            self.prior_scale = nn.Parameter(torch.ones(self.config.latent_dim))

    def encode(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input sequence (batch, seq_len)

        Returns:
            Tuple of (location, scale) or (mu, logvar)
        """
        return self.encoder(x)

    def decode(
        self,
        z: torch.Tensor,
        seq_len: int | None = None,
    ) -> torch.Tensor:
        """Decode latent to output logits.

        Args:
            z: Latent codes (batch, latent_dim)
            seq_len: Target sequence length

        Returns:
            Output logits (batch, seq_len, vocab_size)
        """
        return self.decoder(z, seq_len)

    def reparameterize(
        self,
        location: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick.

        For tropical prior: z = location + scale * noise
        where noise is sampled from Laplace or Gaussian.

        Args:
            location: Location parameter
            scale: Scale parameter

        Returns:
            Sampled latent codes
        """
        if self.config.use_tropical_prior:
            # Laplace noise (matches tropical geometry better)
            noise = torch.empty_like(location).uniform_(-0.5, 0.5)
            noise = -torch.sign(noise) * torch.log(1 - 2 * noise.abs() + 1e-8)
        else:
            # Gaussian noise
            noise = torch.randn_like(location)

        return location + scale * noise

    def forward(
        self,
        x: torch.Tensor,
        return_components: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input sequence (batch, seq_len)
            return_components: Whether to return all components

        Returns:
            Tuple of (logits, location, scale) or dict of components
        """
        # Encode
        location, scale = self.encode(x)

        # Reparameterize
        z = self.reparameterize(location, scale)

        # Decode
        logits = self.decode(z, x.size(1))

        if return_components:
            return {
                "logits": logits,
                "location": location,
                "scale": scale,
                "z": z,
            }
        return logits, location, scale

    def loss(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        location: torch.Tensor,
        scale: torch.Tensor,
        reduction: str = "mean",
    ) -> dict[str, torch.Tensor]:
        """Compute VAE loss.

        Args:
            x: Target sequence (batch, seq_len)
            logits: Predicted logits (batch, seq_len, vocab_size)
            location: Latent location
            scale: Latent scale
            reduction: Loss reduction

        Returns:
            Dict with total loss and components
        """
        # Reconstruction loss
        recon_loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            x.view(-1),
            reduction=reduction,
        )

        # KL divergence
        if self.config.use_tropical_prior:
            # Tropical KL: based on Laplace distributions
            kl_loss = self._tropical_kl(location, scale)
        else:
            # Gaussian KL
            kl_loss = -0.5 * torch.sum(
                1 + 2 * scale.log() - location.pow(2) - scale.pow(2),
                dim=-1,
            )

        if reduction == "mean":
            kl_loss = kl_loss.mean()

        # Total loss
        total_loss = recon_loss + self.config.beta * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def _tropical_kl(
        self,
        location: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Compute tropical KL divergence.

        Uses Laplace distribution as tropical analog of Gaussian.
        KL(Laplace(loc, scale) || Laplace(prior_loc, prior_scale))

        Args:
            location: Posterior location
            scale: Posterior scale

        Returns:
            KL divergence
        """
        prior_loc = self.prior_location
        prior_scale = self.prior_scale.clamp(min=0.01)

        # Laplace KL divergence
        diff = (location - prior_loc).abs()
        kl = torch.log(prior_scale / scale) + (scale + diff) / prior_scale - 1

        return kl.sum(dim=-1)

    def sample(
        self,
        n_samples: int = 1,
        seq_len: int | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample from the prior and decode.

        Args:
            n_samples: Number of samples
            seq_len: Sequence length
            temperature: Sampling temperature

        Returns:
            Sampled sequences (n_samples, seq_len)
        """
        device = self.prior_location.device

        if self.config.use_tropical_prior:
            # Sample from Laplace prior
            noise = torch.empty(n_samples, self.config.latent_dim, device=device).uniform_(-0.5, 0.5)
            noise = -torch.sign(noise) * torch.log(1 - 2 * noise.abs() + 1e-8)
            z = self.prior_location + self.prior_scale * noise * temperature
        else:
            # Sample from Gaussian prior
            z = torch.randn(n_samples, self.config.latent_dim, device=device) * temperature

        # Decode
        logits = self.decode(z, seq_len)

        # Sample from logits
        probs = F.softmax(logits / temperature, dim=-1)
        samples = torch.multinomial(probs.view(-1, self.config.vocab_size), 1)
        samples = samples.view(n_samples, -1)

        return samples

    def tropical_distance(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute tropical distance in latent space.

        Tropical distance: d(x, y) = max_i |x_i - y_i|

        Args:
            z1: First points (batch, latent_dim)
            z2: Second points (batch, latent_dim)

        Returns:
            Tropical distances (batch,)
        """
        return (z1 - z2).abs().max(dim=-1)[0]

    def tropical_geodesic(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        n_points: int = 10,
    ) -> torch.Tensor:
        """Compute tropical geodesic between points.

        In tropical space, geodesics are piecewise linear.

        Args:
            z1: Start point (latent_dim,)
            z2: End point (latent_dim,)
            n_points: Number of points on geodesic

        Returns:
            Points on geodesic (n_points, latent_dim)
        """
        t = torch.linspace(0, 1, n_points, device=z1.device).unsqueeze(-1)

        # Linear interpolation (geodesic in tropical space)
        geodesic = z1.unsqueeze(0) * (1 - t) + z2.unsqueeze(0) * t

        return geodesic


class TropicalVAELoss(nn.Module):
    """Loss function for Tropical VAE with additional regularization."""

    def __init__(
        self,
        beta: float = 1.0,
        tree_reg_weight: float = 0.1,
        ultrametric_weight: float = 0.1,
    ):
        """Initialize loss.

        Args:
            beta: KL weight
            tree_reg_weight: Tree structure regularization weight
            ultrametric_weight: Ultrametric constraint weight
        """
        super().__init__()
        self.beta = beta
        self.tree_reg_weight = tree_reg_weight
        self.ultrametric_weight = ultrametric_weight

    def forward(
        self,
        model: TropicalVAE,
        x: torch.Tensor,
        logits: torch.Tensor,
        location: torch.Tensor,
        scale: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute loss with regularization.

        Args:
            model: Tropical VAE model
            x: Target sequence
            logits: Predicted logits
            location: Latent location
            scale: Latent scale

        Returns:
            Loss dict
        """
        # Base loss
        losses = model.loss(x, logits, location, scale)

        # Tree structure regularization
        if self.tree_reg_weight > 0:
            tree_loss = self._tree_regularization(model, location)
            losses["tree_loss"] = tree_loss
            losses["loss"] = losses["loss"] + self.tree_reg_weight * tree_loss

        # Ultrametric constraint
        if self.ultrametric_weight > 0:
            ultra_loss = self._ultrametric_loss(model, location)
            losses["ultrametric_loss"] = ultra_loss
            losses["loss"] = losses["loss"] + self.ultrametric_weight * ultra_loss

        return losses

    def _tree_regularization(
        self,
        model: TropicalVAE,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Regularize for tree-like structure.

        Encourages pairwise distances to satisfy ultrametric inequality:
        d(x, z) ≤ max(d(x, y), d(y, z))

        Args:
            model: Tropical VAE
            z: Latent codes

        Returns:
            Tree regularization loss
        """
        batch_size = z.size(0)
        if batch_size < 3:
            return torch.tensor(0.0, device=z.device)

        # Sample triplets
        n_triplets = min(100, batch_size * (batch_size - 1) * (batch_size - 2) // 6)
        idx = torch.randperm(batch_size)[: n_triplets * 3].view(n_triplets, 3)

        loss = torch.tensor(0.0, device=z.device)
        for i in range(n_triplets):
            a, b, c = idx[i]
            d_ab = model.tropical_distance(z[a : a + 1], z[b : b + 1])
            d_bc = model.tropical_distance(z[b : b + 1], z[c : c + 1])
            d_ac = model.tropical_distance(z[a : a + 1], z[c : c + 1])

            # Ultrametric: d_ac ≤ max(d_ab, d_bc)
            violation = F.relu(d_ac - torch.maximum(d_ab, d_bc))
            loss = loss + violation

        return loss / n_triplets

    def _ultrametric_loss(
        self,
        model: TropicalVAE,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Explicit ultrametric constraint loss.

        Args:
            model: Tropical VAE
            z: Latent codes

        Returns:
            Ultrametric loss
        """
        # Similar to tree regularization but stricter
        return self._tree_regularization(model, z)


__all__ = [
    "TropicalVAEConfig",
    "TropicalEncoder",
    "TropicalDecoder",
    "TropicalPooling",
    "TropicalVAE",
    "TropicalVAELoss",
]
