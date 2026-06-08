# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Hierarchical Variational Autoencoder for Biological Sequences.

Implements a multi-level hierarchical VAE that captures features at different
resolutions - from local sequence motifs to global protein structure. This
architecture is particularly useful for:

- Multi-scale feature learning (local mutations, domain structure, global fold)
- Disentangled representations (different levels capture different features)
- Better generation quality through progressive refinement
- Improved resistance prediction by capturing both local and global effects

Architecture:
    Level L (top): Global/coarse features (e.g., protein family, fold)
    Level L-1: Intermediate features (e.g., domain structure)
    ...
    Level 1 (bottom): Local/fine features (e.g., specific mutations)

The model uses:
- Bottom-up inference: x -> z1 -> z2 -> ... -> zL
- Top-down generation: zL -> z(L-1) -> ... -> z1 -> x

Example:
    from src.models.hierarchical_vae import HierarchicalVAE, HierarchicalVAEConfig

    config = HierarchicalVAEConfig(
        input_dim=99,
        n_levels=3,
        latent_dims=[8, 16, 32],  # Bottom to top
        hidden_dims=[256, 128, 64],
    )

    model = HierarchicalVAE(config)
    outputs = model(x)  # Returns reconstructions and all level latents
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.models.base_vae import BaseVAE, VAEConfig


@dataclass
class HierarchicalVAEConfig(VAEConfig):
    """Configuration for Hierarchical VAE.

    Attributes:
        n_levels: Number of hierarchy levels (default: 3)
        latent_dims: Latent dimension at each level (bottom to top)
        hidden_dims_per_level: Hidden dims for encoder/decoder at each level
        use_skip_connections: Use residual/skip connections between levels
        use_attention: Use self-attention in encoder/decoder
        top_down_residual: Use residual connections in top-down path
        ladder_sigma_learn: Learn sigma at each ladder step
        free_bits: Minimum KL per dimension for each level (prevents posterior collapse)
        kl_weights: Weight for KL at each level
    """

    n_levels: int = 3
    latent_dims: list[int] = field(default_factory=lambda: [8, 16, 32])
    hidden_dims_per_level: list[int] = field(default_factory=lambda: [256, 128, 64])
    use_skip_connections: bool = True
    use_attention: bool = False
    top_down_residual: bool = True
    ladder_sigma_learn: bool = True
    free_bits: float = 0.25
    kl_weights: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])


class LadderEncoderBlock(nn.Module):
    """Encoder block for one level of the ladder network.

    Each block encodes features and produces latent distribution parameters.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Output mu and logvar
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Pass-through for next level
        self.to_next = nn.Linear(hidden_dim, hidden_dim // 2 if hidden_dim > 32 else hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input.

        Args:
            x: Input features (batch, input_dim)

        Returns:
            Tuple of (mu, logvar, features_for_next_level)
        """
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        next_features = self.to_next(h)

        return mu, logvar, next_features


class LadderDecoderBlock(nn.Module):
    """Decoder block for one level of the ladder network.

    Each block takes latent from current level and features from higher level,
    producing reconstruction features for lower level.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()

        self.use_residual = use_residual

        # Combine latent with top-down features
        self.combiner = nn.Linear(latent_dim + hidden_dim, hidden_dim)

        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        if use_residual and output_dim == hidden_dim:
            self.residual = nn.Identity()
        elif use_residual:
            self.residual = nn.Linear(hidden_dim, output_dim)
        else:
            self.residual = None

    def forward(
        self,
        z: torch.Tensor,
        top_down_features: torch.Tensor,
    ) -> torch.Tensor:
        """Decode from latent and top-down features.

        Args:
            z: Latent vector (batch, latent_dim)
            top_down_features: Features from higher level (batch, hidden_dim)

        Returns:
            Features for lower level (batch, output_dim)
        """
        combined = torch.cat([z, top_down_features], dim=-1)
        h = self.combiner(combined)
        out = self.decoder(h)

        if self.residual is not None:
            out = out + self.residual(top_down_features)

        return out


class TopDownPrior(nn.Module):
    """Top-down prior network for hierarchical VAE.

    Computes the prior distribution at each level conditioned on samples
    from higher levels. This allows the prior to be data-dependent.
    """

    def __init__(
        self,
        input_dim: int,  # From higher level
        latent_dim: int,  # This level
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute prior parameters.

        Args:
            features: Features from higher level (batch, input_dim)

        Returns:
            Tuple of (prior_mu, prior_logvar)
        """
        h = self.network(features)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar


class HierarchicalVAE(BaseVAE):
    """Hierarchical Variational Autoencoder with ladder structure.

    Implements LVAE/Ladder VAE architecture with:
    - Bottom-up deterministic encoder path
    - Top-down stochastic decoder path
    - Data-dependent learned priors at each level
    - KL-balancing with free bits to prevent posterior collapse
    """

    def __init__(self, config: HierarchicalVAEConfig | None = None, **kwargs):
        """Initialize Hierarchical VAE.

        Args:
            config: Model configuration
            **kwargs: Override config parameters
        """
        if config is None:
            config = HierarchicalVAEConfig(**kwargs)

        # Set latent_dim to top level for base class
        super().__init__(config, latent_dim=config.latent_dims[-1])

        self.n_levels = config.n_levels
        self.latent_dims = config.latent_dims
        self.hidden_dims = config.hidden_dims_per_level
        self.kl_weights = config.kl_weights
        self.free_bits = config.free_bits

        assert len(self.latent_dims) == self.n_levels
        assert len(self.hidden_dims) == self.n_levels

        # Build encoder blocks (bottom-up)
        self.encoder_blocks = nn.ModuleList()
        prev_dim = config.input_dim

        for level in range(self.n_levels):
            block = LadderEncoderBlock(
                input_dim=prev_dim,
                hidden_dim=self.hidden_dims[level],
                latent_dim=self.latent_dims[level],
                dropout=config.dropout,
            )
            self.encoder_blocks.append(block)
            # Next level input is features from this block
            prev_dim = self.hidden_dims[level] // 2 if self.hidden_dims[level] > 32 else self.hidden_dims[level]

        # Build decoder blocks (top-down)
        self.decoder_blocks = nn.ModuleList()

        for level in range(self.n_levels - 1, -1, -1):  # Top to bottom
            if level == self.n_levels - 1:
                # Top level: no top-down features, just latent
                block = nn.Sequential(
                    nn.Linear(self.latent_dims[level], self.hidden_dims[level]),
                    nn.GELU(),
                    nn.Linear(self.hidden_dims[level], self.hidden_dims[level]),
                )
            else:
                # Lower levels: combine latent with top-down
                block = LadderDecoderBlock(
                    latent_dim=self.latent_dims[level],
                    hidden_dim=self.hidden_dims[level + 1],  # From higher level
                    output_dim=self.hidden_dims[level],
                    dropout=config.dropout,
                    use_residual=config.top_down_residual,
                )
            self.decoder_blocks.append(block)

        # Build top-down priors for each level (except top)
        self.top_down_priors = nn.ModuleList()
        for level in range(self.n_levels - 1):
            prior = TopDownPrior(
                input_dim=self.hidden_dims[level + 1],
                latent_dim=self.latent_dims[level],
            )
            self.top_down_priors.append(prior)

        # Final reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.hidden_dims[0], self.hidden_dims[0]),
            nn.GELU(),
            nn.Linear(self.hidden_dims[0], config.input_dim * config.output_classes),
        )

        self.output_classes = config.output_classes

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to top-level latent (for base class compatibility).

        For hierarchical VAE, use encode_all() to get all levels.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Tuple of (top_mu, top_logvar)
        """
        all_mus, all_logvars, _ = self.encode_all(x)
        return all_mus[-1], all_logvars[-1]

    def encode_all(
        self,
        x: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Encode input through all hierarchy levels (bottom-up).

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Tuple of:
            - all_mus: List of mu at each level (bottom to top)
            - all_logvars: List of logvar at each level
            - all_features: List of deterministic features at each level
        """
        all_mus = []
        all_logvars = []
        all_features = []

        features = x
        for block in self.encoder_blocks:
            mu, logvar, next_features = block(features)
            all_mus.append(mu)
            all_logvars.append(logvar)
            all_features.append(next_features)
            features = next_features

        return all_mus, all_logvars, all_features

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from top-level latent (for base class compatibility).

        For hierarchical VAE, use decode_all() with all level latents.

        Args:
            z: Top-level latent (batch, top_latent_dim)

        Returns:
            Reconstruction logits (batch, input_dim, output_classes)
        """
        # Decode from top level with zeros for lower levels
        zs = [None] * (self.n_levels - 1) + [z]
        return self.decode_all(zs)

    def decode_all(self, zs: list[torch.Tensor]) -> torch.Tensor:
        """Decode from all hierarchy levels (top-down).

        Args:
            zs: List of latent vectors at each level (bottom to top)

        Returns:
            Reconstruction logits (batch, input_dim, output_classes)
        """
        # Start from top level
        top_z = zs[-1]
        features = self.decoder_blocks[0](top_z)  # Top block is first

        # Go through remaining levels (top-down, so reversed order)
        for i, (block, z) in enumerate(
            zip(self.decoder_blocks[1:], reversed(zs[:-1]), strict=False),
            start=1,
        ):
            if z is None:
                # Sample from prior if z not provided
                batch_size = features.size(0)
                level = self.n_levels - 1 - i
                z = torch.randn(batch_size, self.latent_dims[level], device=features.device)

            features = block(z, features)

        # Final reconstruction
        logits = self.reconstruction_head(features)

        # Reshape to (batch, input_dim, output_classes)
        batch_size = logits.size(0)
        input_dim = logits.size(-1) // self.output_classes
        logits = logits.view(batch_size, input_dim, self.output_classes)

        return logits

    def sample_from_prior(
        self,
        features: torch.Tensor,
        level: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample from top-down prior at a level.

        Args:
            features: Top-down features from higher level
            level: Hierarchy level (0 = bottom)

        Returns:
            Tuple of (z, prior_mu, prior_logvar)
        """
        prior_mu, prior_logvar = self.top_down_priors[level](features)

        # Sample
        std = torch.exp(0.5 * prior_logvar)
        eps = torch.randn_like(std)
        z = prior_mu + eps * std

        return z, prior_mu, prior_logvar

    def forward(self, x: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass through hierarchical VAE.

        Args:
            x: Input tensor (batch, input_dim)

        Returns:
            Dictionary with:
            - logits: Reconstruction logits
            - mu: Top-level mu (for base class compatibility)
            - logvar: Top-level logvar
            - z: Top-level z
            - all_mus: List of mu at each level
            - all_logvars: List of logvar at each level
            - all_zs: List of z at each level
            - prior_mus: List of prior mu at each level (except top)
            - prior_logvars: List of prior logvar at each level
        """
        # Bottom-up encoding
        all_mus, all_logvars, all_features = self.encode_all(x)

        # Sample latents at each level
        all_zs = []
        for mu, logvar in zip(all_mus, all_logvars, strict=False):
            z = self.reparameterize(mu, logvar)
            all_zs.append(z)

        # Top-down decoding with prior computation
        prior_mus = []
        prior_logvars = []

        # Start from top level
        top_z = all_zs[-1]
        features = self.decoder_blocks[0](top_z)

        # Go through remaining levels
        for i, (block, z) in enumerate(
            zip(self.decoder_blocks[1:], reversed(all_zs[:-1]), strict=False),
            start=1,
        ):
            # Compute prior from top-down features
            level = self.n_levels - 1 - i
            p_mu, p_logvar = self.top_down_priors[level](features)
            prior_mus.insert(0, p_mu)  # Insert at beginning (bottom to top order)
            prior_logvars.insert(0, p_logvar)

            features = block(z, features)

        # Final reconstruction
        logits = self.reconstruction_head(features)
        batch_size = logits.size(0)
        input_dim = logits.size(-1) // self.output_classes
        logits = logits.view(batch_size, input_dim, self.output_classes)

        return {
            "logits": logits,
            "mu": all_mus[-1],
            "logvar": all_logvars[-1],
            "z": all_zs[-1],
            "all_mus": all_mus,
            "all_logvars": all_logvars,
            "all_zs": all_zs,
            "prior_mus": prior_mus,
            "prior_logvars": prior_logvars,
        }

    def kl_divergence_hierarchical(
        self,
        all_mus: list[torch.Tensor],
        all_logvars: list[torch.Tensor],
        prior_mus: list[torch.Tensor],
        prior_logvars: list[torch.Tensor],
        free_bits: float | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Compute hierarchical KL divergence with learned priors.

        Uses KL between posterior q(z|x) and learned prior p(z|z_higher).
        Top level uses standard normal prior.

        Args:
            all_mus: Posterior means at each level
            all_logvars: Posterior log variances at each level
            prior_mus: Prior means at each level (except top)
            prior_logvars: Prior log variances at each level
            free_bits: Minimum KL per dimension (prevents collapse)

        Returns:
            Tuple of (total_kl, list of kl at each level)
        """
        if free_bits is None:
            free_bits = self.free_bits

        kl_per_level = []

        # Top level: standard normal prior
        top_mu, top_logvar = all_mus[-1], all_logvars[-1]
        top_kl = -0.5 * (1 + top_logvar - top_mu.pow(2) - top_logvar.exp())
        top_kl = torch.maximum(top_kl, torch.tensor(free_bits, device=top_kl.device))
        kl_per_level.append(top_kl.sum(dim=-1).mean())

        # Lower levels: learned prior
        for i in range(self.n_levels - 2, -1, -1):  # From second-top to bottom
            q_mu, q_logvar = all_mus[i], all_logvars[i]
            p_mu, p_logvar = prior_mus[i], prior_logvars[i]

            # KL between two Gaussians
            kl = 0.5 * (p_logvar - q_logvar + (q_logvar.exp() + (q_mu - p_mu).pow(2)) / (p_logvar.exp() + 1e-8) - 1)
            kl = torch.maximum(kl, torch.tensor(free_bits, device=kl.device))
            kl_per_level.insert(0, kl.sum(dim=-1).mean())

        # Weighted sum
        total_kl = sum(w * kl for w, kl in zip(self.kl_weights, kl_per_level, strict=False))

        return total_kl, kl_per_level

    def compute_loss(
        self,
        x: torch.Tensor,
        outputs: dict[str, torch.Tensor] | None = None,
        beta: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute hierarchical VAE loss.

        Args:
            x: Input tensor
            outputs: Pre-computed model outputs
            beta: KL weight

        Returns:
            Dictionary with loss components
        """
        if outputs is None:
            outputs = self.forward(x)

        if beta is None:
            beta = self.beta

        # Reconstruction loss
        recon_loss = self.reconstruction_loss(outputs["logits"], x)

        # Hierarchical KL
        total_kl, kl_per_level = self.kl_divergence_hierarchical(
            outputs["all_mus"],
            outputs["all_logvars"],
            outputs["prior_mus"],
            outputs["prior_logvars"],
        )

        total_loss = recon_loss + beta * total_kl

        result = {
            "total": total_loss,
            "recon": recon_loss,
            "kl": total_kl,
        }

        # Add per-level KL
        for i, kl in enumerate(kl_per_level):
            result[f"kl_level_{i}"] = kl

        return result

    def get_level_embeddings(
        self,
        x: torch.Tensor,
        level: int | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Get embeddings at specified level(s).

        Args:
            x: Input tensor
            level: Level to get (None = all levels)

        Returns:
            Latent embeddings at specified level(s)
        """
        all_mus, _, _ = self.encode_all(x)

        if level is not None:
            return all_mus[level]
        return all_mus

    def sample(
        self,
        n_samples: int,
        device: torch.device | None = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample from the hierarchical prior.

        Args:
            n_samples: Number of samples
            device: Device to use
            temperature: Sampling temperature

        Returns:
            Generated samples
        """
        if device is None:
            device = next(self.parameters()).device

        # Sample top level from standard normal
        top_z = temperature * torch.randn(n_samples, self.latent_dims[-1], device=device)

        # Top-down generation
        features = self.decoder_blocks[0](top_z)
        all_zs = [None] * (self.n_levels - 1) + [top_z]

        # Sample lower levels from learned priors
        for i in range(self.n_levels - 2, -1, -1):
            z, _, _ = self.sample_from_prior(features, i)
            z = temperature * z
            all_zs[i] = z

            if i > 0:
                block_idx = self.n_levels - 1 - i
                features = self.decoder_blocks[block_idx](z, features)

        return self.decode_all(all_zs)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct input.

        Args:
            x: Input tensor

        Returns:
            Reconstructed values
        """
        outputs = self.forward(x)
        logits = outputs["logits"]

        # Ternary classification: convert to {-1, 0, 1}
        classes = torch.argmax(logits, dim=-1)
        return classes.float() - 1.0

    def count_parameters(self) -> dict[str, int]:
        """Count parameters with level breakdown."""
        result = super().count_parameters()

        # Add per-level counts
        for i, block in enumerate(self.encoder_blocks):
            result[f"encoder_level_{i}"] = sum(p.numel() for p in block.parameters())

        for i, block in enumerate(self.decoder_blocks):
            result[f"decoder_level_{i}"] = sum(p.numel() for p in block.parameters())

        return result


__all__ = [
    "HierarchicalVAE",
    "HierarchicalVAEConfig",
    "LadderEncoderBlock",
    "LadderDecoderBlock",
    "TopDownPrior",
]
