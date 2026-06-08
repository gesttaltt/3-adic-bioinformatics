# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Multimodal DDG VAE that fuses embeddings from specialist VAEs.

This module implements the multimodal fusion architecture that combines
latent representations from all three specialist VAEs:
- VAE-S669 (benchmark specialist)
- VAE-ProTherm (high-quality specialist)
- VAE-Wide (diversity specialist)

The fusion uses attention-based cross-modal alignment to learn
complementary information from each data regime.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.bioinformatics.models.ddg_vae import DDGVAE


@dataclass
class MultimodalConfig:
    """Configuration for multimodal DDG VAE."""

    # Modality dimensions (from specialist VAEs)
    s669_latent_dim: int = 16
    protherm_latent_dim: int = 32
    wide_latent_dim: int = 64

    # Fusion output dimension
    fused_dim: int = 128

    # Fusion type: "concat", "attention", "gated"
    fusion_type: str = "attention"

    # Attention parameters (if using attention fusion)
    n_heads: int = 8
    attention_dropout: float = 0.1

    # Decoder configuration
    decoder_hidden_dim: int = 256
    decoder_n_layers: int = 2
    decoder_dropout: float = 0.1

    # VAE parameters
    beta: float = 1.0  # KL weight
    logvar_min: float = -10.0
    logvar_max: float = 2.0

    @property
    def total_input_dim(self) -> int:
        """Total input dimension from all modalities."""
        return self.s669_latent_dim + self.protherm_latent_dim + self.wide_latent_dim


class CrossModalFusion(nn.Module):
    """Cross-modal fusion module for combining specialist embeddings.

    Supports three fusion strategies:
    1. concat: Simple concatenation + projection
    2. attention: Multi-head cross-attention
    3. gated: Gated fusion with learned weights
    """

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config

        if config.fusion_type == "concat":
            self._init_concat_fusion()
        elif config.fusion_type == "attention":
            self._init_attention_fusion()
        elif config.fusion_type == "gated":
            self._init_gated_fusion()
        else:
            raise ValueError(f"Unknown fusion type: {config.fusion_type}")

    def _init_concat_fusion(self):
        """Initialize concatenation-based fusion."""
        self.projection = nn.Sequential(
            nn.Linear(self.config.total_input_dim, self.config.fused_dim),
            nn.LayerNorm(self.config.fused_dim),
            nn.SiLU(),
            nn.Dropout(self.config.attention_dropout),
            nn.Linear(self.config.fused_dim, self.config.fused_dim),
        )

    def _init_attention_fusion(self):
        """Initialize attention-based fusion."""
        # Project each modality to common dimension
        self.proj_s669 = nn.Linear(self.config.s669_latent_dim, self.config.fused_dim)
        self.proj_protherm = nn.Linear(self.config.protherm_latent_dim, self.config.fused_dim)
        self.proj_wide = nn.Linear(self.config.wide_latent_dim, self.config.fused_dim)

        # Cross-attention between modalities
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.config.fused_dim,
            num_heads=self.config.n_heads,
            dropout=self.config.attention_dropout,
            batch_first=True,
        )

        # Final fusion layer
        self.fusion_proj = nn.Sequential(
            nn.LayerNorm(self.config.fused_dim),
            nn.Linear(self.config.fused_dim, self.config.fused_dim),
            nn.SiLU(),
        )

    def _init_gated_fusion(self):
        """Initialize gated fusion."""
        # Project each modality to common dimension
        self.proj_s669 = nn.Linear(self.config.s669_latent_dim, self.config.fused_dim)
        self.proj_protherm = nn.Linear(self.config.protherm_latent_dim, self.config.fused_dim)
        self.proj_wide = nn.Linear(self.config.wide_latent_dim, self.config.fused_dim)

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(self.config.fused_dim * 3, self.config.fused_dim),
            nn.LayerNorm(self.config.fused_dim),
            nn.SiLU(),
            nn.Linear(self.config.fused_dim, 3),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        z_s669: torch.Tensor,
        z_protherm: torch.Tensor,
        z_wide: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse embeddings from all three modalities.

        Args:
            z_s669: S669 latent (batch, s669_latent_dim)
            z_protherm: ProTherm latent (batch, protherm_latent_dim)
            z_wide: Wide latent (batch, wide_latent_dim)

        Returns:
            Fused embedding (batch, fused_dim)
        """
        if self.config.fusion_type == "concat":
            return self._forward_concat(z_s669, z_protherm, z_wide)
        elif self.config.fusion_type == "attention":
            return self._forward_attention(z_s669, z_protherm, z_wide)
        elif self.config.fusion_type == "gated":
            return self._forward_gated(z_s669, z_protherm, z_wide)

    def _forward_concat(
        self,
        z_s669: torch.Tensor,
        z_protherm: torch.Tensor,
        z_wide: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenation fusion."""
        z_concat = torch.cat([z_s669, z_protherm, z_wide], dim=-1)
        return self.projection(z_concat)

    def _forward_attention(
        self,
        z_s669: torch.Tensor,
        z_protherm: torch.Tensor,
        z_wide: torch.Tensor,
    ) -> torch.Tensor:
        """Attention-based fusion."""
        # Project to common dimension
        h_s669 = self.proj_s669(z_s669)
        h_protherm = self.proj_protherm(z_protherm)
        h_wide = self.proj_wide(z_wide)

        # Stack as sequence for attention
        # Shape: (batch, 3, fused_dim)
        h_stack = torch.stack([h_s669, h_protherm, h_wide], dim=1)

        # Cross-attention (each modality attends to all)
        h_attended, _ = self.cross_attention(h_stack, h_stack, h_stack)

        # Mean pool across modalities
        h_fused = h_attended.mean(dim=1)

        return self.fusion_proj(h_fused)

    def _forward_gated(
        self,
        z_s669: torch.Tensor,
        z_protherm: torch.Tensor,
        z_wide: torch.Tensor,
    ) -> torch.Tensor:
        """Gated fusion."""
        # Project to common dimension
        h_s669 = self.proj_s669(z_s669)
        h_protherm = self.proj_protherm(z_protherm)
        h_wide = self.proj_wide(z_wide)

        # Compute gates
        h_concat = torch.cat([h_s669, h_protherm, h_wide], dim=-1)
        gates = self.gate(h_concat)  # (batch, 3)

        # Weighted combination
        h_fused = gates[:, 0:1] * h_s669 + gates[:, 1:2] * h_protherm + gates[:, 2:3] * h_wide

        return h_fused


class MultimodalDecoder(nn.Module):
    """Decoder for multimodal DDG VAE."""

    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config

        layers = []
        in_dim = config.fused_dim

        for i in range(config.decoder_n_layers):
            out_dim = config.decoder_hidden_dim // (2**i)
            out_dim = max(out_dim, 32)

            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.SiLU(),
                    nn.Dropout(config.decoder_dropout),
                ]
            )
            in_dim = out_dim

        # Final prediction
        layers.append(nn.Linear(in_dim, 1))

        self.decoder = nn.Sequential(*layers)

    def forward(self, z_fused: torch.Tensor) -> torch.Tensor:
        """Decode fused embedding to DDG prediction."""
        return self.decoder(z_fused)


class MultimodalDDGVAE(nn.Module):
    """Multimodal VAE that fuses embeddings from specialist VAEs.

    This model:
    1. Takes pre-trained specialist VAE encoders (frozen)
    2. Fuses their latent representations via attention
    3. Learns a shared decoder for DDG prediction

    The multimodal approach combines:
    - S669: Benchmark-calibrated representations
    - ProTherm: High-quality curated representations
    - Wide: Diverse, generalizable representations
    """

    def __init__(
        self,
        vae_s669: DDGVAE,
        vae_protherm: DDGVAE,
        vae_wide: DDGVAE,
        config: MultimodalConfig | None = None,
        freeze_specialists: bool = True,
    ):
        """Initialize multimodal VAE.

        Args:
            vae_s669: Pre-trained S669 specialist VAE
            vae_protherm: Pre-trained ProTherm specialist VAE
            vae_wide: Pre-trained Wide specialist VAE
            config: MultimodalConfig
            freeze_specialists: Whether to freeze specialist encoders
        """
        super().__init__()

        if config is None:
            config = MultimodalConfig(
                s669_latent_dim=vae_s669.config.latent_dim,
                protherm_latent_dim=vae_protherm.config.latent_dim,
                wide_latent_dim=vae_wide.config.latent_dim,
            )

        self.config = config

        # Store specialist encoders
        self.encoder_s669 = vae_s669.encoder
        self.encoder_protherm = vae_protherm.encoder
        self.encoder_wide = vae_wide.encoder

        # Freeze if requested
        if freeze_specialists:
            self._freeze_encoder(self.encoder_s669)
            self._freeze_encoder(self.encoder_protherm)
            self._freeze_encoder(self.encoder_wide)

        # Fusion module
        self.fusion = CrossModalFusion(config)

        # Variational layers for fused representation
        self.fc_mu = nn.Linear(config.fused_dim, config.fused_dim)
        self.fc_logvar = nn.Linear(config.fused_dim, config.fused_dim)

        # Decoder
        self.decoder = MultimodalDecoder(config)

    def _freeze_encoder(self, encoder: nn.Module) -> None:
        """Freeze encoder parameters."""
        for param in encoder.parameters():
            param.requires_grad = False

    def encode_specialists(
        self,
        x_s669: torch.Tensor,
        x_protherm: torch.Tensor,
        x_wide: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get latent representations from all specialists.

        Args:
            x_s669: Input for S669 encoder
            x_protherm: Input for ProTherm encoder
            x_wide: Input for Wide encoder

        Returns:
            Tuple of (z_s669, z_protherm, z_wide) latent means
        """
        mu_s669, _ = self.encoder_s669(x_s669)
        mu_protherm, _ = self.encoder_protherm(x_protherm)
        mu_wide, _ = self.encoder_wide(x_wide)

        return mu_s669, mu_protherm, mu_wide

    def fuse_and_encode(
        self,
        z_s669: torch.Tensor,
        z_protherm: torch.Tensor,
        z_wide: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse specialist latents and encode to shared space.

        Args:
            z_s669: S669 latent
            z_protherm: ProTherm latent
            z_wide: Wide latent

        Returns:
            Tuple of (mu, logvar) for fused distribution
        """
        # Fuse
        h_fused = self.fusion(z_s669, z_protherm, z_wide)

        # Variational encoding
        mu = self.fc_mu(h_fused)
        logvar = self.fc_logvar(h_fused)
        logvar = logvar.clamp(self.config.logvar_min, self.config.logvar_max)

        return mu, logvar

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        training: bool = True,
    ) -> torch.Tensor:
        """Reparameterization trick."""
        if training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z_fused: torch.Tensor) -> torch.Tensor:
        """Decode fused latent to DDG prediction."""
        return self.decoder(z_fused)

    def forward(
        self,
        x_s669: torch.Tensor,
        x_protherm: torch.Tensor,
        x_wide: torch.Tensor,
        return_latent: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through multimodal VAE.

        Args:
            x_s669: Input for S669 encoder
            x_protherm: Input for ProTherm encoder
            x_wide: Input for Wide encoder
            return_latent: Whether to return latent representations

        Returns:
            Dictionary with predictions and latent info
        """
        # Encode with specialists
        z_s669, z_protherm, z_wide = self.encode_specialists(x_s669, x_protherm, x_wide)

        # Fuse and get distribution
        mu, logvar = self.fuse_and_encode(z_s669, z_protherm, z_wide)

        # Sample
        z_fused = self.reparameterize(mu, logvar, self.training)

        # Decode
        ddg_pred = self.decode(z_fused)

        result = {
            "ddg_pred": ddg_pred,
            "mu": mu,
            "logvar": logvar,
        }

        if return_latent:
            result.update(
                {
                    "z_fused": z_fused,
                    "z_s669": z_s669,
                    "z_protherm": z_protherm,
                    "z_wide": z_wide,
                }
            )

        return result

    def loss(
        self,
        x_s669: torch.Tensor,
        x_protherm: torch.Tensor,
        x_wide: torch.Tensor,
        y: torch.Tensor,
        reduction: str = "mean",
    ) -> dict[str, torch.Tensor]:
        """Compute multimodal VAE loss.

        Args:
            x_s669, x_protherm, x_wide: Inputs for each specialist
            y: Target DDG values
            reduction: Loss reduction method

        Returns:
            Dictionary with loss components
        """
        output = self.forward(x_s669, x_protherm, x_wide)

        ddg_pred = output["ddg_pred"]
        mu = output["mu"]
        logvar = output["logvar"]

        if y.dim() == 1:
            y = y.unsqueeze(-1)

        # Reconstruction loss
        recon_loss = F.mse_loss(ddg_pred, y, reduction=reduction)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        if reduction == "mean":
            kl_loss = kl_loss.mean()
        elif reduction == "sum":
            kl_loss = kl_loss.sum()

        total_loss = recon_loss + self.config.beta * kl_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def predict(
        self,
        x_s669: torch.Tensor,
        x_protherm: torch.Tensor,
        x_wide: torch.Tensor,
    ) -> torch.Tensor:
        """Make DDG predictions (no sampling).

        Args:
            x_s669, x_protherm, x_wide: Inputs for each specialist

        Returns:
            DDG predictions
        """
        self.eval()
        with torch.no_grad():
            z_s669, z_protherm, z_wide = self.encode_specialists(x_s669, x_protherm, x_wide)
            mu, _ = self.fuse_and_encode(z_s669, z_protherm, z_wide)
            return self.decode(mu)

    def get_fused_latent(
        self,
        x_s669: torch.Tensor,
        x_protherm: torch.Tensor,
        x_wide: torch.Tensor,
    ) -> torch.Tensor:
        """Get fused latent representation.

        Args:
            x_s669, x_protherm, x_wide: Inputs for each specialist

        Returns:
            Fused latent mean
        """
        self.eval()
        with torch.no_grad():
            z_s669, z_protherm, z_wide = self.encode_specialists(x_s669, x_protherm, x_wide)
            mu, _ = self.fuse_and_encode(z_s669, z_protherm, z_wide)
            return mu


__all__ = [
    "MultimodalConfig",
    "CrossModalFusion",
    "MultimodalDecoder",
    "MultimodalDDGVAE",
]
