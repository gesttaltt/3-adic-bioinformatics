# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""MLP Refinement module for DDG predictions.

This module implements the MLP refinement layer that takes the fused
VAE latent representation and refines the DDG predictions through
a deep MLP with residual connections.

The refiner:
1. Takes frozen multimodal VAE latent representations
2. Learns residual corrections to the initial VAE prediction
3. Can be fine-tuned end-to-end with small learning rate
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RefinerConfig:
    """Configuration for DDG MLP Refiner."""

    # Input dimension (from multimodal VAE fused latent)
    latent_dim: int = 128

    # MLP architecture
    hidden_dims: list[int] = None  # Default: [256, 256, 128, 64]
    n_layers: int = 5

    # Regularization
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Residual learning
    use_residual: bool = True
    initial_residual_weight: float = 0.5  # Learnable

    # Activation
    activation: str = "silu"

    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256, 128, 64]


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        activation: str = "silu",
    ):
        super().__init__()

        self.use_projection = in_dim != out_dim

        # Main path
        layers = [
            nn.Linear(in_dim, out_dim),
        ]
        if use_layer_norm:
            layers.append(nn.LayerNorm(out_dim))

        if activation == "silu":
            layers.append(nn.SiLU())
        elif activation == "gelu":
            layers.append(nn.GELU())
        else:
            layers.append(nn.ReLU())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(out_dim, out_dim))

        self.main = nn.Sequential(*layers)

        # Projection for skip connection if dimensions differ
        if self.use_projection:
            self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x) if self.use_projection else x
        return residual + self.main(x)


class DDGMLPRefiner(nn.Module):
    """MLP Refiner for DDG predictions.

    Takes fused VAE latent representations and refines DDG predictions
    through a deep MLP with residual learning.

    Architecture:
    1. Initial VAE prediction (from frozen multimodal VAE)
    2. Deep MLP refinement stack
    3. Residual connection: final = vae_pred + residual_weight * mlp_delta
    4. End-to-end fine-tuning with small LR
    """

    def __init__(
        self,
        multimodal_vae: nn.Module | None = None,
        config: RefinerConfig | None = None,
        freeze_vae: bool = True,
    ):
        """Initialize MLP Refiner.

        Args:
            multimodal_vae: Pre-trained multimodal VAE (optional)
            config: RefinerConfig
            freeze_vae: Whether to freeze the VAE
        """
        super().__init__()

        if config is None:
            config = RefinerConfig()

        self.config = config
        self.multimodal_vae = multimodal_vae

        if multimodal_vae is not None and freeze_vae:
            for param in multimodal_vae.parameters():
                param.requires_grad = False

        # Build refinement MLP
        self.refiner = self._build_refiner()

        # Learnable residual weight
        if config.use_residual:
            self.residual_weight = nn.Parameter(torch.tensor(config.initial_residual_weight))

    def _build_refiner(self) -> nn.Module:
        """Build the refinement MLP."""
        layers = []
        in_dim = self.config.latent_dim

        for _i, out_dim in enumerate(self.config.hidden_dims):
            layers.append(
                ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=self.config.dropout,
                    use_layer_norm=self.config.use_layer_norm,
                    activation=self.config.activation,
                )
            )
            in_dim = out_dim

        # Final prediction layer
        layers.append(nn.Linear(in_dim, 1))

        return nn.Sequential(*layers)

    def forward(
        self,
        z_fused: torch.Tensor,
        vae_pred: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            z_fused: Fused latent from multimodal VAE (batch, latent_dim)
            vae_pred: Initial VAE prediction (batch, 1) - optional

        Returns:
            Dictionary with 'ddg_pred', 'mlp_pred', 'vae_pred', 'residual_weight'
        """
        # MLP refinement
        mlp_pred = self.refiner(z_fused)

        if self.config.use_residual and vae_pred is not None:
            # Residual learning: final = vae + weight * delta
            delta = mlp_pred - vae_pred
            weight = torch.sigmoid(self.residual_weight)  # Constrain to [0, 1]
            ddg_pred = vae_pred + weight * delta
        else:
            ddg_pred = mlp_pred
            weight = None

        return {
            "ddg_pred": ddg_pred,
            "mlp_pred": mlp_pred,
            "vae_pred": vae_pred,
            "residual_weight": weight,
        }

    def forward_from_inputs(
        self,
        x_s669: torch.Tensor,
        x_protherm: torch.Tensor,
        x_wide: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass from raw inputs (uses multimodal VAE).

        Args:
            x_s669, x_protherm, x_wide: Inputs for specialist VAEs

        Returns:
            Dictionary with refined predictions
        """
        if self.multimodal_vae is None:
            raise ValueError("Multimodal VAE not set. Use forward() with latent input.")

        # Get VAE predictions and latent
        vae_output = self.multimodal_vae(x_s669, x_protherm, x_wide, return_latent=True)

        z_fused = vae_output["z_fused"]
        vae_pred = vae_output["ddg_pred"]

        return self.forward(z_fused, vae_pred)

    def loss(
        self,
        z_fused: torch.Tensor,
        y: torch.Tensor,
        vae_pred: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> dict[str, torch.Tensor]:
        """Compute refiner loss.

        Args:
            z_fused: Fused latent from VAE
            y: Target DDG values
            vae_pred: Initial VAE prediction
            reduction: Loss reduction method

        Returns:
            Dictionary with loss components
        """
        output = self.forward(z_fused, vae_pred)

        if y.dim() == 1:
            y = y.unsqueeze(-1)

        # Main prediction loss
        pred_loss = F.mse_loss(output["ddg_pred"], y, reduction=reduction)

        # Optional: auxiliary loss on MLP prediction alone
        mlp_loss = F.mse_loss(output["mlp_pred"], y, reduction=reduction)

        return {
            "loss": pred_loss,
            "pred_loss": pred_loss,
            "mlp_loss": mlp_loss,
        }

    def predict(
        self,
        z_fused: torch.Tensor,
        vae_pred: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Make refined DDG predictions.

        Args:
            z_fused: Fused latent from VAE
            vae_pred: Initial VAE prediction

        Returns:
            Refined DDG predictions
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(z_fused, vae_pred)
            return output["ddg_pred"]

    def unfreeze_vae(self, lr_scale: float = 0.01) -> list[dict]:
        """Unfreeze VAE for end-to-end fine-tuning.

        Args:
            lr_scale: Learning rate scale for VAE parameters

        Returns:
            Parameter groups for optimizer
        """
        if self.multimodal_vae is None:
            return []

        for param in self.multimodal_vae.parameters():
            param.requires_grad = True

        return [
            {
                "params": self.refiner.parameters(),
                "lr_scale": 1.0,
            },
            {
                "params": self.multimodal_vae.parameters(),
                "lr_scale": lr_scale,
            },
        ]


__all__ = [
    "RefinerConfig",
    "ResidualBlock",
    "DDGMLPRefiner",
]
