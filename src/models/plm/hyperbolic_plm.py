# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Hyperbolic Protein Language Model encoder.

Combines pretrained protein language models with hyperbolic geometry
for hierarchical sequence representations.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.geometry import project_to_poincare
from src.models.plm.base import PLMEncoderBase
from src.models.plm.esm_encoder import ESM2Config, ESM2Encoder


class HyperbolicPLMEncoder(nn.Module):
    """PLM encoder with hyperbolic projection.

    Combines pretrained protein language model embeddings with
    hyperbolic geometry to capture hierarchical structure.

    Architecture:
        Sequence → PLM → Linear → LayerNorm → Hyperbolic Projection

    The hyperbolic projection uses the exponential map to project
    Euclidean embeddings onto the Poincaré ball, preserving
    hierarchical relationships.

    Example:
        >>> encoder = HyperbolicPLMEncoder(
        ...     plm_dim=1280,
        ...     hyperbolic_dim=64,
        ...     curvature=-1.0
        ... )
        >>> z = encoder(["MKWVTFISLLLLFSSAYS"])
        >>> print(z.shape)  # (1, 64)
        >>> print(z.norm(dim=-1) < 1)  # Inside Poincaré ball
    """

    def __init__(
        self,
        plm_dim: int = 1280,
        hyperbolic_dim: int = 64,
        hidden_dim: int = 512,
        curvature: float = -1.0,
        dropout: float = 0.1,
        plm_encoder: PLMEncoderBase | None = None,
        plm_config: ESM2Config | None = None,
        device: str = "cuda",
    ):
        """Initialize hyperbolic PLM encoder.

        Args:
            plm_dim: Dimension of PLM embeddings
            hyperbolic_dim: Dimension of hyperbolic output
            hidden_dim: Hidden layer dimension
            curvature: Poincaré ball curvature (negative)
            dropout: Dropout rate
            plm_encoder: Optional pretrained PLM encoder
            plm_config: Config for creating new PLM encoder
            device: Computation device
        """
        super().__init__()

        self.plm_dim = plm_dim
        self.hyperbolic_dim = hyperbolic_dim
        self.curvature = curvature
        self._device = device

        # PLM encoder (optional - can use external embeddings)
        if plm_encoder is not None:
            self.plm = plm_encoder
        elif plm_config is not None:
            self.plm = ESM2Encoder(plm_config, device=device)
        else:
            self.plm = None

        # Euclidean projection layers
        self.projection = nn.Sequential(
            nn.Linear(plm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hyperbolic_dim),
            nn.LayerNorm(hyperbolic_dim),
        )

        # Learnable curvature scaling
        self.curvature_scale = nn.Parameter(torch.tensor(1.0))

        self.to(device)

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    @property
    def output_dim(self) -> int:
        return self.hyperbolic_dim

    def forward(
        self,
        x: str | list[str] | torch.Tensor,
        return_euclidean: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode sequences to hyperbolic space.

        Args:
            x: Sequences (str/list) or precomputed PLM embeddings (tensor)
            return_euclidean: Also return Euclidean embeddings

        Returns:
            Hyperbolic embeddings on Poincaré ball
            Optionally also Euclidean embeddings
        """
        # Get PLM embeddings if needed
        if isinstance(x, (str, list)):
            if self.plm is None:
                raise ValueError("PLM encoder required for sequence input. Provide plm_encoder or plm_config.")
            plm_embeddings = self.plm.encode(x)
        else:
            plm_embeddings = x

        # Ensure on correct device
        if plm_embeddings.device != self.device:
            plm_embeddings = plm_embeddings.to(self.device)

        # Project to target dimension
        euclidean = self.projection(plm_embeddings)

        # Project to Poincaré ball using exponential map
        # Scale by learnable curvature factor
        c = abs(self.curvature) * self.curvature_scale
        hyperbolic = self._to_poincare(euclidean, c)

        if return_euclidean:
            return hyperbolic, euclidean

        return hyperbolic

    def _to_poincare(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Project Euclidean vectors to Poincaré ball.

        Uses the exponential map at the origin to project
        tangent vectors onto the manifold.

        Args:
            x: Euclidean vectors
            c: Curvature magnitude

        Returns:
            Points on Poincaré ball
        """
        # Exponential map at origin
        x_norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sqrt_c = c.sqrt()

        # exp_0(v) = tanh(sqrt(c) * ||v||) * v / (sqrt(c) * ||v||)
        hyperbolic = torch.tanh(sqrt_c * x_norm) * x / (sqrt_c * x_norm)

        # Project to ensure inside ball
        hyperbolic = project_to_poincare(hyperbolic, c=c.item(), eps=1e-5)

        return hyperbolic

    def encode_with_attention(
        self,
        sequences: str | list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode with attention weights for interpretability.

        Args:
            sequences: Input sequences

        Returns:
            Tuple of (hyperbolic_embeddings, attention_weights)
        """
        if self.plm is None:
            raise ValueError("PLM encoder required")

        plm_embeddings, attention = self.plm.encode(sequences, return_attention=True)
        hyperbolic = self.forward(plm_embeddings)

        return hyperbolic, attention

    def compute_pairwise_distances(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hyperbolic distances between embeddings.

        Uses the Poincaré distance formula:
            d(u, v) = arccosh(1 + 2 * ||u-v||^2 / ((1-||u||^2)(1-||v||^2)))

        Args:
            z1, z2: Hyperbolic embeddings

        Returns:
            Distance matrix
        """
        c = abs(self.curvature) * self.curvature_scale

        # Norms
        z1_norm_sq = z1.pow(2).sum(dim=-1, keepdim=True)
        z2_norm_sq = z2.pow(2).sum(dim=-1, keepdim=True)

        # Pairwise squared distances in Euclidean space
        diff = z1.unsqueeze(1) - z2.unsqueeze(0)
        diff_norm_sq = diff.pow(2).sum(dim=-1)

        # Poincaré distance formula
        numerator = 2 * diff_norm_sq
        denominator = (1 - z1_norm_sq) * (1 - z2_norm_sq.squeeze(-1).unsqueeze(0))
        denominator = denominator.clamp(min=1e-8)

        arg = 1 + numerator / denominator
        distance = torch.acosh(arg.clamp(min=1.0 + 1e-8)) / c.sqrt()

        return distance


class DualHyperbolicPLMEncoder(nn.Module):
    """Dual encoder for comparing two sequences in hyperbolic space.

    Useful for drug-target interaction, protein-protein interaction,
    and mutation effect prediction.
    """

    def __init__(
        self,
        plm_dim: int = 1280,
        hyperbolic_dim: int = 64,
        shared_encoder: bool = True,
        device: str = "cuda",
    ):
        """Initialize dual encoder.

        Args:
            plm_dim: PLM embedding dimension
            hyperbolic_dim: Output dimension
            shared_encoder: Share projection weights between encoders
            device: Computation device
        """
        super().__init__()

        self.encoder_a = HyperbolicPLMEncoder(
            plm_dim=plm_dim,
            hyperbolic_dim=hyperbolic_dim,
            device=device,
        )

        if shared_encoder:
            self.encoder_b = self.encoder_a
        else:
            self.encoder_b = HyperbolicPLMEncoder(
                plm_dim=plm_dim,
                hyperbolic_dim=hyperbolic_dim,
                device=device,
            )

        self.to(device)

    def forward(
        self,
        seq_a: str | list[str] | torch.Tensor,
        seq_b: str | list[str] | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode two sequences and compute their distance.

        Args:
            seq_a: First sequence(s)
            seq_b: Second sequence(s)

        Returns:
            Tuple of (embedding_a, embedding_b, distance)
        """
        z_a = self.encoder_a(seq_a)
        z_b = self.encoder_b(seq_b)

        # Compute hyperbolic distance
        distance = self.encoder_a.compute_pairwise_distances(z_a, z_b)

        return z_a, z_b, distance.diag()  # Return diagonal for paired inputs
