# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Holographic Decoder for VAE.

Replaces standard MLP decoder with AdS/CFT-inspired bulk-to-boundary
propagation. Key advantages:
1. Parameter efficiency: O(seq_len) vs O(seq_len × latent_dim)
2. Interpretability: Conformal dimension has physical meaning
3. Geometric consistency: Respects hyperbolic structure of latent space

The decoder models the sequence as living on the "boundary" of AdS space,
with the latent code as a "bulk" field that propagates outward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geometry import poincare_distance
from src.models.holographic.bulk_boundary import (
    BulkBoundaryPropagator,
    DecayType,
    GeodesicPropagator,
    PropagatorConfig,
)


@dataclass
class HolographicDecoderConfig:
    """Configuration for holographic decoder.

    Attributes:
        latent_dim: Latent space dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output vocabulary size (e.g., 21 for amino acids)
        max_seq_len: Maximum sequence length
        n_layers: Number of refinement layers
        decay_type: Type of radial decay function
        conformal_dim: Initial conformal dimension
        curvature: Hyperbolic curvature
        use_attention: Whether to use attention over boundary
        dropout: Dropout rate
    """

    latent_dim: int = 16
    hidden_dim: int = 64
    output_dim: int = 21  # Amino acids + gap
    max_seq_len: int = 512
    n_layers: int = 2
    decay_type: DecayType = DecayType.POWER_LAW
    conformal_dim: float = 1.0
    curvature: float = 1.0
    use_attention: bool = True
    dropout: float = 0.1


class HolographicDecoder(nn.Module):
    """Holographic decoder using bulk-to-boundary propagation.

    Architecture:
    1. Bulk field (latent code) propagates to boundary via geodesics
    2. Each boundary position receives signal weighted by geodesic distance
    3. Boundary operator transforms to output vocabulary
    4. Optional attention refinement for long-range dependencies

    The key insight is that in hyperbolic space:
    - Points near origin (bulk) represent general/ancestral features
    - Points near boundary represent specific/derived features
    - Geodesic distance controls information flow
    """

    def __init__(
        self,
        config: HolographicDecoderConfig | None = None,
    ):
        """Initialize holographic decoder.

        Args:
            config: Decoder configuration
        """
        super().__init__()
        self.config = config or HolographicDecoderConfig()

        # Configure propagator
        prop_config = PropagatorConfig(
            latent_dim=self.config.latent_dim,
            boundary_dim=self.config.hidden_dim,
            conformal_dim=self.config.conformal_dim,
            decay_type=self.config.decay_type,
            curvature=self.config.curvature,
            learnable_delta=True,
        )

        # Main bulk-to-boundary propagator
        self.propagator = BulkBoundaryPropagator(prop_config)

        # Refinement layers
        self.refinement_layers = nn.ModuleList()
        for _ in range(self.config.n_layers):
            self.refinement_layers.append(
                HolographicRefinementLayer(
                    self.config.hidden_dim,
                    self.config.use_attention,
                    self.config.dropout,
                )
            )

        # Output projection to vocabulary
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.output_dim),
        )

        # Learnable position embeddings for sequence structure
        self.position_embeddings = nn.Embedding(
            self.config.max_seq_len,
            self.config.hidden_dim,
        )

        # Geodesic operations for advanced features
        self.geodesic = GeodesicPropagator(self.config.curvature)

    def forward(
        self,
        z: torch.Tensor,
        seq_len: int | None = None,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Decode latent code to sequence logits.

        Args:
            z: Latent codes in Poincaré ball (batch, latent_dim)
            seq_len: Target sequence length (default: max_seq_len)
            return_intermediates: Whether to return intermediate representations

        Returns:
            logits: Output logits (batch, seq_len, output_dim)
            intermediates: Optional dict of intermediate values
        """
        z.size(0)
        device = z.device
        seq_len = seq_len or self.config.max_seq_len

        intermediates = {}

        # Step 1: Bulk-to-boundary propagation
        # This is the core holographic operation
        boundary_repr = self.propagator(z, n_boundary_points=seq_len)  # (batch, seq_len, hidden_dim)

        if return_intermediates:
            intermediates["after_propagation"] = boundary_repr.clone()
            intermediates["conformal_dim"] = self.propagator.get_conformal_dimension()

        # Step 2: Add position embeddings
        positions = torch.arange(seq_len, device=device)
        pos_emb = self.position_embeddings(positions)  # (seq_len, hidden_dim)
        boundary_repr = boundary_repr + pos_emb.unsqueeze(0)

        # Step 3: Refinement layers
        for i, layer in enumerate(self.refinement_layers):
            boundary_repr = layer(boundary_repr)
            if return_intermediates:
                intermediates[f"after_refinement_{i}"] = boundary_repr.clone()

        # Step 4: Project to output vocabulary
        logits = self.output_proj(boundary_repr)

        if return_intermediates:
            return logits, intermediates
        return logits

    def decode_with_radius_conditioning(
        self,
        z: torch.Tensor,
        target_radius: torch.Tensor | None = None,
        seq_len: int | None = None,
    ) -> torch.Tensor:
        """Decode with explicit radius conditioning.

        Allows controlling the "specificity" of generated sequences:
        - Small radius → more general/ancestral
        - Large radius → more specific/derived

        Args:
            z: Latent codes (batch, latent_dim)
            target_radius: Target radii (batch,) - if None, use z's radius
            seq_len: Target sequence length

        Returns:
            logits: Output logits
        """
        if target_radius is not None:
            # V5.12.2: Rescale z to target radius using hyperbolic distance
            origin = torch.zeros_like(z)
            current_radius = poincare_distance(z, origin, c=self.config.curvature).unsqueeze(-1)
            direction = z / current_radius.clamp(min=1e-8)
            z = direction * target_radius.unsqueeze(-1).clamp(max=self.config.curvature - 1e-3)

        return self.forward(z, seq_len)

    def compute_reconstruction_loss(
        self,
        z: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute reconstruction loss with optional masking.

        Args:
            z: Latent codes (batch, latent_dim)
            target: Target sequence indices (batch, seq_len)
            mask: Optional mask for valid positions (batch, seq_len)
            reduction: Loss reduction method

        Returns:
            Reconstruction loss
        """
        seq_len = target.size(1)
        logits = self.forward(z, seq_len=seq_len)

        # Compute cross-entropy
        logits_flat = logits.view(-1, self.config.output_dim)
        target_flat = target.view(-1)

        if mask is not None:
            mask_flat = mask.view(-1).bool()
            logits_flat = logits_flat[mask_flat]
            target_flat = target_flat[mask_flat]

        loss = F.cross_entropy(logits_flat, target_flat, reduction=reduction)

        return loss

    def get_parameter_efficiency(self) -> dict[str, int]:
        """Compare parameter count with standard MLP decoder.

        Returns:
            Dict with parameter counts and efficiency ratio
        """
        holographic_params = sum(p.numel() for p in self.parameters())

        # Estimate MLP decoder params: latent_dim → hidden × n_layers → output
        mlp_params = (
            self.config.latent_dim * self.config.hidden_dim  # First layer
            + self.config.hidden_dim * self.config.hidden_dim * self.config.n_layers
            + self.config.hidden_dim * self.config.output_dim * self.config.max_seq_len
        )

        return {
            "holographic_params": holographic_params,
            "mlp_params_estimate": mlp_params,
            "efficiency_ratio": mlp_params / holographic_params,
        }


class HolographicRefinementLayer(nn.Module):
    """Refinement layer for holographic decoder.

    Combines local convolution with optional global attention
    to refine the propagated boundary representation.
    """

    def __init__(
        self,
        hidden_dim: int,
        use_attention: bool = True,
        dropout: float = 0.1,
    ):
        """Initialize refinement layer.

        Args:
            hidden_dim: Hidden dimension
            use_attention: Whether to use self-attention
            dropout: Dropout rate
        """
        super().__init__()
        self.use_attention = use_attention

        # Local convolution for short-range dependencies
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Optional attention for long-range dependencies
        if use_attention:
            self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
            self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply refinement.

        Args:
            x: Input (batch, seq_len, hidden_dim)

        Returns:
            Refined representation
        """
        # Local convolution
        x_conv = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm1(x + x_conv)

        # Optional attention
        if self.use_attention:
            x_attn, _ = self.attention(x, x, x)
            x = self.norm2(x + x_attn)

        # Feed-forward
        x_ff = self.ff(x)
        x = self.norm3(x + x_ff)

        return x


class HierarchicalHolographicDecoder(nn.Module):
    """Hierarchical holographic decoder with multi-scale propagation.

    Uses multiple conformal dimensions to capture features at different
    scales:
    - Low Δ: Coarse-grained, global features
    - High Δ: Fine-grained, local features

    This mirrors the renormalization group in physics.
    """

    def __init__(
        self,
        config: HolographicDecoderConfig | None = None,
        n_scales: int = 3,
    ):
        """Initialize hierarchical decoder.

        Args:
            config: Base decoder configuration
            n_scales: Number of scales (conformal dimensions)
        """
        super().__init__()
        self.config = config or HolographicDecoderConfig()
        self.n_scales = n_scales

        # Create propagators at different scales
        self.propagators = nn.ModuleList()
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

        for i in range(n_scales):
            # Conformal dimension increases with scale
            delta = self.config.conformal_dim * (2**i)
            prop_config = PropagatorConfig(
                latent_dim=self.config.latent_dim,
                boundary_dim=self.config.hidden_dim,
                conformal_dim=delta,
                decay_type=self.config.decay_type,
                curvature=self.config.curvature,
            )
            self.propagators.append(BulkBoundaryPropagator(prop_config))

        # Scale combination network
        self.combine = nn.Sequential(
            nn.Linear(self.config.hidden_dim * n_scales, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.SiLU(),
        )

        # Output projection
        self.output_proj = nn.Linear(self.config.hidden_dim, self.config.output_dim)

    def forward(
        self,
        z: torch.Tensor,
        seq_len: int | None = None,
    ) -> torch.Tensor:
        """Decode with multi-scale propagation.

        Args:
            z: Latent codes (batch, latent_dim)
            seq_len: Target sequence length

        Returns:
            logits: Output logits (batch, seq_len, output_dim)
        """
        seq_len = seq_len or self.config.max_seq_len

        # Propagate at each scale
        scale_outputs = []
        weights = F.softmax(self.scale_weights, dim=0)

        for i, propagator in enumerate(self.propagators):
            output = propagator(z, n_boundary_points=seq_len)
            scale_outputs.append(output * weights[i])

        # Concatenate and combine scales
        multi_scale = torch.cat(scale_outputs, dim=-1)
        combined = self.combine(multi_scale)

        # Project to output
        logits = self.output_proj(combined)

        return logits

    def get_scale_weights(self) -> list[float]:
        """Get current scale weights."""
        return F.softmax(self.scale_weights, dim=0).tolist()


__all__ = [
    "HolographicDecoderConfig",
    "HolographicDecoder",
    "HolographicRefinementLayer",
    "HierarchicalHolographicDecoder",
]
