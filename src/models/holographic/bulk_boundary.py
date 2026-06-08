# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Bulk-to-Boundary Propagation for Holographic Decoding.

Implements the core physics of AdS/CFT correspondence:
- Bulk fields (latent codes) propagate to boundary (sequence)
- Signal strength decays with geodesic distance
- Conformal dimension controls decay rate

Mathematical foundation:
  In AdS/CFT, boundary correlators are computed via:
    <O(x)O(y)> ~ 1/|x-y|^{2Δ}

  where Δ is the conformal dimension. We adapt this:
    signal(bulk_point, boundary_position) ~ K(d_geodesic) * transform(bulk)

  where K is the bulk-to-boundary propagator kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecayType(Enum):
    """Types of radial decay functions."""

    POWER_LAW = "power_law"  # 1/r^{2Δ} - AdS/CFT standard
    EXPONENTIAL = "exponential"  # exp(-αr) - faster decay
    GAUSSIAN = "gaussian"  # exp(-r²/2σ²) - smooth decay
    LEARNABLE = "learnable"  # Fully learnable decay profile


@dataclass
class PropagatorConfig:
    """Configuration for bulk-to-boundary propagator.

    Attributes:
        latent_dim: Dimension of bulk (latent) space
        boundary_dim: Dimension of boundary operators
        conformal_dim: Conformal dimension Δ (controls power-law decay)
        decay_type: Type of radial decay
        curvature: Hyperbolic curvature parameter
        max_radius: Maximum radius in Poincaré ball
        n_harmonics: Number of angular harmonics for expansion
        learnable_delta: Whether conformal dimension is learnable
    """

    latent_dim: int = 16
    boundary_dim: int = 64
    conformal_dim: float = 1.0
    decay_type: DecayType = DecayType.POWER_LAW
    curvature: float = 1.0
    max_radius: float = 0.95
    n_harmonics: int = 8
    learnable_delta: bool = True


class RadialDecayFunction(nn.Module):
    """Radial decay function for bulk-to-boundary propagation.

    Implements K(r) where r is geodesic distance from bulk point to boundary.
    """

    def __init__(
        self,
        decay_type: DecayType = DecayType.POWER_LAW,
        conformal_dim: float = 1.0,
        learnable: bool = True,
    ):
        """Initialize radial decay function.

        Args:
            decay_type: Type of decay function
            conformal_dim: Initial conformal dimension
            learnable: Whether parameters are learnable
        """
        super().__init__()
        self.decay_type = decay_type

        if learnable:
            self.delta = nn.Parameter(torch.tensor(conformal_dim))
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer("delta", torch.tensor(conformal_dim))
            self.register_buffer("scale", torch.ones(1))

        if decay_type == DecayType.LEARNABLE:
            # Learnable MLP for arbitrary decay profile
            self.decay_net = nn.Sequential(
                nn.Linear(1, 32),
                nn.SiLU(),
                nn.Linear(32, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
                nn.Softplus(),  # Ensure positive
            )

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Compute decay factor for given geodesic distance.

        Args:
            distance: Geodesic distance (batch,) or (batch, seq_len)

        Returns:
            Decay factor K(distance)
        """
        # Ensure positive distance with small epsilon
        d = distance.clamp(min=1e-6)

        if self.decay_type == DecayType.POWER_LAW:
            # K(d) = scale / d^{2Δ}
            # Add 1 to denominator for numerical stability at origin
            return self.scale / (1 + d).pow(2 * self.delta.abs())

        elif self.decay_type == DecayType.EXPONENTIAL:
            # K(d) = scale * exp(-Δ * d)
            return self.scale * torch.exp(-self.delta.abs() * d)

        elif self.decay_type == DecayType.GAUSSIAN:
            # K(d) = scale * exp(-d²/(2Δ²))
            sigma = self.delta.abs() + 0.1
            return self.scale * torch.exp(-d.pow(2) / (2 * sigma.pow(2)))

        elif self.decay_type == DecayType.LEARNABLE:
            # Fully learnable profile
            d_input = d.unsqueeze(-1) if d.dim() == 1 else d.unsqueeze(-1)
            return self.decay_net(d_input).squeeze(-1)

        raise ValueError(f"Unknown decay type: {self.decay_type}")

    def get_conformal_dim(self) -> float:
        """Get current conformal dimension."""
        return self.delta.abs().item()


class GeodesicPropagator(nn.Module):
    """Compute geodesic distances in Poincaré ball.

    Used to determine how bulk field values propagate to boundary.
    """

    def __init__(self, curvature: float = 1.0):
        """Initialize geodesic propagator.

        Args:
            curvature: Hyperbolic curvature (c > 0)
        """
        super().__init__()
        self.register_buffer("curvature", torch.tensor(curvature))

    def poincare_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Poincaré ball distance.

        Args:
            x: Points in Poincaré ball (batch, dim)
            y: Points in Poincaré ball (batch, dim) or (batch, n_points, dim)

        Returns:
            Geodesic distances
        """
        c = self.curvature

        # Handle broadcasting for batch of boundary points
        if y.dim() == 3 and x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, dim)

        diff = x - y
        norm_x_sq = (x * x).sum(dim=-1)
        norm_y_sq = (y * y).sum(dim=-1)
        norm_diff_sq = (diff * diff).sum(dim=-1)

        numerator = 2 * c * norm_diff_sq
        denominator = (1 - c * norm_x_sq) * (1 - c * norm_y_sq)

        # Clamp for numerical stability
        ratio = 1 + numerator / (denominator.clamp(min=1e-8))
        ratio = ratio.clamp(min=1.0 + 1e-8)

        return (1 / c.sqrt()) * torch.acosh(ratio)

    def distance_to_origin(self, x: torch.Tensor) -> torch.Tensor:
        """Compute distance from points to origin.

        Args:
            x: Points in Poincaré ball (batch, dim)

        Returns:
            Distances to origin
        """
        c = self.curvature
        norm_x = torch.norm(x, dim=-1)

        # d(x, 0) = (2/√c) * arctanh(√c * ||x||)
        sqrt_c_norm = (c.sqrt() * norm_x).clamp(max=1 - 1e-6)
        return (2 / c.sqrt()) * torch.atanh(sqrt_c_norm)

    def geodesic_midpoint(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute geodesic midpoint (for ancestral reconstruction).

        Args:
            x: First point (batch, dim)
            y: Second point (batch, dim)

        Returns:
            Midpoint on geodesic
        """
        return self.geodesic_interpolation(x, y, t=0.5)

    def geodesic_interpolation(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: float = 0.5,
    ) -> torch.Tensor:
        """Interpolate along geodesic from x to y.

        Args:
            x: Start point (batch, dim)
            y: End point (batch, dim)
            t: Interpolation parameter (0=x, 1=y)

        Returns:
            Point at position t along geodesic
        """
        c = self.curvature

        # Möbius operations for geodesic
        # γ(t) = x ⊕ (t ⊗ (-x ⊕ y))
        neg_x = self._mobius_neg(x, c)
        v = self._mobius_add(neg_x, y, c)
        tv = self._mobius_scalar(t, v, c)
        result = self._mobius_add(x, tv, c)

        return result

    def _mobius_add(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Möbius addition in Poincaré ball."""
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)

        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        denom = 1 + 2 * c * xy + c.pow(2) * x2 * y2

        return num / denom.clamp(min=1e-8)

    def _mobius_neg(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Möbius negation (inverse under Möbius addition)."""
        return -x

    def _mobius_scalar(
        self,
        t: float,
        x: torch.Tensor,
        c: torch.Tensor,
    ) -> torch.Tensor:
        """Möbius scalar multiplication."""
        norm_x = torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8)
        sqrt_c = c.sqrt()

        # t ⊗ x = (1/√c) * tanh(t * arctanh(√c * ||x||)) * (x / ||x||)
        sqrt_c_norm = (sqrt_c * norm_x).clamp(max=1 - 1e-6)
        factor = torch.tanh(t * torch.atanh(sqrt_c_norm)) / (sqrt_c * norm_x)

        return factor * x


class BulkBoundaryPropagator(nn.Module):
    """Main bulk-to-boundary propagator for holographic decoding.

    Combines:
    1. Geodesic distance computation
    2. Radial decay function
    3. Boundary operator transformation

    The output at each boundary position is computed as:
      output[i] = sum_j K(d(bulk, boundary[j])) * W[j] * bulk
    """

    def __init__(self, config: PropagatorConfig | None = None):
        """Initialize bulk-to-boundary propagator.

        Args:
            config: Propagator configuration
        """
        super().__init__()
        self.config = config or PropagatorConfig()

        # Geodesic distance computation
        self.geodesic = GeodesicPropagator(self.config.curvature)

        # Radial decay function
        self.decay = RadialDecayFunction(
            decay_type=self.config.decay_type,
            conformal_dim=self.config.conformal_dim,
            learnable=self.config.learnable_delta,
        )

        # Boundary operator: transforms bulk field to boundary representation
        self.boundary_operator = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.boundary_dim),
            nn.LayerNorm(self.config.boundary_dim),
            nn.SiLU(),
            nn.Linear(self.config.boundary_dim, self.config.boundary_dim),
        )

        # Angular harmonics for position-dependent modulation
        # Encodes "where" on the boundary we're looking
        self.position_encoding = nn.Parameter(torch.randn(self.config.n_harmonics, self.config.latent_dim) * 0.02)

        # Combine radial and angular information
        self.combine = nn.Linear(
            self.config.boundary_dim + self.config.n_harmonics,
            self.config.boundary_dim,
        )

    def forward(
        self,
        bulk_points: torch.Tensor,
        n_boundary_points: int,
        boundary_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Propagate bulk field to boundary.

        Args:
            bulk_points: Points in Poincaré ball (batch, latent_dim)
            n_boundary_points: Number of boundary positions (sequence length)
            boundary_positions: Optional explicit boundary positions
                               (batch, n_positions, latent_dim)

        Returns:
            Boundary field values (batch, n_boundary_points, boundary_dim)
        """
        batch_size = bulk_points.size(0)
        device = bulk_points.device

        # Generate boundary positions if not provided
        if boundary_positions is None:
            boundary_positions = self._generate_boundary_positions(batch_size, n_boundary_points, device)

        # Compute geodesic distances from bulk to each boundary position
        distances = self.geodesic.poincare_distance(bulk_points, boundary_positions)  # (batch, n_positions)

        # Compute decay factors
        decay_factors = self.decay(distances)  # (batch, n_positions)

        # Transform bulk field via boundary operator
        bulk_transformed = self.boundary_operator(bulk_points)  # (batch, boundary_dim)

        # Expand for broadcasting
        bulk_transformed = bulk_transformed.unsqueeze(1)  # (batch, 1, boundary_dim)
        decay_factors = decay_factors.unsqueeze(-1)  # (batch, n_positions, 1)

        # Apply radial decay
        radial_contribution = decay_factors * bulk_transformed  # (batch, n_pos, bdim)

        # Compute angular harmonics for each position
        # This encodes position-specific information
        angular = self._compute_angular_harmonics(bulk_points, boundary_positions)  # (batch, n_positions, n_harmonics)

        # Combine radial and angular
        combined = torch.cat([radial_contribution, angular], dim=-1)
        output = self.combine(combined)

        return output

    def _generate_boundary_positions(
        self,
        batch_size: int,
        n_positions: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate boundary positions on the Poincaré disk edge.

        Boundary positions are points with radius ≈ max_radius,
        distributed angularly around the disk.

        Args:
            batch_size: Batch size
            n_positions: Number of boundary positions
            device: Device for tensors

        Returns:
            Boundary positions (batch, n_positions, latent_dim)
        """
        # Generate angles uniformly around the circle
        angles = torch.linspace(
            0,
            2 * torch.pi * (1 - 1 / n_positions),
            n_positions,
            device=device,
        )

        # Create 2D boundary points at max radius
        # For higher dimensions, we embed in first 2 dims
        x = self.config.max_radius * torch.cos(angles)
        y = self.config.max_radius * torch.sin(angles)

        # Embed in full latent dimension
        boundary = torch.zeros(n_positions, self.config.latent_dim, device=device)
        boundary[:, 0] = x
        boundary[:, 1] = y

        # Expand for batch
        boundary = boundary.unsqueeze(0).expand(batch_size, -1, -1)

        return boundary

    def _compute_angular_harmonics(
        self,
        bulk_points: torch.Tensor,
        boundary_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute angular harmonics based on bulk-boundary relationship.

        Args:
            bulk_points: Bulk field positions (batch, latent_dim)
            boundary_positions: Boundary positions (batch, n_positions, latent_dim)

        Returns:
            Angular harmonic values (batch, n_positions, n_harmonics)
        """
        # Direction from bulk to each boundary point
        directions = boundary_positions - bulk_points.unsqueeze(1)
        directions = F.normalize(directions, dim=-1)

        # Project onto harmonic basis
        # position_encoding: (n_harmonics, latent_dim)
        # directions: (batch, n_positions, latent_dim)
        harmonics = torch.einsum(
            "bpd,hd->bph",
            directions,
            self.position_encoding,
        )

        # Apply nonlinearity to create harmonic response
        harmonics = torch.sin(harmonics)

        return harmonics

    def get_conformal_dimension(self) -> float:
        """Get current conformal dimension."""
        return self.decay.get_conformal_dim()


__all__ = [
    "DecayType",
    "PropagatorConfig",
    "RadialDecayFunction",
    "GeodesicPropagator",
    "BulkBoundaryPropagator",
]
