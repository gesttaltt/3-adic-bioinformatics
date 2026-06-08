# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Holographic Poincare Manifold with AdS/CFT-inspired boundary encoding.

This module applies holographic principles to Poincare ball embeddings,
inspired by black hole physics and AdS/CFT correspondence.

Key Concepts:
    - Event horizon geometry analogous to Poincare ball boundary
    - Information preservation at horizons informs boundary handling
    - Holographic principles improve latent space interpretation
    - Boundary data encodes bulk properties (AdS/CFT duality)

Research Reference:
    RESEARCH_PROPOSALS/08_HOLOGRAPHIC_POINCARE_EMBEDDINGS.md
    Maldacena (1998) "The Large N Limit of Superconformal Field Theories"
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .poincare import (
    PoincareModule,
    poincare_distance,
)


class BoundaryPoint:
    """Represents a point on the Poincare ball boundary.

    In the holographic interpretation, boundary points encode
    bulk information via the AdS/CFT-like correspondence.
    """

    def __init__(
        self,
        direction: torch.Tensor,
        conformal_weight: float = 1.0,
        encoded_info: torch.Tensor | None = None,
    ):
        """Initialize boundary point.

        Args:
            direction: Unit vector pointing to boundary (dim,)
            conformal_weight: Scaling factor for holographic reconstruction
            encoded_info: Optional additional encoded information
        """
        self.direction = F.normalize(direction, dim=-1)
        self.conformal_weight = conformal_weight
        self.encoded_info = encoded_info

    def to_bulk(self, radial_coord: float = 0.5, c: float = 1.0) -> torch.Tensor:
        """Project boundary point back into bulk at specified radius.

        Args:
            radial_coord: Radial coordinate in bulk (0, 1)
            c: Curvature parameter

        Returns:
            Bulk point on Poincare ball
        """
        max_radius = 1.0 / math.sqrt(c) - 1e-6
        radius = radial_coord * max_radius
        return self.direction * radius


class HolographicPoincareManifold(PoincareModule):
    """Poincare manifold with holographic boundary encoding.

    Implements AdS/CFT-inspired operations where boundary data
    encodes information about the bulk interior.
    """

    def __init__(
        self,
        c: float = 1.0,
        max_norm: float = 0.95,
        boundary_resolution: int = 64,
        bulk_reconstruction_order: int = 4,
        latent_dim: int = 16,
    ):
        """Initialize holographic manifold.

        Args:
            c: Curvature parameter
            max_norm: Maximum norm for bulk points
            boundary_resolution: Number of directions for boundary discretization
            bulk_reconstruction_order: Order of bulk reconstruction polynomial
            latent_dim: Dimension of latent/bulk space
        """
        super().__init__(c, max_norm)
        self.boundary_resolution = boundary_resolution
        self.bulk_reconstruction_order = bulk_reconstruction_order
        self.latent_dim = latent_dim

        # Reconstruction network: boundary -> bulk
        self.bulk_reconstructor = nn.Sequential(
            nn.Linear(boundary_resolution, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim + 1),  # direction + radial
            nn.ReLU(),
        )

    def project_to_boundary(
        self,
        z: torch.Tensor,
        eps: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project bulk points to boundary while preserving information.

        Uses conformal mapping to encode bulk information in boundary data.

        Args:
            z: Bulk points on Poincare ball (B, dim)
            eps: Small epsilon for numerical stability

        Returns:
            Tuple of (boundary_directions, radial_info)
            - boundary_directions: Unit vectors to boundary (B, dim)
            - radial_info: Radial coordinate information (B, 1)
        """
        # Compute radial coordinates using hyperbolic distance
        # V5.12.2: Use poincare_distance for proper hyperbolic radial coordinate
        origin = torch.zeros_like(z)
        hyp_dist = poincare_distance(z, origin, c=self.c).unsqueeze(-1)
        float("inf")  # In hyperbolic space, distance to boundary is infinite
        # Normalize by arctanh of max_norm for finite ratio
        radial_info = hyp_dist / (torch.arctanh(torch.tensor(self.max_norm)) + eps)

        # Direction to boundary
        direction = F.normalize(z + eps, dim=-1)

        return direction, radial_info

    def boundary_encoding(
        self,
        z: torch.Tensor,
        n_samples: int = 16,
    ) -> torch.Tensor:
        """Encode bulk point using boundary samples.

        Computes "holographic shadow" - how the point looks from boundary.

        Args:
            z: Bulk points (B, dim)
            n_samples: Number of boundary samples

        Returns:
            Boundary encoding (B, n_samples)
        """
        B, dim = z.shape
        device = z.device

        # Generate uniform directions on sphere
        # Using Fibonacci sphere for better coverage
        directions = self._fibonacci_sphere(n_samples, dim).to(device)  # (n_samples, dim)

        # Compute "visibility" from each boundary direction
        # This is related to the geodesic distance to the boundary at each direction
        # Geodesic distance from z to boundary in direction d:
        # d_boundary(z, d) = arctanh(||z||/max_norm) + arctanh(<z,d>/||z||)

        z_norm = torch.norm(z, dim=-1, keepdim=True)  # (B, 1)
        z_normed = z / (z_norm + 1e-10)  # (B, dim)

        # Projection onto each direction
        projections = torch.matmul(z_normed, directions.T)  # (B, n_samples)

        # Combine with radial information using conformal factor
        # lambda(z) = 2 / (1 - c * ||z||^2)
        conformal = 2.0 / (1.0 - self.c * z_norm**2 + 1e-10)  # (B, 1)

        # Holographic encoding: projection weighted by conformal factor
        encoding = projections * conformal  # (B, n_samples)

        # Add radial information
        radial_term = torch.arctanh(z_norm.clamp(-0.99, 0.99))  # (B, 1)
        encoding = encoding + radial_term

        return encoding

    def bulk_reconstruction(
        self,
        boundary_data: torch.Tensor,
        target_radius: float = 0.5,
    ) -> torch.Tensor:
        """Reconstruct bulk point from boundary encoding.

        Inverse of boundary_encoding - recovers bulk position from
        holographic boundary data.

        Args:
            boundary_data: Boundary encoding (B, n_samples)
            target_radius: Target radial coordinate for reconstruction

        Returns:
            Reconstructed bulk points (B, latent_dim)
        """
        # Use neural network for reconstruction
        features = self.bulk_reconstructor(boundary_data)  # (B, latent_dim + 1)

        # Use first latent_dim features as direction, last feature as radial
        direction_raw = features[:, : self.latent_dim]
        radial_raw = features[:, self.latent_dim : self.latent_dim + 1]

        # Normalize direction
        direction = F.normalize(direction_raw, dim=-1)

        # Radial coordinate (sigmoid to keep in [0, max_norm])
        radius = torch.sigmoid(radial_raw) * self.max_norm

        # Reconstruct bulk point
        z_reconstructed = direction * radius

        return z_reconstructed

    def holographic_distance(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Compute holographic distance combining bulk and boundary.

        This distance incorporates both the standard Poincare distance
        and boundary correlation (holographic duality).

        Args:
            z1: First set of points (B, dim) or (dim,)
            z2: Second set of points (B, dim) or (dim,)
            alpha: Weight for boundary term (0 = pure bulk, 1 = pure boundary)

        Returns:
            Holographic distances
        """
        # Handle 1D inputs
        z1_batch = z1.unsqueeze(0) if z1.dim() == 1 else z1
        z2_batch = z2.unsqueeze(0) if z2.dim() == 1 else z2

        # Standard bulk distance
        bulk_dist = self.dist(z1_batch, z2_batch)

        # Check if points are the same (within tolerance)
        diff = torch.norm(z1_batch - z2_batch, dim=-1)
        same_point_mask = diff < 1e-6

        # Boundary encodings
        enc1 = self.boundary_encoding(z1_batch)
        enc2 = self.boundary_encoding(z2_batch)

        # Boundary correlation distance
        enc1_norm = F.normalize(enc1, dim=-1)
        enc2_norm = F.normalize(enc2, dim=-1)
        boundary_similarity = (enc1_norm * enc2_norm).sum(dim=-1)
        boundary_dist = 1.0 - boundary_similarity

        # Combine
        holographic_dist = (1 - alpha) * bulk_dist + alpha * boundary_dist

        # Set distance to 0 for identical points
        holographic_dist = torch.where(same_point_mask, torch.zeros_like(holographic_dist), holographic_dist)

        return holographic_dist

    def conformal_flow(
        self,
        z: torch.Tensor,
        steps: int = 10,
        step_size: float = 0.01,
    ) -> torch.Tensor:
        """Apply conformal flow toward boundary.

        This simulates "information flow" toward the horizon.

        Args:
            z: Initial bulk points (B, dim)
            steps: Number of flow steps
            step_size: Size of each step

        Returns:
            Final positions after flow
        """
        z_current = z.clone()

        for _ in range(steps):
            # Flow direction: radially outward, scaled by conformal factor
            norm = torch.norm(z_current, dim=-1, keepdim=True)
            direction = z_current / (norm + 1e-10)

            # Conformal scaling
            conformal = 2.0 / (1.0 - self.c * norm**2 + 1e-10)

            # Step toward boundary
            z_current = z_current + step_size * direction * conformal

            # Project back to ball
            z_current = self.proj(z_current)

        return z_current

    def geodesic_slice(
        self,
        z: torch.Tensor,
        direction: torch.Tensor,
        n_points: int = 20,
    ) -> torch.Tensor:
        """Compute geodesic slice through bulk.

        Returns points along geodesic from -boundary through z to +boundary.

        Args:
            z: Central point (dim,)
            direction: Direction of slice (dim,)
            n_points: Number of points on slice

        Returns:
            Points along geodesic (n_points, dim)
        """
        direction = F.normalize(direction, dim=-1)

        # Parameterize geodesic: from -max_norm*direction to +max_norm*direction
        t = torch.linspace(-self.max_norm, self.max_norm, n_points, device=z.device)

        # Simple linear geodesic approximation (exact for geodesics through origin)
        # For non-origin geodesics, use Mobius operations
        points = t.unsqueeze(1) * direction.unsqueeze(0)

        return points

    def horizon_entropy(
        self,
        z_batch: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute holographic entropy at horizon.

        Inspired by Bekenstein-Hawking entropy: S = A / 4G
        Here we compute an analogous quantity for the boundary encoding.

        Args:
            z_batch: Batch of bulk points (B, dim)
            temperature: Temperature parameter

        Returns:
            Entropy values (B,)
        """
        # Project to boundary
        directions, radial = self.project_to_boundary(z_batch)

        # Boundary encoding
        encoding = self.boundary_encoding(z_batch)

        # Softmax to get probability distribution
        probs = F.softmax(encoding / temperature, dim=-1)

        # Shannon entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)

        return entropy

    def bulk_boundary_correspondence(
        self,
        bulk_points: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute full AdS/CFT-like correspondence data.

        Args:
            bulk_points: Points in bulk (B, dim)

        Returns:
            Dictionary with correspondence data
        """
        # Project to boundary
        directions, radial_info = self.project_to_boundary(bulk_points)

        # Full boundary encoding - use boundary_resolution samples
        encoding = self.boundary_encoding(bulk_points, n_samples=self.boundary_resolution)

        # Attempt reconstruction
        reconstructed = self.bulk_reconstruction(encoding)

        # Reconstruction error - compare with original bulk points
        # reconstructed is (B, latent_dim), bulk_points is (B, dim)
        min_dim = min(reconstructed.shape[1], bulk_points.shape[1])
        recon_error = F.mse_loss(reconstructed[:, :min_dim], bulk_points[:, :min_dim], reduction="none").mean(dim=-1)

        # Horizon entropy
        entropy = self.horizon_entropy(bulk_points)

        # Conformal factor at each point
        norm_sq = (bulk_points**2).sum(dim=-1, keepdim=True)
        conformal = 2.0 / (1.0 - self.c * norm_sq + 1e-10)

        return {
            "boundary_directions": directions,
            "radial_coordinates": radial_info,
            "boundary_encoding": encoding,
            "reconstructed_bulk": reconstructed,
            "reconstruction_error": recon_error,
            "horizon_entropy": entropy,
            "conformal_factor": conformal.squeeze(-1),
        }

    def _fibonacci_sphere(self, n: int, dim: int) -> torch.Tensor:
        """Generate n approximately uniform points on unit sphere.

        Uses generalized Fibonacci lattice for arbitrary dimensions.

        Args:
            n: Number of points
            dim: Dimension

        Returns:
            Points on unit sphere (n, dim)
        """
        if dim == 2:
            # Circle
            angles = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
            return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)

        elif dim == 3:
            # Standard Fibonacci sphere
            idx = torch.arange(0, n, dtype=torch.float32) + 0.5
            phi = math.pi * (1 + math.sqrt(5))

            y = 1 - 2 * idx / n
            radius = torch.sqrt(1 - y * y)
            theta = phi * idx

            points = torch.stack([torch.cos(theta) * radius, y, torch.sin(theta) * radius], dim=1)
            return points

        else:
            # Higher dimensions: use normalized Gaussian
            points = torch.randn(n, dim)
            return F.normalize(points, dim=1)


class HolographicLoss(nn.Module):
    """Loss function based on holographic principles.

    Encourages consistency between bulk representations and
    their boundary encodings.
    """

    def __init__(
        self,
        c: float = 1.0,
        reconstruction_weight: float = 1.0,
        entropy_weight: float = 0.1,
        consistency_weight: float = 0.5,
        latent_dim: int = 16,
        boundary_resolution: int = 64,
    ):
        """Initialize holographic loss.

        Args:
            c: Curvature parameter
            reconstruction_weight: Weight for boundary reconstruction loss
            entropy_weight: Weight for entropy regularization
            consistency_weight: Weight for bulk-boundary consistency
            latent_dim: Dimension of latent space
            boundary_resolution: Resolution for boundary encoding
        """
        super().__init__()
        self.manifold = HolographicPoincareManifold(c=c, latent_dim=latent_dim, boundary_resolution=boundary_resolution)
        self.reconstruction_weight = reconstruction_weight
        self.entropy_weight = entropy_weight
        self.consistency_weight = consistency_weight

    def forward(
        self,
        z: torch.Tensor,
        z_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute holographic loss.

        Args:
            z: Bulk latent representations (B, dim)
            z_target: Optional target representations for supervision

        Returns:
            Dictionary of loss components
        """
        # Get correspondence data
        correspondence = self.manifold.bulk_boundary_correspondence(z)

        # Reconstruction loss
        recon_loss = correspondence["reconstruction_error"].mean()

        # Entropy regularization (encourage moderate entropy)
        entropy = correspondence["horizon_entropy"]
        target_entropy = 0.5 * math.log(z.shape[1])  # Target: log(dim)/2
        entropy_loss = (entropy - target_entropy).pow(2).mean()

        # Consistency loss: nearby bulk points should have similar boundary encodings
        if z.shape[0] > 1:
            # Random pairs
            idx1 = torch.randint(0, z.shape[0], (min(z.shape[0], 32),), device=z.device)
            idx2 = torch.randint(0, z.shape[0], (min(z.shape[0], 32),), device=z.device)

            bulk_dist = self.manifold.dist(z[idx1], z[idx2])
            enc = correspondence["boundary_encoding"]
            enc_dist = F.pairwise_distance(enc[idx1], enc[idx2])

            # Distances should be correlated
            consistency_loss = (bulk_dist - enc_dist).pow(2).mean()
        else:
            consistency_loss = torch.tensor(0.0, device=z.device)

        # Supervised loss if target provided
        if z_target is not None:
            supervised_loss = self.manifold.holographic_distance(z, z_target).mean()
        else:
            supervised_loss = torch.tensor(0.0, device=z.device)

        total_loss = (
            self.reconstruction_weight * recon_loss
            + self.entropy_weight * entropy_loss
            + self.consistency_weight * consistency_loss
            + supervised_loss
        )

        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "entropy_loss": entropy_loss,
            "consistency_loss": consistency_loss,
            "supervised_loss": supervised_loss,
            "mean_entropy": entropy.mean(),
            "mean_conformal": correspondence["conformal_factor"].mean(),
        }


class HolographicProjection(nn.Module):
    """Project embeddings using holographic principles.

    Combines standard Poincare projection with holographic encoding
    for richer representations.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        c: float = 1.0,
        boundary_resolution: int = 32,
    ):
        """Initialize holographic projection.

        Args:
            input_dim: Input dimension
            output_dim: Output (latent) dimension
            c: Curvature parameter
            boundary_resolution: Resolution for boundary encoding
        """
        super().__init__()
        self.output_dim = output_dim
        self.boundary_resolution = boundary_resolution
        self.manifold = HolographicPoincareManifold(
            c=c,
            boundary_resolution=boundary_resolution,
            latent_dim=output_dim,
        )

        # Projection to latent space
        self.projector = nn.Linear(input_dim, output_dim)

        # Holographic enhancement layer - use 16 samples for encoding
        self.n_boundary_samples = 16
        self.holographic_enhance = nn.Linear(output_dim + self.n_boundary_samples, output_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Project input with holographic enhancement.

        Args:
            x: Input features (B, input_dim)

        Returns:
            Tuple of (projected_z, holographic_info)
        """
        # Initial projection
        z = self.projector(x)

        # Project to Poincare ball
        z = self.manifold.proj(z)

        # Get boundary encoding with fixed n_samples
        encoding = self.manifold.boundary_encoding(z, n_samples=self.n_boundary_samples)

        # Holographic enhancement
        z_enhanced = self.holographic_enhance(torch.cat([z, encoding], dim=-1))
        z_enhanced = self.manifold.proj(z_enhanced)

        # Compute holographic info
        info = self.manifold.bulk_boundary_correspondence(z_enhanced)

        return z_enhanced, info
