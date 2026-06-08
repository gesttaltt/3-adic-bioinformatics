# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Lattice-Aware Hyperbolic Projection.

Extends hyperbolic projection to respect resistance lattice ordering.
The key insight is that hyperbolic space naturally encodes hierarchies:
- Points closer to the origin are "ancestors" (less resistant)
- Points near the boundary are "descendants" (more resistant)

This module enforces that the lattice structure of resistance profiles
is preserved in the hyperbolic embedding:
- If profile A ⊂ B (A is less resistant), then A is closer to origin
- Chains in the lattice map to geodesics in hyperbolic space
- Antichains map to equidistant points from origin
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.analysis.set_theory.lattice import (
    ResistanceLattice,
    ResistanceLevel,
)
from src.analysis.set_theory.mutation_sets import MutationSet
from src.geometry import poincare_distance
from src.models.hyperbolic_projection import HyperbolicProjection


@dataclass
class LatticeProjectionConfig:
    """Configuration for lattice-aware projection.

    Attributes:
        latent_dim: Latent space dimension
        hidden_dim: Hidden layer dimension
        max_radius: Maximum radius in Poincare ball
        curvature: Hyperbolic curvature
        lattice_margin: Margin for lattice ordering constraints
        level_spacing: Radial spacing between resistance levels
        use_level_targets: Use level-based radius targets
        learnable_curvature: Allow curvature to be learned
    """

    latent_dim: int = 16
    hidden_dim: int = 64
    max_radius: float = 0.95
    curvature: float = 1.0
    lattice_margin: float = 0.05
    level_spacing: float = 0.15
    use_level_targets: bool = True
    learnable_curvature: bool = False


class LatticeAwareHyperbolicProjection(nn.Module):
    """Hyperbolic projection that respects resistance lattice structure.

    Builds on HyperbolicProjection but adds:
    1. Lattice-informed radius targets (resistance level -> radius)
    2. Ordering loss to preserve lattice relationships
    3. Chain consistency loss for geodesic paths

    Example:
        >>> lattice = ResistanceLattice()
        >>> proj = LatticeAwareHyperbolicProjection(config, lattice)
        >>> embeddings = encoder(sequences)
        >>> hyp_embeddings, losses = proj(embeddings, mutation_sets)
    """

    def __init__(
        self,
        config: LatticeProjectionConfig | None = None,
        lattice: ResistanceLattice | None = None,
    ):
        """Initialize lattice-aware projection.

        Args:
            config: Projection configuration
            lattice: Resistance lattice for constraints
        """
        super().__init__()
        self.config = config or LatticeProjectionConfig()
        self.lattice = lattice or ResistanceLattice()

        # Base hyperbolic projection
        self.projection = HyperbolicProjection(
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.hidden_dim,
            max_radius=self.config.max_radius,
            curvature=self.config.curvature,
            learnable_curvature=self.config.learnable_curvature,
        )

        # Level-to-radius mapping
        # Higher resistance level -> larger radius (closer to boundary)
        n_levels = len(ResistanceLevel)
        level_radii = torch.linspace(
            0.1,  # Susceptible near origin
            self.config.max_radius,  # XDR near boundary
            n_levels,
        )
        self.register_buffer("level_radii", level_radii)

        # Learnable level embeddings for radius guidance
        self.level_embeddings = nn.Embedding(n_levels, self.config.hidden_dim)

        # Radius adjustment network
        # Input: latent_dim (euclidean embedding) + hidden_dim (level embedding)
        self.radius_adjust = nn.Sequential(
            nn.Linear(self.config.latent_dim + self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        mutation_sets: list[MutationSet] | None = None,
        return_losses: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Project to hyperbolic space with lattice awareness.

        Args:
            embeddings: Euclidean embeddings (batch_size, latent_dim)
            mutation_sets: Corresponding mutation sets
            return_losses: Whether to compute lattice losses

        Returns:
            Tuple of (hyperbolic embeddings, loss dict or None)
        """
        batch_size = embeddings.size(0)

        # Get base hyperbolic projection
        hyp_embeddings = self.projection(embeddings)

        losses = None
        if return_losses and mutation_sets and len(mutation_sets) == batch_size:
            losses = self._compute_lattice_losses(hyp_embeddings, mutation_sets)

            # Apply radius adjustment based on lattice levels
            if self.config.use_level_targets:
                hyp_embeddings = self._adjust_radii(hyp_embeddings, embeddings, mutation_sets)

        return hyp_embeddings, losses

    def _adjust_radii(
        self,
        hyp_embeddings: torch.Tensor,
        eucl_embeddings: torch.Tensor,
        mutation_sets: list[MutationSet],
    ) -> torch.Tensor:
        """Adjust radii based on resistance levels.

        Args:
            hyp_embeddings: Current hyperbolic embeddings
            eucl_embeddings: Original Euclidean embeddings
            mutation_sets: Mutation sets for level computation

        Returns:
            Adjusted hyperbolic embeddings
        """
        batch_size = hyp_embeddings.size(0)
        device = hyp_embeddings.device

        # Get resistance levels
        levels = torch.zeros(batch_size, dtype=torch.long, device=device)
        for i, ms in enumerate(mutation_sets):
            level = self.lattice.resistance_level(ms)
            levels[i] = level.value

        # Get level embeddings
        level_embs = self.level_embeddings(levels)

        # Compute radius adjustment
        combined = torch.cat([eucl_embeddings, level_embs], dim=-1)
        adjustment = self.radius_adjust(combined).squeeze(-1)

        # Get target radii from levels
        target_radii = self.level_radii[levels]

        # V5.12.2: Blend current radius with target using hyperbolic distance
        origin = torch.zeros_like(hyp_embeddings)
        current_radii = poincare_distance(hyp_embeddings, origin, c=self.config.curvature)
        adjusted_radii = 0.7 * current_radii + 0.3 * target_radii * adjustment

        # Rescale embeddings
        directions = F.normalize(hyp_embeddings, dim=-1)
        adjusted_embeddings = directions * adjusted_radii.unsqueeze(-1)

        # Clamp to Poincare ball
        norms = torch.norm(adjusted_embeddings, dim=-1, keepdim=True)
        adjusted_embeddings = torch.where(
            norms > self.config.max_radius,
            adjusted_embeddings * self.config.max_radius / norms,
            adjusted_embeddings,
        )

        return adjusted_embeddings

    def _compute_lattice_losses(
        self,
        hyp_embeddings: torch.Tensor,
        mutation_sets: list[MutationSet],
    ) -> dict[str, torch.Tensor]:
        """Compute lattice-based losses.

        Args:
            hyp_embeddings: Hyperbolic embeddings
            mutation_sets: Mutation sets

        Returns:
            Dictionary of losses
        """
        len(mutation_sets)

        losses = {}

        # 1. Ordering loss: if A ⊂ B, then ||A|| < ||B||
        ordering_loss = self._ordering_loss(hyp_embeddings, mutation_sets)
        losses["ordering"] = ordering_loss

        # 2. Level consistency loss: samples at same level should have similar radii
        level_loss = self._level_consistency_loss(hyp_embeddings, mutation_sets)
        losses["level"] = level_loss

        # 3. Chain consistency loss: chains should form geodesics
        chain_loss = self._chain_loss(hyp_embeddings, mutation_sets)
        losses["chain"] = chain_loss

        return losses

    def _ordering_loss(
        self,
        hyp_embeddings: torch.Tensor,
        mutation_sets: list[MutationSet],
    ) -> torch.Tensor:
        """Compute ordering loss for lattice constraints.

        Args:
            hyp_embeddings: Hyperbolic embeddings
            mutation_sets: Mutation sets

        Returns:
            Ordering loss
        """
        device = hyp_embeddings.device
        # V5.12.2: Use hyperbolic distance for radii
        origin = torch.zeros_like(hyp_embeddings)
        radii = poincare_distance(hyp_embeddings, origin, c=self.config.curvature)

        loss = torch.tensor(0.0, device=device)
        n_pairs = 0

        for i in range(len(mutation_sets)):
            for j in range(i + 1, len(mutation_sets)):
                cmp = self.lattice.compare(mutation_sets[i], mutation_sets[j])

                if cmp == -1:  # i < j in lattice
                    # r_i should be < r_j
                    violation = F.relu(radii[i] - radii[j] + self.config.lattice_margin)
                    loss = loss + violation
                    n_pairs += 1
                elif cmp == 1:  # i > j in lattice
                    # r_i should be > r_j
                    violation = F.relu(radii[j] - radii[i] + self.config.lattice_margin)
                    loss = loss + violation
                    n_pairs += 1

        if n_pairs > 0:
            loss = loss / n_pairs

        return loss

    def _level_consistency_loss(
        self,
        hyp_embeddings: torch.Tensor,
        mutation_sets: list[MutationSet],
    ) -> torch.Tensor:
        """Compute level consistency loss.

        Samples at the same resistance level should have similar radii.

        Args:
            hyp_embeddings: Hyperbolic embeddings
            mutation_sets: Mutation sets

        Returns:
            Level consistency loss
        """
        device = hyp_embeddings.device
        # V5.12.2: Use hyperbolic distance for radii
        origin = torch.zeros_like(hyp_embeddings)
        radii = poincare_distance(hyp_embeddings, origin, c=self.config.curvature)

        # Group by level
        levels = [self.lattice.resistance_level(ms).value for ms in mutation_sets]
        levels = torch.tensor(levels, device=device)

        loss = torch.tensor(0.0, device=device)
        n_groups = 0

        for level in range(len(ResistanceLevel)):
            mask = levels == level
            if mask.sum() > 1:
                level_radii = radii[mask]
                # Variance of radii within level should be small
                variance = level_radii.var()
                loss = loss + variance
                n_groups += 1

        if n_groups > 0:
            loss = loss / n_groups

        return loss

    def _chain_loss(
        self,
        hyp_embeddings: torch.Tensor,
        mutation_sets: list[MutationSet],
    ) -> torch.Tensor:
        """Compute chain consistency loss.

        Chains in the lattice should form approximate geodesics
        in hyperbolic space.

        Args:
            hyp_embeddings: Hyperbolic embeddings
            mutation_sets: Mutation sets

        Returns:
            Chain loss
        """
        device = hyp_embeddings.device

        # Find chains of length >= 3 in the batch
        # This is simplified - we look for ordered triples
        loss = torch.tensor(0.0, device=device)
        n_chains = 0

        for i in range(len(mutation_sets)):
            for j in range(len(mutation_sets)):
                for k in range(len(mutation_sets)):
                    if i in (j, k) or j == k:
                        continue

                    # Check if i < j < k in lattice
                    cmp_ij = self.lattice.compare(mutation_sets[i], mutation_sets[j])
                    cmp_jk = self.lattice.compare(mutation_sets[j], mutation_sets[k])

                    if cmp_ij == -1 and cmp_jk == -1:
                        # i < j < k forms a chain
                        # j should be on geodesic between i and k
                        # Approximate: d(i,j) + d(j,k) ≈ d(i,k)
                        d_ij = self._hyperbolic_distance(hyp_embeddings[i], hyp_embeddings[j])
                        d_jk = self._hyperbolic_distance(hyp_embeddings[j], hyp_embeddings[k])
                        d_ik = self._hyperbolic_distance(hyp_embeddings[i], hyp_embeddings[k])

                        # Triangle inequality violation
                        chain_violation = F.relu(d_ij + d_jk - d_ik - 0.01)
                        loss = loss + chain_violation
                        n_chains += 1

        if n_chains > 0:
            loss = loss / n_chains

        return loss

    def _hyperbolic_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hyperbolic distance in Poincare ball.

        Args:
            x: First point
            y: Second point

        Returns:
            Hyperbolic distance
        """
        c = self.config.curvature
        sqrt_c = c**0.5

        diff = x - y
        norm_x_sq = (x * x).sum()
        norm_y_sq = (y * y).sum()
        norm_diff_sq = (diff * diff).sum()

        numerator = 2 * c * norm_diff_sq
        denominator = (1 - c * norm_x_sq) * (1 - c * norm_y_sq)

        # Clamp for numerical stability
        ratio = torch.clamp(1 + numerator / (denominator + 1e-8), min=1.0 + 1e-8)

        return (1 / sqrt_c) * torch.acosh(ratio)

    def get_level_radii(self) -> torch.Tensor:
        """Get target radii for each resistance level.

        Returns:
            Tensor of radii for each level
        """
        return self.level_radii

    def project_to_level(
        self,
        embeddings: torch.Tensor,
        target_level: ResistanceLevel,
    ) -> torch.Tensor:
        """Project embeddings to a specific resistance level radius.

        Args:
            embeddings: Euclidean embeddings
            target_level: Target resistance level

        Returns:
            Hyperbolic embeddings at target level radius
        """
        hyp = self.projection(embeddings)

        # Rescale to target level radius
        directions = F.normalize(hyp, dim=-1)
        target_radius = self.level_radii[target_level.value]

        return directions * target_radius


class LatticeGuidedDecoder(nn.Module):
    """Decoder that uses lattice structure for generation guidance.

    When decoding, uses the hyperbolic radius to determine
    the resistance level of the generated sequence.
    """

    def __init__(
        self,
        base_decoder: nn.Module,
        lattice: ResistanceLattice,
        latent_dim: int = 16,
        curvature: float = 1.0,
    ):
        """Initialize lattice-guided decoder.

        Args:
            base_decoder: Base decoder network
            lattice: Resistance lattice
            latent_dim: Latent dimension
            curvature: Hyperbolic curvature for poincare_distance (V5.12.2)
        """
        super().__init__()
        self.decoder = base_decoder
        self.lattice = lattice
        self.latent_dim = latent_dim
        self.curvature = curvature

        # Level conditioning
        n_levels = len(ResistanceLevel)
        self.level_embeddings = nn.Embedding(n_levels, latent_dim)

    def forward(
        self,
        hyp_embeddings: torch.Tensor,
        target_level: ResistanceLevel | None = None,
    ) -> torch.Tensor:
        """Decode with optional level conditioning.

        Args:
            hyp_embeddings: Hyperbolic embeddings
            target_level: Optional target resistance level

        Returns:
            Decoded output
        """
        batch_size = hyp_embeddings.size(0)
        device = hyp_embeddings.device

        # Infer levels from radii if not provided
        if target_level is None:
            # V5.12.2: Use hyperbolic distance for radii
            origin = torch.zeros_like(hyp_embeddings)
            radii = poincare_distance(hyp_embeddings, origin, c=self.curvature)
            # Map radii to nearest level
            level_radii = self.lattice_projection.level_radii.unsqueeze(0)
            diffs = (radii.unsqueeze(-1) - level_radii).abs()
            levels = diffs.argmin(dim=-1)
        else:
            levels = torch.full((batch_size,), target_level.value, dtype=torch.long, device=device)

        # Add level conditioning
        level_cond = self.level_embeddings(levels)
        conditioned = hyp_embeddings + 0.1 * level_cond

        return self.decoder(conditioned)
