# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Hyperbolic Reconstruction Loss for Pure Hyperbolic VAE (v5.10).

This module provides reconstruction loss computed in hyperbolic space,
replacing Euclidean MSE that fights against tree-like structure.

Key insight: Standard reconstruction loss uses Euclidean MSE:
    ||x - x_hat||^2 (Euclidean)

This implicitly assumes flat geometry, which conflicts with the
curved Poincare ball where our embeddings live.

Hyperbolic reconstruction uses geodesic distance:
    d_poincare(z_enc, z_dec)^2

This maintains geometric consistency throughout the VAE.

For discrete outputs (ternary operations), we can also use a
hyperbolic-weighted cross-entropy where the loss is modulated
by the hyperbolic position of the latent code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# P2 FIX: Use core module for vectorized prefix computation
from ..core import TERNARY
from ..geometry import poincare_distance, project_to_poincare


class HyperbolicReconLoss(nn.Module):
    """Hyperbolic-aware reconstruction loss.

    Provides multiple modes:
    1. geodesic: Pure geodesic distance between latent codes
    2. weighted_ce: Cross-entropy weighted by hyperbolic radius
    3. hybrid: Combination of both

    The weighting by hyperbolic radius implements a natural curriculum:
    - Points near origin (high valuation) have higher loss weight
    - Points near boundary (low valuation) have lower loss weight
    - This encourages learning the tree structure first (roots before leaves)
    """

    def __init__(
        self,
        mode: str = "hybrid",
        curvature: float = 1.0,
        max_norm: float = 0.95,
        geodesic_weight: float = 0.3,
        radius_weighting: bool = True,
        radius_power: float = 2.0,
    ):
        """Initialize Hyperbolic Reconstruction Loss.

        Args:
            mode: 'geodesic', 'weighted_ce', or 'hybrid'
            curvature: Poincare ball curvature
            max_norm: Maximum radius in Poincare ball
            geodesic_weight: Weight for geodesic term in hybrid mode
            radius_weighting: Whether to weight loss by radial position
            radius_power: Power for radius weighting (higher = more emphasis on origin)
        """
        super().__init__()
        self.mode = mode
        self.curvature = curvature
        self.max_norm = max_norm
        self.geodesic_weight = geodesic_weight
        self.radius_weighting = radius_weighting
        self.radius_power = radius_power

    def _project_to_poincare(self, z: torch.Tensor) -> torch.Tensor:
        """Project Euclidean points onto the Poincare ball."""
        return project_to_poincare(z, max_norm=self.max_norm, c=self.curvature)

    def _compute_radius_weights(self, z_hyp: torch.Tensor) -> torch.Tensor:
        """Compute importance weights based on radial position.

        Points near origin get higher weights (they represent
        high-valuation, structurally important operations).

        Args:
            z_hyp: Points on Poincare ball

        Returns:
            Weights (batch_size,) in [0.5, 2.0] range
        """
        # V5.12.2: Use hyperbolic distance instead of Euclidean norm
        origin = torch.zeros_like(z_hyp)
        radius = poincare_distance(z_hyp, origin, c=self.curvature)
        normalized_radius = radius / self.max_norm

        # Weight = (1 - radius)^power + 0.5
        # At origin (r=0): weight = 1.5
        # At boundary (r=max_norm): weight = 0.5
        weights = (1 - normalized_radius) ** self.radius_power + 0.5

        return weights

    def geodesic_reconstruction_loss(
        self,
        z_enc: torch.Tensor,
        z_dec: torch.Tensor,
        z_enc_hyp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute geodesic reconstruction loss.

        This measures how well the decoder can reconstruct the
        encoder's latent code using hyperbolic distance.

        For VAEs, z_dec is typically the mean from another encoder
        on the reconstructed output.

        Args:
            z_enc: Encoder latent codes (batch, latent_dim)
            z_dec: Decoder latent codes (batch, latent_dim)
            z_enc_hyp: Pre-projected encoder codes (optional)

        Returns:
            Geodesic reconstruction loss (scalar)
        """
        # Project to Poincare ball if not already
        if z_enc_hyp is None:
            z_enc_hyp = self._project_to_poincare(z_enc)
        z_dec_hyp = self._project_to_poincare(z_dec)

        # Geodesic distance (using unified implementation from geometry module)
        distances = poincare_distance(z_enc_hyp, z_dec_hyp, c=self.curvature)

        # Apply radius weighting if enabled
        if self.radius_weighting:
            weights = self._compute_radius_weights(z_enc_hyp)
            loss = (weights * distances**2).mean()
        else:
            loss = (distances**2).mean()

        return loss

    def weighted_cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor, z_hyp: torch.Tensor) -> torch.Tensor:
        """Compute radius-weighted cross-entropy loss.

        Args:
            logits: Model logits (batch, 9, 3) for ternary operations
            targets: Target values (batch, 9) in {-1, 0, 1}
            z_hyp: Hyperbolic latent codes

        Returns:
            Weighted cross-entropy loss (scalar)
        """
        batch_size = targets.size(0)

        # Convert targets to class indices
        target_classes = (targets + 1).long()  # {-1,0,1} -> {0,1,2}

        # Compute per-sample cross-entropy
        ce_per_sample = (
            F.cross_entropy(logits.view(-1, 3), target_classes.view(-1), reduction="none")
            .view(batch_size, -1)
            .sum(dim=1)
        )

        # Apply radius weighting if enabled
        if self.radius_weighting:
            weights = self._compute_radius_weights(z_hyp)
            loss = (weights * ce_per_sample).mean()
        else:
            loss = ce_per_sample.mean()

        return loss

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        z_enc: torch.Tensor,
        z_dec: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute hyperbolic reconstruction loss.

        Args:
            logits: Model output logits (batch, 9, 3)
            targets: Target values (batch, 9) in {-1, 0, 1}
            z_enc: Encoder latent codes
            z_dec: Decoder latent codes (for geodesic mode)

        Returns:
            Tuple of (loss, metrics_dict)
        """
        z_enc_hyp = self._project_to_poincare(z_enc)

        # V5.12.2: Use hyperbolic distance instead of Euclidean norm
        origin = torch.zeros_like(z_enc_hyp)
        metrics = {"mean_radius": poincare_distance(z_enc_hyp, origin, c=self.curvature).mean().item()}

        if self.mode == "geodesic":
            if z_dec is None:
                raise ValueError("z_dec required for geodesic mode")
            loss = self.geodesic_reconstruction_loss(z_enc, z_dec, z_enc_hyp)
            metrics["geodesic_loss"] = loss.item()

        elif self.mode == "weighted_ce":
            loss = self.weighted_cross_entropy(logits, targets, z_enc_hyp)
            metrics["weighted_ce"] = loss.item()

        elif self.mode == "hybrid":
            ce_loss = self.weighted_cross_entropy(logits, targets, z_enc_hyp)
            metrics["weighted_ce"] = ce_loss.item()

            if z_dec is not None:
                geo_loss = self.geodesic_reconstruction_loss(z_enc, z_dec, z_enc_hyp)
                metrics["geodesic_loss"] = geo_loss.item()
                loss = ce_loss + self.geodesic_weight * geo_loss
            else:
                loss = ce_loss

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return loss, metrics


class HomeostaticReconLoss(HyperbolicReconLoss):
    """Hyperbolic reconstruction with homeostatic adaptation.

    Extends HyperbolicReconLoss with adaptive parameters that maintain
    training stability through algebraic convergence.

    Homeostatic mechanisms:
    1. Adaptive geodesic weight: Increases when coverage drops
    2. Adaptive radius power: Modulates curriculum difficulty
    3. Loss scaling: Prevents gradient explosion/vanishing
    """

    def __init__(
        self,
        mode: str = "hybrid",
        curvature: float = 1.0,
        max_norm: float = 0.95,
        geodesic_weight: float = 0.3,
        radius_weighting: bool = True,
        radius_power: float = 2.0,
        # Homeostatic parameters
        geodesic_weight_min: float = 0.1,
        geodesic_weight_max: float = 0.8,
        radius_power_min: float = 1.0,
        radius_power_max: float = 4.0,
        adaptation_rate: float = 0.01,
    ):
        """Initialize Homeostatic Reconstruction Loss.

        Args:
            mode: Loss mode ('geodesic', 'weighted_ce', 'hybrid')
            curvature: Poincare ball curvature
            max_norm: Maximum radius
            geodesic_weight: Initial geodesic weight
            radius_weighting: Enable radius-based weighting
            radius_power: Initial radius power
            geodesic_weight_min/max: Bounds for geodesic weight
            radius_power_min/max: Bounds for radius power
            adaptation_rate: Rate of homeostatic adaptation
        """
        super().__init__(
            mode,
            curvature,
            max_norm,
            geodesic_weight,
            radius_weighting,
            radius_power,
        )

        self.geodesic_weight_min = geodesic_weight_min
        self.geodesic_weight_max = geodesic_weight_max
        self.radius_power_min = radius_power_min
        self.radius_power_max = radius_power_max
        self.adaptation_rate = adaptation_rate

        # Adaptive parameters
        self.register_buffer("adaptive_geodesic_weight", torch.tensor(geodesic_weight))
        self.adaptive_geodesic_weight: torch.Tensor  # Type hint for mypy
        self.register_buffer("adaptive_radius_power", torch.tensor(radius_power))
        self.adaptive_radius_power: torch.Tensor  # Type hint for mypy

        # EMA tracking
        self.register_buffer("loss_ema", torch.tensor(1.0))
        self.loss_ema: torch.Tensor  # Type hint for mypy
        self.register_buffer("coverage_ema", torch.tensor(50.0))
        self.coverage_ema: torch.Tensor  # Type hint for mypy

    def update_homeostatic_state(self, loss: torch.Tensor, coverage: float, correlation: float = 0.0):
        """Update homeostatic parameters based on training state.

        Args:
            loss: Current reconstruction loss
            coverage: Current coverage percentage
            correlation: Current 3-adic correlation
        """
        # Update EMAs
        alpha = 0.1
        self.loss_ema = alpha * loss.detach() + (1 - alpha) * self.loss_ema
        self.coverage_ema = alpha * coverage + (1 - alpha) * self.coverage_ema

        # Adapt geodesic weight based on coverage
        # Low coverage -> increase geodesic weight (more structure enforcement)
        # High coverage -> decrease geodesic weight (allow finer distinctions)
        coverage_target = 90.0
        coverage_error = coverage_target - self.coverage_ema
        geodesic_delta = self.adaptation_rate * coverage_error * 0.01

        new_geodesic = self.adaptive_geodesic_weight + geodesic_delta
        self.adaptive_geodesic_weight = torch.clamp(new_geodesic, self.geodesic_weight_min, self.geodesic_weight_max)
        self.geodesic_weight = self.adaptive_geodesic_weight.item()

        # Adapt radius power based on correlation
        # Low correlation -> decrease power (easier curriculum)
        # High correlation -> increase power (harder curriculum)
        correlation_target = 0.8
        correlation_error = correlation - correlation_target
        power_delta = self.adaptation_rate * correlation_error * 0.1

        new_power = self.adaptive_radius_power + power_delta
        self.adaptive_radius_power = torch.clamp(new_power, self.radius_power_min, self.radius_power_max)
        self.radius_power = self.adaptive_radius_power.item()

    def get_homeostatic_state(self) -> dict:
        """Return current homeostatic state."""
        return {
            "geodesic_weight": self.adaptive_geodesic_weight.item(),
            "radius_power": self.adaptive_radius_power.item(),
            "loss_ema": self.loss_ema.item(),
            "coverage_ema": self.coverage_ema.item(),
        }

    def set_from_statenet(
        self,
        delta_geodesic_weight: float = 0.0,
        delta_radius_power: float = 0.0,
    ):
        """Apply StateNet corrections.

        Args:
            delta_geodesic_weight: Correction for geodesic weight
            delta_radius_power: Correction for radius power
        """
        new_geodesic = self.adaptive_geodesic_weight + delta_geodesic_weight * 0.1
        self.adaptive_geodesic_weight = torch.clamp(new_geodesic, self.geodesic_weight_min, self.geodesic_weight_max)
        self.geodesic_weight = self.adaptive_geodesic_weight.item()

        new_power = self.adaptive_radius_power + delta_radius_power * 0.1
        self.adaptive_radius_power = torch.clamp(new_power, self.radius_power_min, self.radius_power_max)
        self.radius_power = self.adaptive_radius_power.item()


class HyperbolicCentroidLoss(nn.Module):
    """Centroid-based loss in hyperbolic space for p-adic structure.

    This loss encourages operations with the same 3-adic "prefix" to
    cluster around hyperbolic centroids, creating the tree structure.

    For 3-adic numbers, operations that share the same first k digits
    in base-3 should be close together. This creates a natural hierarchy:
    - Level 0: All operations (root)
    - Level 1: Three clusters (digit 0, 1, 2)
    - Level 2: Nine clusters
    - etc.

    The loss computes Frechet means (hyperbolic centroids) for each
    cluster and encourages points to be close to their centroid.
    """

    def __init__(
        self,
        max_level: int = 4,
        curvature: float = 1.0,
        max_norm: float = 0.95,
        level_weights: torch.Tensor | None = None,
    ):
        """Initialize Hyperbolic Centroid Loss.

        Args:
            max_level: Maximum tree depth to enforce
            curvature: Poincare ball curvature
            max_norm: Maximum radius
            level_weights: Weights for each level (default: exponential decay)
        """
        super().__init__()
        self.max_level = max_level
        self.curvature = curvature
        self.max_norm = max_norm

        if level_weights is None:
            # Higher levels (finer clusters) get lower weight
            level_weights = torch.tensor([0.5**i for i in range(max_level)])
        self.register_buffer("level_weights", level_weights)

    def _get_prefix(self, idx: int, level: int) -> int:
        """Get base-3 prefix of an index at given level.

        Args:
            idx: Operation index (0 to 19682)
            level: Tree level (0 = root, 1 = first digit, etc.)

        Returns:
            Prefix as integer
        """
        # Convert to base-3 and take first 'level' digits
        divisor = 3 ** (9 - level)  # 19683 = 3^9
        return idx // divisor

    def _compute_frechet_mean(
        self,
        points: torch.Tensor,
        weights: torch.Tensor | None = None,
        n_iter: int = 10,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """Compute Frechet mean (hyperbolic centroid) of points.

        Uses iterative algorithm:
        1. Start with Euclidean mean projected to ball
        2. Iteratively refine using tangent space
        3. Early exit on convergence

        Args:
            points: Points on Poincare ball (n, d)
            weights: Optional weights for each point
            n_iter: Maximum number of iterations
            tol: Convergence tolerance (stops when mean changes < tol)

        Returns:
            Frechet mean (d,)
        """
        if weights is None:
            weights = torch.ones(points.size(0), device=points.device)
        weights = weights / weights.sum()

        # Initialize with weighted Euclidean mean
        mean = (points * weights.unsqueeze(1)).sum(dim=0)
        mean = project_to_poincare(mean.unsqueeze(0), max_norm=self.max_norm, c=self.curvature).squeeze(0)

        # Iterative refinement with convergence check
        for _ in range(n_iter):
            prev_mean = mean.clone()
            # Move mean toward weighted average of points
            direction = (points - mean.unsqueeze(0)) * weights.unsqueeze(1)
            direction = direction.sum(dim=0)
            mean = mean + 0.1 * direction
            # Re-project to ball
            mean = project_to_poincare(mean.unsqueeze(0), max_norm=self.max_norm, c=self.curvature).squeeze(0)
            # Check convergence
            if torch.norm(mean - prev_mean) < tol:
                break

        return mean

    def forward(self, z: torch.Tensor, batch_indices: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute hyperbolic centroid loss.

        Args:
            z: Latent codes (batch, latent_dim)
            batch_indices: Operation indices (batch,)

        Returns:
            Tuple of (loss, metrics)
        """
        z_hyp = project_to_poincare(z, max_norm=self.max_norm, c=self.curvature)
        z.size(0)
        device = z.device

        total_loss = torch.tensor(0.0, device=device)
        metrics = {}

        for level in range(1, self.max_level + 1):
            # P2 FIX: Vectorized prefix computation using core module
            # Instead of Python loop: [self._get_prefix(int(idx), level) for idx in batch_indices]
            prefixes = TERNARY.prefix(batch_indices, level)

            unique_prefixes = torch.unique(prefixes)
            level_loss = torch.tensor(0.0, device=device)

            for prefix in unique_prefixes:
                mask = prefixes == prefix
                if mask.sum() < 2:
                    continue

                cluster_points = z_hyp[mask]

                # Compute centroid
                centroid = self._compute_frechet_mean(cluster_points)

                # Distance to centroid (using unified implementation)
                distances = poincare_distance(
                    cluster_points,
                    centroid.unsqueeze(0).expand_as(cluster_points),
                    c=self.curvature,
                )

                level_loss = level_loss + distances.mean()

            # Weight by level
            weight = self.level_weights[level - 1]
            total_loss = total_loss + weight * level_loss / max(len(unique_prefixes), 1)
            metrics[f"centroid_loss_level_{level}"] = level_loss.item()

        metrics["total_centroid_loss"] = total_loss.item()
        return total_loss, metrics
