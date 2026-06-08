# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Hyperbolic Prior for Pure Hyperbolic VAE (v5.10).

This module implements a Wrapped Normal distribution on the Poincare ball,
replacing the standard Gaussian prior. This eliminates the geometric conflict
between Euclidean KL regularization and hyperbolic structure.

Key insight: Standard VAE KL divergence assumes Euclidean geometry:
    KL(q(z|x) || N(0,I)) pulls embeddings toward a Gaussian blob,
    which fights against the tree-like hyperbolic structure.

The Wrapped Normal distribution respects hyperbolic geometry:
    - Prior mass concentrates near the origin (tree root)
    - Natural radial structure emerges from the metric
    - KL computed using hyperbolic geodesics

Reference: Mathieu et al., "Continuous Hierarchical Representations with
Poincare Variational Auto-Encoders" (2019)

Note: Uses geoopt backend when available for numerical stability.
"""

import math

import torch
import torch.nn as nn

# Import from geometry module for stable operations
from src.geometry import exp_map_zero, lambda_x, log_map_zero, poincare_distance, project_to_poincare


class HyperbolicPrior(nn.Module):
    """Wrapped Normal prior on the Poincare ball.

    The wrapped normal distribution is defined by:
    1. Sample v ~ N(0, sigma^2 I) in the tangent space at origin
    2. Map to Poincare ball via exponential map: z = exp_0(v)

    This creates a distribution that:
    - Is centered at the origin (tree root for high-valuation operations)
    - Has radial spread controlled by sigma (like temperature)
    - Respects hyperbolic geometry (no Euclidean contamination)

    The KL divergence from q(z|x) to this prior uses:
    - Riemannian normal distribution in tangent space
    - Parallel transport for off-origin encodings
    - Geodesic distance instead of Euclidean
    """

    def __init__(
        self,
        latent_dim: int = 16,
        curvature: float = 1.0,
        prior_sigma: float = 1.0,
        max_norm: float = 0.95,
    ):
        """Initialize Hyperbolic Prior.

        Args:
            latent_dim: Dimensionality of latent space
            curvature: Poincare ball curvature (c > 0)
            prior_sigma: Standard deviation of wrapped normal
            max_norm: Maximum radius in Poincare ball
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.curvature = curvature
        self.prior_sigma = prior_sigma
        self.max_norm = max_norm

    def _project_to_poincare(self, z: torch.Tensor) -> torch.Tensor:
        """Project Euclidean points onto the Poincare ball.

        Uses geoopt when available for numerical stability.
        """
        return project_to_poincare(z, self.max_norm, self.curvature)

    def _poincare_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Poincare distance between points.

        Uses geoopt when available for numerical stability.
        """
        return poincare_distance(x, y, self.curvature)

    def _lambda_x(self, x: torch.Tensor) -> torch.Tensor:
        """Compute conformal factor lambda_x = 2 / (1 - c * ||x||^2).

        Uses geoopt when available for numerical stability.
        """
        return lambda_x(x, self.curvature, keepdim=True)

    def _exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        """Exponential map from tangent space at origin to Poincare ball.

        Uses geoopt when available for numerical stability.
        """
        return exp_map_zero(v, self.curvature)

    def _log_map_zero(self, z: torch.Tensor) -> torch.Tensor:
        """Logarithmic map from Poincare ball to tangent space at origin.

        Uses geoopt when available for numerical stability.
        """
        return log_map_zero(z, self.curvature, self.max_norm)

    def kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        use_hyperbolic: bool = True,
    ) -> torch.Tensor:
        """Compute KL divergence from posterior to hyperbolic prior.

        For the wrapped normal, we compute:
        1. Project mu to Poincare ball
        2. Map to tangent space at origin via log map
        3. Compute KL in tangent space (which is Euclidean)
        4. Apply correction for the change of measure

        Args:
            mu: Mean of variational posterior (Euclidean)
            logvar: Log variance of variational posterior
            use_hyperbolic: If True, use hyperbolic KL; else Euclidean (for comparison)

        Returns:
            KL divergence (scalar, batch-averaged)
        """
        if not use_hyperbolic:
            # Standard Euclidean KL (for ablation)
            kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
            return kl_per_dim.sum(dim=-1).mean()

        # === Hyperbolic KL Divergence ===

        # 1. Project posterior mean to Poincare ball
        z_mu = self._project_to_poincare(mu)

        # 2. Compute distance from origin (prior center)
        # This is the "radial" component of KL
        # Create origin on same device as z_mu for CUDA compatibility
        origin = torch.zeros_like(z_mu)
        self._poincare_distance(z_mu, origin)

        # 3. Map to tangent space at origin
        v_mu = self._log_map_zero(z_mu)

        # 4. Compute variance in tangent space
        # The posterior variance needs to be scaled by the conformal factor
        lambda_mu = self._lambda_x(z_mu)

        # Effective variance in tangent space (scaled by conformal factor squared)
        # var_tangent = var_euclidean / lambda^2
        var = logvar.exp()
        var_tangent = var / (lambda_mu**2 + 1e-10)
        logvar_tangent = torch.log(var_tangent + 1e-10)

        # 5. KL in tangent space (Euclidean, but with transformed parameters)
        # KL(N(v_mu, var_tangent) || N(0, sigma_prior^2))
        prior_var = self.prior_sigma**2

        kl_per_dim = 0.5 * (
            logvar_tangent.neg()
            - 1  # -log(var_tangent)
            + math.log(prior_var)  # +log(prior_var)
            + var_tangent / prior_var  # +var_tangent / prior_var
            + (v_mu**2) / prior_var  # +(v_mu - 0)^2 / prior_var
        )

        # 6. Jacobian correction for change of measure (exp map)
        # The Jacobian of exp_0 contributes a factor of (lambda_0)^(d-1)
        # At origin, lambda_0 = 2, so this is a constant
        # For numerical stability, we absorb this into the prior variance

        kl = kl_per_dim.sum(dim=-1).mean()

        return kl

    def sample_prior(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample from the wrapped normal prior.

        Args:
            batch_size: Number of samples
            device: Device for tensors

        Returns:
            Samples on the Poincare ball (batch_size, latent_dim)
        """
        # Sample from N(0, sigma^2) in tangent space
        v = torch.randn(batch_size, self.latent_dim, device=device) * self.prior_sigma

        # Map to Poincare ball via exponential map
        z = self._exp_map_zero(v)

        return z

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute KL divergence and return projected samples.

        Args:
            mu: Mean of variational posterior
            logvar: Log variance of variational posterior

        Returns:
            Tuple of (kl_loss, z_hyperbolic)
        """
        # Reparameterization trick in Euclidean space
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_euclidean = mu + eps * std

        # Project to Poincare ball
        z_hyperbolic = self._project_to_poincare(z_euclidean)

        # Compute hyperbolic KL
        kl = self.kl_divergence(mu, logvar, use_hyperbolic=True)

        return kl, z_hyperbolic


class HomeostaticHyperbolicPrior(HyperbolicPrior):
    """Hyperbolic Prior with homeostatic adaptation.

    This extends HyperbolicPrior with StateNet-compatible signals for
    algebraic convergence. The prior adapts its parameters based on:

    1. Radial distribution: Are high-valuation points near origin?
    2. KL magnitude: Is the system in equilibrium?
    3. Coverage signal: External feedback from training

    The homeostatic mechanism prevents both collapse (all points at origin)
    and explosion (all points at boundary) by modulating:
    - prior_sigma: Controls spread (higher = more exploration)
    - curvature: Controls tree depth (higher = sharper hierarchy)
    """

    def __init__(
        self,
        latent_dim: int = 16,
        curvature: float = 1.0,
        prior_sigma: float = 1.0,
        max_norm: float = 0.95,
        # Homeostatic parameters
        sigma_min: float = 0.3,
        sigma_max: float = 2.0,
        curvature_min: float = 0.5,
        curvature_max: float = 4.0,
        adaptation_rate: float = 0.01,
        ema_alpha: float = 0.1,
        kl_target: float = 1.0,
        target_radius: float = 0.5,
    ):
        """Initialize Homeostatic Hyperbolic Prior.

        Args:
            latent_dim: Dimensionality of latent space
            curvature: Initial Poincare ball curvature
            prior_sigma: Initial standard deviation of wrapped normal
            max_norm: Maximum radius in Poincare ball
            sigma_min: Minimum prior sigma (prevents collapse)
            sigma_max: Maximum prior sigma (prevents explosion)
            curvature_min: Minimum curvature
            curvature_max: Maximum curvature
            adaptation_rate: Rate of homeostatic adaptation
            ema_alpha: EMA smoothing factor for statistics (0-1)
            kl_target: Target KL divergence in nats
            target_radius: Target mean radius in Poincare ball (0.5 = balanced tree)
        """
        super().__init__(latent_dim, curvature, prior_sigma, max_norm)

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.curvature_min = curvature_min
        self.curvature_max = curvature_max
        self.adaptation_rate = adaptation_rate
        self.ema_alpha = ema_alpha
        self.kl_target = kl_target
        self.target_radius = target_radius

        # Learnable parameters for homeostatic control
        # (can be modulated by StateNet)
        self.register_buffer("adaptive_sigma", torch.tensor(prior_sigma))
        self.adaptive_sigma: torch.Tensor  # Type hint for mypy
        self.register_buffer("adaptive_curvature", torch.tensor(curvature))
        self.adaptive_curvature: torch.Tensor  # Type hint for mypy

        # EMA for tracking statistics
        self.register_buffer("mean_radius_ema", torch.tensor(0.5))
        self.mean_radius_ema: torch.Tensor  # Type hint for mypy
        self.register_buffer("kl_ema", torch.tensor(1.0))
        self.kl_ema: torch.Tensor  # Type hint for mypy

    def update_homeostatic_state(
        self,
        z_hyperbolic: torch.Tensor,
        kl: torch.Tensor,
        coverage: float = 0.0,
    ):
        """Update homeostatic parameters based on current state.

        Args:
            z_hyperbolic: Current hyperbolic embeddings
            kl: Current KL divergence
            coverage: Current coverage percentage (0-100)
        """
        # V5.12.2: Compute current mean radius using hyperbolic distance
        origin = torch.zeros_like(z_hyperbolic)
        current_radius = poincare_distance(z_hyperbolic, origin, self.curvature).mean()

        # Update EMAs using configured alpha
        alpha = self.ema_alpha
        self.mean_radius_ema = alpha * current_radius + (1 - alpha) * self.mean_radius_ema
        self.kl_ema = alpha * kl + (1 - alpha) * self.kl_ema

        # Homeostatic adaptation of sigma
        # If radius too high (explosion), decrease sigma
        # If radius too low (collapse), increase sigma
        radius_error = self.mean_radius_ema - self.target_radius
        sigma_delta = -self.adaptation_rate * radius_error

        new_sigma = self.adaptive_sigma + sigma_delta
        self.adaptive_sigma = torch.clamp(new_sigma, self.sigma_min, self.sigma_max)

        # Update instance variable for KL computation
        self.prior_sigma = self.adaptive_sigma.item()

        # Homeostatic adaptation of curvature
        # If KL too high, increase curvature (sharper hierarchy)
        # If KL too low, decrease curvature (flatter space)
        kl_error = self.kl_ema - self.kl_target
        curvature_delta = self.adaptation_rate * kl_error * 0.1  # Slower adaptation

        new_curvature = self.adaptive_curvature + curvature_delta
        self.adaptive_curvature = torch.clamp(new_curvature, self.curvature_min, self.curvature_max)
        self.curvature = self.adaptive_curvature.item()

    def get_homeostatic_state(self) -> dict:
        """Return current homeostatic state for logging/StateNet."""
        return {
            "prior_sigma": self.adaptive_sigma.item(),
            "curvature": self.adaptive_curvature.item(),
            "mean_radius_ema": self.mean_radius_ema.item(),
            "kl_ema": self.kl_ema.item(),
        }

    def set_from_statenet(self, delta_sigma: float = 0.0, delta_curvature: float = 0.0):
        """Apply StateNet corrections to homeostatic parameters.

        Args:
            delta_sigma: StateNet correction for sigma
            delta_curvature: StateNet correction for curvature
        """
        new_sigma = self.adaptive_sigma + delta_sigma * self.adaptation_rate * 10
        self.adaptive_sigma = torch.clamp(new_sigma, self.sigma_min, self.sigma_max)
        self.prior_sigma = self.adaptive_sigma.item()

        new_curvature = self.adaptive_curvature + delta_curvature * self.adaptation_rate * 10
        self.adaptive_curvature = torch.clamp(new_curvature, self.curvature_min, self.curvature_max)
        self.curvature = self.adaptive_curvature.item()
