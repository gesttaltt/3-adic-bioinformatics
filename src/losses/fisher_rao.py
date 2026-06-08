# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Fisher-Rao Information Geometry Loss Functions.

Implements loss functions based on the Fisher-Rao metric, which measures
distances in the space of probability distributions.

Key Concepts:
- Fisher Information Matrix: Measures curvature of log-likelihood
- Fisher-Rao Distance: Geodesic distance on statistical manifold
- Natural Gradient: Gradient adjusted for information geometry

The Fisher-Rao metric is the unique (up to scale) Riemannian metric that
is invariant under reparametrization of the probability distribution.

For Gaussian distributions:
    d_FR(p, q)^2 = ||mu_p - mu_q||^2 / sigma^2 + 2 * log(sigma_q/sigma_p)^2

For VAEs, this encourages:
1. Smooth latent space structure
2. Meaningful distances between representations
3. Proper uncertainty quantification

References:
- Amari (1985): Differential-Geometrical Methods in Statistics
- Nielsen (2020): An Elementary Introduction to Information Geometry
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.base import LossComponent, LossResult


@dataclass
class FisherRaoConfig:
    """Configuration for Fisher-Rao loss."""

    distance_type: str = "gaussian"  # 'gaussian', 'categorical', 'mixed'
    use_empirical_fisher: bool = False
    regularization_weight: float = 0.1
    geodesic_weight: float = 1.0
    curvature_penalty_weight: float = 0.01
    min_variance: float = 1e-6
    max_variance: float = 10.0


class FisherRaoDistance(nn.Module):
    """Compute Fisher-Rao distance between probability distributions.

    For Gaussian distributions N(mu, sigma^2), the Fisher-Rao distance is:
        d_FR(p, q)^2 = (mu_p - mu_q)^2 / sigma^2 + 2 * (log(sigma_p) - log(sigma_q))^2

    This is the geodesic distance on the manifold of Gaussian distributions.
    """

    def __init__(
        self,
        distribution: str = "gaussian",
        diagonal_cov: bool = True,
    ):
        """Initialize Fisher-Rao distance.

        Args:
            distribution: Type of distribution ('gaussian', 'laplace', 'categorical')
            diagonal_cov: Whether to assume diagonal covariance
        """
        super().__init__()
        self.distribution = distribution
        self.diagonal_cov = diagonal_cov

    def forward(
        self,
        mu1: torch.Tensor,
        logvar1: torch.Tensor,
        mu2: torch.Tensor,
        logvar2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Fisher-Rao distance between two Gaussian distributions.

        Args:
            mu1: Mean of first distribution (batch, dim)
            logvar1: Log-variance of first distribution (batch, dim)
            mu2: Mean of second distribution (batch, dim)
            logvar2: Log-variance of second distribution (batch, dim)

        Returns:
            Distance tensor (batch,)
        """
        if self.distribution == "gaussian":
            return self._gaussian_distance(mu1, logvar1, mu2, logvar2)
        elif self.distribution == "laplace":
            return self._laplace_distance(mu1, logvar1, mu2, logvar2)
        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

    def _gaussian_distance(
        self,
        mu1: torch.Tensor,
        logvar1: torch.Tensor,
        mu2: torch.Tensor,
        logvar2: torch.Tensor,
    ) -> torch.Tensor:
        """Fisher-Rao distance for Gaussian distributions.

        For diagonal Gaussians, the Fisher information metric is:
            ds^2 = sum_i [(dmu_i)^2 / sigma_i^2 + 2 * (d log sigma_i)^2]

        The geodesic distance is:
            d^2 = sum_i [(mu1_i - mu2_i)^2 / sigma_ref^2 + 2*(log sigma1_i - log sigma2_i)^2]

        where sigma_ref is typically the geometric mean.
        """
        # Mean term: (mu1 - mu2)^2 / sigma^2
        # Use geometric mean of variances as reference
        logvar_ref = 0.5 * (logvar1 + logvar2)
        var_ref = torch.exp(logvar_ref).clamp(min=1e-8)

        mean_term = ((mu1 - mu2) ** 2) / var_ref

        # Variance term: 2 * (log sigma1 - log sigma2)^2
        # = 2 * (0.5 * logvar1 - 0.5 * logvar2)^2
        # = 0.5 * (logvar1 - logvar2)^2
        var_term = 0.5 * (logvar1 - logvar2) ** 2

        # Sum over dimensions
        distance_sq = (mean_term + var_term).sum(dim=-1)

        return torch.sqrt(distance_sq.clamp(min=1e-10))

    def _laplace_distance(
        self,
        mu1: torch.Tensor,
        logvar1: torch.Tensor,
        mu2: torch.Tensor,
        logvar2: torch.Tensor,
    ) -> torch.Tensor:
        """Fisher-Rao distance for Laplace distributions.

        For Laplace(mu, b), the Fisher metric is:
            ds^2 = (dmu)^2 / b^2 + 2 * (d log b)^2

        Using logvar as log(2*b^2), so b = sqrt(exp(logvar)/2).
        """
        # Scale parameter b from variance: var = 2*b^2, so b = sqrt(var/2)
        logb1 = 0.5 * (logvar1 - math.log(2))
        logb2 = 0.5 * (logvar2 - math.log(2))
        logb_ref = 0.5 * (logb1 + logb2)
        b_ref = torch.exp(logb_ref).clamp(min=1e-8)

        # Mean term
        mean_term = ((mu1 - mu2) ** 2) / (b_ref**2)

        # Scale term
        scale_term = 2 * (logb1 - logb2) ** 2

        distance_sq = (mean_term + scale_term).sum(dim=-1)

        return torch.sqrt(distance_sq.clamp(min=1e-10))


class FisherRaoLoss(LossComponent):
    """Fisher-Rao based loss for VAE training.

    This loss encourages the latent space to respect the information
    geometry of the underlying probability distributions.

    Components:
    1. Prior distance: d_FR(q(z|x), p(z)) - replaces standard KL
    2. Geodesic smoothness: Encourages smooth geodesics in latent space
    3. Curvature regularization: Penalizes high Fisher curvature
    """

    def __init__(
        self,
        config: FisherRaoConfig | None = None,
        weight: float = 1.0,
        name: str = "fisher_rao",
    ):
        """Initialize Fisher-Rao loss.

        Args:
            config: Configuration for Fisher-Rao loss
            weight: Weight for this loss component
            name: Name for logging
        """
        super().__init__(weight=weight, name=name)
        self.config = config or FisherRaoConfig()

        self.distance_fn = FisherRaoDistance(distribution=self.config.distance_type)

        # Prior parameters (standard normal)
        self.register_buffer("prior_mu", torch.tensor(0.0))
        self.register_buffer("prior_logvar", torch.tensor(0.0))

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        **kwargs,
    ) -> LossResult:
        """Compute Fisher-Rao loss.

        Args:
            outputs: Model outputs containing:
                - mu_A, logvar_A: Encoder distribution A
                - mu_B, logvar_B: Encoder distribution B
                - z_A, z_B: Sampled latent codes
            targets: Target values (unused for this loss)
            **kwargs: Additional arguments

        Returns:
            LossResult with Fisher-Rao loss and metrics
        """
        metrics = {}
        total_loss = torch.tensor(0.0, device=targets.device)

        # Process both VAEs if present
        for vae in ["A", "B"]:
            mu_key = f"mu_{vae}"
            logvar_key = f"logvar_{vae}"

            if mu_key not in outputs:
                continue

            mu = outputs[mu_key]
            logvar = outputs[logvar_key]
            batch_size, latent_dim = mu.shape

            # 1. Prior distance (replaces KL divergence)
            prior_mu = self.prior_mu.expand(batch_size, latent_dim)
            prior_logvar = self.prior_logvar.expand(batch_size, latent_dim)

            prior_distance = self.distance_fn(mu, logvar, prior_mu, prior_logvar)
            prior_loss = prior_distance.mean()

            metrics[f"prior_distance_{vae}"] = prior_loss.item()
            total_loss = total_loss + prior_loss

            # 2. Geodesic smoothness (within batch)
            if self.config.geodesic_weight > 0 and batch_size > 1:
                geodesic_loss = self._geodesic_smoothness(mu, logvar)
                metrics[f"geodesic_loss_{vae}"] = geodesic_loss.item()
                total_loss = total_loss + self.config.geodesic_weight * geodesic_loss

            # 3. Curvature regularization
            if self.config.curvature_penalty_weight > 0:
                curvature_loss = self._curvature_penalty(logvar)
                metrics[f"curvature_loss_{vae}"] = curvature_loss.item()
                total_loss = total_loss + self.config.curvature_penalty_weight * curvature_loss

        return LossResult(
            loss=total_loss,
            metrics=metrics,
            weight=self.weight,
        )

    def _geodesic_smoothness(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Encourage smooth geodesics between neighboring points.

        Samples pairs of nearby points and encourages the geodesic
        between them to be short (similar distributions close in latent space).
        """
        batch_size = mu.size(0)
        if batch_size < 2:
            return torch.tensor(0.0, device=mu.device)

        # Compute pairwise Fisher-Rao distances
        # For efficiency, use random subset of pairs
        n_pairs = min(batch_size * 2, 100)
        idx1 = torch.randint(0, batch_size, (n_pairs,), device=mu.device)
        idx2 = torch.randint(0, batch_size, (n_pairs,), device=mu.device)

        # Ensure different indices
        mask = idx1 != idx2
        idx1 = idx1[mask]
        idx2 = idx2[mask]

        if len(idx1) == 0:
            return torch.tensor(0.0, device=mu.device)

        # Compute distances
        distances = self.distance_fn(mu[idx1], logvar[idx1], mu[idx2], logvar[idx2])

        # Encourage moderate distances (not too large, not collapsed)
        # Target: distances should be around sqrt(latent_dim)
        target_dist = math.sqrt(mu.size(-1))
        smoothness_loss = ((distances - target_dist) ** 2).mean()

        return smoothness_loss

    def _curvature_penalty(self, logvar: torch.Tensor) -> torch.Tensor:
        """Penalize extreme variances (high curvature regions).

        Very small or large variances indicate regions of high Fisher
        curvature, which can lead to optimization difficulties.
        """
        # Penalize variance outside reasonable range
        var = torch.exp(logvar)

        # Log barrier penalty
        low_penalty = F.softplus(self.config.min_variance - var)
        high_penalty = F.softplus(var - self.config.max_variance)

        return (low_penalty + high_penalty).mean()


class FisherRaoKL(nn.Module):
    """Fisher-Rao aware KL divergence.

    Standard KL divergence for Gaussians:
        KL(q||p) = 0.5 * (var_q/var_p + (mu_p - mu_q)^2/var_p - 1 + log(var_p/var_q))

    Fisher-Rao weighted KL uses the Fisher metric to weight terms:
        KL_FR = integral of (grad log q - grad log p)^T F (grad log q - grad log p)

    This provides a more geometrically natural regularization.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        use_fisher_weighting: bool = True,
    ):
        """Initialize Fisher-Rao KL.

        Args:
            alpha: Interpolation between standard KL (0) and Fisher-Rao KL (1)
            use_fisher_weighting: Whether to use Fisher metric weighting
        """
        super().__init__()
        self.alpha = alpha
        self.use_fisher_weighting = use_fisher_weighting

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        prior_mu: torch.Tensor | None = None,
        prior_logvar: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Fisher-Rao weighted KL divergence.

        Args:
            mu: Posterior mean (batch, dim)
            logvar: Posterior log-variance (batch, dim)
            prior_mu: Prior mean (defaults to 0)
            prior_logvar: Prior log-variance (defaults to 0)

        Returns:
            KL divergence (batch,)
        """
        if prior_mu is None:
            prior_mu = torch.zeros_like(mu)
        if prior_logvar is None:
            prior_logvar = torch.zeros_like(logvar)

        var = torch.exp(logvar)
        prior_var = torch.exp(prior_logvar)

        # Standard KL
        kl_standard = 0.5 * (var / prior_var + (prior_mu - mu) ** 2 / prior_var - 1 + prior_logvar - logvar)

        if not self.use_fisher_weighting:
            return kl_standard.sum(dim=-1)

        # Fisher-Rao weighting
        # The Fisher information for Gaussian is diag(1/var, 2/var^2)
        # Weight mean terms by 1/var, variance terms by 2/var^2

        # Mean difference weighted by Fisher (1/var)
        mean_term = (prior_mu - mu) ** 2 / var

        # Variance difference weighted by Fisher (scales as 1/var^2)
        var_ratio = var / prior_var
        logvar_diff = prior_logvar - logvar
        var_term = 0.5 * (var_ratio - 1 - logvar_diff)

        # Fisher-Rao weighted
        kl_fisher_rao = mean_term + 2 * var_term

        # Interpolate
        kl_final = (1 - self.alpha) * kl_standard + self.alpha * kl_fisher_rao

        return kl_final.sum(dim=-1)


class NaturalGradientRegularizer(nn.Module):
    """Regularizer that encourages natural gradient-friendly geometry.

    This regularizer encourages the encoder to produce representations
    where natural gradients are well-defined and stable:
    1. Fisher information should be well-conditioned
    2. Variance estimates should be accurate
    3. Latent geometry should match data geometry
    """

    def __init__(
        self,
        condition_number_target: float = 10.0,
        min_eigenvalue: float = 0.01,
    ):
        """Initialize regularizer.

        Args:
            condition_number_target: Target condition number for Fisher
            min_eigenvalue: Minimum eigenvalue to encourage
        """
        super().__init__()
        self.condition_number_target = condition_number_target
        self.min_eigenvalue = min_eigenvalue

    def forward(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute natural gradient regularization.

        Args:
            mu: Posterior mean (batch, dim)
            logvar: Posterior log-variance (batch, dim)

        Returns:
            Regularization loss and metrics
        """
        var = torch.exp(logvar)
        batch_size, latent_dim = mu.shape

        # 1. Encourage similar variances (well-conditioned Fisher)
        # Condition number ~ max(var) / min(var)
        var_log_range = var.log().max(dim=-1)[0] - var.log().min(dim=-1)[0]
        target_log_range = math.log(self.condition_number_target)
        condition_loss = F.relu(var_log_range - target_log_range).mean()

        # 2. Encourage minimum variance (avoid Fisher singularity)
        min_var_loss = F.relu(self.min_eigenvalue - var).mean()

        # 3. Encourage diversity across batch (avoid collapse)
        if batch_size > 1:
            mu_centered = mu - mu.mean(dim=0, keepdim=True)
            cov = (mu_centered.T @ mu_centered) / (batch_size - 1)
            # Encourage covariance eigenvalues to be positive
            diversity_loss = -torch.logdet(cov + 0.01 * torch.eye(latent_dim, device=mu.device))
        else:
            diversity_loss = torch.tensor(0.0, device=mu.device)

        total_loss = condition_loss + min_var_loss + 0.1 * diversity_loss

        metrics = {
            "condition_loss": condition_loss.item(),
            "min_var_loss": min_var_loss.item(),
            "diversity_loss": diversity_loss.item() if batch_size > 1 else 0.0,
        }

        return total_loss, metrics


__all__ = [
    "FisherRaoConfig",
    "FisherRaoDistance",
    "FisherRaoLoss",
    "FisherRaoKL",
    "NaturalGradientRegularizer",
]
