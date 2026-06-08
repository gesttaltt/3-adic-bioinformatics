# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Loss functions for Dual-Neural VAE.

This module contains all loss computation separated from model architecture:
- Reconstruction loss (cross-entropy)
- KL divergence with free bits
- Entropy regularization
- Repulsion loss
- Entropy alignment
- Gradient normalization
- p-Adic metric loss (Phase 1A)
- p-Adic norm loss (Phase 1B)

Single responsibility: Loss computation only.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hyperbolic_prior import HomeostaticHyperbolicPrior, HyperbolicPrior
from .hyperbolic_recon import HomeostaticReconLoss, HyperbolicCentroidLoss, HyperbolicReconLoss
from .padic import (
    PAdicMetricLoss,
    PAdicNormLoss,
    PAdicRankingLoss,
    PAdicRankingLossHyperbolic,
    PAdicRankingLossV2,
)


class ReconstructionLoss(nn.Module):
    """Cross-entropy reconstruction loss for ternary operations."""

    def forward(self, logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss.

        Args:
            logits: Model logits (batch_size, 9, 3)
            x: Input data (batch_size, 9) with values in {-1, 0, 1}

        Returns:
            Reconstruction loss (scalar)
        """
        batch_size = x.size(0)
        x_classes = (x + 1).long()  # Convert {-1, 0, 1} to {0, 1, 2}

        loss = F.cross_entropy(logits.view(-1, 3), x_classes.view(-1), reduction="sum") / batch_size

        return loss


class KLDivergenceLoss(nn.Module):
    """KL divergence with optional free bits.

    Free bits allows the first 'free_bits' nats of KL per dimension to be
    ignored, preventing posterior collapse while still regularizing.
    """

    def __init__(self, free_bits: float = 0.0):
        """Initialize KL divergence loss.

        Args:
            free_bits: Minimum KL per dimension (in nats) before penalty applies
        """
        super().__init__()
        self.free_bits = free_bits

    def forward(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence KL(q(z|x) || p(z)).

        Args:
            mu: Mean of variational posterior
            logvar: Log variance of variational posterior

        Returns:
            KL divergence (scalar)
        """
        # Compute KL per dimension: -0.5 * (1 + logvar - mu^2 - exp(logvar))
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        if self.free_bits > 0:
            # Apply free bits: only penalize KL above threshold
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)

        # Sum over dimensions, average over batch
        kl = torch.sum(kl_per_dim) / mu.size(0)
        return kl


class EntropyRegularization(nn.Module):
    """Entropy regularization for output distribution."""

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute negative entropy of output distribution.

        Args:
            logits: Model logits (batch_size, 9, 3)

        Returns:
            Negative entropy (lower means more entropy/diversity)
        """
        probs = F.softmax(logits, dim=-1).mean(dim=0)  # Average over batch
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        return -entropy  # Return negative entropy as loss


class RepulsionLoss(nn.Module):
    """Repulsion loss to encourage diversity in latent space."""

    def __init__(self, sigma: float = 0.5):
        """Initialize repulsion loss.

        Args:
            sigma: Bandwidth for RBF kernel
        """
        super().__init__()
        self.sigma = sigma

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute repulsion loss.

        Args:
            z: Latent codes (batch_size, latent_dim)

        Returns:
            Repulsion loss (scalar)
        """
        if z.size(0) < 2:
            return torch.tensor(0.0, device=z.device)

        # Pairwise distances
        dists = torch.cdist(z, z, p=2)

        # Mask diagonal (self-distances)
        mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)

        # RBF kernel: penalize points that are too close
        repulsion = torch.exp(-(dists[mask] ** 2) / (self.sigma**2)).mean()

        return repulsion


class DualVAELoss(nn.Module):
    """Complete loss for Dual-Neural VAE system.

    Combines:
    - Reconstruction losses for both VAEs
    - KL divergences with optional free bits
    - Entropy regularization for VAE-B
    - Repulsion loss for VAE-B
    - Entropy alignment between VAEs
    - Gradient normalization
    - p-Adic metric loss (Phase 1A) - aligns latent geometry to 3-adic metric
    - p-Adic norm loss (Phase 1B) - enforces MSB/LSB hierarchy
    """

    def __init__(
        self,
        free_bits: float = 0.0,
        repulsion_sigma: float = 0.5,
        padic_config: dict[str, Any] | None = None,
    ):
        """Initialize Dual VAE loss.

        Args:
            free_bits: Free bits per dimension for KL (0.0 = disabled)
            repulsion_sigma: Bandwidth for repulsion loss
            padic_config: Configuration for p-adic losses (from config['padic_losses'])
        """
        super().__init__()
        self.reconstruction_loss = ReconstructionLoss()
        self.kl_loss = KLDivergenceLoss(free_bits)
        self.entropy_loss = EntropyRegularization()
        self.repulsion_loss = RepulsionLoss(repulsion_sigma)

        # p-Adic losses (Phase 1A/1B from implement.md)
        self.padic_config = padic_config or {}
        self.enable_metric_loss = self.padic_config.get("enable_metric_loss", False)
        self.enable_ranking_loss = self.padic_config.get("enable_ranking_loss", False)
        self.enable_norm_loss = self.padic_config.get("enable_norm_loss", False)

        # Phase 1A: Metric loss (MSE-based, for small scale mismatches)
        if self.enable_metric_loss:
            self.padic_metric_loss = PAdicMetricLoss(
                scale=self.padic_config.get("metric_loss_scale", 1.0),
                n_pairs=self.padic_config.get("metric_n_pairs", 1000),
            )
            self.metric_loss_weight = self.padic_config.get("metric_loss_weight", 0.1)

        # Phase 1A-alt: Ranking loss (triplet-based, for large scale mismatches)
        if self.enable_ranking_loss:
            self.padic_ranking_loss = PAdicRankingLoss(
                margin=self.padic_config.get("ranking_margin", 0.1),
                n_triplets=self.padic_config.get("ranking_n_triplets", 500),
            )
            self.ranking_loss_weight = self.padic_config.get("ranking_loss_weight", 0.5)

        # Phase 1B: Norm loss
        if self.enable_norm_loss:
            self.padic_norm_loss = PAdicNormLoss(latent_dim=16)
            self.norm_loss_weight = self.padic_config.get("norm_loss_weight", 0.05)

        # v5.8: PAdicRankingLossV2 (hard negative mining + hierarchical margin)
        self.enable_ranking_loss_v2 = self.padic_config.get("enable_ranking_loss_v2", False)
        if self.enable_ranking_loss_v2:
            v2_config = self.padic_config.get("ranking_v2", {})
            self.padic_ranking_loss_v2 = PAdicRankingLossV2(
                base_margin=v2_config.get("base_margin", 0.05),
                margin_scale=v2_config.get("margin_scale", 0.15),
                n_triplets=v2_config.get("n_triplets", 500),
                hard_negative_ratio=v2_config.get("hard_negative_ratio", 0.5),
                semi_hard=v2_config.get("semi_hard", True),
            )
            self.ranking_v2_weight = self.padic_config.get("ranking_v2_weight", 0.5)

        # v5.9: PAdicRankingLossHyperbolic (Poincare distance + radial hierarchy)
        self.enable_ranking_loss_hyperbolic = self.padic_config.get("enable_ranking_loss_hyperbolic", False)
        if self.enable_ranking_loss_hyperbolic:
            hyp_config = self.padic_config.get("ranking_hyperbolic", {})
            self.padic_ranking_loss_hyperbolic = PAdicRankingLossHyperbolic(
                base_margin=hyp_config.get("base_margin", 0.05),
                margin_scale=hyp_config.get("margin_scale", 0.15),
                n_triplets=hyp_config.get("n_triplets", 500),
                hard_negative_ratio=hyp_config.get("hard_negative_ratio", 0.5),
                curvature=hyp_config.get("curvature", 1.0),
                radial_weight=hyp_config.get("radial_weight", 0.1),
                max_norm=hyp_config.get("max_norm", 0.95),
            )
            self.ranking_hyperbolic_weight = hyp_config.get("weight", 0.5)

        # v5.10: Pure Hyperbolic Geometry (modular replacements)
        hyp_config_v10 = self.padic_config.get("hyperbolic_v10", {})

        # Hyperbolic Prior (replaces Gaussian KL)
        self.use_hyperbolic_prior = hyp_config_v10.get("use_hyperbolic_prior", False)
        if self.use_hyperbolic_prior:
            prior_config = hyp_config_v10.get("prior", {})
            use_homeostatic = prior_config.get("homeostatic", True)
            PriorClass = HomeostaticHyperbolicPrior if use_homeostatic else HyperbolicPrior
            self.hyperbolic_prior_A = PriorClass(
                latent_dim=prior_config.get("latent_dim", 16),
                curvature=prior_config.get("curvature", 2.0),
                prior_sigma=prior_config.get("prior_sigma", 1.0),
                max_norm=prior_config.get("max_norm", 0.95),
            )
            self.hyperbolic_prior_B = PriorClass(
                latent_dim=prior_config.get("latent_dim", 16),
                curvature=prior_config.get("curvature", 2.0),
                prior_sigma=prior_config.get("prior_sigma", 1.0),
                max_norm=prior_config.get("max_norm", 0.95),
            )

        # Hyperbolic Reconstruction Loss (replaces or augments MSE)
        self.use_hyperbolic_recon = hyp_config_v10.get("use_hyperbolic_recon", False)
        if self.use_hyperbolic_recon:
            recon_config = hyp_config_v10.get("recon", {})
            use_homeostatic = recon_config.get("homeostatic", True)
            ReconClass = HomeostaticReconLoss if use_homeostatic else HyperbolicReconLoss
            self.hyperbolic_recon_A = ReconClass(
                mode=recon_config.get("mode", "hybrid"),
                curvature=recon_config.get("curvature", 2.0),
                max_norm=recon_config.get("max_norm", 0.95),
                geodesic_weight=recon_config.get("geodesic_weight", 0.3),
                radius_weighting=recon_config.get("radius_weighting", True),
                radius_power=recon_config.get("radius_power", 2.0),
            )
            self.hyperbolic_recon_B = ReconClass(
                mode=recon_config.get("mode", "hybrid"),
                curvature=recon_config.get("curvature", 2.0),
                max_norm=recon_config.get("max_norm", 0.95),
                geodesic_weight=recon_config.get("geodesic_weight", 0.3),
                radius_weighting=recon_config.get("radius_weighting", True),
                radius_power=recon_config.get("radius_power", 2.0),
            )
            self.hyperbolic_recon_weight = recon_config.get("weight", 0.5)

        # Hyperbolic Centroid Loss (tree structure enforcement)
        self.use_centroid_loss = hyp_config_v10.get("use_centroid_loss", False)
        if self.use_centroid_loss:
            centroid_config = hyp_config_v10.get("centroid", {})
            self.centroid_loss = HyperbolicCentroidLoss(
                max_level=centroid_config.get("max_level", 4),
                curvature=centroid_config.get("curvature", 2.0),
                max_norm=centroid_config.get("max_norm", 0.95),
            )
            self.centroid_loss_weight = centroid_config.get("weight", 0.2)

    def forward(
        self,
        x: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        lambda1: float,
        lambda2: float,
        lambda3: float,
        entropy_weight_B: float,
        repulsion_weight_B: float,
        grad_norm_A_ema: torch.Tensor,
        grad_norm_B_ema: torch.Tensor,
        gradient_balance: bool,
        training: bool,
        batch_indices: torch.Tensor | None = None,
        ranking_weight_override: float | None = None,
    ) -> dict[str, Any]:
        """Compute complete Dual VAE loss.

        Args:
            x: Input data (batch_size, 9)
            outputs: Model outputs dict
            lambda1: Weight for VAE-A loss
            lambda2: Weight for VAE-B loss
            lambda3: Weight for entropy alignment
            entropy_weight_B: Weight for VAE-B entropy regularization
            repulsion_weight_B: Weight for VAE-B repulsion loss
            grad_norm_A_ema: EMA of VAE-A gradient norm
            grad_norm_B_ema: EMA of VAE-B gradient norm
            gradient_balance: Whether to apply gradient balancing
            training: Whether in training mode
            batch_indices: Operation indices for p-adic losses (batch_size,)
            ranking_weight_override: Optional weight override for continuous feedback

        Returns:
            Dictionary of losses and metrics
        """
        # Reconstruction losses
        ce_A = self.reconstruction_loss(outputs["logits_A"], x)
        ce_B = self.reconstruction_loss(outputs["logits_B"], x)

        # KL divergences
        kl_A = self.kl_loss(outputs["mu_A"], outputs["logvar_A"])
        kl_B = self.kl_loss(outputs["mu_B"], outputs["logvar_B"])

        # VAE-A loss (simple β-VAE)
        loss_A = ce_A + outputs["beta_A"] * kl_A

        # VAE-B losses
        entropy_B = -self.entropy_loss(outputs["logits_B"])  # Negate to get actual entropy
        entropy_loss_B = -entropy_B  # Loss term (negative entropy)
        repulsion_B = self.repulsion_loss(outputs["z_B"])

        loss_B = ce_B + outputs["beta_B"] * kl_B + entropy_weight_B * entropy_loss_B + repulsion_weight_B * repulsion_B

        # Gradient normalization
        if gradient_balance and training:
            with torch.no_grad():
                grad_scale_A = grad_norm_B_ema / (grad_norm_A_ema + 1e-8)
                grad_scale_B = grad_norm_A_ema / (grad_norm_B_ema + 1e-8)
                grad_scale_A = torch.clamp(grad_scale_A, 0.5, 2.0)
                grad_scale_B = torch.clamp(grad_scale_B, 0.5, 2.0)
        else:
            grad_scale_A = 1.0
            grad_scale_B = 1.0

        # Entropy alignment
        entropy_align = torch.abs(outputs["H_A"] - outputs["H_B"])

        # Total loss (base)
        total_loss = lambda1 * loss_A * grad_scale_A + lambda2 * loss_B * grad_scale_B + lambda3 * entropy_align

        # P0 FIX: Use Python floats instead of GPU tensor zeros for disabled modules.
        # Only allocate tensors when modules are actually enabled and called.
        # This saves 14 GPU allocations per batch (1078/epoch) when modules disabled.

        # p-Adic losses (Phase 1A/1B from implement.md) - lazy init
        padic_metric_A = 0.0
        padic_metric_B = 0.0
        padic_ranking_A = 0.0
        padic_ranking_B = 0.0
        padic_norm_A = 0.0
        padic_norm_B = 0.0

        # v5.8/v5.9 p-Adic loss defaults - lazy init
        padic_ranking_v2_A = 0.0
        padic_ranking_v2_B = 0.0
        metrics_v2_A = {
            "hard_ratio": 0.0,
            "violations": 0,
            "mean_margin": 0.0,
            "total_triplets": 0,
        }
        metrics_v2_B = {
            "hard_ratio": 0.0,
            "violations": 0,
            "mean_margin": 0.0,
            "total_triplets": 0,
        }
        padic_hyp_A = 0.0
        padic_hyp_B = 0.0
        metrics_hyp_A = {
            "hard_ratio": 0.0,
            "violations": 0,
            "poincare_dist_mean": 0.0,
            "radial_loss": 0.0,
            "ranking_loss": 0.0,
        }
        metrics_hyp_B = {
            "hard_ratio": 0.0,
            "violations": 0,
            "poincare_dist_mean": 0.0,
            "radial_loss": 0.0,
            "ranking_loss": 0.0,
        }
        hyp_weight = 0.0

        if batch_indices is not None:
            # Phase 1A: p-Adic Metric Loss (MSE-based)
            if self.enable_metric_loss:
                padic_metric_A = self.padic_metric_loss(outputs["z_A"], batch_indices)
                padic_metric_B = self.padic_metric_loss(outputs["z_B"], batch_indices)
                total_loss = total_loss + self.metric_loss_weight * (padic_metric_A + padic_metric_B)

            # Phase 1A-alt: p-Adic Ranking Loss (triplet-based, better for scale mismatch)
            if self.enable_ranking_loss:
                padic_ranking_A = self.padic_ranking_loss(outputs["z_A"], batch_indices)
                padic_ranking_B = self.padic_ranking_loss(outputs["z_B"], batch_indices)
                total_loss = total_loss + self.ranking_loss_weight * (padic_ranking_A + padic_ranking_B)

            # Phase 1B: p-Adic Norm Loss
            if self.enable_norm_loss:
                padic_norm_A = self.padic_norm_loss(outputs["z_A"], batch_indices)
                padic_norm_B = self.padic_norm_loss(outputs["z_B"], batch_indices)
                total_loss = total_loss + self.norm_loss_weight * (padic_norm_A + padic_norm_B)

            # v5.8: PAdicRankingLossV2 (hard negative mining + hierarchical margin)
            if self.enable_ranking_loss_v2:
                padic_ranking_v2_A, metrics_v2_A = self.padic_ranking_loss_v2(outputs["z_A"], batch_indices)
                padic_ranking_v2_B, metrics_v2_B = self.padic_ranking_loss_v2(outputs["z_B"], batch_indices)
                total_loss = total_loss + self.ranking_v2_weight * (padic_ranking_v2_A + padic_ranking_v2_B)

            # v5.9: PAdicRankingLossHyperbolic (Poincare distance + radial hierarchy)
            if self.enable_ranking_loss_hyperbolic:
                # Use override weight if provided (for continuous feedback)
                hyp_weight = (
                    ranking_weight_override if ranking_weight_override is not None else self.ranking_hyperbolic_weight
                )
                padic_hyp_A, metrics_hyp_A = self.padic_ranking_loss_hyperbolic(outputs["z_A"], batch_indices)
                padic_hyp_B, metrics_hyp_B = self.padic_ranking_loss_hyperbolic(outputs["z_B"], batch_indices)
                total_loss = total_loss + hyp_weight * (padic_hyp_A + padic_hyp_B)

        # v5.10: Hyperbolic metrics initialization - lazy init
        hyp_v10_metrics = {}
        hyp_kl_A = 0.0
        hyp_kl_B = 0.0

        # v5.10: Hyperbolic Centroid Loss (tree structure) - requires batch_indices
        if batch_indices is not None and self.use_centroid_loss:
            centroid_loss_A, centroid_metrics_A = self.centroid_loss(outputs["z_A"], batch_indices)
            centroid_loss_B, centroid_metrics_B = self.centroid_loss(outputs["z_B"], batch_indices)
            total_loss = total_loss + self.centroid_loss_weight * (centroid_loss_A + centroid_loss_B)
            hyp_v10_metrics["centroid_loss_A"] = centroid_loss_A.item()
            hyp_v10_metrics["centroid_loss_B"] = centroid_loss_B.item()

        if self.use_hyperbolic_prior:
            hyp_kl_A, z_hyp_A = self.hyperbolic_prior_A(outputs["mu_A"], outputs["logvar_A"])
            hyp_kl_B, z_hyp_B = self.hyperbolic_prior_B(outputs["mu_B"], outputs["logvar_B"])

            # Replace Euclidean KL in loss_A and loss_B
            # Note: This modifies the already-computed total_loss
            kl_replacement_A = outputs["beta_A"] * (hyp_kl_A - kl_A)
            kl_replacement_B = outputs["beta_B"] * (hyp_kl_B - kl_B)
            total_loss = total_loss + lambda1 * kl_replacement_A + lambda2 * kl_replacement_B

            hyp_v10_metrics["hyp_kl_A"] = hyp_kl_A.item()
            hyp_v10_metrics["hyp_kl_B"] = hyp_kl_B.item()

            # Update homeostatic state if using homeostatic prior
            if hasattr(self.hyperbolic_prior_A, "update_homeostatic_state"):
                self.hyperbolic_prior_A.update_homeostatic_state(z_hyp_A, hyp_kl_A)
                self.hyperbolic_prior_B.update_homeostatic_state(z_hyp_B, hyp_kl_B)
                hyp_v10_metrics.update(
                    {
                        "prior_sigma_A": self.hyperbolic_prior_A.adaptive_sigma.item(),
                        "prior_sigma_B": self.hyperbolic_prior_B.adaptive_sigma.item(),
                        "prior_curvature_A": self.hyperbolic_prior_A.adaptive_curvature.item(),
                        "prior_curvature_B": self.hyperbolic_prior_B.adaptive_curvature.item(),
                    }
                )

        # v5.10: Hyperbolic Reconstruction Loss - lazy init
        hyp_recon_A = 0.0
        hyp_recon_B = 0.0

        if self.use_hyperbolic_recon:
            hyp_recon_A, recon_metrics_A = self.hyperbolic_recon_A(outputs["logits_A"], x, outputs["z_A"])
            hyp_recon_B, recon_metrics_B = self.hyperbolic_recon_B(outputs["logits_B"], x, outputs["z_B"])
            total_loss = total_loss + self.hyperbolic_recon_weight * (hyp_recon_A + hyp_recon_B)

            hyp_v10_metrics["hyp_recon_A"] = hyp_recon_A.item()
            hyp_v10_metrics["hyp_recon_B"] = hyp_recon_B.item()
            hyp_v10_metrics["mean_radius_A"] = recon_metrics_A.get("mean_radius", 0.0)
            hyp_v10_metrics["mean_radius_B"] = recon_metrics_B.get("mean_radius", 0.0)

            # Update homeostatic state if using homeostatic recon
            if hasattr(self.hyperbolic_recon_A, "update_homeostatic_state"):
                self.hyperbolic_recon_A.update_homeostatic_state(hyp_recon_A, 0.0)
                self.hyperbolic_recon_B.update_homeostatic_state(hyp_recon_B, 0.0)

        return {
            "loss": total_loss,
            "ce_A": ce_A,
            "ce_B": ce_B,
            "kl_A": kl_A,
            "kl_B": kl_B,
            "loss_A": loss_A,
            "loss_B": loss_B,
            "entropy_B": entropy_B,
            "repulsion_B": repulsion_B,
            "entropy_align": entropy_align,
            "H_A": outputs["H_A"],
            "H_B": outputs["H_B"],
            "grad_scale_A": (grad_scale_A.item() if isinstance(grad_scale_A, torch.Tensor) else grad_scale_A),
            "grad_scale_B": (grad_scale_B.item() if isinstance(grad_scale_B, torch.Tensor) else grad_scale_B),
            "lambda1": lambda1,
            "lambda2": lambda2,
            "lambda3": lambda3,
            # p-Adic losses (Phase 1A/1B)
            "padic_metric_A": (padic_metric_A.item() if torch.is_tensor(padic_metric_A) else padic_metric_A),
            "padic_metric_B": (padic_metric_B.item() if torch.is_tensor(padic_metric_B) else padic_metric_B),
            "padic_ranking_A": (padic_ranking_A.item() if torch.is_tensor(padic_ranking_A) else padic_ranking_A),
            "padic_ranking_B": (padic_ranking_B.item() if torch.is_tensor(padic_ranking_B) else padic_ranking_B),
            "padic_norm_A": (padic_norm_A.item() if torch.is_tensor(padic_norm_A) else padic_norm_A),
            "padic_norm_B": (padic_norm_B.item() if torch.is_tensor(padic_norm_B) else padic_norm_B),
            # v5.8: PAdicRankingLossV2 metrics
            "padic_ranking_v2_A": (
                padic_ranking_v2_A.item() if torch.is_tensor(padic_ranking_v2_A) else padic_ranking_v2_A
            ),
            "padic_ranking_v2_B": (
                padic_ranking_v2_B.item() if torch.is_tensor(padic_ranking_v2_B) else padic_ranking_v2_B
            ),
            "ranking_v2_hard_ratio": (metrics_v2_A.get("hard_ratio", 0) + metrics_v2_B.get("hard_ratio", 0)) / 2,
            "ranking_v2_violations": metrics_v2_A.get("violations", 0) + metrics_v2_B.get("violations", 0),
            # v5.9: Hyperbolic p-Adic metrics
            "padic_hyp_A": (padic_hyp_A.item() if torch.is_tensor(padic_hyp_A) else padic_hyp_A),
            "padic_hyp_B": (padic_hyp_B.item() if torch.is_tensor(padic_hyp_B) else padic_hyp_B),
            "hyp_ranking_weight": hyp_weight,
            "hyp_hard_ratio": (metrics_hyp_A.get("hard_ratio", 0) + metrics_hyp_B.get("hard_ratio", 0)) / 2,
            "hyp_violations": metrics_hyp_A.get("violations", 0) + metrics_hyp_B.get("violations", 0),
            "hyp_poincare_dist_mean": (
                metrics_hyp_A.get("poincare_dist_mean", 0) + metrics_hyp_B.get("poincare_dist_mean", 0)
            )
            / 2,
            "hyp_radial_loss": (metrics_hyp_A.get("radial_loss", 0) + metrics_hyp_B.get("radial_loss", 0)) / 2,
            "hyp_ranking_loss": (metrics_hyp_A.get("ranking_loss", 0) + metrics_hyp_B.get("ranking_loss", 0)) / 2,
            # v5.10: Pure Hyperbolic metrics
            "hyp_kl_A": (hyp_kl_A.item() if torch.is_tensor(hyp_kl_A) else hyp_kl_A),
            "hyp_kl_B": (hyp_kl_B.item() if torch.is_tensor(hyp_kl_B) else hyp_kl_B),
            "hyp_recon_A": (hyp_recon_A.item() if torch.is_tensor(hyp_recon_A) else hyp_recon_A),
            "hyp_recon_B": (hyp_recon_B.item() if torch.is_tensor(hyp_recon_B) else hyp_recon_B),
            **hyp_v10_metrics,  # Spread all v5.10 homeostatic metrics
        }
