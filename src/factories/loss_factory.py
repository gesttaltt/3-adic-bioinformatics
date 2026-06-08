# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Factory for creating hyperbolic loss components.

This module provides factory functions for instantiating loss modules
based on configuration, centralizing complex construction logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from src.losses.hyperbolic_prior import HomeostaticHyperbolicPrior
from src.losses.hyperbolic_recon import HomeostaticReconLoss, HyperbolicCentroidLoss
from src.losses.padic import PAdicRankingLossHyperbolic
from src.losses.radial_stratification import RadialStratificationLoss


@dataclass
class HyperbolicLossComponents:
    """Container for hyperbolic loss components.

    Attributes:
        ranking_loss: P-adic ranking loss (hyperbolic version)
        prior_A: Hyperbolic prior for VAE-A
        prior_B: Hyperbolic prior for VAE-B
        recon_A: Hyperbolic reconstruction loss for VAE-A
        recon_B: Hyperbolic reconstruction loss for VAE-B
        centroid_loss: Hyperbolic centroid clustering loss
        radial_stratification_A: Radial stratification for VAE-A
        radial_stratification_B: Radial stratification for VAE-B

        Flags:
        use_hyperbolic_prior: Whether prior is enabled
        use_hyperbolic_recon: Whether recon is enabled
        use_centroid_loss: Whether centroid is enabled
        use_radial_stratification: Whether stratification is enabled

        Weights:
        hyperbolic_recon_weight: Weight for recon loss
        centroid_loss_weight: Weight for centroid loss
        radial_stratification_weight: Weight for stratification loss

        Geometry:
        curvature: Hyperbolic curvature
        max_norm: Maximum Poincaré ball radius
    """

    # Loss modules (Optional)
    ranking_loss: nn.Module | None = None
    prior_A: nn.Module | None = None
    prior_B: nn.Module | None = None
    recon_A: nn.Module | None = None
    recon_B: nn.Module | None = None
    centroid_loss: nn.Module | None = None
    radial_stratification_A: nn.Module | None = None
    radial_stratification_B: nn.Module | None = None

    # Flags
    use_hyperbolic_prior: bool = False
    use_hyperbolic_recon: bool = False
    use_centroid_loss: bool = False
    use_radial_stratification: bool = False

    # Weights
    hyperbolic_recon_weight: float = 0.5
    centroid_loss_weight: float = 0.2
    radial_stratification_weight: float = 1.0

    # Geometry
    curvature: float = 2.0
    max_norm: float = 0.95


class HyperbolicLossFactory:
    """Factory for creating hyperbolic loss components from configuration.

    Example:
        >>> config = {"padic_losses": {"enable_ranking_loss_hyperbolic": True}}
        >>> factory = HyperbolicLossFactory()
        >>> components = factory.create_all(config, device="cuda")
        >>> if components.ranking_loss:
        ...     loss = components.ranking_loss(z_hyp, indices)
    """

    @staticmethod
    def create_ranking_loss(config: dict[str, Any]) -> PAdicRankingLossHyperbolic | None:
        """Create P-adic ranking loss (hyperbolic version).

        Args:
            config: Configuration dict with padic_losses.ranking_hyperbolic

        Returns:
            PAdicRankingLossHyperbolic or None if disabled
        """
        padic_config = config.get("padic_losses", {})
        if not padic_config.get("enable_ranking_loss_hyperbolic", False):
            return None

        hyp_config = padic_config.get("ranking_hyperbolic", {})
        return PAdicRankingLossHyperbolic(
            base_margin=hyp_config.get("base_margin", 0.05),
            margin_scale=hyp_config.get("margin_scale", 0.15),
            n_triplets=hyp_config.get("n_triplets", 500),
            hard_negative_ratio=hyp_config.get("hard_negative_ratio", 0.5),
            curvature=hyp_config.get("curvature", 2.0),
            radial_weight=hyp_config.get("radial_weight", 0.4),
            max_norm=hyp_config.get("max_norm", 0.95),
        )

    @staticmethod
    def create_hyperbolic_priors(
        config: dict[str, Any],
        device: str,
    ) -> tuple[HomeostaticHyperbolicPrior | None, HomeostaticHyperbolicPrior | None]:
        """Create hyperbolic prior modules for VAE-A and VAE-B.

        Args:
            config: Configuration dict with padic_losses.hyperbolic_v10.prior
            device: Device to place modules on

        Returns:
            Tuple of (prior_A, prior_B) or (None, None) if disabled
        """
        padic_config = config.get("padic_losses", {})
        hyp_v10 = padic_config.get("hyperbolic_v10", {})

        if not hyp_v10.get("use_hyperbolic_prior", False):
            return None, None

        prior_config = hyp_v10.get("prior", {})

        def create_prior():
            return HomeostaticHyperbolicPrior(
                latent_dim=prior_config.get("latent_dim", 16),
                curvature=prior_config.get("curvature", 2.2),
                prior_sigma=prior_config.get("prior_sigma", 1.0),
                max_norm=prior_config.get("max_norm", 0.95),
                sigma_min=prior_config.get("sigma_min", 0.8),
                sigma_max=prior_config.get("sigma_max", 1.2),
                curvature_min=prior_config.get("curvature_min", 2.0),
                curvature_max=prior_config.get("curvature_max", 2.5),
                adaptation_rate=prior_config.get("adaptation_rate", 0.005),
                ema_alpha=prior_config.get("ema_alpha", 0.05),
                kl_target=prior_config.get("kl_target", 50.0),
                target_radius=prior_config.get("target_radius", 0.5),
            ).to(device)

        return create_prior(), create_prior()

    @staticmethod
    def create_hyperbolic_recons(
        config: dict[str, Any],
        device: str,
    ) -> tuple[HomeostaticReconLoss | None, HomeostaticReconLoss | None, float]:
        """Create hyperbolic reconstruction loss modules.

        Args:
            config: Configuration dict
            device: Device to place modules on

        Returns:
            Tuple of (recon_A, recon_B, weight) or (None, None, 0) if disabled
        """
        padic_config = config.get("padic_losses", {})
        hyp_v10 = padic_config.get("hyperbolic_v10", {})

        if not hyp_v10.get("use_hyperbolic_recon", False):
            return None, None, 0.0

        recon_config = hyp_v10.get("recon", {})

        def create_recon():
            return HomeostaticReconLoss(
                mode=recon_config.get("mode", "weighted_ce"),
                curvature=recon_config.get("curvature", 2.0),
                max_norm=recon_config.get("max_norm", 0.95),
                radius_weighting=recon_config.get("radius_weighting", True),
                radius_power=recon_config.get("radius_power", 2.0),
            ).to(device)

        weight = recon_config.get("weight", 0.5)
        return create_recon(), create_recon(), weight

    @staticmethod
    def create_centroid_loss(
        config: dict[str, Any],
        device: str,
    ) -> tuple[HyperbolicCentroidLoss | None, float]:
        """Create hyperbolic centroid clustering loss.

        Args:
            config: Configuration dict
            device: Device to place module on

        Returns:
            Tuple of (centroid_loss, weight) or (None, 0) if disabled
        """
        padic_config = config.get("padic_losses", {})
        hyp_v10 = padic_config.get("hyperbolic_v10", {})

        if not hyp_v10.get("use_centroid_loss", False):
            return None, 0.0

        centroid_config = hyp_v10.get("centroid", {})
        loss = HyperbolicCentroidLoss(
            max_level=centroid_config.get("max_level", 4),
            curvature=centroid_config.get("curvature", 2.0),
            max_norm=centroid_config.get("max_norm", 0.95),
        ).to(device)
        weight = centroid_config.get("weight", 0.2)
        return loss, weight

    @staticmethod
    def create_radial_stratification(
        config: dict[str, Any],
        device: str,
    ) -> tuple[RadialStratificationLoss | None, RadialStratificationLoss | None, float]:
        """Create radial stratification loss modules.

        Args:
            config: Configuration dict
            device: Device to place modules on

        Returns:
            Tuple of (strat_A, strat_B, weight) or (None, None, 0) if disabled
        """
        padic_config = config.get("padic_losses", {})
        hyp_v10 = padic_config.get("hyperbolic_v10", {})

        if not hyp_v10.get("use_radial_stratification", False):
            return None, None, 0.0

        strat_config = hyp_v10.get("radial_stratification", {})

        def create_strat():
            return RadialStratificationLoss(
                n_strata=strat_config.get("n_strata", 10),
                inner_radius=strat_config.get("inner_radius", 0.1),
                outer_radius=strat_config.get("outer_radius", 0.85),
                curvature=strat_config.get("curvature", 2.0),
            ).to(device)

        weight = strat_config.get("weight", 1.0)
        return create_strat(), create_strat(), weight

    def create_all(self, config: dict[str, Any], device: str = "cpu") -> HyperbolicLossComponents:
        """Create all hyperbolic loss components from configuration.

        Args:
            config: Full configuration dictionary
            device: Device to place modules on

        Returns:
            HyperbolicLossComponents with all created modules
        """
        # Extract geometry params
        padic_config = config.get("padic_losses", {})
        hyp_config = padic_config.get("ranking_hyperbolic", {})
        curvature = hyp_config.get("curvature", 2.0)
        max_norm = hyp_config.get("max_norm", 0.95)

        # Create components
        ranking_loss = self.create_ranking_loss(config)
        prior_A, prior_B = self.create_hyperbolic_priors(config, device)
        recon_A, recon_B, recon_weight = self.create_hyperbolic_recons(config, device)
        centroid_loss, centroid_weight = self.create_centroid_loss(config, device)
        strat_A, strat_B, strat_weight = self.create_radial_stratification(config, device)

        return HyperbolicLossComponents(
            ranking_loss=ranking_loss,
            prior_A=prior_A,
            prior_B=prior_B,
            recon_A=recon_A,
            recon_B=recon_B,
            centroid_loss=centroid_loss,
            radial_stratification_A=strat_A,
            radial_stratification_B=strat_B,
            use_hyperbolic_prior=prior_A is not None,
            use_hyperbolic_recon=recon_A is not None,
            use_centroid_loss=centroid_loss is not None,
            use_radial_stratification=strat_A is not None,
            hyperbolic_recon_weight=recon_weight,
            centroid_loss_weight=centroid_weight,
            radial_stratification_weight=strat_weight,
            curvature=curvature,
            max_norm=max_norm,
        )
