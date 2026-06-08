# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Loss components with LossComponent interface.

This module provides LossComponent wrappers for all loss types,
enabling them to be used with LossRegistry.

Each component:
    - Inherits from LossComponent or DualVAELossComponent
    - Has a consistent forward() signature
    - Returns LossResult with relevant metrics
    - Is independently testable
"""

from typing import Any

import torch
import torch.nn.functional as F

from ..geometry import poincare_distance
from .base import DualVAELossComponent, LossComponent, LossResult
from .padic import PAdicRankingLossHyperbolic, PAdicRankingLossV2


class ReconstructionLossComponent(LossComponent):
    """Cross-entropy reconstruction loss for ternary operations.

    Computes separate losses for VAE-A and VAE-B, returns combined result.
    """

    def __init__(self, weight: float = 1.0):
        """Initialize reconstruction loss.

        Args:
            weight: Loss weight in composition
        """
        super().__init__(weight=weight, name="reconstruction")

    def forward(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor, **kwargs) -> LossResult:
        """Compute reconstruction loss for both VAEs.

        Args:
            outputs: Model outputs with logits_A, logits_B
            targets: Input data (batch_size, 9) with values in {-1, 0, 1}

        Returns:
            LossResult with combined loss and per-VAE metrics
        """
        batch_size = targets.size(0)
        x_classes = (targets + 1).long()  # Convert {-1, 0, 1} to {0, 1, 2}

        # VAE-A reconstruction
        ce_A = (
            F.cross_entropy(
                outputs["logits_A"].view(-1, 3),
                x_classes.view(-1),
                reduction="sum",
            )
            / batch_size
        )

        # VAE-B reconstruction
        ce_B = (
            F.cross_entropy(
                outputs["logits_B"].view(-1, 3),
                x_classes.view(-1),
                reduction="sum",
            )
            / batch_size
        )

        return LossResult(
            loss=ce_A + ce_B,
            metrics={"ce_A": ce_A.item(), "ce_B": ce_B.item()},
            weight=self.weight,
        )


class KLDivergenceLossComponent(LossComponent):
    """KL divergence with optional free bits.

    Computes KL(q(z|x) || p(z)) for both VAEs with beta weighting.
    """

    def __init__(self, weight: float = 1.0, free_bits: float = 0.0):
        """Initialize KL divergence loss.

        Args:
            weight: Loss weight in composition
            free_bits: Minimum KL per dimension before penalty
        """
        super().__init__(weight=weight, name="kl")
        self.free_bits = free_bits

    def _compute_kl(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence for one VAE.

        Args:
            mu: Mean of variational posterior
            logvar: Log variance of variational posterior

        Returns:
            KL divergence scalar
        """
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

        if self.free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=self.free_bits)

        return torch.sum(kl_per_dim) / mu.size(0)

    def forward(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor, **kwargs) -> LossResult:
        """Compute KL divergence for both VAEs.

        Args:
            outputs: Model outputs with mu_A, logvar_A, mu_B, logvar_B, beta_A, beta_B
            targets: Not used for KL

        Returns:
            LossResult with combined KL and per-VAE metrics
        """
        kl_A = self._compute_kl(outputs["mu_A"], outputs["logvar_A"])
        kl_B = self._compute_kl(outputs["mu_B"], outputs["logvar_B"])

        # Apply beta weighting
        beta_A = outputs.get("beta_A", 1.0)
        beta_B = outputs.get("beta_B", 1.0)

        loss = beta_A * kl_A + beta_B * kl_B

        return LossResult(
            loss=loss,
            metrics={
                "kl_A": kl_A.item(),
                "kl_B": kl_B.item(),
                "beta_A": (beta_A if isinstance(beta_A, float) else beta_A.item()),
                "beta_B": (beta_B if isinstance(beta_B, float) else beta_B.item()),
            },
            weight=self.weight,
        )


class EntropyLossComponent(LossComponent):
    """Entropy regularization for output distribution.

    Encourages diverse output distribution for VAE-B.
    """

    def __init__(self, weight: float = 0.01, vae: str = "B"):
        """Initialize entropy loss.

        Args:
            weight: Loss weight
            vae: Which VAE to apply to ('A', 'B', or 'both')
        """
        super().__init__(weight=weight, name="entropy")
        self.vae = vae

    def _compute_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute entropy of output distribution.

        Args:
            logits: Model logits (batch_size, 9, 3)

        Returns:
            Entropy value (higher = more diverse)
        """
        probs = F.softmax(logits, dim=-1).mean(dim=0)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        return entropy

    def forward(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor, **kwargs) -> LossResult:
        """Compute entropy regularization.

        Args:
            outputs: Model outputs with logits_A, logits_B
            targets: Not used

        Returns:
            LossResult with negative entropy (as loss)
        """
        metrics = {}

        if self.vae in ("B", "both"):
            entropy_B = self._compute_entropy(outputs["logits_B"])
            metrics["entropy_B"] = entropy_B.item()
            loss = -entropy_B  # Negative because we maximize entropy
        else:
            loss = torch.tensor(0.0, device=outputs["logits_A"].device)

        if self.vae in ("A", "both"):
            entropy_A = self._compute_entropy(outputs["logits_A"])
            metrics["entropy_A"] = entropy_A.item()
            loss = loss - entropy_A

        return LossResult(loss=loss, metrics=metrics, weight=self.weight)


class RepulsionLossComponent(LossComponent):
    """Repulsion loss to encourage latent space diversity.

    Uses RBF kernel to penalize points that are too close in latent space.
    """

    def __init__(self, weight: float = 0.01, sigma: float = 0.5, vae: str = "B"):
        """Initialize repulsion loss.

        Args:
            weight: Loss weight
            sigma: RBF kernel bandwidth
            vae: Which VAE ('A', 'B', or 'both')
        """
        super().__init__(weight=weight, name="repulsion")
        self.sigma = sigma
        self.vae = vae

    def _compute_repulsion(self, z: torch.Tensor) -> torch.Tensor:
        """Compute repulsion for one VAE.

        Args:
            z: Latent codes (batch_size, latent_dim)

        Returns:
            Repulsion loss scalar
        """
        if z.size(0) < 2:
            return torch.tensor(0.0, device=z.device)

        dists = torch.cdist(z, z, p=2)
        mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        repulsion = torch.exp(-(dists[mask] ** 2) / (self.sigma**2)).mean()

        return repulsion

    def forward(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor, **kwargs) -> LossResult:
        """Compute repulsion loss.

        Args:
            outputs: Model outputs with z_A, z_B
            targets: Not used

        Returns:
            LossResult with repulsion loss
        """
        loss = torch.tensor(0.0, device=outputs["z_A"].device)
        metrics = {}

        if self.vae in ("B", "both"):
            rep_B = self._compute_repulsion(outputs["z_B"])
            metrics["repulsion_B"] = rep_B.item()
            loss = loss + rep_B

        if self.vae in ("A", "both"):
            rep_A = self._compute_repulsion(outputs["z_A"])
            metrics["repulsion_A"] = rep_A.item()
            loss = loss + rep_A

        return LossResult(loss=loss, metrics=metrics, weight=self.weight)


class EntropyAlignmentComponent(LossComponent):
    """Entropy alignment between VAEs.

    Encourages similar output entropy between VAE-A and VAE-B.
    """

    def __init__(self, weight: float = 0.1):
        """Initialize entropy alignment.

        Args:
            weight: Loss weight (lambda3 in original)
        """
        super().__init__(weight=weight, name="entropy_align")

    def forward(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor, **kwargs) -> LossResult:
        """Compute entropy alignment loss.

        Args:
            outputs: Model outputs with H_A, H_B
            targets: Not used

        Returns:
            LossResult with alignment loss
        """
        H_A = outputs["H_A"]
        H_B = outputs["H_B"]
        alignment = torch.abs(H_A - H_B)

        return LossResult(
            loss=alignment,
            metrics={
                "H_A": H_A.item(),
                "H_B": H_B.item(),
                "alignment": alignment.item(),
            },
            weight=self.weight,
        )


class PAdicRankingLossComponent(DualVAELossComponent):
    """p-Adic ranking loss with hard negative mining.

    Wraps PAdicRankingLossV2 with the LossComponent interface.
    """

    def __init__(self, weight: float = 0.5, config: dict[str, Any] | None = None):
        """Initialize p-adic ranking loss.

        Args:
            weight: Loss weight
            config: Configuration for PAdicRankingLossV2
        """
        super().__init__(weight=weight, name="padic_ranking")
        config = config or {}

        self.loss_fn = PAdicRankingLossV2(
            base_margin=config.get("base_margin", 0.05),
            margin_scale=config.get("margin_scale", 0.15),
            n_triplets=config.get("n_triplets", 500),
            hard_negative_ratio=config.get("hard_negative_ratio", 0.5),
            semi_hard=config.get("semi_hard", True),
        )

    def compute_single(
        self, z: torch.Tensor, outputs: dict[str, torch.Tensor], targets: torch.Tensor, vae: str, **kwargs
    ) -> LossResult:
        """Compute ranking loss for single VAE.

        Args:
            z: Latent codes
            outputs: Full model outputs
            targets: Not used
            vae: Which VAE

        Returns:
            LossResult with ranking loss
        """
        batch_indices = kwargs.get("batch_indices")

        if batch_indices is None:
            return LossResult(
                loss=torch.tensor(0.0, device=z.device),
                metrics={"skipped": True},
                weight=self.weight,
            )

        loss, metrics = self.loss_fn(z, batch_indices)

        return LossResult(
            loss=loss,
            metrics={
                "hard_ratio": metrics.get("hard_ratio", 0.0),
                "violations": metrics.get("violations", 0),
                "mean_margin": metrics.get("mean_margin", 0.0),
            },
            weight=self.weight,
        )


class PAdicHyperbolicLossComponent(DualVAELossComponent):
    """Hyperbolic p-adic ranking loss.

    Wraps PAdicRankingLossHyperbolic with the LossComponent interface.
    """

    def __init__(self, weight: float = 0.5, config: dict[str, Any] | None = None):
        """Initialize hyperbolic p-adic loss.

        Args:
            weight: Loss weight
            config: Configuration for PAdicRankingLossHyperbolic
        """
        super().__init__(weight=weight, name="padic_hyperbolic")
        config = config or {}

        self.loss_fn = PAdicRankingLossHyperbolic(
            base_margin=config.get("base_margin", 0.05),
            margin_scale=config.get("margin_scale", 0.15),
            n_triplets=config.get("n_triplets", 500),
            hard_negative_ratio=config.get("hard_negative_ratio", 0.5),
            curvature=config.get("curvature", 1.0),
            radial_weight=config.get("radial_weight", 0.1),
            max_norm=config.get("max_norm", 0.95),
        )

    def compute_single(
        self, z: torch.Tensor, outputs: dict[str, torch.Tensor], targets: torch.Tensor, vae: str, **kwargs
    ) -> LossResult:
        """Compute hyperbolic ranking loss for single VAE.

        Args:
            z: Latent codes
            outputs: Full model outputs
            targets: Not used
            vae: Which VAE

        Returns:
            LossResult with hyperbolic ranking loss
        """
        batch_indices = kwargs.get("batch_indices")

        if batch_indices is None:
            return LossResult(
                loss=torch.tensor(0.0, device=z.device),
                metrics={"skipped": True},
                weight=self.weight,
            )

        loss, metrics = self.loss_fn(z, batch_indices)

        return LossResult(
            loss=loss,
            metrics={
                "hard_ratio": metrics.get("hard_ratio", 0.0),
                "violations": metrics.get("violations", 0),
                "poincare_dist_mean": metrics.get("poincare_dist_mean", 0.0),
                "radial_loss": metrics.get("radial_loss", 0.0),
                "ranking_loss": metrics.get("ranking_loss", 0.0),
            },
            weight=self.weight,
        )


class RadialStratificationLossComponent(DualVAELossComponent):
    """Radial stratification loss for 3-adic hierarchy.

    Enforces radial positioning based on 3-adic valuation.
    """

    def __init__(
        self,
        weight: float = 0.3,
        inner_radius: float = 0.1,
        outer_radius: float = 0.85,
        max_valuation: int = 9,
        valuation_weighting: bool = True,
        curvature: float = 1.0,
    ):
        """Initialize radial stratification loss.

        Args:
            weight: Loss weight
            inner_radius: Target radius for high-valuation points
            outer_radius: Target radius for low-valuation points
            max_valuation: Maximum possible valuation (log_3(19683) = 9)
            valuation_weighting: Weight high-valuation points more
            curvature: Hyperbolic curvature for poincare_distance (V5.12.2)
        """
        super().__init__(weight=weight, name="radial_stratification")
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.max_valuation = max_valuation
        self.valuation_weighting = valuation_weighting
        self.curvature = curvature

        # Import here to avoid circular dependency
        from ..core import TERNARY

        self.ternary = TERNARY

    def compute_single(
        self, z: torch.Tensor, outputs: dict[str, torch.Tensor], targets: torch.Tensor, vae: str, **kwargs
    ) -> LossResult:
        """Compute radial stratification for single VAE.

        Args:
            z: Latent codes (batch_size, latent_dim)
            outputs: Full model outputs
            targets: Not used
            vae: Which VAE

        Returns:
            LossResult with radial stratification loss
        """
        batch_indices = kwargs.get("batch_indices")

        if batch_indices is None:
            return LossResult(
                loss=torch.tensor(0.0, device=z.device),
                metrics={"skipped": True},
                weight=self.weight,
            )

        # Compute 3-adic valuations using core module
        valuations = self.ternary.valuation(batch_indices).float()
        normalized_v = valuations / self.max_valuation

        # V5.12.2: Compute actual radius using hyperbolic distance
        origin = torch.zeros_like(z)
        actual_radius = poincare_distance(z, origin, c=self.curvature)

        # Compute target radius (inverse relationship: high v -> small r)
        target_radius = self.outer_radius - normalized_v * (self.outer_radius - self.inner_radius)

        # Compute loss with optional valuation weighting
        weights = 1.0 + normalized_v if self.valuation_weighting else torch.ones_like(normalized_v)

        loss = F.smooth_l1_loss(actual_radius, target_radius, reduction="none")
        weighted_loss = (loss * weights).mean()

        # Compute correlation for monitoring
        if actual_radius.numel() > 1:
            radial_corr = torch.corrcoef(torch.stack([actual_radius, valuations]))[0, 1].item()
        else:
            radial_corr = 0.0

        return LossResult(
            loss=weighted_loss,
            metrics={
                "mean_actual_radius": actual_radius.mean().item(),
                "mean_target_radius": target_radius.mean().item(),
                "radial_correlation": (radial_corr if not torch.isnan(torch.tensor(radial_corr)) else 0.0),
            },
            weight=self.weight,
        )


__all__ = [
    "ReconstructionLossComponent",
    "KLDivergenceLossComponent",
    "EntropyLossComponent",
    "RepulsionLossComponent",
    "EntropyAlignmentComponent",
    "PAdicRankingLossComponent",
    "PAdicHyperbolicLossComponent",
    "RadialStratificationLossComponent",
]
