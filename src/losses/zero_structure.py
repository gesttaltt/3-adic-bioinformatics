# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Zero-Structure Loss for exploiting ternary zero semantics.

This module implements losses that exploit the semantic meaning of zeros
in ternary operations {-1, 0, +1}:

1. ZeroValuationLoss: Trailing zeros in operation → closer to origin
2. ZeroSparsityLoss: Higher zero-count → efficiency bonus

Key insight: Zeros in balanced ternary are not just "off" - they represent
"neutral/absent/don't-care" states. Operations with structured zeros
(trailing zeros = higher valuation) should cluster near the manifold origin,
matching p-adic geometry where d(a, 0) = 3^{-v(a)}.

V5.11.9: Zero-structure exploitation.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geometry.poincare import poincare_distance


def compute_operation_zero_valuation(operations: torch.Tensor) -> torch.Tensor:
    """Compute zero-valuation directly from operation vectors.

    Zero-valuation = count of trailing zeros (from position 0).
    This is different from index-based 3-adic valuation.

    Args:
        operations: Ternary operations (batch, 9) with values in {-1, 0, +1}

    Returns:
        valuations: Zero-valuation per operation (batch,) in range [0, 9]
    """
    batch_size = operations.size(0)
    valuations = torch.zeros(batch_size, device=operations.device)

    # Count trailing zeros (from position 0)
    for pos in range(9):
        is_zero = (operations[:, pos] == 0).float()
        # Only count if ALL previous positions were also zero
        if pos == 0:
            valuations += is_zero
        else:
            prev_all_zero = (valuations == pos).float()
            valuations += is_zero * prev_all_zero

    return valuations


def compute_operation_zero_count(operations: torch.Tensor) -> torch.Tensor:
    """Count total zeros in each operation.

    Args:
        operations: Ternary operations (batch, 9) with values in {-1, 0, +1}

    Returns:
        zero_counts: Number of zeros per operation (batch,) in range [0, 9]
    """
    return (operations == 0).sum(dim=1).float()


def compute_position_weighted_zeros(operations: torch.Tensor) -> torch.Tensor:
    """Compute position-weighted zero score.

    Lower positions (LSB in p-adic sense) are weighted more heavily.
    Position 0 has weight 1, position 8 has weight 1/3^8.

    Args:
        operations: Ternary operations (batch, 9) with values in {-1, 0, +1}

    Returns:
        scores: Position-weighted zero score (batch,)
    """
    weights = torch.pow(3.0, -torch.arange(9, device=operations.device).float())
    is_zero = (operations == 0).float()
    return (is_zero * weights).sum(dim=1)


class ZeroValuationLoss(nn.Module):
    """Enforces radial hierarchy based on operation zero-valuation.

    Maps zero structure to radial position:
    - High zero-valuation (many trailing zeros) → small radius (near origin)
    - Low zero-valuation (no trailing zeros) → large radius (near boundary)

    This exploits the semantic meaning of zeros as "neutral" states,
    creating natural hierarchy based on operation structure.

    Formula:
        r_target = outer_radius - (zero_val / 9) * (outer_radius - inner_radius)
    """

    def __init__(
        self,
        inner_radius: float = 0.1,
        outer_radius: float = 0.85,
        weight: float = 1.0,
        loss_type: str = "smooth_l1",
        curvature: float = 1.0,
    ):
        """Initialize Zero Valuation Loss.

        Args:
            inner_radius: Target radius for high zero-valuation (default: 0.1)
            outer_radius: Target radius for zero zero-valuation (default: 0.85)
            weight: Loss weight multiplier
            loss_type: 'smooth_l1' or 'mse'
            curvature: Hyperbolic curvature for poincare_distance (V5.12.2)
        """
        super().__init__()
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.weight = weight
        self.loss_type = loss_type
        self.radius_range = outer_radius - inner_radius
        self.curvature = curvature

    def forward(
        self,
        z: torch.Tensor,
        operations: torch.Tensor,
        return_metrics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Compute zero-valuation radial loss.

        Args:
            z: Latent codes (batch, latent_dim)
            operations: Ternary operations (batch, 9)
            return_metrics: If True, return metrics dict

        Returns:
            loss: Scalar loss value
        """
        # Compute zero-valuation from operation vectors
        valuations = compute_operation_zero_valuation(operations)

        # V5.12.2: Compute actual radius using hyperbolic distance
        origin = torch.zeros_like(z)
        actual_radius = poincare_distance(z, origin, c=self.curvature)

        # Compute target radius (high valuation → small radius)
        normalized_v = valuations / 9.0
        target_radius = self.outer_radius - normalized_v * self.radius_range

        # Compute loss
        if self.loss_type == "smooth_l1":
            loss = F.smooth_l1_loss(actual_radius, target_radius)
        else:
            loss = F.mse_loss(actual_radius, target_radius)

        loss = loss * self.weight

        if return_metrics:
            with torch.no_grad():
                # Correlation between valuation and radius
                corr = torch.corrcoef(torch.stack([valuations, actual_radius]))[0, 1]

                metrics = {
                    "zero_val_loss": loss.item(),
                    "zero_val_radius_corr": (corr.item() if not torch.isnan(corr) else 0.0),
                    "mean_zero_valuation": valuations.mean().item(),
                    "high_val_radius": (
                        actual_radius[valuations >= 3].mean().item() if (valuations >= 3).any() else 0.0
                    ),
                    "low_val_radius": (
                        actual_radius[valuations == 0].mean().item() if (valuations == 0).any() else 0.0
                    ),
                }
                return loss, metrics

        return loss


class ZeroSparsityLoss(nn.Module):
    """Encourages the model to exploit zero-sparsity.

    Operations with more zeros should have:
    1. Smaller latent norms (efficiency)
    2. More compact cluster structure

    This loss creates pressure for the model to "prefer" sparse (zero-heavy)
    representations when multiple encodings are equivalent.
    """

    def __init__(self, target_correlation: float = -0.3, weight: float = 0.5, curvature: float = 1.0):
        """Initialize Zero Sparsity Loss.

        Args:
            target_correlation: Target correlation between zero-count and radius
                               Negative = more zeros means smaller radius
            weight: Loss weight multiplier
            curvature: Hyperbolic curvature for poincare_distance (V5.12.2)
        """
        super().__init__()
        self.target_correlation = target_correlation
        self.weight = weight
        self.curvature = curvature

    def forward(
        self,
        z: torch.Tensor,
        operations: torch.Tensor,
        return_metrics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Compute zero-sparsity loss.

        Args:
            z: Latent codes (batch, latent_dim)
            operations: Ternary operations (batch, 9)
            return_metrics: If True, return metrics dict

        Returns:
            loss: Scalar loss encouraging zero-count/radius correlation
        """
        # Compute zero count
        zero_counts = compute_operation_zero_count(operations)

        # V5.12.2: Compute radius using hyperbolic distance
        origin = torch.zeros_like(z)
        radius = poincare_distance(z, origin, c=self.curvature)

        # Compute current correlation
        # Standardize
        zero_std = (zero_counts - zero_counts.mean()) / (zero_counts.std() + 1e-8)
        radius_std = (radius - radius.mean()) / (radius.std() + 1e-8)

        # Correlation
        current_corr = (zero_std * radius_std).mean()

        # Loss: push correlation toward target (negative)
        loss = F.mse_loss(
            current_corr,
            torch.tensor(self.target_correlation, device=z.device),
        )
        loss = loss * self.weight

        if return_metrics:
            with torch.no_grad():
                metrics = {
                    "zero_sparsity_loss": loss.item(),
                    "zero_count_radius_corr": current_corr.item(),
                    "mean_zero_count": zero_counts.mean().item(),
                    "high_zero_radius": (radius[zero_counts >= 6].mean().item() if (zero_counts >= 6).any() else 0.0),
                    "low_zero_radius": (radius[zero_counts <= 2].mean().item() if (zero_counts <= 2).any() else 0.0),
                }
                return loss, metrics

        return loss


class CombinedZeroStructureLoss(nn.Module):
    """Combined loss for zero-structure exploitation.

    Combines:
    1. ZeroValuationLoss: Trailing zeros → radial hierarchy
    2. ZeroSparsityLoss: Zero-count → radius correlation

    This is the recommended loss for V5.11.9 zero-structure training.
    """

    def __init__(
        self,
        valuation_weight: float = 1.0,
        sparsity_weight: float = 0.5,
        inner_radius: float = 0.1,
        outer_radius: float = 0.85,
        target_correlation: float = -0.3,
        curvature: float = 1.0,
    ):
        """Initialize Combined Zero Structure Loss.

        Args:
            valuation_weight: Weight for zero-valuation loss
            sparsity_weight: Weight for zero-sparsity loss
            inner_radius: Target radius for high-valuation operations
            outer_radius: Target radius for low-valuation operations
            target_correlation: Target zero-count/radius correlation
            curvature: Hyperbolic curvature for poincare_distance (V5.12.2)
        """
        super().__init__()

        self.valuation_loss = ZeroValuationLoss(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            weight=valuation_weight,
            curvature=curvature,
        )

        self.sparsity_loss = ZeroSparsityLoss(
            target_correlation=target_correlation,
            weight=sparsity_weight,
            curvature=curvature,
        )

    def forward(
        self,
        z: torch.Tensor,
        operations: torch.Tensor,
        return_metrics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Compute combined zero-structure loss.

        Args:
            z: Latent codes (batch, latent_dim)
            operations: Ternary operations (batch, 9)
            return_metrics: If True, return metrics dict

        Returns:
            loss: Combined scalar loss
        """
        if return_metrics:
            val_loss, val_metrics = self.valuation_loss(z, operations, return_metrics=True)
            spar_loss, spar_metrics = self.sparsity_loss(z, operations, return_metrics=True)

            combined_loss = val_loss + spar_loss

            metrics = {
                "zero_structure_loss": combined_loss.item(),
                **val_metrics,
                **spar_metrics,
            }
            return combined_loss, metrics
        else:
            val_loss = self.valuation_loss(z, operations)
            spar_loss = self.sparsity_loss(z, operations)
            return val_loss + spar_loss
