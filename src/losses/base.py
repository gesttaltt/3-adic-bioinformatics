# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Base interfaces for loss components.

This module defines the protocol/interface that all loss components must follow.
Each loss is self-contained, reports its own metrics, and has no dependencies
on other loss components.

Design Principles:
    - Single Responsibility: Each loss does one thing
    - Consistent Interface: All losses follow LossComponent protocol
    - Self-Reporting: Each loss returns its own metrics
    - Stateless: No side effects, pure computation
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class LossResult:
    """Standardized loss output.

    Every loss component returns a LossResult containing:
    - loss: The scalar loss value (for backprop)
    - metrics: Dictionary of values to log (for observability)
    - weight: The weight applied to this loss in the total

    This decouples loss computation from metric aggregation.
    """

    loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)
    weight: float = 1.0

    @property
    def weighted_loss(self) -> torch.Tensor:
        """Return loss multiplied by weight."""
        return self.weight * self.loss

    def to_dict(self, prefix: str = "") -> dict[str, Any]:
        """Convert to flat dictionary for logging.

        Args:
            prefix: Optional prefix for metric names

        Returns:
            Flattened dictionary of metrics
        """
        result = {}

        # Add the loss value
        loss_key = f"{prefix}_loss" if prefix else "loss"
        result[loss_key] = self.loss.item() if torch.is_tensor(self.loss) else self.loss

        # Add all metrics with optional prefix
        for key, value in self.metrics.items():
            metric_key = f"{prefix}_{key}" if prefix else key
            result[metric_key] = value

        return result


class LossComponent(ABC, nn.Module):
    """Abstract base class for all loss components.

    All loss components must:
    1. Inherit from this class
    2. Implement the forward method with consistent signature
    3. Return a LossResult

    This enables dynamic composition via LossRegistry.

    Example:
        class MyLoss(LossComponent):
            def __init__(self, config):
                super().__init__()
                self.param = config.get('param', 1.0)

            def forward(self, outputs, targets, **kwargs) -> LossResult:
                loss = compute_my_loss(outputs, targets, self.param)
                return LossResult(
                    loss=loss,
                    metrics={'my_metric': loss.item()},
                    weight=self.weight
                )
    """

    def __init__(self, weight: float = 1.0, name: str | None = None):
        """Initialize loss component.

        Args:
            weight: Weight for this loss in composition
            name: Optional name for logging (defaults to class name)
        """
        super().__init__()
        self.weight = weight
        self._name = name or self.__class__.__name__

    @property
    def name(self) -> str:
        """Return the name of this loss component."""
        return self._name

    @abstractmethod
    def forward(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor, **kwargs) -> LossResult:
        """Compute the loss.

        Args:
            outputs: Model outputs dictionary (z_A, z_B, mu_A, logvar_A, etc.)
            targets: Target values (typically input x for reconstruction)
            **kwargs: Additional arguments (batch_indices, step, etc.)

        Returns:
            LossResult containing loss, metrics, and weight
        """
        pass

    def enabled(self) -> bool:
        """Return True if this loss should be computed.

        Override in subclasses to add enable/disable logic.
        """
        return True


class DualVAELossComponent(LossComponent):
    """Loss component that operates on both VAE-A and VAE-B.

    Most losses in the dual VAE system compute separately for each VAE.
    This base class handles the common pattern of computing for both
    and combining the results.
    """

    def __init__(
        self,
        weight: float = 1.0,
        name: str | None = None,
        combine: str = "sum",
    ):
        """Initialize dual VAE loss component.

        Args:
            weight: Weight for this loss
            name: Optional name for logging
            combine: How to combine A and B losses ('sum', 'mean')
        """
        super().__init__(weight, name)
        self.combine = combine

    @abstractmethod
    def compute_single(
        self,
        z: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        vae: str,
        **kwargs,
    ) -> LossResult:
        """Compute loss for a single VAE.

        Args:
            z: Latent code for this VAE
            outputs: Full model outputs
            targets: Target values
            vae: Which VAE ('A' or 'B')
            **kwargs: Additional arguments

        Returns:
            LossResult for this VAE
        """
        pass

    def forward(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor, **kwargs) -> LossResult:
        """Compute loss for both VAEs and combine.

        Args:
            outputs: Model outputs with z_A, z_B
            targets: Target values
            **kwargs: Additional arguments

        Returns:
            Combined LossResult
        """
        # Compute for VAE-A
        result_A = self.compute_single(outputs["z_A"], outputs, targets, vae="A", **kwargs)

        # Compute for VAE-B
        result_B = self.compute_single(outputs["z_B"], outputs, targets, vae="B", **kwargs)

        # Combine losses
        if self.combine == "sum":
            combined_loss = result_A.loss + result_B.loss
        else:  # mean
            combined_loss = (result_A.loss + result_B.loss) / 2

        # Merge metrics with A/B suffix
        metrics = {}
        for key, value in result_A.metrics.items():
            metrics[f"{key}_A"] = value
        for key, value in result_B.metrics.items():
            metrics[f"{key}_B"] = value

        return LossResult(loss=combined_loss, metrics=metrics, weight=self.weight)


__all__ = ["LossResult", "LossComponent", "DualVAELossComponent"]
