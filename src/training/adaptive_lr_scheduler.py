# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Adaptive learning rate scheduler based on validation metrics.

This module provides an adaptive LR scheduler that adjusts learning rates
based on training dynamics and validation performance.
"""

from dataclasses import dataclass

import torch
from torch.optim.lr_scheduler import _LRScheduler


@dataclass
class ValidationMetrics:
    """Metrics used to guide LR adaptation."""

    loss: float
    coverage: float = 1.0
    hierarchy: float = 0.0
    richness: float = 0.0
    Q: float = 0.0

    @property
    def is_improving(self) -> bool:
        """Check if metrics indicate improvement."""
        return self.coverage >= 0.99 and self.hierarchy < -0.5


@dataclass
class AdaptiveLRConfig:
    """Configuration for adaptive LR scheduler."""

    initial_lr: float = 1e-4
    min_lr: float = 1e-7
    max_lr: float = 1e-2
    patience: int = 10
    factor: float = 0.5
    cooldown: int = 5
    warmup_epochs: int = 5


class AdaptiveLRScheduler(_LRScheduler):
    """Adaptive learning rate scheduler based on validation metrics.

    This scheduler reduces LR when validation metrics plateau and can
    also increase LR when training shows signs of being stuck.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, config: AdaptiveLRConfig | None = None, last_epoch: int = -1):
        self.config = config or AdaptiveLRConfig()
        self.best_loss = float("inf")
        self.bad_epochs = 0
        self.cooldown_counter = 0
        self.history: list[ValidationMetrics] = []

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """Get current learning rates for all param groups."""
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, metrics: ValidationMetrics | None = None, epoch: int | None = None):
        """Step the scheduler based on validation metrics.

        Args:
            metrics: Current validation metrics
            epoch: Current epoch number (optional)
        """
        if epoch is not None:
            self.last_epoch = epoch
        else:
            self.last_epoch += 1

        if metrics is None:
            return

        self.history.append(metrics)

        # Warmup phase - don't adjust
        if self.last_epoch < self.config.warmup_epochs:
            return

        # Cooldown phase - don't adjust
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        # Check for improvement
        if metrics.loss < self.best_loss * 0.99:  # 1% improvement threshold
            self.best_loss = metrics.loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        # Reduce LR if no improvement
        if self.bad_epochs >= self.config.patience:
            self._reduce_lr()
            self.bad_epochs = 0
            self.cooldown_counter = self.config.cooldown

    def _reduce_lr(self):
        """Reduce learning rate by factor."""
        for param_group in self.optimizer.param_groups:
            new_lr = max(param_group["lr"] * self.config.factor, self.config.min_lr)
            param_group["lr"] = new_lr


def create_adaptive_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    initial_lr: float = 1e-4,
    min_lr: float = 1e-7,
    patience: int = 10,
    factor: float = 0.5,
    warmup_epochs: int = 5,
) -> AdaptiveLRScheduler:
    """Create an adaptive LR scheduler.

    Args:
        optimizer: The optimizer to schedule
        initial_lr: Initial learning rate
        min_lr: Minimum learning rate
        patience: Epochs to wait before reducing LR
        factor: Factor to reduce LR by
        warmup_epochs: Number of warmup epochs

    Returns:
        AdaptiveLRScheduler instance
    """
    config = AdaptiveLRConfig(
        initial_lr=initial_lr, min_lr=min_lr, patience=patience, factor=factor, warmup_epochs=warmup_epochs
    )
    return AdaptiveLRScheduler(optimizer, config)


__all__ = ["ValidationMetrics", "AdaptiveLRConfig", "AdaptiveLRScheduler", "create_adaptive_lr_scheduler"]
