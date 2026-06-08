# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Base trainer with defensive patterns.

This module provides a BaseTrainer class with:
- Safe division helpers to prevent division-by-zero crashes
- Validation guards for optional val_loader
- Common checkpointing boilerplate

All trainers should inherit from this class to prevent:
- P0 bugs: Division by zero (S5.2-S5.5 from TECHNICAL_DEBT_AUDIT)
- Forked logic: Changes in one trainer don't propagate to others

Single responsibility: Provide safe training utilities and common boilerplate.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import DataLoader


class BaseTrainer(ABC):
    """Abstract base trainer with defensive patterns.

    All concrete trainers should inherit from this class to ensure:
    1. Division-by-zero protection in loss averaging
    2. Proper handling of optional val_loader
    3. Consistent checkpointing interface
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict[str, Any],
        device: str = "cuda",
    ):
        """Initialize base trainer.

        Args:
            model: PyTorch model to train
            config: Training configuration dict
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.epoch = 0

    @staticmethod
    def safe_average_losses(
        epoch_losses: dict[str, float],
        num_batches: int,
        exclude_keys: set | None = None,
    ) -> dict[str, float]:
        """Safely average accumulated losses, guarding against division by zero.

        Args:
            epoch_losses: Dict of accumulated losses
            num_batches: Number of batches (may be 0)
            exclude_keys: Keys to exclude from averaging (e.g., learning rates)

        Returns:
            Dict with averaged losses (unchanged if num_batches == 0)
        """
        if num_batches == 0:
            return dict(epoch_losses)

        exclude = exclude_keys or set()
        averaged = {}
        for key, val in epoch_losses.items():
            if key in exclude:
                averaged[key] = val
            else:
                averaged[key] = val / num_batches
        return averaged

    @staticmethod
    def accumulate_losses(epoch_losses: dict[str, float], batch_losses: dict[str, Any]) -> None:
        """Accumulate batch losses into epoch losses (in-place).

        Handles both Tensor and scalar values.

        Args:
            epoch_losses: Accumulator dict (modified in-place)
            batch_losses: Current batch losses
        """
        for key, val in batch_losses.items():
            if isinstance(val, torch.Tensor):
                epoch_losses[key] += val.item()
            else:
                epoch_losses[key] += val

    def run_validation(
        self, val_loader: DataLoader | None, train_losses: dict[str, float]
    ) -> tuple[dict[str, float], bool]:
        """Run validation with proper None-check.

        Args:
            val_loader: Optional validation DataLoader
            train_losses: Training losses (used as fallback)

        Returns:
            Tuple of (val_losses, is_best)
        """
        if val_loader is not None:
            val_losses = self.validate(val_loader)
            is_best = self._check_best(val_losses)
        else:
            # Manifold approach: use train losses
            val_losses = train_losses
            is_best = self._check_best(train_losses)
        return val_losses, is_best

    @abstractmethod
    def validate(self, val_loader: DataLoader) -> dict[str, float]:
        """Validation pass (must be implemented by subclass).

        Args:
            val_loader: Validation data loader

        Returns:
            Dict of validation losses
        """
        pass

    @abstractmethod
    def _check_best(self, losses: dict[str, float]) -> bool:
        """Check if current losses represent best model (subclass-specific).

        Args:
            losses: Current validation/training losses

        Returns:
            True if this is the best model so far
        """
        pass

    def create_epoch_losses(self) -> dict[str, float]:
        """Create a new epoch losses accumulator with defaultdict(float).

        Returns:
            Empty accumulator for batch losses
        """
        return defaultdict(float)


# Default keys to exclude from averaging (learning rates, deltas, etc.)
STATENET_KEYS = frozenset(
    [
        "lr_corrected",
        "delta_lr",
        "delta_lambda1",
        "delta_lambda2",
        "delta_lambda3",
        "delta_curriculum",
        "delta_sigma",
        "delta_curvature",
    ]
)


__all__ = [
    "BaseTrainer",
    "STATENET_KEYS",
]
