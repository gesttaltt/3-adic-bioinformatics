# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Early stopping callback implementations.

This module provides various early stopping strategies:
- Loss-based: Stop when validation loss stops improving
- Coverage-based: Stop when coverage plateaus
- Correlation-based: Stop when correlation drops

Usage:
    callback = EarlyStoppingCallback(
        monitor="val_loss",
        patience=50,
        min_delta=0.0001
    )
    trainer = Trainer(callbacks=[callback])
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import TrainingCallback

if TYPE_CHECKING:
    from ..base import BaseTrainer

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(TrainingCallback):
    """Stop training when a monitored metric stops improving.

    Attributes:
        monitor: Metric name to monitor
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as improvement
        mode: 'min' (lower is better) or 'max' (higher is better)
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 50,
        min_delta: float = 0.0001,
        mode: str = "min",
        verbose: bool = True,
    ):
        """Initialize early stopping callback.

        Args:
            monitor: Metric name to monitor (e.g., 'val_loss', 'coverage')
            patience: Epochs to wait before stopping
            min_delta: Minimum improvement required
            mode: 'min' for loss, 'max' for accuracy/coverage
            verbose: Whether to log stopping messages
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.best_value: float | None = None
        self.counter = 0
        self.best_epoch = 0

    def on_train_start(self, trainer: BaseTrainer) -> None:
        """Reset state at training start."""
        self.best_value = None
        self.counter = 0
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: BaseTrainer) -> bool | None:
        """Check if training should stop.

        Returns:
            True if training should stop, None otherwise
        """
        current = metrics.get(self.monitor)

        if current is None:
            logger.warning(f"EarlyStopping: metric '{self.monitor}' not found in metrics")
            return None

        if self.best_value is None:
            self.best_value = current
            self.best_epoch = epoch
            return None

        # Check for improvement
        if self.mode == "min":
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta

        if improved:
            self.best_value = current
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.verbose:
                logger.info(
                    f"Early stopping triggered: {self.monitor} did not improve "
                    f"for {self.patience} epochs. Best: {self.best_value:.6f} "
                    f"at epoch {self.best_epoch}"
                )
            return True

        return None


class CoveragePlateauCallback(TrainingCallback):
    """Stop training when coverage stops improving.

    Specifically designed for the Ternary VAE where we want to
    stop when the model has covered most of the operation space.
    """

    def __init__(
        self,
        patience: int = 100,
        min_delta: float = 0.0005,
        target_coverage: float = 99.7,
        verbose: bool = True,
    ):
        """Initialize coverage plateau callback.

        Args:
            patience: Epochs to wait after plateau
            min_delta: Minimum coverage improvement
            target_coverage: Stop when this coverage is reached
            verbose: Whether to log messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.target_coverage = target_coverage
        self.verbose = verbose

        self.best_coverage = 0.0
        self.counter = 0
        self.best_epoch = 0

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: BaseTrainer) -> bool | None:
        """Check for coverage plateau or target reached."""
        # Try different metric names
        coverage = metrics.get("coverage") or metrics.get("coverage_A") or metrics.get("coverage_percent")

        if coverage is None:
            return None

        # Check if target reached
        if coverage >= self.target_coverage:
            if self.verbose:
                logger.info(f"Target coverage {self.target_coverage}% reached ({coverage:.2f}%) at epoch {epoch}")
            return True

        # Check for improvement
        if coverage > self.best_coverage + self.min_delta:
            self.best_coverage = coverage
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.verbose:
                logger.info(
                    f"Coverage plateau: {coverage:.2f}% has not improved "
                    f"for {self.patience} epochs. Best: {self.best_coverage:.2f}%"
                )
            return True

        return None


class CorrelationDropCallback(TrainingCallback):
    """Stop training if correlation drops significantly.

    Monitors the ranking correlation and stops if it drops
    below a threshold or degrades significantly from peak.
    """

    def __init__(
        self,
        drop_threshold: float = 0.05,
        patience: int = 10,
        min_correlation: float = 0.8,
        verbose: bool = True,
    ):
        """Initialize correlation drop callback.

        Args:
            drop_threshold: Maximum allowed drop from peak
            patience: Epochs to wait after drop detected
            min_correlation: Minimum acceptable correlation
            verbose: Whether to log messages
        """
        self.drop_threshold = drop_threshold
        self.patience = patience
        self.min_correlation = min_correlation
        self.verbose = verbose

        self.best_correlation = 0.0
        self.counter = 0

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: BaseTrainer) -> bool | None:
        """Check for correlation drop."""
        correlation = metrics.get("correlation") or metrics.get("ranking_correlation") or metrics.get("correlation_A")

        if correlation is None:
            return None

        # Update best
        if correlation > self.best_correlation:
            self.best_correlation = correlation
            self.counter = 0
            return None

        # Check for significant drop
        drop = self.best_correlation - correlation
        if drop > self.drop_threshold:
            self.counter += 1

            if self.counter >= self.patience:
                if self.verbose:
                    logger.info(
                        f"Correlation drop: {correlation:.4f} is {drop:.4f} below best ({self.best_correlation:.4f})"
                    )
                return True

        return None


__all__ = [
    "EarlyStoppingCallback",
    "CoveragePlateauCallback",
    "CorrelationDropCallback",
]
