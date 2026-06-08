# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Correlation-based early stopping controller.

This module provides early stopping based on correlation degradation.
When correlation drops significantly from its best value, training
can be stopped to prevent overfitting or model collapse.

Single responsibility: Correlation-based early stopping logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CorrelationEarlyStopConfig:
    """Configuration for correlation-based early stopping.

    Attributes:
        enabled: Whether correlation early stopping is enabled
        correlation_loss_weight: Weight for correlation in loss (for future use)
        target_correlation: Target correlation value
        correlation_drop_threshold: Drop threshold to trigger counter
        correlation_patience: Epochs to wait after drop before stopping
    """

    enabled: bool = False
    correlation_loss_weight: float = 0.1
    target_correlation: float = 0.95
    correlation_drop_threshold: float = 0.05
    correlation_patience: int = 10

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> CorrelationEarlyStopConfig:
        """Create config from dictionary.

        Args:
            config: Configuration dict with optional overrides

        Returns:
            CorrelationEarlyStopConfig instance
        """
        return cls(
            enabled=config.get("enabled", False),
            correlation_loss_weight=config.get("correlation_loss_weight", 0.1),
            target_correlation=config.get("target_correlation", 0.95),
            correlation_drop_threshold=config.get("correlation_drop_threshold", 0.05),
            correlation_patience=config.get("correlation_patience", 10),
        )


@dataclass
class CorrelationEarlyStop:
    """Controller for correlation-based early stopping.

    This controller monitors the correlation metric and triggers early
    stopping when correlation drops significantly from its best value
    for a sustained period (patience epochs).

    Use case: Prevent training from continuing when the model starts
    to lose geometric structure (correlation degradation).

    Attributes:
        config: Early stopping configuration
        drop_counter: Number of epochs since drop began
        best_correlation: Best correlation observed for stopping logic
    """

    config: CorrelationEarlyStopConfig
    drop_counter: int = field(default=0)
    best_correlation: float = field(default=0.0)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> CorrelationEarlyStop:
        """Create controller from configuration dictionary.

        Args:
            config_dict: Configuration dict (typically from YAML)

        Returns:
            CorrelationEarlyStop instance
        """
        config = CorrelationEarlyStopConfig.from_dict(config_dict)
        return cls(config=config)

    def check_should_stop(self, current_correlation: float) -> bool:
        """Check if training should stop due to correlation drop.

        Args:
            current_correlation: Current epoch's correlation (mean of A and B)

        Returns:
            True if training should stop due to correlation degradation
        """
        if not self.config.enabled:
            return False

        # Update best correlation
        if current_correlation > self.best_correlation:
            self.best_correlation = current_correlation
            self.drop_counter = 0
            return False

        # Check if correlation dropped significantly
        drop = self.best_correlation - current_correlation
        if drop > self.config.correlation_drop_threshold:
            self.drop_counter += 1
            if self.drop_counter >= self.config.correlation_patience:
                return True

        return False

    def reset(self) -> None:
        """Reset controller state for new training run."""
        self.drop_counter = 0
        self.best_correlation = 0.0

    def get_state(self) -> dict[str, Any]:
        """Get current controller state for checkpointing.

        Returns:
            Dict containing drop_counter and best_correlation
        """
        return {
            "drop_counter": self.drop_counter,
            "best_correlation": self.best_correlation,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore controller state from checkpoint.

        Args:
            state: State dict from get_state()
        """
        self.drop_counter = state.get("drop_counter", 0)
        self.best_correlation = state.get("best_correlation", 0.0)

    @property
    def is_triggered(self) -> bool:
        """Check if early stopping has been triggered.

        Returns:
            True if drop counter exceeds patience
        """
        return self.drop_counter >= self.config.correlation_patience


__all__ = ["CorrelationEarlyStopConfig", "CorrelationEarlyStop"]
