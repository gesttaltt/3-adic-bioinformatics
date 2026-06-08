# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Continuous feedback controller for coverage-based ranking weight adaptation.

This module provides adaptive ranking weight modulation based on coverage
metrics. When coverage is high, focus on structure (high ranking weight).
When coverage is low, focus on exploration (low ranking weight).

Single responsibility: Coverage-based ranking weight computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from src.config.constants import (
    DEFAULT_COVERAGE_SENSITIVITY,
    DEFAULT_COVERAGE_THRESHOLD,
    DEFAULT_COVERAGE_TREND_SENSITIVITY,
    DEFAULT_EMA_ALPHA,
    DEFAULT_MAX_RANKING_WEIGHT,
    DEFAULT_MIN_RANKING_WEIGHT,
    DEFAULT_RANKING_WEIGHT,
)


@dataclass
class ContinuousFeedbackConfig:
    """Configuration for continuous feedback controller.

    Attributes:
        enabled: Whether continuous feedback is enabled
        base_ranking_weight: Base ranking weight when feedback is disabled
        coverage_threshold: Coverage percentage threshold for weight adjustment
        coverage_sensitivity: Sensitivity to coverage gap
        coverage_trend_sensitivity: Sensitivity to coverage trend
        min_ranking_weight: Minimum ranking weight
        max_ranking_weight: Maximum ranking weight
        coverage_ema_alpha: EMA smoothing factor for coverage
    """

    enabled: bool = True
    base_ranking_weight: float = DEFAULT_RANKING_WEIGHT
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD
    coverage_sensitivity: float = DEFAULT_COVERAGE_SENSITIVITY
    coverage_trend_sensitivity: float = DEFAULT_COVERAGE_TREND_SENSITIVITY
    min_ranking_weight: float = DEFAULT_MIN_RANKING_WEIGHT
    max_ranking_weight: float = DEFAULT_MAX_RANKING_WEIGHT
    coverage_ema_alpha: float = DEFAULT_EMA_ALPHA

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ContinuousFeedbackConfig:
        """Create config from dictionary.

        Args:
            config: Configuration dict with optional overrides

        Returns:
            ContinuousFeedbackConfig instance
        """
        return cls(
            enabled=config.get("enabled", True),
            base_ranking_weight=config.get("base_ranking_weight", DEFAULT_RANKING_WEIGHT),
            coverage_threshold=config.get("coverage_threshold", DEFAULT_COVERAGE_THRESHOLD),
            coverage_sensitivity=config.get("coverage_sensitivity", DEFAULT_COVERAGE_SENSITIVITY),
            coverage_trend_sensitivity=config.get("coverage_trend_sensitivity", DEFAULT_COVERAGE_TREND_SENSITIVITY),
            min_ranking_weight=config.get("min_ranking_weight", DEFAULT_MIN_RANKING_WEIGHT),
            max_ranking_weight=config.get("max_ranking_weight", DEFAULT_MAX_RANKING_WEIGHT),
            coverage_ema_alpha=config.get("coverage_ema_alpha", DEFAULT_EMA_ALPHA),
        )


@dataclass
class ContinuousFeedbackController:
    """Controller for coverage-based ranking weight adaptation.

    This controller implements a sigmoid-based continuous feedback system
    that modulates ranking weight based on:
    1. Coverage gap from threshold (coverage - threshold)
    2. Coverage trend (improvement rate)

    The ranking weight increases when coverage is high (focus on structure)
    and decreases when coverage is low (focus on exploration).

    Attributes:
        config: Feedback configuration
        coverage_ema: Exponential moving average of coverage
        prev_coverage: Previous epoch's coverage for trend calculation
    """

    config: ContinuousFeedbackConfig
    coverage_ema: float | None = field(default=None)
    prev_coverage: float | None = field(default=None)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> ContinuousFeedbackController:
        """Create controller from configuration dictionary.

        Args:
            config_dict: Configuration dict (typically from YAML)

        Returns:
            ContinuousFeedbackController instance
        """
        config = ContinuousFeedbackConfig.from_dict(config_dict)
        return cls(config=config)

    def compute_ranking_weight(self, current_coverage: float) -> float:
        """Compute ranking weight using sigmoid-based continuous feedback.

        The ranking weight modulates how strongly the hyperbolic ranking loss
        affects training. It increases when coverage is high (can focus on
        structure) and decreases when coverage is low (focus on exploration).

        Args:
            current_coverage: Current mean coverage percentage

        Returns:
            Ranking weight in [min_ranking_weight, max_ranking_weight]
        """
        if not self.config.enabled:
            return self.config.base_ranking_weight

        # Update coverage EMA
        if self.coverage_ema is None:
            self.coverage_ema = current_coverage
        else:
            alpha = self.config.coverage_ema_alpha
            self.coverage_ema = alpha * self.coverage_ema + (1 - alpha) * current_coverage

        # Compute coverage trend
        coverage_trend = 0.0 if self.prev_coverage is None else current_coverage - self.prev_coverage

        self.prev_coverage = current_coverage

        # Sigmoid modulation
        coverage_gap = current_coverage - self.config.coverage_threshold
        signal = (
            self.config.coverage_sensitivity * coverage_gap + self.config.coverage_trend_sensitivity * coverage_trend
        )

        modulation = torch.sigmoid(torch.tensor(signal)).item()

        # Scale to [min, max] range
        weight = self.config.min_ranking_weight + modulation * (
            self.config.max_ranking_weight - self.config.min_ranking_weight
        )

        return weight

    def reset(self) -> None:
        """Reset controller state for new training run."""
        self.coverage_ema = None
        self.prev_coverage = None

    def get_state(self) -> dict[str, Any]:
        """Get current controller state for checkpointing.

        Returns:
            Dict containing coverage_ema and prev_coverage
        """
        return {
            "coverage_ema": self.coverage_ema,
            "prev_coverage": self.prev_coverage,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore controller state from checkpoint.

        Args:
            state: State dict from get_state()
        """
        self.coverage_ema = state.get("coverage_ema")
        self.prev_coverage = state.get("prev_coverage")


__all__ = ["ContinuousFeedbackConfig", "ContinuousFeedbackController"]
