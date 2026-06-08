# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Exploration boost controller for coverage stall detection.

This module provides adaptive exploration boosting when coverage
improvement stalls. It increases temperature and reduces ranking
weight to encourage exploration of new manifold regions.

Single responsibility: Coverage stall detection and exploration boost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExplorationBoostConfig:
    """Configuration for exploration boost controller.

    Attributes:
        enabled: Whether exploration boost is enabled
        coverage_stall_threshold: Minimum coverage delta to not count as stall
        coverage_stall_patience: Epochs of stall before boosting
        temp_boost_factor: Multiplicative factor for temperature boost
        temp_boost_max: Maximum temperature multiplier
        ranking_reduction_factor: Multiplicative factor for ranking weight reduction
        ranking_reduction_min: Minimum ranking weight multiplier
    """

    enabled: bool = False
    coverage_stall_threshold: float = 0.5
    coverage_stall_patience: int = 5
    temp_boost_factor: float = 1.15
    temp_boost_max: float = 2.0
    ranking_reduction_factor: float = 0.9
    ranking_reduction_min: float = 0.05

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ExplorationBoostConfig:
        """Create config from dictionary.

        Args:
            config: Configuration dict with optional overrides

        Returns:
            ExplorationBoostConfig instance
        """
        return cls(
            enabled=config.get("enabled", False),
            coverage_stall_threshold=config.get("coverage_stall_threshold", 0.5),
            coverage_stall_patience=config.get("coverage_stall_patience", 5),
            temp_boost_factor=config.get("temp_boost_factor", 1.15),
            temp_boost_max=config.get("temp_boost_max", 2.0),
            ranking_reduction_factor=config.get("ranking_reduction_factor", 0.9),
            ranking_reduction_min=config.get("ranking_reduction_min", 0.05),
        )


@dataclass
class ExplorationBoostController:
    """Controller for coverage-triggered exploration boost.

    This controller monitors coverage improvement and applies exploration
    boost when coverage stalls:
    1. Increases temperature multiplier (encourage diverse sampling)
    2. Reduces ranking weight multiplier (reduce structure pressure)

    When coverage improves, the multipliers gradually return to baseline.

    Attributes:
        config: Exploration boost configuration
        stall_counter: Number of consecutive stall epochs
        prev_coverage: Previous epoch's coverage for stall detection
        temp_multiplier: Current temperature multiplier (>=1.0)
        ranking_multiplier: Current ranking weight multiplier (<=1.0)
    """

    config: ExplorationBoostConfig
    stall_counter: int = field(default=0)
    prev_coverage: float = field(default=0.0)
    temp_multiplier: float = field(default=1.0)
    ranking_multiplier: float = field(default=1.0)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> ExplorationBoostController:
        """Create controller from configuration dictionary.

        Args:
            config_dict: Configuration dict (typically from YAML)

        Returns:
            ExplorationBoostController instance
        """
        config = ExplorationBoostConfig.from_dict(config_dict)
        return cls(config=config)

    def check_coverage_stall(self, current_coverage: float) -> bool:
        """Check if coverage is stalled and apply exploration boost if needed.

        Args:
            current_coverage: Current epoch's coverage percentage

        Returns:
            True if exploration boost was applied this epoch
        """
        if not self.config.enabled:
            return False

        # Check coverage delta
        coverage_delta = abs(current_coverage - self.prev_coverage)
        self.prev_coverage = current_coverage

        if coverage_delta < self.config.coverage_stall_threshold:
            self.stall_counter += 1
        else:
            self.stall_counter = 0
            # Gradually return to baseline when coverage improves
            self.temp_multiplier = max(1.0, self.temp_multiplier * 0.95)
            self.ranking_multiplier = min(1.0, self.ranking_multiplier * 1.05)

        # Apply boost if stalled long enough
        if self.stall_counter >= self.config.coverage_stall_patience:
            self.temp_multiplier = min(
                self.temp_multiplier * self.config.temp_boost_factor,
                self.config.temp_boost_max,
            )
            self.ranking_multiplier = max(
                self.ranking_multiplier * self.config.ranking_reduction_factor,
                self.config.ranking_reduction_min,
            )
            return True

        return False

    def get_multipliers(self) -> tuple[float, float]:
        """Get current exploration boost multipliers.

        Returns:
            Tuple of (temp_multiplier, ranking_multiplier)
        """
        return (self.temp_multiplier, self.ranking_multiplier)

    def reset(self) -> None:
        """Reset controller state for new training run."""
        self.stall_counter = 0
        self.prev_coverage = 0.0
        self.temp_multiplier = 1.0
        self.ranking_multiplier = 1.0

    def get_state(self) -> dict[str, Any]:
        """Get current controller state for checkpointing.

        Returns:
            Dict containing all tracking state
        """
        return {
            "stall_counter": self.stall_counter,
            "prev_coverage": self.prev_coverage,
            "temp_multiplier": self.temp_multiplier,
            "ranking_multiplier": self.ranking_multiplier,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore controller state from checkpoint.

        Args:
            state: State dict from get_state()
        """
        self.stall_counter = state.get("stall_counter", 0)
        self.prev_coverage = state.get("prev_coverage", 0.0)
        self.temp_multiplier = state.get("temp_multiplier", 1.0)
        self.ranking_multiplier = state.get("ranking_multiplier", 1.0)

    @property
    def is_boosting(self) -> bool:
        """Check if exploration boost is currently active.

        Returns:
            True if stall counter exceeds patience
        """
        return self.stall_counter >= self.config.coverage_stall_patience


__all__ = ["ExplorationBoostConfig", "ExplorationBoostController"]
