# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Metrics tracking for training monitoring.

This module handles history tracking and best value management:
- Coverage history for VAE-A and VAE-B
- Entropy history tracking
- Correlation history (hyperbolic and Euclidean)
- Best value tracking
- Plateau detection and early stopping logic

Single responsibility: Metrics history and tracking only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config.constants import N_TERNARY_OPERATIONS


@dataclass
class MetricsTracker:
    """Tracks training metrics history and best values.

    Handles all history management, best value tracking,
    plateau detection, and early stopping logic.

    Attributes:
        best_val_loss: Best validation loss seen
        patience_counter: Counter for early stopping patience
        coverage_A_history: VAE-A coverage counts over epochs
        coverage_B_history: VAE-B coverage counts over epochs
        H_A_history: VAE-A entropy values over epochs
        H_B_history: VAE-B entropy values over epochs
        correlation_hyp_history: Mean hyperbolic correlation over epochs
        correlation_euc_history: Mean Euclidean correlation over epochs
        best_corr_hyp: Best hyperbolic correlation seen
        best_corr_euc: Best Euclidean correlation seen
        best_coverage: Best coverage percentage seen
        global_step: Global batch step counter
    """

    best_val_loss: float = field(default=float("inf"))
    patience_counter: int = field(default=0)

    # Coverage tracking
    coverage_A_history: list[int] = field(default_factory=list)
    coverage_B_history: list[int] = field(default_factory=list)

    # Entropy tracking
    H_A_history: list[float] = field(default_factory=list)
    H_B_history: list[float] = field(default_factory=list)

    # Correlation tracking (v5.10+)
    correlation_hyp_history: list[float] = field(default_factory=list)
    correlation_euc_history: list[float] = field(default_factory=list)
    best_corr_hyp: float = field(default=0.0)
    best_corr_euc: float = field(default=0.0)
    best_coverage: float = field(default=0.0)

    # Step counter for TensorBoard
    global_step: int = field(default=0)
    batches_per_epoch: int = field(default=0)

    def update_histories(
        self,
        H_A: float,
        H_B: float,
        coverage_A: int,
        coverage_B: int,
    ) -> None:
        """Update all tracked histories.

        Args:
            H_A: VAE-A entropy
            H_B: VAE-B entropy
            coverage_A: VAE-A coverage count
            coverage_B: VAE-B coverage count
        """
        self.H_A_history.append(H_A)
        self.H_B_history.append(H_B)
        self.coverage_A_history.append(coverage_A)
        self.coverage_B_history.append(coverage_B)

    def update_correlation(
        self,
        corr_A_hyp: float,
        corr_B_hyp: float,
        corr_A_euc: float,
        corr_B_euc: float,
    ) -> None:
        """Update correlation history and best values.

        Args:
            corr_A_hyp: VAE-A hyperbolic correlation
            corr_B_hyp: VAE-B hyperbolic correlation
            corr_A_euc: VAE-A Euclidean correlation
            corr_B_euc: VAE-B Euclidean correlation
        """
        corr_mean_hyp = (corr_A_hyp + corr_B_hyp) / 2
        corr_mean_euc = (corr_A_euc + corr_B_euc) / 2

        self.correlation_hyp_history.append(corr_mean_hyp)
        self.correlation_euc_history.append(corr_mean_euc)

        if corr_mean_hyp > self.best_corr_hyp:
            self.best_corr_hyp = corr_mean_hyp
        if corr_mean_euc > self.best_corr_euc:
            self.best_corr_euc = corr_mean_euc

    def update_coverage(self, cov_A: float, cov_B: float) -> None:
        """Update best coverage if current is better.

        Args:
            cov_A: VAE-A coverage percentage
            cov_B: VAE-B coverage percentage
        """
        current_coverage = (cov_A + cov_B) / 2
        if current_coverage > self.best_coverage:
            self.best_coverage = current_coverage

    def check_best(self, val_loss: float) -> bool:
        """Check if current validation loss is best.

        Args:
            val_loss: Current validation loss

        Returns:
            True if this is the best loss so far
        """
        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        return is_best

    def should_stop(self, patience: int) -> bool:
        """Check if early stopping criterion is met.

        Args:
            patience: Patience threshold

        Returns:
            True if should stop training
        """
        return self.patience_counter >= patience

    def has_coverage_plateaued(
        self,
        patience: int = 50,
        min_delta: float = 0.001,
    ) -> bool:
        """Check if coverage improvement has plateaued.

        Useful for manifold approach where 100% coverage is the goal.
        Triggers when coverage improvement over `patience` epochs
        is below threshold.

        Args:
            patience: Number of epochs to check for improvement
            min_delta: Minimum improvement fraction required

        Returns:
            True if coverage has plateaued, False otherwise
        """
        if len(self.coverage_A_history) < patience:
            return False

        # Use max of A and B as coverage metric
        recent_A = self.coverage_A_history[-patience:]
        recent_B = self.coverage_B_history[-patience:]
        recent_max = [max(a, b) for a, b in zip(recent_A, recent_B, strict=False)]

        # Compute improvement as fraction of total operations
        improvement = (recent_max[-1] - recent_max[0]) / N_TERNARY_OPERATIONS

        return improvement < min_delta

    def increment_step(self) -> int:
        """Increment and return global step counter.

        Returns:
            Current global step after increment
        """
        self.global_step += 1
        return self.global_step

    def get_metadata(self) -> dict[str, Any]:
        """Get all tracked metadata for checkpointing.

        Returns:
            Dict of all tracked metrics and history
        """
        return {
            "best_val_loss": self.best_val_loss,
            "H_A_history": self.H_A_history,
            "H_B_history": self.H_B_history,
            "coverage_A_history": self.coverage_A_history,
            "coverage_B_history": self.coverage_B_history,
            "correlation_hyp_history": self.correlation_hyp_history,
            "correlation_euc_history": self.correlation_euc_history,
            "best_corr_hyp": self.best_corr_hyp,
            "best_corr_euc": self.best_corr_euc,
            "best_coverage": self.best_coverage,
            "global_step": self.global_step,
        }

    def restore_from_metadata(self, metadata: dict[str, Any]) -> None:
        """Restore tracker state from metadata.

        Args:
            metadata: Dict from get_metadata()
        """
        self.best_val_loss = metadata.get("best_val_loss", float("inf"))
        self.H_A_history = metadata.get("H_A_history", [])
        self.H_B_history = metadata.get("H_B_history", [])
        self.coverage_A_history = metadata.get("coverage_A_history", [])
        self.coverage_B_history = metadata.get("coverage_B_history", [])
        self.correlation_hyp_history = metadata.get("correlation_hyp_history", [])
        self.correlation_euc_history = metadata.get("correlation_euc_history", [])
        self.best_corr_hyp = metadata.get("best_corr_hyp", 0.0)
        self.best_corr_euc = metadata.get("best_corr_euc", 0.0)
        self.best_coverage = metadata.get("best_coverage", 0.0)
        self.global_step = metadata.get("global_step", 0)

    def reset(self) -> None:
        """Reset all tracking state."""
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.coverage_A_history = []
        self.coverage_B_history = []
        self.H_A_history = []
        self.H_B_history = []
        self.correlation_hyp_history = []
        self.correlation_euc_history = []
        self.best_corr_hyp = 0.0
        self.best_corr_euc = 0.0
        self.best_coverage = 0.0
        self.global_step = 0
        self.batches_per_epoch = 0


__all__ = ["MetricsTracker"]
