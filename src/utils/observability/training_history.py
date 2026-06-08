# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Training history tracking and state management.

This module provides a dedicated class for tracking training metrics,
history, and state. Extracted from TrainingMonitor for single responsibility.

Single responsibility: Training state and history management only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config.constants import (
    DEFAULT_PLATEAU_MIN_DELTA,
    DEFAULT_PLATEAU_PATIENCE,
    N_TERNARY_OPERATIONS,
)


@dataclass
class TrainingState:
    """Snapshot of training state at a given point.

    Useful for checkpointing and resuming training.
    """

    epoch: int
    global_step: int
    best_val_loss: float
    best_coverage: float
    best_corr_hyp: float
    best_corr_euc: float
    patience_counter: int


@dataclass
class TrainingHistory:
    """Tracks training metrics and history across epochs.

    This class is responsible for:
    - Maintaining metric histories
    - Tracking best values
    - Early stopping logic
    - Coverage plateau detection

    Attributes:
        best_val_loss: Best validation loss observed
        best_coverage: Best coverage percentage
        best_corr_hyp: Best hyperbolic correlation
        best_corr_euc: Best Euclidean correlation
        patience_counter: Epochs since last improvement
        global_step: Total batches processed
    """

    # Best metrics
    best_val_loss: float = float("inf")
    best_coverage: float = 0.0
    best_corr_hyp: float = 0.0
    best_corr_euc: float = 0.0

    # Early stopping
    patience_counter: int = 0
    global_step: int = 0

    # Histories
    coverage_A_history: list[int] = field(default_factory=list)
    coverage_B_history: list[int] = field(default_factory=list)
    H_A_history: list[float] = field(default_factory=list)
    H_B_history: list[float] = field(default_factory=list)
    correlation_hyp_history: list[float] = field(default_factory=list)
    correlation_euc_history: list[float] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)

    def update_entropies(self, H_A: float, H_B: float) -> None:
        """Update entropy histories.

        Args:
            H_A: VAE-A entropy
            H_B: VAE-B entropy
        """
        self.H_A_history.append(H_A)
        self.H_B_history.append(H_B)

    def update_coverage(self, coverage_A: int, coverage_B: int) -> None:
        """Update coverage histories.

        Args:
            coverage_A: VAE-A unique operations count
            coverage_B: VAE-B unique operations count
        """
        self.coverage_A_history.append(coverage_A)
        self.coverage_B_history.append(coverage_B)

        # Update best coverage (as percentage)
        current_pct = max(coverage_A, coverage_B) / N_TERNARY_OPERATIONS * 100
        if current_pct > self.best_coverage:
            self.best_coverage = current_pct

    def update_correlations(self, corr_hyp: float, corr_euc: float) -> None:
        """Update correlation histories.

        Args:
            corr_hyp: Mean hyperbolic correlation
            corr_euc: Mean Euclidean correlation
        """
        self.correlation_hyp_history.append(corr_hyp)
        self.correlation_euc_history.append(corr_euc)

        if corr_hyp > self.best_corr_hyp:
            self.best_corr_hyp = corr_hyp
        if corr_euc > self.best_corr_euc:
            self.best_corr_euc = corr_euc

    def update_loss(self, loss: float) -> None:
        """Update loss history.

        Args:
            loss: Current epoch loss
        """
        self.loss_history.append(loss)

    def check_best_val_loss(self, val_loss: float) -> bool:
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

    def should_stop_early(self, patience: int) -> bool:
        """Check if early stopping criterion is met.

        Args:
            patience: Patience threshold

        Returns:
            True if should stop training
        """
        return self.patience_counter >= patience

    def has_coverage_plateaued(
        self,
        patience: int = DEFAULT_PLATEAU_PATIENCE,
        min_delta: float = DEFAULT_PLATEAU_MIN_DELTA,
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

    def increment_step(self) -> None:
        """Increment the global step counter."""
        self.global_step += 1

    def get_state(self, epoch: int) -> TrainingState:
        """Get current training state for checkpointing.

        Args:
            epoch: Current epoch number

        Returns:
            TrainingState snapshot
        """
        return TrainingState(
            epoch=epoch,
            global_step=self.global_step,
            best_val_loss=self.best_val_loss,
            best_coverage=self.best_coverage,
            best_corr_hyp=self.best_corr_hyp,
            best_corr_euc=self.best_corr_euc,
            patience_counter=self.patience_counter,
        )

    def restore_state(self, state: TrainingState) -> None:
        """Restore training state from checkpoint.

        Args:
            state: TrainingState to restore from
        """
        self.global_step = state.global_step
        self.best_val_loss = state.best_val_loss
        self.best_coverage = state.best_coverage
        self.best_corr_hyp = state.best_corr_hyp
        self.best_corr_euc = state.best_corr_euc
        self.patience_counter = state.patience_counter

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dict of all tracked metrics and history
        """
        return {
            "best_val_loss": self.best_val_loss,
            "best_coverage": self.best_coverage,
            "best_corr_hyp": self.best_corr_hyp,
            "best_corr_euc": self.best_corr_euc,
            "patience_counter": self.patience_counter,
            "global_step": self.global_step,
            "coverage_A_history": self.coverage_A_history,
            "coverage_B_history": self.coverage_B_history,
            "H_A_history": self.H_A_history,
            "H_B_history": self.H_B_history,
            "correlation_hyp_history": self.correlation_hyp_history,
            "correlation_euc_history": self.correlation_euc_history,
            "loss_history": self.loss_history,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingHistory:
        """Create from dictionary.

        Args:
            data: Dictionary with history data

        Returns:
            TrainingHistory instance
        """
        history = cls()
        history.best_val_loss = data.get("best_val_loss", float("inf"))
        history.best_coverage = data.get("best_coverage", 0.0)
        history.best_corr_hyp = data.get("best_corr_hyp", 0.0)
        history.best_corr_euc = data.get("best_corr_euc", 0.0)
        history.patience_counter = data.get("patience_counter", 0)
        history.global_step = data.get("global_step", 0)
        history.coverage_A_history = data.get("coverage_A_history", [])
        history.coverage_B_history = data.get("coverage_B_history", [])
        history.H_A_history = data.get("H_A_history", [])
        history.H_B_history = data.get("H_B_history", [])
        history.correlation_hyp_history = data.get("correlation_hyp_history", [])
        history.correlation_euc_history = data.get("correlation_euc_history", [])
        history.loss_history = data.get("loss_history", [])
        return history

    def get_summary_stats(self) -> dict[str, float]:
        """Get summary statistics for logging.

        Returns:
            Dict of summary statistics
        """
        stats = {
            "best_val_loss": self.best_val_loss,
            "best_coverage": self.best_coverage,
            "best_corr_hyp": self.best_corr_hyp,
            "best_corr_euc": self.best_corr_euc,
            "epochs_trained": len(self.loss_history),
        }

        if self.coverage_A_history:
            stats["final_coverage_A"] = self.coverage_A_history[-1]
            stats["final_coverage_B"] = self.coverage_B_history[-1]
            stats["final_coverage_A_pct"] = self.coverage_A_history[-1] / N_TERNARY_OPERATIONS * 100
            stats["final_coverage_B_pct"] = self.coverage_B_history[-1] / N_TERNARY_OPERATIONS * 100

        return stats


__all__ = ["TrainingHistory", "TrainingState"]
