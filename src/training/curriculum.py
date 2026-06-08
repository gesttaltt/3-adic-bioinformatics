# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Adaptive curriculum learning for Ternary VAE training.

This module provides curriculum learning strategies that adaptively adjust
training parameters based on model performance.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CurriculumState:
    """Current state of the adaptive curriculum."""

    tau_frozen: bool = False
    frozen_tau: float | None = None
    should_stop: bool = False
    new_best: bool = False
    triggered_freeze: bool = False
    triggered_stop: bool = False


class AdaptiveCurriculum:
    """Adaptive curriculum with threshold-based tau freezing and early stopping.

    Key features:
    1. Tau freezes when hierarchy threshold is reached (stops pushing curriculum)
    2. Early stopping triggers after patience epochs without improvement
    3. Best model selected by composite score (hierarchy + loss balance)

    Example:
        >>> curriculum = AdaptiveCurriculum(
        ...     hierarchy_threshold=-0.70,
        ...     patience=20,
        ...     n_epochs=100
        ... )
        >>> for epoch in range(100):
        ...     tau = curriculum.compute_tau(epoch)
        ...     # ... train with tau ...
        ...     status = curriculum.update(epoch, radial_corr, loss)
        ...     if status.should_stop:
        ...         break
    """

    def __init__(
        self,
        hierarchy_threshold: float = -0.62,
        patience: int = 20,
        min_epochs: int = 30,
        n_epochs: int = 100,
        enabled: bool = True,
        loss_weight: float = 0.5,
    ):
        """Initialize adaptive curriculum.

        Args:
            hierarchy_threshold: Radial correlation to freeze tau at
            patience: Epochs without improvement before early stopping
            min_epochs: Minimum epochs before early stopping can trigger
            n_epochs: Total planned epochs (for tau schedule)
            enabled: Whether curriculum is active
            loss_weight: Weight for loss in composite score
        """
        self.hierarchy_threshold = hierarchy_threshold
        self.patience = patience
        self.min_epochs = min_epochs
        self.n_epochs = n_epochs
        self.enabled = enabled
        self.loss_weight = loss_weight

        # State
        self.tau_frozen = False
        self.frozen_tau: float | None = None
        self.frozen_epoch: int | None = None
        self.best_score = float("-inf")  # Higher is better (composite)
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.should_stop = False

    def compute_tau(self, epoch: int) -> float:
        """Compute tau for current epoch, respecting frozen state.

        Args:
            epoch: Current training epoch

        Returns:
            Tau value (0.0 to 1.0)
        """
        if self.tau_frozen and self.frozen_tau is not None:
            return self.frozen_tau
        # Default schedule: 0 -> 1 over 70% of training
        return min(1.0, epoch / (self.n_epochs * 0.7))

    def compute_composite_score(self, radial_corr: float, loss: float) -> float:
        """Compute composite score balancing hierarchy and loss.

        Score = -radial_corr - loss_weight * loss

        - radial_corr is negative (more negative = better), so -radial_corr is positive
        - loss is positive (lower = better), so we subtract it
        - loss_weight balances the two (default 0.5)

        Args:
            radial_corr: Radial correlation (negative = good hierarchy)
            loss: Training loss

        Returns:
            Composite score (higher is better)
        """
        return -radial_corr - self.loss_weight * loss

    def update(self, epoch: int, radial_corr: float, loss: float) -> CurriculumState:
        """Update curriculum state based on current metrics.

        Args:
            epoch: Current epoch
            radial_corr: Radial correlation metric
            loss: Training loss

        Returns:
            CurriculumState with status information
        """
        if not self.enabled:
            return CurriculumState()

        state = CurriculumState(
            tau_frozen=self.tau_frozen,
            frozen_tau=self.frozen_tau,
        )

        # Check if we should freeze tau
        if not self.tau_frozen and radial_corr <= self.hierarchy_threshold:
            self.tau_frozen = True
            self.frozen_tau = self.compute_tau(epoch)
            self.frozen_epoch = epoch
            state.triggered_freeze = True
            state.tau_frozen = True
            state.frozen_tau = self.frozen_tau

        # Compute composite score
        score = self.compute_composite_score(radial_corr, loss)

        # Check for improvement
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            state.new_best = True
        else:
            self.epochs_without_improvement += 1

        # Check for early stopping
        if epoch >= self.min_epochs and self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            state.should_stop = True
            state.triggered_stop = True

        return state

    def reset(self) -> None:
        """Reset curriculum state for new training run."""
        self.tau_frozen = False
        self.frozen_tau = None
        self.frozen_epoch = None
        self.best_score = float("-inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.should_stop = False
