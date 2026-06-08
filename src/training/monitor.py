# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Training monitoring and logging (composition facade).

This module provides a unified TrainingMonitor that composes:
- MetricsTracker: History and best value tracking
- TensorBoardLogger: TensorBoard visualization
- FileLogger: File and console logging
- CoverageEvaluator: Model coverage evaluation

The TrainingMonitor maintains backward-compatible API while
delegating to specialized components for each concern.

Single responsibility: Orchestrating monitoring components.
"""

from __future__ import annotations

import datetime
from typing import Any

import torch

from .monitoring import (
    CoverageEvaluator,
    FileLogger,
    MetricsTracker,
    TensorBoardLogger,
)


class TrainingMonitor:
    """Monitors and logs training progress with unified observability.

    Composes specialized components for each monitoring concern
    while maintaining a backward-compatible API.

    Attributes:
        metrics: MetricsTracker for history and best values
        tensorboard: TensorBoardLogger for visualization
        file_logger: FileLogger for file/console output
        coverage_evaluator: CoverageEvaluator for model evaluation
        experiment_name: Name of this experiment
    """

    def __init__(
        self,
        eval_num_samples: int = 100000,
        tensorboard_dir: str | None = None,
        experiment_name: str | None = None,
        log_dir: str | None = "logs",
        log_to_file: bool = True,
    ):
        """Initialize training monitor.

        Args:
            eval_num_samples: Number of samples for coverage evaluation
            tensorboard_dir: Base directory for TensorBoard logs
            experiment_name: Name for this experiment run
            log_dir: Directory for persistent log files
            log_to_file: Whether to enable file logging
        """
        self.eval_num_samples = eval_num_samples

        # Generate experiment name if not provided
        if experiment_name is None:
            experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name

        # Initialize components
        self.metrics = MetricsTracker()
        self.file_logger = FileLogger(
            log_dir=log_dir,
            experiment_name=experiment_name,
            log_to_file=log_to_file,
        )
        self.tensorboard = TensorBoardLogger(
            tensorboard_dir=tensorboard_dir,
            experiment_name=experiment_name,
            log_callback=self._log,
        )
        self.coverage_evaluator = CoverageEvaluator(num_samples=eval_num_samples)

    # =========================================================================
    # Delegation properties for backward compatibility
    # =========================================================================

    @property
    def best_val_loss(self) -> float:
        """Best validation loss seen."""
        return self.metrics.best_val_loss

    @best_val_loss.setter
    def best_val_loss(self, value: float) -> None:
        self.metrics.best_val_loss = value

    @property
    def patience_counter(self) -> int:
        """Early stopping patience counter."""
        return self.metrics.patience_counter

    @patience_counter.setter
    def patience_counter(self, value: int) -> None:
        self.metrics.patience_counter = value

    @property
    def coverage_A_history(self) -> list[int]:
        """VAE-A coverage history."""
        return self.metrics.coverage_A_history

    @property
    def coverage_B_history(self) -> list[int]:
        """VAE-B coverage history."""
        return self.metrics.coverage_B_history

    @property
    def H_A_history(self) -> list[float]:
        """VAE-A entropy history."""
        return self.metrics.H_A_history

    @property
    def H_B_history(self) -> list[float]:
        """VAE-B entropy history."""
        return self.metrics.H_B_history

    @property
    def correlation_hyp_history(self) -> list[float]:
        """Hyperbolic correlation history."""
        return self.metrics.correlation_hyp_history

    @property
    def correlation_euc_history(self) -> list[float]:
        """Euclidean correlation history."""
        return self.metrics.correlation_euc_history

    @property
    def best_corr_hyp(self) -> float:
        """Best hyperbolic correlation."""
        return self.metrics.best_corr_hyp

    @property
    def best_corr_euc(self) -> float:
        """Best Euclidean correlation."""
        return self.metrics.best_corr_euc

    @property
    def best_coverage(self) -> float:
        """Best coverage percentage."""
        return self.metrics.best_coverage

    @property
    def global_step(self) -> int:
        """Global batch step counter."""
        return self.metrics.global_step

    @property
    def batches_per_epoch(self) -> int:
        """Batches per epoch."""
        return self.metrics.batches_per_epoch

    @batches_per_epoch.setter
    def batches_per_epoch(self, value: int) -> None:
        self.metrics.batches_per_epoch = value

    @property
    def writer(self):
        """TensorBoard SummaryWriter (for direct access)."""
        return self.tensorboard.writer

    @property
    def logger(self):
        """Python logger instance."""
        return self.file_logger.logger

    # =========================================================================
    # Logging methods
    # =========================================================================

    def _log(self, message: str) -> None:
        """Log message to both file and console."""
        self.file_logger.log(message)

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
        self.metrics.update_histories(H_A, H_B, coverage_A, coverage_B)

    def log_batch(
        self,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        loss: float,
        ce_A: float = 0.0,
        ce_B: float = 0.0,
        kl_A: float = 0.0,
        kl_B: float = 0.0,
        log_interval: int = 10,
    ) -> None:
        """Log batch-level metrics.

        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            total_batches: Total batches in epoch
            loss: Current batch loss
            ce_A: VAE-A cross-entropy
            ce_B: VAE-B cross-entropy
            kl_A: VAE-A KL divergence
            kl_B: VAE-B KL divergence
            log_interval: Log every N batches
        """
        step = self.metrics.increment_step()

        # TensorBoard batch logging
        self.tensorboard.log_batch(step, loss, ce_A, ce_B, kl_A, kl_B)

        # Console/file batch logging
        self.file_logger.log_batch(epoch, batch_idx, total_batches, loss, log_interval)

    def log_hyperbolic_batch(
        self,
        ranking_loss: float = 0.0,
        radial_loss: float = 0.0,
        hyp_kl_A: float = 0.0,
        hyp_kl_B: float = 0.0,
        centroid_loss: float = 0.0,
    ) -> None:
        """Log v5.10 hyperbolic metrics at batch level."""
        self.tensorboard.log_hyperbolic_batch(
            self.metrics.global_step,
            ranking_loss,
            radial_loss,
            hyp_kl_A,
            hyp_kl_B,
            centroid_loss,
        )

    def log_hyperbolic_epoch(
        self,
        epoch: int,
        corr_A_hyp: float,
        corr_B_hyp: float,
        corr_A_euc: float,
        corr_B_euc: float,
        mean_radius_A: float,
        mean_radius_B: float,
        ranking_weight: float,
        ranking_loss: float = 0.0,
        radial_loss: float = 0.0,
        hyp_kl_A: float = 0.0,
        hyp_kl_B: float = 0.0,
        centroid_loss: float = 0.0,
        homeostatic_metrics: dict[str, float] | None = None,
    ) -> None:
        """Log v5.10 hyperbolic metrics at epoch level."""
        # Update correlation tracking
        self.metrics.update_correlation(corr_A_hyp, corr_B_hyp, corr_A_euc, corr_B_euc)

        # TensorBoard logging
        self.tensorboard.log_hyperbolic_epoch(
            epoch,
            corr_A_hyp,
            corr_B_hyp,
            corr_A_euc,
            corr_B_euc,
            mean_radius_A,
            mean_radius_B,
            ranking_weight,
            ranking_loss,
            radial_loss,
            hyp_kl_A,
            hyp_kl_B,
            centroid_loss,
            homeostatic_metrics,
        )

    def log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        loss: float,
        cov_A: float,
        cov_B: float,
        corr_A_hyp: float,
        corr_B_hyp: float,
        corr_A_euc: float,
        corr_B_euc: float,
        mean_radius_A: float,
        mean_radius_B: float,
        ranking_weight: float,
        coverage_evaluated: bool = True,
        correlation_evaluated: bool = True,
        hyp_kl_A: float = 0.0,
        hyp_kl_B: float = 0.0,
        centroid_loss: float = 0.0,
        radial_loss: float = 0.0,
        homeostatic_metrics: dict[str, float] | None = None,
    ) -> None:
        """Log comprehensive epoch summary."""
        # Update best coverage tracking
        self.metrics.update_coverage(cov_A, cov_B)

        # File/console logging
        self.file_logger.log_epoch_summary(
            epoch,
            total_epochs,
            loss,
            cov_A,
            cov_B,
            corr_A_hyp,
            corr_B_hyp,
            corr_A_euc,
            corr_B_euc,
            mean_radius_A,
            mean_radius_B,
            ranking_weight,
            self.metrics.best_coverage,
            self.metrics.best_corr_hyp,
            coverage_evaluated,
            correlation_evaluated,
            hyp_kl_A,
            hyp_kl_B,
            centroid_loss,
            radial_loss,
            homeostatic_metrics,
        )

    def check_best(self, val_loss: float) -> bool:
        """Check if current validation loss is best."""
        return self.metrics.check_best(val_loss)

    def should_stop(self, patience: int) -> bool:
        """Check if early stopping criterion is met."""
        return self.metrics.should_stop(patience)

    def has_coverage_plateaued(
        self,
        patience: int = 50,
        min_delta: float = 0.001,
    ) -> bool:
        """Check if coverage improvement has plateaued."""
        return self.metrics.has_coverage_plateaued(patience, min_delta)

    def evaluate_coverage(
        self,
        model: torch.nn.Module,
        num_samples: int,
        device: str,
        vae: str = "A",
    ) -> tuple[int, float]:
        """Evaluate operation coverage."""
        return self.coverage_evaluator.evaluate(model, device, vae, num_samples)

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_losses: dict[str, Any],
        val_losses: dict[str, Any],
        unique_A: int,
        cov_A: float,
        unique_B: int,
        cov_B: float,
        is_best: bool,
        use_statenet: bool,
        grad_balance_achieved: bool,
    ) -> None:
        """Log epoch results to console and file."""
        self.file_logger.log_epoch(
            epoch,
            total_epochs,
            train_losses,
            val_losses,
            unique_A,
            cov_A,
            unique_B,
            cov_B,
            is_best,
            self.metrics.best_val_loss,
            use_statenet,
            grad_balance_achieved,
        )

    def log_tensorboard(
        self,
        epoch: int,
        train_losses: dict[str, Any],
        val_losses: dict[str, Any],
        unique_A: int,
        unique_B: int,
        cov_A: float,
        cov_B: float,
    ) -> None:
        """Log metrics to TensorBoard."""
        self.tensorboard.log_epoch(
            epoch,
            train_losses,
            val_losses,
            unique_A,
            unique_B,
            cov_A,
            cov_B,
        )

    def log_histograms(self, epoch: int, model: torch.nn.Module) -> None:
        """Log model weight histograms to TensorBoard."""
        self.tensorboard.log_histograms(epoch, model)

    def log_manifold_embedding(
        self,
        model: torch.nn.Module,
        epoch: int,
        device: str,
        n_samples: int = 5000,
        include_all: bool = False,
    ) -> None:
        """Log latent embeddings to TensorBoard."""
        self.tensorboard.log_manifold_embedding(model, epoch, device, n_samples, include_all)

    def close(self) -> None:
        """Close TensorBoard writer and flush all pending events."""
        self.tensorboard.close()

    def get_metadata(self) -> dict[str, Any]:
        """Get all tracked metadata for checkpointing."""
        return self.metrics.get_metadata()

    def print_training_summary(self) -> None:
        """Print training completion summary."""
        self.file_logger.print_training_summary(
            self.metrics.best_val_loss,
            self.metrics.best_corr_hyp,
            self.metrics.best_corr_euc,
            self.metrics.best_coverage,
            self.metrics.coverage_A_history,
            self.metrics.coverage_B_history,
        )
