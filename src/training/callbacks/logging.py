# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Logging callbacks for training observability.

This module provides callbacks for structured logging of training
events, replacing scattered print() statements throughout the codebase.

Callbacks:
    - LoggingCallback: General structured logging
    - TensorBoardCallback: TensorBoard integration
    - ProgressCallback: Progress bar/status updates

Usage:
    callback = LoggingCallback(log_interval=10)
    trainer = Trainer(callbacks=[callback])
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base import TrainingCallback

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

    from ..base import BaseTrainer

logger = logging.getLogger(__name__)


class LoggingCallback(TrainingCallback):
    """Structured logging callback for training events.

    Logs training progress to the standard Python logging system.
    Replaces ad-hoc print() statements with structured logs.
    """

    def __init__(
        self,
        log_interval: int = 10,
        log_level: int = logging.INFO,
        include_batch_logs: bool = False,
    ):
        """Initialize logging callback.

        Args:
            log_interval: Log every N batches (0 = epoch only)
            log_level: Logging level for messages
            include_batch_logs: Whether to log batch-level metrics
        """
        self.log_interval = log_interval
        self.log_level = log_level
        self.include_batch_logs = include_batch_logs

    def on_train_start(self, trainer: BaseTrainer) -> None:
        """Log training start."""
        logger.log(
            self.log_level,
            "Training started",
            extra={
                "event": "train_start",
                "total_epochs": getattr(trainer, "total_epochs", "unknown"),
                "device": str(getattr(trainer, "device", "unknown")),
            },
        )

    def on_train_end(self, trainer: BaseTrainer) -> None:
        """Log training end."""
        logger.log(
            self.log_level,
            "Training completed",
            extra={
                "event": "train_end",
                "final_epoch": getattr(trainer, "current_epoch", "unknown"),
            },
        )

    def on_epoch_start(self, epoch: int, trainer: BaseTrainer) -> None:
        """Log epoch start."""
        logger.debug(
            f"Epoch {epoch + 1} started",
            extra={"event": "epoch_start", "epoch": epoch},
        )

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: BaseTrainer) -> None:
        """Log epoch summary with key metrics."""
        # Extract key metrics for logging
        loss = metrics.get("loss", metrics.get("total_loss", 0))
        coverage = metrics.get("coverage", metrics.get("coverage_A", 0))
        correlation = metrics.get("correlation", metrics.get("correlation_A", 0))

        logger.log(
            self.log_level,
            f"Epoch {epoch + 1}: loss={loss:.4f}, coverage={coverage:.2f}%, correlation={correlation:.4f}",
            extra={
                "event": "epoch_end",
                "epoch": epoch,
                "metrics": metrics,
            },
        )

    def on_batch_end(
        self,
        batch_idx: int,
        loss: Any,
        metrics: dict[str, float],
        trainer: BaseTrainer,
    ) -> None:
        """Log batch progress if enabled."""
        if not self.include_batch_logs:
            return

        if self.log_interval > 0 and batch_idx % self.log_interval == 0:
            loss_val = loss.item() if hasattr(loss, "item") else loss
            logger.debug(
                f"Batch {batch_idx}: loss={loss_val:.4f}",
                extra={
                    "event": "batch_end",
                    "batch_idx": batch_idx,
                    "loss": loss_val,
                },
            )


class TensorBoardCallback(TrainingCallback):
    """TensorBoard logging callback.

    Writes metrics to TensorBoard for visualization.
    Supports scalars, histograms, and embeddings.
    """

    def __init__(
        self,
        writer: SummaryWriter | None = None,
        log_dir: str | None = None,
        histogram_interval: int = 10,
        embedding_interval: int = 50,
    ):
        """Initialize TensorBoard callback.

        Args:
            writer: Existing SummaryWriter (or created from log_dir)
            log_dir: Directory for TensorBoard logs
            histogram_interval: Epochs between histogram logs
            embedding_interval: Epochs between embedding logs
        """
        self.writer = writer
        self.log_dir = log_dir
        self.histogram_interval = histogram_interval
        self.embedding_interval = embedding_interval
        self._owns_writer = False

    def on_train_start(self, trainer: BaseTrainer) -> None:
        """Initialize TensorBoard writer if needed."""
        if self.writer is None and self.log_dir:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(self.log_dir)
            self._owns_writer = True

    def on_train_end(self, trainer: BaseTrainer) -> None:
        """Close TensorBoard writer if we own it."""
        if self._owns_writer and self.writer is not None:
            self.writer.close()

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: BaseTrainer) -> None:
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"train/{name}", value, epoch)

    def on_validation_end(self, epoch: int, metrics: dict[str, float], trainer: BaseTrainer) -> None:
        """Log validation metrics to TensorBoard."""
        if self.writer is None:
            return

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"val/{name}", value, epoch)


class ProgressCallback(TrainingCallback):
    """Progress indicator callback.

    Provides lightweight progress updates without full logging.
    Useful for interactive sessions.
    """

    def __init__(self, update_interval: int = 1):
        """Initialize progress callback.

        Args:
            update_interval: Epochs between progress updates
        """
        self.update_interval = update_interval
        self.total_epochs = 0

    def on_train_start(self, trainer: BaseTrainer) -> None:
        """Record total epochs."""
        self.total_epochs = getattr(trainer, "total_epochs", 0)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: BaseTrainer) -> None:
        """Print progress update."""
        if (epoch + 1) % self.update_interval != 0:
            return

        loss = metrics.get("loss", 0)
        coverage = metrics.get("coverage", metrics.get("coverage_A", 0))

        progress = (epoch + 1) / self.total_epochs * 100 if self.total_epochs else 0
        logger.info(
            f"[{progress:5.1f}%] Epoch {epoch + 1}/{self.total_epochs} | Loss: {loss:.4f} | Coverage: {coverage:.2f}%"
        )


__all__ = [
    "LoggingCallback",
    "TensorBoardCallback",
    "ProgressCallback",
]
