# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Checkpointing callbacks for model saving.

This module provides callbacks for saving and managing model checkpoints:
- Periodic checkpointing
- Best model checkpointing
- Checkpoint rotation (keep N most recent)

Usage:
    callback = CheckpointCallback(
        checkpoint_dir="checkpoints",
        save_best=True,
        monitor="coverage"
    )
    trainer = Trainer(callbacks=[callback])
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .base import TrainingCallback

if TYPE_CHECKING:
    from ..base import BaseTrainer

logger = logging.getLogger(__name__)


class CheckpointCallback(TrainingCallback):
    """Checkpoint saving callback.

    Saves model checkpoints periodically and/or when monitored metric improves.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        save_interval: int = 10,
        save_best: bool = True,
        monitor: str = "coverage",
        mode: str = "max",
        keep_last_n: int = 5,
        include_optimizer: bool = True,
    ):
        """Initialize checkpoint callback.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_interval: Save every N epochs (0 = best only)
            save_best: Whether to save best model
            monitor: Metric to monitor for best model
            mode: 'min' or 'max' for best model selection
            keep_last_n: Number of recent checkpoints to keep
            include_optimizer: Whether to include optimizer state
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_interval = save_interval
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.keep_last_n = keep_last_n
        self.include_optimizer = include_optimizer

        self.best_value: float | None = None
        self.saved_checkpoints: list[Path] = []

    def on_train_start(self, trainer: BaseTrainer) -> None:
        """Create checkpoint directory."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_value = None
        self.saved_checkpoints = []

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: BaseTrainer) -> None:
        """Save checkpoint if conditions are met."""
        # Periodic save
        if self.save_interval > 0 and (epoch + 1) % self.save_interval == 0:
            path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            self._save_checkpoint(trainer, path, epoch, metrics)
            self._rotate_checkpoints()

        # Best model save
        if self.save_best:
            current = metrics.get(self.monitor)
            if current is not None:
                is_best = False
                if (
                    self.best_value is None
                    or self.mode == "max"
                    and current > self.best_value
                    or self.mode == "min"
                    and current < self.best_value
                ):
                    is_best = True

                if is_best:
                    self.best_value = current
                    path = self.checkpoint_dir / "best_model.pt"
                    self._save_checkpoint(trainer, path, epoch, metrics)
                    logger.info(f"New best model: {self.monitor}={current:.4f} at epoch {epoch + 1}")

    def _save_checkpoint(
        self,
        trainer: BaseTrainer,
        path: Path,
        epoch: int,
        metrics: dict[str, float],
    ) -> None:
        """Save a checkpoint to disk."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": trainer.model.state_dict(),
            "metrics": metrics,
        }

        if self.include_optimizer and hasattr(trainer, "optimizer"):
            checkpoint["optimizer_state_dict"] = trainer.optimizer.state_dict()

        torch.save(checkpoint, path)
        self.saved_checkpoints.append(path)

        # Trigger callback notification
        trainer_callbacks = getattr(trainer, "callbacks", None)
        if trainer_callbacks:
            trainer_callbacks.on_checkpoint_save(epoch, str(path), trainer)

        logger.debug(f"Saved checkpoint: {path}")

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints to keep only last N."""
        if self.keep_last_n <= 0:
            return

        # Filter to only periodic checkpoints (not best_model.pt)
        periodic = [p for p in self.saved_checkpoints if "best_model" not in p.name]

        while len(periodic) > self.keep_last_n:
            oldest = periodic.pop(0)
            if oldest.exists():
                oldest.unlink()
                logger.debug(f"Removed old checkpoint: {oldest}")
            if oldest in self.saved_checkpoints:
                self.saved_checkpoints.remove(oldest)


class BestModelCallback(TrainingCallback):
    """Simplified callback for saving only the best model.

    This is a lighter-weight alternative to CheckpointCallback
    when you only need to track the best model.
    """

    def __init__(
        self,
        save_path: str = "best_model.pt",
        monitor: str = "coverage",
        mode: str = "max",
    ):
        """Initialize best model callback.

        Args:
            save_path: Path to save best model
            monitor: Metric to monitor
            mode: 'min' or 'max'
        """
        self.save_path = Path(save_path)
        self.monitor = monitor
        self.mode = mode
        self.best_value: float | None = None

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: BaseTrainer) -> None:
        """Check and save if this is the best model."""
        current = metrics.get(self.monitor)
        if current is None:
            return

        is_best = False
        if (
            self.best_value is None
            or self.mode == "max"
            and current > self.best_value
            or self.mode == "min"
            and current < self.best_value
        ):
            is_best = True

        if is_best:
            self.best_value = current
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": trainer.model.state_dict(),
                    "best_value": current,
                    "monitor": self.monitor,
                },
                self.save_path,
            )
            logger.info(f"Saved best model: {self.monitor}={current:.4f}")


__all__ = [
    "CheckpointCallback",
    "BestModelCallback",
]
