# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Training callback protocol and base implementation.

This module provides a callback system for training hooks, allowing
custom logic to be injected at various points in the training loop
without modifying the trainer code.

Design:
    - Protocol-based for flexibility (duck typing)
    - Optional methods via default implementations
    - Composable via CallbackList
    - No circular dependencies

Hook Points:
    - on_train_start: Before training begins
    - on_train_end: After training completes
    - on_epoch_start: Before each epoch
    - on_epoch_end: After each epoch
    - on_batch_start: Before each batch
    - on_batch_end: After each batch
    - on_validation_start: Before validation
    - on_validation_end: After validation

Usage:
    class MyCallback(TrainingCallback):
        def on_epoch_end(self, epoch, metrics, trainer):
            print(f"Epoch {epoch}: loss={metrics['loss']:.4f}")

    trainer = Trainer(callbacks=[MyCallback()])
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..base import BaseTrainer


class TrainingCallback(ABC):
    """Base class for training callbacks.

    All methods are optional - override only what you need.
    Methods receive the trainer instance for full access to state.
    """

    def on_train_start(self, trainer: BaseTrainer) -> None:
        """Called before training begins.

        Args:
            trainer: The trainer instance
        """
        pass

    def on_train_end(self, trainer: BaseTrainer) -> None:
        """Called after training completes (including early stopping).

        Args:
            trainer: The trainer instance
        """
        pass

    def on_epoch_start(self, epoch: int, trainer: BaseTrainer) -> None:
        """Called at the beginning of each epoch.

        Args:
            epoch: Current epoch number (0-indexed)
            trainer: The trainer instance
        """
        pass

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: BaseTrainer) -> bool | None:
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number (0-indexed)
            metrics: Dictionary of metrics from this epoch
            trainer: The trainer instance

        Returns:
            Optional bool - return True to stop training early
        """
        pass

    def on_batch_start(self, batch_idx: int, batch: torch.Tensor, trainer: BaseTrainer) -> None:
        """Called before processing each batch.

        Args:
            batch_idx: Current batch index within epoch
            batch: The batch data
            trainer: The trainer instance
        """
        pass

    def on_batch_end(
        self,
        batch_idx: int,
        loss: torch.Tensor,
        metrics: dict[str, float],
        trainer: BaseTrainer,
    ) -> None:
        """Called after processing each batch.

        Args:
            batch_idx: Current batch index within epoch
            loss: Batch loss value
            metrics: Batch metrics
            trainer: The trainer instance
        """
        pass

    def on_validation_start(self, epoch: int, trainer: BaseTrainer) -> None:
        """Called before validation begins.

        Args:
            epoch: Current epoch number
            trainer: The trainer instance
        """
        pass

    def on_validation_end(self, epoch: int, metrics: dict[str, float], trainer: BaseTrainer) -> None:
        """Called after validation completes.

        Args:
            epoch: Current epoch number
            metrics: Validation metrics
            trainer: The trainer instance
        """
        pass

    def on_checkpoint_save(self, epoch: int, path: str, trainer: BaseTrainer) -> None:
        """Called when a checkpoint is saved.

        Args:
            epoch: Current epoch number
            path: Path where checkpoint was saved
            trainer: The trainer instance
        """
        pass


class CallbackList:
    """Manages a collection of callbacks and dispatches events.

    This class aggregates multiple callbacks and calls them in order.
    If any on_epoch_end callback returns True, training stops.
    """

    def __init__(self, callbacks: list[TrainingCallback] | None = None):
        """Initialize callback list.

        Args:
            callbacks: List of callbacks to manage
        """
        self.callbacks = callbacks or []

    def add(self, callback: TrainingCallback) -> CallbackList:
        """Add a callback to the list.

        Args:
            callback: Callback to add

        Returns:
            Self for chaining
        """
        self.callbacks.append(callback)
        return self

    def remove(self, callback: TrainingCallback) -> CallbackList:
        """Remove a callback from the list.

        Args:
            callback: Callback to remove

        Returns:
            Self for chaining
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
        return self

    def on_train_start(self, trainer: BaseTrainer) -> None:
        """Dispatch on_train_start to all callbacks."""
        for callback in self.callbacks:
            callback.on_train_start(trainer)

    def on_train_end(self, trainer: BaseTrainer) -> None:
        """Dispatch on_train_end to all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_epoch_start(self, epoch: int, trainer: BaseTrainer) -> None:
        """Dispatch on_epoch_start to all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_start(epoch, trainer)

    def on_epoch_end(self, epoch: int, metrics: dict[str, float], trainer: BaseTrainer) -> bool:
        """Dispatch on_epoch_end to all callbacks.

        Returns:
            True if any callback requested early stopping
        """
        stop_training = False
        for callback in self.callbacks:
            result = callback.on_epoch_end(epoch, metrics, trainer)
            if result is True:
                stop_training = True
        return stop_training

    def on_batch_start(self, batch_idx: int, batch: torch.Tensor, trainer: BaseTrainer) -> None:
        """Dispatch on_batch_start to all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_start(batch_idx, batch, trainer)

    def on_batch_end(
        self,
        batch_idx: int,
        loss: torch.Tensor,
        metrics: dict[str, float],
        trainer: BaseTrainer,
    ) -> None:
        """Dispatch on_batch_end to all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, loss, metrics, trainer)

    def on_validation_start(self, epoch: int, trainer: BaseTrainer) -> None:
        """Dispatch on_validation_start to all callbacks."""
        for callback in self.callbacks:
            callback.on_validation_start(epoch, trainer)

    def on_validation_end(self, epoch: int, metrics: dict[str, float], trainer: BaseTrainer) -> None:
        """Dispatch on_validation_end to all callbacks."""
        for callback in self.callbacks:
            callback.on_validation_end(epoch, metrics, trainer)

    def on_checkpoint_save(self, epoch: int, path: str, trainer: BaseTrainer) -> None:
        """Dispatch on_checkpoint_save to all callbacks."""
        for callback in self.callbacks:
            callback.on_checkpoint_save(epoch, path, trainer)


__all__ = ["TrainingCallback", "CallbackList"]
