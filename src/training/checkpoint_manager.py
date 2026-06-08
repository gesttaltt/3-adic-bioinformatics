# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Checkpoint management for training.

This module handles saving and loading model checkpoints with metadata.

P3 FIX: Async checkpoint saving to avoid blocking training.

Single responsibility: Checkpoint persistence only.
"""

import copy
import logging
import queue
import threading
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class AsyncCheckpointSaver:
    """Background thread for non-blocking checkpoint saves.

    P3 FIX: torch.save() is blocking I/O that stalls training.
    This class offloads saves to a background thread.

    Usage:
        saver = AsyncCheckpointSaver()
        saver.save_async(checkpoint_dict, path)  # Returns immediately
        saver.shutdown()  # Call before exit
    """

    def __init__(self, max_queue_size: int = 3):
        """Initialize async saver.

        Args:
            max_queue_size: Max pending saves before blocking
        """
        self._queue: queue.Queue[tuple[dict[str, Any], Path]] = queue.Queue(maxsize=max_queue_size)
        self._running = True
        self._thread = threading.Thread(target=self._save_loop, daemon=True)
        self._thread.start()
        self._saves_completed = 0

    def save_async(self, checkpoint: dict[str, Any], path: Path) -> None:
        """Queue a checkpoint for async saving.

        Args:
            checkpoint: Checkpoint dict to save
            path: Destination path

        Note:
            If queue is full, this will block until space is available.
            This provides backpressure if saves are slower than training.
        """
        # Deep copy state dicts to avoid race conditions
        # Model/optimizer states are tensors that could be modified
        checkpoint_copy = self._deep_copy_checkpoint(checkpoint)
        self._queue.put((checkpoint_copy, path))

    def _deep_copy_checkpoint(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        """Deep copy checkpoint, handling tensor state dicts specially."""
        result = {}
        for key, value in checkpoint.items():
            if isinstance(value, dict):
                # State dicts need deep copy
                result[key] = copy.deepcopy(value)
            else:
                # Scalar values can be shallow copied
                result[key] = value
        return result

    def _save_loop(self) -> None:
        """Background thread that processes save queue."""
        while self._running or not self._queue.empty():
            try:
                checkpoint, path = self._queue.get(timeout=0.5)
                torch.save(checkpoint, path)
                self._saves_completed += 1
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"AsyncCheckpointSaver error: {e}")

    def shutdown(self, timeout: float = 10.0) -> None:
        """Shutdown the saver, waiting for pending saves.

        Args:
            timeout: Max seconds to wait for pending saves
        """
        self._running = False
        self._thread.join(timeout=timeout)

    @property
    def pending_saves(self) -> int:
        """Number of saves waiting in queue."""
        return self._queue.qsize()

    @property
    def saves_completed(self) -> int:
        """Total saves completed."""
        return self._saves_completed


class CheckpointManager:
    """Manages saving and loading model checkpoints.

    P3 FIX: Now supports async saving via background thread.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        checkpoint_freq: int = 10,
        async_save: bool = True,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            checkpoint_freq: Frequency (in epochs) to save numbered checkpoints
            async_save: Use async saving (default True for P3 optimization)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_freq = checkpoint_freq
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # P3 FIX: Async saver for non-blocking saves
        self._async_save = async_save
        self._async_saver = AsyncCheckpointSaver() if async_save else None

    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metadata: dict[str, Any],
        is_best: bool = False,
    ) -> None:
        """Save checkpoint with model, optimizer, and metadata.

        P3 FIX: Uses async saving by default to avoid blocking training.

        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer to save
            metadata: Additional metadata to save
            is_best: Whether this is the best checkpoint
        """
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            **metadata,
        }

        # Choose save method based on async setting
        save_fn = self._save_async if self._async_save else self._save_sync

        # Always save latest
        save_fn(checkpoint, self.checkpoint_dir / "latest.pt")

        # Save best if indicated
        if is_best:
            save_fn(checkpoint, self.checkpoint_dir / "best.pt")

        # Save numbered checkpoint at specified frequency
        if epoch % self.checkpoint_freq == 0:
            save_fn(checkpoint, self.checkpoint_dir / f"epoch_{epoch}.pt")

    def _save_sync(self, checkpoint: dict[str, Any], path: Path) -> None:
        """Synchronous (blocking) save."""
        torch.save(checkpoint, path)

    def _save_async(self, checkpoint: dict[str, Any], path: Path) -> None:
        """Asynchronous (non-blocking) save."""
        assert self._async_saver is not None, "Async saver not initialized"
        self._async_saver.save_async(checkpoint, path)

    def shutdown(self) -> None:
        """Shutdown async saver, waiting for pending saves."""
        if self._async_saver is not None:
            self._async_saver.shutdown()

    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        checkpoint_name: str = "latest",
        device: str = "cuda",
    ) -> dict[str, Any]:
        """Load checkpoint and restore model/optimizer state.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into (optional)
            checkpoint_name: Name of checkpoint ('latest', 'best', or 'epoch_N')
            device: Device to load checkpoint on

        Returns:
            Checkpoint metadata dict

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        # Construct checkpoint path
        if checkpoint_name in ["latest", "best"]:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"
        else:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint (weights_only=False needed for optimizer state, metrics)
        # Security note: Only load checkpoints from trusted sources
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Restore model state
        model.load_state_dict(checkpoint["model"])

        # Restore optimizer state if provided
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

        return checkpoint

    def list_checkpoints(self) -> dict[str, list]:
        """List all available checkpoints.

        Returns:
            Dict with 'special' (latest/best) and 'epochs' (numbered) checkpoints
        """
        special = []
        epochs = []

        for ckpt_file in self.checkpoint_dir.glob("*.pt"):
            name = ckpt_file.stem
            if name in ["latest", "best"]:
                special.append(name)
            elif name.startswith("epoch_"):
                epochs.append(name)

        return {
            "special": sorted(special),
            "epochs": sorted(epochs, key=lambda x: int(x.split("_")[1])),
        }

    def get_latest_epoch(self) -> int | None:
        """Get the latest saved epoch number.

        Returns:
            Latest epoch number or None if no checkpoints exist
        """
        latest_path = self.checkpoint_dir / "latest.pt"
        if not latest_path.exists():
            return None

        # Security note: Only load checkpoints from trusted sources
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        return checkpoint.get("epoch")
