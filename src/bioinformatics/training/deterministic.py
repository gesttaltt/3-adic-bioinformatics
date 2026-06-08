# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Deterministic training utilities for reproducibility.

This module provides utilities to enable fully deterministic training,
ensuring reproducible results across runs.

Note: Full determinism may cause 10-20% performance reduction.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DeterministicConfig:
    """Configuration for deterministic training."""

    seed: int = 42
    deterministic: bool = True
    benchmark: bool = False
    hash_seed: bool = True

    # DataLoader settings
    num_workers: int = 0  # 0 for full determinism
    pin_memory: bool = True


def set_deterministic_mode(
    seed: int = 42,
    deterministic: bool = True,
    benchmark: bool = False,
    hash_seed: bool = True,
) -> int:
    """Enable fully deterministic training.

    Sets random seeds for Python, NumPy, and PyTorch. Configures
    cuDNN for deterministic behavior.

    Args:
        seed: Random seed
        deterministic: Enable cuDNN deterministic mode
        benchmark: Disable cuDNN benchmark (auto-tuning)
        hash_seed: Set Python hash seed

    Returns:
        The seed used
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # cuDNN settings
    if deterministic:
        torch.backends.cudnn.deterministic = True
    if not benchmark:
        torch.backends.cudnn.benchmark = False

    # Python hash seed (affects dict ordering, set operations)
    if hash_seed:
        os.environ["PYTHONHASHSEED"] = str(seed)

    return seed


def seed_worker(worker_id: int) -> None:
    """Worker init function for deterministic DataLoader.

    Each worker gets a unique but deterministic seed based on the
    base seed + worker_id, ensuring reproducibility across processes.

    Args:
        worker_id: DataLoader worker ID
    """
    # Get worker seed from PyTorch
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_deterministic_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """Create reproducible DataLoader.

    Uses a seeded generator for shuffling and initializes workers
    with deterministic seeds.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        seed: Random seed for shuffling
        num_workers: Number of worker processes (0 for full determinism)
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop last incomplete batch

    Returns:
        Deterministic DataLoader
    """
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
    )


class DeterministicTrainer:
    """Base class for deterministic training.

    Ensures reproducibility through:
    - Seeded random states
    - Deterministic cuDNN operations
    - Reproducible DataLoaders
    - Checkpointing with RNG states
    """

    def __init__(self, config: DeterministicConfig | None = None):
        """Initialize deterministic trainer.

        Args:
            config: DeterministicConfig
        """
        if config is None:
            config = DeterministicConfig()

        self.config = config
        self._seed = None

    def setup_determinism(self) -> int:
        """Set up deterministic training.

        Returns:
            The seed used
        """
        self._seed = set_deterministic_mode(
            seed=self.config.seed,
            deterministic=self.config.deterministic,
            benchmark=self.config.benchmark,
            hash_seed=self.config.hash_seed,
        )
        return self._seed

    def create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create deterministic DataLoader.

        Args:
            dataset: PyTorch Dataset
            batch_size: Batch size
            shuffle: Whether to shuffle

        Returns:
            Deterministic DataLoader
        """
        return get_deterministic_dataloader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=self.config.seed,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def save_rng_state(self) -> dict:
        """Save random number generator states.

        Returns:
            Dictionary with RNG states
        """
        state = {
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda_rng"] = torch.cuda.get_rng_state_all()
        return state

    def load_rng_state(self, state: dict) -> None:
        """Restore random number generator states.

        Args:
            state: Dictionary with RNG states
        """
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])

    def save_checkpoint(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict | None = None,
    ) -> None:
        """Save checkpoint with RNG states.

        Args:
            path: Checkpoint path
            model: Model to save
            optimizer: Optimizer state
            epoch: Current epoch
            metrics: Optional metrics dictionary
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_state": self.save_rng_state(),
            "seed": self._seed,
            "config": self.config.__dict__,
        }
        if metrics is not None:
            checkpoint["metrics"] = metrics

        torch.save(checkpoint, path)

    def load_checkpoint(
        self,
        path: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> dict:
        """Load checkpoint and restore RNG states.

        Args:
            path: Checkpoint path
            model: Model to load into
            optimizer: Optimizer to restore state

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "rng_state" in checkpoint:
            self.load_rng_state(checkpoint["rng_state"])

        return checkpoint


__all__ = [
    "DeterministicConfig",
    "set_deterministic_mode",
    "seed_worker",
    "get_deterministic_dataloader",
    "DeterministicTrainer",
]
