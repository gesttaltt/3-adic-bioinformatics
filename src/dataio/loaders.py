# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Data loader factory functions.

This module provides factory functions for creating DataLoaders
with proper splitting and configuration.

Single responsibility: DataLoader creation only.
"""

import torch
from torch.utils.data import DataLoader, random_split

from .dataset import TernaryOperationDataset
from .generation import generate_all_ternary_operations


def create_ternary_data_loaders(
    batch_size: int = 256,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create train, validation, and test data loaders for ternary operations.

    Generates all 19,683 ternary operations and splits them into
    train/val/test sets.

    Args:
        batch_size: Batch size for all loaders
        train_split: Fraction for training (default 0.8)
        val_split: Fraction for validation (default 0.1)
        test_split: Fraction for test (default 0.1)
        num_workers: Number of data loading workers
        seed: Random seed for reproducible splits
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
        test_loader is None if test_split is 0

    Raises:
        ValueError: If splits don't sum to 1.0
    """
    # Validate splits
    total = train_split + val_split + test_split
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Splits must sum to 1.0, got {total}")

    # Generate dataset
    operations = generate_all_ternary_operations()
    dataset = TernaryOperationDataset(operations)

    # Compute sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size

    # Split dataset
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )

    val_loader = None
    if val_size > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )

    test_loader = None
    if test_size > 0:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
        )

    return train_loader, val_loader, test_loader


def get_data_loader_info(loader: DataLoader) -> dict:
    """Get information about a DataLoader.

    Args:
        loader: DataLoader to inspect

    Returns:
        Dict with loader statistics
    """
    dataset = loader.dataset
    size = len(dataset)

    return {
        "size": size,
        "batch_size": loader.batch_size,
        "num_batches": len(loader),
        "num_workers": loader.num_workers,
        "shuffle": isinstance(loader.sampler, torch.utils.data.sampler.RandomSampler),
    }


__all__ = ["create_ternary_data_loaders", "get_data_loader_info"]
