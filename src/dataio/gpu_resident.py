# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""GPU-Resident Dataset for ternary operations.

P2 FIX: The full dataset (19,683 samples × 9 floats = 710 KB) fits trivially
in GPU memory. Instead of transferring batches each iteration, we load
the entire dataset to GPU once and sample from it.

This eliminates:
- Per-batch CPU→GPU transfers
- DataLoader overhead
- Memory fragmentation from repeated allocations

Usage:
    dataset = GPUResidentTernaryDataset(device='cuda')
    for epoch in range(epochs):
        for batch in dataset.get_batches(batch_size=256, shuffle=True):
            x, indices = batch  # Already on GPU
            ...
"""

from collections.abc import Iterator

import torch

from ..core import TERNARY


class GPUResidentTernaryDataset:
    """GPU-resident dataset for ternary operations.

    Loads all 19,683 ternary operations to GPU memory once at initialization.
    Provides an iterator interface for batched access.

    Memory footprint:
        - Ternary data: 19,683 × 9 × 4 bytes = 708 KB
        - Indices: 19,683 × 8 bytes = 157 KB
        - Total: ~865 KB (trivial for any GPU)

    Benefits over DataLoader:
        - Zero per-batch transfer overhead
        - No DataLoader worker threads
        - Simpler memory allocation pattern
        - Direct tensor indexing (faster than Dataset.__getitem__)
    """

    def __init__(
        self,
        device: str = "cuda",
        train_split: float = 0.8,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        """Initialize GPU-resident dataset.

        Args:
            device: Device to store data on ('cuda' or 'cpu')
            train_split: Fraction for training
            val_split: Fraction for validation (rest is test)
            seed: Random seed for reproducible splits
        """
        self.device = torch.device(device)

        # Use core module's precomputed ternary representations
        self.all_data = TERNARY.all_ternary(self.device)
        self.all_indices = TERNARY.all_indices(self.device)

        # Create reproducible split
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(TERNARY.N_OPERATIONS, generator=generator, device="cpu")

        n_train = int(train_split * TERNARY.N_OPERATIONS)
        n_val = int(val_split * TERNARY.N_OPERATIONS)

        train_perm = perm[:n_train].to(self.device)
        val_perm = perm[n_train : n_train + n_val].to(self.device)
        test_perm = perm[n_train + n_val :].to(self.device)

        # Store split indices
        self.train_indices = train_perm
        self.val_indices = val_perm
        self.test_indices = test_perm

        # Pre-index split data for fast access
        self.train_data = self.all_data[train_perm]
        self.val_data = self.all_data[val_perm]
        self.test_data = self.all_data[test_perm]

        # Store original indices for p-adic losses
        self.train_original_indices = self.all_indices[train_perm]
        self.val_original_indices = self.all_indices[val_perm]
        self.test_original_indices = self.all_indices[test_perm]

    def get_batches(
        self,
        split: str = "train",
        batch_size: int = 256,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over batches.

        Args:
            split: 'train', 'val', or 'test'
            batch_size: Batch size
            shuffle: Whether to shuffle indices each epoch
            drop_last: Drop last incomplete batch

        Yields:
            Tuple of (batch_data, batch_indices) - both already on GPU
        """
        if split == "train":
            data = self.train_data
            indices = self.train_original_indices
        elif split == "val":
            data = self.val_data
            indices = self.val_original_indices
        else:
            data = self.test_data
            indices = self.test_original_indices

        n_samples = data.size(0)

        if shuffle:
            perm = torch.randperm(n_samples, device=self.device)
            data = data[perm]
            indices = indices[perm]

        for start in range(0, n_samples, batch_size):
            end = start + batch_size
            if end > n_samples and drop_last:
                break
            yield data[start:end], indices[start:end]

    def get_split_size(self, split: str = "train") -> int:
        """Get size of a split."""
        if split == "train":
            return len(self.train_indices)
        elif split == "val":
            return len(self.val_indices)
        else:
            return len(self.test_indices)

    def num_batches(self, split: str = "train", batch_size: int = 256) -> int:
        """Get number of batches for a split."""
        return (self.get_split_size(split) + batch_size - 1) // batch_size

    def __len__(self) -> int:
        """Return total dataset size for compatibility with standard DataLoader pattern."""
        return len(self.train_indices) + len(self.val_indices) + len(self.test_indices)


class GPUBatchIterator:
    """Drop-in replacement for DataLoader iteration.

    Provides compatible interface for existing training loops.

    Usage:
        loader = GPUBatchIterator(dataset, 'train', batch_size=256)
        for batch_data, batch_indices in loader:
            ...
    """

    def __init__(
        self,
        dataset: GPUResidentTernaryDataset,
        split: str = "train",
        batch_size: int = 256,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.split = split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        return self.dataset.get_batches(self.split, self.batch_size, self.shuffle, self.drop_last)

    def __len__(self):
        return self.dataset.num_batches(self.split, self.batch_size)


def create_gpu_resident_loaders(
    device: str = "cuda",
    batch_size: int = 256,
    train_split: float = 0.8,
    val_split: float = 0.1,
    seed: int = 42,
) -> tuple["GPUBatchIterator", "GPUBatchIterator", "GPUBatchIterator"]:
    """Create GPU-resident train/val/test loaders.

    Drop-in replacement for create_ternary_data_loaders().

    Args:
        device: GPU device
        batch_size: Batch size for all splits
        train_split: Training fraction
        val_split: Validation fraction
        seed: Random seed

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset = GPUResidentTernaryDataset(device=device, train_split=train_split, val_split=val_split, seed=seed)

    train_loader = GPUBatchIterator(dataset, "train", batch_size, shuffle=True)
    val_loader = GPUBatchIterator(dataset, "val", batch_size, shuffle=False)
    test_loader = GPUBatchIterator(dataset, "test", batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


__all__ = [
    "GPUResidentTernaryDataset",
    "GPUBatchIterator",
    "create_gpu_resident_loaders",
]
