# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Ternary operation dataset classes.

This module provides PyTorch dataset classes for ternary operations.

Single responsibility: Dataset definition only.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TernaryOperationDataset(Dataset):
    """Dataset of ternary operations.

    Each sample is a 9-element vector representing a ternary operation
    with values in {-1, 0, 1}.
    """

    def __init__(self, operations: np.ndarray | torch.Tensor):
        """Initialize dataset with operations.

        Args:
            operations: Array of operations (N, 9) with values in {-1, 0, 1}
        """
        if isinstance(operations, np.ndarray):
            self.operations = torch.FloatTensor(operations)
        else:
            self.operations = operations.float()

        # Validate shape
        if len(self.operations.shape) != 2 or self.operations.shape[1] != 9:
            raise ValueError(f"Operations must have shape (N, 9), got {self.operations.shape}")

        # Validate values
        unique_vals = torch.unique(self.operations)
        expected = torch.tensor([-1.0, 0.0, 1.0])
        if not torch.allclose(torch.sort(unique_vals)[0], expected):
            raise ValueError(f"Operations must contain only {{-1, 0, 1}}, got unique values: {unique_vals.tolist()}")

    def __len__(self) -> int:
        """Return number of operations in dataset."""
        return len(self.operations)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get operation at index.

        Args:
            idx: Index

        Returns:
            Tensor of shape (9,) with values in {-1, 0, 1}
        """
        return self.operations[idx]

    def get_statistics(self) -> dict:
        """Get dataset statistics.

        Returns:
            Dict with mean, std, and value distribution
        """
        return {
            "size": len(self),
            "shape": tuple(self.operations.shape),
            "mean": self.operations.mean().item(),
            "std": self.operations.std().item(),
            "min": self.operations.min().item(),
            "max": self.operations.max().item(),
            "value_counts": {
                "-1": (self.operations == -1).sum().item(),
                "0": (self.operations == 0).sum().item(),
                "1": (self.operations == 1).sum().item(),
            },
        }
