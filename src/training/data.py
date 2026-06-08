# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Data loading and sampling strategies for training.

This module re-exports data utilities from src/data for backwards compatibility.
The canonical implementations are now in src/data/stratified.py.

For new code, prefer importing directly from src.data:
    from src.data import TernaryDataset, StratifiedBatchSampler, create_stratified_batches
"""

# Re-export from canonical location (src/data)
from src.data import (
    StratifiedBatchSampler,
    TernaryDataset,
    create_stratified_batches,
)

__all__ = [
    "TernaryDataset",
    "StratifiedBatchSampler",
    "create_stratified_batches",
]
