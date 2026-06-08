# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Data loaders and preprocessing for DDG prediction.

This module provides:
- ProThermLoader: Curated high-quality ProTherm mutations
- S669Loader: S669 benchmark dataset
- ProteinGymLoader: Large-scale ProteinGym dataset
- DatasetRegistry: Unified registry for all datasets
- preprocessing utilities
"""

from src.bioinformatics.data.dataset_registry import DatasetRegistry
from src.bioinformatics.data.preprocessing import (
    MutationFeatures,
    compute_features,
    compute_hyperbolic_features,
)
from src.bioinformatics.data.proteingym_loader import ProteinGymDataset, ProteinGymLoader
from src.bioinformatics.data.protherm_loader import ProThermDataset, ProThermLoader
from src.bioinformatics.data.s669_loader import S669Dataset, S669Loader

__all__ = [
    "ProThermLoader",
    "ProThermDataset",
    "S669Loader",
    "S669Dataset",
    "ProteinGymLoader",
    "ProteinGymDataset",
    "DatasetRegistry",
    "MutationFeatures",
    "compute_features",
    "compute_hyperbolic_features",
]
