# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Utilities for disease analysis."""

from src.diseases.utils.synthetic_data import (
    augment_synthetic_dataset,
    create_mutation_based_dataset,
    generate_correlated_targets,
)

__all__ = [
    "generate_correlated_targets",
    "augment_synthetic_dataset",
    "create_mutation_based_dataset",
]
