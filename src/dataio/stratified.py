# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Stratified sampling for ternary operations.

This module provides stratified sampling strategies that ensure balanced
representation of valuation levels in training batches.

This is the canonical location for:
- TernaryDataset: Dataset wrapper for ternary operations
- StratifiedBatchSampler: PyTorch Sampler for valuation-balanced batches
- create_stratified_batches: Functional API for stratified sampling
- get_valuation_distribution: Utility for analyzing valuation distribution
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch
from torch.utils.data import Dataset, Sampler

from src.core import TERNARY


class TernaryDataset(Dataset):
    """Dataset for ternary operations.

    Wraps ternary operation tensors for use with DataLoader.

    Args:
        operations: Tensor of ternary operations
        indices: Tensor of operation indices
        valuations: Optional pre-computed valuations
        valuation_fn: Function to compute valuations if not provided
    """

    def __init__(
        self,
        operations: torch.Tensor,
        indices: torch.Tensor,
        valuations: torch.Tensor | None = None,
        valuation_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.operations = operations
        self.indices = indices

        if valuations is not None:
            self.valuations = valuations
        elif valuation_fn is not None:
            self.valuations = valuation_fn(indices)
        else:
            self.valuations = None

    def __len__(self) -> int:
        return len(self.operations)

    def __getitem__(self, idx: int):
        item = {
            "operation": self.operations[idx],
            "index": self.indices[idx],
        }
        if self.valuations is not None:
            item["valuation"] = self.valuations[idx]
        return item


class StratifiedBatchSampler(Sampler):
    """Stratified batch sampler ensuring valuation-balanced batches.

    V5.11.2 FIX: High-valuation points are extremely rare (v>=7 is ~9 out of 19683).
    Random sampling means most batches have NO high-valuation points.

    Solution: Stratified sampling - ensure each batch contains points from
    all valuation levels, with oversampling of rare high-valuation points.

    Args:
        valuations: Tensor of valuation levels for each sample
        batch_size: Target batch size
        high_v_threshold: Valuation level above which to oversample (default: 4)
        high_v_budget_ratio: Fraction of batch reserved for high-v (default: 0.2)
        drop_last: Whether to drop last incomplete batch (default: False)
    """

    def __init__(
        self,
        valuations: torch.Tensor,
        batch_size: int,
        high_v_threshold: int = 4,
        high_v_budget_ratio: float = 0.2,
        drop_last: bool = False,
    ):
        self.valuations = valuations.cpu()
        self.batch_size = batch_size
        self.high_v_threshold = high_v_threshold
        self.high_v_budget_ratio = high_v_budget_ratio
        self.drop_last = drop_last
        self.n_samples = len(valuations)

        # Pre-compute valuation groups
        self._build_valuation_groups()

    def _build_valuation_groups(self):
        """Group indices by valuation level."""
        self.valuation_groups = {}
        valuations_np = self.valuations.numpy()

        for i, v in enumerate(valuations_np):
            v = int(v)
            if v not in self.valuation_groups:
                self.valuation_groups[v] = []
            self.valuation_groups[v].append(i)

        # Convert to tensors
        for v in self.valuation_groups:
            self.valuation_groups[v] = torch.tensor(self.valuation_groups[v], dtype=torch.long)

        # Separate high and low valuation levels
        self.high_v_levels = [v for v in self.valuation_groups if v >= self.high_v_threshold]
        self.low_v_levels = [v for v in self.valuation_groups if v < self.high_v_threshold]

    def __iter__(self) -> Iterator[list[int]]:
        """Generate stratified batch indices."""
        high_v_budget = int(self.batch_size * self.high_v_budget_ratio)
        low_v_budget = self.batch_size - high_v_budget

        n_batches = len(self)

        for _ in range(n_batches):
            batch_indices = []

            # Sample from high-valuation levels (with replacement if needed)
            if self.high_v_levels:
                per_high_v = max(1, high_v_budget // len(self.high_v_levels))
                for v in self.high_v_levels:
                    group = self.valuation_groups[v]
                    # Sample with replacement for rare groups
                    sample_idx = torch.randint(0, len(group), (per_high_v,))
                    batch_indices.extend(group[sample_idx].tolist())

            # Sample from low-valuation levels (proportional to size)
            if self.low_v_levels:
                total_low = sum(len(self.valuation_groups[v]) for v in self.low_v_levels)
                for v in self.low_v_levels:
                    group = self.valuation_groups[v]
                    n_to_sample = max(1, int(low_v_budget * len(group) / total_low))
                    sample_idx = torch.randint(0, len(group), (n_to_sample,))
                    batch_indices.extend(group[sample_idx].tolist())

            # Trim to exact batch size or pad if needed
            if len(batch_indices) > self.batch_size:
                indices = torch.randperm(len(batch_indices))[: self.batch_size]
                batch_indices = [batch_indices[i] for i in indices.tolist()]
            elif len(batch_indices) < self.batch_size:
                # Pad with random samples
                extra = torch.randint(0, self.n_samples, (self.batch_size - len(batch_indices),))
                batch_indices.extend(extra.tolist())

            yield batch_indices

    def __len__(self) -> int:
        """Number of batches."""
        if self.drop_last:
            return self.n_samples // self.batch_size
        return (self.n_samples + self.batch_size - 1) // self.batch_size


def create_stratified_batches(
    indices: torch.Tensor,
    batch_size: int,
    device: str = "cpu",
    high_valuation_fraction: float = 0.2,
    high_valuation_threshold: int = 4,
) -> list[torch.Tensor]:
    """Create stratified batch indices ensuring all valuation levels represented.

    High-valuation points are extremely rare (v>=7 is ~9 out of 19683).
    Random sampling means most batches have NO high-valuation points.

    Solution: Stratified sampling - ensure each batch contains points from
    all valuation levels, with oversampling of rare high-valuation points.

    Args:
        indices: All operation indices
        batch_size: Target batch size
        device: Torch device
        high_valuation_fraction: Fraction of batch reserved for high-v points
        high_valuation_threshold: Valuation level considered "high"

    Returns:
        List of batch index tensors, each containing stratified samples

    Example:
        >>> indices = torch.arange(19683)
        >>> batches = create_stratified_batches(indices, batch_size=512)
        >>> for batch_idx in batches:
        ...     x_batch = x[batch_idx]
        ...     # Each batch has balanced valuation representation
    """
    n_samples = len(indices)
    valuations = TERNARY.valuation(indices).cpu().numpy()

    # Group indices by valuation level
    valuation_groups = {}
    for i, v in enumerate(valuations):
        v = int(v)
        if v not in valuation_groups:
            valuation_groups[v] = []
        valuation_groups[v].append(i)

    # Convert to tensors
    for v in valuation_groups:
        valuation_groups[v] = torch.tensor(valuation_groups[v], device=device)

    # Allocation: reserve fraction for high-v, rest proportional
    high_v_budget = int(batch_size * high_valuation_fraction)
    low_v_budget = batch_size - high_v_budget

    # Separate valuation levels
    high_v_levels = [v for v in valuation_groups if v >= high_valuation_threshold]
    low_v_levels = [v for v in valuation_groups if v < high_valuation_threshold]

    batches = []
    n_batches = (n_samples + batch_size - 1) // batch_size

    for _ in range(n_batches):
        batch_indices = []

        # Sample from high-valuation levels (with replacement if needed)
        if high_v_levels:
            per_high_v = max(1, high_v_budget // len(high_v_levels))
            for v in high_v_levels:
                group = valuation_groups[v]
                n_to_sample = min(per_high_v, len(group))
                if len(group) <= n_to_sample:
                    # Take all, with replacement if oversampling needed
                    sample_idx = torch.randint(0, len(group), (per_high_v,), device=device)
                else:
                    sample_idx = torch.randperm(len(group), device=device)[:n_to_sample]
                batch_indices.append(group[sample_idx])

        # Sample from low-valuation levels (proportional to size)
        if low_v_levels:
            total_low = sum(len(valuation_groups[v]) for v in low_v_levels)
            for v in low_v_levels:
                group = valuation_groups[v]
                n_to_sample = max(1, int(low_v_budget * len(group) / total_low))
                sample_idx = torch.randint(0, len(group), (n_to_sample,), device=device)
                batch_indices.append(group[sample_idx])

        # Combine and shuffle
        batch = torch.cat(batch_indices)

        # Trim to exact batch size or pad if needed
        if len(batch) > batch_size:
            batch = batch[torch.randperm(len(batch), device=device)[:batch_size]]
        elif len(batch) < batch_size:
            # Pad with random samples
            extra = torch.randint(0, n_samples, (batch_size - len(batch),), device=device)
            batch = torch.cat([batch, extra])

        batches.append(batch)

    return batches


def get_valuation_distribution(indices: torch.Tensor) -> dict:
    """Get distribution of valuation levels in indices.

    Args:
        indices: Tensor of operation indices

    Returns:
        Dictionary mapping valuation level to count

    Example:
        >>> indices = torch.arange(19683)
        >>> dist = get_valuation_distribution(indices)
        >>> print(dist)
        {0: 13122, 1: 4374, 2: 1458, ..., 9: 1}
    """
    valuations = TERNARY.valuation(indices).cpu().numpy()
    distribution = {}
    for v in valuations:
        v = int(v)
        distribution[v] = distribution.get(v, 0) + 1
    return dict(sorted(distribution.items()))
