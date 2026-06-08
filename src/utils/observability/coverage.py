# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Coverage evaluation - Decoupled from training loop.

This module provides efficient coverage evaluation separate from
the training monitor. It uses vectorized operations and can be
called asynchronously.

Usage:
    evaluator = CoverageEvaluator(model, device)

    # Evaluate coverage (can be async)
    count, pct = evaluator.evaluate('A', n_samples=1000)

    # Get detailed stats
    stats = evaluator.evaluate_detailed('A')
"""

from dataclasses import dataclass

import torch

from src.core import TERNARY


@dataclass
class CoverageStats:
    """Detailed coverage statistics."""

    unique_count: int
    coverage_pct: float
    total_possible: int = TERNARY.N_OPERATIONS

    # Valuation distribution (how many operations at each tree depth)
    valuation_histogram: dict[int, int] | None = None

    @property
    def missing_count(self) -> int:
        return self.total_possible - self.unique_count


class CoverageEvaluator:
    """Efficient coverage evaluator using vectorized operations.

    Evaluates how many unique ternary operations a model can generate.
    Uses torch.unique for O(n log n) instead of Python sets.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cuda",
        batch_size: int = 1000,
    ):
        """Initialize coverage evaluator.

        Args:
            model: VAE model with sample() method
            device: Device to run evaluation on
            batch_size: Samples per batch during evaluation
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size

    def evaluate(self, vae: str = "A", n_samples: int = 10000) -> tuple[int, float]:
        """Evaluate operation coverage.

        Uses vectorized torch.unique - O(n log n) instead of O(n) with Python sets,
        but much faster due to GPU acceleration.

        Args:
            vae: Which VAE to evaluate ('A' or 'B')
            n_samples: Number of samples to generate

        Returns:
            Tuple of (unique_count, coverage_percentage)
        """
        self.model.eval()

        with torch.no_grad():
            num_batches = max(1, n_samples // self.batch_size)

            # Collect all samples on GPU
            all_samples = []
            for _ in range(num_batches):
                samples = self.model.sample(self.batch_size, self.device, vae)
                samples_rounded = torch.round(samples).long()
                all_samples.append(samples_rounded)

            # Concatenate and find unique (vectorized)
            all_samples = torch.cat(all_samples, dim=0)
            unique_samples = torch.unique(all_samples, dim=0)
            unique_count = unique_samples.size(0)

        coverage_pct = (unique_count / TERNARY.N_OPERATIONS) * 100
        return unique_count, coverage_pct

    def evaluate_detailed(self, vae: str = "A", n_samples: int = 10000) -> CoverageStats:
        """Evaluate coverage with detailed statistics.

        Args:
            vae: Which VAE to evaluate ('A' or 'B')
            n_samples: Number of samples to generate

        Returns:
            CoverageStats with detailed information
        """
        self.model.eval()

        with torch.no_grad():
            num_batches = max(1, n_samples // self.batch_size)

            # Collect all samples on GPU
            all_samples = []
            for _ in range(num_batches):
                samples = self.model.sample(self.batch_size, self.device, vae)
                samples_rounded = torch.round(samples).long()
                all_samples.append(samples_rounded)

            # Concatenate and find unique
            all_samples = torch.cat(all_samples, dim=0)
            unique_samples = torch.unique(all_samples, dim=0)
            unique_count = unique_samples.size(0)

            # Convert to indices and compute valuation histogram
            indices = TERNARY.from_ternary(unique_samples)
            valuation_hist = TERNARY.valuation_histogram(indices)

        return CoverageStats(
            unique_count=unique_count,
            coverage_pct=(unique_count / TERNARY.N_OPERATIONS) * 100,
            valuation_histogram=valuation_hist,
        )

    def evaluate_union(self, n_samples: int = 10000) -> tuple[int, float, int, float]:
        """Evaluate coverage for both VAEs and their union.

        Args:
            n_samples: Number of samples per VAE

        Returns:
            Tuple of (unique_A, pct_A, unique_B, pct_B)
        """
        count_A, pct_A = self.evaluate("A", n_samples)
        count_B, pct_B = self.evaluate("B", n_samples)
        return count_A, pct_A, count_B, pct_B


def evaluate_model_coverage(
    model: torch.nn.Module,
    device: str = "cuda",
    n_samples: int = 10000,
    vae: str = "A",
) -> tuple[int, float]:
    """Convenience function for one-off coverage evaluation.

    Args:
        model: VAE model
        device: Device to use
        n_samples: Number of samples
        vae: Which VAE ('A' or 'B')

    Returns:
        Tuple of (unique_count, coverage_percentage)
    """
    evaluator = CoverageEvaluator(model, device)
    return evaluator.evaluate(vae, n_samples)


__all__ = ["CoverageEvaluator", "CoverageStats", "evaluate_model_coverage"]
