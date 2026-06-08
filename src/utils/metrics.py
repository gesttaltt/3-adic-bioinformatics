# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Metrics for evaluating Ternary VAE performance."""

from collections import defaultdict

import numpy as np
import torch

from src.config.constants import N_TERNARY_OPERATIONS


def evaluate_coverage(samples: torch.Tensor, total_operations: int = N_TERNARY_OPERATIONS) -> tuple[int, float]:
    """Evaluate operation coverage from generated samples.

    Args:
        samples: Generated samples, shape (N, 9)
        total_operations: Total possible operations (default: N_TERNARY_OPERATIONS)

    Returns:
        Tuple of (unique_count, coverage_percentage)
    """
    # Round to nearest ternary value
    samples_rounded = torch.round(samples).long()

    # Use vectorized torch.unique for efficiency
    unique_samples = torch.unique(samples_rounded, dim=0)
    unique_count = unique_samples.size(0)
    coverage_pct = (unique_count / total_operations) * 100

    return unique_count, coverage_pct


def compute_latent_entropy(
    z: torch.Tensor,
    num_bins: int = 50,
    range_min: float = -3.0,
    range_max: float = 3.0,
) -> torch.Tensor:
    """Estimate latent entropy using histogram method.

    Args:
        z: Latent codes, shape (N, latent_dim)
        num_bins: Number of histogram bins
        range_min: Minimum value for histogram
        range_max: Maximum value for histogram

    Returns:
        torch.Tensor: Mean entropy across latent dimensions
    """
    batch_size, latent_dim = z.shape

    entropies = []
    for i in range(latent_dim):
        z_i = z[:, i]

        # Compute histogram
        hist = torch.histc(z_i, bins=num_bins, min=range_min, max=range_max)

        # Normalize to probability
        hist = hist / hist.sum()

        # Remove zeros to avoid log(0)
        hist = hist[hist > 0]

        # Compute entropy: H = -Σ p(x) log p(x)
        entropy = -(hist * torch.log(hist)).sum()
        entropies.append(entropy)

    return torch.stack(entropies).mean()


def compute_diversity_score(samples_A: torch.Tensor, samples_B: torch.Tensor) -> float:
    """Compute diversity score between two sets of samples.

    Diversity score measures how different the two sample sets are.
    Score of 1.0 = completely different, 0.0 = identical.

    Args:
        samples_A: Samples from VAE-A, shape (N, 9)
        samples_B: Samples from VAE-B, shape (M, 9)

    Returns:
        float: Diversity score in [0, 1]
    """
    # Round and get unique samples (vectorized)
    samples_A_rounded = torch.round(samples_A).long()
    samples_B_rounded = torch.round(samples_B).long()

    unique_A = torch.unique(samples_A_rounded, dim=0)
    unique_B = torch.unique(samples_B_rounded, dim=0)

    # Convert to numpy for efficient set creation
    np_A = unique_A.cpu().numpy()
    np_B = unique_B.cpu().numpy()

    # Use set comprehension with tobytes() for faster hashing
    ops_A = {row.tobytes() for row in np_A}
    ops_B = {row.tobytes() for row in np_B}

    # Compute Jaccard distance: 1 - |A ∩ B| / |A ∪ B|
    intersection = len(ops_A & ops_B)
    union = len(ops_A | ops_B)

    if union == 0:
        return 0.0

    jaccard_similarity = intersection / union
    diversity = 1.0 - jaccard_similarity

    return diversity


def compute_reconstruction_accuracy(inputs: torch.Tensor, outputs: torch.Tensor, threshold: float = 0.5) -> float:
    """Compute reconstruction accuracy.

    Args:
        inputs: Original inputs, shape (N, 9)
        outputs: Reconstructed outputs (logits or soft values), shape (N, 9, 3) or (N, 9)
        threshold: Threshold for rounding

    Returns:
        float: Accuracy in [0, 100]
    """
    if outputs.dim() == 3:
        # Logits: take argmax
        predicted = torch.argmax(outputs, dim=-1)  # (N, 9)
        # Convert back to {-1, 0, +1}
        predicted = predicted - 1
    else:
        # Soft values: round
        predicted = torch.round(outputs)

    # Convert inputs to same range if needed
    if inputs.max() <= 1.0 and inputs.min() >= 0.0:
        inputs = inputs * 2 - 1  # [0,1,2] → [-1,0,+1]

    # Compute accuracy
    correct = (predicted == inputs).float().sum()
    total = inputs.numel()
    accuracy = (correct / total) * 100

    return accuracy.item()


def analyze_coverage_distribution(samples: torch.Tensor, dimension: int = 9) -> dict[str, float | dict[int, float]]:
    """Analyze the distribution of covered operations.

    Args:
        samples: Generated samples, shape (N, 9)
        dimension: Operation dimension (9 for ternary)

    Returns:
        dict: Distribution statistics
    """
    samples_rounded = torch.round(samples).long()

    # Value distribution
    value_counts: dict[int, float] = {-1: 0.0, 0: 0.0, 1: 0.0}
    for v in [-1, 0, 1]:
        value_counts[v] = float((samples_rounded == v).sum().item())

    total = samples_rounded.numel()
    value_dist = {k: v / total for k, v in value_counts.items()}

    # Sparsity (fraction of zeros)
    sparsity = value_dist[0]

    # Balance (std of distribution)
    balance_std = float(np.std(list(value_dist.values())))

    # Per-dimension statistics
    dim_stats = []
    for d in range(dimension):
        dim_samples = samples_rounded[:, d]
        dim_unique = len(torch.unique(dim_samples))
        dim_stats.append(dim_unique)

    return {
        "value_distribution": value_dist,
        "sparsity": sparsity,
        "balance_std": balance_std,
        "avg_unique_per_dim": float(np.mean(dim_stats)),
        "min_unique_per_dim": min(dim_stats),
        "max_unique_per_dim": max(dim_stats),
    }


class CoverageTracker:
    """Track coverage over training epochs."""

    def __init__(self):
        self.history = defaultdict(list)
        self.best_coverage = 0
        self.best_epoch = 0

    def update(
        self,
        epoch: int,
        coverage_A: int,
        coverage_B: int,
        intersection: int | None = None,
    ):
        """Update coverage history.

        Args:
            epoch: Current epoch
            coverage_A: Coverage for VAE-A (unique operations covered)
            coverage_B: Coverage for VAE-B (unique operations covered)
            intersection: Number of operations covered by BOTH VAE-A and VAE-B.
                         If provided, computes true union. Otherwise uses max as estimate.
        """
        self.history["epoch"].append(epoch)
        self.history["coverage_A"].append(coverage_A)
        self.history["coverage_B"].append(coverage_B)

        # A8.1 FIX: Compute true union when intersection is available
        # Union formula: |A ∪ B| = |A| + |B| - |A ∩ B|
        if intersection is not None:
            coverage_union = coverage_A + coverage_B - intersection
        else:
            # Fallback: max is a lower bound (actual union >= max)
            coverage_union = max(coverage_A, coverage_B)

        self.history["coverage_union"].append(coverage_union)

        if coverage_union > self.best_coverage:
            self.best_coverage = coverage_union
            self.best_epoch = epoch

    def get_statistics(self) -> dict:
        """Get coverage statistics.

        Returns:
            dict: Statistics including best, current, improvement rate
        """
        if not self.history["epoch"]:
            return {}

        current_A = self.history["coverage_A"][-1]
        current_B = self.history["coverage_B"][-1]
        current_union = self.history["coverage_union"][-1]

        # Improvement rate (ops per epoch)
        if len(self.history["epoch"]) > 1:
            delta_epochs = self.history["epoch"][-1] - self.history["epoch"][0]
            delta_coverage = current_union - self.history["coverage_union"][0]
            improvement_rate = delta_coverage / delta_epochs if delta_epochs > 0 else 0
        else:
            improvement_rate = 0

        return {
            "current_coverage_A": current_A,
            "current_coverage_B": current_B,
            "current_coverage_union": current_union,
            "current_coverage_pct": (current_union / N_TERNARY_OPERATIONS) * 100,
            "best_coverage": self.best_coverage,
            "best_coverage_pct": (self.best_coverage / N_TERNARY_OPERATIONS) * 100,
            "best_epoch": self.best_epoch,
            "improvement_rate": improvement_rate,
            "epochs_tracked": len(self.history["epoch"]),
        }

    def has_plateaued(self, patience: int = 50, min_delta: float = 0.001) -> bool:
        """Check if coverage has plateaued.

        Args:
            patience: Number of epochs to check
            min_delta: Minimum improvement required

        Returns:
            bool: True if plateaued, False otherwise
        """
        if len(self.history["coverage_union"]) < patience:
            return False

        recent = self.history["coverage_union"][-patience:]
        improvement = (recent[-1] - recent[0]) / N_TERNARY_OPERATIONS

        return improvement < min_delta
