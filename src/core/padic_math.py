# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""P-adic mathematics - Single Source of Truth.

This module consolidates all p-adic mathematical operations from:
- src/utils/padic_shift.py (padic_valuation, padic_norm, padic_distance, padic_digits)
- src/analysis/immunology/padic_utils.py (compute_padic_valuation, compute_padic_distance)
- src/analysis/crispr/padic_distance.py (padic_valuation)

The 3-adic framework is fundamental to the project because:
- Ternary operations have natural 3-adic structure
- Codons have 3 positions (wobble effect)
- Hyperbolic geometry encodes p-adic hierarchies

Key Concepts:
    - Valuation v_p(n): Highest power of p dividing n
    - Norm |n|_p = p^(-v_p(n)): Ultrametric norm
    - Distance d_p(a,b) = |a-b|_p: Ultrametric distance
    - Goldilocks zone: Immune escape sweet spot

Usage:
    from src.core.padic_math import (
        padic_valuation,
        padic_distance,
        padic_distance_vectorized,
        compute_goldilocks_score,
    )

References:
    - 2006_Kozyrev_Padic_Analysis_Methods.md
    - 1975_Wong_CoEvolution_Theory.md
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

# ============================================================================
# Constants
# ============================================================================

DEFAULT_P = 3  # Default prime base (3-adic for ternary)
PADIC_INFINITY = float("inf")
PADIC_INFINITY_INT = 100  # Legacy integer representation


# ============================================================================
# Core P-adic Operations
# ============================================================================


def padic_valuation(n: int, p: int = DEFAULT_P) -> int:
    """Compute p-adic valuation v_p(n).

    The p-adic valuation is the largest power of p that divides n.
    v_p(0) is defined as infinity (returns 100 for compatibility).

    Args:
        n: Integer to compute valuation for
        p: Prime base (default 3)

    Returns:
        Valuation v_p(n), or 100 for n=0 (representing infinity)

    Examples:
        >>> padic_valuation(9, 3)   # 9 = 3^2
        2
        >>> padic_valuation(6, 3)   # 6 = 2 * 3^1
        1
        >>> padic_valuation(5, 3)   # 5 is not divisible by 3
        0
    """
    if n == 0:
        return PADIC_INFINITY_INT

    n = abs(n)
    v = 0
    while n % p == 0:
        v += 1
        n //= p
    return v


def padic_norm(n: int, p: int = DEFAULT_P) -> float:
    """Compute p-adic norm |n|_p = p^(-v_p(n)).

    The p-adic norm is the multiplicative inverse of the valuation.
    Smaller norm = more divisible by p = "closer to zero" in p-adic topology.

    Args:
        n: Integer to compute norm for
        p: Prime base (default 3)

    Returns:
        p-adic norm (0 for n=0)

    Examples:
        >>> padic_norm(9, 3)   # |9|_3 = 3^(-2) = 1/9
        0.1111...
        >>> padic_norm(1, 3)   # |1|_3 = 3^0 = 1
        1.0
    """
    if n == 0:
        return 0.0
    v = padic_valuation(n, p)
    if v >= PADIC_INFINITY_INT:
        return 0.0
    return float(p) ** (-v)


def padic_distance(a: int, b: int, p: int = DEFAULT_P) -> float:
    """Compute p-adic distance d_p(a, b) = |a - b|_p.

    This is an ultrametric: d(a,c) <= max(d(a,b), d(b,c))
    The strong triangle inequality means "all triangles are isoceles".

    Args:
        a: First integer
        b: Second integer
        p: Prime base (default 3)

    Returns:
        p-adic distance (0 if equal)

    Examples:
        >>> padic_distance(0, 9, 3)  # |9|_3 = 1/9
        0.1111...
        >>> padic_distance(1, 4, 3)  # |3|_3 = 1/3
        0.333...
    """
    if a == b:
        return 0.0
    return padic_norm(a - b, p)


def padic_digits(n: int, p: int = DEFAULT_P, n_digits: int = 4) -> list[int]:
    """Compute p-adic digit expansion of n.

    Returns the first n_digits of the p-adic expansion:
    n = a_0 + a_1*p + a_2*p^2 + ...

    Args:
        n: Integer to expand
        p: Prime base (default 3)
        n_digits: Number of digits to compute

    Returns:
        List of p-adic digits [a_0, a_1, a_2, ...]

    Examples:
        >>> padic_digits(10, 3, 4)  # 10 = 1 + 0*3 + 1*9 = [1,0,1,0]
        [1, 0, 1, 0]
    """
    n = abs(n)
    digits = []
    for _ in range(n_digits):
        digits.append(n % p)
        n //= p
    return digits


# ============================================================================
# Vectorized Operations (GPU-optimized)
# ============================================================================


def padic_valuation_vectorized(
    diff: torch.Tensor,
    p: int = DEFAULT_P,
    max_depth: int = 10,
) -> torch.Tensor:
    """Compute p-adic valuations for a tensor of differences.

    Vectorized version that avoids Python loops for GPU efficiency.

    Args:
        diff: Tensor of absolute differences
        p: Prime base (default 3)
        max_depth: Maximum valuation to check (default 10)

    Returns:
        Tensor of valuations
    """
    diff = diff.abs().long()
    valuations = torch.zeros_like(diff, dtype=torch.float)

    # Check divisibility by increasing powers of p
    for k in range(1, max_depth + 1):
        divisible = (diff % (p**k) == 0) & (diff > 0)
        valuations[divisible] = k

    return valuations


def padic_distance_vectorized(
    a: torch.Tensor,
    b: torch.Tensor,
    p: int = DEFAULT_P,
) -> torch.Tensor:
    """Compute p-adic distances between tensor elements.

    Fully vectorized for batch processing on GPU.

    Args:
        a: First tensor (any shape)
        b: Second tensor (same shape as a)
        p: Prime base (default 3)

    Returns:
        Tensor of p-adic distances
    """
    diff = (a - b).abs().long()
    valuations = padic_valuation_vectorized(diff, p)

    # Zero difference = zero distance
    zero_mask = diff == 0

    # Compute distances: p^(-v)
    distances = torch.pow(float(p), -valuations)
    distances[zero_mask] = 0.0

    return distances


def padic_distance_matrix(
    indices: torch.Tensor,
    p: int = DEFAULT_P,
) -> torch.Tensor:
    """Compute pairwise p-adic distance matrix.

    Vectorized computation avoiding O(n^3) loops.

    Args:
        indices: Tensor of indices (n,)
        p: Prime base (default 3)

    Returns:
        Distance matrix (n, n)
    """
    len(indices)
    indices = indices.long()

    # Compute all pairwise differences using broadcasting
    diff = torch.abs(indices.unsqueeze(0) - indices.unsqueeze(1))

    # Compute valuations
    valuations = padic_valuation_vectorized(diff, p)

    # Compute distances
    distances = torch.where(
        diff == 0,
        torch.zeros_like(valuations),
        torch.pow(float(p), -valuations),
    )

    return distances


def padic_distance_batch(
    indices: torch.Tensor,
    precomputed_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise distances using precomputed matrix.

    Uses advanced indexing instead of loops for O(n^2) instead of O(n^3).

    Args:
        indices: Codon indices (batch, seq_len)
        precomputed_matrix: Distance matrix (64, 64) for codons

    Returns:
        Distance matrix (batch, seq_len, seq_len)
    """
    batch_size, seq_len = indices.shape
    indices = indices.long()

    # Use advanced indexing: precomputed_matrix[i, j] for all pairs
    # Expand indices for broadcasting
    i_idx = indices.unsqueeze(2).expand(-1, -1, seq_len)  # (batch, seq, seq)
    j_idx = indices.unsqueeze(1).expand(-1, seq_len, -1)  # (batch, seq, seq)

    # Flatten, index, reshape
    flat_i = i_idx.reshape(-1)
    flat_j = j_idx.reshape(-1)
    distances = precomputed_matrix[flat_i, flat_j].reshape(batch_size, seq_len, seq_len)

    return distances


# ============================================================================
# Goldilocks Zone (Autoimmune Risk)
# ============================================================================


def compute_goldilocks_score(
    distance: float,
    center: float = 0.5,
    width: float = 0.15,
    normalize: bool = True,
) -> float:
    """Compute Goldilocks zone score for a given distance.

    The Goldilocks zone is the "just right" region for autoimmune risk:
    - Too close (distance < center - width): indistinguishable from self
    - Too far (distance > center + width): clearly foreign, no cross-reactivity
    - Just right (near center): immunogenic yet cross-reactive

    Args:
        distance: P-adic or structural distance from self
        center: Center of Goldilocks zone (default: 0.5)
        width: Width parameter (sigma) of Gaussian (default: 0.15)
        normalize: If True, normalize distance to [0, 1] first

    Returns:
        Score between 0 and 1 (1 = in zone, 0 = outside zone)
    """
    if normalize and distance > 1.0:
        distance = distance / (1.0 + distance)  # Sigmoid-like normalization

    # Gaussian scoring centered on the Goldilocks zone
    deviation = abs(distance - center)
    score = np.exp(-(deviation**2) / (2 * width**2))

    return float(score)


def is_in_goldilocks_zone(
    distance: float,
    center: float = 0.5,
    width: float = 0.15,
    threshold: float = 0.5,
) -> bool:
    """Check if distance falls within the Goldilocks zone.

    Args:
        distance: P-adic or structural distance
        center: Center of zone
        width: Width of zone
        threshold: Minimum score to be considered "in zone"

    Returns:
        True if in Goldilocks zone
    """
    score = compute_goldilocks_score(distance, center, width)
    return score >= threshold


def compute_goldilocks_tensor(
    distances: torch.Tensor,
    center: float = 0.5,
    width: float = 0.15,
) -> torch.Tensor:
    """Compute Goldilocks scores for a tensor of distances.

    Vectorized version for batch processing.

    Args:
        distances: Tensor of distances
        center: Center of Goldilocks zone
        width: Width parameter

    Returns:
        Tensor of Goldilocks scores
    """
    deviation = torch.abs(distances - center)
    scores = torch.exp(-(deviation**2) / (2 * width**2))
    return scores


# ============================================================================
# Hierarchical Embeddings
# ============================================================================


def compute_hierarchical_embedding(
    indices: torch.Tensor,
    n_digits: int = 9,
    p: int = DEFAULT_P,
) -> torch.Tensor:
    """Compute hierarchical p-adic embedding from indices.

    Creates a multi-scale representation where each level corresponds
    to a p-adic digit, enabling hierarchical similarity comparisons.

    Args:
        indices: Tensor of integer indices
        n_digits: Number of p-adic digits (depth)
        p: Prime base

    Returns:
        Tensor of shape (..., n_digits) with p-adic digits
    """
    shape = indices.shape
    embedding = torch.zeros(*shape, n_digits, dtype=torch.float, device=indices.device)

    remaining = indices.clone().long()
    for d in range(n_digits):
        embedding[..., d] = (remaining % p).float()
        remaining = remaining // p

    return embedding


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class PAdicShiftResult:
    """Result of p-adic shift operation."""

    shift_value: float  # The actual shift value
    valuation: int  # p-adic valuation
    digits: list[int]  # p-adic digit expansion
    canonical_form: str  # String representation


def padic_shift(
    value: int,
    shift_amount: int = 1,
    p: int = DEFAULT_P,
) -> PAdicShiftResult:
    """Perform p-adic shift operation.

    Shifts the p-adic representation by the specified amount.
    This is equivalent to multiplication/division by powers of p.

    Args:
        value: Input integer value
        shift_amount: Number of positions to shift (positive = right, negative = left)
        p: Prime base (default 3)

    Returns:
        PAdicShiftResult with shifted value and metadata
    """
    shifted = value // p**shift_amount if shift_amount >= 0 else value * p ** abs(shift_amount)

    digits = padic_digits(shifted, p)
    val = padic_valuation(shifted, p)
    canonical = f"{shifted}_({p})"

    return PAdicShiftResult(
        shift_value=float(shifted),
        valuation=val,
        digits=digits,
        canonical_form=canonical,
    )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Constants
    "DEFAULT_P",
    "PADIC_INFINITY",
    "PADIC_INFINITY_INT",
    # Core operations
    "padic_valuation",
    "padic_norm",
    "padic_distance",
    "padic_digits",
    "padic_shift",
    "PAdicShiftResult",
    # Vectorized operations
    "padic_valuation_vectorized",
    "padic_distance_vectorized",
    "padic_distance_matrix",
    "padic_distance_batch",
    # Goldilocks zone
    "compute_goldilocks_score",
    "is_in_goldilocks_zone",
    "compute_goldilocks_tensor",
    # Embeddings
    "compute_hierarchical_embedding",
]
