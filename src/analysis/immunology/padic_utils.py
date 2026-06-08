# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""P-adic valuation and Goldilocks zone utilities for immunology.

This module provides immunology-specific p-adic utilities, building on
the core p-adic mathematics from src.core.padic_math.

The Goldilocks zone represents the immune escape region where antigens are:
- Different enough from self to be immunogenic
- Similar enough to self to cause cross-reactive autoimmunity

Note:
    Core p-adic operations are imported from src.core.padic_math.
    This module provides immunology-specific wrappers and defaults.

References:
    - 2006_Kozyrev_Padic_Analysis_Methods.md
    - Goldilocks zone concept from autoimmune research
"""

import torch

# Import all core p-adic operations from centralized module
from src.core.padic_math import (
    DEFAULT_P,
    PADIC_INFINITY,
    PADIC_INFINITY_INT,
    compute_hierarchical_embedding,
    padic_distance,
    padic_distance_vectorized,
    padic_valuation,
)
from src.core.padic_math import (
    compute_goldilocks_score as _core_goldilocks_score,
)
from src.core.padic_math import (
    is_in_goldilocks_zone as _core_is_in_goldilocks,
)


def compute_padic_valuation(
    n: int,
    p: int = DEFAULT_P,
    return_infinity: bool = True,
) -> int | float:
    """Compute p-adic valuation of integer n.

    The p-adic valuation v_p(n) is the highest power of p that divides n.

    This is a thin wrapper around the core padic_valuation function,
    providing backward compatibility for immunology modules.

    Args:
        n: Integer to compute valuation for
        p: Prime base (default: 3 for ternary)
        return_infinity: If True, return float("inf") for n=0;
                        if False, return 100 (legacy compatibility)

    Returns:
        P-adic valuation (int) or infinity for n=0
    """
    val = padic_valuation(n, p)
    if val == PADIC_INFINITY_INT and return_infinity:
        return PADIC_INFINITY
    return val


def compute_padic_distance(
    a: int,
    b: int,
    p: int = DEFAULT_P,
) -> float:
    """Compute p-adic distance between two integers.

    d_p(a, b) = p^(-v_p(a - b))

    This is a direct pass-through to core padic_distance.

    Args:
        a: First integer
        b: Second integer
        p: Prime base (default: 3)

    Returns:
        P-adic distance (0 to 1, where 0 means equal)
    """
    return padic_distance(a, b, p)


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

    return _core_goldilocks_score(distance, center=center, width=width)


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
    return _core_is_in_goldilocks(distance, center=center, width=width, threshold=threshold)


def compute_padic_distance_tensor(
    a: torch.Tensor,
    b: torch.Tensor,
    p: int = DEFAULT_P,
) -> torch.Tensor:
    """Compute p-adic distance between tensor elements.

    Vectorized version for batch processing.
    Uses centralized padic_distance_vectorized.

    Args:
        a: First tensor (any shape)
        b: Second tensor (same shape as a)
        p: Prime base

    Returns:
        Tensor of p-adic distances
    """
    return padic_distance_vectorized(a, b, p)


def compute_hierarchical_padic_embedding(
    indices: torch.Tensor,
    n_digits: int = 9,
    p: int = DEFAULT_P,
) -> torch.Tensor:
    """Compute hierarchical p-adic embedding from indices.

    Creates a multi-scale representation where each level corresponds
    to a p-adic digit, enabling hierarchical similarity comparisons.

    Uses centralized compute_hierarchical_embedding.

    Args:
        indices: Tensor of integer indices
        n_digits: Number of p-adic digits (depth)
        p: Prime base

    Returns:
        Tensor of shape (..., n_digits) with p-adic digits
    """
    return compute_hierarchical_embedding(indices, n_digits=n_digits, p=p)


__all__ = [
    # Re-exported from core
    "DEFAULT_P",
    "PADIC_INFINITY",
    "PADIC_INFINITY_INT",
    # Immunology-specific wrappers
    "compute_padic_valuation",
    "compute_padic_distance",
    "compute_goldilocks_score",
    "is_in_goldilocks_zone",
    "compute_padic_distance_tensor",
    "compute_hierarchical_padic_embedding",
]
