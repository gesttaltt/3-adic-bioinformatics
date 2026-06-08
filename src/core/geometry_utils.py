# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Geometry utilities - Centralized geometric operations.

.. deprecated:: V5.12.2
    This module is DEPRECATED. Use `src.geometry` instead, which provides
    geoopt-backed implementations with better numerical stability.

    Migration guide:
        # OLD (deprecated):
        from src.core.geometry_utils import poincare_distance, exp_map_zero

        # NEW (recommended):
        from src.geometry import poincare_distance, exp_map_zero

    All functions in this module have equivalent implementations in src.geometry.
    This file will be archived in a future release.

---

This module provides unified geometric operations for Euclidean, hyperbolic,
and other geometric spaces used throughout the project.

Consolidated from:
- src/geometry/poincare.py (project_to_poincare, exp/log maps)
- src/graphs/hyperbolic_gnn.py (PoincareOperations, LorentzOperations)
- src/visualization/projections/poincare.py (projection utilities)

Key Features:
- Poincare ball operations (Mobius addition, exp/log maps)
- Lorentz model operations
- Safe projections to manifolds
- Curvature-aware computations

Usage:
    # DEPRECATED - use src.geometry instead
    from src.core.geometry_utils import (
        project_to_ball,
        mobius_add,
        exp_map_zero,
        log_map_zero,
        poincare_distance,
    )

Mathematical Background:
    The Poincare ball model represents hyperbolic space as the interior
    of a unit ball with the metric:
        ds^2 = (2 / (1 - |x|^2))^2 * |dx|^2

    The curvature parameter c controls the "hyperbolicness":
        c = 0: Euclidean space
        c > 0: Hyperbolic space (stronger curvature = tighter hierarchies)

References:
    - Nickel & Kiela (2017): Poincare Embeddings
    - Ganea et al. (2018): Hyperbolic Neural Networks
    - Chami et al. (2019): Hyperbolic Graph Convolutional Networks
"""

from __future__ import annotations

import warnings

import torch

# Emit deprecation warning on import
warnings.warn(
    "src.core.geometry_utils is deprecated as of V5.12.2. "
    "Use src.geometry instead:\n"
    "  from src.geometry import poincare_distance, exp_map_zero, log_map_zero\n"
    "This module will be archived in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from src.config.constants import (
    DEFAULT_CURVATURE,
    DEFAULT_MAX_RADIUS,
    EPSILON,
    EPSILON_NORM,
)

# ============================================================================
# Core Mathematical Operations
# ============================================================================


def lambda_x(
    x: torch.Tensor,
    c: float = DEFAULT_CURVATURE,
    keepdim: bool = True,
) -> torch.Tensor:
    """Compute conformal factor lambda_x = 2 / (1 - c * |x|^2).

    The conformal factor relates the hyperbolic metric to the Euclidean metric.

    Args:
        x: Points in the Poincare ball
        c: Curvature parameter (c > 0 for hyperbolic)
        keepdim: Keep the last dimension

    Returns:
        Conformal factor at each point
    """
    x_sqnorm = torch.sum(x * x, dim=-1, keepdim=keepdim)
    return 2.0 / (1.0 - c * x_sqnorm).clamp(min=EPSILON)


def gyration(
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    c: float = DEFAULT_CURVATURE,
) -> torch.Tensor:
    """Compute gyration gyr[u,v]w.

    Gyration is the hyperbolic analog of rotation that arises from
    the non-associativity of Mobius addition.

    Args:
        u, v, w: Points in the Poincare ball
        c: Curvature parameter

    Returns:
        Gyrated point gyr[u,v]w
    """
    u_dot_w = torch.sum(u * w, dim=-1, keepdim=True)
    v_dot_w = torch.sum(v * w, dim=-1, keepdim=True)
    u_dot_v = torch.sum(u * v, dim=-1, keepdim=True)

    u_sqnorm = torch.sum(u * u, dim=-1, keepdim=True)
    v_sqnorm = torch.sum(v * v, dim=-1, keepdim=True)

    c2 = c * c

    A = -c2 * u_dot_w * v_sqnorm + c * v_dot_w + 2 * c2 * u_dot_v * v_dot_w
    B = -c2 * v_dot_w * u_sqnorm - c * u_dot_w
    D = 1 + 2 * c * u_dot_v + c2 * u_sqnorm * v_sqnorm

    return w + 2 * (A * u + B * v) / D.clamp(min=EPSILON)


# ============================================================================
# Mobius Operations
# ============================================================================


def mobius_add(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = DEFAULT_CURVATURE,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Mobius addition in the Poincare ball.

    The hyperbolic analog of vector addition:
        x (+) y = ((1 + 2c<x,y> + c|y|^2)x + (1 - c|x|^2)y) /
                  (1 + 2c<x,y> + c^2|x|^2|y|^2)

    Args:
        x: First point(s) in Poincare ball
        y: Second point(s) in Poincare ball
        c: Curvature parameter
        eps: Small value for numerical stability

    Returns:
        x (+) y in the Poincare ball
    """
    x_sqnorm = torch.sum(x * x, dim=-1, keepdim=True)
    y_sqnorm = torch.sum(y * y, dim=-1, keepdim=True)
    xy_inner = torch.sum(x * y, dim=-1, keepdim=True)

    numerator = (1 + 2 * c * xy_inner + c * y_sqnorm) * x + (1 - c * x_sqnorm) * y
    denominator = 1 + 2 * c * xy_inner + c * c * x_sqnorm * y_sqnorm

    return numerator / (denominator.clamp(min=eps))


def mobius_scalar_mul(
    r: torch.Tensor,
    x: torch.Tensor,
    c: float = DEFAULT_CURVATURE,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Mobius scalar multiplication.

    The hyperbolic analog of scalar multiplication:
        r (*) x = tanh(r * arctanh(sqrt(c)|x|)) * x / (sqrt(c)|x|)

    Args:
        r: Scalar(s) to multiply by
        x: Point(s) in the Poincare ball
        c: Curvature parameter
        eps: Small value for numerical stability

    Returns:
        r (*) x in the Poincare ball
    """
    sqrt_c = c**0.5
    x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=eps)

    # arctanh(sqrt(c) * |x|)
    sqrt_c_x_norm = (sqrt_c * x_norm).clamp(max=1 - eps)
    arctanh_term = torch.arctanh(sqrt_c_x_norm)

    # tanh(r * arctanh(...))
    tanh_term = torch.tanh(r.unsqueeze(-1) * arctanh_term)

    # Scale
    result = tanh_term * x / (sqrt_c * x_norm)

    return result


def mobius_matvec(
    M: torch.Tensor,
    x: torch.Tensor,
    c: float = DEFAULT_CURVATURE,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Mobius matrix-vector multiplication.

    Applies a linear transformation in hyperbolic space:
        M (*) x = exp_0(M @ log_0(x))

    Args:
        M: Transformation matrix (..., out_dim, in_dim)
        x: Point(s) in Poincare ball (..., in_dim)
        c: Curvature parameter
        eps: Numerical stability

    Returns:
        Transformed point(s) in Poincare ball
    """
    # Map to tangent space at origin
    x_tangent = log_map_zero(x, c, eps)

    # Linear transformation in tangent space
    Mx = torch.einsum("...ij,...j->...i", M, x_tangent)

    # Map back to manifold
    return exp_map_zero(Mx, c, eps)


# ============================================================================
# Exponential and Logarithmic Maps
# ============================================================================


def exp_map_zero(
    v: torch.Tensor,
    c: float = DEFAULT_CURVATURE,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Exponential map from tangent space at origin.

    Maps a tangent vector at the origin to the manifold:
        exp_0(v) = tanh(sqrt(c)|v|) * v / (sqrt(c)|v|)

    Args:
        v: Tangent vector(s) at origin
        c: Curvature parameter
        eps: Numerical stability

    Returns:
        Point(s) on the Poincare ball
    """
    sqrt_c = c**0.5
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=eps)

    # tanh(sqrt(c) * |v|) * v / (sqrt(c) * |v|)
    return torch.tanh(sqrt_c * v_norm) * v / (sqrt_c * v_norm)


def log_map_zero(
    x: torch.Tensor,
    c: float = DEFAULT_CURVATURE,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Logarithmic map to tangent space at origin.

    Maps a point on the manifold to the tangent space at origin:
        log_0(x) = arctanh(sqrt(c)|x|) * x / (sqrt(c)|x|)

    Args:
        x: Point(s) on the Poincare ball
        c: Curvature parameter
        eps: Numerical stability

    Returns:
        Tangent vector(s) at origin
    """
    sqrt_c = c**0.5
    x_norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=eps)

    # Clamp to avoid arctanh of values >= 1
    sqrt_c_x_norm = (sqrt_c * x_norm).clamp(max=1 - eps)

    # arctanh(sqrt(c) * |x|) * x / (sqrt(c) * |x|)
    return torch.arctanh(sqrt_c_x_norm) * x / (sqrt_c * x_norm)


def exp_map(
    x: torch.Tensor,
    v: torch.Tensor,
    c: float = DEFAULT_CURVATURE,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Exponential map from tangent space at x.

    Args:
        x: Base point(s) on manifold
        v: Tangent vector(s) at x
        c: Curvature parameter
        eps: Numerical stability

    Returns:
        Point(s) on manifold
    """
    v_norm = torch.norm(v, dim=-1, keepdim=True).clamp(min=eps)
    lam = lambda_x(x, c)
    sqrt_c = c**0.5

    second_term = torch.tanh(sqrt_c * lam * v_norm / 2) * v / (sqrt_c * v_norm)

    return mobius_add(x, second_term, c, eps)


def log_map(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = DEFAULT_CURVATURE,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Logarithmic map from y to tangent space at x.

    Args:
        x: Base point(s) on manifold
        y: Target point(s) on manifold
        c: Curvature parameter
        eps: Numerical stability

    Returns:
        Tangent vector(s) at x pointing toward y
    """
    # Compute -x (+) y
    neg_x = -x
    diff = mobius_add(neg_x, y, c, eps)

    diff_norm = torch.norm(diff, dim=-1, keepdim=True).clamp(min=eps)
    lam = lambda_x(x, c)
    sqrt_c = c**0.5

    # Clamp arctanh argument
    sqrt_c_diff_norm = (sqrt_c * diff_norm).clamp(max=1 - eps)

    return 2 / (sqrt_c * lam) * torch.arctanh(sqrt_c_diff_norm) * diff / diff_norm


# ============================================================================
# Distance Functions
# ============================================================================


def poincare_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = DEFAULT_CURVATURE,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Compute geodesic distance in the Poincare ball.

    d(x, y) = (2/sqrt(c)) * arctanh(sqrt(c) * |(-x) (+) y|)

    Args:
        x: First point(s)
        y: Second point(s)
        c: Curvature parameter
        eps: Numerical stability

    Returns:
        Geodesic distance(s)
    """
    sqrt_c = c**0.5

    # Compute Mobius addition: (-x) (+) y
    neg_x = -x
    diff = mobius_add(neg_x, y, c, eps)
    diff_norm = torch.norm(diff, dim=-1)

    # Clamp for arctanh
    sqrt_c_diff_norm = (sqrt_c * diff_norm).clamp(max=1 - eps)

    return 2.0 / sqrt_c * torch.arctanh(sqrt_c_diff_norm)


def poincare_distance_squared(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = DEFAULT_CURVATURE,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Compute squared geodesic distance (more efficient for optimization)."""
    dist = poincare_distance(x, y, c, eps)
    return dist * dist


# ============================================================================
# Projection Functions
# ============================================================================


def project_to_ball(
    x: torch.Tensor,
    max_norm: float = DEFAULT_MAX_RADIUS,
    eps: float = EPSILON_NORM,
) -> torch.Tensor:
    """Project points to interior of unit ball.

    Clamps the norm of points to ensure they stay inside the ball.

    Args:
        x: Points to project
        max_norm: Maximum allowed norm (< 1 for Poincare ball)
        eps: Numerical stability

    Returns:
        Projected points with norm <= max_norm
    """
    norm = torch.norm(x, dim=-1, keepdim=True)
    scale = torch.where(
        norm > max_norm,
        max_norm / (norm + eps),
        torch.ones_like(norm),
    )
    return x * scale


def project_to_poincare(
    z: torch.Tensor,
    max_norm: float = DEFAULT_MAX_RADIUS,
    c: float = DEFAULT_CURVATURE,
    eps: float = EPSILON_NORM,
) -> torch.Tensor:
    """Project to Poincare ball with curvature.

    First normalizes by curvature, then clamps to max_norm.

    Args:
        z: Points to project
        max_norm: Maximum radius in ball
        c: Curvature parameter
        eps: Numerical stability

    Returns:
        Points inside Poincare ball
    """
    # The ball has radius 1/sqrt(c)
    ball_radius = 1.0 / (c**0.5) if c > 0 else float("inf")
    effective_max = min(max_norm, ball_radius - eps)

    return project_to_ball(z, effective_max, eps)


def project_polar(
    x: torch.Tensor,
    max_norm: float = DEFAULT_MAX_RADIUS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project to polar coordinates (radius, direction).

    Args:
        x: Points to decompose
        max_norm: Maximum radius

    Returns:
        Tuple of (radii, directions) where directions are unit vectors
    """
    norm = torch.norm(x, dim=-1, keepdim=True).clamp(min=EPSILON_NORM)
    direction = x / norm
    radius = norm.clamp(max=max_norm)

    return radius.squeeze(-1), direction


# ============================================================================
# Lorentz Model Operations
# ============================================================================


def lorentz_inner(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Minkowski inner product for Lorentz model.

    <x, y>_L = -x_0 * y_0 + sum(x_i * y_i)

    Args:
        x, y: Points in Lorentz model (first dim is time)

    Returns:
        Lorentz inner product
    """
    time_part = -x[..., 0:1] * y[..., 0:1]
    space_part = (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
    return (time_part + space_part).squeeze(-1)


def lorentz_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Distance in Lorentz model.

    d(x, y) = arccosh(-<x, y>_L)

    Args:
        x, y: Points in Lorentz model
        eps: Numerical stability

    Returns:
        Geodesic distance
    """
    inner = -lorentz_inner(x, y)
    # arccosh is only defined for x >= 1
    inner = inner.clamp(min=1 + eps)
    return torch.acosh(inner)


def lorentz_to_poincare(
    x: torch.Tensor,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Convert from Lorentz to Poincare model.

    Args:
        x: Points in Lorentz model (..., d+1)
        eps: Numerical stability

    Returns:
        Points in Poincare ball (..., d)
    """
    return x[..., 1:] / (x[..., 0:1] + 1 + eps)


def poincare_to_lorentz(
    x: torch.Tensor,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Convert from Poincare to Lorentz model.

    Args:
        x: Points in Poincare ball (..., d)
        eps: Numerical stability

    Returns:
        Points in Lorentz model (..., d+1)
    """
    x_sqnorm = (x * x).sum(dim=-1, keepdim=True)

    # Time component
    t = (1 + x_sqnorm) / (1 - x_sqnorm + eps)

    # Space components
    space = 2 * x / (1 - x_sqnorm + eps)

    return torch.cat([t, space], dim=-1)


# ============================================================================
# Parallel Transport
# ============================================================================


def parallel_transport(
    x: torch.Tensor,
    y: torch.Tensor,
    v: torch.Tensor,
    c: float = DEFAULT_CURVATURE,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Parallel transport tangent vector from x to y.

    Transports a tangent vector v at x to the tangent space at y
    along the geodesic connecting them.

    Args:
        x: Source point
        y: Target point
        v: Tangent vector at x
        c: Curvature parameter
        eps: Numerical stability

    Returns:
        Transported tangent vector at y
    """
    lam_x = lambda_x(x, c)
    lam_y = lambda_x(y, c)

    return v * lam_x / lam_y


# ============================================================================
# Utilities
# ============================================================================


def hyperbolic_midpoint(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = DEFAULT_CURVATURE,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Compute hyperbolic midpoint between x and y.

    Args:
        x, y: Points in Poincare ball
        c: Curvature parameter
        eps: Numerical stability

    Returns:
        Midpoint in Poincare ball
    """
    # Log map from x to y
    v = log_map(x, y, c, eps)

    # Half the tangent vector
    v_half = v / 2

    # Exp map back
    return exp_map(x, v_half, c, eps)


def hyperbolic_mean(
    points: torch.Tensor,
    weights: torch.Tensor | None = None,
    c: float = DEFAULT_CURVATURE,
    max_iter: int = 10,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Compute weighted Frechet mean in hyperbolic space.

    Uses iterative algorithm to find the point minimizing
    weighted sum of squared distances.

    Args:
        points: Points in Poincare ball (n, d)
        weights: Optional weights (n,). If None, uniform weights.
        c: Curvature parameter
        max_iter: Maximum iterations
        eps: Convergence threshold

    Returns:
        Frechet mean point
    """
    if weights is None:
        weights = torch.ones(len(points), device=points.device)
    weights = weights / weights.sum()

    # Initialize with Euclidean mean projected to ball
    mean = project_to_ball(points.mean(dim=0, keepdim=True))

    for _ in range(max_iter):
        # Compute weighted tangent vectors
        tangents = torch.stack(
            [w * log_map(mean, p.unsqueeze(0), c, eps).squeeze(0) for w, p in zip(weights, points, strict=False)]
        )

        # Update mean
        update = tangents.sum(dim=0, keepdim=True)
        mean = exp_map(mean, update, c, eps)
        mean = project_to_ball(mean)

        # Check convergence
        if torch.norm(update) < eps:
            break

    return mean.squeeze(0)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Core operations
    "lambda_x",
    "gyration",
    # Mobius operations
    "mobius_add",
    "mobius_scalar_mul",
    "mobius_matvec",
    # Exp/Log maps
    "exp_map_zero",
    "log_map_zero",
    "exp_map",
    "log_map",
    # Distance
    "poincare_distance",
    "poincare_distance_squared",
    # Projection
    "project_to_ball",
    "project_to_poincare",
    "project_polar",
    # Lorentz model
    "lorentz_inner",
    "lorentz_distance",
    "lorentz_to_poincare",
    "poincare_to_lorentz",
    # Parallel transport
    "parallel_transport",
    # Utilities
    "hyperbolic_midpoint",
    "hyperbolic_mean",
]
