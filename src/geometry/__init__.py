# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Geometry module for hyperbolic and p-adic operations.

This module provides geoopt-backed implementations of hyperbolic geometry
operations, providing numerical stability and optimized performance.

Key components:
- PoincareBall: Manifold operations via geoopt
- Stable exp/log maps with automatic edge-case handling
- RiemannianAdam optimizer support
"""

from .holographic_poincare import (
    BoundaryPoint,
    HolographicLoss,
    HolographicPoincareManifold,
    HolographicProjection,
)
from .poincare import (
    ManifoldParameter,
    ManifoldTensor,
    PoincareModule,
    RiemannianAdam,
    RiemannianSGD,
    create_manifold_parameter,
    create_manifold_tensor,
    exp_map_zero,
    get_manifold,
    get_riemannian_optimizer,
    lambda_x,
    log_map_zero,
    mobius_add,
    parallel_transport,
    poincare_distance,
    poincare_distance_matrix,
    project_to_poincare,
)

__all__ = [
    # Core Poincare operations
    "get_manifold",
    "poincare_distance",
    "poincare_distance_matrix",
    "project_to_poincare",
    "exp_map_zero",
    "log_map_zero",
    "mobius_add",
    "lambda_x",
    "parallel_transport",
    "PoincareModule",
    "create_manifold_parameter",
    "create_manifold_tensor",
    "get_riemannian_optimizer",
    "ManifoldParameter",
    "ManifoldTensor",
    "RiemannianAdam",
    "RiemannianSGD",
    # Holographic extensions
    "HolographicPoincareManifold",
    "HolographicProjection",
    "HolographicLoss",
    "BoundaryPoint",
]
