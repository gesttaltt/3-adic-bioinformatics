# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Core domain layer - Single Source of Truth.

This module contains the fundamental domain concepts that all other
modules depend on. Changes here affect the entire codebase.

Submodules:
    ternary: Ternary algebra (valuation, distance, operations)
    padic_math: P-adic mathematics (valuations, distances, embeddings)
    metrics: Hyperbolic geometry metrics (consolidated from metrics/)
    tensor_utils: Tensor manipulation utilities (broadcasting, indexing)
    geometry_utils: Geometric operations (hyperbolic, projections)
    config_base: Configuration base classes
    types: Type definitions and protocols

Usage:
    # Ternary operations
    from src.core import TERNARY
    v = TERNARY.valuation(indices)

    # P-adic operations
    from src.core import padic_valuation, padic_distance

    # Tensor utilities
    from src.core import pairwise_broadcast, batch_index_select

    # Geometry utilities
    from src.core import mobius_add, exp_map_zero, poincare_distance

    # Configuration
    from src.core import BaseConfig, PAdicConfig, TrainingConfig
"""

# =============================================================================
# Ternary Operations
# =============================================================================
# =============================================================================
# Configuration Base Classes
# =============================================================================
from .config_base import (
    BaseConfig,
    ContrastiveConfig,
    ExperimentConfig,
    HyperbolicConfig,
    MetaLearningConfig,
    PAdicConfig,
    PhysicsConfig,
    TrainingConfig,
)

# =============================================================================
# Geometry Utilities
# =============================================================================
from .geometry_utils import (
    exp_map,
    exp_map_zero,
    gyration,
    hyperbolic_mean,
    hyperbolic_midpoint,
    lambda_x,
    log_map,
    log_map_zero,
    lorentz_distance,
    lorentz_inner,
    lorentz_to_poincare,
    mobius_add,
    mobius_matvec,
    mobius_scalar_mul,
    parallel_transport,
    poincare_distance,
    poincare_distance_squared,
    poincare_to_lorentz,
    project_polar,
    project_to_ball,
    project_to_poincare,
)

# =============================================================================
# Metrics (consolidated from metrics/)
# =============================================================================
from .metrics import (
    ComprehensiveMetrics,
    compute_3adic_valuation,
    compute_comprehensive_metrics,
    compute_ranking_correlation_hyperbolic,
    poincare_distance,
    project_to_poincare,
)

# =============================================================================
# P-adic Mathematics
# =============================================================================
from .padic_math import (
    DEFAULT_P,
    PADIC_INFINITY,
    PADIC_INFINITY_INT,
    PAdicShiftResult,
    compute_goldilocks_score,
    compute_goldilocks_tensor,
    compute_hierarchical_embedding,
    is_in_goldilocks_zone,
    padic_digits,
    padic_distance,
    padic_distance_batch,
    padic_distance_matrix,
    padic_distance_vectorized,
    padic_norm,
    padic_shift,
    padic_valuation,
    padic_valuation_vectorized,
)

# =============================================================================
# Tensor Utilities
# =============================================================================
from .tensor_utils import (
    apply_mask,
    batch_index_select,
    clamp_norm,
    create_causal_mask,
    create_padding_mask,
    ensure_4d,
    flatten_batch,
    gather_from_indices,
    masked_mean,
    masked_softmax,
    pairwise_broadcast,
    pairwise_difference,
    safe_normalize,
    safe_normalize_l1,
    scatter_mean,
    soft_clamp,
    unflatten_batch,
)
from .ternary import (
    TERNARY,
    TernarySpace,
    distance,
    from_ternary,
    to_ternary,
    valuation,
)

# =============================================================================
# Type Definitions
# =============================================================================
from .types import (
    Array,
    Batch,
    Curvature,
    DataIterator,
    Decoder,
    Device,
    DType,
    Encoder,
    LossDict,
    LossFunction,
    LossValue,
    Manifold,
    MetricsDict,
    NamedBatch,
    Number,
    PAdicDigits,
    PAdicExpansion,
    PAdicIndex,
    Point,
    Radius,
    Result,
    Sampler,
    Shape,
    Task,
    TaskSampler,
    Tensor,
    TensorOrArray,
    VAELike,
    ValuationType,
    ensure_array,
    ensure_tensor,
    is_array,
    is_numeric,
    is_tensor,
)

# =============================================================================
# Module-level Constants
# =============================================================================
N_OPERATIONS = TERNARY.N_OPERATIONS
N_DIGITS = TERNARY.N_DIGITS
MAX_VALUATION = TERNARY.MAX_VALUATION

# =============================================================================
# Exports
# =============================================================================
__all__ = [
    # Ternary
    "TernarySpace",
    "TERNARY",
    "valuation",
    "distance",
    "to_ternary",
    "from_ternary",
    "N_OPERATIONS",
    "N_DIGITS",
    "MAX_VALUATION",
    # Hyperbolic metrics (consolidated from metrics/)
    "project_to_poincare",
    "poincare_distance",
    "compute_3adic_valuation",
    "ComprehensiveMetrics",
    "compute_comprehensive_metrics",
    "compute_ranking_correlation_hyperbolic",
    # P-adic math
    "DEFAULT_P",
    "PADIC_INFINITY",
    "PADIC_INFINITY_INT",
    "padic_valuation",
    "padic_valuation_vectorized",
    "padic_norm",
    "padic_distance",
    "padic_digits",
    "padic_shift",
    "PAdicShiftResult",
    "padic_distance_vectorized",
    "padic_distance_matrix",
    "padic_distance_batch",
    "compute_goldilocks_score",
    "compute_goldilocks_tensor",
    "is_in_goldilocks_zone",
    "compute_hierarchical_embedding",
    # Tensor utilities
    "pairwise_broadcast",
    "pairwise_difference",
    "batch_index_select",
    "safe_normalize",
    "safe_normalize_l1",
    "clamp_norm",
    "soft_clamp",
    "create_causal_mask",
    "create_padding_mask",
    "apply_mask",
    "masked_mean",
    "masked_softmax",
    "gather_from_indices",
    "scatter_mean",
    "flatten_batch",
    "unflatten_batch",
    "ensure_4d",
    # Geometry utilities
    "lambda_x",
    "gyration",
    "mobius_add",
    "mobius_scalar_mul",
    "mobius_matvec",
    "exp_map_zero",
    "log_map_zero",
    "exp_map",
    "log_map",
    "poincare_distance",
    "poincare_distance_squared",
    "project_to_ball",
    "project_to_poincare",
    "project_polar",
    "lorentz_inner",
    "lorentz_distance",
    "lorentz_to_poincare",
    "poincare_to_lorentz",
    "parallel_transport",
    "hyperbolic_midpoint",
    "hyperbolic_mean",
    # Configuration
    "BaseConfig",
    "PAdicConfig",
    "TrainingConfig",
    "HyperbolicConfig",
    "ContrastiveConfig",
    "MetaLearningConfig",
    "PhysicsConfig",
    "ExperimentConfig",
    # Types
    "Tensor",
    "Array",
    "TensorOrArray",
    "Number",
    "Shape",
    "DType",
    "Device",
    "PAdicIndex",
    "PAdicDigits",
    "ValuationType",
    "PAdicExpansion",
    "Curvature",
    "Radius",
    "Point",
    "LossValue",
    "LossDict",
    "MetricsDict",
    "Batch",
    "NamedBatch",
    "DataIterator",
    "Manifold",
    "Encoder",
    "Decoder",
    "VAELike",
    "LossFunction",
    "Sampler",
    "TaskSampler",
    "Task",
    "Result",
    "is_tensor",
    "is_array",
    "is_numeric",
    "ensure_tensor",
    "ensure_array",
]
