# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Centralized configuration module.

This module provides:
- Constants: All magic numbers and defaults in one place
- Schema: Typed, validated configuration classes
- Loader: YAML + environment variable configuration loading

Usage:
    from src.config import load_config, TrainingConfig, EPSILON

    # Load configuration
    config = load_config("config.yaml")

    # Access constants
    from src.config.constants import EPSILON, DEFAULT_CURVATURE

    # Create config programmatically
    config = TrainingConfig(epochs=500, batch_size=128)
"""

# Constants
from .constants import (
    # Coverage tracking
    COVERAGE_TARGET_PCT,
    # Geometry
    CURVATURE_MAX,
    CURVATURE_MIN,
    # Training
    DEFAULT_BATCH_SIZE,
    # Observability
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_CURVATURE,
    DEFAULT_EMBEDDING_INTERVAL,
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_SAMPLES,
    DEFAULT_FREE_BITS,
    # Hyperbolic geometry
    DEFAULT_GEODESIC_WEIGHT,
    DEFAULT_GRAD_CLIP,
    # Loss functions
    DEFAULT_HARD_NEGATIVE_RATIO,
    DEFAULT_HISTOGRAM_INTERVAL,
    DEFAULT_LATENT_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_MARGIN_BASE,
    DEFAULT_MARGIN_SCALE,
    DEFAULT_MAX_RADIUS,
    DEFAULT_METRIC_LOSS_SCALE,
    DEFAULT_METRIC_N_PAIRS,
    DEFAULT_N_TRIPLETS,
    DEFAULT_PATIENCE,
    DEFAULT_PLATEAU_MIN_DELTA,
    DEFAULT_PLATEAU_PATIENCE,
    DEFAULT_PRIOR_SIGMA,
    DEFAULT_RADIAL_WEIGHT,
    DEFAULT_RADIUS_POWER,
    DEFAULT_RANKING_MARGIN,
    DEFAULT_REPULSION_SIGMA,
    DEFAULT_TENSORBOARD_DIR,
    DEFAULT_WEIGHT_DECAY,
    # Numerical stability
    EPSILON,
    EPSILON_LOG,
    EPSILON_NORM,
    EPSILON_TEMP,
    # Gradient balance
    GRAD_EMA_MOMENTUM,
    GRAD_SCALE_MAX,
    GRAD_SCALE_MIN,
    # Homeostatic control
    HOMEOSTATIC_CURVATURE_RATE,
    HOMEOSTATIC_EMA_MOMENTUM,
    HOMEOSTATIC_SIGMA_RATE,
    HOMEOSTATIC_TARGET_RADIUS,
    HYPERBOLIC_CURVATURE,
    HYPERBOLIC_MAX_NORM,
    # Ternary space
    MAX_VALUATION,
    N_TERNARY_DIGITS,
    N_TERNARY_OPERATIONS,
    RADIUS_MAX,
    RADIUS_MIN,
    TERNARY_BASE,
)

# Environment
from .environment import EnvConfig, Environment, get_env_config, reset_env_config

# Loader
from .loader import load_config, save_config

# Paths
from .paths import (
    CACHE_DIR,
    CHECKPOINTS_DIR,
    CONFIG_DIR,
    DATA_DIR,
    DELIVERABLES_DIR,
    DOCS_DIR,
    EXTERNAL_DATA_DIR,
    LOGS_DIR,
    OUTPUT_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
    RESULTS_DIR,
    RUNS_DIR,
    SCRIPTS_DIR,
    SRC_DIR,
    TESTS_DIR,
    VIZ_DIR,
    ensure_dirs,
    get_checkpoint_path,
    get_data_path,
    get_results_path,
    init_project_dirs,
    resolve_legacy_path,
)

# Schema
from .schema import (
    ConfigValidationError,
    GeometryConfig,
    LossWeights,
    OptimizerConfig,
    RankingConfig,
    TrainingConfig,
    VAEConfig,
)

__all__ = [
    # Loader functions
    "load_config",
    "save_config",
    # Schema classes
    "TrainingConfig",
    "GeometryConfig",
    "LossWeights",
    "OptimizerConfig",
    "RankingConfig",
    "VAEConfig",
    "ConfigValidationError",
    # Environment
    "Environment",
    "EnvConfig",
    "get_env_config",
    "reset_env_config",
    # Key constants (most commonly used)
    "EPSILON",
    "DEFAULT_CURVATURE",
    "DEFAULT_MAX_RADIUS",
    "DEFAULT_LATENT_DIM",
    "N_TERNARY_OPERATIONS",
    "HYPERBOLIC_CURVATURE",
    "HYPERBOLIC_MAX_NORM",
    # Path configuration
    "PROJECT_ROOT",
    "CONFIG_DIR",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "EXTERNAL_DATA_DIR",
    "CACHE_DIR",
    "OUTPUT_DIR",
    "RESULTS_DIR",
    "CHECKPOINTS_DIR",
    "RUNS_DIR",
    "REPORTS_DIR",
    "VIZ_DIR",
    "LOGS_DIR",
    "DOCS_DIR",
    "SRC_DIR",
    "TESTS_DIR",
    "SCRIPTS_DIR",
    "DELIVERABLES_DIR",
    # Path helper functions
    "ensure_dirs",
    "get_checkpoint_path",
    "get_results_path",
    "get_data_path",
    "resolve_legacy_path",
    "init_project_dirs",
]
