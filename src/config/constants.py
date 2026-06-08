# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Centralized constants - Single Source of Truth for magic numbers.

This module contains ALL numerical constants, thresholds, and default values
used throughout the codebase. Import from here instead of hardcoding values.

Usage:
    from src.config.constants import EPSILON, DEFAULT_CURVATURE

Categories:
    - Numerical stability (epsilons)
    - Geometry defaults
    - Training defaults
    - Gradient balance bounds
    - Ternary space constants
"""

# =============================================================================
# NUMERICAL STABILITY
# =============================================================================

# Standard epsilon for division and normalization
EPSILON = 1e-8

# Epsilon for logarithm operations (prevents log(0))
EPSILON_LOG = 1e-10

# Epsilon for norm computations
EPSILON_NORM = 1e-6

# Epsilon for softmax temperature scaling
EPSILON_TEMP = 1e-4


# =============================================================================
# GEOMETRY DEFAULTS (Poincaré Ball / Hyperbolic)
# =============================================================================

# Default hyperbolic curvature (c > 0 for hyperbolic space)
DEFAULT_CURVATURE = 1.0

# Maximum radius in Poincaré ball (must be < 1/sqrt(c))
DEFAULT_MAX_RADIUS = 0.95

# Default latent space dimension
DEFAULT_LATENT_DIM = 16

# Curvature bounds for learnable curvature
CURVATURE_MIN = 0.1
CURVATURE_MAX = 10.0

# Radius bounds for projections
RADIUS_MIN = 0.0
RADIUS_MAX = 0.99


# =============================================================================
# TRAINING DEFAULTS
# =============================================================================

# Batch size
DEFAULT_BATCH_SIZE = 256

# Learning rate
DEFAULT_LEARNING_RATE = 1e-3

# Gradient clipping threshold
DEFAULT_GRAD_CLIP = 1.0

# Weight decay for AdamW
DEFAULT_WEIGHT_DECAY = 1e-4

# Default number of epochs
DEFAULT_EPOCHS = 300

# Early stopping patience
DEFAULT_PATIENCE = 150

# Free bits for KL divergence
DEFAULT_FREE_BITS = 0.3


# =============================================================================
# GRADIENT BALANCE BOUNDS
# =============================================================================

# Minimum gradient scale factor
GRAD_SCALE_MIN = 0.5

# Maximum gradient scale factor
GRAD_SCALE_MAX = 2.0

# EMA momentum for gradient tracking
GRAD_EMA_MOMENTUM = 0.99


# =============================================================================
# TERNARY SPACE CONSTANTS
# =============================================================================

# Number of ternary digits (3^9 = 19,683 operations)
N_TERNARY_DIGITS = 9

# Base for ternary operations
TERNARY_BASE = 3

# Total number of unique ternary operations
N_TERNARY_OPERATIONS = TERNARY_BASE**N_TERNARY_DIGITS  # 19,683

# Maximum 3-adic valuation possible
MAX_VALUATION = N_TERNARY_DIGITS  # 9


# =============================================================================
# LOSS FUNCTION DEFAULTS
# =============================================================================

# Repulsion loss bandwidth
DEFAULT_REPULSION_SIGMA = 0.5

# Ranking loss margin (base margin for triplet losses)
DEFAULT_RANKING_MARGIN = 0.1

# Base margin for hierarchical ranking (v5.8+)
DEFAULT_MARGIN_BASE = 0.05

# Margin scale for hierarchical ranking (v5.8+)
DEFAULT_MARGIN_SCALE = 0.15

# Number of triplets for ranking loss
DEFAULT_N_TRIPLETS = 500

# Hard negative mining ratio
DEFAULT_HARD_NEGATIVE_RATIO = 0.5

# Number of pairs for metric loss
DEFAULT_METRIC_N_PAIRS = 1000

# Metric loss scale factor
DEFAULT_METRIC_LOSS_SCALE = 1.0


# =============================================================================
# HYPERBOLIC GEOMETRY DEFAULTS (v5.9+)
# =============================================================================

# Default curvature for hyperbolic losses (Note: 2.0 for most hyperbolic modules)
HYPERBOLIC_CURVATURE = 2.0

# Default max norm for Poincare ball projection
HYPERBOLIC_MAX_NORM = 0.95

# Radial weight for hyperbolic ranking loss
DEFAULT_RADIAL_WEIGHT = 0.1

# Prior sigma for hyperbolic prior
DEFAULT_PRIOR_SIGMA = 1.0

# Geodesic weight for hyperbolic reconstruction
DEFAULT_GEODESIC_WEIGHT = 0.3

# Radius power for radius weighting
DEFAULT_RADIUS_POWER = 2.0


# =============================================================================
# HOMEOSTATIC CONTROL DEFAULTS (v5.10+)
# =============================================================================

# EMA momentum for homeostatic adaptation
HOMEOSTATIC_EMA_MOMENTUM = 0.99

# Target radius for homeostatic control
HOMEOSTATIC_TARGET_RADIUS = 0.7

# Sigma adaptation rate
HOMEOSTATIC_SIGMA_RATE = 0.01

# Curvature adaptation rate
HOMEOSTATIC_CURVATURE_RATE = 0.001

# -----------------------------------------------------------------------------
# V5.11.8 Hierarchical Homeostasis Controller Thresholds
# -----------------------------------------------------------------------------

# Coverage thresholds for encoder_A freeze/unfreeze decisions
HOMEOSTATIC_COVERAGE_FREEZE_THRESHOLD = 0.995  # Freeze when drops below
HOMEOSTATIC_COVERAGE_UNFREEZE_THRESHOLD = 1.0  # Unfreeze when reaches
HOMEOSTATIC_COVERAGE_FLOOR = 0.95  # Minimum coverage threshold (floor for annealing)

# Hierarchy thresholds for encoder_B (VAE-B hierarchy-gated)
HOMEOSTATIC_HIERARCHY_PLATEAU_THRESHOLD = 0.001  # Freeze when change < this
HOMEOSTATIC_HIERARCHY_PLATEAU_PATIENCE = 5  # Epochs of plateau before freeze
HOMEOSTATIC_HIERARCHY_PATIENCE_CEILING = 15  # Max patience for hierarchy (annealing ceiling)

# Controller gradient thresholds (gradient-gated freeze)
HOMEOSTATIC_CONTROLLER_GRAD_THRESHOLD = 0.01  # Freeze when grad norm < this
HOMEOSTATIC_CONTROLLER_GRAD_PATIENCE = 3  # Epochs of low grad before freeze
HOMEOSTATIC_CONTROLLER_PATIENCE_CEILING = 10  # Max patience for controller (annealing ceiling)

# General homeostasis settings
HOMEOSTATIC_WINDOW_SIZE = 5  # Moving average window for metric tracking
HOMEOSTATIC_HYSTERESIS_EPOCHS = 3  # Minimum epochs between state changes
HOMEOSTATIC_WARMUP_EPOCHS = 5  # Epochs before homeostasis activates

# Q-gated annealing settings (V5.11.8)
HOMEOSTATIC_ANNEALING_STEP = 0.005  # How much to relax thresholds per cycle


# =============================================================================
# COVERAGE TRACKING DEFAULTS
# =============================================================================

# Plateau detection patience (epochs)
DEFAULT_PLATEAU_PATIENCE = 100

# Minimum delta for plateau detection (fraction of N_TERNARY_OPERATIONS)
DEFAULT_PLATEAU_MIN_DELTA = 0.0005

# Coverage target percentage (100% = all 19,683 operations)
COVERAGE_TARGET_PCT = 100.0


# =============================================================================
# CONTINUOUS FEEDBACK DEFAULTS
# =============================================================================

# Default base ranking weight
DEFAULT_RANKING_WEIGHT = 0.5

# Coverage threshold for feedback modulation (percentage)
DEFAULT_COVERAGE_THRESHOLD = 90.0

# Sensitivity to coverage gap
DEFAULT_COVERAGE_SENSITIVITY = 0.1

# Sensitivity to coverage trend (change rate)
DEFAULT_COVERAGE_TREND_SENSITIVITY = 2.0

# Minimum ranking weight
DEFAULT_MIN_RANKING_WEIGHT = 0.0

# Maximum ranking weight
DEFAULT_MAX_RANKING_WEIGHT = 1.0

# EMA alpha for coverage smoothing
DEFAULT_EMA_ALPHA = 0.9


# =============================================================================
# OBSERVABILITY DEFAULTS
# =============================================================================

# TensorBoard log interval (batches)
DEFAULT_LOG_INTERVAL = 10

# Checkpoint save frequency (epochs)
DEFAULT_CHECKPOINT_FREQ = 10

# Evaluation sample count
DEFAULT_EVAL_SAMPLES = 1000

# Histogram logging interval (epochs)
DEFAULT_HISTOGRAM_INTERVAL = 10

# Embedding visualization interval (epochs)
DEFAULT_EMBEDDING_INTERVAL = 50


# =============================================================================
# PATH DEFAULTS (can be overridden by environment variables)
# =============================================================================

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = "checkpoints"

# Default log directory
DEFAULT_LOG_DIR = "logs"

# Default TensorBoard runs directory
DEFAULT_TENSORBOARD_DIR = "runs"


__all__ = [
    # Numerical stability
    "EPSILON",
    "EPSILON_LOG",
    "EPSILON_NORM",
    "EPSILON_TEMP",
    # Geometry
    "DEFAULT_CURVATURE",
    "DEFAULT_MAX_RADIUS",
    "DEFAULT_LATENT_DIM",
    "CURVATURE_MIN",
    "CURVATURE_MAX",
    "RADIUS_MIN",
    "RADIUS_MAX",
    # Training
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_GRAD_CLIP",
    "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_EPOCHS",
    "DEFAULT_PATIENCE",
    "DEFAULT_FREE_BITS",
    # Gradient balance
    "GRAD_SCALE_MIN",
    "GRAD_SCALE_MAX",
    "GRAD_EMA_MOMENTUM",
    # Ternary space
    "N_TERNARY_DIGITS",
    "TERNARY_BASE",
    "N_TERNARY_OPERATIONS",
    "MAX_VALUATION",
    # Loss functions
    "DEFAULT_REPULSION_SIGMA",
    "DEFAULT_RANKING_MARGIN",
    "DEFAULT_MARGIN_BASE",
    "DEFAULT_MARGIN_SCALE",
    "DEFAULT_N_TRIPLETS",
    "DEFAULT_HARD_NEGATIVE_RATIO",
    "DEFAULT_METRIC_N_PAIRS",
    "DEFAULT_METRIC_LOSS_SCALE",
    # Hyperbolic geometry
    "HYPERBOLIC_CURVATURE",
    "HYPERBOLIC_MAX_NORM",
    "DEFAULT_RADIAL_WEIGHT",
    "DEFAULT_PRIOR_SIGMA",
    "DEFAULT_GEODESIC_WEIGHT",
    "DEFAULT_RADIUS_POWER",
    # Homeostatic control (basic)
    "HOMEOSTATIC_EMA_MOMENTUM",
    "HOMEOSTATIC_TARGET_RADIUS",
    "HOMEOSTATIC_SIGMA_RATE",
    "HOMEOSTATIC_CURVATURE_RATE",
    # Homeostatic control (V5.11.8 hierarchical)
    "HOMEOSTATIC_COVERAGE_FREEZE_THRESHOLD",
    "HOMEOSTATIC_COVERAGE_UNFREEZE_THRESHOLD",
    "HOMEOSTATIC_COVERAGE_FLOOR",
    "HOMEOSTATIC_HIERARCHY_PLATEAU_THRESHOLD",
    "HOMEOSTATIC_HIERARCHY_PLATEAU_PATIENCE",
    "HOMEOSTATIC_HIERARCHY_PATIENCE_CEILING",
    "HOMEOSTATIC_CONTROLLER_GRAD_THRESHOLD",
    "HOMEOSTATIC_CONTROLLER_GRAD_PATIENCE",
    "HOMEOSTATIC_CONTROLLER_PATIENCE_CEILING",
    "HOMEOSTATIC_WINDOW_SIZE",
    "HOMEOSTATIC_HYSTERESIS_EPOCHS",
    "HOMEOSTATIC_WARMUP_EPOCHS",
    "HOMEOSTATIC_ANNEALING_STEP",
    # Coverage tracking
    "DEFAULT_PLATEAU_PATIENCE",
    "DEFAULT_PLATEAU_MIN_DELTA",
    "COVERAGE_TARGET_PCT",
    # Continuous feedback
    "DEFAULT_RANKING_WEIGHT",
    "DEFAULT_COVERAGE_THRESHOLD",
    "DEFAULT_COVERAGE_SENSITIVITY",
    "DEFAULT_COVERAGE_TREND_SENSITIVITY",
    "DEFAULT_MIN_RANKING_WEIGHT",
    "DEFAULT_MAX_RANKING_WEIGHT",
    "DEFAULT_EMA_ALPHA",
    # Observability
    "DEFAULT_LOG_INTERVAL",
    "DEFAULT_CHECKPOINT_FREQ",
    "DEFAULT_EVAL_SAMPLES",
    "DEFAULT_HISTOGRAM_INTERVAL",
    "DEFAULT_EMBEDDING_INTERVAL",
    # Paths
    "DEFAULT_CHECKPOINT_DIR",
    "DEFAULT_LOG_DIR",
    "DEFAULT_TENSORBOARD_DIR",
]
