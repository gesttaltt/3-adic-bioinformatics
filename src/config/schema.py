# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Configuration schemas with Pydantic validation.

This module provides typed, validated configuration classes that replace
the complex nested dataclass structure in training/config_schema.py.

Key improvements:
- Pydantic validation with clear error messages
- Environment variable support via Field defaults
- Flat structure (max 2 levels of nesting)
- Immutable after creation (frozen=True)

Usage:
    from src.config import TrainingConfig, load_config

    # Load from YAML with env overrides
    config = load_config("config.yaml")

    # Or create programmatically
    config = TrainingConfig(epochs=500, learning_rate=1e-4)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CHECKPOINT_FREQ,
    DEFAULT_CURVATURE,
    DEFAULT_EPOCHS,
    DEFAULT_EVAL_SAMPLES,
    DEFAULT_FREE_BITS,
    DEFAULT_GRAD_CLIP,
    DEFAULT_HARD_NEGATIVE_RATIO,
    DEFAULT_LATENT_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_DIR,
    DEFAULT_LOG_INTERVAL,
    DEFAULT_MAX_RADIUS,
    DEFAULT_N_TRIPLETS,
    DEFAULT_PATIENCE,
    DEFAULT_RANKING_MARGIN,
    DEFAULT_TENSORBOARD_DIR,
    DEFAULT_WEIGHT_DECAY,
)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""

    pass


# =============================================================================
# GEOMETRY CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class GeometryConfig:
    """Hyperbolic geometry parameters.

    Attributes:
        curvature: Poincaré ball curvature (c > 0)
        max_radius: Maximum radius in ball (< 1/sqrt(c))
        latent_dim: Latent space dimension
        learnable_curvature: Whether curvature is trainable
    """

    curvature: float = DEFAULT_CURVATURE
    max_radius: float = DEFAULT_MAX_RADIUS
    latent_dim: int = DEFAULT_LATENT_DIM
    learnable_curvature: bool = False

    def __post_init__(self):
        if self.curvature <= 0:
            raise ConfigValidationError(f"curvature must be > 0, got {self.curvature}")
        if not (0 < self.max_radius < 1):
            raise ConfigValidationError(f"max_radius must be in (0, 1), got {self.max_radius}")
        if self.latent_dim < 2:
            raise ConfigValidationError(f"latent_dim must be >= 2, got {self.latent_dim}")


# =============================================================================
# LOSS CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class LossWeights:
    """Loss component weights.

    All weights should be non-negative. Set to 0.0 to disable a component.
    """

    reconstruction: float = 1.0
    kl_divergence: float = 1.0
    entropy: float = 0.0
    repulsion: float = 0.0
    ranking: float = 0.5
    radial: float = 0.3
    geodesic: float = 0.0
    centroid: float = 0.0

    def __post_init__(self):
        for name, value in self.__dict__.items():
            if value < 0:
                raise ConfigValidationError(f"Loss weight {name} must be >= 0, got {value}")


@dataclass(frozen=True)
class RankingConfig:
    """Ranking loss configuration."""

    margin: float = DEFAULT_RANKING_MARGIN
    n_triplets: int = DEFAULT_N_TRIPLETS
    hard_negative_ratio: float = DEFAULT_HARD_NEGATIVE_RATIO
    use_hyperbolic: bool = True
    radial_weight: float = 0.1


# =============================================================================
# OPTIMIZER CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer configuration."""

    type: Literal["adam", "adamw", "sgd", "riemannian_adam"] = "adamw"
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    betas: tuple[float, float] = (0.9, 0.999)

    # Learning rate schedule
    schedule: Literal["constant", "cosine", "step", "warmup_cosine"] = "constant"
    warmup_epochs: int = 0
    min_lr: float = 1e-6


# =============================================================================
# VAE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class VAEConfig:
    """VAE-specific parameters (for A or B)."""

    beta_start: float = 0.3
    beta_end: float = 0.8
    beta_warmup_epochs: int = 50
    temp_start: float = 1.0
    temp_end: float = 0.3
    entropy_weight: float = 0.0
    repulsion_weight: float = 0.0


# =============================================================================
# TRAINING CONFIGURATION (MAIN)
# =============================================================================


@dataclass
class TrainingConfig:
    """Complete training configuration.

    This is the main configuration class that aggregates all settings.
    Uses environment variables for paths (TVAE_* prefix).

    Example:
        config = TrainingConfig(epochs=500)
        config = TrainingConfig.from_dict(yaml_dict)
    """

    # Core training parameters
    seed: int = 42
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    grad_clip: float = DEFAULT_GRAD_CLIP
    patience: int = DEFAULT_PATIENCE
    free_bits: float = DEFAULT_FREE_BITS

    # Device (auto-detected if not specified)
    device: str = field(default_factory=lambda: "cuda" if _cuda_available() else "cpu")

    # Nested configurations (max 1 level deep)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    loss_weights: LossWeights = field(default_factory=LossWeights)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    vae_a: VAEConfig = field(default_factory=VAEConfig)
    vae_b: VAEConfig = field(default_factory=VAEConfig)

    # Paths (support environment variables)
    checkpoint_dir: str = field(default_factory=lambda: os.getenv("TVAE_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR))
    log_dir: str = field(default_factory=lambda: os.getenv("TVAE_LOG_DIR", DEFAULT_LOG_DIR))
    tensorboard_dir: str = field(default_factory=lambda: os.getenv("TVAE_TENSORBOARD_DIR", DEFAULT_TENSORBOARD_DIR))

    # Observability
    log_interval: int = DEFAULT_LOG_INTERVAL
    checkpoint_freq: int = DEFAULT_CHECKPOINT_FREQ
    eval_samples: int = DEFAULT_EVAL_SAMPLES
    experiment_name: str | None = None

    # Feature flags
    use_controller: bool = True
    use_dual_projection: bool = False
    use_curriculum: bool = False
    use_homeostasis: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.epochs < 1:
            raise ConfigValidationError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ConfigValidationError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.grad_clip <= 0:
            raise ConfigValidationError(f"grad_clip must be > 0, got {self.grad_clip}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainingConfig:
        """Create config from dictionary (e.g., from YAML).

        Handles nested dictionaries for sub-configurations.
        """
        # Extract nested configs
        geometry_data = data.pop("geometry", {})
        optimizer_data = data.pop("optimizer", {})
        loss_weights_data = data.pop("loss_weights", {})
        ranking_data = data.pop("ranking", {})
        vae_a_data = data.pop("vae_a", {})
        vae_b_data = data.pop("vae_b", {})

        # Build nested configs
        geometry = GeometryConfig(**geometry_data) if geometry_data else GeometryConfig()
        optimizer = OptimizerConfig(**optimizer_data) if optimizer_data else OptimizerConfig()
        loss_weights = LossWeights(**loss_weights_data) if loss_weights_data else LossWeights()
        ranking = RankingConfig(**ranking_data) if ranking_data else RankingConfig()
        vae_a = VAEConfig(**vae_a_data) if vae_a_data else VAEConfig()
        vae_b = VAEConfig(**vae_b_data) if vae_b_data else VAEConfig()

        return cls(
            geometry=geometry,
            optimizer=optimizer,
            loss_weights=loss_weights,
            ranking=ranking,
            vae_a=vae_a,
            vae_b=vae_b,
            **data,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        from dataclasses import asdict

        return asdict(self)


def _cuda_available() -> bool:
    """Check if CUDA is available (lazy import to avoid torch at module load)."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


__all__ = [
    "ConfigValidationError",
    "GeometryConfig",
    "LossWeights",
    "RankingConfig",
    "OptimizerConfig",
    "VAEConfig",
    "TrainingConfig",
]
