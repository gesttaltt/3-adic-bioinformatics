# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Configuration base classes.

This module provides standardized configuration patterns with validation,
serialization, and documentation support. All configuration dataclasses
should inherit from these base classes.

Consolidated from:
- src/contrastive/padic_contrastive.py (ContrastiveConfig)
- src/physics/statistical_physics.py (various configs)
- Multiple scattered configuration patterns

Key Features:
- Automatic validation on creation
- JSON/dict serialization
- Documentation generation
- Type checking integration

Usage:
    from src.core.config_base import BaseConfig, PAdicConfig

    @dataclass
    class MyConfig(BaseConfig):
        learning_rate: float = 0.001
        hidden_dim: int = 256

        def validate(self) -> None:
            if self.learning_rate <= 0:
                raise ValueError("learning_rate must be positive")

Examples:
    config = MyConfig(learning_rate=0.01)
    config.validate()
    print(config.to_dict())
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from src.config.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CURVATURE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_RADIUS,
    DEFAULT_PATIENCE,
    MAX_VALUATION,
    TERNARY_BASE,
)

T = TypeVar("T", bound="BaseConfig")


# ============================================================================
# Base Configuration
# ============================================================================


@dataclass
class BaseConfig:
    """Base configuration class with common utilities.

    All configuration dataclasses should inherit from this class
    to get automatic validation, serialization, and documentation.

    Attributes:
        _frozen: If True, config cannot be modified after creation
    """

    _frozen: ClassVar[bool] = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate configuration values.

        Override in subclasses to add validation logic.
        Called automatically after __init__.

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of config
        """
        return asdict(self)

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create configuration from dictionary.

        Args:
            data: Dictionary with configuration values

        Returns:
            New configuration instance
        """
        # Filter to only known fields
        known_fields = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered_data)

    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string.

        Args:
            indent: Indentation level for pretty printing

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls: type[T], json_str: str) -> T:
        """Create configuration from JSON string.

        Args:
            json_str: JSON string with configuration values

        Returns:
            New configuration instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path to save configuration
        """
        path = Path(path)
        path.write_text(self.to_json())

    @classmethod
    def load(cls: type[T], path: str | Path) -> T:
        """Load configuration from JSON file.

        Args:
            path: Path to load configuration from

        Returns:
            New configuration instance
        """
        path = Path(path)
        return cls.from_json(path.read_text())

    def update(self: T, **kwargs: Any) -> T:
        """Create new config with updated values.

        Args:
            **kwargs: Values to update

        Returns:
            New configuration with updates applied
        """
        current = self.to_dict()
        current.update(kwargs)
        return self.__class__.from_dict(current)

    def diff(self, other: BaseConfig) -> dict[str, tuple]:
        """Compare with another config and return differences.

        Args:
            other: Configuration to compare with

        Returns:
            Dict mapping field names to (self_value, other_value) tuples
        """
        self_dict = self.to_dict()
        other_dict = other.to_dict()

        diffs = {}
        all_keys = set(self_dict.keys()) | set(other_dict.keys())

        for key in all_keys:
            self_val = self_dict.get(key)
            other_val = other_dict.get(key)
            if self_val != other_val:
                diffs[key] = (self_val, other_val)

        return diffs

    def describe(self) -> str:
        """Generate human-readable description of configuration.

        Returns:
            Formatted string with field descriptions
        """
        lines = [f"{self.__class__.__name__}:"]
        for f in fields(self):
            value = getattr(self, f.name)
            doc = f.metadata.get("doc", "")
            lines.append(f"  {f.name}: {value}" + (f" - {doc}" if doc else ""))
        return "\n".join(lines)


# ============================================================================
# P-adic Configuration
# ============================================================================


@dataclass
class PAdicConfig(BaseConfig):
    """Configuration for p-adic mathematical operations.

    Standard configuration for modules using p-adic number theory,
    including valuation computations, distance metrics, and hierarchical
    sampling.

    Attributes:
        prime: Prime base for p-adic operations (typically 3 for ternary)
        max_valuation: Maximum valuation to consider (higher = deeper hierarchy)
        use_padic_structure: Whether to enable p-adic-aware processing
    """

    prime: int = TERNARY_BASE
    max_valuation: int = MAX_VALUATION
    use_padic_structure: bool = True

    def validate(self) -> None:
        """Validate p-adic configuration."""
        if self.prime < 2:
            raise ValueError(f"Prime must be >= 2, got {self.prime}")
        if self.max_valuation < 1:
            raise ValueError(f"Max valuation must be >= 1, got {self.max_valuation}")

        # Check primality for small primes
        if self.prime <= 100:
            for i in range(2, int(self.prime**0.5) + 1):
                if self.prime % i == 0:
                    raise ValueError(f"{self.prime} is not prime")


# ============================================================================
# Training Configuration
# ============================================================================


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for model training.

    Standard training hyperparameters with sensible defaults
    from project constants.

    Attributes:
        batch_size: Training batch size
        learning_rate: Initial learning rate
        epochs: Maximum training epochs
        patience: Early stopping patience
        weight_decay: L2 regularization weight
        grad_clip: Maximum gradient norm
        warmup_epochs: Learning rate warmup epochs
    """

    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    epochs: int = DEFAULT_EPOCHS
    patience: int = DEFAULT_PATIENCE
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_epochs: int = 5

    def validate(self) -> None:
        """Validate training configuration."""
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0, got {self.patience}")


# ============================================================================
# Geometry Configuration
# ============================================================================


@dataclass
class HyperbolicConfig(BaseConfig):
    """Configuration for hyperbolic geometry operations.

    Attributes:
        curvature: Hyperbolic curvature (c > 0)
        max_radius: Maximum radius in Poincare ball
        use_learnable_curvature: Whether curvature is trainable
        manifold_type: Type of hyperbolic manifold ('poincare' or 'lorentz')
    """

    curvature: float = DEFAULT_CURVATURE
    max_radius: float = DEFAULT_MAX_RADIUS
    use_learnable_curvature: bool = False
    manifold_type: str = "poincare"

    def validate(self) -> None:
        """Validate hyperbolic configuration."""
        if self.curvature <= 0:
            raise ValueError(f"curvature must be > 0, got {self.curvature}")
        if not 0 < self.max_radius < 1:
            raise ValueError(f"max_radius must be in (0, 1), got {self.max_radius}")
        if self.manifold_type not in ("poincare", "lorentz"):
            raise ValueError("manifold_type must be 'poincare' or 'lorentz'")


# ============================================================================
# Contrastive Learning Configuration
# ============================================================================


@dataclass
class ContrastiveConfig(BaseConfig):
    """Configuration for contrastive learning.

    Attributes:
        temperature: Softmax temperature for contrastive loss
        projection_dim: Dimension of projection head output
        hidden_dim: Dimension of projection head hidden layer
        use_padic_sampling: Use p-adic distance for positive sampling
        momentum: Momentum for MoCo-style updates
        queue_size: Size of negative sample queue
    """

    temperature: float = 0.07
    projection_dim: int = 128
    hidden_dim: int = 256
    use_padic_sampling: bool = True
    momentum: float = 0.999
    queue_size: int = 65536

    def validate(self) -> None:
        """Validate contrastive configuration."""
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.projection_dim < 1:
            raise ValueError("projection_dim must be >= 1")
        if not 0 <= self.momentum <= 1:
            raise ValueError(f"momentum must be in [0, 1], got {self.momentum}")


# ============================================================================
# Meta-Learning Configuration
# ============================================================================


@dataclass
class MetaLearningConfig(BaseConfig):
    """Configuration for meta-learning algorithms.

    Attributes:
        inner_lr: Learning rate for inner loop adaptation
        n_inner_steps: Number of inner loop gradient steps
        first_order: Use first-order approximation (FOMAML)
        n_support: Number of support samples per task
        n_query: Number of query samples per task
    """

    inner_lr: float = 0.01
    n_inner_steps: int = 5
    first_order: bool = False
    n_support: int = 5
    n_query: int = 15

    def validate(self) -> None:
        """Validate meta-learning configuration."""
        if self.inner_lr <= 0:
            raise ValueError(f"inner_lr must be > 0, got {self.inner_lr}")
        if self.n_inner_steps < 1:
            raise ValueError("n_inner_steps must be >= 1")
        if self.n_support < 1:
            raise ValueError("n_support must be >= 1")
        if self.n_query < 1:
            raise ValueError("n_query must be >= 1")


# ============================================================================
# Physics Simulation Configuration
# ============================================================================


@dataclass
class PhysicsConfig(BaseConfig):
    """Configuration for physics-inspired methods.

    Attributes:
        n_replicas: Number of replicas for parallel tempering
        temp_min: Minimum temperature
        temp_max: Maximum temperature
        n_sweeps: Monte Carlo sweeps between exchanges
        coupling_type: Type of spin-spin coupling
    """

    n_replicas: int = 8
    temp_min: float = 0.1
    temp_max: float = 10.0
    n_sweeps: int = 100
    coupling_type: str = "gaussian"

    def validate(self) -> None:
        """Validate physics configuration."""
        if self.n_replicas < 2:
            raise ValueError("n_replicas must be >= 2")
        if self.temp_min <= 0:
            raise ValueError("temp_min must be > 0")
        if self.temp_max <= self.temp_min:
            raise ValueError("temp_max must be > temp_min")
        if self.coupling_type not in ("gaussian", "uniform", "hopfield"):
            raise ValueError(f"Unknown coupling_type: {self.coupling_type}")


# ============================================================================
# Combined Configuration
# ============================================================================


@dataclass
class ExperimentConfig(BaseConfig):
    """Complete experiment configuration combining all components.

    Attributes:
        name: Experiment name for logging
        seed: Random seed for reproducibility
        training: Training configuration
        padic: P-adic configuration
        hyperbolic: Hyperbolic geometry configuration
    """

    name: str = "experiment"
    seed: int = 42
    training: TrainingConfig = field(default_factory=TrainingConfig)
    padic: PAdicConfig = field(default_factory=PAdicConfig)
    hyperbolic: HyperbolicConfig = field(default_factory=HyperbolicConfig)

    def validate(self) -> None:
        """Validate all sub-configurations."""
        self.training.validate()
        self.padic.validate()
        self.hyperbolic.validate()

    def to_dict(self) -> dict[str, Any]:
        """Convert to nested dictionary."""
        return {
            "name": self.name,
            "seed": self.seed,
            "training": self.training.to_dict(),
            "padic": self.padic.to_dict(),
            "hyperbolic": self.hyperbolic.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Create from nested dictionary."""
        return cls(
            name=data.get("name", "experiment"),
            seed=data.get("seed", 42),
            training=TrainingConfig.from_dict(data.get("training", {})),
            padic=PAdicConfig.from_dict(data.get("padic", {})),
            hyperbolic=HyperbolicConfig.from_dict(data.get("hyperbolic", {})),
        )


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "BaseConfig",
    "PAdicConfig",
    "TrainingConfig",
    "HyperbolicConfig",
    "ContrastiveConfig",
    "MetaLearningConfig",
    "PhysicsConfig",
    "ExperimentConfig",
]
