# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Type definitions for the project.

This module provides centralized type aliases, protocols, and type
definitions for consistent typing across all modules.

Usage:
    from src.core.types import (
        Tensor,
        Array,
        PAdicIndex,
        Manifold,
    )

Note:
    For Python 3.9+ compatibility, use generic types directly
    (list[int] instead of List[int]).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import (
    Any,
    Generic,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
import torch

# ============================================================================
# Basic Type Aliases
# ============================================================================

# Tensor types
Tensor = torch.Tensor
Array = np.ndarray
TensorOrArray = Union[Tensor, Array]

# Numeric types
Number = Union[int, float]
NumberSequence = Union[list[Number], tuple[Number, ...], Array, Tensor]

# Shape types
Shape = tuple[int, ...]
DType = Union[torch.dtype, np.dtype]

# Device type
Device = Union[str, torch.device]


# ============================================================================
# P-adic Types
# ============================================================================

# P-adic index (integer that can be decomposed in base p)
PAdicIndex = int

# P-adic digits (list of remainders mod p)
PAdicDigits = list[int]

# Valuation type (int for finite, float('inf') for infinite)
ValuationType = Union[int, float]

# P-adic expansion (valuation and digits)
PAdicExpansion = tuple[ValuationType, PAdicDigits]


# ============================================================================
# Geometry Types
# ============================================================================

# Curvature parameter (positive for hyperbolic)
Curvature = float

# Radius in Poincare ball
Radius = float

# Point in a manifold
Point = Tensor

# Tangent vector
TangentVector = Tensor

# Metric tensor
MetricTensor = Tensor


# ============================================================================
# Loss and Training Types
# ============================================================================

# Loss value (single scalar)
LossValue = Union[float, Tensor]

# Loss dictionary (named losses)
LossDict = dict[str, LossValue]

# Metrics dictionary
MetricsDict = dict[str, float]

# Learning rate schedule function
LRSchedule = Callable[[int], float]


# ============================================================================
# Model Types
# ============================================================================

# Type variable for model classes
M = TypeVar("M", bound=torch.nn.Module)

# Optimizer type
Optimizer = torch.optim.Optimizer

# Scheduler type
Scheduler = torch.optim.lr_scheduler._LRScheduler

# Activation function
ActivationFn = Callable[[Tensor], Tensor]


# ============================================================================
# Data Types
# ============================================================================

# Batch of data (input, target)
Batch = tuple[Tensor, Tensor]

# Named batch with metadata
NamedBatch = dict[str, Tensor]

# Data loader iterator
DataIterator = Iterator[Batch]


# ============================================================================
# Protocols
# ============================================================================


@runtime_checkable
class Manifold(Protocol):
    """Protocol for geometric manifolds.

    Defines the interface that all manifold implementations must satisfy.
    Used for type checking hyperbolic and other geometric operations.
    """

    def expmap(self, x: Tensor, v: Tensor) -> Tensor:
        """Exponential map from tangent space."""
        ...

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        """Logarithmic map to tangent space."""
        ...

    def dist(self, x: Tensor, y: Tensor) -> Tensor:
        """Geodesic distance."""
        ...

    def projx(self, x: Tensor) -> Tensor:
        """Project point to manifold."""
        ...


@runtime_checkable
class Encoder(Protocol):
    """Protocol for encoder networks."""

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent representation."""
        ...

    @property
    def output_dim(self) -> int:
        """Dimension of encoded representation."""
        ...


@runtime_checkable
class Decoder(Protocol):
    """Protocol for decoder networks."""

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation to output."""
        ...

    @property
    def input_dim(self) -> int:
        """Dimension of latent input."""
        ...


@runtime_checkable
class VAELike(Protocol):
    """Protocol for VAE-like models."""

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode to mean and log variance."""
        ...

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent to reconstruction."""
        ...

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Sample using reparameterization trick."""
        ...


@runtime_checkable
class LossFunction(Protocol):
    """Protocol for loss functions."""

    def __call__(self, *args: Any, **kwargs: Any) -> LossDict:
        """Compute loss and return named components."""
        ...


@runtime_checkable
class Sampler(Protocol):
    """Protocol for data samplers."""

    def sample(self, n: int) -> Tensor:
        """Sample n items."""
        ...

    def __len__(self) -> int:
        """Return number of items available."""
        ...


@runtime_checkable
class TaskSampler(Protocol):
    """Protocol for meta-learning task samplers."""

    def sample_task(self) -> Task:
        """Sample a single task."""
        ...

    def sample_batch(self, n_tasks: int) -> list[Task]:
        """Sample a batch of tasks."""
        ...


# Forward reference for Task (defined in meta module)
class Task(Protocol):
    """Protocol for meta-learning tasks."""

    support_x: Tensor
    support_y: Tensor
    query_x: Tensor
    query_y: Tensor


# ============================================================================
# Generic Types
# ============================================================================

# Type variable for generic containers
T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class Result(Generic[T]):
    """Result type for operations that can fail.

    Usage:
        def compute() -> Result[float]:
            try:
                value = expensive_computation()
                return Result.ok(value)
            except Exception as e:
                return Result.error(str(e))
    """

    def __init__(self, value: T | None, error: str | None = None):
        self._value = value
        self._error = error

    @classmethod
    def ok(cls, value: T) -> Result[T]:
        """Create successful result."""
        return cls(value, None)

    @classmethod
    def error(cls, message: str) -> Result[T]:
        """Create error result."""
        return cls(None, message)

    @property
    def is_ok(self) -> bool:
        """Check if result is successful."""
        return self._error is None

    @property
    def is_error(self) -> bool:
        """Check if result is an error."""
        return self._error is not None

    def unwrap(self) -> T:
        """Get value, raising if error."""
        if self._error is not None:
            raise ValueError(f"Cannot unwrap error result: {self._error}")
        return self._value  # type: ignore

    def unwrap_or(self, default: T) -> T:
        """Get value or default if error."""
        if self._error is not None:
            return default
        return self._value  # type: ignore

    @property
    def error_message(self) -> str | None:
        """Get error message if any."""
        return self._error


# ============================================================================
# Type Guards
# ============================================================================


def is_tensor(obj: Any) -> bool:
    """Check if object is a PyTorch tensor."""
    return isinstance(obj, torch.Tensor)


def is_array(obj: Any) -> bool:
    """Check if object is a NumPy array."""
    return isinstance(obj, np.ndarray)


def is_numeric(obj: Any) -> bool:
    """Check if object is numeric (int, float, tensor, or array)."""
    return isinstance(obj, (int, float, torch.Tensor, np.ndarray))


def ensure_tensor(
    obj: TensorOrArray,
    device: Device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Convert to PyTorch tensor if needed."""
    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)
    if device is not None:
        obj = obj.to(device)
    if dtype is not None:
        obj = obj.to(dtype)
    return obj


def ensure_array(obj: TensorOrArray) -> Array:
    """Convert to NumPy array if needed."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    return obj


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Basic types
    "Tensor",
    "Array",
    "TensorOrArray",
    "Number",
    "NumberSequence",
    "Shape",
    "DType",
    "Device",
    # P-adic types
    "PAdicIndex",
    "PAdicDigits",
    "ValuationType",
    "PAdicExpansion",
    # Geometry types
    "Curvature",
    "Radius",
    "Point",
    "TangentVector",
    "MetricTensor",
    # Loss/Training types
    "LossValue",
    "LossDict",
    "MetricsDict",
    "LRSchedule",
    # Model types
    "M",
    "Optimizer",
    "Scheduler",
    "ActivationFn",
    # Data types
    "Batch",
    "NamedBatch",
    "DataIterator",
    # Protocols
    "Manifold",
    "Encoder",
    "Decoder",
    "VAELike",
    "LossFunction",
    "Sampler",
    "TaskSampler",
    "Task",
    # Generic types
    "T",
    "K",
    "V",
    "Result",
    # Type guards
    "is_tensor",
    "is_array",
    "is_numeric",
    "ensure_tensor",
    "ensure_array",
]
