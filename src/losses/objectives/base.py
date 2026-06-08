# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Base classes for multi-objective optimization.

Provides the abstract Objective interface and ObjectiveRegistry for
combining multiple objectives in Pareto optimization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class ObjectiveResult:
    """Result from evaluating an objective function.

    Attributes:
        score: The objective score (lower is better)
        name: Name of the objective
        metadata: Additional information about the evaluation
    """

    score: torch.Tensor
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"ObjectiveResult(name={self.name}, score={self.score.mean().item():.4f})"


class Objective(ABC):
    """Abstract base class for optimization objectives.

    All objectives should return scores where LOWER is BETTER,
    to maintain consistency with standard optimization conventions.
    """

    def __init__(self, name: str, weight: float = 1.0):
        """Initialize objective.

        Args:
            name: Human-readable name for this objective
            weight: Weight for combining with other objectives
        """
        self.name = name
        self.weight = weight

    @abstractmethod
    def evaluate(
        self,
        latent: torch.Tensor,
        decoded: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ObjectiveResult:
        """Evaluate the objective for given inputs.

        Args:
            latent: Latent space vectors, shape (Batch, Dim)
            decoded: Decoded sequences/structures, shape varies
            **kwargs: Additional inputs specific to the objective

        Returns:
            ObjectiveResult with scores (lower is better)
        """
        pass

    def __call__(
        self,
        latent: torch.Tensor,
        decoded: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ObjectiveResult:
        """Alias for evaluate."""
        return self.evaluate(latent, decoded, **kwargs)


class ObjectiveRegistry:
    """Registry for managing multiple objectives in multi-objective optimization.

    Provides methods for:
    - Registering objectives with weights
    - Evaluating all objectives on a batch
    - Computing weighted combination or Pareto scores
    """

    def __init__(self):
        """Initialize empty registry."""
        self._objectives: dict[str, Objective] = {}
        self._weights: dict[str, float] = {}

    def register(
        self,
        name: str,
        objective: Objective,
        weight: float = 1.0,
    ) -> ObjectiveRegistry:
        """Register an objective function.

        Args:
            name: Unique identifier for this objective
            objective: The objective instance
            weight: Weight for weighted combination (default 1.0)

        Returns:
            Self for method chaining
        """
        self._objectives[name] = objective
        self._weights[name] = weight
        return self

    def unregister(self, name: str) -> ObjectiveRegistry:
        """Remove an objective from the registry.

        Args:
            name: Identifier of objective to remove

        Returns:
            Self for method chaining
        """
        if name in self._objectives:
            del self._objectives[name]
            del self._weights[name]
        return self

    @property
    def objective_names(self) -> list[str]:
        """Get list of registered objective names."""
        return list(self._objectives.keys())

    @property
    def num_objectives(self) -> int:
        """Get number of registered objectives."""
        return len(self._objectives)

    def evaluate_all(
        self,
        latent: torch.Tensor,
        decoded: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, ObjectiveResult]:
        """Evaluate all registered objectives.

        Args:
            latent: Latent space vectors, shape (Batch, Dim)
            decoded: Decoded sequences/structures
            **kwargs: Additional inputs passed to all objectives

        Returns:
            Dictionary mapping objective names to results
        """
        results = {}
        for name, objective in self._objectives.items():
            results[name] = objective.evaluate(latent, decoded, **kwargs)
        return results

    def get_score_matrix(
        self,
        latent: torch.Tensor,
        decoded: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Get matrix of all objective scores for Pareto optimization.

        Args:
            latent: Latent space vectors, shape (Batch, Dim)
            decoded: Decoded sequences/structures
            **kwargs: Additional inputs

        Returns:
            Score matrix, shape (Batch, NumObjectives)
            Lower scores are better for all objectives
        """
        results = self.evaluate_all(latent, decoded, **kwargs)
        batch_size = latent.shape[0]

        scores = torch.zeros(batch_size, self.num_objectives, device=latent.device)

        for i, name in enumerate(self.objective_names):
            result = results[name]
            # Ensure scores are per-sample
            if result.score.dim() == 0:
                scores[:, i] = result.score.expand(batch_size)
            else:
                scores[:, i] = result.score

        return scores

    def weighted_sum(
        self,
        latent: torch.Tensor,
        decoded: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute weighted sum of all objectives.

        Args:
            latent: Latent space vectors, shape (Batch, Dim)
            decoded: Decoded sequences/structures
            **kwargs: Additional inputs

        Returns:
            Weighted sum scores, shape (Batch,)
        """
        results = self.evaluate_all(latent, decoded, **kwargs)

        total = torch.zeros(latent.shape[0], device=latent.device)
        for name, result in results.items():
            weight = self._weights[name]
            score = result.score
            if score.dim() == 0:
                score = score.expand(latent.shape[0])
            total += weight * score

        return total

    def create_from_config(
        self,
        config: dict[str, dict[str, Any]],
        objective_factory: Callable[[str, dict[str, Any]], Objective],
    ) -> ObjectiveRegistry:
        """Create registry from configuration dictionary.

        Args:
            config: Dict mapping names to objective configs
            objective_factory: Factory function to create objectives

        Returns:
            Self with objectives registered
        """
        for name, obj_config in config.items():
            weight = obj_config.pop("weight", 1.0)
            objective = objective_factory(name, obj_config)
            self.register(name, objective, weight)
        return self
