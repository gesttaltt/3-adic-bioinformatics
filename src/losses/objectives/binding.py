# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Binding affinity objectives for multi-objective optimization.

These objectives evaluate how well a designed sequence is predicted
to bind to target molecules (receptors, antibodies, etc.).
"""

from __future__ import annotations

from typing import Any

import torch

from .base import Objective, ObjectiveResult


class BindingObjective(Objective):
    """Objective for maximizing binding affinity to target.

    Uses a learned or heuristic model to predict binding strength.
    Returns negative binding score (lower = stronger binding = better).

    The binding prediction uses features from the latent space to
    estimate interaction strength with a target receptor profile.
    """

    def __init__(
        self,
        target_profile: torch.Tensor | None = None,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        weight: float = 1.0,
    ):
        """Initialize binding objective.

        Args:
            target_profile: Target receptor/antibody embedding, shape (Dim,)
            hidden_dim: Hidden dimension for binding prediction
            temperature: Softmax temperature for binding probability
            weight: Weight for multi-objective combination
        """
        super().__init__(name="binding", weight=weight)
        self.target_profile = target_profile
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Simple binding prediction network (can be replaced with learned model)
        self.binding_predictor: torch.nn.Module | None = None

    def _compute_binding_score(
        self,
        latent: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute binding affinity score.

        Uses cosine similarity as a proxy for binding affinity.
        More sophisticated models can be plugged in.

        Args:
            latent: Candidate embeddings, shape (Batch, Dim)
            target: Target profile, shape (Dim,) or (Batch, Dim)

        Returns:
            Binding scores, shape (Batch,)
            Negative values = stronger binding (lower is better)
        """
        # Normalize vectors
        latent_norm = torch.nn.functional.normalize(latent, dim=-1)

        if target.dim() == 1:
            target = target.unsqueeze(0)
        target_norm = torch.nn.functional.normalize(target, dim=-1)

        # Cosine similarity as binding proxy
        # Higher similarity = stronger binding
        similarity = (latent_norm * target_norm).sum(dim=-1)

        # Convert to score where lower is better
        # We want to maximize binding, so return negative similarity
        binding_score = -similarity / self.temperature

        return binding_score

    def evaluate(
        self,
        latent: torch.Tensor,
        decoded: torch.Tensor | None = None,
        target_profile: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ObjectiveResult:
        """Evaluate binding affinity objective.

        Args:
            latent: Latent space vectors, shape (Batch, Dim)
            decoded: Decoded sequences (unused for latent-based binding)
            target_profile: Override target profile for this evaluation
            **kwargs: Additional arguments

        Returns:
            ObjectiveResult with binding scores (lower = better binding)
        """
        # Use provided target or default
        target = target_profile if target_profile is not None else self.target_profile

        if target is None:
            # Default: use zero vector (generic binding)
            target = torch.zeros(latent.shape[-1], device=latent.device)

        target = target.to(latent.device)
        scores = self._compute_binding_score(latent, target)

        return ObjectiveResult(
            score=scores,
            name=self.name,
            metadata={
                "target_dim": target.shape[-1],
                "mean_similarity": -scores.mean().item(),
            },
        )


class EpitopeBindingObjective(Objective):
    """Objective for binding to specific epitope regions.

    Evaluates how well the designed sequence can present epitopes
    that will bind to target immune receptors (MHC, TCR, BCR).
    """

    def __init__(
        self,
        epitope_embeddings: torch.Tensor | None = None,
        num_epitopes: int = 10,
        min_binding_threshold: float = 0.5,
        weight: float = 1.0,
    ):
        """Initialize epitope binding objective.

        Args:
            epitope_embeddings: Known epitope embeddings, shape (NumEpitopes, Dim)
            num_epitopes: Number of epitopes to consider
            min_binding_threshold: Minimum binding similarity threshold
            weight: Weight for multi-objective combination
        """
        super().__init__(name="epitope_binding", weight=weight)
        self.epitope_embeddings = epitope_embeddings
        self.num_epitopes = num_epitopes
        self.min_binding_threshold = min_binding_threshold

    def evaluate(
        self,
        latent: torch.Tensor,
        decoded: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ObjectiveResult:
        """Evaluate epitope binding coverage.

        Args:
            latent: Latent space vectors, shape (Batch, Dim)
            decoded: Decoded sequences
            **kwargs: Additional arguments

        Returns:
            ObjectiveResult with epitope coverage scores
        """
        if self.epitope_embeddings is None:
            # Generate random epitope targets for demonstration
            epitopes = torch.randn(
                self.num_epitopes,
                latent.shape[-1],
                device=latent.device,
            )
        else:
            epitopes = self.epitope_embeddings.to(latent.device)

        # Normalize
        latent_norm = torch.nn.functional.normalize(latent, dim=-1)
        epitope_norm = torch.nn.functional.normalize(epitopes, dim=-1)

        # Compute similarity to all epitopes
        # (Batch, Dim) @ (Dim, NumEpitopes) -> (Batch, NumEpitopes)
        similarities = latent_norm @ epitope_norm.T

        # Count epitopes above threshold
        bound_epitopes = (similarities > self.min_binding_threshold).float().sum(dim=-1)

        # Score: maximize epitope coverage
        # Lower is better, so return negative coverage fraction
        coverage = bound_epitopes / self.num_epitopes
        scores = 1.0 - coverage  # Lower = better coverage

        return ObjectiveResult(
            score=scores,
            name=self.name,
            metadata={
                "mean_coverage": coverage.mean().item(),
                "num_epitopes": self.num_epitopes,
            },
        )
