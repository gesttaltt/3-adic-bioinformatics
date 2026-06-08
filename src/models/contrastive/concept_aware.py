# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Concept-Aware Contrastive Learning.

Uses Formal Concept Analysis to inform contrastive learning:
- Samples in the same FCA concept are positive pairs
- Samples in different concepts are negative pairs
- Concept hierarchy informs similarity structure

This creates semantically meaningful representations where
samples with similar mutation-resistance patterns are close.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.analysis.set_theory.formal_concepts import (
    ConceptLattice,
)
from src.analysis.set_theory.lattice import ResistanceLattice
from src.analysis.set_theory.mutation_sets import MutationSet
from src.geometry import poincare_distance


@dataclass
class ConceptContrastiveConfig:
    """Configuration for concept-aware contrastive learning.

    Attributes:
        temperature: Contrastive temperature
        use_hard_negatives: Use concept hierarchy for hard negatives
        positive_weight: Weight for positive pairs
        negative_weight: Weight for negative pairs
        hierarchy_margin: Margin based on concept distance
    """

    temperature: float = 0.1
    use_hard_negatives: bool = True
    positive_weight: float = 1.0
    negative_weight: float = 1.0
    hierarchy_margin: float = 0.1


class ConceptAwareContrastive(nn.Module):
    """Contrastive learning using FCA concept structure.

    Learns representations where:
    1. Samples in the same concept cluster together
    2. Samples in sibling concepts are close but separated
    3. Samples in distant concepts are far apart

    Example:
        >>> concept_lattice = ConceptLattice(context)
        >>> contrastive = ConceptAwareContrastive(concept_lattice, config)
        >>> loss = contrastive(embeddings, sample_ids)
    """

    def __init__(
        self,
        concept_lattice: ConceptLattice,
        config: ConceptContrastiveConfig | None = None,
    ):
        """Initialize concept-aware contrastive module.

        Args:
            concept_lattice: Formal concept lattice
            config: Configuration
        """
        super().__init__()
        self.lattice = concept_lattice
        self.config = config or ConceptContrastiveConfig()

        # Build sample-to-concept mapping
        self._build_concept_membership()

        # Build concept distance matrix
        self._build_concept_distances()

    def _build_concept_membership(self):
        """Build mapping from samples to their concepts."""
        self.sample_concepts: dict[str, list[int]] = {}
        self.concept_samples: dict[int, set[str]] = {}

        for idx, concept in enumerate(self.lattice.concepts):
            self.concept_samples[idx] = set(concept.extent)

            for sample in concept.extent:
                if sample not in self.sample_concepts:
                    self.sample_concepts[sample] = []
                self.sample_concepts[sample].append(idx)

    def _build_concept_distances(self):
        """Build pairwise concept distances in lattice."""
        n_concepts = len(self.lattice.concepts)
        distances = torch.zeros(n_concepts, n_concepts)

        for i in range(n_concepts):
            for j in range(i + 1, n_concepts):
                # Distance based on extent overlap
                extent_i = self.lattice.concepts[i].extent
                extent_j = self.lattice.concepts[j].extent

                if extent_i and extent_j:
                    intersection = len(extent_i & extent_j)
                    union = len(extent_i | extent_j)
                    similarity = intersection / union if union > 0 else 0
                    distance = 1 - similarity
                else:
                    distance = 1.0

                distances[i, j] = distance
                distances[j, i] = distance

        self.register_buffer("concept_distances", distances)

    def forward(
        self,
        embeddings: torch.Tensor,
        sample_ids: list[str],
    ) -> torch.Tensor:
        """Compute concept-aware contrastive loss.

        Args:
            embeddings: Sample embeddings (batch_size, embed_dim)
            sample_ids: Sample identifiers

        Returns:
            Contrastive loss
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)

        # Compute pairwise similarities
        similarity = torch.mm(embeddings, embeddings.t()) / self.config.temperature

        # Build concept-based masks
        positive_mask, negative_mask, weights = self._build_masks(sample_ids, batch_size, device)

        # Compute loss
        loss = self._compute_loss(similarity, positive_mask, negative_mask, weights)

        return loss

    def _build_masks(
        self,
        sample_ids: list[str],
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build positive/negative masks based on concept membership.

        Args:
            sample_ids: Sample IDs
            batch_size: Batch size
            device: Device

        Returns:
            Tuple of (positive_mask, negative_mask, weights)
        """
        positive_mask = torch.zeros(batch_size, batch_size, device=device)
        negative_mask = torch.zeros(batch_size, batch_size, device=device)
        weights = torch.ones(batch_size, batch_size, device=device)

        for i in range(batch_size):
            concepts_i = set(self.sample_concepts.get(sample_ids[i], []))

            for j in range(batch_size):
                if i == j:
                    continue

                concepts_j = set(self.sample_concepts.get(sample_ids[j], []))

                # Positive if share at least one concept
                if concepts_i & concepts_j:
                    positive_mask[i, j] = 1

                    # Weight by number of shared concepts
                    n_shared = len(concepts_i & concepts_j)
                    n_total = len(concepts_i | concepts_j)
                    weights[i, j] = n_shared / n_total if n_total > 0 else 1
                else:
                    negative_mask[i, j] = 1

                    # Hard negatives: samples in sibling concepts
                    if self.config.use_hard_negatives:
                        # Find minimum concept distance
                        if concepts_i and concepts_j:
                            min_dist = float("inf")
                            for ci in concepts_i:
                                for cj in concepts_j:
                                    if ci < len(self.concept_distances) and cj < len(self.concept_distances):
                                        dist = self.concept_distances[ci, cj].item()
                                        min_dist = min(min_dist, dist)

                            # Closer concepts = harder negatives = higher weight
                            if min_dist < float("inf"):
                                weights[i, j] = 1 + (1 - min_dist)

        return positive_mask, negative_mask, weights

    def _compute_loss(
        self,
        similarity: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted contrastive loss.

        Args:
            similarity: Pairwise similarity matrix
            positive_mask: Mask for positive pairs
            negative_mask: Mask for negative pairs
            weights: Per-pair weights

        Returns:
            Contrastive loss
        """
        batch_size = similarity.size(0)

        # For each sample, compute InfoNCE-style loss
        loss = torch.tensor(0.0, device=similarity.device)

        for i in range(batch_size):
            pos_mask = positive_mask[i]
            neg_mask = negative_mask[i]

            if pos_mask.sum() == 0:
                continue

            # Positive similarities
            pos_sim = similarity[i] * pos_mask * weights[i]

            # Negative similarities
            neg_sim = similarity[i] * neg_mask * weights[i]

            # Log-sum-exp over negatives (handle empty case)
            neg_indices = neg_mask.bool()
            if neg_indices.sum() == 0:
                # No negatives - skip this sample
                continue

            neg_logsumexp = torch.logsumexp(neg_sim[neg_indices], dim=0)

            # Loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
            for j in range(batch_size):
                if pos_mask[j] > 0:
                    sample_loss = -pos_sim[j] + neg_logsumexp
                    loss = loss + sample_loss * self.config.positive_weight

        # Normalize
        n_positives = positive_mask.sum()
        if n_positives > 0:
            loss = loss / n_positives

        return loss

    def get_positive_pairs(
        self,
        sample_ids: list[str],
    ) -> list[tuple[str, str]]:
        """Get positive pairs based on concept membership.

        Args:
            sample_ids: Available sample IDs

        Returns:
            List of (id1, id2) positive pairs
        """
        pairs = []
        sample_set = set(sample_ids)

        for concept in self.lattice.concepts:
            # Samples in same concept are positive pairs
            extent = [s for s in concept.extent if s in sample_set]

            for i, s1 in enumerate(extent):
                for s2 in extent[i + 1 :]:
                    pairs.append((s1, s2))

        return pairs


class LatticeContrastiveLoss(nn.Module):
    """Contrastive loss respecting resistance lattice structure.

    Uses the resistance lattice hierarchy to define similarity:
    - Samples at same level are positive pairs
    - Samples in parent-child relationships have moderate similarity
    - Samples in different branches are negative pairs
    """

    def __init__(
        self,
        lattice: ResistanceLattice,
        temperature: float = 0.1,
        level_margin: float = 0.1,
    ):
        """Initialize lattice contrastive loss.

        Args:
            lattice: Resistance lattice
            temperature: Contrastive temperature
            level_margin: Margin per level difference
        """
        super().__init__()
        self.lattice = lattice
        self.temperature = temperature
        self.level_margin = level_margin

    def forward(
        self,
        embeddings: torch.Tensor,
        mutation_sets: list[MutationSet],
    ) -> torch.Tensor:
        """Compute lattice-aware contrastive loss.

        Args:
            embeddings: Sample embeddings
            mutation_sets: Corresponding mutation sets

        Returns:
            Contrastive loss
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=-1)

        # Compute similarities
        similarity = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Build target similarity based on lattice
        target_similarity = self._compute_target_similarity(mutation_sets, device)

        # MSE loss between actual and target similarity
        mask = 1 - torch.eye(batch_size, device=device)
        loss = ((similarity - target_similarity) ** 2 * mask).sum() / mask.sum()

        return loss

    def _compute_target_similarity(
        self,
        mutation_sets: list[MutationSet],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute target similarity matrix from lattice.

        Args:
            mutation_sets: Mutation sets
            device: Device

        Returns:
            Target similarity matrix
        """
        n = len(mutation_sets)
        target = torch.zeros(n, n, device=device)

        for i in range(n):
            for j in range(n):
                if i == j:
                    target[i, j] = 1.0
                    continue

                # Compare in lattice
                cmp = self.lattice.compare(mutation_sets[i], mutation_sets[j])

                if cmp is None:
                    # Incomparable - moderate similarity
                    target[i, j] = 0.3
                elif cmp == 0:
                    # Equal - high similarity
                    target[i, j] = 0.9
                else:
                    # Parent-child relationship
                    # Similarity decreases with level difference
                    level_i = self.lattice.resistance_level(mutation_sets[i]).value
                    level_j = self.lattice.resistance_level(mutation_sets[j]).value
                    level_diff = abs(level_i - level_j)
                    target[i, j] = max(0.1, 0.8 - level_diff * self.level_margin)

        return target


class ConceptPrototypeNetwork(nn.Module):
    """Learn prototype embeddings for each FCA concept.

    Each concept has a learnable prototype, and samples
    are classified based on distance to prototypes.
    """

    def __init__(
        self,
        concept_lattice: ConceptLattice,
        embed_dim: int = 64,
        use_hyperbolic: bool = True,
        curvature: float = 1.0,
    ):
        """Initialize prototype network.

        Args:
            concept_lattice: Concept lattice
            embed_dim: Embedding dimension
            use_hyperbolic: V5.12.2 - Use poincare_distance for hyperbolic embeddings (default True)
            curvature: Hyperbolic curvature for poincare_distance
        """
        super().__init__()
        self.lattice = concept_lattice
        self.use_hyperbolic = use_hyperbolic
        self.curvature = curvature

        n_concepts = len(concept_lattice.concepts)
        self.prototypes = nn.Parameter(torch.randn(n_concepts, embed_dim))

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distances to concept prototypes.

        Args:
            embeddings: Sample embeddings (batch_size, embed_dim)

        Returns:
            Distances to each prototype (batch_size, n_concepts)
        """
        # V5.12.2: Support both Euclidean and hyperbolic distance
        if self.use_hyperbolic:
            # Compute hyperbolic distances to each prototype
            batch_size = embeddings.size(0)
            n_concepts = self.prototypes.size(0)
            distances = torch.zeros(batch_size, n_concepts, device=embeddings.device)
            for i in range(n_concepts):
                prototype = self.prototypes[i].unsqueeze(0).expand(batch_size, -1)
                distances[:, i] = poincare_distance(embeddings, prototype, c=self.curvature)
        else:
            # Euclidean distance to prototypes
            diff = embeddings.unsqueeze(1) - self.prototypes.unsqueeze(0)
            distances = torch.norm(diff, dim=-1)

        return distances

    def classify(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Classify samples to nearest concept.

        Args:
            embeddings: Sample embeddings

        Returns:
            Concept indices
        """
        distances = self.forward(embeddings)
        return distances.argmin(dim=-1)

    def prototype_loss(
        self,
        embeddings: torch.Tensor,
        sample_ids: list[str],
    ) -> torch.Tensor:
        """Compute loss to pull samples toward their concept prototypes.

        Args:
            embeddings: Sample embeddings
            sample_ids: Sample identifiers

        Returns:
            Prototype loss
        """
        distances = self.forward(embeddings)
        device = embeddings.device

        loss = torch.tensor(0.0, device=device)
        n_samples = 0

        for i, sample_id in enumerate(sample_ids):
            # Find concepts containing this sample
            for j, concept in enumerate(self.lattice.concepts):
                if sample_id in concept.extent:
                    # Pull toward this concept's prototype
                    loss = loss + distances[i, j]
                    n_samples += 1

        if n_samples > 0:
            loss = loss / n_samples

        return loss
