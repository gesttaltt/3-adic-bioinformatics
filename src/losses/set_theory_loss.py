# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Set-theoretic loss functions for resistance prediction.

Provides loss functions that incorporate set-theoretic constraints:
- Lattice ordering consistency
- Concept consistency (samples in same FCA concept should be similar)
- Cross-resistance awareness
- Rough set boundary handling
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.analysis.set_theory.formal_concepts import ConceptLattice
from src.analysis.set_theory.lattice import ResistanceLattice, ResistanceLevel
from src.analysis.set_theory.mutation_sets import MutationSet
from src.analysis.set_theory.rough_sets import RoughClassifier
from src.geometry import poincare_distance


@dataclass
class SetLossConfig:
    """Configuration for set-theoretic losses.

    Attributes:
        lattice_weight: Weight for lattice ordering loss
        concept_weight: Weight for concept consistency loss
        rough_weight: Weight for rough set boundary loss
        cross_resistance_weight: Weight for cross-resistance loss
        margin: Margin for ranking losses
        temperature: Temperature for softmax operations
    """

    lattice_weight: float = 0.1
    concept_weight: float = 0.1
    rough_weight: float = 0.1
    cross_resistance_weight: float = 0.05
    margin: float = 0.1
    temperature: float = 1.0


class LatticeOrderingLoss(nn.Module):
    """Loss enforcing lattice ordering in embeddings.

    If mutation set A ⊆ B in the lattice (A less resistant than B),
    then the embedding of A should be "closer to origin" than B.

    This is particularly suited for hyperbolic embeddings where
    distance from origin indicates hierarchy depth.
    """

    def __init__(
        self,
        lattice: ResistanceLattice,
        margin: float = 0.1,
        distance_fn: str = "hyperbolic",
        curvature: float = 1.0,
    ):
        """Initialize lattice ordering loss.

        Args:
            lattice: Resistance lattice structure
            margin: Ranking margin
            distance_fn: V5.12.2 - Distance function ('hyperbolic' default, or 'euclidean')
            curvature: Hyperbolic curvature for poincare_distance
        """
        super().__init__()
        self.lattice = lattice
        self.margin = margin
        self.distance_fn = distance_fn
        self.curvature = curvature

    def forward(
        self,
        embeddings: torch.Tensor,
        mutation_sets: list[MutationSet],
    ) -> torch.Tensor:
        """Compute lattice ordering loss.

        Args:
            embeddings: Batch of embeddings (batch_size, embed_dim)
            mutation_sets: Corresponding mutation sets

        Returns:
            Ordering loss
        """
        if len(mutation_sets) < 2:
            return torch.tensor(0.0, device=embeddings.device)

        loss = torch.tensor(0.0, device=embeddings.device)
        n_pairs = 0

        # Compute distances from origin
        if self.distance_fn == "euclidean":
            distances = torch.norm(embeddings, dim=-1)
        else:
            # V5.12.2: Use proper poincare_distance from origin
            origin = torch.zeros_like(embeddings)
            distances = poincare_distance(embeddings, origin, c=self.curvature)

        # Check all pairs for ordering violations
        for i in range(len(mutation_sets)):
            for j in range(i + 1, len(mutation_sets)):
                ms_i = mutation_sets[i]
                ms_j = mutation_sets[j]

                # Compare in lattice
                cmp = self.lattice.compare(ms_i, ms_j)

                if cmp == -1:
                    # ms_i < ms_j: dist_i should be < dist_j
                    violation = F.relu(distances[i] - distances[j] + self.margin)
                    loss = loss + violation
                    n_pairs += 1
                elif cmp == 1:
                    # ms_i > ms_j: dist_i should be > dist_j
                    violation = F.relu(distances[j] - distances[i] + self.margin)
                    loss = loss + violation
                    n_pairs += 1
                # If cmp is 0 or None (incomparable), no constraint

        if n_pairs > 0:
            loss = loss / n_pairs

        return loss


class HierarchicalResistanceLoss(nn.Module):
    """Loss that penalizes predictions violating resistance hierarchy.

    Predictions that jump multiple levels in the resistance hierarchy
    (e.g., predicting XDR as Susceptible) are penalized more heavily.
    """

    def __init__(
        self,
        lattice: ResistanceLattice,
        level_penalty_scale: float = 1.0,
    ):
        """Initialize hierarchical resistance loss.

        Args:
            lattice: Resistance lattice for level computation
            level_penalty_scale: Scale factor for level distance penalty
        """
        super().__init__()
        self.lattice = lattice
        self.level_penalty_scale = level_penalty_scale

        # Build level distance matrix
        n_levels = len(ResistanceLevel)
        level_distances = torch.zeros(n_levels, n_levels)
        for i in range(n_levels):
            for j in range(n_levels):
                level_distances[i, j] = abs(i - j)
        self.register_buffer("level_distances", level_distances)

    def forward(
        self,
        predictions: torch.Tensor,
        mutation_sets: list[MutationSet],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hierarchical resistance loss.

        Args:
            predictions: Predicted resistance levels (batch_size, n_levels)
            mutation_sets: Mutation sets for computing true levels
            targets: True resistance level indices

        Returns:
            Weighted cross-entropy loss
        """
        batch_size = predictions.size(0)
        device = predictions.device

        # Standard cross-entropy
        ce_loss = F.cross_entropy(predictions, targets, reduction="none")

        # Get predicted levels
        pred_levels = predictions.argmax(dim=-1)

        # Compute level distance penalty
        penalties = torch.zeros(batch_size, device=device)
        for i in range(batch_size):
            true_level = targets[i].item()
            pred_level = pred_levels[i].item()
            penalties[i] = self.level_distances[true_level, pred_level]

        # Weighted loss: base CE + penalty for hierarchy violations
        weighted_loss = ce_loss * (1 + self.level_penalty_scale * penalties)

        return weighted_loss.mean()


class ConceptConsistencyLoss(nn.Module):
    """Loss enforcing that samples in the same FCA concept are similar.

    Samples belonging to the same formal concept should have
    similar embeddings, as they share the same attribute structure.
    """

    def __init__(
        self,
        concept_lattice: ConceptLattice,
        temperature: float = 0.1,
    ):
        """Initialize concept consistency loss.

        Args:
            concept_lattice: Formal concept lattice
            temperature: Temperature for similarity computation
        """
        super().__init__()
        self.concept_lattice = concept_lattice
        self.temperature = temperature

        # Build concept membership for efficient lookup
        self._build_concept_membership()

    def _build_concept_membership(self):
        """Build mapping from samples to concepts."""
        self.sample_to_concepts: dict[str, list[int]] = {}

        for idx, concept in enumerate(self.concept_lattice.concepts):
            for sample in concept.extent:
                if sample not in self.sample_to_concepts:
                    self.sample_to_concepts[sample] = []
                self.sample_to_concepts[sample].append(idx)

    def forward(
        self,
        embeddings: torch.Tensor,
        sample_ids: list[str],
    ) -> torch.Tensor:
        """Compute concept consistency loss.

        Args:
            embeddings: Sample embeddings (batch_size, embed_dim)
            sample_ids: Sample identifiers

        Returns:
            Concept consistency loss
        """
        batch_size = embeddings.size(0)
        device = embeddings.device

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Compute pairwise cosine similarity
        normalized = F.normalize(embeddings, dim=-1)
        similarity = torch.mm(normalized, normalized.t()) / self.temperature

        # Create concept mask (1 if same concept, 0 otherwise)
        concept_mask = torch.zeros(batch_size, batch_size, device=device)

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                concepts_i = set(self.sample_to_concepts.get(sample_ids[i], []))
                concepts_j = set(self.sample_to_concepts.get(sample_ids[j], []))

                if concepts_i & concepts_j:  # Share at least one concept
                    concept_mask[i, j] = 1
                    concept_mask[j, i] = 1

        # Contrastive loss: pull together same-concept samples
        # Push apart different-concept samples
        positive_sim = (similarity * concept_mask).sum() / (concept_mask.sum() + 1e-8)

        negative_mask = 1 - concept_mask - torch.eye(batch_size, device=device)
        negative_sim = (similarity * negative_mask).sum() / (negative_mask.sum() + 1e-8)

        # Maximize positive, minimize negative
        loss = -positive_sim + F.relu(negative_sim + 0.5)

        return loss


class RoughBoundaryLoss(nn.Module):
    """Loss for handling rough set boundary region samples.

    Samples in the boundary region (uncertain) should have
    higher prediction entropy, reflecting the inherent uncertainty.
    """

    def __init__(
        self,
        rough_classifiers: dict[str, RoughClassifier],
        entropy_target: float = 0.5,
    ):
        """Initialize rough boundary loss.

        Args:
            rough_classifiers: Per-drug rough classifiers
            entropy_target: Target entropy for boundary samples
        """
        super().__init__()
        self.rough_classifiers = rough_classifiers
        self.entropy_target = entropy_target

    def forward(
        self,
        predictions: torch.Tensor,
        mutation_sets: list[MutationSet],
        drug_idx: int,
        drug_name: str,
    ) -> torch.Tensor:
        """Compute rough boundary loss for a specific drug.

        Args:
            predictions: Predictions for this drug (batch_size,)
            mutation_sets: Mutation sets
            drug_idx: Drug index
            drug_name: Drug name

        Returns:
            Boundary-aware loss
        """
        device = predictions.device
        len(mutation_sets)

        if drug_name not in self.rough_classifiers:
            return torch.tensor(0.0, device=device)

        classifier = self.rough_classifiers[drug_name]
        loss = torch.tensor(0.0, device=device)
        n_boundary = 0

        for i, mutations in enumerate(mutation_sets):
            # Check if any mutation is in boundary
            is_boundary = any(classifier.positive_mutations.uncertain(mut) for mut in mutations)

            if is_boundary:
                # For boundary samples, encourage uncertain predictions
                pred = torch.sigmoid(predictions[i])
                # Binary entropy
                entropy = -pred * torch.log(pred + 1e-8) - (1 - pred) * torch.log(1 - pred + 1e-8)
                # Penalize low entropy for boundary samples
                loss = loss + F.relu(self.entropy_target - entropy)
                n_boundary += 1

        if n_boundary > 0:
            loss = loss / n_boundary

        return loss


class CrossResistanceLoss(nn.Module):
    """Loss encouraging consistent predictions for cross-resistant drugs.

    If two drugs share significant cross-resistance (based on mutation overlap),
    their predictions should be correlated.
    """

    def __init__(
        self,
        cross_resistance_matrix: dict[tuple[str, str], float],
        drug_names: list[str],
        threshold: float = 0.3,
    ):
        """Initialize cross-resistance loss.

        Args:
            cross_resistance_matrix: Pairwise cross-resistance scores
            drug_names: Drug name list
            threshold: Minimum cross-resistance for constraint
        """
        super().__init__()
        self.drug_names = drug_names
        self.threshold = threshold

        # Build correlation matrix
        n_drugs = len(drug_names)
        corr = torch.zeros(n_drugs, n_drugs)

        for i, drug1 in enumerate(drug_names):
            for j, drug2 in enumerate(drug_names):
                score = cross_resistance_matrix.get((drug1, drug2), 0.0)
                if score >= threshold:
                    corr[i, j] = score

        self.register_buffer("correlation", corr)

    def forward(
        self,
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-resistance consistency loss.

        Args:
            predictions: Per-drug predictions (batch_size, n_drugs)

        Returns:
            Cross-resistance loss
        """
        batch_size, n_drugs = predictions.shape
        device = predictions.device

        # Convert predictions to probabilities
        probs = torch.sigmoid(predictions)

        loss = torch.tensor(0.0, device=device)
        n_pairs = 0

        for i in range(n_drugs):
            for j in range(i + 1, n_drugs):
                if self.correlation[i, j] > 0:
                    # Highly cross-resistant drugs should have similar predictions
                    # Penalize large differences
                    weight = self.correlation[i, j]
                    diff = (probs[:, i] - probs[:, j]).abs().mean()
                    loss = loss + weight * diff
                    n_pairs += 1

        if n_pairs > 0:
            loss = loss / n_pairs

        return loss


class SetTheoryAwareLoss(nn.Module):
    """Combined loss incorporating all set-theoretic constraints.

    Combines standard prediction loss with:
    - Lattice ordering constraints
    - Hierarchical resistance penalties
    - Concept consistency
    - Rough boundary handling
    - Cross-resistance awareness
    """

    def __init__(
        self,
        config: SetLossConfig | None = None,
        lattice: ResistanceLattice | None = None,
        concept_lattice: ConceptLattice | None = None,
        rough_classifiers: dict[str, RoughClassifier] | None = None,
        cross_resistance_matrix: dict[tuple[str, str], float] | None = None,
        drug_names: list[str] | None = None,
    ):
        """Initialize combined set-aware loss.

        Args:
            config: Loss configuration
            lattice: Resistance lattice
            concept_lattice: Formal concept lattice
            rough_classifiers: Per-drug rough classifiers
            cross_resistance_matrix: Pairwise cross-resistance
            drug_names: Drug names
        """
        super().__init__()
        self.config = config or SetLossConfig()

        # Initialize component losses
        self.lattice_loss = None
        if lattice:
            self.lattice_loss = LatticeOrderingLoss(lattice, self.config.margin)

        self.hierarchical_loss = None
        if lattice:
            self.hierarchical_loss = HierarchicalResistanceLoss(lattice)

        self.concept_loss = None
        if concept_lattice:
            self.concept_loss = ConceptConsistencyLoss(concept_lattice, self.config.temperature)

        self.rough_loss = None
        if rough_classifiers:
            self.rough_loss = RoughBoundaryLoss(rough_classifiers)
            self.rough_classifiers = rough_classifiers

        self.cross_loss = None
        if cross_resistance_matrix and drug_names:
            self.cross_loss = CrossResistanceLoss(cross_resistance_matrix, drug_names)

        self.drug_names = drug_names or []

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        embeddings: torch.Tensor | None = None,
        mutation_sets: list[MutationSet] | None = None,
        sample_ids: list[str] | None = None,
        resistance_levels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            predictions: Per-drug predictions (batch_size, n_drugs)
            targets: Ground truth labels
            embeddings: Sample embeddings
            mutation_sets: Mutation sets
            sample_ids: Sample identifiers
            resistance_levels: True resistance levels

        Returns:
            Dictionary with total loss and component losses
        """
        device = predictions.device
        losses = {}

        # Standard BCE loss
        pred_loss = F.binary_cross_entropy_with_logits(predictions, targets)
        losses["prediction"] = pred_loss
        total_loss = pred_loss

        # Lattice ordering loss
        if self.lattice_loss and embeddings is not None and mutation_sets:
            lattice_loss = self.lattice_loss(embeddings, mutation_sets)
            losses["lattice"] = lattice_loss
            total_loss = total_loss + self.config.lattice_weight * lattice_loss

        # Concept consistency loss
        if self.concept_loss and embeddings is not None and sample_ids:
            concept_loss = self.concept_loss(embeddings, sample_ids)
            losses["concept"] = concept_loss
            total_loss = total_loss + self.config.concept_weight * concept_loss

        # Rough boundary loss (per drug)
        if self.rough_loss and mutation_sets:
            rough_total = torch.tensor(0.0, device=device)
            for idx, drug in enumerate(self.drug_names):
                if idx < predictions.size(1):
                    rough_loss = self.rough_loss(predictions[:, idx], mutation_sets, idx, drug)
                    rough_total = rough_total + rough_loss
            losses["rough"] = rough_total
            total_loss = total_loss + self.config.rough_weight * rough_total

        # Cross-resistance loss
        if self.cross_loss:
            cross_loss = self.cross_loss(predictions)
            losses["cross_resistance"] = cross_loss
            total_loss = total_loss + self.config.cross_resistance_weight * cross_loss

        losses["total"] = total_loss

        return losses
