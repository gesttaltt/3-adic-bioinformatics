# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unified Epistasis Loss for mutation interaction modeling.

This loss combines multiple components to model epistatic effects:
1. Coevolution constraints from coevolution_loss.py
2. Drug interaction penalties from drug_interaction.py
3. Learned epistasis patterns from epistasis_module.py

The goal is to train VAE models that capture how mutations interact
to affect drug resistance, fitness, and other phenotypes.

Usage:
    from src.losses.epistasis_loss import EpistasisLoss

    loss_fn = EpistasisLoss(latent_dim=16)
    result = loss_fn(model_output, targets, mutation_info)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.coevolution_loss import CoEvolutionLoss


@dataclass
class EpistasisLossResult:
    """Container for epistasis loss computation results.

    Attributes:
        total_loss: Combined loss value
        epistasis_loss: Learned epistasis component
        coevolution_loss: Evolutionary constraint loss
        drug_interaction_loss: Drug cross-resistance loss
        margin_loss: Margin ranking loss for fitness ordering
        metrics: Dictionary of additional metrics
    """

    total_loss: torch.Tensor
    epistasis_loss: torch.Tensor
    coevolution_loss: torch.Tensor
    drug_interaction_loss: torch.Tensor
    margin_loss: torch.Tensor
    metrics: dict[str, float]


class LearnedEpistasisLoss(nn.Module):
    """Loss that encourages learning of epistatic interaction patterns.

    This loss penalizes predictions that don't account for mutation
    interactions. It assumes additive effects unless epistasis is detected.
    """

    def __init__(
        self,
        margin: float = 0.1,
        interaction_weight: float = 1.0,
    ):
        """Initialize learned epistasis loss.

        Args:
            margin: Margin for ranking loss
            interaction_weight: Weight for interaction term
        """
        super().__init__()
        self.margin = margin
        self.interaction_weight = interaction_weight

    def forward(
        self,
        single_effects: torch.Tensor,
        combined_effects: torch.Tensor,
        predicted_combined: torch.Tensor,
        target_combined: torch.Tensor,
    ) -> torch.Tensor:
        """Compute epistasis loss.

        If mutations are additive: combined ≈ sum(single)
        If epistatic: combined ≠ sum(single)

        Loss encourages model to predict epistasis when needed.

        Args:
            single_effects: Predicted effects of individual mutations (batch, n_muts)
            combined_effects: Sum of single effects (batch,)
            predicted_combined: Model's prediction for combination (batch,)
            target_combined: True combined effect (batch,)

        Returns:
            Epistasis loss (scalar)
        """
        # Additive assumption
        additive_prediction = combined_effects

        # True epistasis signal
        true_epistasis = target_combined - additive_prediction

        # Predicted epistasis
        predicted_epistasis = predicted_combined - additive_prediction

        # Loss: predict epistasis correctly
        epistasis_mse = F.mse_loss(predicted_epistasis, true_epistasis)

        # Regularization: don't hallucinate epistasis
        # (small when true epistasis is small)
        magnitude_reg = (predicted_epistasis**2 * (1 - true_epistasis.abs().clamp(0, 1))).mean()

        return epistasis_mse + self.interaction_weight * magnitude_reg


class DrugInteractionLoss(nn.Module):
    """Loss for drug cross-resistance patterns.

    Based on hyperbolic contrastive learning to capture hierarchical
    drug class relationships. Drugs in the same class should have
    similar resistance patterns.
    """

    def __init__(
        self,
        n_drugs: int,
        embed_dim: int = 32,
        curvature: float = 1.0,
        temperature: float = 0.1,
    ):
        """Initialize drug interaction loss.

        Args:
            n_drugs: Number of drugs
            embed_dim: Drug embedding dimension
            curvature: Hyperbolic curvature
            temperature: Contrastive temperature
        """
        super().__init__()

        self.n_drugs = n_drugs
        self.curvature = curvature
        self.temperature = temperature

        # Drug embeddings in hyperbolic space
        self.drug_embeddings = nn.Parameter(torch.randn(n_drugs, embed_dim) * 0.01)

        # Drug class assignments (to be set externally)
        self.register_buffer("drug_classes", torch.zeros(n_drugs, dtype=torch.long))

    def set_drug_classes(self, drug_classes: torch.Tensor):
        """Set drug class assignments.

        Args:
            drug_classes: Class index for each drug (n_drugs,)
        """
        self.drug_classes = drug_classes

    def hyperbolic_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise hyperbolic distances."""
        c = self.curvature

        # Project to Poincare ball (ensure norm < 1)
        x_norm = torch.norm(x, dim=-1, keepdim=True)
        y_norm = torch.norm(y, dim=-1, keepdim=True)
        x = x / x_norm.clamp(min=1e-8) * torch.clamp(x_norm, max=1 - 1e-5)
        y = y / y_norm.clamp(min=1e-8) * torch.clamp(y_norm, max=1 - 1e-5)

        # Distance using Poincare formula
        diff_norm = torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1)
        x_sq = (x**2).sum(-1, keepdim=True)
        y_sq = (y**2).sum(-1, keepdim=True).t()

        num = diff_norm**2
        denom = (1 - c * x_sq) * (1 - c * y_sq)

        # Compute arcosh safely (argument >= 1)
        ratio = 1 + 2 * c * num / denom.clamp(min=1e-8)
        ratio = ratio.clamp(min=1.0)  # Ensure >= 1 for arcosh

        return torch.acosh(ratio)

    def forward(
        self,
        resistance_predictions: torch.Tensor,
        drug_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute drug interaction loss.

        Encourages similar drugs to have similar resistance predictions.

        Args:
            resistance_predictions: Predicted resistance per drug (batch, n_drugs)
            drug_indices: Optional subset of drug indices to use

        Returns:
            Drug interaction loss (scalar)
        """
        # Handle edge cases
        if resistance_predictions.shape[0] < 2:
            return torch.tensor(0.0, device=resistance_predictions.device)

        # Same class mask
        same_class = (self.drug_classes.unsqueeze(0) == self.drug_classes.unsqueeze(1)).float()

        # Check if all same class (no contrastive signal)
        if same_class.sum() == same_class.numel():
            return torch.tensor(0.0, device=resistance_predictions.device)

        # Prediction similarity (cosine) - handle zero vectors
        pred_norm = F.normalize(resistance_predictions + 1e-8, dim=0)
        pred_sim = torch.mm(pred_norm.t(), pred_norm)

        # Clamp similarity to valid range for log
        pred_sim = pred_sim.clamp(min=1e-6, max=1 - 1e-6)

        # Contrastive loss with numerical stability
        # Positive pairs (same class) should have high similarity
        pos_mask = same_class.bool()
        neg_mask = ~pos_mask

        # Binary cross-entropy style loss
        pos_loss = -torch.log(pred_sim[pos_mask]).mean() if pos_mask.any() else torch.tensor(0.0)
        neg_loss = -torch.log(1 - pred_sim[neg_mask]).mean() if neg_mask.any() else torch.tensor(0.0)

        return pos_loss + neg_loss


class MarginRankingLoss(nn.Module):
    """Margin ranking loss for fitness/resistance ordering.

    Ensures that mutations with higher true resistance are predicted
    to have higher resistance than those with lower.
    """

    def __init__(self, margin: float = 0.1):
        """Initialize margin ranking loss.

        Args:
            margin: Minimum margin between ordered pairs
        """
        super().__init__()
        self.margin = margin
        self.loss_fn = nn.MarginRankingLoss(margin=margin)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute margin ranking loss.

        Args:
            predictions: Model predictions (batch,) or (batch, n_outputs)
            targets: True values (batch,) or (batch, n_outputs)
            mask: Optional mask for valid comparisons

        Returns:
            Ranking loss (scalar)
        """
        if predictions.dim() > 1:
            predictions = predictions.mean(dim=-1)
        if targets.dim() > 1:
            targets = targets.mean(dim=-1)

        batch_size = predictions.shape[0]
        if batch_size < 2:
            return torch.tensor(0.0, device=predictions.device)

        # Create all pairs
        total_loss = torch.tensor(0.0, device=predictions.device)
        count = 0

        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                if mask is not None and (not mask[i] or not mask[j]):
                    continue

                target_diff = targets[i] - targets[j]
                if abs(target_diff) < 1e-6:
                    continue  # Skip equal targets

                # y = 1 if targets[i] > targets[j], else -1
                y = torch.sign(target_diff).unsqueeze(0)
                loss = self.loss_fn(predictions[i].unsqueeze(0), predictions[j].unsqueeze(0), y)
                total_loss += loss
                count += 1

        return total_loss / max(count, 1)


class EpistasisLoss(nn.Module):
    """Unified epistasis loss combining all components.

    This is the main loss function for training models that capture
    mutation interactions and epistatic effects.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        n_drugs: int = 1,
        p: int = 3,
        weights: dict[str, float] | None = None,
        use_coevolution: bool = True,
        use_drug_interaction: bool = True,
        use_margin_ranking: bool = True,
    ):
        """Initialize unified epistasis loss.

        Args:
            latent_dim: Latent space dimension
            n_drugs: Number of drugs for drug interaction loss
            p: Prime for p-adic calculations in coevolution
            weights: Component weights
            use_coevolution: Whether to include coevolution loss
            use_drug_interaction: Whether to include drug interaction loss
            use_margin_ranking: Whether to include margin ranking loss
        """
        super().__init__()

        self.weights = weights or {
            "epistasis": 0.3,
            "coevolution": 0.3,
            "drug_interaction": 0.2,
            "margin": 0.2,
        }

        # Learned epistasis
        self.learned_epistasis = LearnedEpistasisLoss()

        # Coevolution (from existing module)
        if use_coevolution:
            self.coevolution = CoEvolutionLoss(latent_dim=latent_dim, p=p)
        else:
            self.coevolution = None

        # Drug interaction
        if use_drug_interaction and n_drugs > 1:
            self.drug_interaction = DrugInteractionLoss(n_drugs=n_drugs)
        else:
            self.drug_interaction = None

        # Margin ranking
        if use_margin_ranking:
            self.margin_ranking = MarginRankingLoss()
        else:
            self.margin_ranking = None

    def forward(
        self,
        model_output: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        mutation_info: dict[str, Any] | None = None,
    ) -> EpistasisLossResult:
        """Compute unified epistasis loss.

        Args:
            model_output: Model outputs with keys:
                - predictions: Drug resistance predictions (batch, n_drugs)
                - z: Latent vectors (batch, latent_dim)
                - single_effects: Optional individual mutation effects
                - codon_embeddings: Optional for coevolution loss
            targets: Target values with keys:
                - resistance: True resistance values
                - codon_indices: For coevolution loss
            mutation_info: Optional mutation details:
                - positions: Mutation positions
                - amino_acids: Mutant amino acids

        Returns:
            EpistasisLossResult with all loss components
        """
        device = model_output.get("predictions", model_output.get("z")).device

        # Initialize losses
        epistasis_loss = torch.tensor(0.0, device=device)
        coevolution_loss = torch.tensor(0.0, device=device)
        drug_interaction_loss = torch.tensor(0.0, device=device)
        margin_loss = torch.tensor(0.0, device=device)

        # 1. Learned epistasis loss
        if "single_effects" in model_output and "predictions" in model_output:
            single_effects = model_output["single_effects"]
            # Sum over mutation dimension (dim=1 if 3D), keep drug dimension
            if single_effects.dim() == 3:
                # (batch, n_muts, n_drugs) -> (batch, n_drugs)
                combined_effects = single_effects.sum(dim=1)
            else:
                combined_effects = single_effects

            predictions = model_output["predictions"]

            if "resistance" in targets:
                target_combined = targets["resistance"]

                # Ensure same dimensions - average if needed
                if combined_effects.dim() != target_combined.dim():
                    combined_effects = combined_effects.mean(dim=-1) if combined_effects.dim() > 1 else combined_effects
                    target_combined = target_combined.mean(dim=-1) if target_combined.dim() > 1 else target_combined
                    predictions = predictions.mean(dim=-1) if predictions.dim() > 1 else predictions

                epistasis_loss = self.learned_epistasis(
                    single_effects=single_effects,
                    combined_effects=combined_effects,
                    predicted_combined=predictions,
                    target_combined=target_combined,
                )

        # 2. Coevolution loss
        if self.coevolution is not None and "codon_embeddings" in model_output:
            codon_emb = model_output["codon_embeddings"]
            codon_idx = targets.get("codon_indices", torch.zeros_like(codon_emb[:, :, 0]).long())
            codon_probs = model_output.get("codon_probabilities")

            coev_result = self.coevolution(codon_emb, codon_idx, codon_probs)
            coevolution_loss = coev_result["loss"]

        # 3. Drug interaction loss
        if self.drug_interaction is not None and "predictions" in model_output:
            predictions = model_output["predictions"]
            if predictions.dim() > 1 and predictions.shape[1] > 1:
                drug_interaction_loss = self.drug_interaction(predictions)

        # 4. Margin ranking loss
        if self.margin_ranking is not None and "predictions" in model_output:
            predictions = model_output["predictions"]
            if "resistance" in targets:
                resistance_targets = targets["resistance"]
                margin_loss = self.margin_ranking(predictions, resistance_targets)

        # Combine losses
        total_loss = (
            self.weights["epistasis"] * epistasis_loss
            + self.weights["coevolution"] * coevolution_loss
            + self.weights["drug_interaction"] * drug_interaction_loss
            + self.weights["margin"] * margin_loss
        )

        # Compute metrics
        metrics = {
            "epistasis_loss": epistasis_loss.item(),
            "coevolution_loss": coevolution_loss.item(),
            "drug_interaction_loss": drug_interaction_loss.item(),
            "margin_loss": margin_loss.item(),
            "total_loss": total_loss.item(),
        }

        return EpistasisLossResult(
            total_loss=total_loss,
            epistasis_loss=epistasis_loss,
            coevolution_loss=coevolution_loss,
            drug_interaction_loss=drug_interaction_loss,
            margin_loss=margin_loss,
            metrics=metrics,
        )


__all__ = [
    "EpistasisLoss",
    "EpistasisLossResult",
    "LearnedEpistasisLoss",
    "DrugInteractionLoss",
    "MarginRankingLoss",
]
