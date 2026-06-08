# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""6-Component Loss Functions for PeptideVAE.

This module implements the loss functions for training the PeptideVAE model
for antimicrobial peptide activity prediction. Following the TrainableCodonEncoder
pattern (Spearman 0.60 for DDG), it uses a multi-objective loss combining:

1. ReconstructionLoss: Sequence cross-entropy (decoder regularization)
2. MICPredictionLoss: Primary supervision (Smooth L1)
3. PropertyAlignmentLoss: Embed distance ~ property distance
4. RadialHierarchyLoss: Low MIC → small radius (active = central)
5. CohesionLoss: Same pathogen clusters together
6. SeparationLoss: Different pathogens separate

Curriculum Schedule:
    - Epochs 0-10: Reconstruction focus (learn sequence structure)
    - Epochs 10-30: Add MIC prediction + property alignment
    - Epochs 30-50: Full multi-task with all 6 components

Usage:
    from src.losses.peptide_losses import PeptideLossManager

    loss_manager = PeptideLossManager()
    total_loss, metrics = loss_manager.compute_total_loss(
        model_output, target_mic, target_tokens, pathogen_labels, epoch
    )
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.geometry import poincare_distance
from src.losses.base import LossComponent, LossResult

# =============================================================================
# Reconstruction Loss
# =============================================================================


class ReconstructionLoss(LossComponent):
    """Sequence reconstruction loss using cross-entropy.

    Encourages the decoder to reconstruct the input sequence, providing
    regularization and ensuring the latent space captures sequence information.
    """

    def __init__(
        self,
        weight: float = 1.0,
        ignore_index: int = 21,  # PAD_IDX
        label_smoothing: float = 0.0,
    ):
        """Initialize reconstruction loss.

        Args:
            weight: Loss weight
            ignore_index: Index to ignore in loss (padding)
            label_smoothing: Label smoothing factor
        """
        super().__init__(weight=weight, name="reconstruction")
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
            reduction="mean",
        )

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: Tensor,
        **kwargs,
    ) -> LossResult:
        """Compute reconstruction loss.

        Args:
            outputs: Model outputs with 'logits' (batch, seq_len, vocab_size)
            targets: Target tokens (batch, seq_len)

        Returns:
            LossResult with cross-entropy loss
        """
        logits = outputs["logits"]

        # Flatten for cross-entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        loss = self.criterion(logits_flat, targets_flat)

        # Compute accuracy (excluding padding)
        with torch.no_grad():
            preds = logits_flat.argmax(dim=-1)
            mask = targets_flat != self.ignore_index
            correct = (preds == targets_flat) & mask
            accuracy = correct.float().sum() / mask.float().sum().clamp(min=1)

        return LossResult(
            loss=loss,
            metrics={
                "recon_ce": loss.item(),
                "recon_accuracy": accuracy.item(),
            },
            weight=self.weight,
        )


# =============================================================================
# MIC Prediction Loss
# =============================================================================


class MICPredictionLoss(LossComponent):
    """MIC prediction loss using Smooth L1 (Huber loss).

    Primary supervision signal for learning activity-predictive embeddings.
    Uses Smooth L1 for robustness to outliers in MIC measurements.
    """

    def __init__(
        self,
        weight: float = 2.0,
        beta: float = 1.0,
    ):
        """Initialize MIC prediction loss.

        Args:
            weight: Loss weight (higher = prioritize MIC prediction)
            beta: Smooth L1 beta parameter
        """
        super().__init__(weight=weight, name="mic_prediction")
        self.beta = beta

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: Tensor,
        mic_targets: Tensor | None = None,
        **kwargs,
    ) -> LossResult:
        """Compute MIC prediction loss.

        Args:
            outputs: Model outputs with 'mic_pred' (batch, 1)
            targets: Unused (for interface compatibility)
            mic_targets: Target log10(MIC) values (batch,) or (batch, 1)

        Returns:
            LossResult with Smooth L1 loss
        """
        if mic_targets is None:
            return LossResult(
                loss=torch.tensor(0.0, device=outputs["mic_pred"].device),
                metrics={"mic_loss": 0.0, "mic_mae": 0.0},
                weight=self.weight,
            )

        mic_pred = outputs["mic_pred"].squeeze(-1)
        mic_targets = mic_targets.squeeze(-1) if mic_targets.dim() > 1 else mic_targets

        loss = F.smooth_l1_loss(mic_pred, mic_targets, beta=self.beta)

        # Compute MAE for interpretability
        with torch.no_grad():
            mae = F.l1_loss(mic_pred, mic_targets)

        return LossResult(
            loss=loss,
            metrics={
                "mic_loss": loss.item(),
                "mic_mae": mae.item(),
            },
            weight=self.weight,
        )


# =============================================================================
# Property Alignment Loss
# =============================================================================


class PropertyAlignmentLoss(LossComponent):
    """Property alignment loss encouraging embed distances to match property distances.

    Peptides with similar physicochemical properties should cluster together
    in the embedding space.
    """

    def __init__(
        self,
        weight: float = 1.0,
        n_pairs: int = 100,
        temperature: float = 1.0,
    ):
        """Initialize property alignment loss.

        Args:
            weight: Loss weight
            n_pairs: Number of random pairs to sample per batch
            temperature: Temperature for distance scaling
        """
        super().__init__(weight=weight, name="property_alignment")
        self.n_pairs = n_pairs
        self.temperature = temperature

    def compute_property_distance(
        self,
        properties: Tensor,
        i: Tensor,
        j: Tensor,
    ) -> Tensor:
        """Compute normalized property distance between peptide pairs.

        Args:
            properties: Peptide properties (batch, n_props)
            i: First peptide indices
            j: Second peptide indices

        Returns:
            Property distances (n_pairs,)
        """
        prop_i = properties[i]
        prop_j = properties[j]
        dist = torch.norm(prop_i - prop_j, dim=-1)
        # Normalize to [0, 1] range
        return dist / (dist.max() + 1e-8)

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: Tensor,
        peptide_properties: Tensor | None = None,
        **kwargs,
    ) -> LossResult:
        """Compute property alignment loss.

        Args:
            outputs: Model outputs with 'z_hyp' (batch, latent_dim)
            targets: Unused
            peptide_properties: Peptide property features (batch, n_props)

        Returns:
            LossResult with correlation-based loss
        """
        if peptide_properties is None:
            return LossResult(
                loss=torch.tensor(0.0, device=outputs["z_hyp"].device),
                metrics={"property_loss": 0.0, "property_corr": 0.0},
                weight=self.weight,
            )

        z_hyp = outputs["z_hyp"]
        batch_size = z_hyp.shape[0]

        if batch_size < 2:
            return LossResult(
                loss=torch.tensor(0.0, device=z_hyp.device),
                metrics={"property_loss": 0.0, "property_corr": 0.0},
                weight=self.weight,
            )

        # Sample random pairs
        n_pairs = min(self.n_pairs, batch_size * (batch_size - 1) // 2)
        i = torch.randint(0, batch_size, (n_pairs,), device=z_hyp.device)
        j = torch.randint(0, batch_size, (n_pairs,), device=z_hyp.device)
        # Ensure i != j
        j = (i + j.remainder(batch_size - 1) + 1) % batch_size

        # Compute hyperbolic distances
        z_i = z_hyp[i]
        z_j = z_hyp[j]
        hyp_dists = poincare_distance(z_i, z_j, c=1.0) / self.temperature

        # Compute property distances
        prop_dists = self.compute_property_distance(peptide_properties, i, j)

        # Correlation-based loss: 1 - correlation
        hyp_centered = hyp_dists - hyp_dists.mean()
        prop_centered = prop_dists - prop_dists.mean()

        numerator = (hyp_centered * prop_centered).sum()
        denominator = torch.sqrt((hyp_centered**2).sum() * (prop_centered**2).sum() + 1e-8)
        correlation = numerator / denominator

        loss = 1.0 - correlation

        return LossResult(
            loss=loss,
            metrics={
                "property_loss": loss.item(),
                "property_corr": correlation.item(),
            },
            weight=self.weight,
        )


# =============================================================================
# Radial Hierarchy Loss
# =============================================================================


class RadialHierarchyLoss(LossComponent):
    """Radial hierarchy loss mapping MIC values to radial position.

    Active peptides (low MIC) should be positioned near the center of the
    Poincaré ball, while inactive peptides (high MIC) should be at the boundary.

    This encodes biological activity hierarchy in the geometric structure.
    """

    def __init__(
        self,
        weight: float = 0.5,
        max_radius: float = 0.95,
        min_radius: float = 0.1,
        mic_range: tuple[float, float] = (-1.0, 3.0),  # log10(MIC) range
        curvature: float = 1.0,
    ):
        """Initialize radial hierarchy loss.

        Args:
            weight: Loss weight
            max_radius: Target radius for inactive peptides (high MIC)
            min_radius: Target radius for active peptides (low MIC)
            mic_range: Expected range of log10(MIC) values
            curvature: Poincaré ball curvature
        """
        super().__init__(weight=weight, name="radial_hierarchy")
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.mic_range = mic_range
        self.curvature = curvature

    def mic_to_target_radius(self, mic_values: Tensor) -> Tensor:
        """Map MIC values to target radii.

        Args:
            mic_values: log10(MIC) values

        Returns:
            Target radii (low MIC → small radius, high MIC → large radius)
        """
        # Normalize MIC to [0, 1]
        mic_min, mic_max = self.mic_range
        mic_normalized = (mic_values - mic_min) / (mic_max - mic_min + 1e-8)
        mic_normalized = mic_normalized.clamp(0, 1)

        # Map to radius: low MIC (active) → small radius, high MIC → large radius
        target_radii = self.min_radius + mic_normalized * (self.max_radius - self.min_radius)

        return target_radii

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: Tensor,
        mic_targets: Tensor | None = None,
        **kwargs,
    ) -> LossResult:
        """Compute radial hierarchy loss.

        Args:
            outputs: Model outputs with 'z_hyp' and 'radius'
            targets: Unused
            mic_targets: Target log10(MIC) values

        Returns:
            LossResult with MSE between actual and target radii
        """
        if mic_targets is None:
            return LossResult(
                loss=torch.tensor(0.0, device=outputs["z_hyp"].device),
                metrics={"radial_loss": 0.0, "radius_mean": 0.0},
                weight=self.weight,
            )

        z_hyp = outputs["z_hyp"]
        mic_targets = mic_targets.squeeze(-1) if mic_targets.dim() > 1 else mic_targets

        # Get actual radii from hyperbolic embeddings
        origin = torch.zeros(1, z_hyp.shape[-1], device=z_hyp.device)
        actual_radii = poincare_distance(z_hyp, origin.expand(z_hyp.shape[0], -1), c=self.curvature)

        # Compute target radii from MIC values
        target_radii = self.mic_to_target_radius(mic_targets)

        # MSE loss
        loss = F.mse_loss(actual_radii, target_radii)

        return LossResult(
            loss=loss,
            metrics={
                "radial_loss": loss.item(),
                "radius_mean": actual_radii.mean().item(),
                "radius_std": actual_radii.std().item(),
                "target_radius_mean": target_radii.mean().item(),
            },
            weight=self.weight,
        )


# =============================================================================
# Cohesion Loss
# =============================================================================


class CohesionLoss(LossComponent):
    """Cohesion loss encouraging same-pathogen peptides to cluster.

    Peptides effective against the same pathogen should be nearby in
    the embedding space.
    """

    def __init__(
        self,
        weight: float = 0.3,
        curvature: float = 1.0,
    ):
        """Initialize cohesion loss.

        Args:
            weight: Loss weight
            curvature: Poincaré ball curvature
        """
        super().__init__(weight=weight, name="cohesion")
        self.curvature = curvature

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: Tensor,
        pathogen_labels: Tensor | None = None,
        **kwargs,
    ) -> LossResult:
        """Compute cohesion loss.

        Args:
            outputs: Model outputs with 'z_hyp'
            targets: Unused
            pathogen_labels: Pathogen class labels (batch,)

        Returns:
            LossResult with intra-class distance loss
        """
        if pathogen_labels is None:
            return LossResult(
                loss=torch.tensor(0.0, device=outputs["z_hyp"].device),
                metrics={"cohesion_loss": 0.0},
                weight=self.weight,
            )

        z_hyp = outputs["z_hyp"]
        batch_size = z_hyp.shape[0]

        if batch_size < 2:
            return LossResult(
                loss=torch.tensor(0.0, device=z_hyp.device),
                metrics={"cohesion_loss": 0.0},
                weight=self.weight,
            )

        total_loss = torch.tensor(0.0, device=z_hyp.device)
        n_groups = 0

        # Get unique pathogen labels
        unique_labels = pathogen_labels.unique()

        for label in unique_labels:
            mask = pathogen_labels == label
            group_z = z_hyp[mask]

            if group_z.shape[0] < 2:
                continue

            # Compute pairwise hyperbolic distances within group
            n = group_z.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    dist = poincare_distance(group_z[i : i + 1], group_z[j : j + 1], c=self.curvature)
                    total_loss = total_loss + dist

            n_groups += n * (n - 1) // 2

        if n_groups == 0:
            return LossResult(
                loss=torch.tensor(0.0, device=z_hyp.device),
                metrics={"cohesion_loss": 0.0},
                weight=self.weight,
            )

        avg_loss = total_loss / n_groups

        return LossResult(
            loss=avg_loss,
            metrics={
                "cohesion_loss": avg_loss.item(),
                "n_intra_pairs": n_groups,
            },
            weight=self.weight,
        )


# =============================================================================
# Separation Loss
# =============================================================================


class SeparationLoss(LossComponent):
    """Separation loss encouraging different-pathogen peptides to separate.

    Peptides targeting different pathogens should be distant in the
    embedding space, using a margin-based triplet approach.
    """

    def __init__(
        self,
        weight: float = 0.3,
        margin: float = 0.5,
        curvature: float = 1.0,
    ):
        """Initialize separation loss.

        Args:
            weight: Loss weight
            margin: Minimum distance margin between different pathogens
            curvature: Poincaré ball curvature
        """
        super().__init__(weight=weight, name="separation")
        self.margin = margin
        self.curvature = curvature

    def forward(
        self,
        outputs: dict[str, Tensor],
        targets: Tensor,
        pathogen_labels: Tensor | None = None,
        **kwargs,
    ) -> LossResult:
        """Compute separation loss.

        Args:
            outputs: Model outputs with 'z_hyp'
            targets: Unused
            pathogen_labels: Pathogen class labels (batch,)

        Returns:
            LossResult with margin-based inter-class distance loss
        """
        if pathogen_labels is None:
            return LossResult(
                loss=torch.tensor(0.0, device=outputs["z_hyp"].device),
                metrics={"separation_loss": 0.0},
                weight=self.weight,
            )

        z_hyp = outputs["z_hyp"]
        batch_size = z_hyp.shape[0]

        if batch_size < 2:
            return LossResult(
                loss=torch.tensor(0.0, device=z_hyp.device),
                metrics={"separation_loss": 0.0},
                weight=self.weight,
            )

        # Compute pathogen centroids
        unique_labels = pathogen_labels.unique()
        n_pathogens = len(unique_labels)

        if n_pathogens < 2:
            return LossResult(
                loss=torch.tensor(0.0, device=z_hyp.device),
                metrics={"separation_loss": 0.0},
                weight=self.weight,
            )

        centroids = []
        for label in unique_labels:
            mask = pathogen_labels == label
            centroid = z_hyp[mask].mean(dim=0)
            centroids.append(centroid)

        centroids = torch.stack(centroids)

        # Compute pairwise distances between centroids and apply margin
        total_loss = torch.tensor(0.0, device=z_hyp.device)
        n_pairs = 0

        for i in range(n_pathogens):
            for j in range(i + 1, n_pathogens):
                dist = poincare_distance(centroids[i : i + 1], centroids[j : j + 1], c=self.curvature)
                # Penalize if distance < margin
                loss = F.relu(self.margin - dist)
                total_loss = total_loss + loss
                n_pairs += 1

        if n_pairs == 0:
            return LossResult(
                loss=torch.tensor(0.0, device=z_hyp.device),
                metrics={"separation_loss": 0.0},
                weight=self.weight,
            )

        avg_loss = total_loss / n_pairs

        # Compute average inter-centroid distance for metrics
        with torch.no_grad():
            avg_dist = 0.0
            for i in range(n_pathogens):
                for j in range(i + 1, n_pathogens):
                    avg_dist += poincare_distance(centroids[i : i + 1], centroids[j : j + 1], c=self.curvature).item()
            avg_dist /= max(n_pairs, 1)

        return LossResult(
            loss=avg_loss,
            metrics={
                "separation_loss": avg_loss.item(),
                "avg_inter_dist": avg_dist,
                "n_pathogens": n_pathogens,
            },
            weight=self.weight,
        )


# =============================================================================
# Loss Manager
# =============================================================================


@dataclass
class CurriculumSchedule:
    """Curriculum learning schedule for loss weights.

    Defines when each loss component becomes active and how its weight
    changes during training.
    """

    reconstruction_start: int = 0
    reconstruction_ramp: int = 10
    mic_start: int = 10
    mic_ramp: int = 20
    property_start: int = 10
    property_ramp: int = 20
    radial_start: int = 20
    radial_ramp: int = 30
    cohesion_start: int = 30
    cohesion_ramp: int = 10
    separation_start: int = 30
    separation_ramp: int = 10


class PeptideLossManager(nn.Module):
    """Manager for computing and combining all 6 loss components.

    Handles curriculum learning where different losses are introduced
    at different training stages.
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        mic_weight: float = 2.0,
        property_weight: float = 1.0,
        radial_weight: float = 0.5,
        cohesion_weight: float = 0.3,
        separation_weight: float = 0.3,
        use_curriculum: bool = True,
        curriculum: CurriculumSchedule | None = None,
    ):
        """Initialize loss manager.

        Args:
            reconstruction_weight: Weight for reconstruction loss
            mic_weight: Weight for MIC prediction loss
            property_weight: Weight for property alignment loss
            radial_weight: Weight for radial hierarchy loss
            cohesion_weight: Weight for cohesion loss
            separation_weight: Weight for separation loss
            use_curriculum: Whether to use curriculum learning
            curriculum: Custom curriculum schedule
        """
        super().__init__()

        self.use_curriculum = use_curriculum
        self.curriculum = curriculum or CurriculumSchedule()

        # Store base weights
        self.base_weights = {
            "reconstruction": reconstruction_weight,
            "mic": mic_weight,
            "property": property_weight,
            "radial": radial_weight,
            "cohesion": cohesion_weight,
            "separation": separation_weight,
        }

        # Initialize loss components
        self.reconstruction_loss = ReconstructionLoss(weight=reconstruction_weight)
        self.mic_loss = MICPredictionLoss(weight=mic_weight)
        self.property_loss = PropertyAlignmentLoss(weight=property_weight)
        self.radial_loss = RadialHierarchyLoss(weight=radial_weight)
        self.cohesion_loss = CohesionLoss(weight=cohesion_weight)
        self.separation_loss = SeparationLoss(weight=separation_weight)

    def get_curriculum_weight(self, epoch: int, start: int, ramp: int) -> float:
        """Get curriculum weight multiplier for a loss component.

        Args:
            epoch: Current training epoch
            start: Epoch when component starts
            ramp: Number of epochs to ramp up to full weight

        Returns:
            Weight multiplier in [0, 1], minimum 0.1 when active
        """
        if epoch < start:
            return 0.0
        elif epoch >= start + ramp:
            return 1.0
        else:
            # Use (epoch - start + 1) to ensure non-zero at start epoch
            # Minimum 0.1 to ensure loss contributes gradients
            return max(0.1, (epoch - start + 1) / (ramp + 1))

    def compute_total_loss(
        self,
        outputs: dict[str, Tensor],
        target_tokens: Tensor,
        mic_targets: Tensor | None = None,
        pathogen_labels: Tensor | None = None,
        peptide_properties: Tensor | None = None,
        epoch: int = 0,
    ) -> tuple[Tensor, dict[str, float]]:
        """Compute total loss with all components.

        Args:
            outputs: Model forward outputs
            target_tokens: Target token sequences
            mic_targets: Target log10(MIC) values
            pathogen_labels: Pathogen class labels
            peptide_properties: Peptide physicochemical properties
            epoch: Current training epoch (for curriculum)

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        all_metrics = {}
        total_loss = torch.tensor(0.0, device=outputs["z_hyp"].device)

        # Get curriculum weights
        if self.use_curriculum:
            weights = {
                "reconstruction": self.get_curriculum_weight(
                    epoch, self.curriculum.reconstruction_start, self.curriculum.reconstruction_ramp
                ),
                "mic": self.get_curriculum_weight(epoch, self.curriculum.mic_start, self.curriculum.mic_ramp),
                "property": self.get_curriculum_weight(
                    epoch, self.curriculum.property_start, self.curriculum.property_ramp
                ),
                "radial": self.get_curriculum_weight(epoch, self.curriculum.radial_start, self.curriculum.radial_ramp),
                "cohesion": self.get_curriculum_weight(
                    epoch, self.curriculum.cohesion_start, self.curriculum.cohesion_ramp
                ),
                "separation": self.get_curriculum_weight(
                    epoch, self.curriculum.separation_start, self.curriculum.separation_ramp
                ),
            }
        else:
            weights = {k: 1.0 for k in self.base_weights}

        # 1. Reconstruction Loss
        if weights["reconstruction"] > 0:
            recon_result = self.reconstruction_loss(outputs, target_tokens)
            total_loss = total_loss + weights["reconstruction"] * recon_result.weighted_loss
            all_metrics.update(recon_result.to_dict("recon"))

        # 2. MIC Prediction Loss
        if weights["mic"] > 0 and mic_targets is not None:
            mic_result = self.mic_loss(outputs, target_tokens, mic_targets=mic_targets)
            total_loss = total_loss + weights["mic"] * mic_result.weighted_loss
            all_metrics.update(mic_result.to_dict("mic"))

        # 3. Property Alignment Loss
        if weights["property"] > 0 and peptide_properties is not None:
            prop_result = self.property_loss(outputs, target_tokens, peptide_properties=peptide_properties)
            total_loss = total_loss + weights["property"] * prop_result.weighted_loss
            all_metrics.update(prop_result.to_dict("property"))

        # 4. Radial Hierarchy Loss
        if weights["radial"] > 0 and mic_targets is not None:
            radial_result = self.radial_loss(outputs, target_tokens, mic_targets=mic_targets)
            total_loss = total_loss + weights["radial"] * radial_result.weighted_loss
            all_metrics.update(radial_result.to_dict("radial"))

        # 5. Cohesion Loss
        if weights["cohesion"] > 0 and pathogen_labels is not None:
            cohesion_result = self.cohesion_loss(outputs, target_tokens, pathogen_labels=pathogen_labels)
            total_loss = total_loss + weights["cohesion"] * cohesion_result.weighted_loss
            all_metrics.update(cohesion_result.to_dict("cohesion"))

        # 6. Separation Loss
        if weights["separation"] > 0 and pathogen_labels is not None:
            sep_result = self.separation_loss(outputs, target_tokens, pathogen_labels=pathogen_labels)
            total_loss = total_loss + weights["separation"] * sep_result.weighted_loss
            all_metrics.update(sep_result.to_dict("separation"))

        # Add curriculum weights to metrics
        for name, w in weights.items():
            all_metrics[f"curriculum_{name}"] = w

        all_metrics["total_loss"] = total_loss.item()

        return total_loss, all_metrics


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ReconstructionLoss",
    "MICPredictionLoss",
    "PropertyAlignmentLoss",
    "RadialHierarchyLoss",
    "CohesionLoss",
    "SeparationLoss",
    "CurriculumSchedule",
    "PeptideLossManager",
]
