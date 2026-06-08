# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Variant Escape Prediction Module.

This module implements EVEscape-inspired prediction heads for viral variant
escape mutations, particularly for HIV and other rapidly evolving pathogens.

Key Features:
- Fitness prediction: Viral replication capacity
- Immune escape prediction: Antibody/T-cell evasion
- Drug resistance prediction: Treatment escape potential
- Receptor binding: ACE2/CD4 binding affinity changes

Based on research from:
- EVEscape (Thadani et al. 2023): Predicting viral escape
- E2VD: Deep generative model for antibody escape
- DeepDTA/DeepDTAGen: Drug-target binding prediction

Usage:
    from src.diseases.variant_escape import VariantEscapeHead

    head = VariantEscapeHead(latent_dim=64, disease="hiv")
    predictions = head(latent_z)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiseaseType(Enum):
    """Supported disease types for escape prediction."""

    HIV = "hiv"
    SARS_COV2 = "sars_cov2"
    INFLUENZA = "influenza"
    GENERAL = "general"


@dataclass
class EscapePrediction:
    """Structured prediction output for variant escape."""

    # Core predictions (0-1 probabilities)
    fitness: torch.Tensor  # Viral fitness/replication capacity
    immune_escape: torch.Tensor  # Immune evasion probability
    drug_resistance: torch.Tensor  # Drug resistance level

    # Disease-specific predictions
    receptor_binding: torch.Tensor | None = None  # Receptor affinity change
    antibody_escape: torch.Tensor | None = None  # Per-antibody escape
    tcell_escape: torch.Tensor | None = None  # T-cell epitope escape

    # Uncertainty estimates
    fitness_uncertainty: torch.Tensor | None = None
    escape_uncertainty: torch.Tensor | None = None

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert to dictionary format."""
        result = {
            "fitness": self.fitness,
            "immune_escape": self.immune_escape,
            "drug_resistance": self.drug_resistance,
        }
        if self.receptor_binding is not None:
            result["receptor_binding"] = self.receptor_binding
        if self.antibody_escape is not None:
            result["antibody_escape"] = self.antibody_escape
        if self.tcell_escape is not None:
            result["tcell_escape"] = self.tcell_escape
        return result


class FitnessPredictor(nn.Module):
    """Predict viral fitness from latent representation.

    Fitness represents the replication capacity of a variant.
    Higher fitness = more successful virus.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Uncertainty estimation head
        self.uncertainty = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        z: torch.Tensor,
        return_uncertainty: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Predict fitness score.

        Args:
            z: Latent representation (batch, latent_dim)
            return_uncertainty: Whether to return uncertainty estimate

        Returns:
            Fitness score (batch, 1) and optional uncertainty
        """
        fitness = self.network(z)

        if return_uncertainty:
            uncertainty = self.uncertainty(z)
            return fitness, uncertainty

        return fitness, None


class ImmuneEscapePredictor(nn.Module):
    """Predict immune escape potential.

    Models escape from:
    - Neutralizing antibodies
    - T-cell responses
    - Innate immunity
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        n_antibody_classes: int = 10,
        n_tcell_epitopes: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Shared encoder
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Overall immune escape
        self.overall_escape = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Antibody-specific escape (per class)
        self.antibody_escape = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_antibody_classes),
            nn.Sigmoid(),
        )

        # T-cell epitope escape
        self.tcell_escape = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_tcell_epitopes),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict immune escape.

        Args:
            z: Latent representation (batch, latent_dim)

        Returns:
            Tuple of (overall_escape, antibody_escape, tcell_escape)
        """
        shared = self.shared(z)

        overall = self.overall_escape(shared)
        antibody = self.antibody_escape(shared)
        tcell = self.tcell_escape(shared)

        return overall, antibody, tcell


class DrugResistancePredictor(nn.Module):
    """Predict drug resistance mutations.

    For HIV: Resistance to RTIs, PIs, INSTIs, etc.
    For SARS-CoV-2: Resistance to antivirals like Paxlovid.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        n_drug_classes: int = 6,  # HIV: 6 drug classes
        dropout: float = 0.1,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-drug-class resistance
        self.drug_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, 1),
                    nn.Sigmoid(),
                )
                for _ in range(n_drug_classes)
            ]
        )

        # Overall resistance (aggregated)
        self.overall = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict drug resistance.

        Args:
            z: Latent representation (batch, latent_dim)

        Returns:
            Tuple of (overall_resistance, per_drug_resistance)
        """
        features = self.network(z)

        # Per-drug predictions
        per_drug = torch.cat([head(features) for head in self.drug_heads], dim=-1)

        # Overall resistance
        overall = self.overall(features)

        return overall, per_drug


class ReceptorBindingPredictor(nn.Module):
    """Predict receptor binding affinity changes.

    For HIV: CD4 and coreceptor (CCR5/CXCR4) binding
    For SARS-CoV-2: ACE2 binding affinity
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        disease: DiseaseType = DiseaseType.HIV,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.disease = disease

        self.network = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # Disease-specific outputs
        if disease == DiseaseType.HIV:
            # CD4 binding, CCR5, CXCR4
            self.binding_head = nn.Linear(hidden_dim // 2, 3)
        elif disease == DiseaseType.SARS_COV2:
            # ACE2 binding
            self.binding_head = nn.Linear(hidden_dim // 2, 1)
        else:
            # Generic receptor
            self.binding_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict receptor binding affinity.

        Args:
            z: Latent representation (batch, latent_dim)

        Returns:
            Binding affinity predictions (batch, n_receptors)
        """
        features = self.network(z)
        binding = self.binding_head(features)
        return torch.sigmoid(binding)


class VariantEscapeHead(nn.Module):
    """Complete variant escape prediction head.

    Combines all prediction components:
    - Fitness
    - Immune escape
    - Drug resistance
    - Receptor binding

    Suitable for few-shot adaptation via meta-learning.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        disease: str = "hiv",
        n_drug_classes: int = 6,
        n_antibody_classes: int = 10,
        n_tcell_epitopes: int = 20,
        dropout: float = 0.1,
    ):
        """Initialize VariantEscapeHead.

        Args:
            latent_dim: Dimension of input latent space
            hidden_dim: Hidden layer dimension
            disease: Disease type ('hiv', 'sars_cov2', 'influenza', 'general')
            n_drug_classes: Number of drug classes for resistance prediction
            n_antibody_classes: Number of antibody classes
            n_tcell_epitopes: Number of T-cell epitopes
            dropout: Dropout rate
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.disease = DiseaseType(disease.lower())

        # Component predictors
        self.fitness_predictor = FitnessPredictor(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self.immune_predictor = ImmuneEscapePredictor(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_antibody_classes=n_antibody_classes,
            n_tcell_epitopes=n_tcell_epitopes,
            dropout=dropout,
        )

        self.resistance_predictor = DrugResistancePredictor(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            n_drug_classes=n_drug_classes,
            dropout=dropout,
        )

        self.binding_predictor = ReceptorBindingPredictor(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            disease=self.disease,
            dropout=dropout,
        )

        # Composite escape score
        self.composite = nn.Sequential(
            nn.Linear(4, hidden_dim // 4),  # 4 main predictions
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
        return_components: bool = True,
        return_uncertainty: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Predict variant escape potential.

        Args:
            z: Latent representation (batch, latent_dim)
            return_components: Whether to return all component predictions
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Dictionary with all predictions
        """
        # Get individual predictions
        fitness, fitness_unc = self.fitness_predictor(z, return_uncertainty)

        immune_overall, antibody_escape, tcell_escape = self.immune_predictor(z)

        resistance_overall, per_drug = self.resistance_predictor(z)

        binding = self.binding_predictor(z)

        # Composite escape score
        composite_input = torch.cat(
            [
                fitness,
                immune_overall,
                resistance_overall,
                binding.mean(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        composite_score = self.composite(composite_input)

        # Build output dictionary
        outputs = {
            "escape_score": composite_score,
            "fitness": fitness,
            "immune_escape": immune_overall,
            "drug_resistance": resistance_overall,
        }

        if return_components:
            outputs.update(
                {
                    "antibody_escape": antibody_escape,
                    "tcell_escape": tcell_escape,
                    "per_drug_resistance": per_drug,
                    "receptor_binding": binding,
                }
            )

        if return_uncertainty and fitness_unc is not None:
            outputs["fitness_uncertainty"] = fitness_unc

        return outputs

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute escape prediction loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            weights: Optional loss weights per component

        Returns:
            Tuple of (total_loss, component_losses_dict)
        """
        if weights is None:
            weights = {
                "fitness": 1.0,
                "immune_escape": 1.0,
                "drug_resistance": 1.0,
                "binding": 0.5,
            }

        losses = {}
        total_loss = torch.tensor(0.0, device=predictions["fitness"].device)

        # Fitness loss
        if "fitness" in targets:
            fitness_loss = F.binary_cross_entropy(
                predictions["fitness"],
                targets["fitness"],
            )
            losses["fitness"] = fitness_loss.item()
            total_loss = total_loss + weights["fitness"] * fitness_loss

        # Immune escape loss
        if "immune_escape" in targets:
            immune_loss = F.binary_cross_entropy(
                predictions["immune_escape"],
                targets["immune_escape"],
            )
            losses["immune_escape"] = immune_loss.item()
            total_loss = total_loss + weights["immune_escape"] * immune_loss

        # Drug resistance loss
        if "drug_resistance" in targets:
            resistance_loss = F.binary_cross_entropy(
                predictions["drug_resistance"],
                targets["drug_resistance"],
            )
            losses["drug_resistance"] = resistance_loss.item()
            total_loss = total_loss + weights["drug_resistance"] * resistance_loss

        # Binding loss
        if "receptor_binding" in targets and "receptor_binding" in predictions:
            binding_loss = F.mse_loss(
                predictions["receptor_binding"],
                targets["receptor_binding"],
            )
            losses["binding"] = binding_loss.item()
            total_loss = total_loss + weights.get("binding", 0.5) * binding_loss

        losses["total"] = total_loss.item()
        return total_loss, losses


class MetaLearningEscapeHead(nn.Module):
    """Variant escape head optimized for meta-learning.

    Uses MAML-compatible architecture with:
    - Smaller parameter count
    - Easy gradient computation
    - Task-specific adaptation layers
    """

    def __init__(
        self,
        latent_dim: int = 64,
        hidden_dim: int = 64,
        n_tasks: int = 4,  # fitness, immune, resistance, binding
    ):
        super().__init__()

        # Shared base (frozen or slow-adapted)
        self.base = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
        )

        # Task-specific heads (fast-adapted)
        self.task_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                    nn.Sigmoid(),
                )
                for _ in range(n_tasks)
            ]
        )

        self.task_names = ["fitness", "immune_escape", "drug_resistance", "binding"]

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            z: Latent representation

        Returns:
            Dictionary of task predictions
        """
        base_features = self.base(z)

        outputs = {}
        for name, head in zip(self.task_names, self.task_heads, strict=False):
            outputs[name] = head(base_features)

        return outputs

    def get_adaptation_parameters(self) -> list[nn.Parameter]:
        """Get parameters for fast adaptation (task heads only)."""
        params = []
        for head in self.task_heads:
            params.extend(head.parameters())
        return params

    def get_base_parameters(self) -> list[nn.Parameter]:
        """Get parameters for slow adaptation (base only)."""
        return list(self.base.parameters())


__all__ = [
    "VariantEscapeHead",
    "MetaLearningEscapeHead",
    "EscapePrediction",
    "DiseaseType",
    "FitnessPredictor",
    "ImmuneEscapePredictor",
    "DrugResistancePredictor",
    "ReceptorBindingPredictor",
]
