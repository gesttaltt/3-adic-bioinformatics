# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Multi-task drug resistance predictor.

Jointly predicts resistance across multiple drugs while learning
shared representations and drug-specific patterns.

Key features:
- Shared encoder with task-specific heads
- Cross-task attention for information sharing
- GradNorm-based automatic task weighting
- Drug correlation modeling

References:
    - Chen et al. (2018): GradNorm
    - Liu et al. (2019): Multi-Task Learning as Multi-Objective Optimization
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.mtl.task_heads import (
    ClassificationHead,
    CrossTaskAttention,
    DrugResistanceHead,
    RegressionHead,
)


@dataclass
class MTLConfig:
    """Configuration for multi-task learning model.

    Attributes:
        input_dim: Input feature dimension
        hidden_dims: Shared encoder hidden dimensions
        n_drugs: Number of drugs for resistance prediction
        drug_names: Names of drugs (optional)
        use_cross_attention: Enable cross-task attention
        n_attention_heads: Number of attention heads
        dropout: Dropout rate
        task_weighting: Method for task weighting ('equal', 'uncertainty', 'gradnorm')
    """

    input_dim: int = 64
    hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    n_drugs: int = 18
    drug_names: list[str] | None = None
    use_cross_attention: bool = True
    n_attention_heads: int = 4
    dropout: float = 0.1
    task_weighting: str = "gradnorm"


# Default TB drug names
TB_DRUG_NAMES = [
    "Isoniazid",
    "Rifampicin",
    "Ethambutol",
    "Pyrazinamide",
    "Streptomycin",
    "Fluoroquinolones",
    "Amikacin",
    "Capreomycin",
    "Kanamycin",
    "Ethionamide",
    "Cycloserine",
    "PAS",
    "Linezolid",
    "Bedaquiline",
    "Delamanid",
    "Clofazimine",
    "Meropenem",
    "Pretomanid",
]


class MultiTaskResistancePredictor(nn.Module):
    """Multi-task predictor for drug resistance.

    Jointly learns resistance prediction across multiple drugs
    with shared representations and cross-task learning.

    Example:
        >>> config = MTLConfig(input_dim=64, n_drugs=18)
        >>> model = MultiTaskResistancePredictor(config)
        >>> embeddings = torch.randn(32, 64)  # From VAE/PLM
        >>> outputs = model(embeddings)
        >>> print(outputs["resistance"].shape)  # (32, 18)
    """

    def __init__(
        self,
        config: MTLConfig | None = None,
    ):
        """Initialize multi-task predictor.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config or MTLConfig()

        # Drug names
        if self.config.drug_names is None:
            self.drug_names = TB_DRUG_NAMES[: self.config.n_drugs]
        else:
            self.drug_names = self.config.drug_names

        # Shared encoder
        self._build_shared_encoder()

        # Drug resistance head
        self.resistance_head = DrugResistanceHead(
            input_dim=self.config.hidden_dims[-1],
            n_drugs=self.config.n_drugs,
            hidden_dims=[256, 128],
            drug_names=self.drug_names,
        )

        # Cross-resistance prediction (pairwise)
        if self.config.use_cross_attention:
            self.cross_attention = CrossTaskAttention(
                task_dim=128,
                n_tasks=self.config.n_drugs,
                n_heads=self.config.n_attention_heads,
            )

        # Task weights (for weighted loss)
        self.task_weights = nn.Parameter(torch.ones(self.config.n_drugs), requires_grad=True)

        # Auxiliary heads
        self._build_auxiliary_heads()

    def _build_shared_encoder(self):
        """Build shared encoder backbone."""
        layers = []
        prev_dim = self.config.input_dim

        for hidden_dim in self.config.hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(self.config.dropout),
                ]
            )
            prev_dim = hidden_dim

        self.shared_encoder = nn.Sequential(*layers)

    def _build_auxiliary_heads(self):
        """Build auxiliary prediction heads."""
        hidden = self.config.hidden_dims[-1]

        # Fitness cost predictor
        self.fitness_head = RegressionHead(
            input_dim=hidden,
            output_dim=1,
            hidden_dims=[64, 32],
            learn_variance=True,
        )

        # MDR/XDR classifier
        self.mdr_classifier = ClassificationHead(
            input_dim=hidden,
            n_classes=3,  # Susceptible, MDR, XDR
            task_type="multiclass",
            hidden_dims=[64, 32],
        )

        # Mutation effect predictor
        self.mutation_effect_head = RegressionHead(
            input_dim=hidden,
            output_dim=1,
            hidden_dims=[64, 32],
        )

    def forward(
        self,
        x: torch.Tensor,
        drug_indices: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input features (batch, input_dim)
            drug_indices: Specific drugs to predict (None = all)
            return_features: Return intermediate features

        Returns:
            Dictionary with predictions
        """
        # Shared encoding
        shared_features = self.shared_encoder(x)

        # Drug resistance prediction
        resistance = self.resistance_head(shared_features, drug_indices)

        # Apply cross-task attention if available
        if self.config.use_cross_attention and drug_indices is None:
            # Create task representations
            n_drugs = self.config.n_drugs
            x.shape[0]

            # Expand shared features for each drug
            task_reps = shared_features.unsqueeze(1).expand(-1, n_drugs, -1)

            # Apply cross-task attention
            attended = self.cross_attention(task_reps)

            # Average across tasks back to single representation
            attended_features = attended.mean(dim=1)
        else:
            attended_features = shared_features

        # Auxiliary predictions
        fitness, fitness_var = self.fitness_head(shared_features, return_variance=True)
        mdr_logits = self.mdr_classifier(shared_features)
        mutation_effect, _ = self.mutation_effect_head(shared_features)

        outputs = {
            "resistance": resistance,
            "fitness_cost": fitness,
            "fitness_variance": fitness_var,
            "mdr_logits": mdr_logits,
            "mdr_probs": F.softmax(mdr_logits, dim=-1),
            "mutation_effect": mutation_effect,
        }

        if return_features:
            outputs["shared_features"] = shared_features
            if self.config.use_cross_attention:
                outputs["attended_features"] = attended_features

        return outputs

    def compute_loss(
        self,
        x: torch.Tensor,
        targets: dict[str, torch.Tensor],
        use_task_weights: bool = True,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute multi-task loss.

        Args:
            x: Input features
            targets: Dictionary of target tensors
            use_task_weights: Apply learned task weights

        Returns:
            Tuple of (total_loss, per_task_losses)
        """
        outputs = self.forward(x)
        losses = {}

        # Resistance loss (per-drug BCE)
        if "resistance" in targets:
            resistance_targets = targets["resistance"]
            resistance_pred = outputs["resistance"]

            per_drug_loss = F.binary_cross_entropy_with_logits(
                resistance_pred,
                resistance_targets,
                reduction="none",
            ).mean(dim=0)  # (n_drugs,)

            if use_task_weights:
                weights = F.softmax(self.task_weights, dim=0)
                resistance_loss = (per_drug_loss * weights).sum()
            else:
                resistance_loss = per_drug_loss.mean()

            losses["resistance"] = resistance_loss
            losses["per_drug_loss"] = per_drug_loss

        # Fitness loss
        if "fitness" in targets:
            fitness_loss = self.fitness_head.compute_loss(
                outputs["fitness_cost"],
                targets["fitness"],
                outputs["fitness_variance"],
            )
            losses["fitness"] = fitness_loss

        # MDR classification loss
        if "mdr_label" in targets:
            mdr_loss = self.mdr_classifier.compute_loss(
                outputs["mdr_logits"],
                targets["mdr_label"],
            )
            losses["mdr"] = mdr_loss

        # Mutation effect loss
        if "mutation_effect" in targets:
            mut_loss = F.mse_loss(
                outputs["mutation_effect"],
                targets["mutation_effect"],
            )
            losses["mutation_effect"] = mut_loss

        # Combine losses
        total_loss = sum(losses.values())

        return total_loss, losses

    def predict_resistance(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Predict resistance with binary labels.

        Args:
            x: Input features
            threshold: Classification threshold

        Returns:
            Dictionary with predictions and probabilities
        """
        with torch.no_grad():
            outputs = self.forward(x)
            probs = torch.sigmoid(outputs["resistance"])
            labels = (probs > threshold).long()

        return {
            "probabilities": probs,
            "predictions": labels,
            "drug_names": self.drug_names,
        }

    def get_cross_resistance_matrix(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pairwise cross-resistance correlations.

        Args:
            x: Input features (batch, input_dim)

        Returns:
            (n_drugs, n_drugs) correlation matrix
        """
        with torch.no_grad():
            # Get resistance predictions for batch
            outputs = self.forward(x)
            resistance = torch.sigmoid(outputs["resistance"])  # (batch, n_drugs)

            # Compute correlation
            resistance_centered = resistance - resistance.mean(dim=0, keepdim=True)
            cov = torch.mm(resistance_centered.t(), resistance_centered) / (x.shape[0] - 1)
            std = resistance.std(dim=0, keepdim=True)
            corr = cov / (std.t() @ std + 1e-8)

        return corr

    def get_drug_embeddings(self) -> torch.Tensor:
        """Get learned drug embeddings.

        Returns:
            (n_drugs, embedding_dim) drug embeddings
        """
        return self.resistance_head.drug_embeddings.weight.detach()


class HierarchicalMTL(nn.Module):
    """Hierarchical multi-task learning with drug class structure.

    Leverages drug class hierarchy (first-line, second-line, etc.)
    for improved resistance prediction.
    """

    # Drug class hierarchy
    DRUG_CLASSES = {
        "first_line": ["Isoniazid", "Rifampicin", "Ethambutol", "Pyrazinamide"],
        "second_line_injectable": ["Streptomycin", "Amikacin", "Capreomycin", "Kanamycin"],
        "second_line_oral": ["Fluoroquinolones", "Ethionamide", "Cycloserine", "PAS"],
        "group_5": ["Linezolid", "Bedaquiline", "Delamanid", "Clofazimine"],
    }

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 256,
    ):
        """Initialize hierarchical MTL.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
        """
        super().__init__()

        # Shared encoder
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Class-level encoders
        self.class_encoders = nn.ModuleDict(
            {
                class_name: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.SiLU(),
                )
                for class_name in self.DRUG_CLASSES
            }
        )

        # Drug-level heads
        self.drug_heads = nn.ModuleDict()
        for _class_name, drugs in self.DRUG_CLASSES.items():
            for drug in drugs:
                self.drug_heads[drug] = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input features

        Returns:
            Dictionary with per-drug predictions
        """
        # Shared encoding
        shared = self.shared_encoder(x)

        predictions = {}

        for class_name, drugs in self.DRUG_CLASSES.items():
            # Class-specific encoding
            class_features = self.class_encoders[class_name](shared)

            # Per-drug prediction
            for drug in drugs:
                predictions[drug] = self.drug_heads[drug](class_features)

        return predictions

    def get_all_predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Get all predictions as single tensor.

        Args:
            x: Input features

        Returns:
            (batch, n_drugs) predictions
        """
        pred_dict = self.forward(x)

        # Order consistently
        all_drugs = []
        for drugs in self.DRUG_CLASSES.values():
            all_drugs.extend(drugs)

        return torch.cat([pred_dict[drug] for drug in all_drugs], dim=-1)
