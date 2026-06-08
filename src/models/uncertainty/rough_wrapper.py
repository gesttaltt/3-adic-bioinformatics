# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Rough Set-Enhanced Uncertainty Wrapper.

Combines neural uncertainty estimation with rough set theory
for three-way decision making:
- Accept: High confidence, definitely in positive region
- Reject: High confidence, definitely in negative region
- Defer: Low confidence or in boundary region

This is particularly valuable for drug resistance where:
- Some mutations definitely confer resistance
- Some mutations may or may not confer resistance
- The relationship depends on context
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.analysis.set_theory.mutation_sets import MutationSet
from src.analysis.set_theory.rough_sets import (
    RoughClassifier,
)


class Decision(Enum):
    """Three-way decision types."""

    ACCEPT = "accept"  # Definitely positive (resistant)
    REJECT = "reject"  # Definitely negative (susceptible)
    DEFER = "defer"  # Uncertain, need more information


@dataclass
class UncertaintyResult:
    """Result from uncertainty-aware prediction.

    Attributes:
        prediction: Predicted value/class
        neural_uncertainty: Uncertainty from neural model (variance/entropy)
        rough_classification: Classification from rough sets
        combined_confidence: Combined confidence score
        decision: Three-way decision
        evidence: Supporting evidence from rough sets
    """

    prediction: float | str
    neural_uncertainty: float
    rough_classification: str
    combined_confidence: float
    decision: Decision
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prediction": self.prediction,
            "neural_uncertainty": self.neural_uncertainty,
            "rough_classification": self.rough_classification,
            "combined_confidence": self.combined_confidence,
            "decision": self.decision.value,
            "evidence": self.evidence,
        }


@dataclass
class RoughWrapperConfig:
    """Configuration for rough uncertainty wrapper.

    Attributes:
        n_samples: Number of MC samples for neural uncertainty
        accept_threshold: Threshold for accept decision
        reject_threshold: Threshold for reject decision
        defer_on_boundary: Always defer for boundary region samples
        combine_method: How to combine neural and rough uncertainty
        boundary_entropy_boost: Boost entropy for boundary samples
    """

    n_samples: int = 30
    accept_threshold: float = 0.7
    reject_threshold: float = 0.3
    defer_on_boundary: bool = True
    combine_method: str = "weighted"  # 'weighted', 'min', 'max', 'product'
    boundary_entropy_boost: float = 0.3


class RoughUncertaintyWrapper(nn.Module):
    """Wrapper combining neural predictions with rough set uncertainty.

    Enhances any neural predictor with:
    1. MC Dropout uncertainty estimation
    2. Rough set membership checking
    3. Three-way decision making
    4. Calibrated confidence scores

    Example:
        >>> model = ResistancePredictor()
        >>> rough_classifiers = {"RIF": RoughClassifier.from_evidence(...)}
        >>> wrapper = RoughUncertaintyWrapper(model, rough_classifiers)
        >>> result = wrapper.predict_with_three_way_decision(x, mutations)
        >>> if result.decision == Decision.DEFER:
        ...     print("Need more testing")
    """

    def __init__(
        self,
        model: nn.Module,
        rough_classifiers: dict[str, RoughClassifier],
        config: RoughWrapperConfig | None = None,
        drug_names: list[str] | None = None,
    ):
        """Initialize rough uncertainty wrapper.

        Args:
            model: Base neural model
            rough_classifiers: Per-drug rough classifiers
            config: Configuration
            drug_names: Drug names (for multi-output models)
        """
        super().__init__()
        self.model = model
        self.rough_classifiers = rough_classifiers
        self.config = config or RoughWrapperConfig()
        self.drug_names = drug_names or list(rough_classifiers.keys())

        # Enable dropout for MC sampling
        self._enable_mc_dropout()

        # Learnable calibration parameters
        self.temperature = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def _enable_mc_dropout(self):
        """Enable dropout layers for MC sampling."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep dropout active during inference

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass.

        Args:
            x: Input tensor

        Returns:
            Model predictions
        """
        return self.model(x)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        mutations: MutationSet | None = None,
    ) -> dict[str, torch.Tensor]:
        """Get predictions with neural uncertainty.

        Args:
            x: Input tensor
            mutations: Optional mutations for rough classification

        Returns:
            Dictionary with predictions, mean, std
        """
        # MC Dropout sampling
        samples = []
        self.model.train()  # Enable dropout

        with torch.no_grad():
            for _ in range(self.config.n_samples):
                output = self.model(x)
                samples.append(output)

        samples = torch.stack(samples)  # (n_samples, batch, output_dim)

        mean = samples.mean(dim=0)
        std = samples.std(dim=0)

        # Apply calibration
        calibrated_mean = mean * self.temperature + self.bias

        return {
            "prediction": calibrated_mean,
            "mean": mean,
            "std": std,
            "samples": samples,
        }

    def predict_with_three_way_decision(
        self,
        x: torch.Tensor,
        mutations: MutationSet,
        drug: str | None = None,
    ) -> UncertaintyResult:
        """Make three-way decision combining neural and rough uncertainty.

        Args:
            x: Input tensor (single sample)
            mutations: Mutation set
            drug: Drug name (for single-drug predictions)

        Returns:
            UncertaintyResult with decision
        """
        # Get neural prediction with uncertainty
        neural_result = self.predict_with_uncertainty(x, mutations)
        prediction = neural_result["mean"]
        std = neural_result["std"]

        # Get rough classification
        if drug and drug in self.rough_classifiers:
            classifier = self.rough_classifiers[drug]
            rough_result = classifier.classify_detailed(mutations)
        else:
            rough_result = {
                "classification": "unknown",
                "confidence": 0.5,
                "evidence_strength": "none",
            }

        # Combine uncertainties
        neural_uncertainty = std.mean().item()
        rough_conf = rough_result.get("confidence", 0.5)

        combined_confidence = self._combine_confidence(neural_uncertainty, rough_conf, rough_result)

        # Make three-way decision
        decision = self._make_decision(prediction, combined_confidence, rough_result)

        # Extract evidence
        evidence = {
            "definite_mutations": rough_result.get("definite_resistance_mutations", []),
            "possible_mutations": rough_result.get("possible_resistance_mutations", []),
            "neural_samples": neural_result["samples"].shape[0],
            "neural_std": neural_uncertainty,
        }

        return UncertaintyResult(
            prediction=float(prediction.squeeze()) if prediction.numel() == 1 else prediction.tolist(),
            neural_uncertainty=neural_uncertainty,
            rough_classification=rough_result.get("classification", "unknown"),
            combined_confidence=combined_confidence,
            decision=decision,
            evidence=evidence,
        )

    def _combine_confidence(
        self,
        neural_uncertainty: float,
        rough_confidence: float,
        rough_result: dict[str, Any],
    ) -> float:
        """Combine neural and rough confidence scores.

        Args:
            neural_uncertainty: Neural model uncertainty
            rough_confidence: Rough set confidence
            rough_result: Full rough classification result

        Returns:
            Combined confidence
        """
        # Convert neural uncertainty to confidence (inverse relationship)
        neural_confidence = max(0, 1 - neural_uncertainty * 2)

        # Apply boundary penalty
        if rough_result.get("classification") == "uncertain":
            neural_confidence *= 1 - self.config.boundary_entropy_boost

        # Combine based on method
        if self.config.combine_method == "weighted":
            # Weight by evidence strength
            evidence = rough_result.get("evidence_strength", "none")
            if evidence == "strong":
                weight = 0.7
            elif evidence == "moderate":
                weight = 0.5
            else:
                weight = 0.3
            return weight * rough_confidence + (1 - weight) * neural_confidence

        elif self.config.combine_method == "min":
            return min(neural_confidence, rough_confidence)

        elif self.config.combine_method == "max":
            return max(neural_confidence, rough_confidence)

        elif self.config.combine_method == "product":
            return neural_confidence * rough_confidence

        return (neural_confidence + rough_confidence) / 2

    def _make_decision(
        self,
        prediction: torch.Tensor,
        confidence: float,
        rough_result: dict[str, Any],
    ) -> Decision:
        """Make three-way decision.

        Args:
            prediction: Neural prediction
            confidence: Combined confidence
            rough_result: Rough classification result

        Returns:
            Decision (accept/reject/defer)
        """
        rough_class = rough_result.get("classification", "unknown")

        # Always defer if in boundary and configured to do so
        if self.config.defer_on_boundary and rough_class == "uncertain":
            return Decision.DEFER

        # Get prediction probability
        pred_prob = torch.sigmoid(prediction).mean().item()

        # High confidence decisions
        if confidence >= self.config.accept_threshold:
            if pred_prob > 0.5 or rough_class == "resistant":
                return Decision.ACCEPT
            else:
                return Decision.REJECT

        # Low confidence decisions
        if confidence <= self.config.reject_threshold:
            return Decision.DEFER

        # Moderate confidence - use rough classification as tiebreaker
        if rough_class == "resistant":
            return Decision.ACCEPT
        elif rough_class == "susceptible":
            return Decision.REJECT
        else:
            return Decision.DEFER

    def batch_predict(
        self,
        x: torch.Tensor,
        mutation_sets: list[MutationSet],
        drug: str,
    ) -> list[UncertaintyResult]:
        """Predict for batch of samples.

        Args:
            x: Batch input tensor
            mutation_sets: List of mutation sets
            drug: Drug name

        Returns:
            List of uncertainty results
        """
        results = []

        for i in range(len(mutation_sets)):
            sample_x = x[i : i + 1] if x.dim() > 1 else x[i : i + 1]
            result = self.predict_with_three_way_decision(sample_x, mutation_sets[i], drug)
            results.append(result)

        return results

    def get_deferred_samples(
        self,
        results: list[UncertaintyResult],
    ) -> list[int]:
        """Get indices of samples that were deferred.

        Args:
            results: List of prediction results

        Returns:
            Indices of deferred samples
        """
        return [i for i, r in enumerate(results) if r.decision == Decision.DEFER]

    def calibrate(
        self,
        val_predictions: torch.Tensor,
        val_targets: torch.Tensor,
    ):
        """Calibrate the uncertainty estimates on validation data.

        Args:
            val_predictions: Validation predictions
            val_targets: Validation targets
        """
        # Simple temperature scaling calibration
        with torch.enable_grad():
            self.temperature.requires_grad_(True)
            self.bias.requires_grad_(True)

            optimizer = torch.optim.LBFGS([self.temperature, self.bias])

            def closure():
                optimizer.zero_grad()
                calibrated = val_predictions * self.temperature + self.bias
                loss = F.binary_cross_entropy_with_logits(calibrated, val_targets)
                loss.backward()
                return loss

            optimizer.step(closure)


class EnsembleRoughWrapper(nn.Module):
    """Ensemble of models with rough set integration.

    Uses ensemble disagreement as uncertainty measure,
    combined with rough set classification.
    """

    def __init__(
        self,
        models: list[nn.Module],
        rough_classifiers: dict[str, RoughClassifier],
        config: RoughWrapperConfig | None = None,
    ):
        """Initialize ensemble wrapper.

        Args:
            models: List of ensemble models
            rough_classifiers: Per-drug rough classifiers
            config: Configuration
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.rough_classifiers = rough_classifiers
        self.config = config or RoughWrapperConfig()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get ensemble mean prediction.

        Args:
            x: Input tensor

        Returns:
            Mean prediction
        """
        predictions = [model(x) for model in self.models]
        return torch.stack(predictions).mean(dim=0)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        mutations: MutationSet | None = None,
    ) -> dict[str, torch.Tensor]:
        """Get predictions with ensemble uncertainty.

        Args:
            x: Input tensor
            mutations: Optional mutations

        Returns:
            Predictions with uncertainty
        """
        with torch.no_grad():
            predictions = torch.stack([model(x) for model in self.models])

        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)

        return {
            "prediction": mean,
            "mean": mean,
            "std": std,
            "ensemble_predictions": predictions,
        }

    def predict_with_three_way_decision(
        self,
        x: torch.Tensor,
        mutations: MutationSet,
        drug: str,
    ) -> UncertaintyResult:
        """Make three-way decision using ensemble.

        Args:
            x: Input tensor
            mutations: Mutation set
            drug: Drug name

        Returns:
            UncertaintyResult
        """
        # Get ensemble predictions
        ensemble_result = self.predict_with_uncertainty(x, mutations)

        # Ensemble disagreement
        disagreement = ensemble_result["std"].mean().item()

        # Rough classification
        if drug in self.rough_classifiers:
            classifier = self.rough_classifiers[drug]
            rough_result = classifier.classify_detailed(mutations)
        else:
            rough_result = {"classification": "unknown", "confidence": 0.5}

        # Combine
        rough_conf = rough_result.get("confidence", 0.5)
        neural_conf = 1 - min(1, disagreement * 3)  # Scale disagreement

        combined = 0.5 * rough_conf + 0.5 * neural_conf

        # Decision
        pred = ensemble_result["mean"]
        pred_prob = torch.sigmoid(pred).mean().item()
        rough_class = rough_result.get("classification", "unknown")

        if combined >= self.config.accept_threshold:
            decision = Decision.ACCEPT if pred_prob > 0.5 or rough_class == "resistant" else Decision.REJECT
        elif rough_class == "uncertain" or combined < self.config.reject_threshold:
            decision = Decision.DEFER
        else:
            decision = Decision.ACCEPT if pred_prob > 0.5 else Decision.REJECT

        return UncertaintyResult(
            prediction=float(pred.squeeze()) if pred.numel() == 1 else pred.tolist(),
            neural_uncertainty=disagreement,
            rough_classification=rough_class,
            combined_confidence=combined,
            decision=decision,
            evidence={
                "n_models": len(self.models),
                "disagreement": disagreement,
                "rough_evidence": rough_result.get("evidence_strength", "none"),
            },
        )
