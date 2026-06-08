# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Uncertainty-aware disease analyzer integration.

This module integrates the existing uncertainty quantification methods
(MC Dropout, Evidential, Ensemble) into the disease prediction pipeline.

Key features:
- Confidence intervals for drug resistance predictions
- Epistemic vs aleatoric uncertainty decomposition
- Calibrated uncertainty estimates
- Clinical-grade uncertainty reporting

Clinical value:
  "This sequence is resistant to Drug X (0.85 ± 0.12, 95% CI: [0.61, 0.97])"
  vs
  "This sequence is resistant to Drug X (0.85)"

Usage:
    from src.diseases.uncertainty_aware_analyzer import UncertaintyAwareAnalyzer

    analyzer = UncertaintyAwareAnalyzer(
        base_analyzer=HIVAnalyzer(),
        uncertainty_method="evidential"
    )

    results = analyzer.analyze_with_uncertainty(sequences)
    print(f"Resistance: {results['resistance']['mean']:.2f} ± {results['resistance']['std']:.2f}")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.diseases.base import DiseaseAnalyzer
from src.models.uncertainty import (
    DeepEnsemble,
    MCDropoutWrapper,
)


@dataclass
class UncertaintyEstimate:
    """Container for uncertainty estimates from models.

    Attributes:
        mean: Mean prediction
        std: Standard deviation (total uncertainty)
        lower: Lower confidence bound
        upper: Upper confidence bound
        epistemic: Epistemic (model) uncertainty
        aleatoric: Aleatoric (data) uncertainty
    """

    mean: torch.Tensor
    std: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    epistemic: torch.Tensor | None = None
    aleatoric: torch.Tensor | None = None

    @classmethod
    def from_dict(cls, d: dict[str, torch.Tensor]) -> UncertaintyEstimate:
        """Create from dictionary of tensors."""
        return cls(
            mean=d.get("mean", d.get("prediction", torch.tensor([]))),
            std=d.get("std", d.get("uncertainty", torch.tensor([]))),
            lower=d.get("lower", d.get("mean", torch.tensor([])) - 1.96 * d.get("std", torch.tensor([]))),
            upper=d.get("upper", d.get("mean", torch.tensor([])) + 1.96 * d.get("std", torch.tensor([]))),
            epistemic=d.get("epistemic"),
            aleatoric=d.get("aleatoric"),
        )


class UncertaintyCalibrator:
    """Simple calibration for uncertainty estimates using temperature scaling."""

    def __init__(self):
        self.temperature = 1.0
        self.fitted = False

    def fit(self, predictions: torch.Tensor, uncertainties: torch.Tensor, targets: torch.Tensor):
        """Fit temperature scaling to calibrate uncertainties.

        Args:
            predictions: Predicted values
            uncertainties: Predicted uncertainties (std)
            targets: True targets
        """
        with torch.no_grad():
            errors = (predictions - targets).abs()
            # Scale temperature to match average error to average uncertainty
            self.temperature = (uncertainties.mean() / errors.mean().clamp(min=1e-8)).item()
            self.fitted = True

    def calibrate(self, uncertainties: torch.Tensor) -> torch.Tensor:
        """Apply calibration to uncertainties."""
        if not self.fitted:
            return uncertainties
        return uncertainties * self.temperature


def evaluate_uncertainty(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """Evaluate quality of uncertainty estimates.

    Args:
        predictions: Predicted values
        uncertainties: Predicted uncertainties (std)
        targets: True targets

    Returns:
        Dictionary with calibration metrics
    """
    with torch.no_grad():
        errors = (predictions - targets).abs()

        # Negative log likelihood (assume Gaussian)
        nll = 0.5 * torch.log(2 * np.pi * uncertainties**2) + (errors**2) / (2 * uncertainties**2 + 1e-8)
        nll = nll.mean().item()

        # 95% coverage
        z = 1.96
        lower = predictions - z * uncertainties
        upper = predictions + z * uncertainties
        in_interval = ((targets >= lower) & (targets <= upper)).float()
        coverage_95 = in_interval.mean().item()

        # Spearman correlation between uncertainty and error
        try:
            from scipy.stats import spearmanr

            corr, _ = spearmanr(uncertainties.cpu().numpy().flatten(), errors.cpu().numpy().flatten())
        except Exception:
            corr = 0.0

        return {
            "nll": nll,
            "coverage_95": coverage_95,
            "error_uncertainty_corr": corr,
        }


class UncertaintyMethod(Enum):
    """Available uncertainty quantification methods."""

    MC_DROPOUT = "mc_dropout"
    ENSEMBLE = "ensemble"
    EVIDENTIAL = "evidential"
    COMBINED = "combined"  # Ensemble of evidential models


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty quantification.

    Attributes:
        method: Uncertainty method to use
        n_samples: Number of MC samples (MC Dropout)
        n_models: Number of ensemble members (Ensemble)
        confidence_level: Confidence level for intervals (e.g., 0.95)
        calibrate: Whether to calibrate uncertainties
        decompose: Whether to decompose into epistemic/aleatoric
    """

    method: UncertaintyMethod = UncertaintyMethod.EVIDENTIAL
    n_samples: int = 50
    n_models: int = 5
    confidence_level: float = 0.95
    calibrate: bool = True
    decompose: bool = True


@dataclass
class UncertaintyResult:
    """Container for predictions with uncertainty.

    Attributes:
        mean: Mean prediction
        std: Standard deviation
        lower: Lower confidence bound
        upper: Upper confidence bound
        epistemic: Epistemic (model) uncertainty
        aleatoric: Aleatoric (data) uncertainty
        calibrated: Whether uncertainties are calibrated
        confidence_level: Confidence level used
    """

    mean: np.ndarray
    std: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    epistemic: np.ndarray | None = None
    aleatoric: np.ndarray | None = None
    calibrated: bool = False
    confidence_level: float = 0.95
    raw_samples: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean": self.mean.tolist() if isinstance(self.mean, np.ndarray) else self.mean,
            "std": self.std.tolist() if isinstance(self.std, np.ndarray) else self.std,
            "lower": self.lower.tolist() if isinstance(self.lower, np.ndarray) else self.lower,
            "upper": self.upper.tolist() if isinstance(self.upper, np.ndarray) else self.upper,
            "epistemic": self.epistemic.tolist() if self.epistemic is not None else None,
            "aleatoric": self.aleatoric.tolist() if self.aleatoric is not None else None,
            "calibrated": self.calibrated,
            "confidence_level": self.confidence_level,
        }


class UncertaintyWrapper(ABC):
    """Abstract base for uncertainty wrappers."""

    @abstractmethod
    def predict_with_uncertainty(self, x: torch.Tensor) -> UncertaintyEstimate:
        """Make predictions with uncertainty estimates."""
        pass


class MCDropoutUncertainty(UncertaintyWrapper):
    """MC Dropout-based uncertainty wrapper."""

    def __init__(self, model: nn.Module, n_samples: int = 50, dropout_rate: float = 0.1):
        self.wrapper = MCDropoutWrapper(model, dropout_rate=dropout_rate)
        self.n_samples = n_samples

    def predict_with_uncertainty(self, x: torch.Tensor) -> UncertaintyEstimate:
        result = self.wrapper.predict_with_uncertainty(x, n_samples=self.n_samples)
        return UncertaintyEstimate.from_dict(
            {
                "mean": result[0] if isinstance(result, tuple) else result.get("mean", x),
                "std": result[1] if isinstance(result, tuple) else result.get("std", torch.zeros_like(x)),
            }
        )


class EnsembleUncertainty(UncertaintyWrapper):
    """Ensemble-based uncertainty wrapper."""

    def __init__(self, model_fn: Callable[[], nn.Module], n_models: int = 5, confidence: float = 0.95):
        self.ensemble = DeepEnsemble(model_fn, n_models=n_models, confidence=confidence)

    def predict_with_uncertainty(self, x: torch.Tensor) -> UncertaintyEstimate:
        return self.ensemble.predict_with_uncertainty(x)


class EvidentialUncertainty(UncertaintyWrapper):
    """Evidential learning-based uncertainty wrapper."""

    def __init__(self, model: nn.Module):
        # Assumes model has predict_with_uncertainty method
        if not hasattr(model, "predict_with_uncertainty"):
            raise ValueError("Model must have predict_with_uncertainty method for evidential uncertainty")
        self.model = model

    def predict_with_uncertainty(self, x: torch.Tensor) -> UncertaintyEstimate:
        return self.model.predict_with_uncertainty(x)


class UncertaintyAwareAnalyzer:
    """Disease analyzer with integrated uncertainty quantification.

    Wraps any DiseaseAnalyzer to add uncertainty estimates to predictions.
    Supports multiple uncertainty methods and provides calibrated confidence intervals.
    """

    def __init__(
        self,
        base_analyzer: DiseaseAnalyzer,
        config: UncertaintyConfig | None = None,
        model: nn.Module | None = None,
        model_fn: Callable[[], nn.Module] | None = None,
    ):
        """Initialize uncertainty-aware analyzer.

        Args:
            base_analyzer: Base disease analyzer (e.g., HIVAnalyzer)
            config: Uncertainty configuration
            model: Pre-trained model for MC Dropout or Evidential
            model_fn: Factory function for Ensemble method
        """
        self.base_analyzer = base_analyzer
        self.config = config or UncertaintyConfig()
        self.calibrator = UncertaintyCalibrator() if self.config.calibrate else None
        self.is_calibrated = False

        # Initialize uncertainty wrapper based on method
        self.uncertainty_wrapper = self._create_uncertainty_wrapper(model, model_fn)

    def _create_uncertainty_wrapper(
        self,
        model: nn.Module | None,
        model_fn: Callable[[], nn.Module] | None,
    ) -> UncertaintyWrapper | None:
        """Create appropriate uncertainty wrapper."""
        method = self.config.method

        if method == UncertaintyMethod.MC_DROPOUT:
            if model is None:
                return None
            return MCDropoutUncertainty(model, self.config.n_samples, self.config.confidence_level)

        elif method == UncertaintyMethod.ENSEMBLE:
            if model_fn is None:
                return None
            return EnsembleUncertainty(model_fn, self.config.n_models, self.config.confidence_level)

        elif method == UncertaintyMethod.EVIDENTIAL:
            if model is None:
                return None
            return EvidentialUncertainty(model)

        return None

    def analyze(self, sequences: Any, **kwargs) -> dict[str, Any]:
        """Run base analysis without uncertainty.

        Args:
            sequences: Input sequences
            **kwargs: Additional arguments for base analyzer

        Returns:
            Analysis results from base analyzer
        """
        return self.base_analyzer.analyze(sequences, **kwargs)

    def analyze_with_uncertainty(
        self,
        sequences: Any,
        encodings: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run analysis with uncertainty quantification.

        Args:
            sequences: Input sequences (dict of gene -> sequences)
            encodings: Optional pre-computed sequence encodings
            **kwargs: Additional arguments for base analyzer

        Returns:
            Analysis results with uncertainty estimates for each drug
        """
        # Run base analysis
        base_results = self.base_analyzer.analyze(sequences, **kwargs)

        # If no uncertainty wrapper or no encodings, return base results
        if self.uncertainty_wrapper is None or encodings is None:
            return base_results

        # Get uncertainty estimates
        try:
            uncertainty_estimate = self.uncertainty_wrapper.predict_with_uncertainty(encodings)

            # Apply calibration if available
            if self.calibrator is not None and self.is_calibrated:
                uncertainty_estimate = self.calibrator.apply(uncertainty_estimate)

            # Add uncertainty to drug resistance results
            if "drug_resistance" in base_results:
                base_results = self._add_uncertainty_to_drug_results(
                    base_results,
                    uncertainty_estimate,
                )

        except Exception as e:
            # If uncertainty fails, add warning but return base results
            base_results["uncertainty_warning"] = str(e)

        return base_results

    def _add_uncertainty_to_drug_results(
        self,
        results: dict[str, Any],
        uncertainty: UncertaintyEstimate,
    ) -> dict[str, Any]:
        """Add uncertainty estimates to drug resistance results."""
        # Convert tensors to numpy
        uncertainty.mean.cpu().numpy() if torch.is_tensor(uncertainty.mean) else uncertainty.mean
        std = uncertainty.std.cpu().numpy() if torch.is_tensor(uncertainty.std) else uncertainty.std
        lower = uncertainty.lower.cpu().numpy() if torch.is_tensor(uncertainty.lower) else uncertainty.lower
        upper = uncertainty.upper.cpu().numpy() if torch.is_tensor(uncertainty.upper) else uncertainty.upper

        for _drug, data in results["drug_resistance"].items():
            # Add uncertainty metrics
            data["uncertainty"] = {
                "std": float(std.mean()) if len(std) > 1 else float(std),
                "confidence_interval": {
                    "lower": float(lower.mean()) if len(lower) > 1 else float(lower),
                    "upper": float(upper.mean()) if len(upper) > 1 else float(upper),
                    "level": self.config.confidence_level,
                },
            }

            # Add decomposition if available
            if self.config.decompose:
                if uncertainty.epistemic is not None:
                    epistemic = (
                        uncertainty.epistemic.cpu().numpy()
                        if torch.is_tensor(uncertainty.epistemic)
                        else uncertainty.epistemic
                    )
                    data["uncertainty"]["epistemic"] = (
                        float(epistemic.mean()) if len(epistemic) > 1 else float(epistemic)
                    )

                if uncertainty.aleatoric is not None:
                    aleatoric = (
                        uncertainty.aleatoric.cpu().numpy()
                        if torch.is_tensor(uncertainty.aleatoric)
                        else uncertainty.aleatoric
                    )
                    data["uncertainty"]["aleatoric"] = (
                        float(aleatoric.mean()) if len(aleatoric) > 1 else float(aleatoric)
                    )

            # Add calibration status
            data["uncertainty"]["calibrated"] = self.is_calibrated

        return results

    def calibrate(
        self,
        validation_encodings: torch.Tensor,
        validation_targets: torch.Tensor,
    ):
        """Calibrate uncertainty estimates using validation data.

        Args:
            validation_encodings: Validation set encodings
            validation_targets: True resistance values
        """
        if self.uncertainty_wrapper is None:
            raise ValueError("No uncertainty wrapper available for calibration")

        if self.calibrator is None:
            self.calibrator = UncertaintyCalibrator()

        # Get predictions with uncertainty
        estimate = self.uncertainty_wrapper.predict_with_uncertainty(validation_encodings)

        # Fit the calibrator with validation data
        self.calibrator.fit(
            predictions=estimate.mean,
            uncertainties=estimate.std,
            targets=validation_targets,
        )

        self.is_calibrated = True

    def evaluate_uncertainty_quality(
        self,
        test_encodings: torch.Tensor,
        test_targets: torch.Tensor,
    ) -> dict[str, float]:
        """Evaluate quality of uncertainty estimates.

        Args:
            test_encodings: Test set encodings
            test_targets: True resistance values

        Returns:
            Dictionary of uncertainty quality metrics
        """
        if self.uncertainty_wrapper is None:
            return {"error": "No uncertainty wrapper available"}

        estimate = self.uncertainty_wrapper.predict_with_uncertainty(test_encodings)

        if self.calibrator is not None and self.is_calibrated:
            estimate = self.calibrator.apply(estimate)

        return evaluate_uncertainty(estimate.mean, estimate.std, test_targets)

    def validate_predictions(
        self,
        predictions: dict[str, torch.Tensor],
        ground_truth: dict[str, Any],
    ) -> dict[str, float]:
        """Validate predictions with uncertainty-aware metrics.

        Extends base validation with uncertainty quality metrics.
        """
        base_metrics = self.base_analyzer.validate_predictions(predictions, ground_truth)

        # Add uncertainty-specific metrics if available
        if "uncertainty" in predictions:
            # Check if confidence intervals contain true values
            for drug, pred_data in predictions.items():
                if drug in ground_truth and "uncertainty" in pred_data:
                    true_vals = ground_truth[drug]
                    ci = pred_data["uncertainty"].get("confidence_interval", {})
                    if ci:
                        coverage = np.mean((true_vals >= ci["lower"]) & (true_vals <= ci["upper"]))
                        base_metrics[f"{drug}_coverage"] = coverage

        return base_metrics


def create_uncertainty_analyzer(
    disease_name: str,
    method: str = "evidential",
    model: nn.Module | None = None,
    **kwargs,
) -> UncertaintyAwareAnalyzer:
    """Factory function to create uncertainty-aware analyzers.

    Args:
        disease_name: Name of disease (hiv, hbv, tb, etc.)
        method: Uncertainty method (mc_dropout, ensemble, evidential)
        model: Pre-trained model
        **kwargs: Additional arguments for analyzer

    Returns:
        UncertaintyAwareAnalyzer instance
    """
    # Import disease analyzers dynamically
    from src.diseases import get_analyzer

    base_analyzer = get_analyzer(disease_name)

    config = UncertaintyConfig(
        method=UncertaintyMethod(method),
        **{k: v for k, v in kwargs.items() if hasattr(UncertaintyConfig, k)},
    )

    return UncertaintyAwareAnalyzer(
        base_analyzer=base_analyzer,
        config=config,
        model=model,
    )


__all__ = [
    "UncertaintyAwareAnalyzer",
    "UncertaintyConfig",
    "UncertaintyMethod",
    "UncertaintyResult",
    "create_uncertainty_analyzer",
]
