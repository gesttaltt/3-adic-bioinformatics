# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Universal uncertainty wrapper for any predictor.

Provides a unified interface for adding uncertainty estimation
to any PyTorch model, with configurable uncertainty methods.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyMethod(Enum):
    """Available uncertainty estimation methods."""

    MC_DROPOUT = "mc_dropout"
    ENSEMBLE = "ensemble"
    EVIDENTIAL = "evidential"
    TEMPERATURE_SCALING = "temperature_scaling"
    LAPLACE = "laplace"


class UncertaintyWrapper(nn.Module):
    """Wrap any predictor with uncertainty estimation.

    Provides a unified interface for adding calibrated uncertainty
    to any model, supporting multiple uncertainty methods.

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
        >>> wrapped = UncertaintyWrapper(model, method="mc_dropout")
        >>> result = wrapped.predict_with_uncertainty(x)
        >>> print(f"Prediction: {result['prediction']}")
        >>> print(f"Uncertainty: {result['uncertainty']}")
    """

    def __init__(
        self,
        model: nn.Module,
        method: str | UncertaintyMethod = "mc_dropout",
        dropout_rate: float = 0.1,
        n_samples: int = 30,
        temperature: float = 1.0,
    ):
        """Initialize uncertainty wrapper.

        Args:
            model: Base model to wrap
            method: Uncertainty estimation method
            dropout_rate: Dropout rate for MC Dropout
            n_samples: Number of samples for MC methods
            temperature: Temperature for calibration
        """
        super().__init__()
        self.model = model
        self.method = UncertaintyMethod(method) if isinstance(method, str) else method
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples
        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=False)

        # Add dropout if using MC Dropout
        if self.method == UncertaintyMethod.MC_DROPOUT:
            self._add_dropout_layers()

        # Temperature for calibration
        self.calibration_temperature = nn.Parameter(torch.ones(1), requires_grad=True)

    def _add_dropout_layers(self):
        """Insert dropout layers into the model."""
        self.dropout = nn.Dropout(p=self.dropout_rate)

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
        n_samples: int | None = None,
        return_samples: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Predict with uncertainty estimation.

        Args:
            x: Input tensor
            n_samples: Override default sample count
            return_samples: Include raw samples in output

        Returns:
            Dictionary with predictions and uncertainties
        """
        n_samples = n_samples or self.n_samples

        if self.method == UncertaintyMethod.MC_DROPOUT:
            return self._mc_dropout_predict(x, n_samples, return_samples)
        elif self.method == UncertaintyMethod.TEMPERATURE_SCALING:
            return self._temperature_scaled_predict(x)
        else:
            # Default to MC Dropout
            return self._mc_dropout_predict(x, n_samples, return_samples)

    def _mc_dropout_predict(
        self,
        x: torch.Tensor,
        n_samples: int,
        return_samples: bool = False,
    ) -> dict[str, torch.Tensor]:
        """MC Dropout-based uncertainty estimation.

        Args:
            x: Input tensor
            n_samples: Number of forward passes
            return_samples: Include raw samples

        Returns:
            Predictions with uncertainty
        """
        # Enable dropout
        was_training = self.training
        self.train()

        samples = []
        with torch.no_grad():
            for _ in range(n_samples):
                # Apply dropout before forward pass
                if hasattr(self, "dropout"):
                    x_dropped = self.dropout(x)
                    output = self.model(x_dropped)
                else:
                    output = self.model(x)
                samples.append(output)

        # Restore training state
        if not was_training:
            self.eval()

        samples = torch.stack(samples, dim=0)

        # Compute statistics
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0)

        # Standard deviation as uncertainty
        uncertainty = variance.sqrt()

        # Confidence (inverse uncertainty, normalized)
        confidence = 1 / (1 + uncertainty)

        result = {
            "prediction": mean,
            "uncertainty": uncertainty,
            "variance": variance,
            "epistemic_uncertainty": variance,
            "confidence": confidence,
        }

        if return_samples:
            result["samples"] = samples

        return result

    def _temperature_scaled_predict(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Temperature-scaled prediction with calibrated confidence.

        Args:
            x: Input tensor

        Returns:
            Calibrated predictions
        """
        with torch.no_grad():
            logits = self.model(x)

        # Apply temperature scaling
        scaled_logits = logits / self.calibration_temperature

        # For classification
        if len(scaled_logits.shape) > 1 and scaled_logits.shape[-1] > 1:
            probs = F.softmax(scaled_logits, dim=-1)
            prediction = probs.argmax(dim=-1)

            # Entropy as uncertainty
            entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1)
            max_entropy = torch.log(torch.tensor(probs.shape[-1], dtype=torch.float))
            uncertainty = entropy / max_entropy

            confidence = probs.max(dim=-1).values
        else:
            prediction = scaled_logits
            uncertainty = torch.zeros_like(prediction)
            confidence = torch.ones_like(prediction)

        return {
            "prediction": prediction,
            "uncertainty": uncertainty,
            "confidence": confidence,
            "logits": logits,
            "scaled_logits": scaled_logits,
        }

    def calibrate_temperature(
        self,
        val_loader: Any,
        device: str = "cuda",
        max_iter: int = 100,
    ) -> float:
        """Calibrate temperature on validation set.

        Args:
            val_loader: Validation data loader
            device: Computation device
            max_iter: Maximum optimization iterations

        Returns:
            Optimized temperature value
        """
        self.eval()
        self.calibration_temperature.requires_grad = True

        optimizer = torch.optim.LBFGS(
            [self.calibration_temperature],
            lr=0.01,
            max_iter=max_iter,
        )

        # Collect all logits and labels
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (tuple, list)):
                    x, y = batch[0], batch[1]
                else:
                    x, y = batch["x"], batch["y"]

                x = x.to(device)
                y = y.to(device)

                logits = self.model(x)
                all_logits.append(logits)
                all_labels.append(y)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        def closure():
            optimizer.zero_grad()
            scaled = all_logits / self.calibration_temperature
            loss = F.cross_entropy(scaled, all_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.calibration_temperature.requires_grad = False

        return self.calibration_temperature.item()

    def get_confidence_interval(
        self,
        x: torch.Tensor,
        confidence_level: float = 0.95,
        n_samples: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get prediction with confidence interval.

        Args:
            x: Input tensor
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            n_samples: Number of samples

        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        n_samples = n_samples or self.n_samples

        # Collect samples
        self.train()
        samples = []

        with torch.no_grad():
            for _ in range(n_samples):
                if hasattr(self, "dropout"):
                    x_dropped = self.dropout(x)
                    output = self.model(x_dropped)
                else:
                    output = self.model(x)
                samples.append(output)

        samples = torch.stack(samples, dim=0)

        # Compute percentiles
        alpha = (1 - confidence_level) / 2
        lower_idx = int(alpha * n_samples)
        upper_idx = int((1 - alpha) * n_samples)

        sorted_samples, _ = samples.sort(dim=0)
        lower = sorted_samples[lower_idx]
        upper = sorted_samples[upper_idx]
        mean = samples.mean(dim=0)

        return mean, lower, upper

    def should_reject(
        self,
        x: torch.Tensor,
        threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine which predictions should be rejected due to low confidence.

        Args:
            x: Input tensor
            threshold: Minimum confidence to accept

        Returns:
            Tuple of (predictions, reject_mask)
        """
        result = self.predict_with_uncertainty(x)

        predictions = result["prediction"]
        confidence = result["confidence"]

        # Reject if confidence below threshold
        reject_mask = confidence.squeeze(-1) < threshold

        return predictions, reject_mask


class CalibratedWrapper(UncertaintyWrapper):
    """Uncertainty wrapper with isotonic calibration.

    Applies isotonic regression for probability calibration
    in addition to uncertainty estimation.
    """

    def __init__(
        self,
        model: nn.Module,
        n_bins: int = 15,
        **kwargs,
    ):
        """Initialize calibrated wrapper.

        Args:
            model: Base model
            n_bins: Number of calibration bins
            **kwargs: Arguments for UncertaintyWrapper
        """
        super().__init__(model, **kwargs)
        self.n_bins = n_bins

        # Calibration mapping (learned during calibration)
        self.register_buffer(
            "calibration_map",
            torch.linspace(0, 1, n_bins + 1),
        )
        self.register_buffer(
            "calibrated_probs",
            torch.linspace(0, 1, n_bins + 1),
        )

    def calibrate(
        self,
        val_loader: Any,
        device: str = "cuda",
    ):
        """Calibrate using histogram binning.

        Args:
            val_loader: Validation data loader
            device: Computation device
        """
        self.eval()

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (tuple, list)):
                    x, y = batch[0], batch[1]
                else:
                    x, y = batch["x"], batch["y"]

                x = x.to(device)
                y = y.to(device)

                output = self.model(x)

                # Get probabilities
                if len(output.shape) > 1 and output.shape[-1] > 1:
                    probs = F.softmax(output, dim=-1)
                    max_probs = probs.max(dim=-1).values
                else:
                    max_probs = torch.sigmoid(output).squeeze(-1)

                all_probs.append(max_probs)
                all_labels.append(y)

        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Histogram binning calibration
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1, device=device)
        calibrated = torch.zeros(self.n_bins + 1, device=device)

        for i in range(self.n_bins):
            mask = (all_probs >= bin_boundaries[i]) & (all_probs < bin_boundaries[i + 1])
            if mask.sum() > 0:
                calibrated[i] = all_labels[mask].float().mean()
            else:
                calibrated[i] = bin_boundaries[i]

        calibrated[-1] = 1.0

        self.calibration_map = bin_boundaries
        self.calibrated_probs = calibrated

    def apply_calibration(self, probs: torch.Tensor) -> torch.Tensor:
        """Apply learned calibration mapping.

        Args:
            probs: Uncalibrated probabilities

        Returns:
            Calibrated probabilities
        """
        # Find bin for each probability
        bin_indices = torch.bucketize(probs, self.calibration_map) - 1
        bin_indices = bin_indices.clamp(0, self.n_bins - 1)

        # Interpolate calibrated probability
        lower = self.calibrated_probs[bin_indices]
        upper = self.calibrated_probs[bin_indices + 1]
        lower_bound = self.calibration_map[bin_indices]
        upper_bound = self.calibration_map[bin_indices + 1]

        # Linear interpolation
        fraction = (probs - lower_bound) / (upper_bound - lower_bound + 1e-8)
        calibrated = lower + fraction * (upper - lower)

        return calibrated


class MultiOutputWrapper(nn.Module):
    """Wrapper for multi-output models with per-output uncertainty.

    Provides independent uncertainty estimates for each output dimension.
    """

    def __init__(
        self,
        model: nn.Module,
        output_dim: int,
        method: str = "mc_dropout",
        **kwargs,
    ):
        """Initialize multi-output wrapper.

        Args:
            model: Base model
            output_dim: Number of output dimensions
            method: Uncertainty method
            **kwargs: Arguments for UncertaintyWrapper
        """
        super().__init__()
        self.model = model
        self.output_dim = output_dim

        # Per-output calibration temperatures
        self.output_temperatures = nn.Parameter(torch.ones(output_dim))

        # Wrap with uncertainty
        self.wrapper = UncertaintyWrapper(model, method=method, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.model(x)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        output_indices: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict with per-output uncertainty.

        Args:
            x: Input tensor
            output_indices: Specific outputs to return (None = all)

        Returns:
            Dictionary with predictions and per-output uncertainties
        """
        result = self.wrapper.predict_with_uncertainty(x, return_samples=True)

        # Apply per-output calibration
        prediction = result["prediction"]
        uncertainty = result["uncertainty"]

        # Scale by learned temperatures
        calibrated_uncertainty = uncertainty * self.output_temperatures.abs()

        # Select specific outputs if requested
        if output_indices is not None:
            prediction = prediction[..., output_indices]
            uncertainty = uncertainty[..., output_indices]
            calibrated_uncertainty = calibrated_uncertainty[..., output_indices]

        return {
            "prediction": prediction,
            "uncertainty": uncertainty,
            "calibrated_uncertainty": calibrated_uncertainty,
            "confidence": 1 / (1 + calibrated_uncertainty),
            "per_output_temperature": self.output_temperatures,
        }
