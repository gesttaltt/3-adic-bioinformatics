# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Bayesian uncertainty quantification via Monte Carlo Dropout.

MC Dropout provides approximate Bayesian inference by performing
multiple forward passes with dropout enabled at inference time.

References:
    - Gal & Ghahramani (2016): Dropout as a Bayesian Approximation
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MCDropoutWrapper(nn.Module):
    """Wrapper that enables MC Dropout for any model.

    Wraps a model and applies dropout during inference to estimate
    epistemic uncertainty through prediction variance.

    Example:
        >>> model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))
        >>> mc_model = MCDropoutWrapper(model, dropout_rate=0.1)
        >>> pred, uncertainty = mc_model.predict_with_uncertainty(x, n_samples=100)
    """

    def __init__(
        self,
        model: nn.Module,
        dropout_rate: float = 0.1,
        dropout_positions: str = "all",
    ):
        """Initialize MC Dropout wrapper.

        Args:
            model: Base model to wrap
            dropout_rate: Dropout probability
            dropout_positions: Where to apply dropout ('all', 'last', 'first')
        """
        super().__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        self.dropout_positions = dropout_positions

        # Add dropout layers
        self._add_dropout_layers()

    def _add_dropout_layers(self):
        """Add dropout layers to the model."""
        self.dropout_layers = nn.ModuleList()

        def add_dropout_after(module):
            """Add dropout after specific layer types."""
            for _name, child in module.named_children():
                if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    self.dropout_layers.append(nn.Dropout(self.dropout_rate))
                add_dropout_after(child)

        add_dropout_after(self.model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        return self.model(x)

    def forward_with_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dropout enabled."""
        # Enable dropout
        self.train()
        output = self.model(x)
        return output

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty estimation via MC Dropout.

        Args:
            x: Input tensor
            n_samples: Number of MC samples

        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        # Collect samples
        samples = []
        for _ in range(n_samples):
            with torch.no_grad():
                sample = self.forward_with_dropout(x)
                samples.append(sample)

        samples = torch.stack(samples, dim=0)

        # Mean and variance
        mean = samples.mean(dim=0)
        variance = samples.var(dim=0)

        # Epistemic uncertainty is the prediction variance
        uncertainty = variance

        return mean, uncertainty


class BayesianPredictor(nn.Module):
    """Full Bayesian predictor with uncertainty decomposition.

    Provides both epistemic (model) and aleatoric (data) uncertainty
    estimates using MC Dropout and learned variance.

    Epistemic uncertainty can be reduced with more data.
    Aleatoric uncertainty is inherent in the data.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: list[int] = None,
        dropout_rate: float = 0.1,
        learn_variance: bool = True,
    ):
        """Initialize Bayesian predictor.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate for MC sampling
            learn_variance: Learn heteroscedastic aleatoric uncertainty
        """
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        super().__init__()
        self.output_dim = output_dim
        self.learn_variance = learn_variance
        self.dropout_rate = dropout_rate

        # Build network with dropout
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Mean head
        self.mean_head = nn.Linear(prev_dim, output_dim)

        # Variance head (for aleatoric uncertainty)
        if learn_variance:
            self.log_var_head = nn.Linear(prev_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        return_variance: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features
            return_variance: Return learned variance

        Returns:
            Predictions, optionally with variance
        """
        features = self.backbone(x)
        mean = self.mean_head(features)

        if return_variance and self.learn_variance:
            log_var = self.log_var_head(features)
            variance = torch.exp(log_var)
            return mean, variance

        return mean

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100,
    ) -> dict:
        """Predict with full uncertainty decomposition.

        Args:
            x: Input features
            n_samples: Number of MC samples

        Returns:
            Dictionary with predictions and uncertainties
        """
        # Collect MC samples
        samples = []
        variances = []

        self.train()  # Enable dropout
        for _ in range(n_samples):
            with torch.no_grad():
                if self.learn_variance:
                    sample, var = self.forward(x, return_variance=True)
                    variances.append(var)
                else:
                    sample = self.forward(x)
                samples.append(sample)

        samples = torch.stack(samples, dim=0)

        # Mean prediction
        mean_prediction = samples.mean(dim=0)

        # Epistemic uncertainty (variance of predictions)
        epistemic = samples.var(dim=0)

        # Aleatoric uncertainty (mean of learned variances)
        if self.learn_variance:
            variances = torch.stack(variances, dim=0)
            aleatoric = variances.mean(dim=0)
        else:
            aleatoric = torch.zeros_like(epistemic)

        # Total uncertainty
        total = epistemic + aleatoric

        # Confidence (inverse of total uncertainty)
        confidence = 1 / (1 + total)

        return {
            "prediction": mean_prediction,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
            "total_uncertainty": total,
            "confidence": confidence,
            "samples": samples,
        }

    def predict_with_rejection(
        self,
        x: torch.Tensor,
        confidence_threshold: float = 0.8,
        n_samples: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict with option to reject uncertain samples.

        Args:
            x: Input features
            confidence_threshold: Minimum confidence to accept
            n_samples: Number of MC samples

        Returns:
            Tuple of (predictions, confidence, accept_mask)
        """
        result = self.predict_with_uncertainty(x, n_samples)

        predictions = result["prediction"]
        confidence = result["confidence"]

        # Accept samples above threshold
        accept_mask = confidence.squeeze(-1) > confidence_threshold

        return predictions, confidence, accept_mask

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss.

        For heteroscedastic models, this accounts for learned variance.

        Args:
            x: Input features
            y: Targets

        Returns:
            Loss value
        """
        if self.learn_variance:
            mean, variance = self.forward(x, return_variance=True)

            # Negative log-likelihood of Gaussian
            nll = 0.5 * (torch.log(variance + 1e-8) + (y - mean) ** 2 / (variance + 1e-8))
            return nll.mean()
        else:
            mean = self.forward(x)
            return nn.functional.mse_loss(mean, y)


class BayesianResistancePredictor(BayesianPredictor):
    """Bayesian predictor specialized for drug resistance.

    Extends BayesianPredictor with drug-specific features
    and calibration for resistance prediction.
    """

    def __init__(
        self,
        input_dim: int,
        n_drugs: int = 18,
        hidden_dims: list[int] = None,
        dropout_rate: float = 0.15,
    ):
        """Initialize resistance predictor.

        Args:
            input_dim: Input feature dimension
            n_drugs: Number of drugs to predict
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate
        """
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        super().__init__(
            input_dim=input_dim,
            output_dim=n_drugs,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            learn_variance=True,
        )

        self.n_drugs = n_drugs

        # Drug-specific calibration
        self.drug_calibration = nn.Parameter(torch.ones(n_drugs))

    def predict_resistance(
        self,
        x: torch.Tensor,
        drug_indices: torch.Tensor | None = None,
        n_samples: int = 100,
    ) -> dict:
        """Predict resistance with calibrated uncertainty.

        Args:
            x: Sequence embeddings
            drug_indices: Which drugs to predict (optional)
            n_samples: MC samples

        Returns:
            Resistance predictions with uncertainty
        """
        result = self.predict_with_uncertainty(x, n_samples)

        # Apply calibration
        calibration = torch.sigmoid(self.drug_calibration)
        result["calibrated_confidence"] = result["confidence"] * calibration

        # Select specific drugs if requested
        if drug_indices is not None:
            for key in [
                "prediction",
                "epistemic_uncertainty",
                "aleatoric_uncertainty",
                "total_uncertainty",
                "confidence",
                "calibrated_confidence",
            ]:
                result[key] = result[key][:, drug_indices]

        return result
