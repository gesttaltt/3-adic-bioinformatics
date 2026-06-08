"""Uncertainty quantification for drug resistance predictions.

Point predictions are insufficient for clinical applications.
This module provides uncertainty estimates through:

1. MC Dropout: Monte Carlo sampling with dropout at inference
2. Deep Ensembles: Multiple models with different initializations
3. Evidential Learning: Direct uncertainty prediction

Clinical value: "This sequence is resistant (0.85 ± 0.12)"
vs "This sequence is resistant (0.85)"

References:
- Gal & Ghahramani (2016): Dropout as a Bayesian Approximation
- Lakshminarayanan et al. (2017): Simple and Scalable Predictive Uncertainty
- Sensoy et al. (2018): Evidential Deep Learning
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class UncertaintyEstimate:
    """Container for prediction with uncertainty."""

    mean: torch.Tensor  # Mean prediction
    std: torch.Tensor  # Standard deviation
    lower: torch.Tensor  # Lower confidence bound
    upper: torch.Tensor  # Upper confidence bound
    samples: torch.Tensor | None = None  # Optional: raw samples
    epistemic: torch.Tensor | None = None  # Model uncertainty
    aleatoric: torch.Tensor | None = None  # Data uncertainty

    @property
    def confidence_width(self) -> torch.Tensor:
        """Width of confidence interval."""
        return self.upper - self.lower

    def to_dict(self) -> dict[str, torch.Tensor]:
        """Convert to dictionary."""
        return {
            "mean": self.mean,
            "std": self.std,
            "lower": self.lower,
            "upper": self.upper,
        }


class MCDropoutWrapper(nn.Module):
    """Wrapper that enables dropout at inference time for uncertainty.

    Usage:
        model = MCDropoutWrapper(base_model, n_samples=50)
        estimate = model.predict_with_uncertainty(x)
        print(f"Prediction: {estimate.mean} ± {estimate.std}")
    """

    def __init__(
        self,
        model: nn.Module,
        n_samples: int = 50,
        confidence: float = 0.95,
    ):
        """Initialize wrapper.

        Args:
            model: Base model with dropout layers
            n_samples: Number of forward passes for uncertainty
            confidence: Confidence level for intervals (e.g., 0.95)
        """
        super().__init__()
        self.model = model
        self.n_samples = n_samples
        self.confidence = confidence
        self.z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%

    def enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Standard forward pass."""
        return self.model(x)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        return_samples: bool = False,
    ) -> UncertaintyEstimate:
        """Predict with uncertainty estimation.

        Args:
            x: Input tensor
            return_samples: Whether to return raw samples

        Returns:
            UncertaintyEstimate with mean, std, and confidence bounds
        """
        self.model.eval()
        self.enable_dropout()

        samples = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                out = self.model(x)
                # Extract prediction (handle different output formats)
                pred = out.get("prediction", out.get("z", out.get("mu"))[:, 0]) if isinstance(out, dict) else out
                samples.append(pred)

        samples = torch.stack(samples, dim=0)  # (n_samples, batch)

        mean = samples.mean(dim=0)
        std = samples.std(dim=0)
        lower = mean - self.z_score * std
        upper = mean + self.z_score * std

        return UncertaintyEstimate(
            mean=mean,
            std=std,
            lower=lower,
            upper=upper,
            samples=samples if return_samples else None,
            epistemic=std,  # For MC Dropout, epistemic = total uncertainty
        )


class DeepEnsemble(nn.Module):
    """Ensemble of models for uncertainty estimation.

    Each model is trained with different random initialization,
    providing diverse predictions that capture model uncertainty.
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        n_models: int = 5,
        confidence: float = 0.95,
    ):
        """Initialize ensemble.

        Args:
            model_fn: Function that creates a new model instance
            n_models: Number of models in ensemble
            confidence: Confidence level for intervals
        """
        super().__init__()
        self.n_models = n_models
        self.confidence = confidence
        self.z_score = 1.96 if confidence == 0.95 else 2.576

        # Create ensemble members
        self.models = nn.ModuleList([model_fn() for _ in range(n_models)])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Average prediction across ensemble."""
        predictions = []
        for model in self.models:
            out = model(x)
            pred = out.get("prediction", out.get("z", out.get("mu"))[:, 0]) if isinstance(out, dict) else out
            predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)
        mean = predictions.mean(dim=0)

        return {"prediction": mean, "predictions": predictions}

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        return_samples: bool = False,
    ) -> UncertaintyEstimate:
        """Predict with uncertainty from ensemble disagreement."""
        self.eval()

        predictions = []
        with torch.no_grad():
            for model in self.models:
                out = model(x)
                pred = out.get("prediction", out.get("z", out.get("mu"))[:, 0]) if isinstance(out, dict) else out
                predictions.append(pred)

        predictions = torch.stack(predictions, dim=0)  # (n_models, batch)

        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        lower = mean - self.z_score * std
        upper = mean + self.z_score * std

        return UncertaintyEstimate(
            mean=mean,
            std=std,
            lower=lower,
            upper=upper,
            samples=predictions if return_samples else None,
            epistemic=std,
        )

    def train_ensemble(
        self,
        train_fn: Callable[[nn.Module, int], None],
    ):
        """Train each model in ensemble.

        Args:
            train_fn: Function that trains a model, takes (model, index)
        """
        for i, model in enumerate(self.models):
            print(f"Training ensemble member {i + 1}/{self.n_models}")
            train_fn(model, i)


class EvidentialVAE(nn.Module):
    """VAE with evidential uncertainty prediction.

    Instead of point predictions, outputs parameters of a
    Normal-Inverse-Gamma distribution, providing both
    epistemic and aleatoric uncertainty.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: list[int] = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        # Encoder
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.GELU(), nn.BatchNorm1d(h), nn.Dropout(0.1)])
            in_dim = h

        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        # Decoder
        layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            layers.extend([nn.Linear(in_dim, h), nn.GELU(), nn.BatchNorm1d(h)])
            in_dim = h
        layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*layers)

        # Evidential head: outputs (gamma, nu, alpha, beta) for NIG distribution
        self.evidence_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 4),  # gamma, log_nu, log_alpha, log_beta
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Encode
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Reparameterize
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        # Decode
        x_recon = self.decoder(z)

        # Evidential parameters
        evidence = self.evidence_head(z)
        gamma = evidence[:, 0]  # Mean prediction
        nu = F.softplus(evidence[:, 1]) + 1e-6  # Precision of mean (>0)
        alpha = F.softplus(evidence[:, 2]) + 1.0  # Shape (>1 for valid variance)
        beta = F.softplus(evidence[:, 3]) + 1e-6  # Scale (>0)

        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "gamma": gamma,
            "nu": nu,
            "alpha": alpha,
            "beta": beta,
        }

    def predict_with_uncertainty(self, x: torch.Tensor) -> UncertaintyEstimate:
        """Get prediction with epistemic and aleatoric uncertainty."""
        self.eval()
        with torch.no_grad():
            out = self(x)

        gamma = out["gamma"]
        nu = out["nu"]
        alpha = out["alpha"]
        beta = out["beta"]

        # Aleatoric uncertainty (data noise)
        aleatoric = beta / (alpha - 1 + 1e-6)

        # Epistemic uncertainty (model uncertainty)
        epistemic = beta / (nu * (alpha - 1 + 1e-6))

        # Total uncertainty
        total_var = aleatoric + epistemic
        std = torch.sqrt(total_var)

        return UncertaintyEstimate(
            mean=gamma,
            std=std,
            lower=gamma - 1.96 * std,
            upper=gamma + 1.96 * std,
            epistemic=torch.sqrt(epistemic),
            aleatoric=torch.sqrt(aleatoric),
        )


def evidential_loss(
    out: dict[str, torch.Tensor],
    target: torch.Tensor,
    x: torch.Tensor,
    lambda_reg: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Loss function for evidential learning.

    Combines NIG negative log-likelihood with regularization
    to prevent overconfident predictions.
    """
    gamma = out["gamma"]
    nu = out["nu"]
    alpha = out["alpha"]
    beta = out["beta"]

    # NIG negative log-likelihood
    error = target - gamma
    omega = 2 * beta * (1 + nu)

    nll = (
        0.5 * torch.log(torch.pi / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log(nu * error**2 + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )

    # Regularization to prevent overconfidence on wrong predictions
    reg = torch.abs(error) * (2 * nu + alpha)

    # Reconstruction loss
    recon = F.mse_loss(out["x_recon"], x)

    # KL divergence
    kl = -0.5 * torch.sum(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
    kl = 0.001 * kl / x.size(0)

    return {
        "nll": nll.mean(),
        "reg": lambda_reg * reg.mean(),
        "recon": recon,
        "kl": kl,
        "total": nll.mean() + lambda_reg * reg.mean() + recon + kl,
    }


class UncertaintyCalibrator:
    """Calibrate uncertainty estimates using held-out data.

    Ensures that 95% confidence intervals actually contain
    95% of true values.
    """

    def __init__(self):
        self.calibration_factor = 1.0

    def calibrate(
        self,
        predictions: torch.Tensor,
        uncertainties: torch.Tensor,
        targets: torch.Tensor,
        target_coverage: float = 0.95,
    ):
        """Calibrate uncertainty based on observed coverage.

        Args:
            predictions: Mean predictions
            uncertainties: Predicted standard deviations
            targets: True values
            target_coverage: Desired coverage (e.g., 0.95)
        """
        # Binary search for calibration factor
        low, high = 0.1, 10.0
        z_score = 1.96  # For 95% coverage

        for _ in range(50):
            mid = (low + high) / 2
            lower = predictions - z_score * mid * uncertainties
            upper = predictions + z_score * mid * uncertainties
            coverage = ((targets >= lower) & (targets <= upper)).float().mean()

            if coverage < target_coverage:
                low = mid
            else:
                high = mid

        self.calibration_factor = mid
        print(f"Calibration factor: {self.calibration_factor:.3f}")

    def apply(self, estimate: UncertaintyEstimate) -> UncertaintyEstimate:
        """Apply calibration to uncertainty estimate."""
        calibrated_std = estimate.std * self.calibration_factor

        return UncertaintyEstimate(
            mean=estimate.mean,
            std=calibrated_std,
            lower=estimate.mean - 1.96 * calibrated_std,
            upper=estimate.mean + 1.96 * calibrated_std,
            samples=estimate.samples,
            epistemic=estimate.epistemic * self.calibration_factor if estimate.epistemic is not None else None,
            aleatoric=estimate.aleatoric,
        )


def evaluate_uncertainty(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """Evaluate quality of uncertainty estimates.

    Returns:
        Dict with calibration metrics
    """
    # Coverage at various levels
    coverages = {}
    for level in [0.5, 0.8, 0.9, 0.95, 0.99]:
        z = {0.5: 0.674, 0.8: 1.282, 0.9: 1.645, 0.95: 1.96, 0.99: 2.576}[level]
        lower = predictions - z * uncertainties
        upper = predictions + z * uncertainties
        coverage = ((targets >= lower) & (targets <= upper)).float().mean().item()
        coverages[f"coverage_{int(level * 100)}"] = coverage

    # Negative log-likelihood
    nll = 0.5 * (torch.log(2 * torch.pi * uncertainties**2) + ((targets - predictions) ** 2) / (uncertainties**2))
    nll = nll.mean().item()

    # Sharpness (average uncertainty)
    sharpness = uncertainties.mean().item()

    # Calibration error
    expected_coverage_95 = 0.95
    actual_coverage_95 = coverages["coverage_95"]
    calibration_error = abs(actual_coverage_95 - expected_coverage_95)

    return {
        **coverages,
        "nll": nll,
        "sharpness": sharpness,
        "calibration_error": calibration_error,
    }


if __name__ == "__main__":
    print("Testing Uncertainty Quantification")
    print("=" * 60)

    # Create a simple model for testing
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(100, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
            )

        def forward(self, x):
            return {"prediction": self.net(x).squeeze(-1)}

    # Test MC Dropout
    model = SimpleModel()
    mc_model = MCDropoutWrapper(model, n_samples=50)

    x = torch.randn(8, 100)
    estimate = mc_model.predict_with_uncertainty(x)

    print("MC Dropout:")
    print(f"  Mean shape: {estimate.mean.shape}")
    print(f"  Std range: [{estimate.std.min():.4f}, {estimate.std.max():.4f}]")

    # Test Ensemble
    ensemble = DeepEnsemble(lambda: SimpleModel(), n_models=5)
    estimate = ensemble.predict_with_uncertainty(x)

    print("\nDeep Ensemble:")
    print(f"  Mean shape: {estimate.mean.shape}")
    print(f"  Std range: [{estimate.std.min():.4f}, {estimate.std.max():.4f}]")

    # Test Evidential VAE
    evidential = EvidentialVAE(input_dim=100)
    out = evidential(x)
    estimate = evidential.predict_with_uncertainty(x)

    print("\nEvidential VAE:")
    print(f"  Mean shape: {estimate.mean.shape}")
    print(f"  Epistemic: {estimate.epistemic.mean():.4f}")
    print(f"  Aleatoric: {estimate.aleatoric.mean():.4f}")

    print("\n" + "=" * 60)
    print("All uncertainty methods working!")
