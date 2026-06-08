# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Evidential deep learning for uncertainty quantification.

Places Dirichlet (classification) or Normal-Inverse-Gamma (regression)
priors over predictions to estimate both aleatoric and epistemic uncertainty.

References:
    - Sensoy et al. (2018): Evidential Deep Learning
    - Amini et al. (2020): Deep Evidential Regression
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialLoss(nn.Module):
    """Loss function for evidential deep learning.

    For regression, uses Normal-Inverse-Gamma (NIG) likelihood.
    For classification, uses Dirichlet-based loss.
    """

    def __init__(
        self,
        task: str = "regression",
        lambda_coeff: float = 0.1,
    ):
        """Initialize evidential loss.

        Args:
            task: 'regression' or 'classification'
            lambda_coeff: Regularization coefficient for evidence
        """
        super().__init__()
        self.task = task
        self.lambda_coeff = lambda_coeff

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute evidential loss.

        Args:
            predictions: Model outputs (evidence parameters)
            targets: Ground truth

        Returns:
            Loss value
        """
        if self.task == "regression":
            return self._regression_loss(predictions, targets)
        else:
            return self._classification_loss(predictions, targets)

    def _regression_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """NIG-based regression loss.

        Args:
            predictions: (batch, 4) - gamma, nu, alpha, beta
            targets: (batch, 1) ground truth

        Returns:
            Loss value
        """
        # Unpack NIG parameters
        gamma = predictions[:, 0:1]  # Mean
        nu = predictions[:, 1:2]  # Precision factor
        alpha = predictions[:, 2:3]  # Shape
        beta = predictions[:, 3:4]  # Rate

        # Ensure positive parameters
        nu = F.softplus(nu) + 1e-6
        alpha = F.softplus(alpha) + 1.0
        beta = F.softplus(beta) + 1e-6

        # NIG negative log-likelihood
        omega = 2 * beta * (1 + nu)
        nll = (
            0.5 * torch.log(torch.pi / nu)
            - alpha * torch.log(omega)
            + (alpha + 0.5) * torch.log((targets - gamma) ** 2 * nu + omega)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

        # Evidence regularization (KL to uniform)
        kl_reg = self.lambda_coeff * (2 * nu + alpha)

        return (nll + kl_reg).mean()

    def _classification_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Dirichlet-based classification loss.

        Args:
            predictions: (batch, n_classes) evidence/alpha
            targets: (batch,) class indices

        Returns:
            Loss value
        """
        # Convert to Dirichlet parameters
        evidence = F.softplus(predictions)
        alpha = evidence + 1

        # Total evidence
        S = alpha.sum(dim=-1, keepdim=True)

        # One-hot targets
        n_classes = predictions.shape[-1]
        targets_onehot = F.one_hot(targets.long(), n_classes).float()

        # Expected probability
        alpha / S

        # Type II Maximum Likelihood loss
        loss = (targets_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)

        # KL regularization
        alpha_tilde = targets_onehot + (1 - targets_onehot) * alpha
        kl = self._kl_dirichlet(alpha_tilde, torch.ones_like(alpha))

        return (loss + self.lambda_coeff * kl).mean()

    def _kl_dirichlet(
        self,
        alpha: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence between Dirichlet distributions."""
        sum_alpha = alpha.sum(dim=-1)
        sum_beta = beta.sum(dim=-1)

        kl = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(sum_beta)
            - (torch.lgamma(alpha) - torch.lgamma(beta)).sum(dim=-1)
            + ((alpha - beta) * (torch.digamma(alpha) - torch.digamma(sum_alpha).unsqueeze(-1))).sum(dim=-1)
        )

        return kl


class EvidentialPredictor(nn.Module):
    """Evidential predictor with built-in uncertainty.

    For regression, outputs Normal-Inverse-Gamma parameters.
    For classification, outputs Dirichlet concentration parameters.

    Example:
        >>> predictor = EvidentialPredictor(input_dim=64, output_dim=1)
        >>> result = predictor(x)
        >>> print(result['prediction'], result['epistemic_uncertainty'])
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: list[int] = None,
        task: str = "regression",
    ):
        """Initialize evidential predictor.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            task: 'regression' or 'classification'
        """
        if hidden_dims is None:
            hidden_dims = [256, 128]
        super().__init__()
        self.output_dim = output_dim
        self.task = task

        # Build backbone
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Evidence head
        if task == "regression":
            # Output 4 parameters per target: gamma, nu, alpha, beta
            self.evidence_head = nn.Linear(prev_dim, output_dim * 4)
        else:
            # Output evidence for each class
            self.evidence_head = nn.Linear(prev_dim, output_dim)

        # Loss function
        self.loss_fn = EvidentialLoss(task=task)

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict:
        """Forward pass with uncertainty estimation.

        Args:
            x: Input features

        Returns:
            Dictionary with predictions and uncertainties
        """
        features = self.backbone(x)
        evidence = self.evidence_head(features)

        if self.task == "regression":
            return self._regression_output(evidence)
        else:
            return self._classification_output(evidence)

    def _regression_output(self, evidence: torch.Tensor) -> dict:
        """Process regression evidence to predictions + uncertainty.

        Args:
            evidence: (batch, output_dim * 4) NIG parameters

        Returns:
            Predictions and uncertainties
        """
        # Reshape to (batch, output_dim, 4)
        evidence = evidence.view(-1, self.output_dim, 4)

        gamma = evidence[:, :, 0]  # Mean prediction
        nu = F.softplus(evidence[:, :, 1]) + 1e-6
        alpha = F.softplus(evidence[:, :, 2]) + 1.0
        beta = F.softplus(evidence[:, :, 3]) + 1e-6

        # Predicted mean
        prediction = gamma

        # Aleatoric uncertainty: E[Var] = beta / (alpha - 1)
        aleatoric = beta / (alpha - 1 + 1e-8)

        # Epistemic uncertainty: Var[E] = beta / (nu * (alpha - 1))
        epistemic = beta / (nu * (alpha - 1 + 1e-8))

        # Total uncertainty
        total = aleatoric + epistemic

        # Confidence (based on evidence strength)
        confidence = nu / (nu + 1)

        return {
            "prediction": prediction,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "total_uncertainty": total,
            "confidence": confidence,
            "evidence": {
                "gamma": gamma,
                "nu": nu,
                "alpha": alpha,
                "beta": beta,
            },
        }

    def _classification_output(self, evidence: torch.Tensor) -> dict:
        """Process classification evidence to predictions + uncertainty.

        Args:
            evidence: (batch, n_classes) Dirichlet evidence

        Returns:
            Predictions and uncertainties
        """
        # Dirichlet concentration parameters
        evidence = F.softplus(evidence)
        alpha = evidence + 1

        # Total evidence (strength)
        S = alpha.sum(dim=-1, keepdim=True)

        # Predicted probabilities
        probs = alpha / S
        prediction = probs.argmax(dim=-1)

        # Epistemic uncertainty (inverse of total evidence)
        epistemic = self.output_dim / S

        # Aleatoric uncertainty (expected entropy)
        aleatoric = -(probs * (probs + 1e-8).log()).sum(dim=-1, keepdim=True)

        # Total uncertainty
        total = epistemic + aleatoric

        # Confidence
        confidence = 1 - epistemic

        return {
            "prediction": prediction,
            "probabilities": probs,
            "aleatoric_uncertainty": aleatoric.squeeze(-1),
            "epistemic_uncertainty": epistemic.squeeze(-1),
            "total_uncertainty": total.squeeze(-1),
            "confidence": confidence.squeeze(-1),
            "evidence": {
                "alpha": alpha,
                "strength": S.squeeze(-1),
            },
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute evidential loss.

        Args:
            x: Input features
            targets: Ground truth

        Returns:
            Loss value
        """
        features = self.backbone(x)
        evidence = self.evidence_head(features)

        return self.loss_fn(evidence, targets)


class EvidentialEnsemble(nn.Module):
    """Ensemble of evidential predictors.

    Combines multiple evidential models for improved
    uncertainty estimation through model disagreement.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        n_models: int = 5,
        **kwargs,
    ):
        """Initialize ensemble.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            n_models: Number of ensemble members
            **kwargs: Arguments for EvidentialPredictor
        """
        super().__init__()

        self.models = nn.ModuleList([EvidentialPredictor(input_dim, output_dim, **kwargs) for _ in range(n_models)])

    def forward(self, x: torch.Tensor) -> dict:
        """Ensemble prediction with aggregated uncertainty.

        Args:
            x: Input features

        Returns:
            Aggregated predictions and uncertainties
        """
        # Collect predictions from all models
        results = [model(x) for model in self.models]

        # Stack predictions
        predictions = torch.stack([r["prediction"] for r in results], dim=0)
        epistemic_all = torch.stack([r["epistemic_uncertainty"] for r in results], dim=0)
        aleatoric_all = torch.stack([r["aleatoric_uncertainty"] for r in results], dim=0)

        # Ensemble mean
        mean_prediction = predictions.mean(dim=0)

        # Mean uncertainties from individual models
        mean_epistemic = epistemic_all.mean(dim=0)
        mean_aleatoric = aleatoric_all.mean(dim=0)

        # Additional epistemic uncertainty from disagreement
        disagreement = predictions.var(dim=0)

        # Total epistemic = model uncertainty + disagreement
        total_epistemic = mean_epistemic + disagreement

        return {
            "prediction": mean_prediction,
            "epistemic_uncertainty": total_epistemic,
            "aleatoric_uncertainty": mean_aleatoric,
            "total_uncertainty": total_epistemic + mean_aleatoric,
            "model_disagreement": disagreement,
            "confidence": 1 / (1 + total_epistemic + mean_aleatoric),
        }
