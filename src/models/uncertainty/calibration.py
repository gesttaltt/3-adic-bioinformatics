# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Calibration methods for uncertainty quantification.

Implements:
- Temperature Scaling: Simple post-hoc calibration
- Platt Scaling: Logistic regression calibration
- Isotonic Regression: Non-parametric calibration
- Vector Scaling: Per-class temperature scaling
- Focal Loss Calibration: Training-time calibration

References:
    - Guo et al., "On Calibration of Modern Neural Networks" (2017)
    - Platt, "Probabilistic Outputs for SVMs" (1999)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import LBFGS, Adam


@dataclass
class CalibrationMetrics:
    """Metrics for evaluating calibration quality."""

    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    brier: float  # Brier Score
    nll: float  # Negative Log-Likelihood
    reliability_diagram: dict[str, np.ndarray] = field(default_factory=dict)


class BaseCalibrator(ABC, nn.Module):
    """Abstract base class for calibration methods."""

    def __init__(self):
        super().__init__()
        self._fitted = False

    @abstractmethod
    def fit(
        self,
        logits: Tensor,
        labels: Tensor,
        **kwargs: Any,
    ) -> BaseCalibrator:
        """Fit calibrator to validation data."""
        pass

    @abstractmethod
    def calibrate(self, logits: Tensor) -> Tensor:
        """Apply calibration to logits."""
        pass

    def forward(self, logits: Tensor) -> Tensor:
        """Forward pass applies calibration."""
        return self.calibrate(logits)

    @property
    def is_fitted(self) -> bool:
        """Check if calibrator has been fitted."""
        return self._fitted


class TemperatureScaling(BaseCalibrator):
    """Temperature Scaling for neural network calibration.

    Learns a single scalar temperature T to scale logits: softmax(z/T).
    Simple yet effective post-hoc calibration method.

    Attributes:
        temperature: Learned temperature parameter (initialized to 1.0)

    Example:
        >>> calibrator = TemperatureScaling()
        >>> calibrator.fit(val_logits, val_labels)
        >>> calibrated_probs = calibrator.calibrate(test_logits)
    """

    def __init__(self, init_temperature: float = 1.0):
        """Initialize temperature scaling.

        Args:
            init_temperature: Initial temperature value (default 1.0)
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([init_temperature]))

    def fit(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
        verbose: bool = False,
    ) -> TemperatureScaling:
        """Fit temperature parameter using NLL loss.

        Args:
            logits: Model logits (N, C) or (N,) for binary
            labels: Ground truth labels (N,)
            lr: Learning rate for optimization
            max_iter: Maximum iterations for LBFGS
            verbose: Print optimization progress

        Returns:
            Self for chaining
        """
        # Ensure logits are 2D
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)
            logits = torch.cat([torch.zeros_like(logits), logits], dim=1)

        # Use LBFGS for efficient optimization
        optimizer = LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = F.cross_entropy(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

        # Clamp temperature to reasonable range
        with torch.no_grad():
            self.temperature.clamp_(min=0.01, max=10.0)

        self._fitted = True

        if verbose:
            print(f"Fitted temperature: {self.temperature.item():.4f}")

        return self

    def calibrate(self, logits: Tensor) -> Tensor:
        """Apply temperature scaling to logits.

        Args:
            logits: Model logits (N, C) or (N,)

        Returns:
            Calibrated probabilities
        """
        if not self._fitted:
            # Return softmax without scaling if not fitted
            return F.softmax(logits, dim=-1)

        scaled_logits = logits / self.temperature
        return F.softmax(scaled_logits, dim=-1)

    def get_temperature(self) -> float:
        """Get current temperature value."""
        return self.temperature.item()


class VectorScaling(BaseCalibrator):
    """Vector Scaling: Per-class temperature and bias.

    Learns W (diagonal matrix) and b (bias vector) to transform logits.
    More expressive than temperature scaling but may overfit on small data.
    """

    def __init__(self, num_classes: int):
        """Initialize vector scaling.

        Args:
            num_classes: Number of output classes
        """
        super().__init__()
        self.num_classes = num_classes
        self.W = nn.Parameter(torch.ones(num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))

    def fit(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.001,
        max_epochs: int = 100,
        patience: int = 10,
        verbose: bool = False,
    ) -> VectorScaling:
        """Fit vector scaling parameters.

        Args:
            logits: Model logits (N, C)
            labels: Ground truth labels (N,)
            lr: Learning rate
            max_epochs: Maximum training epochs
            patience: Early stopping patience
            verbose: Print progress

        Returns:
            Self for chaining
        """
        optimizer = Adam([self.W, self.b], lr=lr)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(max_epochs):
            optimizer.zero_grad()
            scaled = logits * self.W + self.b
            loss = F.cross_entropy(scaled, labels)
            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            if verbose and epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

        self._fitted = True
        return self

    def calibrate(self, logits: Tensor) -> Tensor:
        """Apply vector scaling to logits."""
        if not self._fitted:
            return F.softmax(logits, dim=-1)
        scaled = logits * self.W + self.b
        return F.softmax(scaled, dim=-1)


class PlattScaling(BaseCalibrator):
    """Platt Scaling for binary classification.

    Fits a logistic regression on top of model scores.
    """

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.tensor([1.0]))
        self.b = nn.Parameter(torch.tensor([0.0]))

    def fit(
        self,
        logits: Tensor,
        labels: Tensor,
        lr: float = 0.01,
        max_iter: int = 100,
        verbose: bool = False,
    ) -> PlattScaling:
        """Fit Platt scaling parameters."""
        # For binary classification, use single column
        if logits.dim() == 2:
            scores = logits[:, 1] - logits[:, 0] if logits.shape[1] == 2 else logits[:, 0]
        else:
            scores = logits

        optimizer = LBFGS([self.a, self.b], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = self.a * scores + self.b
            loss = F.binary_cross_entropy_with_logits(scaled, labels.float())
            loss.backward()
            return loss

        optimizer.step(closure)
        self._fitted = True

        if verbose:
            print(f"Platt params: a={self.a.item():.4f}, b={self.b.item():.4f}")

        return self

    def calibrate(self, logits: Tensor) -> Tensor:
        """Apply Platt scaling."""
        if logits.dim() == 2:
            scores = logits[:, 1] - logits[:, 0] if logits.shape[1] == 2 else logits[:, 0]
        else:
            scores = logits

        if not self._fitted:
            return torch.sigmoid(scores)

        scaled = self.a * scores + self.b
        probs = torch.sigmoid(scaled)

        # Return two-class probabilities
        return torch.stack([1 - probs, probs], dim=-1)


class IsotonicCalibration(BaseCalibrator):
    """Isotonic Regression Calibration.

    Non-parametric calibration using sklearn's isotonic regression.
    Requires sklearn for fitting.
    """

    def __init__(self):
        super().__init__()
        self._isotonic_models: list[Any] = []

    def fit(
        self,
        logits: Tensor,
        labels: Tensor,
        verbose: bool = False,
    ) -> IsotonicCalibration:
        """Fit isotonic regression for each class."""
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            raise ImportError("sklearn required for IsotonicCalibration")

        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        num_classes = probs.shape[1]

        self._isotonic_models = []
        for c in range(num_classes):
            ir = IsotonicRegression(out_of_bounds="clip")
            binary_labels = (labels_np == c).astype(float)
            ir.fit(probs[:, c], binary_labels)
            self._isotonic_models.append(ir)

        self._fitted = True
        return self

    def calibrate(self, logits: Tensor) -> Tensor:
        """Apply isotonic calibration."""
        if not self._fitted:
            return F.softmax(logits, dim=-1)

        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        calibrated = np.zeros_like(probs)

        for c, ir in enumerate(self._isotonic_models):
            calibrated[:, c] = ir.predict(probs[:, c])

        # Renormalize
        calibrated = calibrated / calibrated.sum(axis=1, keepdims=True)

        return torch.from_numpy(calibrated).to(logits.device).float()


class FocalLossCalibration(nn.Module):
    """Focal Loss for training-time calibration.

    Reduces contribution of well-classified examples, focusing on hard cases.
    Improves calibration during training.

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(self, gamma: float = 2.0, alpha: Tensor | None = None):
        """Initialize focal loss.

        Args:
            gamma: Focusing parameter (higher = more focus on hard examples)
            alpha: Class weights (optional)
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            logits: Model logits (N, C)
            labels: Ground truth labels (N,)

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[labels]
            focal_loss = alpha_t * focal_loss

        return focal_loss.mean()


class LabelSmoothingLoss(nn.Module):
    """Label Smoothing Loss for improved calibration.

    Softens hard labels to prevent overconfidence during training.

    Args:
        smoothing: Label smoothing factor (0 = no smoothing, 1 = uniform)
        reduction: Loss reduction method ('mean', 'sum', 'none')
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        assert 0 <= smoothing < 1
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute label smoothing loss.

        Args:
            logits: Model logits (N, C)
            labels: Ground truth labels (N,)

        Returns:
            Smoothed cross-entropy loss
        """
        num_classes = logits.shape[-1]
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (num_classes - 1)

        # Create smoothed labels
        with torch.no_grad():
            smooth_labels = torch.full_like(logits, smooth_value)
            smooth_labels.scatter_(1, labels.unsqueeze(1), confidence)

        # KL divergence (equivalent to cross-entropy with soft labels)
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.sum(smooth_labels * log_probs, dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_calibration_metrics(
    probs: Tensor,
    labels: Tensor,
    n_bins: int = 15,
) -> CalibrationMetrics:
    """Compute calibration metrics.

    Args:
        probs: Predicted probabilities (N, C) or (N,)
        labels: Ground truth labels (N,)
        n_bins: Number of bins for ECE/MCE

    Returns:
        CalibrationMetrics dataclass
    """
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    # For multi-class, use predicted class probability
    if probs_np.ndim == 2:
        confidences = probs_np.max(axis=1)
        predictions = probs_np.argmax(axis=1)
    else:
        confidences = probs_np
        predictions = (probs_np >= 0.5).astype(int)

    accuracies = predictions == labels_np

    # Compute ECE and MCE
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            acc_in_bin = accuracies[in_bin].mean()
            conf_in_bin = confidences[in_bin].mean()
            gap = np.abs(acc_in_bin - conf_in_bin)

            ece += prop_in_bin * gap
            mce = max(mce, gap)

            bin_accs.append(acc_in_bin)
            bin_confs.append(conf_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)

    # Brier score
    if probs_np.ndim == 2:
        one_hot = np.zeros_like(probs_np)
        one_hot[np.arange(len(labels_np)), labels_np] = 1
        brier = ((probs_np - one_hot) ** 2).sum(axis=1).mean()
    else:
        brier = ((probs_np - labels_np) ** 2).mean()

    # NLL
    if probs_np.ndim == 2:
        nll = -np.log(probs_np[np.arange(len(labels_np)), labels_np] + 1e-10).mean()
    else:
        eps = 1e-10
        nll = -(labels_np * np.log(probs_np + eps) + (1 - labels_np) * np.log(1 - probs_np + eps)).mean()

    return CalibrationMetrics(
        ece=float(ece),
        mce=float(mce),
        brier=float(brier),
        nll=float(nll),
        reliability_diagram={
            "accuracies": np.array(bin_accs),
            "confidences": np.array(bin_confs),
            "counts": np.array(bin_counts),
            "bin_edges": bin_boundaries,
        },
    )


def auto_calibrate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    method: str = "temperature",
    device: str = "cuda",
    verbose: bool = True,
) -> tuple[BaseCalibrator, CalibrationMetrics]:
    """Automatically calibrate a model on validation data.

    Args:
        model: Neural network model
        val_loader: Validation data loader
        method: Calibration method ('temperature', 'vector', 'platt', 'isotonic')
        device: Device for computation
        verbose: Print calibration results

    Returns:
        Tuple of (fitted calibrator, calibration metrics)
    """
    model.eval()

    # Collect predictions
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0], batch[1]
            else:
                inputs = batch
                labels = None

            inputs = inputs.to(device)
            outputs = model(inputs)

            logits = outputs.get("logits", outputs.get("predictions")) if isinstance(outputs, dict) else outputs

            all_logits.append(logits.cpu())
            if labels is not None:
                all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0) if all_labels else None

    if labels is None:
        raise ValueError("Labels required for calibration")

    # Create and fit calibrator
    if method == "temperature":
        calibrator = TemperatureScaling()
    elif method == "vector":
        num_classes = logits.shape[-1] if logits.dim() == 2 else 2
        calibrator = VectorScaling(num_classes)
    elif method == "platt":
        calibrator = PlattScaling()
    elif method == "isotonic":
        calibrator = IsotonicCalibration()
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    calibrator.fit(logits, labels, verbose=verbose)

    # Compute metrics before and after calibration
    probs_before = F.softmax(logits, dim=-1)
    probs_after = calibrator.calibrate(logits)

    metrics_before = compute_calibration_metrics(probs_before, labels)
    metrics_after = compute_calibration_metrics(probs_after, labels)

    if verbose:
        print(f"\nCalibration Results ({method}):")
        print(f"  ECE: {metrics_before.ece:.4f} -> {metrics_after.ece:.4f}")
        print(f"  MCE: {metrics_before.mce:.4f} -> {metrics_after.mce:.4f}")
        print(f"  Brier: {metrics_before.brier:.4f} -> {metrics_after.brier:.4f}")

    return calibrator, metrics_after


class CalibratedModel(nn.Module):
    """Wrapper that applies calibration to model outputs.

    Combines a trained model with a fitted calibrator for end-to-end
    calibrated predictions.
    """

    def __init__(self, model: nn.Module, calibrator: BaseCalibrator):
        """Initialize calibrated model.

        Args:
            model: Base model
            calibrator: Fitted calibrator
        """
        super().__init__()
        self.model = model
        self.calibrator = calibrator

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with calibration.

        Args:
            x: Input tensor

        Returns:
            Calibrated probabilities
        """
        with torch.no_grad():
            outputs = self.model(x)

        logits = outputs.get("logits", outputs.get("predictions")) if isinstance(outputs, dict) else outputs

        return self.calibrator.calibrate(logits)

    def predict_with_uncertainty(
        self,
        x: Tensor,
    ) -> dict[str, Tensor]:
        """Get calibrated predictions with uncertainty.

        Args:
            x: Input tensor

        Returns:
            Dict with predictions, probabilities, and confidence
        """
        probs = self.forward(x)
        predictions = probs.argmax(dim=-1)
        confidence = probs.max(dim=-1).values

        return {
            "predictions": predictions,
            "probabilities": probs,
            "confidence": confidence,
        }
