# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Conformal Prediction for uncertainty quantification.

Provides distribution-free prediction sets with coverage guarantees.

Implements:
- Split Conformal Prediction
- Adaptive Conformal Prediction (ACI)
- Conformalized Quantile Regression
- RAPS (Regularized Adaptive Prediction Sets)

References:
    - Vovk et al., "Algorithmic Learning in a Random World" (2005)
    - Romano et al., "Conformalized Quantile Regression" (2019)
    - Angelopoulos et al., "Uncertainty Sets for Image Classifiers" (2021)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class PredictionSet:
    """Container for conformal prediction results.

    Attributes:
        sets: List of prediction sets for each sample
        set_sizes: Size of each prediction set
        coverage: Empirical coverage on calibration data
        alpha: Significance level used
        threshold: Conformity score threshold
    """

    sets: list[list[int]]
    set_sizes: np.ndarray
    coverage: float
    alpha: float
    threshold: float

    def average_set_size(self) -> float:
        """Compute average prediction set size."""
        return float(np.mean(self.set_sizes))

    def coverage_rate(self) -> float:
        """Return coverage rate."""
        return self.coverage


@dataclass
class RegressionInterval:
    """Container for conformal regression intervals.

    Attributes:
        lower: Lower bounds (N,)
        upper: Upper bounds (N,)
        coverage: Empirical coverage
        alpha: Significance level
        width: Interval widths (N,)
    """

    lower: np.ndarray
    upper: np.ndarray
    coverage: float
    alpha: float
    width: np.ndarray

    def average_width(self) -> float:
        """Compute average interval width."""
        return float(np.mean(self.width))


class BaseConformalPredictor(ABC):
    """Abstract base class for conformal predictors."""

    def __init__(self, alpha: float = 0.1):
        """Initialize conformal predictor.

        Args:
            alpha: Significance level (1 - alpha = target coverage)
        """
        self.alpha = alpha
        self._calibrated = False
        self._threshold: float | None = None

    @abstractmethod
    def calibrate(
        self,
        cal_scores: np.ndarray,
        cal_labels: np.ndarray,
    ) -> BaseConformalPredictor:
        """Calibrate on held-out data."""
        pass

    @abstractmethod
    def predict(self, test_scores: np.ndarray) -> Any:
        """Generate prediction sets/intervals."""
        pass

    @property
    def is_calibrated(self) -> bool:
        """Check if predictor is calibrated."""
        return self._calibrated


class SplitConformalClassifier(BaseConformalPredictor):
    """Split Conformal Prediction for classification.

    Uses softmax scores as conformity scores. Prediction sets include
    all classes with scores above a calibrated threshold.

    Example:
        >>> predictor = SplitConformalClassifier(alpha=0.1)
        >>> predictor.calibrate(val_probs, val_labels)
        >>> pred_sets = predictor.predict(test_probs)
        >>> print(f"Average set size: {pred_sets.average_set_size():.2f}")
    """

    def __init__(
        self,
        alpha: float = 0.1,
        score_fn: str = "softmax",
    ):
        """Initialize split conformal classifier.

        Args:
            alpha: Significance level (default 0.1 = 90% coverage)
            score_fn: Conformity score function ('softmax', 'aps', 'raps')
        """
        super().__init__(alpha)
        self.score_fn = score_fn

    def _compute_scores(
        self,
        probs: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute conformity scores.

        Args:
            probs: Predicted probabilities (N, C)
            labels: True labels (N,) - only for calibration

        Returns:
            Conformity scores (N,) for calibration or (N, C) for prediction
        """
        if labels is not None:
            # For calibration: score is 1 - P(true class)
            true_probs = probs[np.arange(len(labels)), labels]
            return 1 - true_probs
        else:
            # For prediction: score for each class is 1 - P(class)
            return 1 - probs

    def calibrate(
        self,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
    ) -> SplitConformalClassifier:
        """Calibrate threshold on validation data.

        Args:
            cal_probs: Calibration probabilities (N, C)
            cal_labels: Calibration labels (N,)

        Returns:
            Self for chaining
        """
        # Compute conformity scores for true labels
        scores = self._compute_scores(cal_probs, cal_labels)

        # Find quantile threshold
        n = len(scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        q = min(q, 1.0)
        self._threshold = float(np.quantile(scores, q))

        self._calibrated = True
        return self

    def predict(self, test_probs: np.ndarray) -> PredictionSet:
        """Generate prediction sets for test data.

        Args:
            test_probs: Test probabilities (N, C)

        Returns:
            PredictionSet with coverage guarantee
        """
        if not self._calibrated:
            raise RuntimeError("Must calibrate before prediction")

        # Compute scores for all classes
        scores = self._compute_scores(test_probs)

        # Include classes with score <= threshold
        pred_sets = []
        set_sizes = []

        for i in range(len(test_probs)):
            included = np.where(scores[i] <= self._threshold)[0].tolist()
            # Always include at least the most likely class
            if len(included) == 0:
                included = [int(np.argmax(test_probs[i]))]
            pred_sets.append(included)
            set_sizes.append(len(included))

        return PredictionSet(
            sets=pred_sets,
            set_sizes=np.array(set_sizes),
            coverage=1 - self.alpha,  # Guaranteed coverage
            alpha=self.alpha,
            threshold=self._threshold,
        )


class AdaptiveConformalClassifier(BaseConformalPredictor):
    """Adaptive Prediction Sets (APS) for classification.

    Orders classes by probability and includes classes until cumulative
    probability exceeds threshold. More adaptive than split conformal.

    Reference:
        Romano et al., "Classification with Valid and Adaptive Coverage" (2020)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        randomize: bool = True,
    ):
        """Initialize adaptive conformal classifier.

        Args:
            alpha: Significance level
            randomize: Use randomized conformity scores
        """
        super().__init__(alpha)
        self.randomize = randomize

    def _compute_aps_score(
        self,
        probs: np.ndarray,
        label: int,
        u: float = 0.5,
    ) -> float:
        """Compute APS conformity score.

        Args:
            probs: Class probabilities (C,)
            label: True label
            u: Random uniform for randomization

        Returns:
            APS conformity score
        """
        # Sort probabilities descending
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]

        # Find position of true label
        label_pos = np.where(sorted_indices == label)[0][0]

        # Cumulative sum up to (but not including) true label
        cumsum_before = sorted_probs[:label_pos].sum() if label_pos > 0 else 0.0

        # Score is cumulative probability needed to include true label
        score = cumsum_before + u * probs[label] if self.randomize else cumsum_before + probs[label]

        return score

    def calibrate(
        self,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
    ) -> AdaptiveConformalClassifier:
        """Calibrate on validation data.

        Args:
            cal_probs: Calibration probabilities (N, C)
            cal_labels: Calibration labels (N,)

        Returns:
            Self for chaining
        """
        n = len(cal_probs)
        scores = np.zeros(n)

        for i in range(n):
            u = np.random.uniform() if self.randomize else 0.5
            scores[i] = self._compute_aps_score(cal_probs[i], cal_labels[i], u)

        # Find quantile threshold
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        q = min(q, 1.0)
        self._threshold = float(np.quantile(scores, q))

        self._calibrated = True
        return self

    def predict(self, test_probs: np.ndarray) -> PredictionSet:
        """Generate adaptive prediction sets.

        Args:
            test_probs: Test probabilities (N, C)

        Returns:
            PredictionSet with adaptive sizes
        """
        if not self._calibrated:
            raise RuntimeError("Must calibrate before prediction")

        pred_sets = []
        set_sizes = []

        for i in range(len(test_probs)):
            probs = test_probs[i]
            sorted_indices = np.argsort(-probs)
            sorted_probs = probs[sorted_indices]

            # Include classes until cumsum exceeds threshold
            cumsum = 0.0
            included = []

            for j, idx in enumerate(sorted_indices):
                u = np.random.uniform() if self.randomize else 0.5
                cumsum_with_random = cumsum + u * sorted_probs[j]

                included.append(int(idx))

                if cumsum_with_random >= self._threshold:
                    break

                cumsum += sorted_probs[j]

            # Ensure at least one class included
            if len(included) == 0:
                included = [int(sorted_indices[0])]

            pred_sets.append(included)
            set_sizes.append(len(included))

        return PredictionSet(
            sets=pred_sets,
            set_sizes=np.array(set_sizes),
            coverage=1 - self.alpha,
            alpha=self.alpha,
            threshold=self._threshold,
        )


class RAPSConformalClassifier(BaseConformalPredictor):
    """Regularized Adaptive Prediction Sets (RAPS).

    Adds regularization to reduce set sizes while maintaining coverage.

    Reference:
        Angelopoulos et al., "Uncertainty Sets for Image Classifiers" (2021)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        k_reg: int = 2,
        lambda_reg: float = 0.01,
        randomize: bool = True,
    ):
        """Initialize RAPS classifier.

        Args:
            alpha: Significance level
            k_reg: Number of top classes before regularization
            lambda_reg: Regularization strength
            randomize: Use randomized scores
        """
        super().__init__(alpha)
        self.k_reg = k_reg
        self.lambda_reg = lambda_reg
        self.randomize = randomize

    def _compute_raps_score(
        self,
        probs: np.ndarray,
        label: int,
        u: float = 0.5,
    ) -> float:
        """Compute RAPS conformity score."""
        sorted_indices = np.argsort(-probs)
        sorted_probs = probs[sorted_indices]

        label_pos = np.where(sorted_indices == label)[0][0]

        # Base cumulative probability
        cumsum_before = sorted_probs[:label_pos].sum() if label_pos > 0 else 0.0

        # Add regularization penalty for classes beyond k_reg
        reg_penalty = self.lambda_reg * max(0, label_pos - self.k_reg)

        if self.randomize:
            score = cumsum_before + u * probs[label] + reg_penalty
        else:
            score = cumsum_before + probs[label] + reg_penalty

        return score

    def calibrate(
        self,
        cal_probs: np.ndarray,
        cal_labels: np.ndarray,
    ) -> RAPSConformalClassifier:
        """Calibrate RAPS predictor."""
        n = len(cal_probs)
        scores = np.zeros(n)

        for i in range(n):
            u = np.random.uniform() if self.randomize else 0.5
            scores[i] = self._compute_raps_score(cal_probs[i], cal_labels[i], u)

        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        q = min(q, 1.0)
        self._threshold = float(np.quantile(scores, q))

        self._calibrated = True
        return self

    def predict(self, test_probs: np.ndarray) -> PredictionSet:
        """Generate RAPS prediction sets."""
        if not self._calibrated:
            raise RuntimeError("Must calibrate before prediction")

        pred_sets = []
        set_sizes = []

        for i in range(len(test_probs)):
            probs = test_probs[i]
            sorted_indices = np.argsort(-probs)
            sorted_probs = probs[sorted_indices]

            cumsum = 0.0
            included = []

            for j, idx in enumerate(sorted_indices):
                u = np.random.uniform() if self.randomize else 0.5
                reg_penalty = self.lambda_reg * max(0, j - self.k_reg)
                score_with_random = cumsum + u * sorted_probs[j] + reg_penalty

                included.append(int(idx))

                if score_with_random >= self._threshold:
                    break

                cumsum += sorted_probs[j]

            if len(included) == 0:
                included = [int(sorted_indices[0])]

            pred_sets.append(included)
            set_sizes.append(len(included))

        return PredictionSet(
            sets=pred_sets,
            set_sizes=np.array(set_sizes),
            coverage=1 - self.alpha,
            alpha=self.alpha,
            threshold=self._threshold,
        )


class ConformalRegressor(BaseConformalPredictor):
    """Conformal Prediction for regression.

    Uses absolute residuals as conformity scores.
    """

    def __init__(self, alpha: float = 0.1):
        """Initialize conformal regressor.

        Args:
            alpha: Significance level (1 - alpha = coverage)
        """
        super().__init__(alpha)

    def calibrate(
        self,
        cal_preds: np.ndarray,
        cal_labels: np.ndarray,
    ) -> ConformalRegressor:
        """Calibrate using absolute residuals.

        Args:
            cal_preds: Predictions on calibration set (N,)
            cal_labels: True values (N,)

        Returns:
            Self for chaining
        """
        residuals = np.abs(cal_labels - cal_preds)
        n = len(residuals)

        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        q = min(q, 1.0)
        self._threshold = float(np.quantile(residuals, q))

        self._calibrated = True
        return self

    def predict(
        self,
        test_preds: np.ndarray,
        test_labels: np.ndarray | None = None,
    ) -> RegressionInterval:
        """Generate prediction intervals.

        Args:
            test_preds: Point predictions (N,)
            test_labels: True labels for coverage evaluation (optional)

        Returns:
            RegressionInterval with coverage guarantee
        """
        if not self._calibrated:
            raise RuntimeError("Must calibrate before prediction")

        lower = test_preds - self._threshold
        upper = test_preds + self._threshold
        width = np.full_like(test_preds, 2 * self._threshold)

        # Compute coverage if labels provided
        if test_labels is not None:
            covered = (test_labels >= lower) & (test_labels <= upper)
            coverage = float(covered.mean())
        else:
            coverage = 1 - self.alpha

        return RegressionInterval(
            lower=lower,
            upper=upper,
            coverage=coverage,
            alpha=self.alpha,
            width=width,
        )


class ConformizedQuantileRegressor(BaseConformalPredictor):
    """Conformalized Quantile Regression (CQR).

    Combines quantile regression with conformal prediction for
    adaptive interval widths.

    Reference:
        Romano et al., "Conformalized Quantile Regression" (2019)
    """

    def __init__(
        self,
        alpha: float = 0.1,
    ):
        """Initialize CQR.

        Args:
            alpha: Significance level
        """
        super().__init__(alpha)

    def calibrate(
        self,
        cal_lower: np.ndarray,
        cal_upper: np.ndarray,
        cal_labels: np.ndarray,
    ) -> ConformizedQuantileRegressor:
        """Calibrate using quantile predictions.

        Args:
            cal_lower: Lower quantile predictions (N,)
            cal_upper: Upper quantile predictions (N,)
            cal_labels: True values (N,)

        Returns:
            Self for chaining
        """
        # Conformity score: max of how much label exceeds bounds
        scores = np.maximum(cal_lower - cal_labels, cal_labels - cal_upper)

        n = len(scores)
        q = np.ceil((n + 1) * (1 - self.alpha)) / n
        q = min(q, 1.0)
        self._threshold = float(np.quantile(scores, q))

        self._calibrated = True
        return self

    def predict(
        self,
        test_lower: np.ndarray,
        test_upper: np.ndarray,
        test_labels: np.ndarray | None = None,
    ) -> RegressionInterval:
        """Generate conformalized prediction intervals.

        Args:
            test_lower: Lower quantile predictions (N,)
            test_upper: Upper quantile predictions (N,)
            test_labels: True labels for coverage evaluation

        Returns:
            RegressionInterval with adaptive widths
        """
        if not self._calibrated:
            raise RuntimeError("Must calibrate before prediction")

        # Expand intervals by threshold
        lower = test_lower - self._threshold
        upper = test_upper + self._threshold
        width = upper - lower

        if test_labels is not None:
            covered = (test_labels >= lower) & (test_labels <= upper)
            coverage = float(covered.mean())
        else:
            coverage = 1 - self.alpha

        return RegressionInterval(
            lower=lower,
            upper=upper,
            coverage=coverage,
            alpha=self.alpha,
            width=width,
        )


class ConformalPredictionWrapper(nn.Module):
    """Wrapper to add conformal prediction to any classifier.

    Example:
        >>> wrapper = ConformalPredictionWrapper(model, alpha=0.1)
        >>> wrapper.calibrate(val_loader, device)
        >>> pred_sets = wrapper.predict_sets(test_inputs)
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.1,
        method: str = "aps",
    ):
        """Initialize wrapper.

        Args:
            model: Base classifier model
            alpha: Significance level
            method: Conformal method ('split', 'aps', 'raps')
        """
        super().__init__()
        self.model = model
        self.alpha = alpha

        if method == "split":
            self.conformal = SplitConformalClassifier(alpha)
        elif method == "aps":
            self.conformal = AdaptiveConformalClassifier(alpha)
        elif method == "raps":
            self.conformal = RAPSConformalClassifier(alpha)
        else:
            raise ValueError(f"Unknown method: {method}")

    def calibrate(
        self,
        cal_loader: torch.utils.data.DataLoader,
        device: str = "cuda",
    ) -> ConformalPredictionWrapper:
        """Calibrate on validation data.

        Args:
            cal_loader: Calibration data loader
            device: Device for computation

        Returns:
            Self for chaining
        """
        self.model.eval()

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in cal_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch[0], batch[1]
                else:
                    inputs = batch
                    labels = None

                inputs = inputs.to(device)
                outputs = self.model(inputs)

                logits = outputs.get("logits", outputs.get("predictions")) if isinstance(outputs, dict) else outputs

                probs = torch.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

                if labels is not None:
                    all_labels.append(labels.numpy())

        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0) if all_labels else None

        if labels is None:
            raise ValueError("Labels required for calibration")

        self.conformal.calibrate(probs, labels)
        return self

    def forward(self, x: Tensor) -> Tensor:
        """Standard forward pass (no conformal)."""
        return self.model(x)

    def predict_sets(
        self,
        x: Tensor,
    ) -> PredictionSet:
        """Generate conformal prediction sets.

        Args:
            x: Input tensor

        Returns:
            PredictionSet with coverage guarantee
        """
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(x)

        logits = outputs.get("logits", outputs.get("predictions")) if isinstance(outputs, dict) else outputs

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return self.conformal.predict(probs)

    def predict_with_confidence(
        self,
        x: Tensor,
    ) -> dict[str, Any]:
        """Get predictions with conformal confidence.

        Args:
            x: Input tensor

        Returns:
            Dict with predictions, prediction sets, and set sizes
        """
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(x)

        logits = outputs.get("logits", outputs.get("predictions")) if isinstance(outputs, dict) else outputs

        probs = torch.softmax(logits, dim=-1)
        predictions = probs.argmax(dim=-1)

        pred_sets = self.conformal.predict(probs.cpu().numpy())

        return {
            "predictions": predictions,
            "probabilities": probs,
            "prediction_sets": pred_sets.sets,
            "set_sizes": torch.from_numpy(pred_sets.set_sizes),
            "coverage_guarantee": 1 - self.alpha,
        }


def evaluate_conformal_coverage(
    predictor: BaseConformalPredictor,
    test_probs: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, float]:
    """Evaluate conformal predictor coverage.

    Args:
        predictor: Calibrated conformal predictor
        test_probs: Test probabilities
        test_labels: Test labels

    Returns:
        Dict with coverage metrics
    """
    pred_sets = predictor.predict(test_probs)

    # Check if true label in prediction set
    covered = np.array([test_labels[i] in pred_sets.sets[i] for i in range(len(test_labels))])

    return {
        "empirical_coverage": float(covered.mean()),
        "target_coverage": 1 - predictor.alpha,
        "average_set_size": pred_sets.average_set_size(),
        "median_set_size": float(np.median(pred_sets.set_sizes)),
        "max_set_size": int(pred_sets.set_sizes.max()),
        "min_set_size": int(pred_sets.set_sizes.min()),
    }
