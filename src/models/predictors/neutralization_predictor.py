# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Antibody neutralization predictor using hyperbolic features."""

from __future__ import annotations

import numpy as np

from .base_predictor import BasePredictor

try:
    from scipy.stats import spearmanr
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class NeutralizationPredictor(BasePredictor):
    """Predict antibody neutralization IC50 from epitope features.

    Uses hyperbolic geometry features of epitope sequences to predict
    neutralization sensitivity (IC50 values).
    """

    def __init__(
        self,
        model: any | None = None,
        mode: str = "regression",
        n_estimators: int = 100,
        max_depth: int = 8,
    ):
        """Initialize neutralization predictor.

        Args:
            model: Optional pre-trained model
            mode: "regression" for IC50 prediction, "classification" for sensitive/resistant
            n_estimators: Number of estimators
            max_depth: Maximum depth
        """
        super().__init__(model)
        self.mode = mode
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ic50_threshold: float = 1.0,
        **kwargs,
    ) -> NeutralizationPredictor:
        """Train the neutralization predictor.

        Args:
            X: Training features (n_samples, n_features)
            y: IC50 values (regression) or binary labels (classification)
            ic50_threshold: Threshold for converting IC50 to binary (if regression mode but binary y)

        Returns:
            Self for method chaining
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for NeutralizationPredictor")

        if self.mode == "regression":
            # Log-transform IC50 values
            y_transformed = np.log10(np.clip(y, 1e-10, None))

            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
            )
            self.model.fit(X, y_transformed)
        else:
            # Classification mode
            if y.dtype == float:
                y = (y < ic50_threshold).astype(int)

            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight="balanced",
                random_state=42,
            )
            self.model.fit(X, y)

        self.is_fitted = True
        self._ic50_threshold = ic50_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict IC50 values or sensitivity.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predicted IC50 values (regression) or binary sensitivity (classification)
        """
        if not self.is_fitted:
            raise ValueError("Predictor not fitted. Call fit() first.")

        predictions = self.model.predict(X)

        if self.mode == "regression":
            # Transform back from log scale
            predictions = 10**predictions

        return predictions

    def predict_sensitivity(
        self,
        X: np.ndarray,
        threshold: float | None = None,
    ) -> np.ndarray:
        """Predict sensitivity as binary labels.

        Args:
            X: Features (n_samples, n_features)
            threshold: IC50 threshold (default: use training threshold)

        Returns:
            Binary sensitivity labels (1=sensitive, 0=resistant)
        """
        if threshold is None:
            threshold = getattr(self, "_ic50_threshold", 1.0)

        if self.mode == "classification":
            return self.predict(X)
        else:
            ic50_pred = self.predict(X)
            return (ic50_pred < threshold).astype(int)

    def predict_from_epitope(
        self,
        epitope_sequence: str,
        antibody_class: str | None = None,
    ) -> dict[str, float]:
        """Predict neutralization from epitope sequence.

        Args:
            epitope_sequence: Epitope amino acid sequence
            antibody_class: Optional antibody class for context

        Returns:
            Dictionary with predictions
        """
        features = self.feature_extractor.sequence_features(epitope_sequence)
        X = features.reshape(1, -1)

        if self.mode == "regression":
            ic50 = self.predict(X)[0]
            sensitivity = ic50 < getattr(self, "_ic50_threshold", 1.0)
        else:
            sensitivity = bool(self.predict(X)[0])
            ic50 = 0.1 if sensitivity else 10.0

        return {
            "predicted_ic50": float(ic50),
            "is_sensitive": sensitivity,
            "epitope_length": len(epitope_sequence),
            "antibody_class": antibody_class or "unknown",
        }

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Compute appropriate metrics based on mode.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        if not HAS_SKLEARN:
            return {"error": "scikit-learn not available"}

        if self.mode == "regression":
            # Log-transform for metrics
            log_true = np.log10(np.clip(y_true, 1e-10, None))
            log_pred = np.log10(np.clip(y_pred, 1e-10, None))

            mse = mean_squared_error(log_true, log_pred)
            r2 = r2_score(log_true, log_pred)
            spearman_r, spearman_p = spearmanr(y_true, y_pred)

            return {
                "mse": float(mse),
                "rmse": float(np.sqrt(mse)),
                "r2": float(r2),
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
            }
        else:
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
            }

    @property
    def feature_importance(self) -> dict[str, float]:
        """Get feature importances from trained model."""
        if not self.is_fitted:
            return {}

        feature_names = [
            "mean_radial",
            "std_radial",
            "min_radial",
            "max_radial",
            "range_radial",
            "skew",
        ]

        importances = self.model.feature_importances_
        return dict(zip(feature_names[: len(importances)], importances, strict=False))
