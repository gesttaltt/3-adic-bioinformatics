# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Drug resistance predictor using hyperbolic features."""

from __future__ import annotations

import numpy as np

from .base_predictor import BasePredictor

try:
    from scipy.stats import spearmanr
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class ResistancePredictor(BasePredictor):
    """Predict drug resistance fold-change from mutation features.

    Uses hyperbolic geometry features to predict the fold-change in
    drug resistance caused by mutations.
    """

    def __init__(
        self,
        model: any | None = None,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
    ):
        """Initialize resistance predictor.

        Args:
            model: Optional pre-trained model
            n_estimators: Number of boosting stages
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
        """
        super().__init__(model)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        log_transform: bool = True,
        **kwargs,
    ) -> ResistancePredictor:
        """Train the resistance predictor.

        Args:
            X: Training features (n_samples, n_features)
            y: Fold-change values (n_samples,)
            log_transform: Whether to log-transform targets

        Returns:
            Self for method chaining
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for ResistancePredictor")

        if log_transform:
            y = np.log10(np.clip(y, 1e-10, None))

        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
        )
        self.model.fit(X, y)
        self.is_fitted = True
        self._log_transform = log_transform

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict fold-change values.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predicted fold-change values
        """
        if not self.is_fitted:
            raise ValueError("Predictor not fitted. Call fit() first.")

        predictions = self.model.predict(X)

        if getattr(self, "_log_transform", True):
            predictions = 10**predictions

        return predictions

    def predict_from_mutations(
        self,
        mutations: list[tuple[str, str]],
    ) -> np.ndarray:
        """Predict resistance from mutation pairs.

        Args:
            mutations: List of (wild_type, mutant) amino acid pairs

        Returns:
            Predicted fold-change values
        """
        X = np.array([self.feature_extractor.mutation_features(wt, mut) for wt, mut in mutations])
        return self.predict(X)

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Compute regression metrics.

        Args:
            y_true: True fold-change values
            y_pred: Predicted fold-change values

        Returns:
            Dictionary with MSE, RMSE, R2, Spearman correlation
        """
        if not HAS_SKLEARN:
            return {"error": "scikit-learn not available"}

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

    @property
    def feature_importance(self) -> dict[str, float]:
        """Get feature importances from trained model."""
        if not self.is_fitted:
            return {}

        feature_names = [
            "wt_radial",
            "mut_radial",
            "radial_change",
            "distance",
        ]

        importances = self.model.feature_importances_
        return dict(zip(feature_names[: len(importances)], importances, strict=False))
