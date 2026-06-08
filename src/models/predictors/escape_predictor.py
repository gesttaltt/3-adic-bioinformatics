# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Escape mutation predictor using hyperbolic features."""

from __future__ import annotations

import numpy as np

from .base_predictor import BasePredictor

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class EscapePredictor(BasePredictor):
    """Predict whether a mutation leads to immune/antibody escape.

    Uses hyperbolic geometry features to classify mutations as
    escape vs non-escape based on their geometric properties.
    """

    def __init__(
        self,
        model: any | None = None,
        n_estimators: int = 100,
        max_depth: int = 10,
        class_weight: str = "balanced",
    ):
        """Initialize escape predictor.

        Args:
            model: Optional pre-trained model
            n_estimators: Number of trees in forest
            max_depth: Maximum tree depth
            class_weight: Class weight strategy
        """
        super().__init__(model)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight = class_weight

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> EscapePredictor:
        """Train the escape predictor.

        Args:
            X: Training features (n_samples, n_features)
            y: Binary labels (0=no escape, 1=escape)

        Returns:
            Self for method chaining
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for EscapePredictor")

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            class_weight=self.class_weight,
            random_state=42,
        )
        self.model.fit(X, y)
        self.is_fitted = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict escape labels.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Binary predictions (0 or 1)
        """
        if not self.is_fitted:
            raise ValueError("Predictor not fitted. Call fit() first.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict escape probabilities.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Probability of escape for each sample
        """
        if not self.is_fitted:
            raise ValueError("Predictor not fitted. Call fit() first.")

        return self.model.predict_proba(X)[:, 1]

    def predict_escape_risk(
        self,
        epitope_sequence: str,
        mutation_positions: list[int],
        mutant_aas: list[str],
    ) -> dict[str, float]:
        """Predict escape risk for mutations in an epitope.

        Args:
            epitope_sequence: Wild-type epitope sequence
            mutation_positions: Positions of mutations (0-indexed)
            mutant_aas: Mutant amino acids at each position

        Returns:
            Dictionary with escape risk assessment
        """
        if len(mutation_positions) != len(mutant_aas):
            raise ValueError("mutation_positions and mutant_aas must have same length")

        features = []
        for pos, mut_aa in zip(mutation_positions, mutant_aas, strict=False):
            if pos >= len(epitope_sequence):
                continue

            wt_aa = epitope_sequence[pos]
            mut_feat = self.feature_extractor.mutation_features(wt_aa, mut_aa)
            seq_feat = self.feature_extractor.sequence_features(epitope_sequence)
            features.append(np.concatenate([mut_feat, seq_feat]))

        if not features:
            return {"escape_risk": 0.0, "n_mutations": 0}

        X = np.array(features)
        probs = self.predict_proba(X)

        return {
            "escape_risk": float(np.max(probs)),
            "mean_escape_prob": float(np.mean(probs)),
            "n_mutations": len(features),
            "highest_risk_position": int(mutation_positions[np.argmax(probs)]),
        }

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Compute classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary with accuracy, precision, recall, F1, AUC
        """
        if not HAS_SKLEARN:
            return {"error": "scikit-learn not available"}

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }

        return metrics

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
            "seq_mean_radial",
            "seq_std_radial",
            "seq_min_radial",
            "seq_max_radial",
            "seq_range_radial",
            "seq_skew",
        ]

        importances = self.model.feature_importances_
        return dict(zip(feature_names[: len(importances)], importances, strict=False))
