# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Coreceptor tropism classifier using hyperbolic features."""

from __future__ import annotations

import numpy as np

from .base_predictor import BasePredictor

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.svm import SVC

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class TropismClassifier(BasePredictor):
    """Classify HIV coreceptor tropism (CCR5 vs CXCR4).

    Uses hyperbolic geometry features of V3 loop sequences to predict
    coreceptor usage (R5 vs X4).
    """

    LABELS = {0: "CCR5", 1: "CXCR4"}
    REVERSE_LABELS = {"CCR5": 0, "R5": 0, "CXCR4": 1, "X4": 1}

    def __init__(
        self,
        model: any | None = None,
        classifier_type: str = "random_forest",
        n_estimators: int = 100,
        max_depth: int = 10,
    ):
        """Initialize tropism classifier.

        Args:
            model: Optional pre-trained model
            classifier_type: "random_forest" or "svm"
            n_estimators: Number of estimators (for RF)
            max_depth: Maximum depth (for RF)
        """
        super().__init__(model)
        self.classifier_type = classifier_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _extract_extended_features(self, sequence: str) -> np.ndarray:
        """Extract extended features for V3 loop analysis.

        Includes:
        - Standard hyperbolic features
        - Glycosylation site count
        - Net charge estimate
        - V3 loop-specific features
        """
        # Standard features
        base_features = self.feature_extractor.sequence_features(sequence)

        # Glycosylation sites (N-X-S/T pattern)
        n_glycans = 0
        for i in range(len(sequence) - 2):
            if sequence[i] == "N" and sequence[i + 1] != "P" and sequence[i + 2] in "ST":
                n_glycans += 1

        # Charge estimate
        positive = sum(1 for aa in sequence if aa in "RK")
        negative = sum(1 for aa in sequence if aa in "DE")
        net_charge = positive - negative

        # V3 crown region (positions 11-25 approximately)
        crown_start = min(10, len(sequence) - 1)
        crown_end = min(25, len(sequence))
        crown = sequence[crown_start:crown_end] if len(sequence) > 10 else sequence

        crown_features = self.feature_extractor.sequence_features(crown)

        # Combine all features
        extended = np.concatenate(
            [
                base_features,
                [n_glycans, n_glycans / max(len(sequence), 1)],  # Glycan count and density
                [positive, negative, net_charge],  # Charge features
                crown_features,  # Crown-specific features
            ]
        )

        return extended

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs,
    ) -> TropismClassifier:
        """Train the tropism classifier.

        Args:
            X: Training features (n_samples, n_features)
            y: Binary labels (0=CCR5, 1=CXCR4)

        Returns:
            Self for method chaining
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn required for TropismClassifier")

        if self.classifier_type == "svm":
            self.model = SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=42,
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                class_weight="balanced",
                random_state=42,
            )

        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def fit_from_sequences(
        self,
        sequences: list[str],
        tropisms: list[str],
        **kwargs,
    ) -> TropismClassifier:
        """Fit classifier from V3 sequences and tropism labels.

        Args:
            sequences: List of V3 loop sequences
            tropisms: List of tropism labels ("CCR5", "R5", "CXCR4", "X4")

        Returns:
            Self for method chaining
        """
        X = np.array([self._extract_extended_features(s) for s in sequences])
        y = np.array([self.REVERSE_LABELS.get(t.upper(), 0) for t in tropisms])

        return self.fit(X, y, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict tropism labels.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Binary predictions (0=CCR5, 1=CXCR4)
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")

        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict tropism probabilities.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Probabilities for each class (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Classifier not fitted. Call fit() first.")

        return self.model.predict_proba(X)

    def predict_from_sequences(self, sequences: list[str]) -> list[str]:
        """Predict tropism from V3 sequences.

        Args:
            sequences: List of V3 loop sequences

        Returns:
            List of tropism labels ("CCR5" or "CXCR4")
        """
        X = np.array([self._extract_extended_features(s) for s in sequences])
        predictions = self.predict(X)
        return [self.LABELS[p] for p in predictions]

    def predict_tropism(self, sequence: str) -> dict[str, any]:
        """Predict tropism for a single V3 sequence.

        Args:
            sequence: V3 loop amino acid sequence

        Returns:
            Dictionary with prediction, probabilities, and features
        """
        features = self._extract_extended_features(sequence)
        X = features.reshape(1, -1)

        pred = self.predict(X)[0]
        proba = self.predict_proba(X)[0]

        # Extract interpretable features
        n_glycans = int(features[6])
        net_charge = int(features[10])

        return {
            "tropism": self.LABELS[pred],
            "ccr5_probability": float(proba[0]),
            "cxcr4_probability": float(proba[1]),
            "confidence": float(max(proba)),
            "n_glycans": n_glycans,
            "net_charge": net_charge,
            "sequence_length": len(sequence),
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
            Dictionary with metrics
        """
        if not HAS_SKLEARN:
            return {"error": "scikit-learn not available"}

        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "true_negatives": int(cm[0, 0]),
            "false_positives": int(cm[0, 1]),
            "false_negatives": int(cm[1, 0]),
            "true_positives": int(cm[1, 1]),
        }

        # Sensitivity (CCR5 correctly identified)
        metrics["ccr5_sensitivity"] = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0

        # Specificity (CXCR4 correctly identified)
        metrics["cxcr4_sensitivity"] = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0

        return metrics

    @property
    def feature_importance(self) -> dict[str, float]:
        """Get feature importances from trained model."""
        if not self.is_fitted or not hasattr(self.model, "feature_importances_"):
            return {}

        feature_names = [
            "mean_radial",
            "std_radial",
            "min_radial",
            "max_radial",
            "range_radial",
            "skew",
            "n_glycans",
            "glycan_density",
            "positive_charge",
            "negative_charge",
            "net_charge",
            "crown_mean_radial",
            "crown_std_radial",
            "crown_min_radial",
            "crown_max_radial",
            "crown_range_radial",
            "crown_skew",
        ]

        importances = self.model.feature_importances_
        return dict(zip(feature_names[: len(importances)], importances, strict=False))
