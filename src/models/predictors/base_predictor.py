# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Base predictor class and hyperbolic feature extraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from src.core.padic_math import PADIC_INFINITY
from src.core.padic_math import padic_valuation as core_padic_valuation

try:
    import joblib

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


class HyperbolicFeatureExtractor:
    """Extract hyperbolic features from amino acid sequences.

    Uses p-adic valuation to compute radial positions in hyperbolic space.
    """

    def __init__(self, p: int = 3):
        """Initialize feature extractor.

        Args:
            p: Prime base for p-adic valuation (default: 3 for ternary)
        """
        self.p = p
        self._load_codon_mappings()

    def _load_codon_mappings(self) -> None:
        """Load codon to amino acid mappings."""
        from src.biology.codons import AMINO_ACID_TO_CODONS, codon_to_index

        self.aa_to_codons = AMINO_ACID_TO_CODONS
        self.codon_to_index = codon_to_index

    def padic_valuation(self, n: int) -> int:
        """Compute p-adic valuation of integer n.

        Uses centralized padic_valuation from src.core.padic_math.
        """
        val = core_padic_valuation(n, self.p)
        # Return PADIC_INFINITY for n=0 (core returns PADIC_INFINITY_INT=100)
        return PADIC_INFINITY if n == 0 else val

    def codon_to_radial(self, codon: str) -> float:
        """Map codon to radial position in Poincaré ball."""
        try:
            idx = self.codon_to_index(codon)
            v = self.padic_valuation(idx) if idx > 0 else 0
            # Higher valuation -> closer to origin -> smaller radial
            return 1.0 - (v / 5.0)  # Normalize to [0, 1]
        except (KeyError, ValueError):
            return 0.5  # Default to middle

    def aa_to_radial(self, aa: str) -> float:
        """Map amino acid to radial position using representative codon."""
        codons = self.aa_to_codons.get(aa.upper(), [])
        if codons:
            return self.codon_to_radial(codons[0])
        return 0.5

    def sequence_features(self, sequence: str) -> np.ndarray:
        """Extract features from amino acid sequence.

        Returns:
            Array of shape (n_features,) with:
            - mean_radial: Mean radial position
            - std_radial: Standard deviation
            - min_radial: Minimum radial
            - max_radial: Maximum radial
            - range_radial: Max - min
            - skew: Distribution skewness
        """
        radials = [self.aa_to_radial(aa) for aa in sequence if aa.isalpha()]

        if not radials:
            return np.zeros(6)

        arr = np.array(radials)

        features = np.array(
            [
                np.mean(arr),
                np.std(arr),
                np.min(arr),
                np.max(arr),
                np.max(arr) - np.min(arr),
                self._skewness(arr),
            ]
        )

        return features

    def mutation_features(self, wild_type: str, mutant: str) -> np.ndarray:
        """Extract features for a single amino acid mutation.

        Returns:
            Array with:
            - wt_radial: Wild-type radial position
            - mut_radial: Mutant radial position
            - radial_change: Change in radial position
            - distance: Hyperbolic-like distance
        """
        wt_radial = self.aa_to_radial(wild_type)
        mut_radial = self.aa_to_radial(mutant)

        radial_change = mut_radial - wt_radial
        distance = np.arctanh(min(abs(radial_change), 0.99)) * 2

        return np.array([wt_radial, mut_radial, radial_change, distance])

    def _skewness(self, arr: np.ndarray) -> float:
        """Compute skewness of array."""
        if len(arr) < 3:
            return 0.0
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return 0.0
        return np.mean(((arr - mean) / std) ** 3)


class BasePredictor(ABC):
    """Abstract base class for all HIV predictors."""

    def __init__(self, model: Any = None):
        """Initialize predictor.

        Args:
            model: Optional pre-trained model
        """
        self.model = model
        self.feature_extractor = HyperbolicFeatureExtractor()
        self.is_fitted = model is not None

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> BasePredictor:
        """Train the predictor.

        Args:
            X: Training features
            y: Training labels/targets

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features to predict

        Returns:
            Predictions
        """
        pass

    def fit_from_sequences(
        self,
        sequences: list[str],
        targets: np.ndarray,
        **kwargs,
    ) -> BasePredictor:
        """Fit predictor from raw sequences.

        Args:
            sequences: List of amino acid sequences
            targets: Target values

        Returns:
            Self for method chaining
        """
        X = np.array([self.feature_extractor.sequence_features(s) for s in sequences])
        return self.fit(X, targets, **kwargs)

    def predict_from_sequences(self, sequences: list[str]) -> np.ndarray:
        """Make predictions from raw sequences.

        Args:
            sequences: List of amino acid sequences

        Returns:
            Predictions
        """
        X = np.array([self.feature_extractor.sequence_features(s) for s in sequences])
        return self.predict(X)

    def save(self, path: str | Path) -> None:
        """Save model to file.

        Args:
            path: Path to save model
        """
        if not HAS_JOBLIB:
            raise ImportError("joblib required for model serialization")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            {
                "model": self.model,
                "is_fitted": self.is_fitted,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> BasePredictor:
        """Load model from file.

        Args:
            path: Path to load model from

        Returns:
            Loaded predictor
        """
        if not HAS_JOBLIB:
            raise ImportError("joblib required for model serialization")

        data = joblib.load(path)
        instance = cls(model=data["model"])
        instance.is_fitted = data["is_fitted"]
        return instance

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        """Evaluate predictor on test data.

        Args:
            X: Test features
            y: True values

        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X)
        return self._compute_metrics(y, predictions)

    @abstractmethod
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Compute evaluation metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
        """
        pass
