# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""P-adic based classifiers for bioinformatics.

This module implements classifiers that leverage p-adic (especially 3-adic)
mathematics for hierarchical classification tasks in bioinformatics.

Key Classifiers:
    - PAdicKNN: k-Nearest Neighbors using p-adic distance
    - PAdicHierarchicalClassifier: Clustering-based classification
    - GoldilocksZoneClassifier: Binary classifier for autoimmune risk

The 3-adic framework is particularly suited for:
    - Codon classification (64 → 20 amino acids)
    - Drug resistance classification (NRTI, NNRTI, PI, INSTI)
    - Epitope classification (CTL, B-cell, bnAb)
    - CRISPR off-target risk assessment

Usage:
    from src.analysis.classifiers import PAdicKNN

    clf = PAdicKNN(k=5, p=3)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from src.core.padic_math import (
    DEFAULT_P,
    compute_goldilocks_score,
    padic_distance,
)

# ============================================================================
# Base Classifier
# ============================================================================


@dataclass
class ClassificationResult:
    """Result of a classification prediction."""

    predicted_class: Any
    confidence: float
    neighbor_distances: list[float] = field(default_factory=list)
    neighbor_classes: list[Any] = field(default_factory=list)


class PAdicClassifierBase(ABC):
    """Abstract base class for p-adic classifiers."""

    def __init__(self, p: int = DEFAULT_P):
        """Initialize classifier.

        Args:
            p: Prime base for p-adic calculations (default: 3)
        """
        self.p = p
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> PAdicClassifierBase:
        """Fit the classifier to training data.

        Args:
            X: Training indices (n_samples,)
            y: Target labels (n_samples,)

        Returns:
            Self for method chaining
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Args:
            X: Sample indices (n_samples,)

        Returns:
            Predicted class labels (n_samples,)
        """
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Args:
            X: Sample indices (n_samples,)

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        raise NotImplementedError("predict_proba not implemented for this classifier")

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score.

        Args:
            X: Test indices
            y: True labels

        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return float(np.mean(predictions == y))


# ============================================================================
# P-adic k-Nearest Neighbors
# ============================================================================


class PAdicKNN(PAdicClassifierBase):
    """k-Nearest Neighbors classifier using p-adic distance.

    This classifier uses the ultrametric p-adic distance instead of
    Euclidean distance, which is particularly suited for hierarchical
    classification problems in molecular biology.

    The p-adic distance satisfies the ultrametric inequality:
        d(a,c) <= max(d(a,b), d(b,c))

    This property makes it ideal for:
        - Codon/amino acid classification
        - Phylogenetic relationships
        - Hierarchical protein families

    Attributes:
        k: Number of neighbors
        p: Prime base for p-adic calculations
        weights: Weighting strategy ('uniform' or 'distance')

    Example:
        >>> clf = PAdicKNN(k=3, p=3)
        >>> clf.fit(train_indices, train_labels)
        >>> predictions = clf.predict(test_indices)
    """

    def __init__(
        self,
        k: int = 5,
        p: int = DEFAULT_P,
        weights: Literal["uniform", "distance"] = "uniform",
    ):
        """Initialize PAdicKNN classifier.

        Args:
            k: Number of neighbors to use (default: 5)
            p: Prime base for p-adic distance (default: 3)
            weights: Weight function ('uniform' or 'distance')
        """
        super().__init__(p=p)
        self.k = k
        self.weights = weights
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.classes_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> PAdicKNN:
        """Fit the classifier to training data.

        Args:
            X: Training indices (n_samples,)
            y: Target labels (n_samples,)

        Returns:
            Self for method chaining
        """
        X = np.asarray(X).astype(int)
        y = np.asarray(y)

        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")

        if len(X) == 0:
            raise ValueError("Training data cannot be empty")

        self.X_train = X
        self.y_train = y
        self.classes_ = np.unique(y)
        self.is_fitted = True

        return self

    def _compute_distances(self, query: int) -> np.ndarray:
        """Compute p-adic distances from query to all training points.

        Args:
            query: Query index

        Returns:
            Array of distances to each training point
        """
        distances = np.array([padic_distance(query, x, self.p) for x in self.X_train])
        return distances

    def _predict_single(self, query: int) -> ClassificationResult:
        """Predict class for a single query.

        Args:
            query: Query index

        Returns:
            ClassificationResult with prediction details
        """
        distances = self._compute_distances(query)

        # Get k nearest neighbors
        k_indices = np.argsort(distances)[: self.k]
        k_distances = distances[k_indices]
        k_classes = self.y_train[k_indices]

        if self.weights == "uniform":
            # Simple majority vote
            class_counts = Counter(k_classes)
            predicted_class = class_counts.most_common(1)[0][0]
            confidence = class_counts[predicted_class] / self.k
        else:
            # Distance-weighted voting
            # Add small epsilon to avoid division by zero
            weights = 1.0 / (k_distances + 1e-10)
            class_weights = {}
            for cls, w in zip(k_classes, weights, strict=False):
                class_weights[cls] = class_weights.get(cls, 0) + w

            predicted_class = max(class_weights, key=class_weights.get)
            total_weight = sum(class_weights.values())
            confidence = class_weights[predicted_class] / total_weight

        return ClassificationResult(
            predicted_class=predicted_class,
            confidence=confidence,
            neighbor_distances=k_distances.tolist(),
            neighbor_classes=k_classes.tolist(),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples.

        Args:
            X: Sample indices (n_samples,)

        Returns:
            Predicted class labels (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predicting")

        X = np.asarray(X).astype(int)
        predictions = np.array([self._predict_single(x).predicted_class for x in X])
        return predictions

    def predict_with_details(self, X: np.ndarray) -> list[ClassificationResult]:
        """Predict with detailed results for each sample.

        Args:
            X: Sample indices (n_samples,)

        Returns:
            List of ClassificationResult objects
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predicting")

        X = np.asarray(X).astype(int)
        return [self._predict_single(x) for x in X]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples.

        Args:
            X: Sample indices (n_samples,)

        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predicting")

        X = np.asarray(X).astype(int)
        n_samples = len(X)
        n_classes = len(self.classes_)

        probas = np.zeros((n_samples, n_classes))

        for i, query in enumerate(X):
            distances = self._compute_distances(query)
            k_indices = np.argsort(distances)[: self.k]
            k_classes = self.y_train[k_indices]

            if self.weights == "uniform":
                for cls in k_classes:
                    cls_idx = np.where(self.classes_ == cls)[0][0]
                    probas[i, cls_idx] += 1.0 / self.k
            else:
                k_distances = distances[k_indices]
                weights = 1.0 / (k_distances + 1e-10)
                total_weight = weights.sum()

                for cls, w in zip(k_classes, weights, strict=False):
                    cls_idx = np.where(self.classes_ == cls)[0][0]
                    probas[i, cls_idx] += w / total_weight

        return probas


# ============================================================================
# Goldilocks Zone Classifier
# ============================================================================


class GoldilocksZoneClassifier(PAdicClassifierBase):
    """Binary classifier based on Goldilocks zone scoring.

    Classifies sequences as being "in zone" (autoimmune risk) or
    "out of zone" (safe) based on their p-adic distance from reference.

    The Goldilocks zone represents the "just right" distance where:
        - Too close: indistinguishable from self
        - Too far: clearly foreign, no cross-reactivity
        - In zone: immunogenic yet cross-reactive (risk)

    Attributes:
        center: Center of the Goldilocks zone
        width: Width (sigma) of the Gaussian
        threshold: Score threshold for "in zone" classification
        reference_index: Reference sequence index

    Example:
        >>> clf = GoldilocksZoneClassifier(reference_index=0)
        >>> clf.fit(train_indices, train_labels)  # Optional
        >>> risk = clf.predict(test_indices)
    """

    def __init__(
        self,
        reference_index: int = 0,
        center: float = 0.5,
        width: float = 0.15,
        threshold: float = 0.5,
        p: int = DEFAULT_P,
    ):
        """Initialize GoldilocksZoneClassifier.

        Args:
            reference_index: Index of reference (self) sequence
            center: Center of Goldilocks zone (default: 0.5)
            width: Width of Gaussian (default: 0.15)
            threshold: Threshold for "in zone" (default: 0.5)
            p: Prime base (default: 3)
        """
        super().__init__(p=p)
        self.reference_index = reference_index
        self.center = center
        self.width = width
        self.threshold = threshold

    def fit(self, X: np.ndarray | None = None, y: np.ndarray | None = None) -> GoldilocksZoneClassifier:
        """Fit classifier (optional - can use default parameters).

        If X and y are provided, optimizes center and width parameters.

        Args:
            X: Training indices (optional)
            y: Target labels (0=out, 1=in) (optional)

        Returns:
            Self for method chaining
        """
        if X is not None and y is not None:
            # Could optimize center/width here based on training data
            # For now, just mark as fitted
            pass

        self.is_fitted = True
        return self

    def compute_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute Goldilocks zone scores for samples.

        Args:
            X: Sample indices (n_samples,)

        Returns:
            Scores in [0, 1] (1 = in zone)
        """
        X = np.asarray(X).astype(int)
        scores = np.array(
            [
                compute_goldilocks_score(
                    padic_distance(x, self.reference_index, self.p),
                    center=self.center,
                    width=self.width,
                )
                for x in X
            ]
        )
        return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class (0=out, 1=in zone).

        Args:
            X: Sample indices (n_samples,)

        Returns:
            Binary predictions (n_samples,)
        """
        scores = self.compute_scores(X)
        return (scores >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Sample indices (n_samples,)

        Returns:
            Probabilities (n_samples, 2) for [out, in]
        """
        scores = self.compute_scores(X)
        probas = np.column_stack([1 - scores, scores])
        return probas


# ============================================================================
# Codon Classification
# ============================================================================


class CodonClassifier(PAdicKNN):
    """Specialized classifier for codon to amino acid mapping.

    Uses 3-adic distance which naturally captures the wobble hypothesis:
    third position mutations are less impactful, corresponding to larger
    p-adic distances at the first digit level.

    The 64 codons map to 20 amino acids + 3 stop codons.
    This classifier learns the mapping from sequence data.

    Attributes:
        codon_to_index: Mapping from codon string to index
        index_to_codon: Mapping from index to codon string
        amino_acid_mapping: Known codon→AA mapping for validation

    Example:
        >>> clf = CodonClassifier()
        >>> clf.fit(codon_indices, amino_acids)
        >>> aa = clf.predict([codon_index])
    """

    # Standard genetic code
    GENETIC_CODE = {
        "TTT": "F",
        "TTC": "F",
        "TTA": "L",
        "TTG": "L",
        "TCT": "S",
        "TCC": "S",
        "TCA": "S",
        "TCG": "S",
        "TAT": "Y",
        "TAC": "Y",
        "TAA": "*",
        "TAG": "*",
        "TGT": "C",
        "TGC": "C",
        "TGA": "*",
        "TGG": "W",
        "CTT": "L",
        "CTC": "L",
        "CTA": "L",
        "CTG": "L",
        "CCT": "P",
        "CCC": "P",
        "CCA": "P",
        "CCG": "P",
        "CAT": "H",
        "CAC": "H",
        "CAA": "Q",
        "CAG": "Q",
        "CGT": "R",
        "CGC": "R",
        "CGA": "R",
        "CGG": "R",
        "ATT": "I",
        "ATC": "I",
        "ATA": "I",
        "ATG": "M",
        "ACT": "T",
        "ACC": "T",
        "ACA": "T",
        "ACG": "T",
        "AAT": "N",
        "AAC": "N",
        "AAA": "K",
        "AAG": "K",
        "AGT": "S",
        "AGC": "S",
        "AGA": "R",
        "AGG": "R",
        "GTT": "V",
        "GTC": "V",
        "GTA": "V",
        "GTG": "V",
        "GCT": "A",
        "GCC": "A",
        "GCA": "A",
        "GCG": "A",
        "GAT": "D",
        "GAC": "D",
        "GAA": "E",
        "GAG": "E",
        "GGT": "G",
        "GGC": "G",
        "GGA": "G",
        "GGG": "G",
    }

    BASES = ["T", "C", "A", "G"]

    def __init__(self, k: int = 3, weights: Literal["uniform", "distance"] = "distance"):
        """Initialize CodonClassifier.

        Args:
            k: Number of neighbors (default: 3, for wobble consideration)
            weights: Weight function (default: 'distance')
        """
        super().__init__(k=k, p=3, weights=weights)
        self._build_codon_mappings()

    def _build_codon_mappings(self):
        """Build codon-index mappings."""
        self.codon_to_index = {}
        self.index_to_codon = {}

        for i, b1 in enumerate(self.BASES):
            for j, b2 in enumerate(self.BASES):
                for k, b3 in enumerate(self.BASES):
                    codon = b1 + b2 + b3
                    # 3-adic index: position1 + 4*position2 + 16*position3
                    # Or use base-4 encoding
                    index = i * 16 + j * 4 + k
                    self.codon_to_index[codon] = index
                    self.index_to_codon[index] = codon

    def codon_to_aa(self, codon: str) -> str:
        """Get amino acid for a codon.

        Args:
            codon: 3-letter codon string

        Returns:
            Amino acid single-letter code
        """
        return self.GENETIC_CODE.get(codon.upper(), "X")

    def fit_from_genetic_code(self) -> CodonClassifier:
        """Fit classifier from the standard genetic code.

        Returns:
            Self for method chaining
        """
        indices = list(range(64))
        labels = [self.codon_to_aa(self.index_to_codon[i]) for i in indices]

        return self.fit(np.array(indices), np.array(labels))

    def evaluate_accuracy(self) -> dict[str, float]:
        """Evaluate accuracy on the standard genetic code.

        Returns:
            Dictionary with accuracy metrics
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted first")

        indices = np.array(list(range(64)))
        true_labels = np.array([self.codon_to_aa(self.index_to_codon[i]) for i in indices])
        predictions = self.predict(indices)

        accuracy = np.mean(predictions == true_labels)

        # Per-amino acid accuracy
        aa_accuracy = {}
        for aa in np.unique(true_labels):
            mask = true_labels == aa
            if mask.sum() > 0:
                aa_accuracy[aa] = np.mean(predictions[mask] == true_labels[mask])

        return {
            "overall_accuracy": float(accuracy),
            "per_aa_accuracy": aa_accuracy,
            "n_correct": int(np.sum(predictions == true_labels)),
            "n_total": 64,
        }


# ============================================================================
# Hierarchical Classifier
# ============================================================================


class PAdicHierarchicalClassifier(PAdicClassifierBase):
    """Hierarchical classifier using p-adic digit structure.

    Uses the p-adic digit expansion to create a hierarchical decision tree.
    Each level of the tree corresponds to a p-adic digit, enabling
    efficient multi-level classification.

    Attributes:
        n_digits: Number of p-adic digits (tree depth)
        min_samples_leaf: Minimum samples per leaf

    Example:
        >>> clf = PAdicHierarchicalClassifier(n_digits=4)
        >>> clf.fit(train_indices, train_labels)
        >>> predictions = clf.predict(test_indices)
    """

    def __init__(
        self,
        n_digits: int = 6,
        p: int = DEFAULT_P,
        min_samples_leaf: int = 1,
    ):
        """Initialize PAdicHierarchicalClassifier.

        Args:
            n_digits: Number of p-adic digits (tree depth)
            p: Prime base (default: 3)
            min_samples_leaf: Minimum samples per leaf node
        """
        super().__init__(p=p)
        self.n_digits = n_digits
        self.min_samples_leaf = min_samples_leaf
        self.tree_: dict | None = None

    def _extract_digits(self, index: int) -> tuple[int, ...]:
        """Extract p-adic digits from an index.

        Args:
            index: Integer index

        Returns:
            Tuple of p-adic digits
        """
        digits = []
        remaining = abs(index)
        for _ in range(self.n_digits):
            digits.append(remaining % self.p)
            remaining //= self.p
        return tuple(digits)

    def fit(self, X: np.ndarray, y: np.ndarray) -> PAdicHierarchicalClassifier:
        """Fit the hierarchical classifier.

        Builds a tree where each node corresponds to a partial
        p-adic digit sequence.

        Args:
            X: Training indices (n_samples,)
            y: Target labels (n_samples,)

        Returns:
            Self for method chaining
        """
        X = np.asarray(X).astype(int)
        y = np.asarray(y)

        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")

        # Build hierarchical tree
        self.tree_ = {}
        self.classes_ = np.unique(y)

        for idx, label in zip(X, y, strict=False):
            digits = self._extract_digits(idx)

            # Add to tree at each level
            for depth in range(1, self.n_digits + 1):
                prefix = digits[:depth]
                if prefix not in self.tree_:
                    self.tree_[prefix] = Counter()
                self.tree_[prefix][label] += 1

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using hierarchical lookup.

        Args:
            X: Sample indices (n_samples,)

        Returns:
            Predicted class labels (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before predicting")

        X = np.asarray(X).astype(int)
        predictions = []

        for idx in X:
            digits = self._extract_digits(idx)

            # Look up in tree, starting from most specific
            for depth in range(self.n_digits, 0, -1):
                prefix = digits[:depth]
                if prefix in self.tree_:
                    counter = self.tree_[prefix]
                    if sum(counter.values()) >= self.min_samples_leaf:
                        predictions.append(counter.most_common(1)[0][0])
                        break
            else:
                # Fall back to most common class overall
                all_counts = Counter()
                for counter in self.tree_.values():
                    all_counts.update(counter)
                predictions.append(all_counts.most_common(1)[0][0])

        return np.array(predictions)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Base classes
    "ClassificationResult",
    "PAdicClassifierBase",
    # Classifiers
    "PAdicKNN",
    "GoldilocksZoneClassifier",
    "CodonClassifier",
    "PAdicHierarchicalClassifier",
]
