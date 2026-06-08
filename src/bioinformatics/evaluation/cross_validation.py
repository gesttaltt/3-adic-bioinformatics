# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Cross-validation utilities for DDG prediction.

This module provides:
- Leave-one-out cross-validation (gold standard for small datasets)
- K-fold cross-validation with proper stratification
- Cross-validated predictions with sklearn Pipeline pattern
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.bioinformatics.evaluation.metrics import DDGMetrics, compute_all_metrics


@dataclass
class CVResult:
    """Container for cross-validation results."""

    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: DDGMetrics
    cv_type: str
    n_folds: int
    fold_metrics: list[DDGMetrics] | None = None


class CrossValidator:
    """Cross-validator for DDG prediction models.

    Supports both sklearn-style models and PyTorch models through
    a unified interface.
    """

    def __init__(
        self,
        model_factory: Callable[[], BaseEstimator],
        scaler: bool = True,
        seed: int = 42,
    ):
        """Initialize cross-validator.

        Args:
            model_factory: Callable that returns a fresh model instance
            scaler: Whether to apply StandardScaler
            seed: Random seed
        """
        self.model_factory = model_factory
        self.use_scaler = scaler
        self.seed = seed

    def _create_pipeline(self) -> Pipeline:
        """Create sklearn pipeline with optional scaling."""
        steps = []
        if self.use_scaler:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", self.model_factory()))
        return Pipeline(steps)

    def loo_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bootstrap_ci: bool = True,
    ) -> CVResult:
        """Run leave-one-out cross-validation.

        LOO is the gold standard for small datasets as it uses
        maximum training data while providing unbiased estimates.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            bootstrap_ci: Compute confidence intervals

        Returns:
            CVResult with predictions and metrics
        """
        X = np.asarray(X)
        y = np.asarray(y).flatten()
        n_samples = len(y)

        # Create pipeline for proper data leakage prevention
        pipeline = self._create_pipeline()

        # Leave-one-out predictions
        loo = LeaveOneOut()
        y_pred = cross_val_predict(pipeline, X, y, cv=loo)

        # Compute metrics
        metrics = compute_all_metrics(
            y,
            y_pred,
            bootstrap_ci=bootstrap_ci,
            seed=self.seed,
        )

        return CVResult(
            y_true=y,
            y_pred=y_pred,
            metrics=metrics,
            cv_type="LOO",
            n_folds=n_samples,
        )

    def kfold_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        n_repeats: int = 1,
        stratify: bool = False,
        bootstrap_ci: bool = True,
    ) -> CVResult:
        """Run k-fold cross-validation.

        Args:
            X: Feature matrix
            y: Target values
            n_folds: Number of folds
            n_repeats: Number of repetitions
            stratify: Use stratified folds (bins DDG values)
            bootstrap_ci: Compute confidence intervals

        Returns:
            CVResult with predictions and metrics
        """
        X = np.asarray(X)
        y = np.asarray(y).flatten()

        all_y_true = []
        all_y_pred = []
        fold_metrics = []

        for repeat in range(n_repeats):
            seed = self.seed + repeat

            if stratify:
                # Stratify by DDG bins
                y_bins = np.digitize(y, bins=[-np.inf, -1, 1, np.inf])
                cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
                splits = cv.split(X, y_bins)
            else:
                cv = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
                splits = cv.split(X)

            for train_idx, val_idx in splits:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                pipeline = self._create_pipeline()
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_val)

                all_y_true.extend(y_val)
                all_y_pred.extend(y_pred)

                # Fold metrics
                if len(y_val) > 5:
                    fold_metric = compute_all_metrics(y_val, y_pred, bootstrap_ci=False)
                    fold_metrics.append(fold_metric)

        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)

        metrics = compute_all_metrics(
            all_y_true,
            all_y_pred,
            bootstrap_ci=bootstrap_ci,
            seed=self.seed,
        )

        return CVResult(
            y_true=all_y_true,
            y_pred=all_y_pred,
            metrics=metrics,
            cv_type=f"{n_folds}-fold x {n_repeats}",
            n_folds=n_folds * n_repeats,
            fold_metrics=fold_metrics if fold_metrics else None,
        )


def loo_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable[[], BaseEstimator],
    scaler: bool = True,
    bootstrap_ci: bool = True,
    seed: int = 42,
) -> CVResult:
    """Convenience function for LOO cross-validation.

    Args:
        X: Feature matrix
        y: Target values
        model_factory: Callable returning fresh model
        scaler: Apply StandardScaler
        bootstrap_ci: Compute confidence intervals
        seed: Random seed

    Returns:
        CVResult
    """
    cv = CrossValidator(model_factory, scaler=scaler, seed=seed)
    return cv.loo_cv(X, y, bootstrap_ci=bootstrap_ci)


def kfold_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable[[], BaseEstimator],
    n_folds: int = 5,
    n_repeats: int = 3,
    scaler: bool = True,
    stratify: bool = True,
    bootstrap_ci: bool = True,
    seed: int = 42,
) -> CVResult:
    """Convenience function for k-fold cross-validation.

    Args:
        X: Feature matrix
        y: Target values
        model_factory: Callable returning fresh model
        n_folds: Number of folds
        n_repeats: Number of repetitions
        scaler: Apply StandardScaler
        stratify: Use stratified folds
        bootstrap_ci: Compute confidence intervals
        seed: Random seed

    Returns:
        CVResult
    """
    cv = CrossValidator(model_factory, scaler=scaler, seed=seed)
    return cv.kfold_cv(
        X,
        y,
        n_folds=n_folds,
        n_repeats=n_repeats,
        stratify=stratify,
        bootstrap_ci=bootstrap_ci,
    )


class PyTorchModelWrapper(BaseEstimator, RegressorMixin):
    """Wrapper to make PyTorch models sklearn-compatible for CV.

    This allows using cross_val_predict with PyTorch models.
    """

    def __init__(
        self,
        model_class: type,
        model_kwargs: dict = None,
        train_epochs: int = 100,
        train_lr: float = 1e-4,
        device: str = "cpu",
    ):
        """Initialize wrapper.

        Args:
            model_class: PyTorch model class
            model_kwargs: Model constructor arguments
            train_epochs: Training epochs per fold
            train_lr: Learning rate
            device: Training device
        """
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.train_epochs = train_epochs
        self.train_lr = train_lr
        self.device = device
        self.model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> PyTorchModelWrapper:
        """Fit model on training data."""
        import torch
        from torch.optim import Adam

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device).unsqueeze(-1)

        self.model_ = self.model_class(**self.model_kwargs).to(self.device)
        optimizer = Adam(self.model_.parameters(), lr=self.train_lr)

        self.model_.train()
        for _ in range(self.train_epochs):
            optimizer.zero_grad()
            loss_dict = self.model_.loss(X_tensor, y_tensor)
            loss_dict["loss"].backward()
            optimizer.step()

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        import torch

        if self.model_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

        self.model_.eval()
        with torch.no_grad():
            preds = self.model_.predict(X_tensor)

        return preds.cpu().numpy().flatten()


__all__ = [
    "CVResult",
    "CrossValidator",
    "loo_cv",
    "kfold_cv",
    "PyTorchModelWrapper",
]
