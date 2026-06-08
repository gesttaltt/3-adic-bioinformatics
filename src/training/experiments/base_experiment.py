# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Base experiment framework for reproducible evaluation.

This module provides the foundation for running standardized experiments
with fixed seeds, cross-validation, and comprehensive metric computation.
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run.

    Attributes:
        name: Experiment identifier
        seed: Random seed for reproducibility
        n_folds: Number of cross-validation folds
        n_repeats: Number of repeated runs
        device: Torch device (cuda/cpu)
        output_dir: Directory for results
        save_checkpoints: Whether to save model checkpoints
        save_predictions: Whether to save all predictions
        verbose: Logging verbosity level
    """

    name: str = "experiment"
    seed: int = 42
    n_folds: int = 5
    n_repeats: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "results/experiments"
    save_checkpoints: bool = False
    save_predictions: bool = True
    verbose: int = 1

    # Training parameters
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10

    # Model parameters
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64, 32])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ExperimentResult:
    """Results from an experiment run.

    Contains all metrics, predictions, and metadata for reproducibility.
    """

    config: ExperimentConfig
    metrics: dict[str, float]
    fold_metrics: list[dict[str, float]]
    predictions: dict[str, np.ndarray] | None = None
    training_history: list[dict[str, float]] | None = None
    runtime_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        result = {
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "fold_metrics": self.fold_metrics,
            "runtime_seconds": self.runtime_seconds,
            "timestamp": self.timestamp,
        }
        if self.predictions is not None:
            result["predictions"] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.predictions.items()
            }
        if self.training_history is not None:
            result["training_history"] = self.training_history
        return result

    def save(self, path: Path | None = None) -> Path:
        """Save results to JSON file."""
        if path is None:
            output_dir = Path(self.config.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{self.config.name}_{self.timestamp.replace(':', '-')}.json"

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Results saved to {path}")
        return path

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Experiment: {self.config.name}",
            f"Timestamp: {self.timestamp}",
            f"Runtime: {self.runtime_seconds:.2f}s",
            "",
            "Metrics:",
        ]
        for key, value in sorted(self.metrics.items()):
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")
        return "\n".join(lines)


class MetricComputer:
    """Compute various evaluation metrics.

    Supports both regression and classification tasks with
    proper handling of edge cases.
    """

    @staticmethod
    def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
        """Compute Spearman correlation coefficient."""
        if len(y_true) < 3:
            return 0.0, 1.0
        rho, pval = stats.spearmanr(y_true, y_pred)
        return float(rho) if not np.isnan(rho) else 0.0, float(pval)

    @staticmethod
    def pearson(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
        """Compute Pearson correlation coefficient."""
        if len(y_true) < 3:
            return 0.0, 1.0
        rho, pval = stats.pearsonr(y_true, y_pred)
        return float(rho) if not np.isnan(rho) else 0.0, float(pval)

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Root Mean Squared Error."""
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Error."""
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R-squared score."""
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def auc_roc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute AUC-ROC for binary classification."""
        try:
            return float(roc_auc_score(y_true, y_pred))
        except ValueError:
            return 0.5  # Default for single-class

    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Compute classification metrics."""
        y_pred_binary = (y_pred > 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred_binary)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred_binary, average="binary", zero_division=0)
        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        }

    @classmethod
    def compute_all(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        task_type: str = "regression",
    ) -> dict[str, float]:
        """Compute all relevant metrics for a task type.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values
            task_type: "regression" or "classification"

        Returns:
            Dictionary of metric names to values
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        metrics = {}

        if task_type == "regression":
            spearman_rho, spearman_p = cls.spearman(y_true, y_pred)
            pearson_rho, pearson_p = cls.pearson(y_true, y_pred)
            metrics.update(
                {
                    "spearman": spearman_rho,
                    "spearman_p": spearman_p,
                    "pearson": pearson_rho,
                    "pearson_p": pearson_p,
                    "rmse": cls.rmse(y_true, y_pred),
                    "mae": cls.mae(y_true, y_pred),
                    "r2": cls.r2(y_true, y_pred),
                }
            )
        elif task_type == "classification":
            metrics.update(cls.classification_metrics(y_true, y_pred))
            metrics["auc_roc"] = cls.auc_roc(y_true, y_pred)

        return metrics


class BaseExperiment(ABC):
    """Abstract base class for experiments.

    Provides the scaffolding for reproducible experiments with:
    - Fixed random seeds
    - K-fold cross-validation
    - Metric computation and aggregation
    - Result saving

    Subclasses must implement:
    - create_model: Create a fresh model instance
    - train_fold: Train model on one fold
    - evaluate: Evaluate model on test data
    """

    def __init__(self, config: ExperimentConfig):
        """Initialize experiment.

        Args:
            config: Experiment configuration
        """
        self.config = config
        self.device = torch.device(config.device)
        self._set_seeds(config.seed)

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        if config.verbose > 0:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )

    def _set_seeds(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    @abstractmethod
    def create_model(self) -> torch.nn.Module:
        """Create a fresh model instance."""
        pass

    @abstractmethod
    def train_fold(
        self,
        model: torch.nn.Module,
        train_data: tuple[np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray],
    ) -> dict[str, Any]:
        """Train model on one fold.

        Args:
            model: Model to train
            train_data: (X_train, y_train)
            val_data: (X_val, y_val)

        Returns:
            Training history and metadata
        """
        pass

    @abstractmethod
    def evaluate(
        self,
        model: torch.nn.Module,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Evaluate model on test data.

        Args:
            model: Trained model
            test_data: (X_test, y_test)

        Returns:
            (predictions, metrics)
        """
        pass

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str = "regression",
    ) -> ExperimentResult:
        """Run the full experiment with cross-validation.

        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            task_type: "regression" or "classification"

        Returns:
            ExperimentResult with aggregated metrics
        """
        start_time = time.time()
        logger.info(f"Starting experiment: {self.config.name}")
        logger.info(f"Data shape: X={X.shape}, y={y.shape}")

        all_fold_metrics = []
        all_predictions = {"y_true": [], "y_pred": [], "fold": [], "repeat": []}

        for repeat in range(self.config.n_repeats):
            self._set_seeds(self.config.seed + repeat)

            # Create cross-validation splitter
            if task_type == "classification":
                kfold = StratifiedKFold(
                    n_splits=self.config.n_folds,
                    shuffle=True,
                    random_state=self.config.seed + repeat,
                )
                split_iter = kfold.split(X, y)
            else:
                kfold = KFold(
                    n_splits=self.config.n_folds,
                    shuffle=True,
                    random_state=self.config.seed + repeat,
                )
                split_iter = kfold.split(X)

            for fold, (train_idx, val_idx) in enumerate(split_iter):
                logger.info(f"Repeat {repeat + 1}/{self.config.n_repeats}, Fold {fold + 1}/{self.config.n_folds}")

                # Split data
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Create and train model
                model = self.create_model()
                self.train_fold(model, (X_train, y_train), (X_val, y_val))

                # Evaluate
                predictions, metrics = self.evaluate(model, (X_val, y_val))
                all_fold_metrics.append(metrics)

                # Store predictions
                if self.config.save_predictions:
                    all_predictions["y_true"].extend(y_val.tolist())
                    all_predictions["y_pred"].extend(predictions.tolist())
                    all_predictions["fold"].extend([fold] * len(y_val))
                    all_predictions["repeat"].extend([repeat] * len(y_val))

                logger.info(f"  Spearman: {metrics.get('spearman', 'N/A'):.4f}")

        # Aggregate metrics
        aggregated = self._aggregate_metrics(all_fold_metrics)
        runtime = time.time() - start_time

        # Create result
        result = ExperimentResult(
            config=self.config,
            metrics=aggregated,
            fold_metrics=all_fold_metrics,
            predictions={k: np.array(v) for k, v in all_predictions.items()} if self.config.save_predictions else None,
            runtime_seconds=runtime,
        )

        logger.info(f"\n{result.summary()}")

        # Save results
        result.save()

        return result

    def _aggregate_metrics(self, fold_metrics: list[dict[str, float]]) -> dict[str, float]:
        """Aggregate metrics across folds."""
        aggregated = {}
        keys = fold_metrics[0].keys()

        for key in keys:
            values = [m[key] for m in fold_metrics if key in m]
            if values:
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))
                aggregated[f"{key}_min"] = float(np.min(values))
                aggregated[f"{key}_max"] = float(np.max(values))

        return aggregated
