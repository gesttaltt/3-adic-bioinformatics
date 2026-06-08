# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Grokking Detection and Training Dynamics Analysis.

Implements detection mechanisms for the grokking phenomenon, where neural networks
exhibit sudden generalization long after achieving near-zero training loss.

Key References:
- Power et al. (2022): "Grokking: Generalization Beyond Overfitting on Small
  Algorithmic Datasets" (arXiv:2201.02177)
- Nanda et al. (2023): "Progress Measures for Grokking via Mechanistic
  Interpretability" (arXiv:2301.05217)
- Liu et al. (2024): "Deep Networks Always Grok and Here is Why"
  (arXiv:2402.15555)

Grokking Indicators:
1. Training loss → 0 while validation accuracy stagnant (memorization phase)
2. Local Complexity (LC) double-descent pattern
3. Sudden improvement in generalization after extended training
4. Weight norm dynamics showing compression phase

Anti-Grokking Patterns (what we observed in v5.10.1):
1. Best generalization at training START, not end
2. Phase transitions (β-warmup) causing performance degradation
3. Metrics never recovering after disruption
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import torch
import torch.nn as nn


class TrainingPhase(Enum):
    """Training phase classification."""

    WARMUP = "warmup"
    MEMORIZATION = "memorization"
    PLATEAU = "plateau"
    GROKKING = "grokking"
    DEGRADATION = "degradation"
    CONVERGED = "converged"


@dataclass
class EpochMetrics:
    """Metrics for a single training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float | None = None
    val_accuracy: float | None = None
    correlation: float | None = None
    coverage: float | None = None
    weight_norm: float | None = None
    gradient_norm: float | None = None
    local_complexity: float | None = None
    learning_rate: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class GrokDetectorConfig:
    """Configuration for grokking detection."""

    # Window sizes for moving averages
    short_window: int = 10
    long_window: int = 50

    # Thresholds for phase detection
    memorization_loss_threshold: float = 0.1  # Train loss < this = memorization
    generalization_gap_threshold: float = 0.5  # Val - Train gap
    improvement_threshold: float = 0.05  # Min improvement to detect grokking
    degradation_threshold: float = 0.1  # Max degradation before flagging

    # Grokking detection
    grokking_patience: int = 100  # Epochs to wait for grokking
    min_epochs_for_detection: int = 50

    # Local complexity estimation
    estimate_local_complexity: bool = True
    lc_sample_points: int = 1000


@dataclass
class GrokAnalysis:
    """Results of grokking analysis."""

    current_phase: TrainingPhase
    grokking_probability: float
    epochs_in_memorization: int
    best_generalization_epoch: int
    current_generalization_gap: float
    trend_direction: str  # "improving", "stable", "degrading"
    recommendations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "current_phase": self.current_phase.value,
            "grokking_probability": self.grokking_probability,
            "epochs_in_memorization": self.epochs_in_memorization,
            "best_generalization_epoch": self.best_generalization_epoch,
            "current_generalization_gap": self.current_generalization_gap,
            "trend_direction": self.trend_direction,
            "recommendations": self.recommendations,
            "warnings": self.warnings,
        }


class GrokDetector:
    """Detects grokking phenomenon and training dynamics issues.

    This detector monitors training metrics to identify:
    1. Classic grokking: sudden generalization after memorization
    2. Anti-grokking: best performance early, then degradation
    3. Phase transitions: sudden changes in training dynamics
    4. Stagnation: lack of progress over extended periods

    Example:
        detector = GrokDetector()

        for epoch in range(num_epochs):
            # ... training code ...
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                correlation=correlation,
                coverage=coverage,
            )
            analysis = detector.update(metrics)

            if analysis.current_phase == TrainingPhase.GROKKING:
                print("Grokking detected! Continue training.")
            elif analysis.current_phase == TrainingPhase.DEGRADATION:
                print("Performance degrading, consider early stopping.")
    """

    def __init__(self, config: GrokDetectorConfig | None = None):
        """Initialize the grokking detector.

        Args:
            config: Detection configuration
        """
        self.config = config or GrokDetectorConfig()
        self.history: list[EpochMetrics] = []

        # Moving average buffers
        self._train_loss_short = deque(maxlen=self.config.short_window)
        self._train_loss_long = deque(maxlen=self.config.long_window)
        self._val_loss_short = deque(maxlen=self.config.short_window)
        self._val_loss_long = deque(maxlen=self.config.long_window)
        self._correlation_history = deque(maxlen=self.config.long_window)

        # State tracking
        self._memorization_start: int | None = None
        self._best_val_loss = float("inf")
        self._best_val_epoch = 0
        self._best_correlation = 0.0
        self._best_correlation_epoch = 0
        self._best_coverage = 0.0
        self._best_coverage_epoch = 0

        # Phase transition detection
        self._phase_transitions: list[tuple[int, TrainingPhase, TrainingPhase]] = []
        self._current_phase = TrainingPhase.WARMUP

    def update(self, metrics: EpochMetrics) -> GrokAnalysis:
        """Update detector with new epoch metrics.

        Args:
            metrics: Metrics from the current epoch

        Returns:
            Analysis of current training state
        """
        self.history.append(metrics)

        # Update moving averages
        self._train_loss_short.append(metrics.train_loss)
        self._train_loss_long.append(metrics.train_loss)
        self._val_loss_short.append(metrics.val_loss)
        self._val_loss_long.append(metrics.val_loss)

        if metrics.correlation is not None:
            self._correlation_history.append(metrics.correlation)

        # Track best metrics
        if metrics.val_loss < self._best_val_loss:
            self._best_val_loss = metrics.val_loss
            self._best_val_epoch = metrics.epoch

        if metrics.correlation is not None and metrics.correlation > self._best_correlation:
            self._best_correlation = metrics.correlation
            self._best_correlation_epoch = metrics.epoch

        if metrics.coverage is not None and metrics.coverage > self._best_coverage:
            self._best_coverage = metrics.coverage
            self._best_coverage_epoch = metrics.epoch

        # Detect current phase
        new_phase = self._detect_phase(metrics)
        if new_phase != self._current_phase:
            self._phase_transitions.append((metrics.epoch, self._current_phase, new_phase))
            self._current_phase = new_phase

        # Compute analysis
        return self._compute_analysis(metrics)

    def _detect_phase(self, metrics: EpochMetrics) -> TrainingPhase:
        """Detect current training phase."""
        epoch = metrics.epoch

        # Early epochs = warmup
        if epoch < self.config.min_epochs_for_detection // 2:
            return TrainingPhase.WARMUP

        # Check for memorization (low train loss, high val loss)
        if metrics.train_loss < self.config.memorization_loss_threshold:
            gap = metrics.val_loss - metrics.train_loss
            if gap > self.config.generalization_gap_threshold:
                if self._memorization_start is None:
                    self._memorization_start = epoch
                return TrainingPhase.MEMORIZATION

        # Check for grokking (sudden improvement after memorization)
        if self._memorization_start is not None:
            epochs_since_memorization = epoch - self._memorization_start
            if epochs_since_memorization > 10:
                # Check for sudden generalization improvement
                if len(self._val_loss_short) >= self.config.short_window:
                    recent_avg = sum(self._val_loss_short) / len(self._val_loss_short)
                    if recent_avg < self._best_val_loss * (1 - self.config.improvement_threshold):
                        return TrainingPhase.GROKKING

        # Check for degradation (performance getting worse)
        if len(self.history) >= self.config.min_epochs_for_detection:
            # Compare current to best
            if metrics.correlation is not None:
                corr_degradation = self._best_correlation - metrics.correlation
                if corr_degradation > self.config.degradation_threshold:
                    return TrainingPhase.DEGRADATION

            # Check coverage degradation
            if metrics.coverage is not None:
                cov_degradation = self._best_coverage - metrics.coverage
                if cov_degradation > self.config.degradation_threshold:
                    return TrainingPhase.DEGRADATION

        # Check for plateau
        if len(self._val_loss_long) >= self.config.long_window:
            long_avg = sum(self._val_loss_long) / len(self._val_loss_long)
            short_avg = sum(self._val_loss_short) / len(self._val_loss_short)
            if abs(long_avg - short_avg) < 0.01:
                return TrainingPhase.PLATEAU

        return TrainingPhase.CONVERGED if epoch > 100 else TrainingPhase.WARMUP

    def _compute_analysis(self, metrics: EpochMetrics) -> GrokAnalysis:
        """Compute comprehensive training analysis."""
        epoch = metrics.epoch

        # Calculate grokking probability
        grok_prob = self._estimate_grokking_probability(metrics)

        # Calculate generalization gap
        gen_gap = metrics.val_loss - metrics.train_loss

        # Determine trend direction
        trend = self._compute_trend()

        # Generate recommendations and warnings
        recommendations, warnings = self._generate_recommendations(metrics, trend)

        # Epochs in memorization
        epochs_in_mem = 0
        if self._memorization_start is not None:
            epochs_in_mem = epoch - self._memorization_start

        return GrokAnalysis(
            current_phase=self._current_phase,
            grokking_probability=grok_prob,
            epochs_in_memorization=epochs_in_mem,
            best_generalization_epoch=self._best_val_epoch,
            current_generalization_gap=gen_gap,
            trend_direction=trend,
            recommendations=recommendations,
            warnings=warnings,
        )

    def _estimate_grokking_probability(self, metrics: EpochMetrics) -> float:
        """Estimate probability that grokking will occur.

        Based on conditions favorable to grokking:
        1. Currently in memorization phase
        2. Training loss very low
        3. Weight decay present
        4. Sufficient remaining training budget
        """
        prob = 0.0

        # In memorization = higher grokking chance
        if self._current_phase == TrainingPhase.MEMORIZATION:
            prob += 0.3

        # Very low training loss
        if metrics.train_loss < 0.01:
            prob += 0.2
        elif metrics.train_loss < 0.1:
            prob += 0.1

        # Long memorization phase (grokking takes time)
        if self._memorization_start is not None:
            epochs_in_mem = metrics.epoch - self._memorization_start
            if epochs_in_mem > 50:
                prob += 0.2
            elif epochs_in_mem > 20:
                prob += 0.1

        # Check for improving trend in later epochs
        if len(self._correlation_history) >= 10:
            recent_corr = list(self._correlation_history)[-10:]
            if len(recent_corr) >= 10:
                early = sum(recent_corr[:5]) / 5
                late = sum(recent_corr[5:]) / 5
                if late > early:
                    prob += 0.2

        return min(prob, 1.0)

    def _compute_trend(self) -> str:
        """Compute current training trend."""
        if len(self._val_loss_short) < 5:
            return "stable"

        recent = list(self._val_loss_short)[-5:]
        if len(recent) < 5:
            return "stable"

        # Linear regression slope
        x_mean = 2.0  # (0+1+2+3+4)/5
        y_mean = sum(recent) / 5

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(5))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        if slope < -0.01:
            return "improving"
        elif slope > 0.01:
            return "degrading"
        return "stable"

    def _generate_recommendations(self, metrics: EpochMetrics, trend: str) -> tuple[list[str], list[str]]:
        """Generate actionable recommendations and warnings."""
        recommendations = []
        warnings = []

        epoch = metrics.epoch

        # Anti-grokking detection (best at start)
        if epoch > 50 and self._best_correlation_epoch < 10:
            warnings.append(
                f"ANTI-GROKKING: Best correlation ({self._best_correlation:.4f}) was at "
                f"epoch {self._best_correlation_epoch}. Model may be degrading."
            )
            recommendations.append("Consider using early stopping or checkpoint from early epochs")
            recommendations.append("Review learning rate schedule - may be too aggressive")

        # Memorization without grokking
        if self._current_phase == TrainingPhase.MEMORIZATION:
            if self._memorization_start and epoch - self._memorization_start > 100:
                warnings.append(
                    f"Extended memorization ({epoch - self._memorization_start} epochs) "
                    "without grokking. May need more training or weight decay."
                )
                recommendations.append("Increase weight decay to encourage grokking")
                recommendations.append("Continue training - grokking can take 10x memorization time")

        # Coverage stagnation
        if metrics.coverage is not None and metrics.coverage < 0.1 and epoch > 50:
            warnings.append(f"Low coverage ({metrics.coverage:.1%}) after {epoch} epochs")
            recommendations.append("Review latent space regularization")
            recommendations.append("Check for posterior collapse indicators")

        # Phase transition detected
        if self._phase_transitions and self._phase_transitions[-1][0] == epoch:
            old_phase, new_phase = self._phase_transitions[-1][1], self._phase_transitions[-1][2]
            if new_phase == TrainingPhase.DEGRADATION:
                warnings.append(f"Phase transition: {old_phase.value} → {new_phase.value}")
                recommendations.append("Consider reverting to best checkpoint")

        # Grokking opportunity
        if self._current_phase == TrainingPhase.MEMORIZATION and trend == "stable":
            recommendations.append("In memorization phase - grokking may occur with continued training")

        return recommendations, warnings

    def get_summary(self) -> dict:
        """Get comprehensive training summary."""
        if not self.history:
            return {"status": "no_data"}

        return {
            "total_epochs": len(self.history),
            "current_phase": self._current_phase.value,
            "best_metrics": {
                "val_loss": {"value": self._best_val_loss, "epoch": self._best_val_epoch},
                "correlation": {
                    "value": self._best_correlation,
                    "epoch": self._best_correlation_epoch,
                },
                "coverage": {"value": self._best_coverage, "epoch": self._best_coverage_epoch},
            },
            "phase_transitions": [{"epoch": e, "from": f.value, "to": t.value} for e, f, t in self._phase_transitions],
            "memorization_started_at": self._memorization_start,
            "final_metrics": self.history[-1].to_dict() if self.history else None,
        }

    def save_report(self, path: Path):
        """Save analysis report to file."""
        report = {
            "summary": self.get_summary(),
            "history": [m.to_dict() for m in self.history],
        }
        path.write_text(json.dumps(report, indent=2))


class LocalComplexityEstimator:
    """Estimates local complexity (LC) for grokking detection.

    Local complexity measures the density of linear regions in a neural network's
    input space. The characteristic "double descent" pattern in LC is a strong
    indicator of impending grokking.

    Reference: Liu et al. (2024) - "Deep Networks Always Grok and Here is Why"
    """

    def __init__(self, n_samples: int = 1000, eps: float = 1e-4):
        """Initialize LC estimator.

        Args:
            n_samples: Number of sample points for estimation
            eps: Epsilon for numerical differentiation
        """
        self.n_samples = n_samples
        self.eps = eps
        self.history: list[float] = []

    @torch.no_grad()
    def estimate(self, model: nn.Module, sample_input: torch.Tensor) -> float:
        """Estimate local complexity of the model.

        Args:
            model: Neural network model
            sample_input: Sample input tensor for probing

        Returns:
            Estimated local complexity score
        """
        model.eval()
        device = next(model.parameters()).device

        # Generate random perturbations
        perturbations = torch.randn(self.n_samples, *sample_input.shape[1:], device=device)
        perturbations = perturbations / perturbations.norm(dim=-1, keepdim=True) * self.eps

        # Compute outputs at perturbed points
        base_out = model(sample_input)
        perturbed_inputs = sample_input.unsqueeze(0) + perturbations.unsqueeze(1)

        # Reshape for batch processing
        batch_shape = perturbed_inputs.shape
        perturbed_inputs = perturbed_inputs.view(-1, *batch_shape[2:])

        perturbed_out = model(perturbed_inputs)
        perturbed_out = perturbed_out.view(self.n_samples, -1, *perturbed_out.shape[1:])

        # Count direction changes (approximates linear regions)
        directions = perturbed_out - base_out.unsqueeze(0)
        direction_signs = torch.sign(directions)

        # Estimate LC as variance in directions
        lc = direction_signs.float().var().item()

        self.history.append(lc)
        return lc

    def detect_double_descent(self) -> tuple[bool, int | None]:
        """Detect double-descent pattern in LC history.

        Returns:
            Tuple of (detected, second_descent_start_epoch)
        """
        if len(self.history) < 20:
            return False, None

        # Smooth the history
        window = 5
        smoothed = []
        for i in range(len(self.history) - window + 1):
            smoothed.append(sum(self.history[i : i + window]) / window)

        # Find local minima and maxima
        minima = []
        maxima = []
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] < smoothed[i - 1] and smoothed[i] < smoothed[i + 1]:
                minima.append(i)
            elif smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                maxima.append(i)

        # Double descent: min -> max -> min pattern
        if len(minima) >= 2 and len(maxima) >= 1:
            # Check if we have min -> max -> min
            first_min = minima[0]
            for m in maxima:
                if m > first_min:
                    for second_min in minima[1:]:
                        if second_min > m:
                            return True, second_min + window // 2
        return False, None


class WeightNormTracker:
    """Tracks weight norm dynamics for grokking analysis.

    During grokking, weight norms typically show:
    1. Initial growth (memorization)
    2. Plateau
    3. Compression (generalization via weight decay)
    """

    def __init__(self):
        """Initialize weight norm tracker."""
        self.history: list[dict[str, float]] = []

    @torch.no_grad()
    def compute_norms(self, model: nn.Module) -> dict[str, float]:
        """Compute weight norms for all layers.

        Args:
            model: Neural network model

        Returns:
            Dictionary of layer names to weight norms
        """
        norms = {}
        total_norm = 0.0

        for name, param in model.named_parameters():
            if param.requires_grad:
                norm = param.norm().item()
                norms[name] = norm
                total_norm += norm**2

        norms["total"] = math.sqrt(total_norm)
        self.history.append(norms)
        return norms

    def detect_compression_phase(self) -> tuple[bool, int | None]:
        """Detect if model is in weight compression phase.

        Returns:
            Tuple of (in_compression, compression_start_epoch)
        """
        if len(self.history) < 20:
            return False, None

        total_norms = [h["total"] for h in self.history]

        # Check for sustained decrease in last 10 epochs
        recent = total_norms[-10:]
        if len(recent) < 10:
            return False, None

        # Fit linear trend
        x_mean = 4.5
        y_mean = sum(recent) / 10
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(10))

        if denominator == 0:
            return False, None

        slope = numerator / denominator

        # Negative slope = compression
        if slope < -0.01:
            # Find when compression started
            for i in range(len(total_norms) - 1, 0, -1):
                if total_norms[i] > total_norms[i - 1]:
                    return True, i
            return True, len(total_norms) - 10

        return False, None


def analyze_training_log(log_path: Path) -> GrokAnalysis:
    """Analyze a training log file for grokking patterns.

    Args:
        log_path: Path to training log file

    Returns:
        Grokking analysis results
    """
    import re

    detector = GrokDetector()

    with open(log_path) as f:
        lines = f.readlines()

    epoch_pattern = re.compile(r"Epoch (\d+)/\d+")
    loss_pattern = re.compile(r"Loss: ([\d.]+)")
    corr_pattern = re.compile(r"3-Adic Correlation.*?: A=([\d.]+) B=([\d.]+)")
    cov_pattern = re.compile(r"Coverage.*?: A=([\d.]+)% B=([\d.]+)%")

    current_epoch = None
    current_loss = None
    current_corr = None
    current_cov = None

    for line in lines:
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            # Save previous epoch if complete
            if current_epoch is not None and current_loss is not None:
                metrics = EpochMetrics(
                    epoch=current_epoch,
                    train_loss=current_loss,
                    val_loss=current_loss,  # Using same as train for now
                    correlation=current_corr,
                    coverage=current_cov,
                )
                detector.update(metrics)

            current_epoch = int(epoch_match.group(1))
            current_corr = None
            current_cov = None

        loss_match = loss_pattern.search(line)
        if loss_match and "Batch" not in line:
            current_loss = float(loss_match.group(1))

        corr_match = corr_pattern.search(line)
        if corr_match:
            current_corr = max(float(corr_match.group(1)), float(corr_match.group(2)))

        cov_match = cov_pattern.search(line)
        if cov_match:
            current_cov = max(float(cov_match.group(1)), float(cov_match.group(2))) / 100

    # Final analysis
    if current_epoch is not None and current_loss is not None:
        metrics = EpochMetrics(
            epoch=current_epoch,
            train_loss=current_loss,
            val_loss=current_loss,
            correlation=current_corr,
            coverage=current_cov,
        )
        return detector.update(metrics)

    return GrokAnalysis(
        current_phase=TrainingPhase.WARMUP,
        grokking_probability=0.0,
        epochs_in_memorization=0,
        best_generalization_epoch=0,
        current_generalization_gap=0.0,
        trend_direction="stable",
    )
