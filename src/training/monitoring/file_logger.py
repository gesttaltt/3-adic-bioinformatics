# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""File and console logging for training.

This module handles persistent logging:
- File logging with timestamps
- Console logging (clean output)
- Training progress messages
- Epoch summaries
- Training completion summaries

Single responsibility: Log output management only.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from src.config.constants import N_TERNARY_OPERATIONS

# Module-level logger for fallback
_module_logger = logging.getLogger(__name__)


class FileLogger:
    """Handles file and console logging for training.

    Manages log file creation, formatting, and message output
    for both file and console destinations.

    Attributes:
        logger: Python logger instance
        experiment_name: Name of the experiment
    """

    def __init__(
        self,
        log_dir: str | None = "logs",
        experiment_name: str = "experiment",
        log_to_file: bool = True,
    ):
        """Initialize file logger.

        Args:
            log_dir: Directory for log files
            experiment_name: Name for this experiment
            log_to_file: Whether to enable file logging
        """
        self.experiment_name = experiment_name
        self.logger: logging.Logger | None = None

        if log_to_file and log_dir:
            self.logger = self._setup_file_logging(log_dir, experiment_name)

    def _setup_file_logging(
        self,
        log_dir: str,
        experiment_name: str,
    ) -> logging.Logger:
        """Setup persistent file logging.

        Args:
            log_dir: Directory for log files
            experiment_name: Experiment name for log file

        Returns:
            Configured logger instance
        """
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        log_file = log_path / f"training_{experiment_name}.log"

        logger = logging.getLogger(f"ternary_vae_{experiment_name}")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        # File handler with timestamps
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            "%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)

        # Console handler without timestamps
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info(f"Logging to: {log_file}")
        return logger

    def log(self, message: str) -> None:
        """Log message to both file and console.

        Args:
            message: Message to log
        """
        if self.logger:
            self.logger.info(message)
        else:
            _module_logger.info(message)

    def log_batch(
        self,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        loss: float,
        log_interval: int = 10,
    ) -> None:
        """Log batch progress message.

        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            total_batches: Total batches in epoch
            loss: Current batch loss
            log_interval: Log every N batches
        """
        if batch_idx % log_interval == 0 or batch_idx == total_batches - 1:
            progress = (batch_idx + 1) / total_batches * 100
            self.log(f"  [Epoch {epoch}] Batch {batch_idx + 1}/{total_batches} ({progress:.0f}%) | Loss: {loss:.4f}")

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_losses: dict[str, Any],
        val_losses: dict[str, Any],
        unique_A: int,
        cov_A: float,
        unique_B: int,
        cov_B: float,
        is_best: bool,
        best_val_loss: float,
        use_statenet: bool,
        grad_balance_achieved: bool,
    ) -> None:
        """Log epoch results.

        Args:
            epoch: Current epoch
            total_epochs: Total epochs
            train_losses: Training losses dict
            val_losses: Validation losses dict
            unique_A: VAE-A unique operations
            cov_A: VAE-A coverage percentage
            unique_B: VAE-B unique operations
            cov_B: VAE-B coverage percentage
            is_best: Whether this is best validation loss
            best_val_loss: Best validation loss so far
            use_statenet: Whether StateNet is enabled
            grad_balance_achieved: Whether gradient balance is achieved
        """
        self.log(f"\nEpoch {epoch}/{total_epochs}")
        self.log(f"  Loss: Train={train_losses['loss']:.4f} Val={val_losses['loss']:.4f}")
        self.log(f"  VAE-A: CE={train_losses['ce_A']:.4f} KL={train_losses['kl_A']:.4f} H={train_losses['H_A']:.3f}")
        self.log(f"  VAE-B: CE={train_losses['ce_B']:.4f} KL={train_losses['kl_B']:.4f} H={train_losses['H_B']:.3f}")
        self.log(
            f"  Weights: l1={train_losses['lambda1']:.3f} "
            f"l2={train_losses['lambda2']:.3f} "
            f"l3={train_losses['lambda3']:.3f}"
        )
        self.log(
            f"  Phase {train_losses['phase']}: "
            f"rho={train_losses['rho']:.3f} "
            f"(balance: {'Y' if grad_balance_achieved else 'N'})"
        )
        self.log(f"  Grad: ratio={train_losses['grad_ratio']:.3f} EMA_a={train_losses['ema_momentum']:.2f}")
        self.log(
            f"  Temp: A={train_losses['temp_A']:.3f} "
            f"B={train_losses['temp_B']:.3f} | "
            f"beta: A={train_losses['beta_A']:.3f} "
            f"B={train_losses['beta_B']:.3f}"
        )

        if use_statenet and "lr_corrected" in train_losses:
            self.log(
                f"  LR: {train_losses['lr_scheduled']:.6f} -> "
                f"{train_losses['lr_corrected']:.6f} "
                f"(d={train_losses.get('delta_lr', 0):+.3f})"
            )
            self.log(
                f"  StateNet: dl1={train_losses.get('delta_lambda1', 0):+.3f} "
                f"dl2={train_losses.get('delta_lambda2', 0):+.3f} "
                f"dl3={train_losses.get('delta_lambda3', 0):+.3f}"
            )
        else:
            self.log(f"  LR: {train_losses['lr_scheduled']:.6f}")

        self.log(f"  Coverage: A={unique_A} ({cov_A:.2f}%) | B={unique_B} ({cov_B:.2f}%)")

        # p-Adic losses
        self._log_padic_losses(train_losses)

        if is_best:
            self.log(f"  Best val loss: {best_val_loss:.4f}")

    def _log_padic_losses(self, train_losses: dict[str, Any]) -> None:
        """Log p-adic loss components.

        Args:
            train_losses: Training losses dict
        """
        has_padic = (
            train_losses.get("padic_metric_A", 0) > 0
            or train_losses.get("padic_ranking_A", 0) > 0
            or train_losses.get("padic_norm_A", 0) > 0
        )

        if not has_padic:
            return

        parts = []
        if train_losses.get("padic_metric_A", 0) > 0:
            parts.append(
                f"metric={train_losses.get('padic_metric_A', 0):.4f}/{train_losses.get('padic_metric_B', 0):.4f}"
            )
        if train_losses.get("padic_ranking_A", 0) > 0:
            parts.append(
                f"rank={train_losses.get('padic_ranking_A', 0):.4f}/{train_losses.get('padic_ranking_B', 0):.4f}"
            )
        if train_losses.get("padic_norm_A", 0) > 0:
            parts.append(f"norm={train_losses.get('padic_norm_A', 0):.4f}/{train_losses.get('padic_norm_B', 0):.4f}")
        self.log(f"  p-Adic: {' '.join(parts)}")

    def log_epoch_summary(
        self,
        epoch: int,
        total_epochs: int,
        loss: float,
        cov_A: float,
        cov_B: float,
        corr_A_hyp: float,
        corr_B_hyp: float,
        corr_A_euc: float,
        corr_B_euc: float,
        mean_radius_A: float,
        mean_radius_B: float,
        ranking_weight: float,
        best_coverage: float,
        best_corr_hyp: float,
        coverage_evaluated: bool = True,
        correlation_evaluated: bool = True,
        hyp_kl_A: float = 0.0,
        hyp_kl_B: float = 0.0,
        centroid_loss: float = 0.0,
        radial_loss: float = 0.0,
        homeostatic_metrics: dict[str, float] | None = None,
    ) -> None:
        """Log comprehensive epoch summary (v5.10 hyperbolic).

        Args:
            epoch: Current epoch
            total_epochs: Total epochs
            loss: Total loss
            cov_A: VAE-A coverage percentage
            cov_B: VAE-B coverage percentage
            corr_A_hyp: VAE-A hyperbolic correlation
            corr_B_hyp: VAE-B hyperbolic correlation
            corr_A_euc: VAE-A Euclidean correlation
            corr_B_euc: VAE-B Euclidean correlation
            mean_radius_A: VAE-A mean latent radius
            mean_radius_B: VAE-B mean latent radius
            ranking_weight: Current ranking loss weight
            best_coverage: Best coverage seen
            best_corr_hyp: Best hyperbolic correlation seen
            coverage_evaluated: Whether coverage was freshly evaluated
            correlation_evaluated: Whether correlation was freshly evaluated
            hyp_kl_A: Hyperbolic KL for VAE-A
            hyp_kl_B: Hyperbolic KL for VAE-B
            centroid_loss: Frechet centroid loss
            radial_loss: Radial hierarchy loss
            homeostatic_metrics: Dict of homeostatic adaptation metrics
        """
        cov_status = "FRESH" if coverage_evaluated else "cached"
        corr_status = "FRESH" if correlation_evaluated else "cached"

        self.log(f"\nEpoch {epoch}/{total_epochs}")
        self.log(f"  Loss: {loss:.4f} | Ranking Weight: {ranking_weight:.3f}")
        self.log(f"  Coverage [{cov_status}]: A={cov_A:.1f}% B={cov_B:.1f}% (best={best_coverage:.1f}%)")
        self.log(
            f"  3-Adic Correlation [{corr_status}] (Hyp): "
            f"A={corr_A_hyp:.3f} B={corr_B_hyp:.3f} "
            f"(best={best_corr_hyp:.3f})"
        )

        if correlation_evaluated:
            self.log(f"  3-Adic Correlation (Euclidean): A={corr_A_euc:.3f} B={corr_B_euc:.3f}")

        self.log(f"  Mean Radius: A={mean_radius_A:.3f} B={mean_radius_B:.3f}")

        if radial_loss > 0:
            self.log(f"  Radial Loss: {radial_loss:.4f}")

        if hyp_kl_A > 0:
            self.log(f"  Hyperbolic KL: A={hyp_kl_A:.4f} B={hyp_kl_B:.4f}")

        if centroid_loss > 0:
            self.log(f"  Centroid Loss: {centroid_loss:.4f}")

        if homeostatic_metrics:
            if "prior_sigma_A" in homeostatic_metrics:
                self.log(
                    f"  Homeostatic Sigma: "
                    f"A={homeostatic_metrics['prior_sigma_A']:.3f} "
                    f"B={homeostatic_metrics['prior_sigma_B']:.3f}"
                )
            if "prior_curvature_A" in homeostatic_metrics:
                self.log(
                    f"  Homeostatic Curvature: "
                    f"A={homeostatic_metrics['prior_curvature_A']:.3f} "
                    f"B={homeostatic_metrics['prior_curvature_B']:.3f}"
                )

    def print_training_summary(
        self,
        best_val_loss: float,
        best_corr_hyp: float,
        best_corr_euc: float,
        best_coverage: float,
        coverage_A_history: list,
        coverage_B_history: list,
    ) -> None:
        """Print training completion summary.

        Args:
            best_val_loss: Best validation loss
            best_corr_hyp: Best hyperbolic correlation
            best_corr_euc: Best Euclidean correlation
            best_coverage: Best coverage percentage
            coverage_A_history: VAE-A coverage history
            coverage_B_history: VAE-B coverage history
        """
        self.log(f"\n{'=' * 80}")
        self.log("Training Complete")
        self.log(f"{'=' * 80}")
        self.log(f"Best val loss: {best_val_loss:.4f}")
        self.log(f"Best hyperbolic correlation: {best_corr_hyp:.4f}")
        self.log(f"Best Euclidean correlation: {best_corr_euc:.4f}")
        self.log(f"Best coverage: {best_coverage:.2f}%")

        if coverage_A_history:
            final_cov_A = coverage_A_history[-1]
            final_cov_B = coverage_B_history[-1]
            self.log(f"Final Coverage: A={final_cov_A} ({final_cov_A / N_TERNARY_OPERATIONS * 100:.2f}%)")
            self.log(f"                B={final_cov_B} ({final_cov_B / N_TERNARY_OPERATIONS * 100:.2f}%)")

        self.log("Target: r > 0.99, coverage > 99.7%")


__all__ = ["FileLogger"]
