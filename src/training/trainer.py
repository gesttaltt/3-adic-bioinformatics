# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Refactored trainer using modular components.

This trainer delegates to specialized components for:
- Scheduling: TemperatureScheduler, BetaScheduler, LearningRateScheduler
- Monitoring: TrainingMonitor
- Checkpointing: CheckpointManager
- Compilation: torch.compile (TorchInductor) for 1.4-2x speedup

Single responsibility: Orchestrate training loop only.

Inherits from BaseTrainer for:
- Safe division helpers (prevents P0 division-by-zero bugs)
- Validation guards (handles optional val_loader)
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

from ..config.constants import N_TERNARY_OPERATIONS
from ..losses import DualVAELoss, RadialStratificationLoss
from ..models.curriculum import ContinuousCurriculumModule
from .base import BaseTrainer
from .checkpoint_manager import CheckpointManager
from .monitor import TrainingMonitor
from .schedulers import BetaScheduler, LearningRateScheduler, TemperatureScheduler


class TernaryVAETrainer(BaseTrainer):
    """Refactored trainer with single responsibility: orchestrate training loop.

    All scheduling, monitoring, and checkpoint management delegated to components.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: dict[str, Any],
        device: str = "cuda",
    ):
        """Initialize trainer with model and config.

        Args:
            model: DualNeuralVAE model
            config: Training configuration dict
            device: Device to train on
        """
        super().__init__(model, config, device)

        # TorchInductor compilation (PyTorch 2.x)
        self.compiled = False
        compile_config = config.get("torch_compile", {})
        if compile_config.get("enabled", False) and hasattr(torch, "compile"):
            try:
                backend = compile_config.get("backend", "inductor")
                mode = compile_config.get("mode", "default")
                fullgraph = compile_config.get("fullgraph", False)

                self.model = torch.compile(self.model, backend=backend, mode=mode, fullgraph=fullgraph)  # type: ignore
                self.compiled = True
                logger.info(f"torch.compile enabled: backend={backend}, mode={mode}")
            except Exception as e:
                logger.warning(f"torch.compile failed ({e}), falling back to eager mode")
        elif compile_config.get("enabled", False):
            logger.warning("torch.compile requested but not available (PyTorch < 2.0)")

        # Initialize optimizer
        # Ensure self.model is treated as Module for parameters()
        self.optimizer = optim.AdamW(
            self.model.parameters(),  # type: ignore
            lr=config["optimizer"]["lr_start"],
            weight_decay=config["optimizer"].get("weight_decay", 0.0001),
        )

        # Initialize components
        self.temp_scheduler = TemperatureScheduler(
            config,
            config["phase_transitions"]["ultra_exploration_start"],
            config["controller"]["temp_lag"],
        )

        self.beta_scheduler = BetaScheduler(config, config["controller"]["beta_phase_lag"])

        self.lr_scheduler = LearningRateScheduler(config["optimizer"]["lr_schedule"])

        self.monitor = TrainingMonitor(
            eval_num_samples=config["eval_num_samples"],
            tensorboard_dir=config.get("tensorboard_dir"),
            experiment_name=config.get("experiment_name"),
        )

        self.checkpoint_manager = CheckpointManager(Path(config["checkpoint_dir"]), config["checkpoint_freq"])

        # Initialize loss function (with p-adic losses if configured)
        self.loss_fn = DualVAELoss(
            free_bits=config.get("free_bits", 0.0),
            repulsion_sigma=0.5,
            padic_config=config.get("padic_losses", {}),
        )

        # Initialize radial stratification loss (for curriculum learning)
        radial_config = config.get("radial_stratification", {})
        if radial_config.get("enabled", False):
            self.radial_loss_fn = RadialStratificationLoss(
                inner_radius=radial_config.get("inner_radius", 0.1),
                outer_radius=radial_config.get("outer_radius", 0.85),
                max_valuation=radial_config.get("max_valuation", 9),
                valuation_weighting=radial_config.get("valuation_weighting", True),
                loss_type=radial_config.get("loss_type", "smooth_l1"),
            )
            self.radial_loss_weight = radial_config.get("base_weight", 0.3)
        else:
            self.radial_loss_fn = None
            self.radial_loss_weight = 0.0

        # Initialize curriculum module (StateNet v5 controlled)
        curriculum_config = config.get("curriculum", {})
        if curriculum_config.get("enabled", False):
            self.curriculum = ContinuousCurriculumModule(
                initial_tau=curriculum_config.get("initial_tau", 0.0),
                tau_min=curriculum_config.get("tau_min", 0.0),
                tau_max=curriculum_config.get("tau_max", 1.0),
                tau_scale=curriculum_config.get("tau_scale", 0.1),
                momentum=curriculum_config.get("tau_momentum", 0.95),
            ).to(device)
        else:
            self.curriculum = None

        # P1 FIX: Correlation loss - actually add to total loss (not just logging)
        corr_loss_config = config.get("correlation_loss", {})
        self.correlation_loss_enabled = corr_loss_config.get("enabled", False)
        self.correlation_loss_weight = corr_loss_config.get("weight", 0.5)
        self.correlation_loss_warmup = corr_loss_config.get("warmup_epochs", 5)

        # GAP 7 FIX: Exploration boost temp multiplier (set by hyperbolic_trainer)
        # This unifies exploration_boost with curriculum temp modulation
        self.exploration_temp_multiplier = 1.0

        # Cache phase 4 start for model updates
        self.phase_4_start = config["phase_transitions"]["ultra_exploration_start"]

        # Precompute base-3 weights for index computation
        self._base3_weights = torch.tensor([3**i for i in range(9)], dtype=torch.long)

        self._print_init_summary()

    def _print_init_summary(self) -> None:
        """Log initialization summary."""
        total_params = sum(p.numel() for p in self.model.parameters())

        logger.info("=" * 60)
        logger.info("Dual Neural VAE - Base Trainer Initialized")
        logger.info("=" * 60)
        logger.info(f"Total parameters: {total_params:,}")

        if (
            self.config["model"].get("use_statenet", True)
            and hasattr(self.model, "state_net")
            and self.model.state_net is not None
        ):
            statenet_params = sum(p.numel() for p in self.model.state_net.parameters())
            logger.info(f"StateNet parameters: {statenet_params:,} ({statenet_params / total_params * 100:.2f}%)")

        logger.info(f"Device: {self.device}")
        logger.info(f"Gradient balance: {self.config['model'].get('gradient_balance', True)}")
        logger.info(f"Adaptive scheduling: {self.config['model'].get('adaptive_scheduling', True)}")
        logger.info(f"StateNet enabled: {self.config['model'].get('use_statenet', True)}")
        logger.info(f"torch.compile: {'enabled' if self.compiled else 'disabled'}")

    def _check_best(self, losses: dict[str, Any]) -> bool:
        """Check if current losses represent best model."""
        return self.monitor.check_best(losses["loss"])

    def _compute_batch_indices(self, batch_data: torch.Tensor) -> torch.Tensor:
        """Compute operation indices from ternary data."""
        digits = (batch_data + 1).long()
        weights = self._base3_weights.to(batch_data.device)
        indices = (digits * weights).sum(dim=1)
        return indices

    def _update_model_parameters(self, epoch: int) -> None:
        """Update model's adaptive parameters for current epoch."""
        # Cast to Any to avoid mypy issues with torch.compile return type
        model: Any = self.model

        model.epoch = epoch
        model.rho = model.compute_phase_scheduled_rho(epoch, self.phase_4_start)

        if self.curriculum is not None:
            tau = self.curriculum.get_tau().item()
            rho_curriculum_factor = 1.0 - 0.3 * tau
            model.rho = model.rho * rho_curriculum_factor

        model.lambda3 = model.compute_cyclic_lambda3(epoch, period=30)
        grad_ratio = (model.grad_norm_B_ema / (model.grad_norm_A_ema + 1e-8)).item()
        model.update_adaptive_ema_momentum(grad_ratio)

        if len(self.monitor.coverage_A_history) > 0:
            coverage_a = self.monitor.coverage_A_history[-1]
            coverage_b = self.monitor.coverage_B_history[-1]
            model.update_adaptive_lambdas(grad_ratio, coverage_a, coverage_b)

    def _apply_structure_loss(
        self,
        losses: dict[str, Any],
        outputs: dict[str, Any],
        batch_indices: torch.Tensor,
    ) -> None:
        """Apply curriculum-controlled radial stratification loss."""
        radial_loss = torch.tensor(0.0, device=self.device)
        curriculum_tau = 0.0

        if self.radial_loss_fn is None:
            return

        # Compute radial loss for both VAEs
        radial_loss_a = self.radial_loss_fn(outputs["z_A"], batch_indices)
        radial_loss_b = self.radial_loss_fn(outputs["z_B"], batch_indices)
        radial_loss = radial_loss_a + radial_loss_b

        # Get ranking loss from p-adic losses (if available)
        ranking_loss = torch.tensor(0.0, device=self.device)
        if "padic_ranking_A" in losses and torch.is_tensor(losses["padic_ranking_A"]):
            ranking_loss = losses["padic_ranking_A"] + losses.get("padic_ranking_B", 0.0)

        # Use curriculum to blend radial and ranking losses
        if self.curriculum is not None:
            curriculum_tau = self.curriculum.get_tau().item()
            structure_loss = self.curriculum.modulate_losses(radial_loss, ranking_loss)
            losses["loss"] = losses["loss"] + self.radial_loss_weight * structure_loss
        else:
            losses["loss"] = losses["loss"] + self.radial_loss_weight * radial_loss

        # Log metrics
        losses["radial_stratification_loss"] = radial_loss.item()
        losses["curriculum_tau"] = curriculum_tau
        radial_wt, ranking_wt = (1 - curriculum_tau, curriculum_tau) if self.curriculum else (1.0, 0.0)
        losses["curriculum_radial_weight"] = radial_wt
        losses["curriculum_ranking_weight"] = ranking_wt

    def _apply_optimization_feedback(
        self,
        losses: dict[str, Any],
        epoch_losses: dict[str, Any],
        lr_scheduled: float,
        outputs: dict[str, Any],
    ) -> None:
        """Apply StateNet corrections or standard LR scheduling."""
        if not self.model.use_statenet:
            # Standard scheduling
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr_scheduled
            return

        # Get latest coverage
        cov_a = self.monitor.coverage_A_history[-1] if self.monitor.coverage_A_history else 0
        cov_b = self.monitor.coverage_B_history[-1] if self.monitor.coverage_B_history else 0

        # Compute cross-VAE correlation r_AB (complementarity metric)
        z_a = outputs["z_A"]
        z_b = outputs["z_B"]
        z_a_centered = z_a - z_a.mean(dim=0, keepdim=True)
        z_b_centered = z_b - z_b.mean(dim=0, keepdim=True)
        r_ab_raw = torch.nn.functional.cosine_similarity(
            z_a_centered.flatten(start_dim=1).mean(dim=0),
            z_b_centered.flatten(start_dim=1).mean(dim=0),
            dim=0,
        )
        r_ab = ((r_ab_raw + 1.0) / 2.0).item()

        # StateNet v5
        if self.curriculum is not None and hasattr(self.model, "apply_statenet_v5_corrections"):
            corrections = self.model.apply_statenet_v5_corrections(
                lr_scheduled,
                (losses["H_A"].item() if torch.is_tensor(losses["H_A"]) else losses["H_A"]),
                (losses["H_B"].item() if torch.is_tensor(losses["H_B"]) else losses["H_B"]),
                losses["kl_A"].item(),
                losses["kl_B"].item(),
                (self.model.grad_norm_B_ema / (self.model.grad_norm_A_ema + 1e-8)).item(),
                coverage_A=cov_a,
                coverage_B=cov_b,
                r_AB=r_ab,
                radial_loss=losses.get("radial_stratification_loss", 0.0),
                curriculum_tau=losses.get("curriculum_tau", 0.0),
            )
            self.curriculum.update_tau(corrections["delta_curriculum"])
        else:
            # v4 Fallback
            corrected_lr, *deltas = self.model.apply_statenet_corrections(
                lr_scheduled,
                losses["H_A"].item(),
                losses["H_B"].item(),
                losses["kl_A"].item(),
                losses["kl_B"].item(),
                (self.model.grad_norm_B_ema / (self.model.grad_norm_A_ema + 1e-8)).item(),
                coverage_A=cov_a,
                coverage_B=cov_b,
            )
            corrections = {
                "corrected_lr": corrected_lr,
                "delta_lr": deltas[0],
                "delta_lambda1": deltas[1],
                "delta_lambda2": deltas[2],
                "delta_lambda3": deltas[3],
            }

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = corrections["corrected_lr"]

        epoch_losses.update(corrections)
        if "r_AB" not in epoch_losses:
            epoch_losses["r_AB"] = r_ab

    def _process_batch_metrics(
        self,
        losses: dict[str, Any],
        epoch_losses: dict[str, float],
        batch_idx: int,
        total_batches: int,
        log_interval: int,
    ) -> None:
        """Accummulate losses and log to tensorboard."""
        # Log to TensorBoard
        batch_loss = losses["loss"].item() if torch.is_tensor(losses["loss"]) else losses["loss"]
        self.monitor.log_batch(
            epoch=self.epoch,
            batch_idx=batch_idx,
            total_batches=total_batches,
            loss=batch_loss,
            ce_A=(losses["ce_A"].item() if torch.is_tensor(losses["ce_A"]) else losses["ce_A"]),
            ce_B=(losses["ce_B"].item() if torch.is_tensor(losses["ce_B"]) else losses["ce_B"]),
            kl_A=(losses["kl_A"].item() if torch.is_tensor(losses["kl_A"]) else losses["kl_A"]),
            kl_B=(losses["kl_B"].item() if torch.is_tensor(losses["kl_B"]) else losses["kl_B"]),
            log_interval=log_interval,
        )

        # Accumulate
        for key, val in losses.items():
            if isinstance(val, torch.Tensor):
                epoch_losses[key] += val.item()
            else:
                epoch_losses[key] += val

    def train_epoch(self, train_loader: DataLoader, log_interval: int = 10) -> dict[str, Any]:
        """Train for one epoch with batch-level TensorBoard logging."""
        model: Any = self.model  # Cast to avoid mypy errors
        model.train()
        self._update_model_parameters(self.epoch)

        # Get scheduled parameters
        temp_a = self.temp_scheduler.get_temperature(self.epoch, "A")
        temp_b = self.temp_scheduler.get_temperature(self.epoch, "B")
        beta_a = self.beta_scheduler.get_beta(self.epoch, "A")
        beta_b = self.beta_scheduler.get_beta(self.epoch, "B")
        lr_scheduled = self.lr_scheduler.get_lr(self.epoch)

        if self.curriculum is not None:
            tau = self.curriculum.get_tau().item()
            beta_a = beta_a * (1.0 + 0.15 * tau)
            beta_b = beta_b * (1.0 + 0.30 * tau)
            temp_a = temp_a * (1.0 - 0.10 * tau)
            temp_b = temp_b * (1.0 - 0.20 * tau)

        if self.exploration_temp_multiplier != 1.0:
            temp_a = temp_a * self.exploration_temp_multiplier
            temp_b = temp_b * self.exploration_temp_multiplier

        entropy_weight = self.config["vae_b"]["entropy_weight"]
        repulsion_weight = self.config["vae_b"]["repulsion_weight"]

        epoch_losses: dict[str, float] = defaultdict(float)
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, tuple):
                batch_data, batch_indices = batch
            else:
                batch_data = batch.to(self.device)
                batch_indices = self._compute_batch_indices(batch_data)

            # Forward pass
            outputs = model(batch_data, temp_a, temp_b, beta_a, beta_b)

            # Compute losses
            losses = self.loss_fn(
                batch_data,
                outputs,
                model.lambda1,
                model.lambda2,
                model.lambda3,
                entropy_weight,
                repulsion_weight,
                model.grad_norm_A_ema,
                model.grad_norm_B_ema,
                model.gradient_balance,
                model.training,
                batch_indices=batch_indices,
            )

            losses["rho"] = model.rho
            losses["phase"] = model.current_phase

            # Helper Calls
            self._apply_structure_loss(losses, outputs, batch_indices)

            if batch_idx == 0:
                self._apply_optimization_feedback(losses, epoch_losses, lr_scheduled, outputs)

            # Correlation loss
            if self.correlation_loss_enabled and self.epoch >= self.correlation_loss_warmup:
                if self.monitor.correlation_hyp_history:
                    cached_corr = self.monitor.correlation_hyp_history[-1]
                    correlation_loss_term = -self.correlation_loss_weight * cached_corr
                    losses["loss"] = losses["loss"] + correlation_loss_term
                    losses["correlation_loss_applied"] = correlation_loss_term

            # Backward and optimize
            self.optimizer.zero_grad()
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config["grad_clip"])
            model.update_gradient_norms()
            self.optimizer.step()

            # Process metrics
            self._process_batch_metrics(losses, epoch_losses, batch_idx, num_batches, log_interval)

        # Average losses
        if num_batches > 0:
            for key in epoch_losses:
                # Do not average these accumulated deltas or boosts
                if key not in [
                    "lr_corrected",
                    "delta_lr",
                    "delta_lambda1",
                    "delta_lambda2",
                    "delta_lambda3",
                    "delta_curriculum",
                    "delta_sigma",
                    "delta_curvature",
                    "r_AB",
                    "statenet_lr_boost",
                ]:
                    epoch_losses[key] /= num_batches

        epoch_losses["temp_A"] = temp_a
        epoch_losses["temp_B"] = temp_b
        epoch_losses["beta_A"] = beta_a
        epoch_losses["beta_B"] = beta_b
        epoch_losses["lr_scheduled"] = lr_scheduled
        epoch_losses["grad_ratio"] = (model.grad_norm_B_ema / (model.grad_norm_A_ema + 1e-8)).item()
        epoch_losses["ema_momentum"] = model.grad_ema_momentum

        if hasattr(model, "update_loss_plateau_detection"):
            boost = model.update_loss_plateau_detection(epoch_losses["loss"])
            epoch_losses["statenet_lr_boost"] = boost

        return epoch_losses

    def validate(self, val_loader: DataLoader) -> dict[str, Any]:
        """Validation pass."""
        model: Any = self.model
        model.eval()
        epoch_losses: dict[str, float] = defaultdict(float)
        num_batches = 0

        temp_a = self.temp_scheduler.get_temperature(self.epoch, "A")
        temp_b = self.temp_scheduler.get_temperature(self.epoch, "B")
        beta_a = self.beta_scheduler.get_beta(self.epoch, "A")
        beta_b = self.beta_scheduler.get_beta(self.epoch, "B")
        entropy_weight = self.config["vae_b"]["entropy_weight"]
        repulsion_weight = self.config["vae_b"]["repulsion_weight"]

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, tuple):
                    batch_data, batch_indices = batch
                else:
                    batch_data = batch.to(self.device)
                    batch_indices = self._compute_batch_indices(batch_data)

                outputs = model(batch_data, temp_a, temp_b, beta_a, beta_b)

                losses = self.loss_fn(
                    batch_data,
                    outputs,
                    model.lambda1,
                    model.lambda2,
                    model.lambda3,
                    entropy_weight,
                    repulsion_weight,
                    model.grad_norm_A_ema,
                    model.grad_norm_B_ema,
                    model.gradient_balance,
                    False,
                    batch_indices=batch_indices,
                )

                losses["rho"] = model.rho
                losses["phase"] = model.current_phase

                for key, val in losses.items():
                    if isinstance(val, torch.Tensor):
                        epoch_losses[key] += val.item()
                    else:
                        epoch_losses[key] += val

                num_batches += 1

        if num_batches > 0:
            for key in epoch_losses:
                epoch_losses[key] /= num_batches

        return epoch_losses

    def train(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> None:
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("Starting Dual Neural VAE Training")
        logger.info("=" * 60)

        # Cast to Any to avoid mypy issues with torch.compile return type
        model: Any = self.model
        total_epochs = self.config["total_epochs"]

        for epoch in range(total_epochs):
            self.epoch = epoch

            train_losses = self.train_epoch(train_loader)

            if val_loader is not None:
                val_losses = self.validate(val_loader)
                is_best = self.monitor.check_best(val_losses["loss"])
            else:
                val_losses = train_losses
                is_best = self.monitor.check_best(train_losses["loss"])

            unique_a, cov_a = self.monitor.evaluate_coverage(model, self.config["eval_num_samples"], self.device, "A")
            unique_b, cov_b = self.monitor.evaluate_coverage(model, self.config["eval_num_samples"], self.device, "B")

            self.monitor.update_histories(train_losses["H_A"], train_losses["H_B"], unique_a, unique_b)

            self.monitor.log_epoch(
                epoch,
                total_epochs,
                train_losses,
                val_losses,
                unique_a,
                cov_a,
                unique_b,
                cov_b,
                is_best,
                model.use_statenet,
                model.grad_balance_achieved,
            )

            self.monitor.log_tensorboard(
                epoch,
                train_losses,
                val_losses,
                unique_a,
                unique_b,
                cov_a,
                cov_b,
            )

            if epoch % 10 == 0:
                self.monitor.log_histograms(epoch, model)

            embedding_interval = self.config.get("embedding_interval", 50)
            if embedding_interval > 0 and epoch % embedding_interval == 0:
                self.monitor.log_manifold_embedding(
                    model,
                    epoch,
                    self.device,
                    n_samples=self.config.get("embedding_n_samples", 5000),
                )

            metadata = {
                **self.monitor.get_metadata(),
                "lambda1": model.lambda1,
                "lambda2": model.lambda2,
                "lambda3": model.lambda3,
                "rho": model.rho,
                "phase": model.current_phase,
                "grad_balance_achieved": model.grad_balance_achieved,
                "grad_norm_A_ema": model.grad_norm_A_ema.item(),
                "grad_norm_B_ema": model.grad_norm_B_ema.item(),
                "grad_ema_momentum": model.grad_ema_momentum,
                "statenet_enabled": model.use_statenet,
            }

            if model.use_statenet:
                metadata["statenet_corrections"] = model.statenet_corrections

            self.checkpoint_manager.save_checkpoint(epoch, model, self.optimizer, metadata, is_best)

            if self.monitor.should_stop(self.config["patience"]):
                logger.warning(f"Early stopping triggered (patience={self.config['patience']})")
                break

            coverage_plateau_patience = self.config.get("coverage_plateau_patience", 100)
            coverage_plateau_delta = self.config.get("coverage_plateau_min_delta", 0.0005)
            if self.monitor.has_coverage_plateaued(coverage_plateau_patience, coverage_plateau_delta):
                current_cov = max(
                    (self.monitor.coverage_A_history[-1] if self.monitor.coverage_A_history else 0),
                    (self.monitor.coverage_B_history[-1] if self.monitor.coverage_B_history else 0),
                )
                logger.info(
                    f"Coverage plateaued at {current_cov / N_TERNARY_OPERATIONS * 100:.2f}% (no improvement for {coverage_plateau_patience} epochs)"
                )
                break

        self.monitor.print_training_summary()
        self.monitor.close()
