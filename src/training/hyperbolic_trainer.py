# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Hyperbolic VAE Trainer for v5.10 Pure Hyperbolic Geometry.

This trainer implements pure hyperbolic geometry training with homeostatic
adaptation for the Dual Neural VAE. It wraps a base trainer and adds:

1. Hyperbolic Prior (wrapped normal on Poincare ball)
2. Hyperbolic Reconstruction Loss (radius-weighted)
3. Hyperbolic Centroid Loss (Frechet mean clustering)
4. Continuous feedback for ranking weight adaptation
5. Homeostatic parameter adaptation
6. Unified TensorBoard observability (batch + epoch level)

Key principle: No Euclidean contamination - all geometry is hyperbolic.

Single responsibility: Hyperbolic training orchestration and observability.
"""

from typing import Any

import torch

from ..config.constants import N_TERNARY_OPERATIONS
from ..core.metrics import compute_ranking_correlation_hyperbolic
from ..losses.consequence_predictor import ConsequencePredictor, evaluate_addition_accuracy
from ..losses.hyperbolic_prior import HomeostaticHyperbolicPrior
from ..losses.hyperbolic_recon import HomeostaticReconLoss, HyperbolicCentroidLoss
from ..losses.padic import PAdicRankingLossHyperbolic
from ..losses.radial_stratification import RadialStratificationLoss
from ..models.curriculum import ContinuousCurriculumModule
from .feedback import (
    ContinuousFeedbackController,
    CorrelationEarlyStop,
    ExplorationBoostController,
)
from .monitor import TrainingMonitor


class HyperbolicVAETrainer:
    """Trainer for v5.10 Pure Hyperbolic geometry with homeostatic adaptation.

    This trainer wraps a base TernaryVAETrainer and adds hyperbolic geometry
    losses and metrics. It implements continuous feedback for adaptive ranking
    weight modulation based on coverage.

    Args:
        base_trainer: Base TernaryVAETrainer instance
        model: DualNeuralVAEV5_10 model
        device: Device to train on
        config: Training configuration dict
        monitor: Optional TrainingMonitor for logging
    """

    def __init__(
        self,
        base_trainer,
        model: torch.nn.Module,
        device: str,
        config: dict[str, Any],
        monitor: TrainingMonitor | None = None,
    ):
        self.base_trainer = base_trainer
        self.model = model
        self.device = device
        self.config = config
        self.monitor = monitor or base_trainer.monitor
        self.total_epochs = config.get("total_epochs", 100)

        # Explicit type hints for optional components to satisfy mypy
        self.ranking_loss_hyp: Any | None = None
        self.hyperbolic_prior_A: Any | None = None
        self.hyperbolic_prior_B: Any | None = None
        self.hyperbolic_recon_A: Any | None = None
        self.hyperbolic_recon_B: Any | None = None
        self.centroid_loss: Any | None = None
        self.radial_stratification_A: Any | None = None
        self.radial_stratification_B: Any | None = None
        self.curriculum: Any | None = None

        # Observability config
        self.histogram_interval = config.get("histogram_interval", 10)
        self.embedding_interval = config.get("embedding_interval", 50)
        self.embedding_n_samples = config.get("embedding_n_samples", 5000)
        self.log_interval = config.get("log_interval", 10)

        # Initialize feedback controllers (extracted to dedicated modules)
        self._init_feedback_controllers(config)

        # Initialize hyperbolic loss modules
        self._init_hyperbolic_losses(config, device)

        # Initialize v5.10.1 components: radial stratification, curriculum, consequence
        self._init_radial_stratification(config, device)
        self._init_curriculum(config, device)
        self._init_consequence_predictor(config, device)

        # P2 FIX: Initialize correlation loss config
        self._init_correlation_loss(config)

        # Initialize evaluation intervals and caching
        self._init_evaluation_config(config)

        # Tracking histories
        self.correlation_history_hyp: list[float] = []
        self.correlation_history_euc: list[float] = []
        self.coverage_history: list[float] = []
        self.ranking_weight_history: list[float] = []
        self.radial_loss_history: list[float] = []
        self.homeostatic_history: list[dict[str, float]] = []

        # Best metrics
        self.best_corr_hyp = 0.0
        self.best_corr_euc = 0.0
        self.best_coverage = 0.0

    def _init_feedback_controllers(self, config: dict[str, Any]) -> None:
        """Initialize feedback controllers for adaptive training.

        Creates three controllers:
        - ContinuousFeedbackController: Coverage-based ranking weight
        - CorrelationEarlyStop: Correlation-based early stopping
        - ExplorationBoostController: Coverage stall detection
        """
        # Continuous feedback for ranking weight adaptation
        feedback_config = config.get("continuous_feedback", {})
        self.continuous_feedback = ContinuousFeedbackController.from_dict(feedback_config)

        # Correlation-based early stopping
        corr_config = config.get("correlation_feedback", {})
        self.correlation_early_stop = CorrelationEarlyStop.from_dict(corr_config)

        # Exploration boost for coverage stall
        boost_config = config.get("exploration_boost", {})
        self.exploration_boost = ExplorationBoostController.from_dict(boost_config)

    def _init_correlation_loss(self, config: dict[str, Any]) -> None:
        """P2 FIX: Initialize correlation loss term."""
        corr_loss_config = config.get("correlation_loss", {})
        self.correlation_loss_enabled = corr_loss_config.get("enabled", False)

        if self.correlation_loss_enabled:
            self.correlation_loss_weight = corr_loss_config.get("weight", 0.5)
            self.correlation_loss_warmup = corr_loss_config.get("warmup_epochs", 5)
            self.correlation_loss_use_cached = corr_loss_config.get("use_cached", True)
        else:
            self.correlation_loss_weight = 0.0
            self.correlation_loss_warmup = 5
            self.correlation_loss_use_cached = True

    def _init_hyperbolic_losses(self, config: dict[str, Any], device: str) -> None:
        """Initialize hyperbolic loss modules based on config."""
        padic_config = config.get("padic_losses", {})

        # PAdicRankingLossHyperbolic (v5.9 foundation)
        if padic_config.get("enable_ranking_loss_hyperbolic", False):
            hyp_config = padic_config.get("ranking_hyperbolic", {})
            self.ranking_loss_hyp = PAdicRankingLossHyperbolic(
                base_margin=hyp_config.get("base_margin", 0.05),
                margin_scale=hyp_config.get("margin_scale", 0.15),
                n_triplets=hyp_config.get("n_triplets", 500),
                hard_negative_ratio=hyp_config.get("hard_negative_ratio", 0.5),
                curvature=hyp_config.get("curvature", 2.0),
                radial_weight=hyp_config.get("radial_weight", 0.4),
                max_norm=hyp_config.get("max_norm", 0.95),
            )
            self.max_norm = hyp_config.get("max_norm", 0.95)
            self.curvature = hyp_config.get("curvature", 2.0)
        else:
            self.ranking_loss_hyp = None
            self.max_norm = 0.95
            self.curvature = 2.0

        # v5.10 Hyperbolic modules
        hyp_v10 = padic_config.get("hyperbolic_v10", {})

        # Hyperbolic Prior (replaces Gaussian KL)
        if hyp_v10.get("use_hyperbolic_prior", False):
            prior_config = hyp_v10.get("prior", {})
            # P0 FIX: Wire ALL homeostatic params from config (was only 4/12)
            self.hyperbolic_prior_A = HomeostaticHyperbolicPrior(
                latent_dim=prior_config.get("latent_dim", 16),
                curvature=prior_config.get("curvature", 2.2),
                prior_sigma=prior_config.get("prior_sigma", 1.0),
                max_norm=prior_config.get("max_norm", 0.95),
                # P0 FIX: New params wired from config
                sigma_min=prior_config.get("sigma_min", 0.8),
                sigma_max=prior_config.get("sigma_max", 1.2),
                curvature_min=prior_config.get("curvature_min", 2.0),
                curvature_max=prior_config.get("curvature_max", 2.5),
                adaptation_rate=prior_config.get("adaptation_rate", 0.005),
                ema_alpha=prior_config.get("ema_alpha", 0.05),
                kl_target=prior_config.get("kl_target", 50.0),
                target_radius=prior_config.get("target_radius", 0.5),
            ).to(device)
            self.hyperbolic_prior_B = HomeostaticHyperbolicPrior(
                latent_dim=prior_config.get("latent_dim", 16),
                curvature=prior_config.get("curvature", 2.2),
                prior_sigma=prior_config.get("prior_sigma", 1.0),
                max_norm=prior_config.get("max_norm", 0.95),
                # P0 FIX: New params wired from config
                sigma_min=prior_config.get("sigma_min", 0.8),
                sigma_max=prior_config.get("sigma_max", 1.2),
                curvature_min=prior_config.get("curvature_min", 2.0),
                curvature_max=prior_config.get("curvature_max", 2.5),
                adaptation_rate=prior_config.get("adaptation_rate", 0.005),
                ema_alpha=prior_config.get("ema_alpha", 0.05),
                kl_target=prior_config.get("kl_target", 50.0),
                target_radius=prior_config.get("target_radius", 0.5),
            ).to(device)
            self.use_hyperbolic_prior = True
        else:
            self.hyperbolic_prior_A = None
            self.hyperbolic_prior_B = None
            self.use_hyperbolic_prior = False

        # Hyperbolic Reconstruction
        if hyp_v10.get("use_hyperbolic_recon", False):
            recon_config = hyp_v10.get("recon", {})
            self.hyperbolic_recon_A = HomeostaticReconLoss(
                mode=recon_config.get("mode", "weighted_ce"),
                curvature=recon_config.get("curvature", 2.0),
                max_norm=recon_config.get("max_norm", 0.95),
                radius_weighting=recon_config.get("radius_weighting", True),
                radius_power=recon_config.get("radius_power", 2.0),
            ).to(device)
            self.hyperbolic_recon_B = HomeostaticReconLoss(
                mode=recon_config.get("mode", "weighted_ce"),
                curvature=recon_config.get("curvature", 2.0),
                max_norm=recon_config.get("max_norm", 0.95),
                radius_weighting=recon_config.get("radius_weighting", True),
                radius_power=recon_config.get("radius_power", 2.0),
            ).to(device)
            self.hyperbolic_recon_weight = recon_config.get("weight", 0.5)
            self.use_hyperbolic_recon = True
        else:
            self.hyperbolic_recon_A = None
            self.hyperbolic_recon_B = None
            self.use_hyperbolic_recon = False

        # Hyperbolic Centroid Loss
        if hyp_v10.get("use_centroid_loss", False):
            centroid_config = hyp_v10.get("centroid", {})
            self.centroid_loss = HyperbolicCentroidLoss(
                max_level=centroid_config.get("max_level", 4),
                curvature=centroid_config.get("curvature", 2.0),
                max_norm=centroid_config.get("max_norm", 0.95),
            ).to(device)
            self.centroid_loss_weight = centroid_config.get("weight", 0.2)
            self.use_centroid_loss = True
        else:
            self.centroid_loss = None
            self.use_centroid_loss = False

    def _init_evaluation_config(self, config: dict[str, Any]) -> None:
        """Initialize evaluation intervals and cached values."""
        self.coverage_check_interval = config.get("coverage_check_interval", 5)
        self.eval_interval = config.get("eval_interval", 20)

        # Cached values for non-evaluation epochs
        self.cached_cov_A = 0.0
        self.cached_cov_B = 0.0
        self.cached_unique_A = 0
        self.cached_unique_B = 0
        self.cached_corr_A_hyp = 0.5
        self.cached_corr_B_hyp = 0.5
        self.cached_corr_A_euc = 0.5
        self.cached_corr_B_euc = 0.5
        self.cached_mean_radius_A = 0.0
        self.cached_mean_radius_B = 0.0

    def _init_radial_stratification(self, config: dict[str, Any], device: str) -> None:
        """Initialize RadialStratificationLoss for 3-adic hierarchy enforcement."""
        radial_config = config.get("radial_stratification", {})

        if radial_config.get("enabled", False):
            self.radial_stratification_A = RadialStratificationLoss(
                inner_radius=radial_config.get("inner_radius", 0.1),
                outer_radius=radial_config.get("outer_radius", 0.85),
                max_valuation=radial_config.get("max_valuation", 9),
                valuation_weighting=radial_config.get("valuation_weighting", True),
                loss_type=radial_config.get("loss_type", "smooth_l1"),
            ).to(device)
            self.radial_stratification_B = RadialStratificationLoss(
                inner_radius=radial_config.get("inner_radius", 0.1),
                outer_radius=radial_config.get("outer_radius", 0.85),
                max_valuation=radial_config.get("max_valuation", 9),
                valuation_weighting=radial_config.get("valuation_weighting", True),
                loss_type=radial_config.get("loss_type", "smooth_l1"),
            ).to(device)
            self.radial_stratification_weight = radial_config.get("base_weight", 0.3)
            self.use_radial_stratification = True
        else:
            self.radial_stratification_A = None
            self.radial_stratification_B = None
            self.use_radial_stratification = False

    def _init_curriculum(self, config: dict[str, Any], device: str) -> None:
        """Initialize ContinuousCurriculumModule for StateNet-driven training."""
        curriculum_config = config.get("curriculum", {})

        if curriculum_config.get("enabled", False):
            self.curriculum = ContinuousCurriculumModule(
                initial_tau=curriculum_config.get("initial_tau", 0.0),
                tau_min=curriculum_config.get("tau_min", 0.0),
                tau_max=curriculum_config.get("tau_max", 1.0),
                tau_scale=curriculum_config.get("tau_scale", 0.1),
                momentum=curriculum_config.get("tau_momentum", 0.95),
            ).to(device)
            self.use_curriculum = True
        else:
            self.curriculum = None
            self.use_curriculum = False

    def _init_consequence_predictor(self, config: dict[str, Any], device: str) -> None:
        """Initialize ConsequencePredictor for purpose-aware training."""
        mc = config.get("model", {})
        latent_dim = mc.get("latent_dim", 16)

        # Consequence predictor is always available but training is optional
        self.consequence_predictor = ConsequencePredictor(latent_dim=latent_dim, hidden_dim=32).to(device)
        self.consequence_eval_interval = config.get("consequence_eval_interval", 50)
        self.cached_addition_accuracy = 0.5

    def train_epoch(self, train_loader, val_loader, epoch: int) -> dict[str, Any]:
        """Train one epoch with pure hyperbolic geometry and homeostatic adaptation.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epoch: Current epoch number

        Returns:
            Dict containing all losses and metrics for the epoch
        """
        # Coverage evaluation (only at intervals)
        should_check_coverage = (epoch == 0) or (epoch % self.coverage_check_interval == 0)

        if should_check_coverage:
            unique_A, cov_A = self.base_trainer.monitor.evaluate_coverage(
                self.model, self.config["eval_num_samples"], self.device, "A"
            )
            unique_B, cov_B = self.base_trainer.monitor.evaluate_coverage(
                self.model, self.config["eval_num_samples"], self.device, "B"
            )
            self.cached_cov_A = cov_A
            self.cached_cov_B = cov_B
            self.cached_unique_A = unique_A
            self.cached_unique_B = unique_B
        else:
            cov_A = self.cached_cov_A
            cov_B = self.cached_cov_B
            unique_A = self.cached_unique_A
            unique_B = self.cached_unique_B

        current_coverage = (cov_A + cov_B) / 2

        # Compute adaptive ranking weight using feedback controller
        ranking_weight = self.continuous_feedback.compute_ranking_weight(current_coverage)

        # P2 FIX: Check coverage stall and apply exploration boost
        self.exploration_boost.check_coverage_stall(current_coverage)
        temp_mult, ranking_mult = self.exploration_boost.get_multipliers()

        # Apply P2 exploration boost to ranking weight
        ranking_weight = ranking_weight * ranking_mult

        self.ranking_weight_history.append(ranking_weight)

        # GAP 7 FIX: Apply exploration temp multiplier to base trainer
        # This unifies exploration_boost with curriculum temp modulation
        # Formula: final_temp = base_temp * curriculum_factor * exploration_mult
        self.base_trainer.exploration_temp_multiplier = temp_mult

        # Base training (uses DualVAELoss internally, logs batch metrics)
        train_losses = self.base_trainer.train_epoch(train_loader, log_interval=self.log_interval)

        # P2 FIX: Compute correlation loss contribution (for logging and next epoch adjustment)
        correlation_loss_value = 0.0
        if self.correlation_loss_enabled and epoch >= self.correlation_loss_warmup:
            # Use cached correlation from previous eval
            cached_corr = (
                (self.cached_corr_A_hyp + self.cached_corr_B_hyp) / 2 if hasattr(self, "cached_corr_A_hyp") else 0.0
            )
            # Correlation loss: negative correlation (reward high correlation)
            # This value is for logging; actual loss integration requires base trainer modification
            correlation_loss_value = -self.correlation_loss_weight * cached_corr
            train_losses["correlation_loss"] = correlation_loss_value
            train_losses["correlation_bonus"] = cached_corr * self.correlation_loss_weight

        # P1 FIX: Apply StateNet delta_sigma/delta_curvature to HyperbolicPrior modules
        # This enables StateNet to control the hyperbolic prior's adaptive parameters
        if self.use_hyperbolic_prior and "delta_sigma" in train_losses:
            delta_sigma = train_losses.get("delta_sigma", 0.0)
            delta_curvature = train_losses.get("delta_curvature", 0.0)

            # Apply to both VAE's hyperbolic priors
            if hasattr(self.hyperbolic_prior_A, "set_from_statenet"):
                self.hyperbolic_prior_A.set_from_statenet(delta_sigma, delta_curvature)
            if hasattr(self.hyperbolic_prior_B, "set_from_statenet"):
                self.hyperbolic_prior_B.set_from_statenet(delta_sigma, delta_curvature)

        # v5.10.1: Apply StateNet delta_curriculum to CurriculumModule
        # StateNet v5 controls the radial->ranking curriculum progression
        if self.use_curriculum and "delta_curriculum" in train_losses:
            delta_curriculum = train_losses.get("delta_curriculum", 0.0)
            self.curriculum.update_tau(delta_curriculum)

        # v5.10.1: Evaluate consequence predictor at intervals
        if epoch % self.consequence_eval_interval == 0:
            self.cached_addition_accuracy = evaluate_addition_accuracy(self.model, self.device, n_samples=500)

        # D2.2 FIX: Removed wasted validation call
        # The hyperbolic_trainer uses train_losses for all metrics (line 624).
        # Calling validate() here was wasted computation since val_losses was discarded.
        # The base_trainer.train() method handles validation properly if used directly.

        # Compute hyperbolic losses and metrics
        hyperbolic_metrics = self._compute_hyperbolic_losses(train_loader, ranking_weight, current_coverage)

        # Log hyperbolic metrics at batch level for TensorBoard
        self.log_hyperbolic_batch(hyperbolic_metrics)

        # Correlation evaluation (only at intervals)
        should_check_correlation = (epoch == 0) or (epoch % self.eval_interval == 0)

        if should_check_correlation:
            corr_results = compute_ranking_correlation_hyperbolic(
                self.model,
                self.device,
                max_norm=self.max_norm,
                curvature=self.curvature,
            )
            (
                corr_A_hyp,
                corr_B_hyp,
                corr_A_euc,
                corr_B_euc,
                mean_radius_A,
                mean_radius_B,
            ) = corr_results

            self.cached_corr_A_hyp = corr_A_hyp
            self.cached_corr_B_hyp = corr_B_hyp
            self.cached_corr_A_euc = corr_A_euc
            self.cached_corr_B_euc = corr_B_euc
            self.cached_mean_radius_A = mean_radius_A
            self.cached_mean_radius_B = mean_radius_B
        else:
            corr_A_hyp = self.cached_corr_A_hyp
            corr_B_hyp = self.cached_corr_B_hyp
            corr_A_euc = self.cached_corr_A_euc
            corr_B_euc = self.cached_corr_B_euc
            mean_radius_A = self.cached_mean_radius_A
            mean_radius_B = self.cached_mean_radius_B

        corr_mean_hyp = (corr_A_hyp + corr_B_hyp) / 2
        corr_mean_euc = (corr_A_euc + corr_B_euc) / 2

        # Update best metrics
        if corr_mean_hyp > self.best_corr_hyp:
            self.best_corr_hyp = corr_mean_hyp
        if corr_mean_euc > self.best_corr_euc:
            self.best_corr_euc = corr_mean_euc
        if current_coverage > self.best_coverage:
            self.best_coverage = current_coverage

        # Update histories
        self.correlation_history_hyp.append(corr_mean_hyp)
        self.correlation_history_euc.append(corr_mean_euc)
        self.coverage_history.append(current_coverage)

        # P1 FIX: Check correlation early stopping using controller
        should_stop_correlation = self.correlation_early_stop.check_should_stop(corr_mean_hyp)

        # Build return dict
        return {
            **train_losses,
            "ranking_weight": ranking_weight,
            "ranking_loss_hyp": hyperbolic_metrics["ranking_loss"],
            "radial_loss": hyperbolic_metrics["radial_loss"],
            "radial_stratification_loss": hyperbolic_metrics["radial_stratification_loss"],
            "curriculum_tau": hyperbolic_metrics["curriculum_tau"],
            "hyp_kl_A": hyperbolic_metrics["hyp_kl_A"],
            "hyp_kl_B": hyperbolic_metrics["hyp_kl_B"],
            "hyp_recon_A": hyperbolic_metrics["hyp_recon_A"],
            "hyp_recon_B": hyperbolic_metrics["hyp_recon_B"],
            "centroid_loss": hyperbolic_metrics["centroid_loss"],
            "corr_A_hyp": corr_A_hyp,
            "corr_B_hyp": corr_B_hyp,
            "corr_mean_hyp": corr_mean_hyp,
            "corr_A_euc": corr_A_euc,
            "corr_B_euc": corr_B_euc,
            "corr_mean_euc": corr_mean_euc,
            "cov_A": cov_A,
            "cov_B": cov_B,
            "cov_mean": current_coverage,
            "unique_A": unique_A,
            "unique_B": unique_B,
            "coverage_ema": self.continuous_feedback.coverage_ema or current_coverage,
            "mean_radius_A": mean_radius_A,
            "mean_radius_B": mean_radius_B,
            "coverage_evaluated": should_check_coverage,
            "correlation_evaluated": should_check_correlation,
            # P1 FIX: Correlation early stopping signal
            "should_stop_correlation": should_stop_correlation,
            "correlation_drop_counter": self.correlation_early_stop.drop_counter,
            "best_correlation_for_stopping": self.correlation_early_stop.best_correlation,
            # P2 FIX: Exploration boost metrics
            "exploration_boosted": self.exploration_boost.is_boosting,
            "coverage_stall_counter": self.exploration_boost.stall_counter,
            "temp_multiplier": self.exploration_boost.temp_multiplier,
            "ranking_multiplier": self.exploration_boost.ranking_multiplier,
            **{f"ranking_{k}": v for k, v in hyperbolic_metrics["ranking_metrics"].items()},
            **{f"homeo_{k}": v for k, v in hyperbolic_metrics["homeostatic_metrics"].items()},
        }

    def _compute_hyperbolic_losses(
        self, train_loader, ranking_weight: float, current_coverage: float
    ) -> dict[str, Any]:
        """Compute all hyperbolic geometry losses.

        Args:
            train_loader: Training data loader
            ranking_weight: Current ranking loss weight
            current_coverage: Current coverage for homeostatic adaptation

        Returns:
            Dict of hyperbolic loss values and metrics
        """
        # P0 FIX: Short-circuit when ALL hyperbolic modules are disabled
        # Avoids 2000-sample forward pass when nothing will use the outputs
        all_disabled = (
            self.ranking_loss_hyp is None
            and not self.use_hyperbolic_prior
            and not self.use_hyperbolic_recon
            and not self.use_centroid_loss
            and not self.use_radial_stratification  # v5.10.1
        )

        if all_disabled or ranking_weight <= 0:
            self.radial_loss_history.append(0.0)
            self.homeostatic_history.append({})
            return {
                "ranking_loss": 0.0,
                "radial_loss": 0.0,
                "radial_stratification_loss": 0.0,
                "hyp_kl_A": 0.0,
                "hyp_kl_B": 0.0,
                "hyp_recon_A": 0.0,
                "hyp_recon_B": 0.0,
                "centroid_loss": 0.0,
                "curriculum_tau": (self.curriculum.get_tau().item() if self.use_curriculum else 0.0),
                "ranking_metrics": {},
                "homeostatic_metrics": {},
            }

        ranking_loss = 0.0
        radial_loss = 0.0
        hyp_kl_A = 0.0
        hyp_kl_B = 0.0
        hyp_recon_A = 0.0
        hyp_recon_B = 0.0
        centroid_loss_val = 0.0
        ranking_metrics = {}
        homeostatic_metrics = {}

        self.model.eval()
        with torch.no_grad():
            # Sample for loss computation
            n_samples = min(2000, len(train_loader.dataset))
            indices = torch.randint(0, N_TERNARY_OPERATIONS, (n_samples,), device=self.device)

            ternary_data = torch.zeros(n_samples, 9, device=self.device)
            for i in range(9):
                ternary_data[:, i] = ((indices // (3**i)) % 3) - 1

            outputs = self.model(ternary_data.float(), 1.0, 1.0, 0.5, 0.5)
            z_A = outputs["z_A"]
            z_B = outputs["z_B"]
            mu_A = outputs["mu_A"]
            mu_B = outputs["mu_B"]
            logvar_A = outputs["logvar_A"]
            logvar_B = outputs["logvar_B"]
            logits_A = outputs["logits_A"]
            logits_B = outputs["logits_B"]

        self.model.train()

        # Hyperbolic Ranking Loss
        if self.ranking_loss_hyp is not None:
            loss_A, metrics_A = self.ranking_loss_hyp(z_A, indices)
            loss_B, metrics_B = self.ranking_loss_hyp(z_B, indices)

            ranking_loss = ranking_weight * (loss_A.item() + loss_B.item()) / 2
            radial_loss = (metrics_A.get("radial_loss", 0) + metrics_B.get("radial_loss", 0)) / 2

            ranking_metrics = {
                "hard_ratio": (metrics_A.get("hard_ratio", 0) + metrics_B.get("hard_ratio", 0)) / 2,
                "violations": metrics_A.get("violations", 0) + metrics_B.get("violations", 0),
                "radial_loss": radial_loss,
            }

        # Hyperbolic Prior KL
        if self.use_hyperbolic_prior:
            kl_A, z_hyp_A = self.hyperbolic_prior_A(mu_A, logvar_A)
            kl_B, z_hyp_B = self.hyperbolic_prior_B(mu_B, logvar_B)
            hyp_kl_A = kl_A.item()
            hyp_kl_B = kl_B.item()

            # Update homeostatic state
            self.hyperbolic_prior_A.update_homeostatic_state(z_hyp_A, kl_A, current_coverage)
            self.hyperbolic_prior_B.update_homeostatic_state(z_hyp_B, kl_B, current_coverage)

            homeostatic_metrics.update(
                {
                    "prior_sigma_A": self.hyperbolic_prior_A.adaptive_sigma.item(),
                    "prior_sigma_B": self.hyperbolic_prior_B.adaptive_sigma.item(),
                    "prior_curvature_A": self.hyperbolic_prior_A.adaptive_curvature.item(),
                    "prior_curvature_B": self.hyperbolic_prior_B.adaptive_curvature.item(),
                }
            )

        # Hyperbolic Reconstruction
        if self.use_hyperbolic_recon:
            recon_A, recon_metrics_A = self.hyperbolic_recon_A(logits_A, ternary_data, z_A)
            recon_B, recon_metrics_B = self.hyperbolic_recon_B(logits_B, ternary_data, z_B)
            hyp_recon_A = recon_A.item()
            hyp_recon_B = recon_B.item()

            homeostatic_metrics.update(
                {
                    "recon_radius_A": recon_metrics_A.get("mean_radius", 0),
                    "recon_radius_B": recon_metrics_B.get("mean_radius", 0),
                }
            )

        # Centroid Loss
        if self.use_centroid_loss:
            cent_A, _ = self.centroid_loss(z_A, indices)
            cent_B, _ = self.centroid_loss(z_B, indices)
            centroid_loss_val = (cent_A.item() + cent_B.item()) / 2

            homeostatic_metrics.update({"centroid_loss": centroid_loss_val})

        # v5.10.1: RadialStratificationLoss - enforces 3-adic tree structure via radial position
        radial_stratification_loss = 0.0
        radial_strat_metrics = {}
        if self.use_radial_stratification:
            loss_A, metrics_A = self.radial_stratification_A(z_A, indices, return_metrics=True)
            loss_B, metrics_B = self.radial_stratification_B(z_B, indices, return_metrics=True)
            radial_stratification_loss = (loss_A.item() + loss_B.item()) / 2

            radial_strat_metrics = {
                "radial_correlation_A": metrics_A.get("radial_correlation", 0),
                "radial_correlation_B": metrics_B.get("radial_correlation", 0),
                "mean_radius_error": (metrics_A.get("mean_radius_error", 0) + metrics_B.get("mean_radius_error", 0))
                / 2,
                "high_v_radius": (metrics_A.get("high_v_radius", 0) + metrics_B.get("high_v_radius", 0)) / 2,
                "low_v_radius": (metrics_A.get("low_v_radius", 0) + metrics_B.get("low_v_radius", 0)) / 2,
            }
            homeostatic_metrics.update(radial_strat_metrics)

        # v5.10.1: Curriculum modulation - blend radial and ranking losses
        curriculum_tau = 0.0
        if self.use_curriculum:
            curriculum_tau = self.curriculum.get_tau().item()
            # The modulated loss would be: (1-tau)*radial + tau*ranking
            # But actual gradient flow happens in base_trainer, so we track metrics
            homeostatic_metrics.update(
                {
                    "curriculum_tau": curriculum_tau,
                    "curriculum_radial_weight": 1 - curriculum_tau,
                    "curriculum_ranking_weight": curriculum_tau,
                }
            )

        self.radial_loss_history.append(radial_loss + radial_stratification_loss)
        self.homeostatic_history.append(homeostatic_metrics)

        return {
            "ranking_loss": ranking_loss,
            "radial_loss": radial_loss,
            "radial_stratification_loss": radial_stratification_loss,
            "hyp_kl_A": hyp_kl_A,
            "hyp_kl_B": hyp_kl_B,
            "hyp_recon_A": hyp_recon_A,
            "hyp_recon_B": hyp_recon_B,
            "centroid_loss": centroid_loss_val,
            "curriculum_tau": curriculum_tau,
            "ranking_metrics": ranking_metrics,
            "homeostatic_metrics": homeostatic_metrics,
        }

    def log_epoch(self, epoch: int, losses: dict[str, Any]) -> None:
        """Unified epoch-level logging to TensorBoard and console/file.

        This method centralizes ALL observability for the epoch:
        - Console/file summary via log_epoch_summary()
        - TensorBoard hyperbolic metrics via log_hyperbolic_epoch()
        - TensorBoard standard VAE metrics via log_tensorboard()
        - Weight/gradient histograms at intervals

        Args:
            epoch: Current epoch number
            losses: Dict containing all losses and metrics from train_epoch()
        """
        if self.monitor is None:
            return

        # Extract homeostatic metrics
        homeo = {k.replace("homeo_", ""): v for k, v in losses.items() if k.startswith("homeo_")}
        homeo_dict = homeo if homeo else None

        # 1. Console/file epoch summary
        self.monitor.log_epoch_summary(
            epoch=epoch,
            total_epochs=self.total_epochs,
            loss=losses["loss"],
            cov_A=losses["cov_A"],
            cov_B=losses["cov_B"],
            corr_A_hyp=losses["corr_A_hyp"],
            corr_B_hyp=losses["corr_B_hyp"],
            corr_A_euc=losses["corr_A_euc"],
            corr_B_euc=losses["corr_B_euc"],
            mean_radius_A=losses["mean_radius_A"],
            mean_radius_B=losses["mean_radius_B"],
            ranking_weight=losses["ranking_weight"],
            coverage_evaluated=losses.get("coverage_evaluated", True),
            correlation_evaluated=losses.get("correlation_evaluated", True),
            hyp_kl_A=losses.get("hyp_kl_A", 0),
            hyp_kl_B=losses.get("hyp_kl_B", 0),
            centroid_loss=losses.get("centroid_loss", 0),
            radial_loss=losses.get("radial_loss", 0),
            homeostatic_metrics=homeo_dict,
        )

        # 2. TensorBoard hyperbolic metrics (epoch-level)
        self.monitor.log_hyperbolic_epoch(
            epoch=epoch,
            corr_A_hyp=losses["corr_A_hyp"],
            corr_B_hyp=losses["corr_B_hyp"],
            corr_A_euc=losses["corr_A_euc"],
            corr_B_euc=losses["corr_B_euc"],
            mean_radius_A=losses["mean_radius_A"],
            mean_radius_B=losses["mean_radius_B"],
            ranking_weight=losses["ranking_weight"],
            ranking_loss=losses.get("ranking_loss_hyp", 0),
            radial_loss=losses.get("radial_loss", 0),
            hyp_kl_A=losses.get("hyp_kl_A", 0),
            hyp_kl_B=losses.get("hyp_kl_B", 0),
            centroid_loss=losses.get("centroid_loss", 0),
            homeostatic_metrics=homeo_dict,
        )

        # 3. TensorBoard standard VAE metrics (epoch-level)
        self._log_standard_tensorboard(epoch, losses)

        # 4. Weight/gradient histograms at intervals
        if epoch % self.histogram_interval == 0:
            self.monitor.log_histograms(epoch, self.model)

        # 5. Embedding projections at intervals (for 3D visualization)
        if self.embedding_interval > 0 and epoch % self.embedding_interval == 0:
            self.monitor.log_manifold_embedding(
                self.model,
                epoch,
                self.device,
                n_samples=self.embedding_n_samples,
            )

    def _log_standard_tensorboard(self, epoch: int, losses: dict[str, Any]) -> None:
        """Log standard VAE metrics to TensorBoard.

        Args:
            epoch: Current epoch
            losses: Losses dict from train_epoch()
        """
        if self.monitor.writer is None:
            return

        writer = self.monitor.writer

        # Total loss
        writer.add_scalar("Loss/Total", losses["loss"], epoch)

        # VAE-A metrics
        writer.add_scalar("VAE_A/CrossEntropy", losses.get("ce_A", 0), epoch)
        writer.add_scalar("VAE_A/KL_Divergence", losses.get("kl_A", 0), epoch)
        writer.add_scalar("VAE_A/Entropy", losses.get("H_A", 0), epoch)
        writer.add_scalar("VAE_A/Coverage_Pct", losses["cov_A"], epoch)

        # VAE-B metrics
        writer.add_scalar("VAE_B/CrossEntropy", losses.get("ce_B", 0), epoch)
        writer.add_scalar("VAE_B/KL_Divergence", losses.get("kl_B", 0), epoch)
        writer.add_scalar("VAE_B/Entropy", losses.get("H_B", 0), epoch)
        writer.add_scalar("VAE_B/Coverage_Pct", losses["cov_B"], epoch)

        # Comparative metrics
        writer.add_scalars(
            "Compare/Coverage",
            {"VAE_A": losses["cov_A"], "VAE_B": losses["cov_B"]},
            epoch,
        )

        writer.add_scalars(
            "Compare/Entropy",
            {"VAE_A": losses.get("H_A", 0), "VAE_B": losses.get("H_B", 0)},
            epoch,
        )

        # Training dynamics
        writer.add_scalar("Dynamics/Phase", losses.get("phase", 0), epoch)
        writer.add_scalar("Dynamics/Rho", losses.get("rho", 0), epoch)
        writer.add_scalar("Dynamics/GradRatio", losses.get("grad_ratio", 0), epoch)

        # Lambda weights
        writer.add_scalars(
            "Lambdas",
            {
                "lambda1": losses.get("lambda1", 0),
                "lambda2": losses.get("lambda2", 0),
                "lambda3": losses.get("lambda3", 0),
            },
            epoch,
        )

        # Temperature and beta
        writer.add_scalars(
            "Temperature",
            {
                "VAE_A": losses.get("temp_A", 1.0),
                "VAE_B": losses.get("temp_B", 1.0),
            },
            epoch,
        )

        writer.add_scalars(
            "Beta",
            {
                "VAE_A": losses.get("beta_A", 1.0),
                "VAE_B": losses.get("beta_B", 1.0),
            },
            epoch,
        )

        # Learning rate
        writer.add_scalar("LR/Scheduled", losses.get("lr_scheduled", 0), epoch)
        if "lr_corrected" in losses:
            writer.add_scalar("LR/Corrected", losses["lr_corrected"], epoch)

        # Coverage EMA for continuous feedback
        if "coverage_ema" in losses:
            writer.add_scalar("Feedback/CoverageEMA", losses["coverage_ema"], epoch)

        writer.flush()

    def log_hyperbolic_batch(self, hyperbolic_metrics: dict[str, Any]) -> None:
        """Log hyperbolic metrics at batch level to TensorBoard.

        Called after computing hyperbolic losses for real-time observability.

        Args:
            hyperbolic_metrics: Dict from _compute_hyperbolic_losses()
        """
        if self.monitor is None:
            return

        self.monitor.log_hyperbolic_batch(
            ranking_loss=hyperbolic_metrics.get("ranking_loss", 0),
            radial_loss=hyperbolic_metrics.get("radial_loss", 0),
            hyp_kl_A=hyperbolic_metrics.get("hyp_kl_A", 0),
            hyp_kl_B=hyperbolic_metrics.get("hyp_kl_B", 0),
            centroid_loss=hyperbolic_metrics.get("centroid_loss", 0),
        )

    def update_monitor_state(self, losses: dict[str, Any]) -> None:
        """Update monitor's internal tracking state.

        Args:
            losses: Losses dict from train_epoch()
        """
        if self.monitor is None:
            return

        self.monitor.check_best(losses["loss"])
        self.monitor.update_histories(
            H_A=losses.get("H_A", 0),
            H_B=losses.get("H_B", 0),
            coverage_A=losses["unique_A"],
            coverage_B=losses["unique_B"],
        )

    def print_summary(self) -> None:
        """Print training completion summary."""
        if self.monitor:
            self.monitor.print_training_summary()

    def close(self) -> None:
        """Close TensorBoard writer and cleanup."""
        if self.monitor:
            self.monitor.close()


__all__ = ["HyperbolicVAETrainer"]
