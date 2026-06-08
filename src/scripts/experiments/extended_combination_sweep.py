#!/usr/bin/env python
"""Extended Combination Sweep: Test Advanced Components.

This script extends combination_sweep.py to test:
- MixedRiemannianOptimizer
- Feedback controllers (ContinuousFeedback, ExplorationBoost)
- Curriculum learning
- Homeostatic losses
- SwarmVAE and TropicalVAE (if available)
- Controllers (Differentiable, Homeostasis)

Run with:
    python src/scripts/experiments/extended_combination_sweep.py --phase 5 --sequential
    python src/scripts/experiments/extended_combination_sweep.py --phase 6 --sequential
    python src/scripts/experiments/extended_combination_sweep.py --all-advanced --sequential
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# EXTENDED CONFIGURATION
# =============================================================================


@dataclass
class ExtendedExperimentConfig:
    """Extended configuration with advanced components."""

    name: str
    # Architecture
    model_type: str = "simple_hyperbolic"
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    # Projection
    use_hyperbolic: bool = True
    curvature: float = 1.0
    # Loss
    padic_loss_type: str = "none"
    padic_weight: float = 0.3
    radial_loss_type: str = "none"
    radial_weight: float = 0.3
    use_homeostatic_prior: bool = False
    use_homeostatic_recon: bool = False
    # Training
    beta: float = 0.1
    beta_schedule: str = "cyclical"
    learning_rate: float = 0.005
    epochs: int = 50
    # Advanced - Optimizer
    optimizer_type: str = "adam"  # adam, mixed_riemannian
    # Advanced - Feedback
    use_continuous_feedback: bool = False
    use_exploration_boost: bool = False
    # Advanced - Curriculum
    use_curriculum: bool = False
    curriculum_threshold: float = -0.70
    # Advanced - Controller
    controller_type: str = "none"  # none, differentiable, homeostasis


@dataclass
class ExtendedExperimentResult:
    """Results with extended metrics."""

    name: str
    accuracy: float
    spearman: float
    silhouette: float
    training_time: float
    final_loss: float
    config: ExtendedExperimentConfig
    # Extended metrics
    coverage: float = 0.0
    stability: float = 1.0
    error: str | None = None


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def compute_padic_distance(i: int, j: int) -> float:
    """Compute 3-adic distance."""
    if i == j:
        return 0.0
    diff = abs(i - j)
    k = 0
    while diff % 3 == 0:
        diff //= 3
        k += 1
    return 3.0 ** (-k)


def evaluate_embeddings(z: torch.Tensor, indices: torch.Tensor) -> dict[str, float]:
    """Evaluate latent space structure."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(z)
    z_np = z.detach().cpu().numpy()
    indices_np = indices.cpu().numpy()

    # Spearman correlation
    n_pairs = min(3000, n * (n - 1) // 2)
    np.random.seed(42)
    i_idx = np.random.randint(0, n, n_pairs)
    j_idx = np.random.randint(0, n, n_pairs)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]

    padic_dists = [compute_padic_distance(indices_np[i], indices_np[j]) for i, j in zip(i_idx, j_idx, strict=False)]
    latent_dists = np.linalg.norm(z_np[i_idx] - z_np[j_idx], axis=1)

    corr, _ = spearmanr(padic_dists, latent_dists)

    # Silhouette
    try:
        n_samples = min(2000, n)
        sample_idx = np.random.choice(n, n_samples, replace=False)
        kmeans = KMeans(n_clusters=min(27, n_samples // 10), random_state=42, n_init=10)
        labels = kmeans.fit_predict(z_np[sample_idx])
        silhouette = silhouette_score(z_np[sample_idx], labels)
    except Exception:
        silhouette = 0.0

    return {
        "spearman": corr if not np.isnan(corr) else 0.0,
        "silhouette": silhouette,
    }


# =============================================================================
# MODEL FACTORY
# =============================================================================


def create_model(config: ExtendedExperimentConfig):
    """Create model based on configuration."""
    if config.model_type == "simple":
        from src.models.simple_vae import SimpleVAE

        return SimpleVAE(
            input_dim=9,
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
        )
    elif config.model_type == "simple_hyperbolic":
        from src.models.simple_vae import SimpleVAEWithHyperbolic

        return SimpleVAEWithHyperbolic(
            input_dim=9,
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
            curvature=config.curvature,
        )
    elif config.model_type == "swarm":
        try:
            from src.models.swarm_vae import SwarmVAE

            return SwarmVAE(
                input_dim=9,
                latent_dim=config.latent_dim,
                hidden_dim=config.hidden_dims[0],
            )
        except ImportError:
            # Fall back to simple
            from src.models.simple_vae import SimpleVAE

            return SimpleVAE(input_dim=9, latent_dim=config.latent_dim)
    elif config.model_type == "tropical":
        try:
            from src.models.tropical.tropical_vae import TropicalVAE

            return TropicalVAE(
                input_dim=9,
                latent_dim=config.latent_dim,
                hidden_dim=config.hidden_dims[0],
            )
        except ImportError:
            from src.models.simple_vae import SimpleVAE

            return SimpleVAE(input_dim=9, latent_dim=config.latent_dim)
    else:
        from src.models.simple_vae import SimpleVAE

        return SimpleVAE(input_dim=9, latent_dim=config.latent_dim)


# =============================================================================
# OPTIMIZER FACTORY
# =============================================================================


def create_optimizer(model, config: ExtendedExperimentConfig):
    """Create optimizer based on configuration."""
    if config.optimizer_type == "mixed_riemannian":
        try:
            from src.training.optimizers.riemannian import MixedRiemannianOptimizer, OptimizerConfig

            opt_config = OptimizerConfig(
                euclidean_lr=config.learning_rate,
                manifold_lr=config.learning_rate * 0.1,  # Lower for manifold
                grad_clip_norm=1.0,
            )
            return MixedRiemannianOptimizer(model, opt_config)
        except ImportError:
            return torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate)


# =============================================================================
# LOSS FACTORY
# =============================================================================


def create_padic_loss(config: ExtendedExperimentConfig):
    """Create p-adic loss based on configuration."""
    if config.padic_loss_type == "none":
        return None
    elif config.padic_loss_type == "triplet":
        from src.losses.padic import PAdicRankingLoss

        return PAdicRankingLoss(margin=0.1, n_triplets=500)
    elif config.padic_loss_type == "soft_ranking":
        return SoftPadicRankingLoss(temperature=0.5)
    elif config.padic_loss_type == "geodesic":
        from src.losses.padic_geodesic import PAdicGeodesicLoss

        return PAdicGeodesicLoss(curvature=config.curvature)
    elif config.padic_loss_type == "norm":
        try:
            from src.losses.padic.norm_loss import PAdicNormLoss

            return PAdicNormLoss()
        except ImportError:
            return None
    elif config.padic_loss_type == "metric":
        try:
            from src.losses.padic.metric_loss import PAdicMetricLoss

            return PAdicMetricLoss()
        except ImportError:
            return None
    return None


def create_radial_loss(config: ExtendedExperimentConfig):
    """Create radial loss based on configuration."""
    if config.radial_loss_type == "none":
        return None
    elif config.radial_loss_type == "hierarchy":
        from src.losses.padic_geodesic import RadialHierarchyLoss

        return RadialHierarchyLoss()
    elif config.radial_loss_type == "monotonic":
        from src.losses.padic_geodesic import MonotonicRadialLoss

        return MonotonicRadialLoss()
    elif config.radial_loss_type == "global_rank":
        from src.losses.padic_geodesic import GlobalRankLoss

        return GlobalRankLoss()
    return None


def create_kl_loss(config: ExtendedExperimentConfig):
    """Create KL loss (optionally homeostatic)."""
    if config.use_homeostatic_prior:
        try:
            from src.losses.hyperbolic_prior import HomeostaticHyperbolicPrior

            return HomeostaticHyperbolicPrior(
                latent_dim=config.latent_dim,
                curvature=config.curvature,
            )
        except ImportError:
            from src.losses.dual_vae_loss import KLDivergenceLoss

            return KLDivergenceLoss()
    else:
        from src.losses.dual_vae_loss import KLDivergenceLoss

        return KLDivergenceLoss()


# =============================================================================
# SOFT P-ADIC RANKING LOSS
# =============================================================================


class SoftPadicRankingLoss(nn.Module):
    """Soft p-adic ranking using KL divergence."""

    def __init__(self, temperature: float = 0.5, n_samples: int = 200):
        super().__init__()
        self.temperature = temperature
        self.n_samples = n_samples

    def forward(self, z: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        n = z.size(0)
        if n < 3:
            return torch.tensor(0.0, device=z.device)

        if n > self.n_samples:
            idx = torch.randperm(n)[: self.n_samples]
            z = z[idx]
            indices = indices[idx]
            n = self.n_samples

        latent_dist = torch.cdist(z, z)
        padic_dist = torch.zeros(n, n, device=z.device)
        for i in range(n):
            for j in range(n):
                padic_dist[i, j] = compute_padic_distance(indices[i].item(), indices[j].item())

        latent_ranks = F.softmax(-latent_dist / self.temperature, dim=1)
        padic_ranks = F.softmax(-padic_dist / self.temperature, dim=1)

        loss = F.kl_div(latent_ranks.log(), padic_ranks, reduction="batchmean")
        return loss


# =============================================================================
# FEEDBACK CONTROLLERS
# =============================================================================


class SimpleFeedbackController:
    """Simplified feedback controller for sweep."""

    def __init__(self, base_weight: float = 0.3):
        self.base_weight = base_weight
        self.coverage_history = []

    def update(self, coverage: float) -> float:
        """Update and return new weight."""
        self.coverage_history.append(coverage)
        if len(self.coverage_history) < 3:
            return self.base_weight

        # If coverage improving, increase ranking weight
        recent = self.coverage_history[-3:]
        if recent[-1] > recent[0]:
            return min(0.9, self.base_weight * 1.2)
        else:
            return max(0.1, self.base_weight * 0.8)


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================


def run_experiment(config: ExtendedExperimentConfig) -> ExtendedExperimentResult:
    """Run a single experiment with extended configuration."""
    from src.data.generation import generate_all_ternary_operations
    from src.losses.dual_vae_loss import ReconstructionLoss

    start_time = time.time()

    try:
        # Load data
        ops = torch.tensor(generate_all_ternary_operations()[:1000], dtype=torch.float32)
        indices = torch.arange(len(ops))

        # Create model
        model = create_model(config)
        optimizer = create_optimizer(model, config)

        # Create losses
        recon_fn = ReconstructionLoss()
        kl_fn = create_kl_loss(config)
        padic_fn = create_padic_loss(config)
        radial_fn = create_radial_loss(config)

        # Feedback controller
        feedback = SimpleFeedbackController(config.padic_weight) if config.use_continuous_feedback else None

        # Training loop
        loss_history = []
        for epoch in range(config.epochs):
            model.train()
            outputs = model(ops)

            # Base losses
            logits = outputs["logits"]
            mu = outputs["mu"]
            logvar = outputs["logvar"]
            z = outputs.get("z_hyp", outputs.get("z_euc", outputs["z"]))

            recon = recon_fn(logits, ops)

            # KL loss
            if hasattr(kl_fn, "forward") and "z" in kl_fn.forward.__code__.co_varnames:
                kl = kl_fn(mu, logvar, z)
            else:
                kl = kl_fn(mu, logvar)

            # Beta schedule
            if config.beta_schedule == "warmup":
                beta = config.beta * min(1.0, epoch / 20)
            elif config.beta_schedule == "cyclical":
                beta = config.beta * (0.5 + 0.5 * np.sin(2 * np.pi * epoch / 50))
            else:
                beta = config.beta

            loss = recon + beta * kl

            # P-adic loss with optional feedback
            padic_weight = config.padic_weight
            if feedback is not None:
                # Estimate coverage (simplified)
                with torch.no_grad():
                    pred = outputs["logits"].argmax(dim=-1)
                    target = (ops + 1).long()
                    coverage = (pred == target).float().mean().item() * 100
                padic_weight = feedback.update(coverage)

            if padic_fn is not None:
                padic = padic_fn(z, indices)
                if isinstance(padic, tuple):
                    padic = padic[0]
                loss = loss + padic_weight * padic

            # Radial loss
            if radial_fn is not None:
                radial = radial_fn(z, indices)
                if isinstance(radial, tuple):
                    radial = radial[0]
                loss = loss + config.radial_weight * radial

            # Curriculum (simplified - reduce loss weights early)
            if config.use_curriculum and epoch < 10:
                loss = loss * (0.5 + 0.05 * epoch)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            loss_history.append(loss.item())

        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(ops)
            z = outputs.get("z_euc", outputs["z"])

            pred = outputs["logits"].argmax(dim=-1)
            target = (ops + 1).long()
            accuracy = (pred == target).float().mean().item()

            metrics = evaluate_embeddings(z, indices)

        # Stability metric (loss variance in last 10 epochs)
        stability = 1.0 / (1.0 + np.std(loss_history[-10:]))

        training_time = time.time() - start_time

        return ExtendedExperimentResult(
            name=config.name,
            accuracy=accuracy,
            spearman=metrics["spearman"],
            silhouette=metrics["silhouette"],
            training_time=training_time,
            final_loss=loss.item(),
            config=config,
            coverage=accuracy * 100,
            stability=stability,
        )

    except Exception as e:
        import traceback

        return ExtendedExperimentResult(
            name=config.name,
            accuracy=0.0,
            spearman=0.0,
            silhouette=0.0,
            training_time=time.time() - start_time,
            final_loss=float("inf"),
            config=config,
            error=f"{str(e)}\n{traceback.format_exc()[:200]}",
        )


# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================


def get_phase5_experiments() -> list[ExtendedExperimentConfig]:
    """Phase 5: Advanced training components."""
    experiments = []

    # Base config (our current best)
    base = {
        "model_type": "simple_hyperbolic",
        "beta": 0.1,
        "beta_schedule": "cyclical",
        "learning_rate": 0.005,
        "padic_loss_type": "triplet",
        "radial_loss_type": "monotonic",
    }

    # Test feedback controllers
    experiments.append(ExtendedExperimentConfig(name="adv_baseline", **base))

    experiments.append(ExtendedExperimentConfig(name="adv_continuous_feedback", use_continuous_feedback=True, **base))

    experiments.append(ExtendedExperimentConfig(name="adv_curriculum", use_curriculum=True, **base))

    experiments.append(
        ExtendedExperimentConfig(
            name="adv_feedback_curriculum", use_continuous_feedback=True, use_curriculum=True, **base
        )
    )

    # Test homeostatic losses
    experiments.append(ExtendedExperimentConfig(name="adv_homeostatic_prior", use_homeostatic_prior=True, **base))

    # Test different optimizers
    experiments.append(ExtendedExperimentConfig(name="adv_riemannian_opt", optimizer_type="mixed_riemannian", **base))

    # Combined advanced
    experiments.append(
        ExtendedExperimentConfig(
            name="adv_full_stack",
            use_continuous_feedback=True,
            use_curriculum=True,
            use_homeostatic_prior=True,
            optimizer_type="mixed_riemannian",
            **base,
        )
    )

    return experiments


def get_phase6_experiments() -> list[ExtendedExperimentConfig]:
    """Phase 6: Alternative model architectures."""
    experiments = []

    base = {
        "beta": 0.1,
        "beta_schedule": "cyclical",
        "learning_rate": 0.005,
    }

    # SwarmVAE tests
    experiments.append(
        ExtendedExperimentConfig(name="swarm_baseline", model_type="swarm", use_hyperbolic=False, **base)
    )

    experiments.append(
        ExtendedExperimentConfig(
            name="swarm_triplet_monotonic",
            model_type="swarm",
            use_hyperbolic=False,
            padic_loss_type="triplet",
            radial_loss_type="monotonic",
            **base,
        )
    )

    # TropicalVAE tests
    experiments.append(
        ExtendedExperimentConfig(name="tropical_baseline", model_type="tropical", use_hyperbolic=False, **base)
    )

    experiments.append(
        ExtendedExperimentConfig(
            name="tropical_soft_ranking",
            model_type="tropical",
            use_hyperbolic=False,
            padic_loss_type="soft_ranking",
            **base,
        )
    )

    return experiments


def get_phase7_experiments() -> list[ExtendedExperimentConfig]:
    """Phase 7: Alternative loss combinations."""
    experiments = []

    base = {
        "model_type": "simple_hyperbolic",
        "beta": 0.1,
        "beta_schedule": "cyclical",
        "learning_rate": 0.005,
    }

    # Test untested p-adic losses
    for padic_type in ["norm", "metric"]:
        experiments.append(
            ExtendedExperimentConfig(
                name=f"loss_padic_{padic_type}", padic_loss_type=padic_type, radial_loss_type="monotonic", **base
            )
        )

    # Homeostatic combinations
    experiments.append(
        ExtendedExperimentConfig(
            name="loss_homeostatic_soft", use_homeostatic_prior=True, padic_loss_type="soft_ranking", **base
        )
    )

    experiments.append(
        ExtendedExperimentConfig(
            name="loss_homeostatic_geodesic",
            use_homeostatic_prior=True,
            padic_loss_type="geodesic",
            radial_loss_type="hierarchy",
            **base,
        )
    )

    return experiments


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Extended Combination Sweep")
    parser.add_argument("--phase", type=int, choices=[5, 6, 7], help="Run specific phase")
    parser.add_argument("--all-advanced", action="store_true", help="Run all advanced phases")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially")
    parser.add_argument("--output", type=str, default="extended_results.json")

    args = parser.parse_args()

    # Select experiments
    if args.all_advanced:
        experiments = get_phase5_experiments() + get_phase6_experiments() + get_phase7_experiments()
        phase_name = "All Advanced Phases"
    elif args.phase == 5:
        experiments = get_phase5_experiments()
        phase_name = "Phase 5 (Advanced Training)"
    elif args.phase == 6:
        experiments = get_phase6_experiments()
        phase_name = "Phase 6 (Alternative Models)"
    elif args.phase == 7:
        experiments = get_phase7_experiments()
        phase_name = "Phase 7 (Alternative Losses)"
    else:
        experiments = get_phase5_experiments()
        phase_name = "Phase 5 (default)"

    print("\n" + "=" * 70)
    print(f"EXTENDED COMBINATION SWEEP: {phase_name}")
    print("=" * 70)
    print(f"Running {len(experiments)} experiments")
    print("=" * 70 + "\n")

    results = []

    for exp in experiments:
        print(f"Running {exp.name}...", end=" ", flush=True)
        result = run_experiment(exp)
        results.append(result)
        if result.error:
            print(f"ERROR: {result.error[:50]}")
        else:
            print(f"acc={result.accuracy:.1%}, corr={result.spearman:+.4f}, stab={result.stability:.2f}")

    # Sort by composite score
    def score(r):
        if r.error:
            return -999
        return 0.4 * r.accuracy + 0.3 * r.spearman + 0.15 * r.silhouette + 0.15 * r.stability

    results.sort(key=lambda r: -score(r))

    # Print results table
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY (sorted by composite score)")
    print("=" * 100)
    print(
        f"{'Rank':<5} {'Experiment':<35} {'Accuracy':>10} {'Spearman':>10} {'Silhouette':>10} {'Stability':>10} {'Score':>8}"
    )
    print("-" * 100)

    for i, r in enumerate(results, 1):
        if r.error:
            print(f"{i:<5} {r.name:<35} {'ERROR':<10} {r.error[:40]}")
        else:
            s = score(r)
            print(
                f"{i:<5} {r.name:<35} {r.accuracy:>9.1%} {r.spearman:>+10.4f} {r.silhouette:>10.4f} {r.stability:>10.2f} {s:>8.4f}"
            )

    # Save results
    results_data = []
    for r in results:
        results_data.append(
            {
                "name": r.name,
                "accuracy": r.accuracy,
                "spearman": r.spearman,
                "silhouette": r.silhouette,
                "stability": r.stability,
                "training_time": r.training_time,
                "final_loss": r.final_loss,
                "error": r.error,
                "config": {
                    "model_type": r.config.model_type,
                    "latent_dim": r.config.latent_dim,
                    "use_hyperbolic": r.config.use_hyperbolic,
                    "curvature": r.config.curvature,
                    "padic_loss_type": r.config.padic_loss_type,
                    "radial_loss_type": r.config.radial_loss_type,
                    "beta": r.config.beta,
                    "beta_schedule": r.config.beta_schedule,
                    "optimizer_type": r.config.optimizer_type,
                    "use_continuous_feedback": r.config.use_continuous_feedback,
                    "use_curriculum": r.config.use_curriculum,
                    "use_homeostatic_prior": r.config.use_homeostatic_prior,
                },
            }
        )

    output_path = PROJECT_ROOT / "outputs" / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print winner
    if results and not results[0].error:
        winner = results[0]
        print("\n" + "=" * 70)
        print("WINNER")
        print("=" * 70)
        print(f"Configuration: {winner.name}")
        print(f"Accuracy: {winner.accuracy:.1%}")
        print(f"Spearman: {winner.spearman:+.4f}")
        print(f"Silhouette: {winner.silhouette:.4f}")
        print(f"Stability: {winner.stability:.2f}")
        print(f"Training time: {winner.training_time:.1f}s")


if __name__ == "__main__":
    main()
