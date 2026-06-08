#!/usr/bin/env python
"""Combination Sweep: Test All Promising VAE Configurations.

This script systematically tests combinations of:
- Model architectures (SimpleVAE, Hyperbolic, Tropical, etc.)
- Loss functions (P-adic, Radial, Geodesic, etc.)
- Training strategies (Beta schedules, Curriculum, etc.)
- Advanced components (Controllers, Feedback, etc.)

Run with:
    python src/scripts/experiments/combination_sweep.py --phase 1 --workers 4
    python src/scripts/experiments/combination_sweep.py --top20 --workers 4
    python src/scripts/experiments/combination_sweep.py --cascade --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    # Architecture
    model_type: str = "simple_hyperbolic"  # simple, simple_hyperbolic, tropical, swarm
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [64, 32])
    # Projection
    use_hyperbolic: bool = True
    curvature: float = 1.0
    max_radius: float = 0.95
    # Loss
    padic_loss_type: str = "none"  # none, triplet, soft_ranking, geodesic, contrastive
    padic_weight: float = 0.3
    radial_loss_type: str = "none"  # none, stratification, hierarchy, monotonic, global_rank
    radial_weight: float = 0.3
    use_fisher_rao: bool = False
    fisher_rao_weight: float = 0.1
    # Training
    beta: float = 0.01
    beta_schedule: str = "constant"  # constant, warmup, cyclical
    learning_rate: float = 1e-3
    epochs: int = 50
    # Advanced
    use_curriculum: bool = False
    use_controller: bool = False
    use_feedback: bool = False


@dataclass
class ExperimentResult:
    """Results from experiment."""

    name: str
    accuracy: float
    spearman: float
    silhouette: float
    training_time: float
    final_loss: float
    config: ExperimentConfig
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
        kmeans = KMeans(n_clusters=27, random_state=42, n_init=10)
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


def create_model(config: ExperimentConfig):
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
    elif config.model_type == "tropical":
        # Tropical VAE wrapper
        from src.models.simple_vae import SimpleVAE

        # For now, use simple VAE as base (tropical layers would need integration)
        return SimpleVAE(
            input_dim=9,
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims,
        )
    elif config.model_type == "optimal":
        from src.models.optimal_vae import OptimalVAE, OptimalVAEConfig

        return OptimalVAE(
            OptimalVAEConfig(
                latent_dim=config.latent_dim,
                hidden_dims=config.hidden_dims,
                enable_hyperbolic=config.use_hyperbolic,
                enable_padic_ranking=config.padic_loss_type != "none",
            )
        )
    else:
        from src.models.simple_vae import SimpleVAE

        return SimpleVAE(input_dim=9, latent_dim=config.latent_dim)


# =============================================================================
# LOSS FACTORY
# =============================================================================


def create_padic_loss(config: ExperimentConfig):
    """Create p-adic loss based on configuration."""
    if config.padic_loss_type == "none":
        return None
    elif config.padic_loss_type == "triplet":
        from src.losses.padic import PAdicRankingLoss

        return PAdicRankingLoss(margin=0.1, n_triplets=500)
    elif config.padic_loss_type == "soft_ranking":
        # Use the soft ranking implementation we created earlier
        return SoftPadicRankingLoss(temperature=0.5)
    elif config.padic_loss_type == "geodesic":
        from src.losses.padic_geodesic import PAdicGeodesicLoss

        return PAdicGeodesicLoss(curvature=config.curvature)
    elif config.padic_loss_type == "contrastive":
        return ContrastivePadicLoss(temperature=0.1)
    else:
        return None


def create_radial_loss(config: ExperimentConfig):
    """Create radial loss based on configuration."""
    if config.radial_loss_type == "none":
        return None
    elif config.radial_loss_type == "stratification":
        from src.losses.radial_stratification import RadialStratificationLoss

        return RadialStratificationLoss()
    elif config.radial_loss_type == "hierarchy":
        from src.losses.padic_geodesic import RadialHierarchyLoss

        return RadialHierarchyLoss()
    elif config.radial_loss_type == "monotonic":
        from src.losses.padic_geodesic import MonotonicRadialLoss

        return MonotonicRadialLoss()
    elif config.radial_loss_type == "global_rank":
        from src.losses.padic_geodesic import GlobalRankLoss

        return GlobalRankLoss()
    else:
        return None


# =============================================================================
# P-ADIC LOSS IMPLEMENTATIONS (from our earlier work)
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


class ContrastivePadicLoss(nn.Module):
    """InfoNCE-style contrastive loss with p-adic similarity."""

    def __init__(self, temperature: float = 0.1, n_samples: int = 200):
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

        z_norm = F.normalize(z, dim=1)
        sim = torch.mm(z_norm, z_norm.t()) / self.temperature

        target_sim = torch.zeros(n, n, device=z.device)
        for i in range(n):
            for j in range(n):
                d = compute_padic_distance(indices[i].item(), indices[j].item())
                target_sim[i, j] = 1.0 / (1.0 + d)

        target_probs = F.softmax(target_sim / 0.1, dim=1)
        log_probs = F.log_softmax(sim, dim=1)
        loss = -(target_probs * log_probs).sum(dim=1).mean()

        return loss


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run a single experiment configuration."""
    from src.data.generation import generate_all_ternary_operations
    from src.losses.dual_vae_loss import KLDivergenceLoss, ReconstructionLoss

    start_time = time.time()

    try:
        # Load data
        ops = torch.tensor(generate_all_ternary_operations()[:1000], dtype=torch.float32)
        indices = torch.arange(len(ops))

        # Create model
        model = create_model(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        # Create losses
        recon_fn = ReconstructionLoss()
        kl_fn = KLDivergenceLoss()
        padic_fn = create_padic_loss(config)
        radial_fn = create_radial_loss(config)

        # Training loop
        for epoch in range(config.epochs):
            model.train()
            outputs = model(ops)

            # Base losses
            logits = outputs["logits"]
            mu = outputs["mu"]
            logvar = outputs["logvar"]
            z = outputs.get("z_hyp", outputs.get("z_euc", outputs["z"]))

            recon = recon_fn(logits, ops)
            kl = kl_fn(mu, logvar)

            # Beta schedule
            if config.beta_schedule == "warmup":
                beta = config.beta * min(1.0, epoch / 20)
            elif config.beta_schedule == "cyclical":
                beta = config.beta * (0.5 + 0.5 * np.sin(2 * np.pi * epoch / 50))
            else:
                beta = config.beta

            loss = recon + beta * kl

            # P-adic loss
            if padic_fn is not None:
                padic = padic_fn(z, indices)
                if isinstance(padic, tuple):
                    padic = padic[0]  # Some losses return (loss, metrics)
                loss = loss + config.padic_weight * padic

            # Radial loss
            if radial_fn is not None:
                radial = radial_fn(z, indices)
                if isinstance(radial, tuple):
                    radial = radial[0]
                loss = loss + config.radial_weight * radial

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            outputs = model(ops)
            z = outputs.get("z_euc", outputs["z"])

            pred = outputs["logits"].argmax(dim=-1)
            target = (ops + 1).long()
            accuracy = (pred == target).float().mean().item()

            metrics = evaluate_embeddings(z, indices)

        training_time = time.time() - start_time

        return ExperimentResult(
            name=config.name,
            accuracy=accuracy,
            spearman=metrics["spearman"],
            silhouette=metrics["silhouette"],
            training_time=training_time,
            final_loss=loss.item(),
            config=config,
        )

    except Exception as e:
        return ExperimentResult(
            name=config.name,
            accuracy=0.0,
            spearman=0.0,
            silhouette=0.0,
            training_time=time.time() - start_time,
            final_loss=float("inf"),
            config=config,
            error=str(e),
        )


# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================


def get_phase1_experiments() -> list[ExperimentConfig]:
    """Phase 1: Architecture × Projection sweep."""
    experiments = []

    # Model types to test
    model_types = ["simple", "simple_hyperbolic", "tropical"]

    # Projection variants (via use_hyperbolic and curvature)
    projections = [
        ("euclidean", False, 1.0),
        ("hyperbolic_c1", True, 1.0),
        ("hyperbolic_c05", True, 0.5),
        ("hyperbolic_c2", True, 2.0),
    ]

    for model_type in model_types:
        for proj_name, use_hyp, curv in projections:
            # Skip invalid combos
            if model_type == "simple" and use_hyp:
                continue  # SimpleVAE doesn't have hyperbolic
            if model_type == "simple_hyperbolic" and not use_hyp:
                continue  # Hyperbolic VAE needs hyperbolic

            name = f"{model_type}_{proj_name}"
            experiments.append(
                ExperimentConfig(
                    name=name,
                    model_type=model_type,
                    use_hyperbolic=use_hyp,
                    curvature=curv,
                )
            )

    return experiments


def get_phase2_experiments(base_model: str = "simple_hyperbolic") -> list[ExperimentConfig]:
    """Phase 2: Loss combination sweep."""
    experiments = []

    padic_types = ["none", "triplet", "soft_ranking", "geodesic", "contrastive"]
    radial_types = ["none", "hierarchy", "monotonic", "global_rank"]

    for padic in padic_types:
        for radial in radial_types:
            name = f"{base_model}_padic_{padic}_radial_{radial}"
            experiments.append(
                ExperimentConfig(
                    name=name,
                    model_type=base_model,
                    use_hyperbolic=True,
                    padic_loss_type=padic,
                    radial_loss_type=radial,
                )
            )

    return experiments


def get_phase3_experiments(base_config: dict[str, Any] = None) -> list[ExperimentConfig]:
    """Phase 3: Training strategy sweep."""
    experiments = []

    betas = [0.001, 0.01, 0.1]
    schedules = ["constant", "warmup", "cyclical"]
    lrs = [1e-4, 1e-3, 5e-3]

    for beta in betas:
        for schedule in schedules:
            for lr in lrs:
                name = f"beta_{beta}_sched_{schedule}_lr_{lr}"
                experiments.append(
                    ExperimentConfig(
                        name=name,
                        model_type="simple_hyperbolic",
                        use_hyperbolic=True,
                        beta=beta,
                        beta_schedule=schedule,
                        learning_rate=lr,
                    )
                )

    return experiments


def get_top20_experiments() -> list[ExperimentConfig]:
    """Top 20 most promising configurations."""
    return [
        # Verified best
        ExperimentConfig(name="01_hyp_no_padic", model_type="simple_hyperbolic", padic_loss_type="none"),
        # Soft ranking variants
        ExperimentConfig(name="02_hyp_soft_ranking", model_type="simple_hyperbolic", padic_loss_type="soft_ranking"),
        ExperimentConfig(
            name="03_hyp_soft_w05", model_type="simple_hyperbolic", padic_loss_type="soft_ranking", padic_weight=0.5
        ),
        ExperimentConfig(
            name="04_hyp_soft_w01", model_type="simple_hyperbolic", padic_loss_type="soft_ranking", padic_weight=0.1
        ),
        # Geodesic variants
        ExperimentConfig(name="05_hyp_geodesic", model_type="simple_hyperbolic", padic_loss_type="geodesic"),
        ExperimentConfig(
            name="06_hyp_geodesic_w05", model_type="simple_hyperbolic", padic_loss_type="geodesic", padic_weight=0.5
        ),
        # Radial variants
        ExperimentConfig(name="07_hyp_monotonic", model_type="simple_hyperbolic", radial_loss_type="monotonic"),
        ExperimentConfig(name="08_hyp_global_rank", model_type="simple_hyperbolic", radial_loss_type="global_rank"),
        ExperimentConfig(name="09_hyp_hierarchy", model_type="simple_hyperbolic", radial_loss_type="hierarchy"),
        # Combined
        ExperimentConfig(
            name="10_hyp_soft_monotonic",
            model_type="simple_hyperbolic",
            padic_loss_type="soft_ranking",
            radial_loss_type="monotonic",
        ),
        ExperimentConfig(
            name="11_hyp_geodesic_monotonic",
            model_type="simple_hyperbolic",
            padic_loss_type="geodesic",
            radial_loss_type="monotonic",
        ),
        # Training variants
        ExperimentConfig(name="12_hyp_warmup", model_type="simple_hyperbolic", beta_schedule="warmup"),
        ExperimentConfig(name="13_hyp_cyclical", model_type="simple_hyperbolic", beta_schedule="cyclical"),
        ExperimentConfig(name="14_hyp_beta001", model_type="simple_hyperbolic", beta=0.001),
        ExperimentConfig(name="15_hyp_beta01", model_type="simple_hyperbolic", beta=0.1),
        # Curvature variants
        ExperimentConfig(name="16_hyp_curv05", model_type="simple_hyperbolic", curvature=0.5),
        ExperimentConfig(name="17_hyp_curv2", model_type="simple_hyperbolic", curvature=2.0),
        # Architecture variants
        ExperimentConfig(name="18_hyp_latent32", model_type="simple_hyperbolic", latent_dim=32),
        ExperimentConfig(name="19_hyp_deeper", model_type="simple_hyperbolic", hidden_dims=[128, 64, 32]),
        # Baseline comparison
        ExperimentConfig(name="20_simple_baseline", model_type="simple", use_hyperbolic=False),
    ]


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Combination Sweep")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Run specific phase")
    parser.add_argument("--top20", action="store_true", help="Run top 20 promising configs")
    parser.add_argument("--cascade", action="store_true", help="Run full cascade (all phases)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially (debug)")
    parser.add_argument("--output", type=str, default="combination_results.json", help="Output file")

    args = parser.parse_args()

    # Select experiments
    if args.top20:
        experiments = get_top20_experiments()
        phase_name = "Top 20"
    elif args.phase == 1:
        experiments = get_phase1_experiments()
        phase_name = "Phase 1 (Architecture)"
    elif args.phase == 2:
        experiments = get_phase2_experiments()
        phase_name = "Phase 2 (Losses)"
    elif args.phase == 3:
        experiments = get_phase3_experiments()
        phase_name = "Phase 3 (Training)"
    else:
        experiments = get_top20_experiments()
        phase_name = "Top 20 (default)"

    print("\n" + "=" * 70)
    print(f"COMBINATION SWEEP: {phase_name}")
    print("=" * 70)
    print(f"Running {len(experiments)} experiments with {args.workers} workers")
    print("=" * 70 + "\n")

    results = []

    if args.sequential:
        for exp in experiments:
            print(f"Running {exp.name}...", end=" ", flush=True)
            result = run_experiment(exp)
            results.append(result)
            if result.error:
                print(f"ERROR: {result.error}")
            else:
                print(f"acc={result.accuracy:.1%}, corr={result.spearman:+.4f}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_experiment, exp): exp.name for exp in experiments}

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result.error:
                        print(f"[FAIL] {name}: {result.error}")
                    else:
                        print(f"[DONE] {name}: acc={result.accuracy:.1%}, corr={result.spearman:+.4f}")
                except Exception as e:
                    print(f"[FAIL] {name}: {e}")

    # Sort by composite score
    def score(r):
        if r.error:
            return -999
        return 0.4 * r.accuracy + 0.3 * r.spearman + 0.15 * r.silhouette

    results.sort(key=lambda r: -score(r))

    # Print results table
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY (sorted by composite score)")
    print("=" * 90)
    print(f"{'Rank':<5} {'Experiment':<35} {'Accuracy':>10} {'Spearman':>10} {'Silhouette':>10} {'Score':>8}")
    print("-" * 90)

    for i, r in enumerate(results[:20], 1):
        if r.error:
            print(f"{i:<5} {r.name:<35} {'ERROR':<10} {r.error[:40]}")
        else:
            s = score(r)
            print(f"{i:<5} {r.name:<35} {r.accuracy:>9.1%} {r.spearman:>+10.4f} {r.silhouette:>10.4f} {s:>8.4f}")

    # Save results
    results_data = []
    for r in results:
        results_data.append(
            {
                "name": r.name,
                "accuracy": r.accuracy,
                "spearman": r.spearman,
                "silhouette": r.silhouette,
                "training_time": r.training_time,
                "final_loss": r.final_loss,
                "error": r.error,
                "config": {
                    "model_type": r.config.model_type,
                    "latent_dim": r.config.latent_dim,
                    "use_hyperbolic": r.config.use_hyperbolic,
                    "curvature": r.config.curvature,
                    "padic_loss_type": r.config.padic_loss_type,
                    "padic_weight": r.config.padic_weight,
                    "radial_loss_type": r.config.radial_loss_type,
                    "beta": r.config.beta,
                    "beta_schedule": r.config.beta_schedule,
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
        print(f"Training time: {winner.training_time:.1f}s")


if __name__ == "__main__":
    main()
