#!/usr/bin/env python
"""Test TropicalVAE with p-adic losses - Key Hypothesis.

This tests the combination of:
- TropicalVAE: Best accuracy (96.7%)
- P-adic losses: Best correlation (+0.5465)

Hypothesis: Tropical geometry + p-adic losses could achieve BOTH high accuracy AND high correlation
because both have ultrametric/tree-like structure.

Run with:
    python src/scripts/experiments/tropical_padic_experiment.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass
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
# CONFIGURATION
# =============================================================================


@dataclass
class ExperimentConfig:
    """Configuration for tropical + p-adic experiment."""

    name: str
    # Model
    model_type: str = "tropical"
    latent_dim: int = 16
    hidden_dim: int = 64
    # Loss
    padic_loss_type: str = "none"
    padic_weight: float = 0.3
    radial_loss_type: str = "none"
    radial_weight: float = 0.3
    # Training
    beta: float = 0.1
    beta_schedule: str = "cyclical"
    learning_rate: float = 0.005
    epochs: int = 80  # More epochs for better convergence


@dataclass
class ExperimentResult:
    """Result from experiment."""

    name: str
    accuracy: float
    spearman: float
    silhouette: float
    training_time: float
    final_loss: float
    config: ExperimentConfig
    error: str | None = None
    history: dict[str, list[float]] | None = None


# =============================================================================
# HELPERS
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

    # Spearman correlation between p-adic and latent distances
    n_pairs = min(3000, n * (n - 1) // 2)
    np.random.seed(42)
    i_idx = np.random.randint(0, n, n_pairs)
    j_idx = np.random.randint(0, n, n_pairs)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]

    padic_dists = [compute_padic_distance(indices_np[i], indices_np[j]) for i, j in zip(i_idx, j_idx, strict=False)]
    latent_dists = np.linalg.norm(z_np[i_idx] - z_np[j_idx], axis=1)

    corr, _ = spearmanr(padic_dists, latent_dists)

    # Silhouette score
    try:
        n_samples = min(500, n)
        sample_idx = np.random.choice(n, n_samples, replace=False)
        n_clusters = min(27, n_samples // 10)
        if n_clusters >= 2:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(z_np[sample_idx])
            silhouette = silhouette_score(z_np[sample_idx], labels)
        else:
            silhouette = 0.0
    except Exception:
        silhouette = 0.0

    return {
        "spearman": corr if not np.isnan(corr) else 0.0,
        "silhouette": silhouette,
    }


# =============================================================================
# TROPICAL VAE WITH ENHANCED STRUCTURE
# =============================================================================


class EnhancedTropicalVAE(nn.Module):
    """Enhanced TropicalVAE with stronger tropical structure.

    Uses tropical (max-plus) algebra operations:
    - Tropical addition: max(a, b)
    - Tropical multiplication: a + b

    This creates piecewise-linear mappings that naturally preserve
    tree-like (ultrametric) structure.
    """

    def __init__(
        self,
        input_dim: int = 9,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.temperature = temperature

        # Encoder with tropical-inspired structure
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),  # ReLU approximates tropical max
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

        # Tropical layer: uses logsumexp as smooth approximation to max
        self.tropical_transform = nn.Linear(latent_dim, latent_dim)

        # Decoder with tropical structure
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * 3),
        )

    def tropical_aggregate(self, z: torch.Tensor) -> torch.Tensor:
        """Apply tropical-like aggregation using smooth max (logsumexp)."""
        # Transform to multiple "tropical" representations
        z_transformed = self.tropical_transform(z)

        # Tropical-style combination: logsumexp approximates max
        # Stack and apply logsumexp along a new dimension
        combined = torch.stack([z, z_transformed], dim=-1)
        z_tropical = torch.logsumexp(combined / self.temperature, dim=-1) * self.temperature

        return z_tropical

    def forward(self, x: torch.Tensor) -> dict[str, Any]:
        # Encode
        h = self.encoder(x)
        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h)

        # Reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Apply tropical transformation
        z_tropical = self.tropical_aggregate(z)

        # Decode
        logits = self.decoder(z_tropical).view(-1, self.input_dim, 3)

        return {
            "logits": logits,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "z_euc": z,  # Use pre-tropical for p-adic loss
            "z_tropical": z_tropical,
        }


# =============================================================================
# SOFT P-ADIC RANKING LOSS
# =============================================================================


class SoftPadicRankingLoss(nn.Module):
    """Soft p-adic ranking using KL divergence between rank distributions."""

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
# LOSS FACTORIES
# =============================================================================


def create_padic_loss(config: ExperimentConfig):
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

        return PAdicGeodesicLoss(curvature=1.0)
    return None


def create_radial_loss(config: ExperimentConfig):
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


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run a single experiment with detailed tracking."""
    from src.data.generation import generate_all_ternary_operations
    from src.losses.dual_vae_loss import KLDivergenceLoss, ReconstructionLoss

    start_time = time.time()
    history = {"loss": [], "accuracy": [], "spearman": [], "recon": [], "kl": [], "padic": [], "radial": []}

    try:
        # Load data
        ops = torch.tensor(generate_all_ternary_operations()[:1000], dtype=torch.float32)
        indices = torch.arange(len(ops))

        # Create model
        if config.model_type == "tropical":
            model = EnhancedTropicalVAE(
                input_dim=9,
                latent_dim=config.latent_dim,
                hidden_dim=config.hidden_dim,
            )
        elif config.model_type == "simple_hyperbolic":
            from src.models.simple_vae import SimpleVAEWithHyperbolic

            model = SimpleVAEWithHyperbolic(
                input_dim=9,
                latent_dim=config.latent_dim,
                hidden_dims=[config.hidden_dim, config.hidden_dim // 2],
            )
        else:
            from src.models.simple_vae import SimpleVAE

            model = SimpleVAE(
                input_dim=9,
                latent_dim=config.latent_dim,
                hidden_dims=[config.hidden_dim, config.hidden_dim // 2],
            )

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        # Create losses
        recon_fn = ReconstructionLoss()
        kl_fn = KLDivergenceLoss()
        padic_fn = create_padic_loss(config)
        radial_fn = create_radial_loss(config)

        # Training loop with tracking
        for epoch in range(config.epochs):
            model.train()
            outputs = model(ops)

            logits = outputs["logits"]
            mu = outputs["mu"]
            logvar = outputs["logvar"]
            z = outputs.get("z_euc", outputs["z"])  # Use euclidean for p-adic

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
            padic_val = 0.0
            if padic_fn is not None:
                padic = padic_fn(z, indices)
                if isinstance(padic, tuple):
                    padic = padic[0]
                padic_val = padic.item()
                loss = loss + config.padic_weight * padic

            # Radial loss
            radial_val = 0.0
            if radial_fn is not None:
                radial = radial_fn(z, indices)
                if isinstance(radial, tuple):
                    radial = radial[0]
                radial_val = radial.item()
                loss = loss + config.radial_weight * radial

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track every 10 epochs
            if epoch % 10 == 0:
                with torch.no_grad():
                    pred = logits.argmax(dim=-1)
                    target = (ops + 1).long()
                    acc = (pred == target).float().mean().item()
                    metrics = evaluate_embeddings(z.detach(), indices)

                    history["loss"].append(loss.item())
                    history["accuracy"].append(acc)
                    history["spearman"].append(metrics["spearman"])
                    history["recon"].append(recon.item())
                    history["kl"].append(kl.item())
                    history["padic"].append(padic_val)
                    history["radial"].append(radial_val)

        # Final evaluation
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
            history=history,
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
            error=f"{str(e)}\n{traceback.format_exc()[:500]}",
        )


# =============================================================================
# EXPERIMENT DEFINITIONS
# =============================================================================


def get_experiments() -> list[ExperimentConfig]:
    """Define all experiments to test TropicalVAE + p-adic losses."""
    experiments = []

    # Baselines (for comparison)
    experiments.append(
        ExperimentConfig(
            name="baseline_tropical",
            model_type="tropical",
            padic_loss_type="none",
            radial_loss_type="none",
        )
    )

    experiments.append(
        ExperimentConfig(
            name="baseline_simple_triplet_mono",
            model_type="simple_hyperbolic",
            padic_loss_type="triplet",
            radial_loss_type="monotonic",
        )
    )

    # KEY HYPOTHESIS: TropicalVAE + triplet + monotonic
    experiments.append(
        ExperimentConfig(
            name="tropical_triplet_monotonic",
            model_type="tropical",
            padic_loss_type="triplet",
            radial_loss_type="monotonic",
        )
    )

    # Alternative: TropicalVAE + soft_ranking + hierarchy
    experiments.append(
        ExperimentConfig(
            name="tropical_soft_hierarchy",
            model_type="tropical",
            padic_loss_type="soft_ranking",
            radial_loss_type="hierarchy",
        )
    )

    # TropicalVAE with individual losses
    experiments.append(
        ExperimentConfig(
            name="tropical_triplet_only",
            model_type="tropical",
            padic_loss_type="triplet",
            radial_loss_type="none",
        )
    )

    experiments.append(
        ExperimentConfig(
            name="tropical_monotonic_only",
            model_type="tropical",
            padic_loss_type="none",
            radial_loss_type="monotonic",
        )
    )

    experiments.append(
        ExperimentConfig(
            name="tropical_soft_ranking_only",
            model_type="tropical",
            padic_loss_type="soft_ranking",
            radial_loss_type="none",
        )
    )

    experiments.append(
        ExperimentConfig(
            name="tropical_hierarchy_only",
            model_type="tropical",
            padic_loss_type="none",
            radial_loss_type="hierarchy",
        )
    )

    # Hyperparameter variations for best combination
    for padic_weight in [0.1, 0.3, 0.5]:
        for radial_weight in [0.1, 0.3, 0.5]:
            if padic_weight == 0.3 and radial_weight == 0.3:
                continue  # Already tested above
            experiments.append(
                ExperimentConfig(
                    name=f"tropical_triplet_mono_pw{padic_weight}_rw{radial_weight}",
                    model_type="tropical",
                    padic_loss_type="triplet",
                    radial_loss_type="monotonic",
                    padic_weight=padic_weight,
                    radial_weight=radial_weight,
                )
            )

    return experiments


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("\n" + "=" * 80)
    print("TROPICAL + P-ADIC EXPERIMENT")
    print("=" * 80)
    print("Testing key hypothesis: TropicalVAE + p-adic losses")
    print("Goal: Achieve BOTH high accuracy (like TropicalVAE: 96.7%)")
    print("      AND high correlation (like triplet+monotonic: +0.5465)")
    print("=" * 80 + "\n")

    experiments = get_experiments()
    print(f"Running {len(experiments)} experiments\n")

    results = []
    for i, exp in enumerate(experiments, 1):
        print(f"[{i}/{len(experiments)}] Running {exp.name}...", end=" ", flush=True)
        result = run_experiment(exp)
        results.append(result)

        if result.error:
            print(f"ERROR: {result.error[:60]}")
        else:
            print(f"acc={result.accuracy:.1%}, corr={result.spearman:+.4f}, time={result.training_time:.1f}s")

    # Sort by composite score
    def score(r):
        if r.error:
            return -999
        return 0.4 * r.accuracy + 0.4 * r.spearman + 0.2 * r.silhouette

    results.sort(key=lambda r: -score(r))

    # Print summary
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Rank':<5} {'Experiment':<45} {'Accuracy':>10} {'Spearman':>10} {'Silhouette':>10} {'Score':>8}")
    print("-" * 100)

    for i, r in enumerate(results, 1):
        if r.error:
            print(f"{i:<5} {r.name:<45} {'ERROR':<10} {r.error[:40]}")
        else:
            s = score(r)
            print(f"{i:<5} {r.name:<45} {r.accuracy:>9.1%} {r.spearman:>+10.4f} {r.silhouette:>10.4f} {s:>8.4f}")

    # Key comparison
    print("\n" + "=" * 80)
    print("KEY COMPARISON: Previous Best vs New Combinations")
    print("=" * 80)

    # Find specific results
    tropical_baseline = next((r for r in results if r.name == "baseline_tropical" and not r.error), None)
    simple_baseline = next((r for r in results if r.name == "baseline_simple_triplet_mono" and not r.error), None)
    tropical_triplet_mono = next((r for r in results if r.name == "tropical_triplet_monotonic" and not r.error), None)
    tropical_soft_hier = next((r for r in results if r.name == "tropical_soft_hierarchy" and not r.error), None)

    print("\nPrevious Results (from comprehensive sweep):")
    print("  TropicalVAE alone:           96.7% acc, +0.21 corr")
    print("  SimpleHyp+triplet+monotonic: 74.8% acc, +0.55 corr")

    print("\nNew Results (from this experiment):")
    if tropical_baseline:
        print(
            f"  TropicalVAE (this run):      {tropical_baseline.accuracy:.1%} acc, {tropical_baseline.spearman:+.2f} corr"
        )
    if simple_baseline:
        print(
            f"  SimpleHyp+triplet+mono:      {simple_baseline.accuracy:.1%} acc, {simple_baseline.spearman:+.2f} corr"
        )
    if tropical_triplet_mono:
        print(
            f"  TropicalVAE+triplet+mono:    {tropical_triplet_mono.accuracy:.1%} acc, {tropical_triplet_mono.spearman:+.2f} corr  <-- KEY TEST"
        )
    if tropical_soft_hier:
        print(
            f"  TropicalVAE+soft+hierarchy:  {tropical_soft_hier.accuracy:.1%} acc, {tropical_soft_hier.spearman:+.2f} corr"
        )

    # Conclusion
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    best = results[0] if results else None
    if best and not best.error:
        print(f"\nBest overall: {best.name}")
        print(f"  Accuracy:   {best.accuracy:.1%}")
        print(f"  Spearman:   {best.spearman:+.4f}")
        print(f"  Silhouette: {best.silhouette:.4f}")
        print(f"  Score:      {score(best):.4f}")

        if tropical_triplet_mono and not tropical_triplet_mono.error:
            if tropical_triplet_mono.accuracy > 0.90 and tropical_triplet_mono.spearman > 0.40:
                print("\n*** HYPOTHESIS CONFIRMED: Tropical + p-adic achieves both high accuracy AND correlation! ***")
            elif tropical_triplet_mono.accuracy > 0.85:
                print("\n*** Tropical + p-adic maintains high accuracy with improved structure ***")
            else:
                print("\n*** Tropical + p-adic shows trade-offs - further tuning needed ***")

    # Save results
    output_path = PROJECT_ROOT / "outputs" / "tropical_padic_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
                    "padic_loss_type": r.config.padic_loss_type,
                    "radial_loss_type": r.config.radial_loss_type,
                    "padic_weight": r.config.padic_weight,
                    "radial_weight": r.config.radial_weight,
                    "beta": r.config.beta,
                    "beta_schedule": r.config.beta_schedule,
                    "learning_rate": r.config.learning_rate,
                },
                "history": r.history,
            }
        )

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
