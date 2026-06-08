#!/usr/bin/env python
"""Comprehensive Parallel Analysis of VAE Configuration Space.

Tests all key variables and their interactions to understand:
1. How reconstruction and p-adic losses interact
2. Effect of hyperbolic projection
3. Beta (KL weight) sensitivity
4. P-adic weight sensitivity
5. Two-stage training effectiveness
6. Architecture effects

Runs experiments in parallel and generates comprehensive analysis.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    epochs: int = 50
    beta: float = 0.01
    padic_weight: float = 0.0
    use_hyperbolic: bool = False
    use_padic: bool = False
    hidden_dims: tuple = (64, 32)
    latent_dim: int = 16
    freeze_decoder_at: int | None = None  # Epoch to freeze decoder
    recon_weight: float = 1.0  # Can reduce reconstruction weight


@dataclass
class ExperimentResult:
    """Results from experiment."""

    name: str
    final_accuracy: float
    final_recon_loss: float
    final_padic_loss: float
    spearman_correlation: float
    silhouette_score: float
    training_time: float
    config: ExperimentConfig


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


def evaluate_embeddings(embeddings: np.ndarray, n_samples: int = 5000) -> dict:
    """Evaluate embedding quality."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(embeddings)

    # Sample pairs for correlation
    np.random.seed(42)
    i_idx = np.random.randint(0, n, n_samples)
    j_idx = np.random.randint(0, n, n_samples)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]

    padic_dists = np.array([compute_padic_distance(i, j) for i, j in zip(i_idx, j_idx, strict=False)])
    latent_dists = np.linalg.norm(embeddings[i_idx] - embeddings[j_idx], axis=1)

    corr, _ = spearmanr(padic_dists, latent_dists)

    # Clustering
    try:
        kmeans = KMeans(n_clusters=27, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings[:2000])  # Subsample for speed
        silhouette = silhouette_score(embeddings[:2000], labels)
    except:
        silhouette = 0.0

    return {
        "spearman": corr,
        "silhouette": silhouette,
    }


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run a single experiment."""
    from src.data.generation import generate_all_ternary_operations
    from src.losses.dual_vae_loss import KLDivergenceLoss, ReconstructionLoss
    from src.losses.padic import PAdicRankingLoss
    from src.models.simple_vae import SimpleVAE, SimpleVAEWithHyperbolic
    from src.training import TernaryDataset

    start_time = time.time()

    # Load data
    ops = torch.tensor(generate_all_ternary_operations(), dtype=torch.float32)
    indices = torch.arange(len(ops))
    dataset = TernaryDataset(ops, indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    # Create model
    if config.use_hyperbolic:
        model = SimpleVAEWithHyperbolic(
            input_dim=9,
            latent_dim=config.latent_dim,
            hidden_dims=list(config.hidden_dims),
        )
    else:
        model = SimpleVAE(
            input_dim=9,
            latent_dim=config.latent_dim,
            hidden_dims=list(config.hidden_dims),
        )

    # Loss functions
    recon_loss_fn = ReconstructionLoss()
    kl_loss_fn = KLDivergenceLoss()
    padic_loss_fn = PAdicRankingLoss() if config.use_padic else None

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training
    final_recon = 0.0
    final_padic = 0.0
    final_acc = 0.0

    for epoch in range(config.epochs):
        model.train()

        # Freeze decoder if specified
        if config.freeze_decoder_at is not None and epoch == config.freeze_decoder_at:
            for param in model.decoder.parameters():
                param.requires_grad = False
            # Re-create optimizer with only encoder params
            optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

        epoch_recon = 0.0
        epoch_padic = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch in loader:
            x = batch["operation"]
            batch_idx = batch["index"]

            optimizer.zero_grad()
            outputs = model(x)

            logits = outputs["logits"]
            mu = outputs["mu"]
            logvar = outputs["logvar"]
            z = outputs.get("z_hyp", outputs["z"]) if config.use_hyperbolic else outputs["z"]

            # Reconstruction loss
            recon = recon_loss_fn(logits, x)
            kl = kl_loss_fn(mu, logvar)

            loss = config.recon_weight * recon + config.beta * kl

            # P-adic loss
            if padic_loss_fn is not None and config.padic_weight > 0:
                padic = padic_loss_fn(z, batch_idx)
                loss = loss + config.padic_weight * padic
                epoch_padic += padic.item()

            loss.backward()
            optimizer.step()

            epoch_recon += recon.item()

            # Accuracy
            pred = logits.argmax(dim=-1)
            target = (x + 1).long()
            epoch_correct += (pred == target).float().sum().item()
            epoch_total += x.size(0) * 9

        final_recon = epoch_recon / len(loader)
        final_padic = epoch_padic / len(loader) if epoch_padic > 0 else 0.0
        final_acc = epoch_correct / epoch_total

    # Evaluate embeddings
    model.eval()
    all_z = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["operation"])
            z = outputs.get("z_euc", outputs["z"])
            all_z.append(z)
    embeddings = torch.cat(all_z, dim=0).numpy()

    metrics = evaluate_embeddings(embeddings)

    return ExperimentResult(
        name=config.name,
        final_accuracy=final_acc,
        final_recon_loss=final_recon,
        final_padic_loss=final_padic,
        spearman_correlation=metrics["spearman"],
        silhouette_score=metrics["silhouette"],
        training_time=time.time() - start_time,
        config=config,
    )


# Define all experiments
EXPERIMENTS = [
    # Baseline experiments
    ExperimentConfig(name="baseline_pure", epochs=50, beta=0.01, use_padic=False, use_hyperbolic=False),
    ExperimentConfig(name="hyperbolic_only", epochs=50, beta=0.01, use_padic=False, use_hyperbolic=True),
    ExperimentConfig(name="padic_only", epochs=50, beta=0.01, padic_weight=0.3, use_padic=True, use_hyperbolic=False),
    ExperimentConfig(name="hyp_padic", epochs=50, beta=0.01, padic_weight=0.3, use_padic=True, use_hyperbolic=True),
    # P-adic weight sweep
    ExperimentConfig(name="padic_w0.1", epochs=50, beta=0.01, padic_weight=0.1, use_padic=True, use_hyperbolic=True),
    ExperimentConfig(name="padic_w0.5", epochs=50, beta=0.01, padic_weight=0.5, use_padic=True, use_hyperbolic=True),
    ExperimentConfig(name="padic_w1.0", epochs=50, beta=0.01, padic_weight=1.0, use_padic=True, use_hyperbolic=True),
    ExperimentConfig(name="padic_w2.0", epochs=50, beta=0.01, padic_weight=2.0, use_padic=True, use_hyperbolic=True),
    # Beta sweep (KL weight)
    ExperimentConfig(name="beta_0.001", epochs=50, beta=0.001, padic_weight=0.3, use_padic=True, use_hyperbolic=True),
    ExperimentConfig(name="beta_0.1", epochs=50, beta=0.1, padic_weight=0.3, use_padic=True, use_hyperbolic=True),
    ExperimentConfig(name="beta_0.5", epochs=50, beta=0.5, padic_weight=0.3, use_padic=True, use_hyperbolic=True),
    # Two-stage training
    ExperimentConfig(
        name="2stage_freeze25",
        epochs=50,
        beta=0.01,
        padic_weight=0.5,
        use_padic=True,
        use_hyperbolic=True,
        freeze_decoder_at=25,
    ),
    ExperimentConfig(
        name="2stage_freeze10",
        epochs=50,
        beta=0.01,
        padic_weight=1.0,
        use_padic=True,
        use_hyperbolic=True,
        freeze_decoder_at=10,
    ),
    # Reduced reconstruction weight
    ExperimentConfig(
        name="recon_w0.5", epochs=50, beta=0.01, padic_weight=0.3, use_padic=True, use_hyperbolic=True, recon_weight=0.5
    ),
    ExperimentConfig(
        name="recon_w0.1", epochs=50, beta=0.01, padic_weight=0.3, use_padic=True, use_hyperbolic=True, recon_weight=0.1
    ),
    # Pure p-adic (no reconstruction)
    ExperimentConfig(
        name="pure_padic", epochs=50, beta=0.0, padic_weight=1.0, use_padic=True, use_hyperbolic=True, recon_weight=0.0
    ),
    # Architecture variations
    ExperimentConfig(
        name="deeper_net",
        epochs=50,
        beta=0.01,
        padic_weight=0.3,
        use_padic=True,
        use_hyperbolic=True,
        hidden_dims=(128, 64, 32),
    ),
    ExperimentConfig(
        name="wider_net",
        epochs=50,
        beta=0.01,
        padic_weight=0.3,
        use_padic=True,
        use_hyperbolic=True,
        hidden_dims=(128, 64),
    ),
    ExperimentConfig(
        name="latent32", epochs=50, beta=0.01, padic_weight=0.3, use_padic=True, use_hyperbolic=True, latent_dim=32
    ),
]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--sequential", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("COMPREHENSIVE VAE ANALYSIS")
    print("=" * 80)
    print(f"Running {len(EXPERIMENTS)} experiments")
    print("=" * 80 + "\n")

    results = []

    if args.sequential:
        for exp in EXPERIMENTS:
            print(f"Running {exp.name}...", end=" ", flush=True)
            result = run_experiment(exp)
            results.append(result)
            print(f"done (acc={result.final_accuracy:.1%}, corr={result.spearman_correlation:.4f})")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_experiment, exp): exp.name for exp in EXPERIMENTS}

            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"[DONE] {name}: acc={result.final_accuracy:.1%}, corr={result.spearman_correlation:.4f}")
                except Exception as e:
                    print(f"[FAIL] {name}: {e}")

    # Sort by name for consistent output
    results.sort(key=lambda r: r.name)

    # Print results table
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY")
    print("=" * 100)
    print(f"{'Experiment':<20} {'Acc':>8} {'Recon':>8} {'P-adic':>8} {'Spearman':>10} {'Silhouette':>10} {'Time':>8}")
    print("-" * 100)

    for r in results:
        print(
            f"{r.name:<20} {r.final_accuracy:>7.1%} {r.final_recon_loss:>8.4f} {r.final_padic_loss:>8.4f} {r.spearman_correlation:>10.4f} {r.silhouette_score:>10.4f} {r.training_time:>7.1f}s"
        )

    # Analysis sections
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Group by category
    baselines = [r for r in results if r.name in ["baseline_pure", "hyperbolic_only", "padic_only", "hyp_padic"]]
    padic_sweep = [r for r in results if r.name.startswith("padic_w")]
    beta_sweep = [r for r in results if r.name.startswith("beta_")]
    two_stage = [r for r in results if r.name.startswith("2stage_")]
    recon_sweep = [r for r in results if r.name.startswith("recon_w") or r.name == "pure_padic"]
    arch_sweep = [r for r in results if r.name in ["deeper_net", "wider_net", "latent32"]]

    print("\n1. BASELINE COMPARISON")
    print("-" * 40)
    for r in baselines:
        hyp = "H" if r.config.use_hyperbolic else "-"
        pad = "P" if r.config.use_padic else "-"
        print(f"  [{hyp}{pad}] {r.name:<18}: corr={r.spearman_correlation:+.4f}, acc={r.final_accuracy:.1%}")

    print("\n2. P-ADIC WEIGHT EFFECT")
    print("-" * 40)
    for r in sorted(padic_sweep, key=lambda x: x.config.padic_weight):
        print(
            f"  w={r.config.padic_weight:<4}: corr={r.spearman_correlation:+.4f}, padic_loss={r.final_padic_loss:.4f}, acc={r.final_accuracy:.1%}"
        )

    print("\n3. BETA (KL WEIGHT) EFFECT")
    print("-" * 40)
    for r in sorted(beta_sweep, key=lambda x: x.config.beta):
        print(f"  beta={r.config.beta:<5}: corr={r.spearman_correlation:+.4f}, acc={r.final_accuracy:.1%}")

    print("\n4. TWO-STAGE TRAINING")
    print("-" * 40)
    for r in two_stage:
        print(f"  {r.name}: corr={r.spearman_correlation:+.4f}, acc={r.final_accuracy:.1%}")

    print("\n5. RECONSTRUCTION WEIGHT EFFECT")
    print("-" * 40)
    for r in sorted(recon_sweep, key=lambda x: x.config.recon_weight):
        print(f"  recon_w={r.config.recon_weight:<4}: corr={r.spearman_correlation:+.4f}, acc={r.final_accuracy:.1%}")

    print("\n6. ARCHITECTURE EFFECT")
    print("-" * 40)
    for r in arch_sweep:
        print(f"  {r.name}: corr={r.spearman_correlation:+.4f}, sil={r.silhouette_score:.4f}")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    best_corr = max(results, key=lambda r: r.spearman_correlation)
    best_acc = max(results, key=lambda r: r.final_accuracy)
    best_combined = max(results, key=lambda r: r.spearman_correlation + r.final_accuracy)

    print(f"\n  Best correlation:     {best_corr.name} (corr={best_corr.spearman_correlation:+.4f})")
    print(f"  Best accuracy:        {best_acc.name} (acc={best_acc.final_accuracy:.1%})")
    print(
        f"  Best combined:        {best_combined.name} (corr={best_combined.spearman_correlation:+.4f}, acc={best_combined.final_accuracy:.1%})"
    )

    # Correlation vs Accuracy trade-off
    print("\n  TRADE-OFF ANALYSIS:")
    high_acc = [r for r in results if r.final_accuracy > 0.95]
    if high_acc:
        best_corr_high_acc = max(high_acc, key=lambda r: r.spearman_correlation)
        print(
            f"  Best corr with acc>95%: {best_corr_high_acc.name} (corr={best_corr_high_acc.spearman_correlation:+.4f})"
        )

    positive_corr = [r for r in results if r.spearman_correlation > 0]
    if positive_corr:
        print(f"\n  Experiments with POSITIVE correlation: {len(positive_corr)}/{len(results)}")
        for r in sorted(positive_corr, key=lambda x: -x.spearman_correlation):
            print(f"    {r.name}: {r.spearman_correlation:+.4f}")
    else:
        print("\n  No experiments achieved positive correlation")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
