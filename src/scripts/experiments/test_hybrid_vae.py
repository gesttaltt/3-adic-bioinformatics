#!/usr/bin/env python
"""Test TropicalHyperbolicVAE hybrid architecture.

Tests the new hybrid model that combines tropical and hyperbolic geometry.

Run with:
    python src/scripts/experiments/test_hybrid_vae.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tropical_hyperbolic_vae import TropicalHyperbolicVAE, TropicalHyperbolicVAELight


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


class SoftPadicRankingLoss(torch.nn.Module):
    """Soft p-adic ranking loss."""

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

        return F.kl_div(latent_ranks.log(), padic_ranks, reduction="batchmean")


def train_and_evaluate(model, ops, indices, config, epochs=80):
    """Train model and return metrics."""
    from src.losses.dual_vae_loss import KLDivergenceLoss, ReconstructionLoss
    from src.losses.padic import PAdicRankingLoss
    from src.losses.padic_geodesic import MonotonicRadialLoss

    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.005))

    recon_fn = ReconstructionLoss()
    kl_fn = KLDivergenceLoss()
    padic_fn = PAdicRankingLoss(margin=0.1, n_triplets=500)
    radial_fn = MonotonicRadialLoss()

    padic_weight = config.get("padic_weight", 0.5)
    radial_weight = config.get("radial_weight", 0.3)
    beta = config.get("beta", 0.1)

    history = {"loss": [], "accuracy": [], "spearman": []}

    for epoch in range(epochs):
        model.train()
        outputs = model(ops)

        logits = outputs["logits"]
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        z = outputs.get("z_euc", outputs["z"])

        recon = recon_fn(logits, ops)
        kl = kl_fn(mu, logvar)

        # Cyclical beta
        beta_t = beta * (0.5 + 0.5 * np.sin(2 * np.pi * epoch / 50))

        loss = recon + beta_t * kl

        # P-adic loss
        padic = padic_fn(z, indices)
        if isinstance(padic, tuple):
            padic = padic[0]
        loss = loss + padic_weight * padic

        # Radial loss
        radial = radial_fn(z, indices)
        if isinstance(radial, tuple):
            radial = radial[0]
        loss = loss + radial_weight * radial

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                target = (ops + 1).long()
                acc = (pred == target).float().mean().item()
                metrics = evaluate_embeddings(z.detach(), indices)
                history["loss"].append(loss.item())
                history["accuracy"].append(acc)
                history["spearman"].append(metrics["spearman"])

    # Final evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(ops)
        z = outputs.get("z_euc", outputs["z"])
        z_hyp = outputs.get("z_hyp", z)
        z_tropical = outputs.get("z_tropical", z)

        pred = outputs["logits"].argmax(dim=-1)
        target = (ops + 1).long()
        accuracy = (pred == target).float().mean().item()

        metrics_euc = evaluate_embeddings(z, indices)
        metrics_hyp = evaluate_embeddings(z_hyp, indices)
        metrics_tropical = evaluate_embeddings(z_tropical, indices)

    return {
        "accuracy": accuracy,
        "spearman_euc": metrics_euc["spearman"],
        "spearman_hyp": metrics_hyp["spearman"],
        "spearman_tropical": metrics_tropical["spearman"],
        "silhouette": metrics_euc["silhouette"],
        "history": history,
    }


def main():
    from src.data.generation import generate_all_ternary_operations

    print("\n" + "=" * 80)
    print("TROPICAL-HYPERBOLIC VAE HYBRID TEST")
    print("=" * 80)

    # Load data
    ops = torch.tensor(generate_all_ternary_operations()[:1000], dtype=torch.float32)
    indices = torch.arange(len(ops))

    results = []
    config = {"learning_rate": 0.005, "padic_weight": 0.5, "radial_weight": 0.3, "beta": 0.1}

    # Test 1: TropicalHyperbolicVAE (full)
    print("\n[1/4] Testing TropicalHyperbolicVAE (full)...")
    model = TropicalHyperbolicVAE(input_dim=9, latent_dim=16, hidden_dims=[64, 32])
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    start = time.time()
    metrics = train_and_evaluate(model, ops, indices, config)
    elapsed = time.time() - start
    results.append(
        {
            "name": "TropicalHyperbolicVAE (full)",
            **metrics,
            "time": elapsed,
        }
    )
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Spearman (euc): {metrics['spearman_euc']:+.4f}")
    print(f"  Spearman (hyp): {metrics['spearman_hyp']:+.4f}")
    print(f"  Spearman (tropical): {metrics['spearman_tropical']:+.4f}")
    print(f"  Time: {elapsed:.1f}s")

    # Test 2: TropicalHyperbolicVAELight
    print("\n[2/4] Testing TropicalHyperbolicVAELight...")
    model = TropicalHyperbolicVAELight(input_dim=9, latent_dim=16, hidden_dims=[64, 32])
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    start = time.time()
    metrics = train_and_evaluate(model, ops, indices, config)
    elapsed = time.time() - start
    results.append(
        {
            "name": "TropicalHyperbolicVAELight",
            **metrics,
            "time": elapsed,
        }
    )
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Spearman (euc): {metrics['spearman_euc']:+.4f}")
    print(f"  Spearman (hyp): {metrics['spearman_hyp']:+.4f}")
    print(f"  Spearman (tropical): {metrics['spearman_tropical']:+.4f}")
    print(f"  Time: {elapsed:.1f}s")

    # Test 3: Comparison - SimpleVAEWithHyperbolic
    print("\n[3/4] Testing SimpleVAEWithHyperbolic (baseline)...")
    from src.models.simple_vae import SimpleVAEWithHyperbolic

    model = SimpleVAEWithHyperbolic(input_dim=9, latent_dim=16, hidden_dims=[64, 32])
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    start = time.time()
    metrics = train_and_evaluate(model, ops, indices, config)
    elapsed = time.time() - start
    results.append(
        {
            "name": "SimpleVAEWithHyperbolic (baseline)",
            **metrics,
            "time": elapsed,
        }
    )
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Spearman (euc): {metrics['spearman_euc']:+.4f}")
    print(f"  Spearman (hyp): {metrics['spearman_hyp']:+.4f}")
    print(f"  Time: {elapsed:.1f}s")

    # Test 4: TropicalHyperbolicVAE with different temperature
    print("\n[4/4] Testing TropicalHyperbolicVAE (temp=0.05)...")
    model = TropicalHyperbolicVAE(input_dim=9, latent_dim=16, hidden_dims=[64, 32], temperature=0.05)
    start = time.time()
    metrics = train_and_evaluate(model, ops, indices, config)
    elapsed = time.time() - start
    results.append(
        {
            "name": "TropicalHyperbolicVAE (temp=0.05)",
            **metrics,
            "time": elapsed,
        }
    )
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Spearman (euc): {metrics['spearman_euc']:+.4f}")
    print(f"  Spearman (hyp): {metrics['spearman_hyp']:+.4f}")
    print(f"  Spearman (tropical): {metrics['spearman_tropical']:+.4f}")
    print(f"  Time: {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Model':<40} {'Accuracy':>10} {'Spearman(euc)':>14} {'Spearman(hyp)':>14} {'Time':>8}")
    print("-" * 100)

    for r in results:
        print(
            f"{r['name']:<40} {r['accuracy']:>9.1%} {r['spearman_euc']:>+14.4f} {r['spearman_hyp']:>+14.4f} {r['time']:>7.1f}s"
        )

    # Best model
    best = max(results, key=lambda r: 0.4 * r["accuracy"] + 0.4 * r["spearman_euc"] + 0.2 * r.get("silhouette", 0))
    print(f"\nBest model: {best['name']}")
    print(f"  Accuracy: {best['accuracy']:.1%}, Spearman: {best['spearman_euc']:+.4f}")

    # Save results
    output_path = PROJECT_ROOT / "outputs" / "hybrid_vae_results.json"
    with open(output_path, "w") as f:
        # Remove non-serializable history
        save_results = [{k: v for k, v in r.items() if k != "history"} for r in results]
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
