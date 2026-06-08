#!/usr/bin/env python
"""Test Curriculum Training vs Standard Training.

Compares phased loss introduction (curriculum) vs full loss from start.

Run with:
    python src/scripts/experiments/test_curriculum_training.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


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


def evaluate_embeddings(z, indices):
    """Evaluate latent space structure."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    n = len(z)
    z_np = z.detach().cpu().numpy()
    indices_np = indices.cpu().numpy()

    n_pairs = min(3000, n * (n - 1) // 2)
    np.random.seed(42)
    i_idx = np.random.randint(0, n, n_pairs)
    j_idx = np.random.randint(0, n, n_pairs)
    mask = i_idx != j_idx
    i_idx, j_idx = i_idx[mask], j_idx[mask]

    padic_dists = [compute_padic_distance(indices_np[i], indices_np[j]) for i, j in zip(i_idx, j_idx, strict=False)]
    latent_dists = np.linalg.norm(z_np[i_idx] - z_np[j_idx], axis=1)

    corr, _ = spearmanr(padic_dists, latent_dists)

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

    return {"spearman": corr if not np.isnan(corr) else 0.0, "silhouette": silhouette}


def train_standard(model, ops, indices, epochs=100, padic_weight=0.5, radial_weight=0.3):
    """Standard training with all losses from start."""
    from src.losses.dual_vae_loss import KLDivergenceLoss, ReconstructionLoss
    from src.losses.padic import PAdicRankingLoss
    from src.losses.padic_geodesic import MonotonicRadialLoss

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    recon_fn = ReconstructionLoss()
    kl_fn = KLDivergenceLoss()
    padic_fn = PAdicRankingLoss(margin=0.1, n_triplets=500)
    radial_fn = MonotonicRadialLoss()

    history = {"epoch": [], "loss": [], "accuracy": [], "spearman": []}

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
        beta = 0.1 * (0.5 + 0.5 * np.sin(2 * np.pi * epoch / 50))

        loss = recon + beta * kl

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

                history["epoch"].append(epoch)
                history["loss"].append(loss.item())
                history["accuracy"].append(acc)
                history["spearman"].append(metrics["spearman"])

    # Final eval
    model.eval()
    with torch.no_grad():
        outputs = model(ops)
        z = outputs.get("z_euc", outputs["z"])
        pred = outputs["logits"].argmax(dim=-1)
        target = (ops + 1).long()
        accuracy = (pred == target).float().mean().item()
        metrics = evaluate_embeddings(z, indices)

    return {
        "accuracy": accuracy,
        "spearman": metrics["spearman"],
        "silhouette": metrics["silhouette"],
        "history": history,
    }


def main():
    from src.data.generation import generate_all_ternary_operations
    from src.models.simple_vae import SimpleVAEWithHyperbolic
    from src.models.tropical_hyperbolic_vae import TropicalHyperbolicVAE
    from src.training.curriculum_trainer import CurriculumConfig, CurriculumTrainer

    print("\n" + "=" * 80)
    print("CURRICULUM TRAINING TEST")
    print("=" * 80)
    print("Comparing: Phased loss introduction vs All losses from start")
    print("=" * 80 + "\n")

    # Load data
    ops = torch.tensor(generate_all_ternary_operations()[:1000], dtype=torch.float32)
    indices = torch.arange(len(ops))

    results = []

    # Test 1: Standard training - SimpleVAEWithHyperbolic
    print("[1/4] Standard training - SimpleVAEWithHyperbolic...")
    model = SimpleVAEWithHyperbolic(input_dim=9, latent_dim=16)
    start = time.time()
    result = train_standard(model, ops, indices, epochs=100)
    elapsed = time.time() - start
    results.append(
        {
            "name": "Standard - SimpleHyp",
            "accuracy": result["accuracy"],
            "spearman": result["spearman"],
            "silhouette": result["silhouette"],
            "time": elapsed,
        }
    )
    print(f"  Accuracy: {result['accuracy']:.1%}, Spearman: {result['spearman']:+.4f}, Time: {elapsed:.1f}s")

    # Test 2: Curriculum training - SimpleVAEWithHyperbolic
    print("\n[2/4] Curriculum training - SimpleVAEWithHyperbolic...")
    model = SimpleVAEWithHyperbolic(input_dim=9, latent_dim=16)
    config = CurriculumConfig(
        phase1_epochs=20,
        phase2_epochs=20,
        phase3_epochs=20,
        phase4_epochs=40,
        padic_weight=0.5,
        radial_weight=0.3,
    )
    trainer = CurriculumTrainer(model, config)
    start = time.time()
    result = trainer.train(ops, indices, eval_fn=evaluate_embeddings, verbose=False)
    elapsed = time.time() - start
    results.append(
        {
            "name": "Curriculum - SimpleHyp",
            "accuracy": result["accuracy"],
            "spearman": result["spearman"],
            "silhouette": result["silhouette"],
            "time": elapsed,
        }
    )
    print(f"  Accuracy: {result['accuracy']:.1%}, Spearman: {result['spearman']:+.4f}, Time: {elapsed:.1f}s")

    # Test 3: Standard training - TropicalHyperbolicVAE
    print("\n[3/4] Standard training - TropicalHyperbolicVAE...")
    model = TropicalHyperbolicVAE(input_dim=9, latent_dim=16, temperature=0.05)
    start = time.time()
    result = train_standard(model, ops, indices, epochs=100)
    elapsed = time.time() - start
    results.append(
        {
            "name": "Standard - TropicalHyp",
            "accuracy": result["accuracy"],
            "spearman": result["spearman"],
            "silhouette": result["silhouette"],
            "time": elapsed,
        }
    )
    print(f"  Accuracy: {result['accuracy']:.1%}, Spearman: {result['spearman']:+.4f}, Time: {elapsed:.1f}s")

    # Test 4: Curriculum training - TropicalHyperbolicVAE
    print("\n[4/4] Curriculum training - TropicalHyperbolicVAE...")
    model = TropicalHyperbolicVAE(input_dim=9, latent_dim=16, temperature=0.05)
    trainer = CurriculumTrainer(model, config)
    start = time.time()
    result = trainer.train(ops, indices, eval_fn=evaluate_embeddings, verbose=False)
    elapsed = time.time() - start
    results.append(
        {
            "name": "Curriculum - TropicalHyp",
            "accuracy": result["accuracy"],
            "spearman": result["spearman"],
            "silhouette": result["silhouette"],
            "time": elapsed,
        }
    )
    print(f"  Accuracy: {result['accuracy']:.1%}, Spearman: {result['spearman']:+.4f}, Time: {elapsed:.1f}s")

    # Summary
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"{'Configuration':<30} {'Accuracy':>10} {'Spearman':>10} {'Silhouette':>10} {'Time':>8}")
    print("-" * 100)

    for r in results:
        print(
            f"{r['name']:<30} {r['accuracy']:>9.1%} {r['spearman']:>+10.4f} {r['silhouette']:>10.4f} {r['time']:>7.1f}s"
        )

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON: Curriculum vs Standard")
    print("=" * 80)

    simpleHyp_std = next(r for r in results if "Standard - SimpleHyp" in r["name"])
    simpleHyp_curr = next(r for r in results if "Curriculum - SimpleHyp" in r["name"])
    tropicalHyp_std = next(r for r in results if "Standard - TropicalHyp" in r["name"])
    tropicalHyp_curr = next(r for r in results if "Curriculum - TropicalHyp" in r["name"])

    print("\nSimpleVAEWithHyperbolic:")
    print(f"  Standard:   acc={simpleHyp_std['accuracy']:.1%}, corr={simpleHyp_std['spearman']:+.4f}")
    print(f"  Curriculum: acc={simpleHyp_curr['accuracy']:.1%}, corr={simpleHyp_curr['spearman']:+.4f}")
    acc_diff = simpleHyp_curr["accuracy"] - simpleHyp_std["accuracy"]
    corr_diff = simpleHyp_curr["spearman"] - simpleHyp_std["spearman"]
    print(f"  Difference: acc={acc_diff:+.1%}, corr={corr_diff:+.4f}")

    print("\nTropicalHyperbolicVAE:")
    print(f"  Standard:   acc={tropicalHyp_std['accuracy']:.1%}, corr={tropicalHyp_std['spearman']:+.4f}")
    print(f"  Curriculum: acc={tropicalHyp_curr['accuracy']:.1%}, corr={tropicalHyp_curr['spearman']:+.4f}")
    acc_diff = tropicalHyp_curr["accuracy"] - tropicalHyp_std["accuracy"]
    corr_diff = tropicalHyp_curr["spearman"] - tropicalHyp_std["spearman"]
    print(f"  Difference: acc={acc_diff:+.1%}, corr={corr_diff:+.4f}")

    # Best overall
    best = max(results, key=lambda r: 0.4 * r["accuracy"] + 0.4 * r["spearman"] + 0.2 * r["silhouette"])
    print(f"\nBest overall: {best['name']}")
    print(f"  Accuracy: {best['accuracy']:.1%}, Spearman: {best['spearman']:+.4f}")

    # Save
    output_path = PROJECT_ROOT / "outputs" / "curriculum_training_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
