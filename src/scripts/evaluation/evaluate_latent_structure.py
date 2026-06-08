#!/usr/bin/env python
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Evaluate Latent Space Structure.

This script evaluates how well the VAE latent space preserves:
1. 3-adic distance ordering (p-adic structure)
2. Hierarchical clustering structure
3. Operation similarity

Compares baseline VAE vs hyperbolic + p-adic enhanced VAE.

Usage:
    python src/scripts/evaluation/evaluate_latent_structure.py
    python src/scripts/evaluation/evaluate_latent_structure.py --checkpoint outputs/optimal/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generation import generate_all_ternary_operations
from src.models.optimal_vae import OptimalVAE, OptimalVAEConfig
from src.models.simple_vae import SimpleVAE


def compute_padic_distance(idx1: int, idx2: int, p: int = 3) -> float:
    """Compute p-adic distance between two operation indices.

    The p-adic distance is based on the largest power of p dividing
    the difference.
    """
    if idx1 == idx2:
        return 0.0

    diff = abs(idx1 - idx2)
    if diff == 0:
        return 0.0

    # Count powers of p
    k = 0
    while diff % p == 0:
        diff //= p
        k += 1

    # p-adic distance = p^(-k)
    return float(p ** (-k))


def compute_padic_distance_matrix(n: int, p: int = 3) -> np.ndarray:
    """Compute pairwise p-adic distance matrix."""
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = compute_padic_distance(i, j, p)
            distances[i, j] = d
            distances[j, i] = d
    return distances


def evaluate_distance_preservation(
    embeddings: np.ndarray,
    padic_distances: np.ndarray,
    n_samples: int = 5000,
) -> dict:
    """Evaluate how well embeddings preserve p-adic distance ordering.

    Uses Spearman and Kendall-Tau rank correlation between:
    - p-adic distances in index space
    - Euclidean distances in latent space
    """
    n = len(embeddings)

    # Sample pairs for efficiency
    np.random.seed(42)
    i_idx = np.random.randint(0, n, n_samples)
    j_idx = np.random.randint(0, n, n_samples)

    # Remove self-pairs
    mask = i_idx != j_idx
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]

    # Compute distances
    padic_dists = padic_distances[i_idx, j_idx]
    latent_dists = np.linalg.norm(embeddings[i_idx] - embeddings[j_idx], axis=1)

    # Rank correlations
    spearman_corr, spearman_p = spearmanr(padic_dists, latent_dists)
    kendall_corr, kendall_p = kendalltau(padic_dists, latent_dists)

    return {
        "spearman_correlation": spearman_corr,
        "spearman_pvalue": spearman_p,
        "kendall_correlation": kendall_corr,
        "kendall_pvalue": kendall_p,
        "n_pairs": len(i_idx),
    }


def evaluate_clustering(embeddings: np.ndarray, n_clusters: int = 27) -> dict:
    """Evaluate clustering quality of latent space.

    Uses 27 clusters (3^3) to match the hierarchical structure of
    ternary operations.
    """
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)

    # Silhouette score (higher is better)
    silhouette = silhouette_score(embeddings, labels)

    # Compute cluster compactness
    inertia = kmeans.inertia_ / len(embeddings)

    return {
        "silhouette_score": silhouette,
        "inertia": inertia,
        "n_clusters": n_clusters,
    }


def evaluate_hierarchical_structure(
    embeddings: np.ndarray,
    indices: np.ndarray,
) -> dict:
    """Evaluate hierarchical structure preservation.

    Operations that share the same first 3 positions (first trit)
    should cluster together.
    """
    n = len(embeddings)

    # Group by first trit (index // 3^6 = 729)
    first_trit = indices // 729  # 0-26 (27 groups)

    # Compute within-group and between-group distances
    within_dists = []
    between_dists = []

    np.random.seed(42)
    for _ in range(5000):
        i, j = np.random.randint(0, n, 2)
        if i == j:
            continue

        dist = np.linalg.norm(embeddings[i] - embeddings[j])

        if first_trit[i] == first_trit[j]:
            within_dists.append(dist)
        else:
            between_dists.append(dist)

    within_mean = np.mean(within_dists) if within_dists else 0
    between_mean = np.mean(between_dists) if between_dists else 0

    # Separation ratio (higher is better - means hierarchical structure)
    separation_ratio = between_mean / (within_mean + 1e-8)

    return {
        "within_group_distance": within_mean,
        "between_group_distance": between_mean,
        "separation_ratio": separation_ratio,
    }


def load_model(checkpoint_path: Path = None, use_hyperbolic: bool = False):
    """Load trained model or create new one."""
    if checkpoint_path and checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Determine model type from config
        config = checkpoint.get("config", {})
        if isinstance(config, dict):
            enable_hyp = config.get("enable_hyperbolic", False)
        else:
            enable_hyp = getattr(config, "enable_hyperbolic", False)

        if enable_hyp:
            model = OptimalVAE(
                OptimalVAEConfig(
                    enable_hyperbolic=True,
                    enable_padic_ranking=config.get("enable_padic_ranking", True)
                    if isinstance(config, dict)
                    else getattr(config, "enable_padic_ranking", True),
                )
            )
        else:
            model = SimpleVAE()

        model.load_state_dict(checkpoint["model_state_dict"])
        return model, "optimal" if enable_hyp else "baseline"

    # Create new model
    if use_hyperbolic:
        return OptimalVAE(OptimalVAEConfig()), "optimal"
    else:
        return SimpleVAE(), "baseline"


def get_embeddings(model, operations: torch.Tensor) -> np.ndarray:
    """Get latent embeddings from model."""
    model.eval()
    with torch.no_grad():
        outputs = model(operations)
        z = outputs.get("z_euc", outputs.get("z", outputs["mu"]))
        return z.numpy()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Latent Space Structure")
    parser.add_argument("--checkpoint", type=Path, help="Model checkpoint to load")
    parser.add_argument("--compare", action="store_true", help="Compare baseline vs optimal")
    parser.add_argument("--train-first", action="store_true", help="Train models before evaluating")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs if --train-first")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("LATENT SPACE STRUCTURE EVALUATION")
    print("=" * 70)

    # Load data
    all_operations = generate_all_ternary_operations()
    operations = torch.tensor(all_operations, dtype=torch.float32)
    indices = np.arange(len(operations))
    print(f"\nOperations: {len(operations)}")

    # Compute p-adic distance matrix (subsample for efficiency)
    print("Computing p-adic distance matrix...")
    n_subsample = min(2000, len(operations))
    subsample_idx = np.random.choice(len(operations), n_subsample, replace=False)
    subsample_idx.sort()
    padic_distances = compute_padic_distance_matrix(n_subsample)

    results = {}

    if args.compare or args.checkpoint is None:
        # Compare baseline vs optimal
        models_to_test = [
            ("Baseline (SimpleVAE)", SimpleVAE()),
            ("Optimal (Hyperbolic + P-adic)", OptimalVAE(OptimalVAEConfig())),
        ]

        if args.train_first:
            print("\n--- Training models first ---")
            from src.losses.dual_vae_loss import KLDivergenceLoss, ReconstructionLoss
            from src.training import TernaryDataset

            dataset = TernaryDataset(operations, torch.arange(len(operations)))
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

            recon_loss_fn = ReconstructionLoss()
            kl_loss_fn = KLDivergenceLoss()

            trained_models = []
            for name, model in models_to_test:
                print(f"\nTraining {name}...")
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                for epoch in range(args.epochs):
                    model.train()
                    for batch in train_loader:
                        x = batch["operation"]
                        optimizer.zero_grad()
                        outputs = model(x)
                        recon = recon_loss_fn(outputs["logits"], x)
                        kl = kl_loss_fn(outputs["mu"], outputs["logvar"])
                        loss = recon + 0.01 * kl  # Use beta=0.01
                        loss.backward()
                        optimizer.step()

                    if epoch % 20 == 0:
                        print(f"  Epoch {epoch}: loss={loss.item():.4f}")

                trained_models.append((name, model))

            models_to_test = trained_models

        for name, model in models_to_test:
            print(f"\n--- {name} ---")

            # Get embeddings
            embeddings = get_embeddings(model, operations)
            embeddings_sub = embeddings[subsample_idx]

            # Evaluate distance preservation
            dist_metrics = evaluate_distance_preservation(embeddings_sub, padic_distances)
            print("Distance Preservation:")
            print(f"  Spearman correlation: {dist_metrics['spearman_correlation']:.4f}")
            print(f"  Kendall correlation:  {dist_metrics['kendall_correlation']:.4f}")

            # Evaluate clustering
            cluster_metrics = evaluate_clustering(embeddings)
            print("Clustering Quality:")
            print(f"  Silhouette score: {cluster_metrics['silhouette_score']:.4f}")
            print(f"  Inertia: {cluster_metrics['inertia']:.4f}")

            # Evaluate hierarchical structure
            hier_metrics = evaluate_hierarchical_structure(embeddings, indices)
            print("Hierarchical Structure:")
            print(f"  Within-group dist:  {hier_metrics['within_group_distance']:.4f}")
            print(f"  Between-group dist: {hier_metrics['between_group_distance']:.4f}")
            print(f"  Separation ratio:   {hier_metrics['separation_ratio']:.4f}")

            results[name] = {
                **dist_metrics,
                **cluster_metrics,
                **hier_metrics,
            }

    elif args.checkpoint:
        # Evaluate single checkpoint
        model, model_type = load_model(args.checkpoint)
        name = f"Loaded ({model_type})"

        embeddings = get_embeddings(model, operations)
        embeddings_sub = embeddings[subsample_idx]

        dist_metrics = evaluate_distance_preservation(embeddings_sub, padic_distances)
        cluster_metrics = evaluate_clustering(embeddings)
        hier_metrics = evaluate_hierarchical_structure(embeddings, indices)

        print(f"\n--- {name} ---")
        print("Distance Preservation:")
        print(f"  Spearman correlation: {dist_metrics['spearman_correlation']:.4f}")
        print(f"  Kendall correlation:  {dist_metrics['kendall_correlation']:.4f}")
        print("Clustering Quality:")
        print(f"  Silhouette score: {cluster_metrics['silhouette_score']:.4f}")
        print("Hierarchical Structure:")
        print(f"  Separation ratio: {hier_metrics['separation_ratio']:.4f}")

        results[name] = {**dist_metrics, **cluster_metrics, **hier_metrics}

    # Summary comparison
    if len(results) > 1:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)

        metrics_to_compare = [
            ("spearman_correlation", "Spearman (p-adic)", True),
            ("kendall_correlation", "Kendall (p-adic)", True),
            ("silhouette_score", "Silhouette", True),
            ("separation_ratio", "Separation", True),
        ]

        print(f"\n{'Metric':<25}", end="")
        for name in results:
            print(f"{name[:20]:<22}", end="")
        print()
        print("-" * 70)

        for metric_key, metric_name, higher_better in metrics_to_compare:
            print(f"{metric_name:<25}", end="")
            values = []
            for name in results:
                val = results[name].get(metric_key, 0)
                values.append(val)
                print(f"{val:<22.4f}", end="")
            print()

            # Highlight winner
            if len(values) == 2:
                winner = (0 if values[0] > values[1] else 1) if higher_better else 0 if values[0] < values[1] else 1
                improvement = (values[1] - values[0]) / (abs(values[0]) + 1e-8) * 100
                print(f"  {'Optimal better' if winner == 1 else 'Baseline better'}: {abs(improvement):.1f}%")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
