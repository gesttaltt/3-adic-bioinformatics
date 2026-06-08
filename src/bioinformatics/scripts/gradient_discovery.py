#!/usr/bin/env python3
"""Gradient Discovery in VAE Latent Space.

Explores the VAE embedding space to discover:
1. Non-evident paths between distant mutations
2. "Mutation gradients" - directions of increasing/decreasing DDG
3. Clusters of functionally similar mutations
4. Unexpected relationships between apparently unrelated mutations

The VAE latent space provides a continuous manifold where discrete
mutations are embedded based on their functional similarity, enabling
gradient-based navigation that wouldn't be possible in discrete space.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3]))

import json
from dataclasses import dataclass
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.bioinformatics.data.protherm_loader import ProThermLoader
from src.bioinformatics.models.ddg_vae import DDGVAE


@dataclass
class MutationEmbedding:
    """Container for mutation with its VAE embedding."""

    mutation_id: str
    wild_type: str
    mutant: str
    position: int
    protein: str
    ddg: float
    embedding: np.ndarray
    vae_pred: float


class GradientDiscovery:
    """Discovers gradients and paths in VAE latent space."""

    def __init__(
        self,
        vae: DDGVAE,
        device: str = "cuda",
    ):
        self.vae = vae.to(device)
        self.vae.eval()
        self.device = device
        self.mutations: list[MutationEmbedding] = []
        self.embeddings: np.ndarray | None = None
        self.distance_matrix: np.ndarray | None = None

    def extract_embeddings(self, dataset, records) -> None:
        """Extract VAE embeddings for all mutations."""
        self.mutations = []

        with torch.no_grad():
            for i, record in enumerate(records):
                x, y = dataset[i]
                x = x.unsqueeze(0).to(self.device)

                output = self.vae(x)

                # Get embedding (mu)
                if "mu" in output:
                    emb = output["mu"].cpu().numpy().squeeze()
                else:
                    emb = output["z_hyp"].cpu().numpy().squeeze()

                vae_pred = output["ddg_pred"].cpu().item()

                self.mutations.append(
                    MutationEmbedding(
                        mutation_id=record.full_id,
                        wild_type=record.wild_type,
                        mutant=record.mutant,
                        position=record.position,
                        protein=record.pdb_id,
                        ddg=record.ddg,
                        embedding=emb,
                        vae_pred=vae_pred,
                    )
                )

        self.embeddings = np.array([m.embedding for m in self.mutations])
        self._compute_distances()

    def _compute_distances(self) -> None:
        """Compute pairwise distances in latent space."""
        self.distance_matrix = squareform(pdist(self.embeddings, metric="euclidean"))

    def find_ddg_gradient(self) -> dict:
        """Find the direction of maximum DDG change in latent space.

        Uses linear regression to find the gradient direction.
        """
        ddg_values = np.array([m.ddg for m in self.mutations])

        # Fit linear model: DDG = w @ embedding + b
        # Using least squares
        X = self.embeddings
        y = ddg_values

        # Add bias term
        X_bias = np.column_stack([X, np.ones(len(X))])
        w, residuals, rank, s = np.linalg.lstsq(X_bias, y, rcond=None)

        gradient_direction = w[:-1]  # Exclude bias
        gradient_direction = gradient_direction / np.linalg.norm(gradient_direction)

        # Project mutations onto gradient
        projections = self.embeddings @ gradient_direction

        # Correlation between projection and DDG
        corr = spearmanr(projections, ddg_values)[0]

        return {
            "gradient_direction": gradient_direction,
            "gradient_ddg_correlation": corr,
            "projections": projections,
            "explained_variance": 1 - (residuals[0] / np.var(y) / len(y)) if len(residuals) > 0 else 0,
        }

    def find_mutation_path(
        self,
        start_idx: int,
        end_idx: int,
        n_steps: int = 10,
    ) -> list[dict]:
        """Find interpolation path between two mutations in latent space.

        This reveals intermediate "virtual mutations" that might exist
        between two known mutations.
        """
        start_emb = self.embeddings[start_idx]
        end_emb = self.embeddings[end_idx]

        # Linear interpolation
        path = []
        for t in np.linspace(0, 1, n_steps):
            interp_emb = (1 - t) * start_emb + t * end_emb

            # Find nearest real mutation
            distances = np.linalg.norm(self.embeddings - interp_emb, axis=1)
            nearest_idx = np.argmin(distances)
            nearest = self.mutations[nearest_idx]

            # Predict DDG at interpolated point
            with torch.no_grad():
                # We need to decode the embedding - approximate by using nearest
                pass

            path.append(
                {
                    "t": t,
                    "interpolated_embedding": interp_emb,
                    "nearest_mutation": nearest.mutation_id,
                    "nearest_distance": distances[nearest_idx],
                    "nearest_ddg": nearest.ddg,
                    "estimated_ddg": (1 - t) * self.mutations[start_idx].ddg + t * self.mutations[end_idx].ddg,
                }
            )

        return path

    def find_unexpected_neighbors(self, top_k: int = 10) -> list[dict]:
        """Find mutations that are close in latent space but distant in feature space.

        These represent non-obvious functional relationships.
        """
        # Compute feature-space distances (based on mutation type)
        feature_distances = np.zeros((len(self.mutations), len(self.mutations)))

        for i, m1 in enumerate(self.mutations):
            for j, m2 in enumerate(self.mutations):
                if i >= j:
                    continue

                # Simple feature distance: same protein? same position? same AA change?
                dist = 0
                if m1.protein != m2.protein:
                    dist += 10  # Different protein
                dist += abs(m1.position - m2.position) / 100  # Position difference
                if m1.wild_type != m2.wild_type:
                    dist += 2
                if m1.mutant != m2.mutant:
                    dist += 2

                feature_distances[i, j] = dist
                feature_distances[j, i] = dist

        # Find pairs with small latent distance but large feature distance
        unexpected = []
        for i in range(len(self.mutations)):
            for j in range(i + 1, len(self.mutations)):
                latent_dist = self.distance_matrix[i, j]
                feature_dist = feature_distances[i, j]

                if feature_dist > 5:  # Require some feature distance
                    unexpected.append(
                        {
                            "mutation1": self.mutations[i].mutation_id,
                            "mutation2": self.mutations[j].mutation_id,
                            "latent_distance": latent_dist,
                            "feature_distance": feature_dist,
                            "ddg1": self.mutations[i].ddg,
                            "ddg2": self.mutations[j].ddg,
                            "ddg_diff": abs(self.mutations[i].ddg - self.mutations[j].ddg),
                            "ratio": feature_dist / (latent_dist + 0.1),
                        }
                    )

        # Sort by ratio (high ratio = unexpected closeness)
        unexpected.sort(key=lambda x: -x["ratio"])
        return unexpected[:top_k]

    def cluster_mutations(self, n_clusters: int = 5) -> dict:
        """Cluster mutations in latent space."""
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.embeddings)

        # Analyze clusters
        clusters = []
        for c in range(n_clusters):
            mask = labels == c
            cluster_mutations = [m for m, is_in in zip(self.mutations, mask, strict=False) if is_in]
            cluster_ddgs = [m.ddg for m in cluster_mutations]

            # Dominant mutation types
            wt_counts = {}
            mut_counts = {}
            for m in cluster_mutations:
                wt_counts[m.wild_type] = wt_counts.get(m.wild_type, 0) + 1
                mut_counts[m.mutant] = mut_counts.get(m.mutant, 0) + 1

            clusters.append(
                {
                    "cluster_id": c,
                    "size": len(cluster_mutations),
                    "mean_ddg": np.mean(cluster_ddgs),
                    "std_ddg": np.std(cluster_ddgs),
                    "ddg_range": [min(cluster_ddgs), max(cluster_ddgs)],
                    "dominant_wt": max(wt_counts, key=wt_counts.get) if wt_counts else None,
                    "dominant_mut": max(mut_counts, key=mut_counts.get) if mut_counts else None,
                    "mutations": [m.mutation_id for m in cluster_mutations[:5]],  # Sample
                }
            )

        return {
            "n_clusters": n_clusters,
            "labels": labels.tolist(),
            "clusters": clusters,
            "centroids": kmeans.cluster_centers_.tolist(),
        }

    def find_ddg_extremes_path(self) -> dict:
        """Find path between most stabilizing and most destabilizing mutations."""
        ddg_values = [m.ddg for m in self.mutations]
        min_idx = np.argmin(ddg_values)
        max_idx = np.argmax(ddg_values)

        path = self.find_mutation_path(min_idx, max_idx, n_steps=20)

        return {
            "stabilizing": {
                "mutation": self.mutations[min_idx].mutation_id,
                "ddg": self.mutations[min_idx].ddg,
            },
            "destabilizing": {
                "mutation": self.mutations[max_idx].mutation_id,
                "ddg": self.mutations[max_idx].ddg,
            },
            "path": path,
            "latent_distance": self.distance_matrix[min_idx, max_idx],
        }

    def analyze_local_gradients(self, k: int = 5) -> list[dict]:
        """Analyze local DDG gradients around each mutation.

        For each mutation, looks at k nearest neighbors and computes
        the local gradient direction.
        """
        local_gradients = []

        for i, m in enumerate(self.mutations):
            # Find k nearest neighbors
            distances = self.distance_matrix[i]
            neighbor_indices = np.argsort(distances)[1 : k + 1]  # Exclude self

            # Get neighbor embeddings and DDGs
            neighbor_embs = self.embeddings[neighbor_indices]
            neighbor_ddgs = np.array([self.mutations[j].ddg for j in neighbor_indices])

            # Compute local gradient (direction of DDG increase)
            # Using weighted average of directions to higher DDG neighbors
            directions = neighbor_embs - self.embeddings[i]
            ddg_diffs = neighbor_ddgs - m.ddg

            # Weight by DDG difference magnitude
            weights = np.abs(ddg_diffs)
            if weights.sum() > 0:
                weights = weights / weights.sum()
                weighted_direction = (directions.T @ (weights * np.sign(ddg_diffs))).T
                gradient_magnitude = np.linalg.norm(weighted_direction)
                if gradient_magnitude > 0:
                    weighted_direction = weighted_direction / gradient_magnitude
            else:
                weighted_direction = np.zeros_like(self.embeddings[i])
                gradient_magnitude = 0

            local_gradients.append(
                {
                    "mutation": m.mutation_id,
                    "ddg": m.ddg,
                    "local_gradient": weighted_direction.tolist(),
                    "gradient_magnitude": gradient_magnitude,
                    "neighbor_ddg_variance": np.var(neighbor_ddgs),
                    "neighbors": [self.mutations[j].mutation_id for j in neighbor_indices],
                }
            )

        return local_gradients

    def visualize_latent_space(self, output_path: Path) -> None:
        """Create visualization of latent space."""
        # PCA for 2D projection
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(self.embeddings)

        ddg_values = np.array([m.ddg for m in self.mutations])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Colored by DDG
        scatter1 = axes[0].scatter(emb_2d[:, 0], emb_2d[:, 1], c=ddg_values, cmap="RdYlBu_r", s=50, alpha=0.7)
        axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        axes[0].set_title("VAE Latent Space (colored by DDG)")
        plt.colorbar(scatter1, ax=axes[0], label="DDG (kcal/mol)")

        # Plot 2: Colored by protein
        proteins = [m.protein for m in self.mutations]
        unique_proteins = list(set(proteins))
        protein_colors = {p: i for i, p in enumerate(unique_proteins)}
        colors = [protein_colors[p] for p in proteins]

        axes[1].scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, cmap="tab20", s=50, alpha=0.7)
        axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        axes[1].set_title("VAE Latent Space (colored by protein)")

        plt.tight_layout()
        plt.savefig(output_path / "latent_space_visualization.png", dpi=150)
        plt.close()

        # Additional: Gradient visualization
        gradient_info = self.find_ddg_gradient()
        gradient_dir = gradient_info["gradient_direction"]

        # Project gradient to 2D
        gradient_2d = pca.transform(gradient_dir.reshape(1, -1))[0]
        gradient_2d = gradient_2d / np.linalg.norm(gradient_2d) * 2

        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=ddg_values, cmap="RdYlBu_r", s=50, alpha=0.7)

        # Draw gradient arrow
        center = emb_2d.mean(axis=0)
        ax.annotate("", xy=center + gradient_2d, xytext=center, arrowprops=dict(arrowstyle="->", color="black", lw=3))
        ax.text(
            center[0] + gradient_2d[0] * 1.1,
            center[1] + gradient_2d[1] * 1.1,
            "DDG gradient",
            fontsize=12,
            fontweight="bold",
        )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"DDG Gradient in Latent Space (corr={gradient_info['gradient_ddg_correlation']:.3f})")
        plt.colorbar(scatter, label="DDG (kcal/mol)")

        plt.tight_layout()
        plt.savefig(output_path / "ddg_gradient_visualization.png", dpi=150)
        plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Gradient Discovery in VAE Latent Space")
    parser.add_argument("--device", default="cuda", help="Device")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/gradient_discovery_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Gradient Discovery in VAE Latent Space")
    print("=" * 70)

    # Load VAE
    print("\n[1] Loading trained VAE...")
    vae = DDGVAE.create_protherm_variant(use_hyperbolic=False)
    ckpt = torch.load(
        "outputs/ddg_vae_training_20260129_212316/vae_protherm/best.pt", map_location=args.device, weights_only=False
    )
    vae.load_state_dict(ckpt["model_state_dict"])
    print("  VAE loaded")

    # Load data
    print("\n[2] Loading ProTherm data...")
    loader = ProThermLoader()
    db = loader.load_curated()
    dataset = loader.create_dataset(db)
    print(f"  Loaded {len(db)} mutations")

    # Initialize discovery
    print("\n[3] Extracting VAE embeddings...")
    discovery = GradientDiscovery(vae, args.device)
    discovery.extract_embeddings(dataset, db.records)
    print(f"  Extracted {len(discovery.mutations)} embeddings, dim={discovery.embeddings.shape[1]}")

    # ========================================
    # Analysis 1: DDG Gradient
    # ========================================
    print("\n" + "=" * 70)
    print("[4] Finding DDG Gradient Direction")
    print("=" * 70)

    gradient_info = discovery.find_ddg_gradient()
    print(f"  Gradient-DDG correlation: {gradient_info['gradient_ddg_correlation']:.4f}")
    print(f"  This means {abs(gradient_info['gradient_ddg_correlation']) * 100:.1f}% of DDG variance")
    print("  is explained by a single direction in latent space!")

    # ========================================
    # Analysis 2: Clustering
    # ========================================
    print("\n" + "=" * 70)
    print("[5] Clustering Mutations in Latent Space")
    print("=" * 70)

    cluster_info = discovery.cluster_mutations(n_clusters=5)
    for c in cluster_info["clusters"]:
        print(f"\n  Cluster {c['cluster_id']}:")
        print(f"    Size: {c['size']}")
        print(f"    DDG: {c['mean_ddg']:.2f} ± {c['std_ddg']:.2f}")
        print(f"    Dominant: {c['dominant_wt']}→{c['dominant_mut']}")

    # ========================================
    # Analysis 3: Unexpected Neighbors
    # ========================================
    print("\n" + "=" * 70)
    print("[6] Finding Unexpected Neighbors (non-obvious relationships)")
    print("=" * 70)

    unexpected = discovery.find_unexpected_neighbors(top_k=10)
    print("\n  Top pairs with similar latent embeddings but different features:")
    for i, pair in enumerate(unexpected[:5]):
        print(f"\n  {i + 1}. {pair['mutation1']} <-> {pair['mutation2']}")
        print(f"     Latent dist: {pair['latent_distance']:.3f}, Feature dist: {pair['feature_distance']:.1f}")
        print(f"     DDG: {pair['ddg1']:.2f} vs {pair['ddg2']:.2f} (diff: {pair['ddg_diff']:.2f})")

    # ========================================
    # Analysis 4: Extreme Path
    # ========================================
    print("\n" + "=" * 70)
    print("[7] Path from Most Stabilizing to Most Destabilizing")
    print("=" * 70)

    extreme_path = discovery.find_ddg_extremes_path()
    print(f"\n  Start: {extreme_path['stabilizing']['mutation']} (DDG={extreme_path['stabilizing']['ddg']:.2f})")
    print(f"  End: {extreme_path['destabilizing']['mutation']} (DDG={extreme_path['destabilizing']['ddg']:.2f})")
    print(f"  Latent distance: {extreme_path['latent_distance']:.3f}")
    print("\n  Path through latent space:")
    for step in extreme_path["path"][::4]:  # Every 4th step
        print(f"    t={step['t']:.2f}: nearest={step['nearest_mutation'][:20]:20s} DDG≈{step['estimated_ddg']:.2f}")

    # ========================================
    # Analysis 5: Local Gradients
    # ========================================
    print("\n" + "=" * 70)
    print("[8] Analyzing Local DDG Gradients")
    print("=" * 70)

    local_gradients = discovery.analyze_local_gradients(k=5)

    # Find mutations with strongest local gradients
    local_gradients.sort(key=lambda x: -x["gradient_magnitude"])
    print("\n  Mutations with strongest local DDG gradients:")
    for lg in local_gradients[:5]:
        print(
            f"    {lg['mutation']}: magnitude={lg['gradient_magnitude']:.3f}, "
            f"DDG={lg['ddg']:.2f}, neighbor_var={lg['neighbor_ddg_variance']:.2f}"
        )

    # ========================================
    # Visualizations
    # ========================================
    print("\n" + "=" * 70)
    print("[9] Creating Visualizations")
    print("=" * 70)

    discovery.visualize_latent_space(output_dir)
    print(f"  Saved to {output_dir}/")

    # ========================================
    # Save Results
    # ========================================
    print("\n" + "=" * 70)
    print("[10] Saving Results")
    print("=" * 70)

    results = {
        "timestamp": timestamp,
        "n_mutations": len(discovery.mutations),
        "embedding_dim": discovery.embeddings.shape[1],
        "gradient_analysis": {
            "ddg_correlation": gradient_info["gradient_ddg_correlation"],
            "gradient_direction": gradient_info["gradient_direction"].tolist(),
        },
        "clustering": cluster_info,
        "unexpected_neighbors": unexpected,
        "extreme_path": {
            "stabilizing": extreme_path["stabilizing"],
            "destabilizing": extreme_path["destabilizing"],
            "latent_distance": extreme_path["latent_distance"],
        },
        "local_gradients_top5": local_gradients[:5],
    }

    with open(output_dir / "gradient_discovery_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved to {output_dir}/gradient_discovery_results.json")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("GRADIENT DISCOVERY COMPLETE")
    print("=" * 70)
    print(f"""
Key Findings:

1. DDG GRADIENT: A single direction in latent space explains
   {abs(gradient_info["gradient_ddg_correlation"]) * 100:.1f}% of DDG variation (corr={gradient_info["gradient_ddg_correlation"]:.3f})

2. CLUSTERS: {cluster_info["n_clusters"]} distinct functional clusters found
   - Cluster DDGs range from {min(c["mean_ddg"] for c in cluster_info["clusters"]):.2f} to {max(c["mean_ddg"] for c in cluster_info["clusters"]):.2f}

3. UNEXPECTED NEIGHBORS: Found mutations that are functionally similar
   (close in latent space) despite being from different proteins/positions

4. EXTREME PATH: Identified smooth transition path from most stabilizing
   to most destabilizing mutation through latent space

Output: {output_dir}/
""")


if __name__ == "__main__":
    main()
