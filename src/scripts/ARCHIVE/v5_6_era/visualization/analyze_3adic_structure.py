# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""3-Adic Algebraic Structure Analysis of Ternary VAE Manifold.

Analyzes whether the learned VAE embedding preserves the intrinsic
3-adic ultrametric topology and algebraic structure of ternary operations.

Key analyses:
1. 3-adic digit distance vs latent distance correlation
2. Composition geodesics: does f∘g lie on path between f and g?
3. PCA alignment with algebraic directions (symmetric, projection ops)
4. Cayley graph structure preservation

Usage:
    python src/scripts/visualization/analyze_3adic_structure.py --checkpoint latest.pt
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ternary_vae_v5_6 import DualNeuralVAEV5

from src.config.paths import CHECKPOINTS_DIR, VIZ_DIR
from src.data.generation import generate_all_ternary_operations


def load_model_and_encode(checkpoint_path: Path, device: str = "cpu"):
    """Load model and encode all operations."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    use_statenet = checkpoint.get("statenet_enabled", False)

    model = DualNeuralVAEV5(input_dim=9, latent_dim=16, use_statenet=use_statenet)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, dtype=torch.float32, device=device)

    with torch.no_grad():
        mu_A, _ = model.encoder_A(x)
        mu_B, _ = model.encoder_B(x)

    return {
        "operations": operations,
        "z_A": mu_A.cpu().numpy(),
        "z_B": mu_B.cpu().numpy(),
        "model": model,
        "device": device,
    }


def op_to_index(op: np.ndarray) -> int:
    """Convert operation vector to index (3-adic representation)."""
    idx = 0
    for i, val in enumerate(op):
        idx += (int(val) + 1) * (3**i)
    return idx


def index_to_op(idx: int) -> np.ndarray:
    """Convert index to operation vector."""
    op = np.zeros(9)
    for i in range(9):
        op[i] = (idx % 3) - 1
        idx //= 3
    return op


def compute_3adic_distance(idx1: int, idx2: int) -> int:
    """Compute 3-adic distance: number of differing 3-adic digits."""
    diff = 0
    for _ in range(9):
        if (idx1 % 3) != (idx2 % 3):
            diff += 1
        idx1 //= 3
        idx2 //= 3
    return diff


def compute_3adic_valuation(idx1: int, idx2: int) -> int:
    """Compute 3-adic valuation: position of first differing digit.

    In 3-adic topology, distance = 3^(-v) where v is the valuation.
    Higher valuation = closer in 3-adic metric.
    """
    diff = abs(idx1 - idx2)
    if diff == 0:
        return 9  # Same operation, infinite valuation
    v = 0
    while diff % 3 == 0:
        diff //= 3
        v += 1
    return v


def compose_operations(op1: np.ndarray, op2: np.ndarray) -> np.ndarray:
    """Compose two ternary operations: (op1 ∘ op2)(a,b) = op1(op2(a,b), op2(a,b)).

    Actually, for ternary logic ops, composition is:
    (f ∘ g)(a,b) = f(g(a,b), g(a,b)) -- but this loses info

    Better: think of ops as functions {-1,0,1}^2 -> {-1,0,1}
    Composition in the monoid sense would be pointwise or via lookup.

    For our analysis, we use: result[i] = op1[op2[i] + 1 + 3*(op2[i] + 1)]
    which represents f(g(input)) where input maps to position i.
    """
    # Simplified composition: apply op1 to outputs of op2
    # This treats the output as a new "input" in a specific way
    result = np.zeros(9)
    for i in range(9):
        # op2 output at position i
        out2 = int(op2[i]) + 1  # Map {-1,0,1} to {0,1,2}
        # Use this as index into op1 (diagonal application)
        result[i] = op1[out2 * 3 + out2]  # f(g(x), g(x)) pattern
    return result


def analyze_3adic_clustering(data: dict, output_path: Path):
    """Analyze if 3-adic neighbors cluster together in latent space."""
    operations = data["operations"]
    z_A = data["z_A"]
    z_B = data["z_B"]
    n_ops = len(operations)

    print("\n" + "=" * 60)
    print("3-ADIC DISTANCE vs LATENT DISTANCE ANALYSIS")
    print("=" * 60)

    # Sample pairs for analysis (full pairwise is 19683^2 / 2 ~ 193M pairs)
    n_samples = 50000
    np.random.seed(42)
    pairs = np.random.choice(n_ops, size=(n_samples, 2), replace=True)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]  # Remove self-pairs

    # Compute distances
    adic_dists = []
    adic_vals = []
    latent_dists_A = []
    latent_dists_B = []

    for i, j in pairs:
        adic_dists.append(compute_3adic_distance(i, j))
        adic_vals.append(compute_3adic_valuation(i, j))
        latent_dists_A.append(np.linalg.norm(z_A[i] - z_A[j]))
        latent_dists_B.append(np.linalg.norm(z_B[i] - z_B[j]))

    adic_dists = np.array(adic_dists)
    adic_vals = np.array(adic_vals)
    latent_dists_A = np.array(latent_dists_A)
    latent_dists_B = np.array(latent_dists_B)

    # Correlation analysis
    corr_A_dist, p_A_dist = spearmanr(adic_dists, latent_dists_A)
    corr_B_dist, p_B_dist = spearmanr(adic_dists, latent_dists_B)
    corr_A_val, p_A_val = spearmanr(adic_vals, latent_dists_A)
    corr_B_val, p_B_val = spearmanr(adic_vals, latent_dists_B)

    print(f"\nSpearman correlations (n={len(pairs)} pairs):")
    print(f"  VAE-A: 3-adic digit distance vs latent: r={corr_A_dist:.4f} (p={p_A_dist:.2e})")
    print(f"  VAE-B: 3-adic digit distance vs latent: r={corr_B_dist:.4f} (p={p_B_dist:.2e})")
    print(f"  VAE-A: 3-adic valuation vs latent:      r={corr_A_val:.4f} (p={p_A_val:.2e})")
    print(f"  VAE-B: 3-adic valuation vs latent:      r={corr_B_val:.4f} (p={p_B_val:.2e})")

    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # VAE-A: 3-adic digit distance vs latent distance
    ax = axes[0, 0]
    for d in range(1, 10):
        mask = adic_dists == d
        if mask.sum() > 0:
            ax.scatter(
                np.full(mask.sum(), d) + np.random.randn(mask.sum()) * 0.1,
                latent_dists_A[mask],
                alpha=0.1,
                s=1,
            )
    ax.set_xlabel("3-adic Digit Distance")
    ax.set_ylabel("Latent Distance (VAE-A)")
    ax.set_title(f"VAE-A: 3-adic vs Latent Distance\nr={corr_A_dist:.3f}")

    # VAE-B: 3-adic digit distance vs latent distance
    ax = axes[0, 1]
    for d in range(1, 10):
        mask = adic_dists == d
        if mask.sum() > 0:
            ax.scatter(
                np.full(mask.sum(), d) + np.random.randn(mask.sum()) * 0.1,
                latent_dists_B[mask],
                alpha=0.1,
                s=1,
            )
    ax.set_xlabel("3-adic Digit Distance")
    ax.set_ylabel("Latent Distance (VAE-B)")
    ax.set_title(f"VAE-B: 3-adic vs Latent Distance\nr={corr_B_dist:.3f}")

    # Mean latent distance by 3-adic distance
    ax = axes[0, 2]
    means_A = [latent_dists_A[adic_dists == d].mean() for d in range(1, 10)]
    means_B = [latent_dists_B[adic_dists == d].mean() for d in range(1, 10)]
    stds_A = [latent_dists_A[adic_dists == d].std() for d in range(1, 10)]
    stds_B = [latent_dists_B[adic_dists == d].std() for d in range(1, 10)]
    x = np.arange(1, 10)
    ax.errorbar(x - 0.1, means_A, yerr=stds_A, fmt="o-", label="VAE-A", capsize=3)
    ax.errorbar(x + 0.1, means_B, yerr=stds_B, fmt="s-", label="VAE-B", capsize=3)
    ax.set_xlabel("3-adic Digit Distance")
    ax.set_ylabel("Mean Latent Distance")
    ax.set_title("Mean Latent Distance by 3-adic Distance")
    ax.legend()

    # 3-adic valuation analysis
    ax = axes[1, 0]
    for v in range(10):
        mask = adic_vals == v
        if mask.sum() > 0:
            ax.scatter(
                np.full(mask.sum(), v) + np.random.randn(mask.sum()) * 0.1,
                latent_dists_A[mask],
                alpha=0.1,
                s=1,
                c="blue",
            )
    ax.set_xlabel("3-adic Valuation (higher = closer)")
    ax.set_ylabel("Latent Distance (VAE-A)")
    ax.set_title(f"VAE-A: Valuation vs Latent\nr={corr_A_val:.3f}")

    ax = axes[1, 1]
    for v in range(10):
        mask = adic_vals == v
        if mask.sum() > 0:
            ax.scatter(
                np.full(mask.sum(), v) + np.random.randn(mask.sum()) * 0.1,
                latent_dists_B[mask],
                alpha=0.1,
                s=1,
                c="orange",
            )
    ax.set_xlabel("3-adic Valuation (higher = closer)")
    ax.set_ylabel("Latent Distance (VAE-B)")
    ax.set_title(f"VAE-B: Valuation vs Latent\nr={corr_B_val:.3f}")

    # Histogram of distances
    ax = axes[1, 2]
    ax.hist(latent_dists_A, bins=50, alpha=0.5, label="VAE-A", density=True)
    ax.hist(latent_dists_B, bins=50, alpha=0.5, label="VAE-B", density=True)
    ax.set_xlabel("Latent Distance")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Latent Distances")
    ax.legend()

    plt.tight_layout()
    plt.savefig(
        output_path / "3adic_distance_analysis.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {output_path / '3adic_distance_analysis.png'}")

    return {
        "corr_A_dist": corr_A_dist,
        "corr_B_dist": corr_B_dist,
        "corr_A_val": corr_A_val,
        "corr_B_val": corr_B_val,
    }


def analyze_single_digit_neighbors(data: dict, output_path: Path):
    """Analyze operations that differ by exactly one 3-adic digit."""
    operations = data["operations"]
    z_A = data["z_A"]
    z_B = data["z_B"]
    n_ops = len(operations)

    print("\n" + "=" * 60)
    print("SINGLE-DIGIT NEIGHBOR ANALYSIS")
    print("=" * 60)

    # For each position, find pairs differing only at that position
    neighbor_dists_A = {pos: [] for pos in range(9)}
    neighbor_dists_B = {pos: [] for pos in range(9)}

    # Sample operations
    np.random.seed(42)
    sample_indices = np.random.choice(n_ops, size=min(5000, n_ops), replace=False)

    for idx in sample_indices:
        op = operations[idx].copy()
        for pos in range(9):
            original_val = op[pos]
            for new_val in [-1, 0, 1]:
                if new_val != original_val:
                    op[pos] = new_val
                    neighbor_idx = op_to_index(op)
                    if neighbor_idx < n_ops:
                        neighbor_dists_A[pos].append(np.linalg.norm(z_A[idx] - z_A[neighbor_idx]))
                        neighbor_dists_B[pos].append(np.linalg.norm(z_B[idx] - z_B[neighbor_idx]))
            op[pos] = original_val

    # Analyze by position
    print("\nMean latent distance for single-digit changes by position:")
    print("Position | VAE-A mean (std) | VAE-B mean (std)")
    print("-" * 50)

    means_A = []
    means_B = []
    for pos in range(9):
        if neighbor_dists_A[pos]:
            mean_A = np.mean(neighbor_dists_A[pos])
            std_A = np.std(neighbor_dists_A[pos])
            mean_B = np.mean(neighbor_dists_B[pos])
            std_B = np.std(neighbor_dists_B[pos])
            means_A.append(mean_A)
            means_B.append(mean_B)
            print(f"   {pos}     | {mean_A:.4f} ({std_A:.4f}) | {mean_B:.4f} ({std_B:.4f})")

    # In 3-adic topology, lower positions should have larger "jumps"
    # because changing digit 0 is a larger change than digit 8
    print("\nPosition-distance correlation (expect negative if 3-adic preserved):")
    corr_A, _ = spearmanr(range(9), means_A)
    corr_B, _ = spearmanr(range(9), means_B)
    print(f"  VAE-A: r={corr_A:.4f}")
    print(f"  VAE-B: r={corr_B:.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Bar chart of mean distances by position
    ax = axes[0]
    x = np.arange(9)
    width = 0.35
    ax.bar(x - width / 2, means_A, width, label="VAE-A", color="blue", alpha=0.7)
    ax.bar(x + width / 2, means_B, width, label="VAE-B", color="orange", alpha=0.7)
    ax.set_xlabel("Digit Position (0=LSB, 8=MSB)")
    ax.set_ylabel("Mean Latent Distance")
    ax.set_title("Single-Digit Change: Latent Distance by Position\n(3-adic: lower pos = larger jump)")
    ax.legend()
    ax.set_xticks(x)

    # 3D visualization of digit-0 neighbors
    ax = axes[1]
    pca = PCA(n_components=2)
    z_A_2d = pca.fit_transform(z_A)

    # Show a subset of digit-0 neighbor pairs
    np.random.seed(42)
    for _ in range(100):
        idx = np.random.randint(n_ops)
        op = operations[idx].copy()
        for new_val in [-1, 0, 1]:
            if new_val != op[0]:
                op[0] = new_val
                neighbor_idx = op_to_index(op)
                if neighbor_idx < n_ops:
                    ax.plot(
                        [z_A_2d[idx, 0], z_A_2d[neighbor_idx, 0]],
                        [z_A_2d[idx, 1], z_A_2d[neighbor_idx, 1]],
                        "b-",
                        alpha=0.1,
                        linewidth=0.5,
                    )
                op[0] = operations[idx][0]
    ax.scatter(z_A_2d[:, 0], z_A_2d[:, 1], c="gray", s=1, alpha=0.3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("VAE-A: Digit-0 Neighbor Connections\n(Blue lines = single LSB change)")

    # Same for digit-8 (MSB)
    ax = axes[2]
    for _ in range(100):
        idx = np.random.randint(n_ops)
        op = operations[idx].copy()
        for new_val in [-1, 0, 1]:
            if new_val != op[8]:
                op[8] = new_val
                neighbor_idx = op_to_index(op)
                if neighbor_idx < n_ops:
                    ax.plot(
                        [z_A_2d[idx, 0], z_A_2d[neighbor_idx, 0]],
                        [z_A_2d[idx, 1], z_A_2d[neighbor_idx, 1]],
                        "r-",
                        alpha=0.1,
                        linewidth=0.5,
                    )
                op[8] = operations[idx][8]
    ax.scatter(z_A_2d[:, 0], z_A_2d[:, 1], c="gray", s=1, alpha=0.3)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("VAE-A: Digit-8 Neighbor Connections\n(Red lines = single MSB change)")

    plt.tight_layout()
    plt.savefig(
        output_path / "3adic_single_digit_analysis.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {output_path / '3adic_single_digit_analysis.png'}")

    return {"position_corr_A": corr_A, "position_corr_B": corr_B}


def analyze_algebraic_structure(data: dict, output_path: Path):
    """Analyze special algebraic operations and their positions."""
    operations = data["operations"]
    z_A = data["z_A"]
    z_B = data["z_B"]
    n_ops = len(operations)

    print("\n" + "=" * 60)
    print("ALGEBRAIC STRUCTURE ANALYSIS")
    print("=" * 60)

    # Find special operations
    special_ops = {}

    # Identity-like operations
    for idx, op in enumerate(operations):
        # Projection to first argument: op(a,b) = a
        if np.array_equal(op, np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])):
            special_ops["proj_1"] = idx
        # Projection to second argument: op(a,b) = b
        if np.array_equal(op, np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])):
            special_ops["proj_2"] = idx
        # Constant -1
        if np.all(op == -1):
            special_ops["const_-1"] = idx
        # Constant 0
        if np.all(op == 0):
            special_ops["const_0"] = idx
        # Constant 1
        if np.all(op == 1):
            special_ops["const_1"] = idx
        # Min operation
        if np.array_equal(op, np.array([-1, -1, -1, -1, 0, 0, -1, 0, 1])):
            special_ops["min"] = idx
        # Max operation
        if np.array_equal(op, np.array([-1, -1, -1, -1, 0, 1, -1, 1, 1])):
            special_ops["max"] = idx
        # Addition mod 3 (shifted)
        if np.array_equal(op, np.array([-1, 0, 1, 0, 1, -1, 1, -1, 0])):
            special_ops["add_mod3"] = idx
        # Multiplication (ternary)
        if np.array_equal(op, np.array([1, 0, -1, 0, 0, 0, -1, 0, 1])):
            special_ops["mult"] = idx

    print(f"\nFound {len(special_ops)} special operations:")
    for name, idx in special_ops.items():
        print(f"  {name}: index {idx}, op = {operations[idx]}")

    # Compute distances between special ops in latent space
    if len(special_ops) >= 2:
        print("\nLatent distances between special operations:")
        names = list(special_ops.keys())
        print("\nVAE-A distances:")
        for i, name1 in enumerate(names):
            for name2 in names[i + 1 :]:
                idx1, idx2 = special_ops[name1], special_ops[name2]
                dist_A = np.linalg.norm(z_A[idx1] - z_A[idx2])
                np.linalg.norm(z_B[idx1] - z_B[idx2])
                adic = compute_3adic_distance(idx1, idx2)
                print(f"  {name1} <-> {name2}: latent={dist_A:.3f}, 3-adic={adic}")

    # PCA analysis: do algebraic structures align with principal components?
    pca_A = PCA(n_components=3)
    pca_B = PCA(n_components=3)
    z_A_3d = pca_A.fit_transform(z_A)
    z_B_3d = pca_B.fit_transform(z_B)

    # Analyze what each PC captures
    print("\n" + "=" * 60)
    print("PCA COMPONENT INTERPRETATION")
    print("=" * 60)

    # Correlate PCs with operation properties
    # Sum of outputs (bias)
    output_sums = operations.sum(axis=1)
    # Variance of outputs (spread)
    output_vars = operations.var(axis=1)
    # First element (LSB behavior)
    first_elem = operations[:, 0]
    # Last element (MSB behavior)
    last_elem = operations[:, 8]
    # Center element (identity behavior)
    center_elem = operations[:, 4]

    print("\nVAE-A PC correlations with operation properties:")
    for pc_idx in range(3):
        pc = z_A_3d[:, pc_idx]
        corr_sum, _ = spearmanr(pc, output_sums)
        corr_var, _ = spearmanr(pc, output_vars)
        corr_first, _ = spearmanr(pc, first_elem)
        corr_center, _ = spearmanr(pc, center_elem)
        corr_last, _ = spearmanr(pc, last_elem)
        print(
            f"  PC{pc_idx + 1}: sum={corr_sum:.3f}, var={corr_var:.3f}, "
            f"elem[0]={corr_first:.3f}, elem[4]={corr_center:.3f}, elem[8]={corr_last:.3f}"
        )

    print("\nVAE-B PC correlations with operation properties:")
    for pc_idx in range(3):
        pc = z_B_3d[:, pc_idx]
        corr_sum, _ = spearmanr(pc, output_sums)
        corr_var, _ = spearmanr(pc, output_vars)
        corr_first, _ = spearmanr(pc, first_elem)
        corr_center, _ = spearmanr(pc, center_elem)
        corr_last, _ = spearmanr(pc, last_elem)
        print(
            f"  PC{pc_idx + 1}: sum={corr_sum:.3f}, var={corr_var:.3f}, "
            f"elem[0]={corr_first:.3f}, elem[4]={corr_center:.3f}, elem[8]={corr_last:.3f}"
        )

    # Visualization
    fig = plt.figure(figsize=(18, 12))

    # 3D plot with special ops highlighted
    ax = fig.add_subplot(231, projection="3d")
    ax.scatter(z_A_3d[:, 0], z_A_3d[:, 1], z_A_3d[:, 2], c="gray", s=1, alpha=0.3)
    colors = plt.cm.tab10(np.linspace(0, 1, len(special_ops)))
    for (name, idx), color in zip(special_ops.items(), colors, strict=False):
        ax.scatter(
            [z_A_3d[idx, 0]],
            [z_A_3d[idx, 1]],
            [z_A_3d[idx, 2]],
            c=[color],
            s=100,
            marker="*",
            label=name,
            edgecolors="black",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("VAE-A: Special Operations in Latent Space")
    ax.legend(loc="upper left", fontsize=8)

    ax = fig.add_subplot(232, projection="3d")
    ax.scatter(z_B_3d[:, 0], z_B_3d[:, 1], z_B_3d[:, 2], c="gray", s=1, alpha=0.3)
    for (name, idx), color in zip(special_ops.items(), colors, strict=False):
        ax.scatter(
            [z_B_3d[idx, 0]],
            [z_B_3d[idx, 1]],
            [z_B_3d[idx, 2]],
            c=[color],
            s=100,
            marker="*",
            label=name,
            edgecolors="black",
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("VAE-B: Special Operations in Latent Space")
    ax.legend(loc="upper left", fontsize=8)

    # Color by output sum (bias)
    ax = fig.add_subplot(233, projection="3d")
    scatter = ax.scatter(
        z_A_3d[:, 0],
        z_A_3d[:, 1],
        z_A_3d[:, 2],
        c=output_sums,
        cmap="RdBu",
        s=2,
        alpha=0.5,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("VAE-A: Colored by Output Sum (Bias)")
    fig.colorbar(scatter, ax=ax, shrink=0.5)

    # Color by center element (identity behavior)
    ax = fig.add_subplot(234, projection="3d")
    scatter = ax.scatter(
        z_A_3d[:, 0],
        z_A_3d[:, 1],
        z_A_3d[:, 2],
        c=center_elem,
        cmap="viridis",
        s=2,
        alpha=0.5,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("VAE-A: Colored by op(0,0) Value")
    fig.colorbar(scatter, ax=ax, shrink=0.5)

    # Color by 3-adic index directly
    ax = fig.add_subplot(235, projection="3d")
    indices = np.arange(n_ops)
    scatter = ax.scatter(
        z_A_3d[:, 0],
        z_A_3d[:, 1],
        z_A_3d[:, 2],
        c=indices,
        cmap="twilight",
        s=2,
        alpha=0.5,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("VAE-A: Colored by 3-adic Index")
    fig.colorbar(scatter, ax=ax, shrink=0.5)

    # Same for VAE-B
    ax = fig.add_subplot(236, projection="3d")
    scatter = ax.scatter(
        z_B_3d[:, 0],
        z_B_3d[:, 1],
        z_B_3d[:, 2],
        c=indices,
        cmap="twilight",
        s=2,
        alpha=0.5,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("VAE-B: Colored by 3-adic Index")
    fig.colorbar(scatter, ax=ax, shrink=0.5)

    plt.tight_layout()
    plt.savefig(
        output_path / "3adic_algebraic_structure.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {output_path / '3adic_algebraic_structure.png'}")

    return special_ops


def analyze_cayley_structure(data: dict, output_path: Path):
    """Visualize the Cayley graph structure in latent space."""
    operations = data["operations"]
    z_A = data["z_A"]
    n_ops = len(operations)

    print("\n" + "=" * 60)
    print("CAYLEY GRAPH STRUCTURE")
    print("=" * 60)

    pca = PCA(n_components=3)
    z_3d = pca.fit_transform(z_A)

    fig = plt.figure(figsize=(18, 6))

    # Visualize 3-adic "shells" - operations at fixed Hamming distance from origin
    ax = fig.add_subplot(131, projection="3d")

    # Reference: the "zero" operation (all zeros)
    zero_idx = op_to_index(np.zeros(9))

    # Color by Hamming distance from zero op
    hamming_from_zero = np.array([compute_3adic_distance(zero_idx, i) for i in range(n_ops)])
    scatter = ax.scatter(
        z_3d[:, 0],
        z_3d[:, 1],
        z_3d[:, 2],
        c=hamming_from_zero,
        cmap="viridis",
        s=2,
        alpha=0.5,
    )
    ax.scatter(
        [z_3d[zero_idx, 0]],
        [z_3d[zero_idx, 1]],
        [z_3d[zero_idx, 2]],
        c="red",
        s=200,
        marker="*",
        label="Zero Op",
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("VAE-A: 3-adic Distance from Zero Op\n(Cayley Graph Shells)")
    ax.legend()
    fig.colorbar(scatter, ax=ax, shrink=0.5, label="Hamming dist")

    # Visualize operations by their first 3 3-adic digits (coarse structure)
    ax = fig.add_subplot(132, projection="3d")
    coarse_idx = np.array([i % 27 for i in range(n_ops)])  # First 3 digits = mod 27
    scatter = ax.scatter(
        z_3d[:, 0],
        z_3d[:, 1],
        z_3d[:, 2],
        c=coarse_idx,
        cmap="tab20",
        s=2,
        alpha=0.5,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("VAE-A: Colored by First 3 Digits\n(27 Coarse Classes)")
    fig.colorbar(scatter, ax=ax, shrink=0.5, label="mod 27")

    # Visualize operations by their last 3 3-adic digits
    ax = fig.add_subplot(133, projection="3d")
    fine_idx = np.array([i // (3**6) for i in range(n_ops)])  # Last 3 digits
    scatter = ax.scatter(
        z_3d[:, 0],
        z_3d[:, 1],
        z_3d[:, 2],
        c=fine_idx,
        cmap="tab20",
        s=2,
        alpha=0.5,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("VAE-A: Colored by Last 3 Digits\n(27 Fine Classes)")
    fig.colorbar(scatter, ax=ax, shrink=0.5, label="div 729")

    plt.tight_layout()
    plt.savefig(
        output_path / "3adic_cayley_structure.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {output_path / '3adic_cayley_structure.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze 3-adic structure of Ternary VAE")
    parser.add_argument("--checkpoint", type=str, default="latest.pt")
    parser.add_argument("--output", type=str, default=str(VIZ_DIR / "manifold_viz"))
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()

    checkpoint_dir = CHECKPOINTS_DIR / "v5_5"
    checkpoint_path = checkpoint_dir / args.checkpoint
    output_path = PROJECT_ROOT / args.output
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("3-ADIC ALGEBRAIC STRUCTURE ANALYSIS")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")

    # Load and encode
    data = load_model_and_encode(checkpoint_path, args.device)
    print(f"Loaded {len(data['operations'])} operations")

    # Run analyses
    clustering_results = analyze_3adic_clustering(data, output_path)
    neighbor_results = analyze_single_digit_neighbors(data, output_path)
    analyze_algebraic_structure(data, output_path)
    analyze_cayley_structure(data, output_path)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: 3-ADIC STRUCTURE PRESERVATION")
    print("=" * 60)
    print("\n3-adic distance correlation:")
    print(f"  VAE-A: r={clustering_results['corr_A_dist']:.4f}")
    print(f"  VAE-B: r={clustering_results['corr_B_dist']:.4f}")
    print("\nPosition-distance correlation (negative = 3-adic preserved):")
    print(f"  VAE-A: r={neighbor_results['position_corr_A']:.4f}")
    print(f"  VAE-B: r={neighbor_results['position_corr_B']:.4f}")

    interpretation = ""
    if clustering_results["corr_A_dist"] > 0.3:
        interpretation = "STRONG 3-adic structure preservation"
    elif clustering_results["corr_A_dist"] > 0.1:
        interpretation = "MODERATE 3-adic structure preservation"
    else:
        interpretation = "WEAK 3-adic structure (VAE learned different coordinates)"

    print(f"\nInterpretation: {interpretation}")
    print("\nGenerated files:")
    for f in sorted(output_path.glob("3adic*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
