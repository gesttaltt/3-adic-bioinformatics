# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Analyze encoder structure for p-adic representation capacity.

Key insight: A frozen encoder that naturally achieves 100% coverage has learned
a bijective mapping from the p-adic operation space. This is the foundation for:
1. Clean 3-adic semantics (no geometrical contamination)
2. Generalization to p-adic spaces (p=5, 7, 11, ...)

This script analyzes:
1. What encoder configurations achieve 100% coverage
2. The algebraic structure preserved by the encoder
3. How to enforce this for arbitrary p

Usage:
    python src/scripts/epsilon_vae/analyze_padic_structure.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR, OUTPUT_DIR
from src.data.generation import generate_all_ternary_operations


def load_encoder_weights(checkpoint_path: Path) -> dict:
    """Load encoder weights from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model_state = ckpt.get("model_state_dict") or ckpt.get("model_state") or {}

    encoder_A = {}
    encoder_B = {}

    for key, value in model_state.items():
        if "encoder_A" in key:
            clean_key = key.replace("encoder_A.", "")
            encoder_A[clean_key] = value.numpy()
        elif "encoder_B" in key:
            clean_key = key.replace("encoder_B.", "")
            encoder_B[clean_key] = value.numpy()

    metrics = ckpt.get("metrics", {})

    return {
        "encoder_A": encoder_A,
        "encoder_B": encoder_B,
        "metrics": metrics,
        "epoch": ckpt.get("epoch", -1),
    }


def analyze_encoder_bijectivity(encoder_weights: dict, p: int = 3) -> dict:
    """Analyze if encoder preserves bijective structure for p-adic operations.

    For clean p-adic representation:
    - The encoder must map EVERY p^9 operation to a UNIQUE latent point
    - This requires the encoder to have sufficient rank at each layer
    - The total information capacity must be >= log2(p^9) bits

    Args:
        encoder_weights: Dict of encoder weight matrices
        p: Prime base (3 for ternary)

    Returns:
        Analysis of bijectivity properties
    """
    n_operations = p**9  # Total operations in p-adic space
    required_bits = np.log2(n_operations)

    analysis = {
        "p": p,
        "n_operations": n_operations,
        "required_bits": required_bits,
        "layer_analysis": [],
    }

    # Analyze each layer's information capacity
    layer_keys = sorted([k for k in encoder_weights if "weight" in k])

    cumulative_capacity = float("inf")

    for key in layer_keys:
        w = encoder_weights[key]
        if w.ndim != 2:
            continue

        out_dim, in_dim = w.shape

        # Singular value decomposition
        U, S, Vh = np.linalg.svd(w, full_matrices=False)

        # Effective rank (number of significant singular values)
        threshold = 0.01 * S[0]
        effective_rank = int(np.sum(threshold < S))

        # Information capacity of this layer
        # Each dimension can encode ~log2(p) bits for p-adic
        layer_capacity = effective_rank * np.log2(p)

        # Update cumulative (bottleneck)
        cumulative_capacity = min(cumulative_capacity, layer_capacity)

        # Condition number (stability of mapping)
        condition = S[0] / (S[-1] + 1e-10)

        analysis["layer_analysis"].append(
            {
                "layer": key,
                "shape": f"{in_dim} -> {out_dim}",
                "effective_rank": effective_rank,
                "max_rank": min(in_dim, out_dim),
                "rank_ratio": effective_rank / min(in_dim, out_dim),
                "layer_capacity_bits": layer_capacity,
                "condition_number": condition,
                "top_singular_values": S[:5].tolist(),
            }
        )

    analysis["total_capacity_bits"] = cumulative_capacity
    analysis["can_represent_all"] = cumulative_capacity >= required_bits
    analysis["capacity_margin"] = cumulative_capacity - required_bits

    return analysis


def compare_high_vs_low_coverage(high_cov_ckpt: Path, low_cov_ckpt: Path) -> dict:
    """Compare encoder weights between high and low coverage checkpoints.

    This reveals what changes in the encoder cause coverage loss.

    Args:
        high_cov_ckpt: Path to checkpoint with ~100% coverage
        low_cov_ckpt: Path to checkpoint with lower coverage

    Returns:
        Comparison analysis
    """
    high = load_encoder_weights(high_cov_ckpt)
    low = load_encoder_weights(low_cov_ckpt)

    comparison = {
        "high_coverage": high["metrics"].get("coverage", 0),
        "low_coverage": low["metrics"].get("coverage", 0),
        "weight_changes": [],
    }

    # Compare each weight matrix
    for key in high["encoder_A"]:
        if key not in low["encoder_A"]:
            continue

        w_high = high["encoder_A"][key]
        w_low = low["encoder_A"][key]

        if w_high.shape != w_low.shape:
            continue

        # Compute changes
        diff = w_low - w_high
        relative_change = np.linalg.norm(diff) / (np.linalg.norm(w_high) + 1e-10)

        # Direction change (cosine distance)
        cos_sim = np.dot(w_high.flatten(), w_low.flatten()) / (
            np.linalg.norm(w_high.flatten()) * np.linalg.norm(w_low.flatten()) + 1e-10
        )

        # Singular value changes
        _, S_high, _ = np.linalg.svd(
            w_high.reshape(-1, w_high.shape[-1]) if w_high.ndim > 1 else w_high.reshape(1, -1), full_matrices=False
        )
        _, S_low, _ = np.linalg.svd(
            w_low.reshape(-1, w_low.shape[-1]) if w_low.ndim > 1 else w_low.reshape(1, -1), full_matrices=False
        )

        comparison["weight_changes"].append(
            {
                "layer": key,
                "relative_change": float(relative_change),
                "cosine_similarity": float(cos_sim),
                "norm_high": float(np.linalg.norm(w_high)),
                "norm_low": float(np.linalg.norm(w_low)),
                "norm_change": float(np.linalg.norm(w_low) - np.linalg.norm(w_high)),
            }
        )

    # Sort by relative change
    comparison["weight_changes"].sort(key=lambda x: x["relative_change"], reverse=True)

    return comparison


def analyze_latent_coverage(checkpoint_path: Path, device: str = "cpu") -> dict:
    """Analyze how well the latent space covers all ternary operations.

    This directly tests if the encoder achieves bijective mapping.

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to use

    Returns:
        Coverage analysis
    """
    from scripts.epsilon_vae.extract_embeddings import load_model_from_checkpoint

    # Load model
    model = load_model_from_checkpoint(checkpoint_path, device)
    if model is None:
        return {"error": "Could not load model"}

    # Generate all ternary operations
    all_ops = generate_all_ternary_operations()  # (19683, 9)
    all_ops_tensor = torch.tensor(all_ops, dtype=torch.float32, device=device)

    # Get latent representations
    model.eval()
    with torch.no_grad():
        outputs = model(all_ops_tensor, compute_control=False)
        z_A = outputs["z_A_euc"].cpu().numpy()  # Euclidean latent (before projection)
        outputs["z_A_hyp"].cpu().numpy()  # Hyperbolic (after projection)

    # Analyze coverage
    n_ops = len(all_ops)

    # Check for collisions (same latent for different operations)
    # Use approximate nearest neighbors for efficiency
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(z_A)
    distances, indices = nn.kneighbors(z_A)

    # distances[:, 1] is distance to nearest neighbor (excluding self)
    nn_distances = distances[:, 1]

    # Operations with very close neighbors might be "colliding"
    collision_threshold = 0.01
    potential_collisions = np.sum(nn_distances < collision_threshold)

    # Compute spread of latent space
    latent_std = np.std(z_A, axis=0)
    latent_range = np.ptp(z_A, axis=0)

    # Effective dimensionality (how many dimensions are actually used)
    from sklearn.decomposition import PCA

    pca = PCA()
    pca.fit(z_A)
    explained_var_ratio = pca.explained_variance_ratio_
    effective_dim = np.sum(np.cumsum(explained_var_ratio) < 0.99) + 1

    return {
        "n_operations": n_ops,
        "latent_dim": z_A.shape[1],
        "effective_dim": int(effective_dim),
        "potential_collisions": int(potential_collisions),
        "collision_rate": float(potential_collisions / n_ops),
        "mean_nn_distance": float(np.mean(nn_distances)),
        "min_nn_distance": float(np.min(nn_distances)),
        "latent_std_per_dim": latent_std.tolist(),
        "latent_range_per_dim": latent_range.tolist(),
        "pca_explained_variance": explained_var_ratio[:10].tolist(),
    }


def visualize_padic_analysis(
    bijectivity: dict,
    coverage_analysis: dict,
    output_dir: Path,
    run_name: str,
):
    """Create visualizations for p-adic structure analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Layer capacity vs required bits
    ax1 = axes[0, 0]
    layers = [la["layer"].split(".")[-1] for la in bijectivity["layer_analysis"]]
    capacities = [la["layer_capacity_bits"] for la in bijectivity["layer_analysis"]]

    ax1.bar(range(len(layers)), capacities, color="steelblue", alpha=0.7)
    ax1.axhline(
        y=bijectivity["required_bits"],
        color="r",
        linestyle="--",
        label=f"Required: {bijectivity['required_bits']:.1f} bits",
    )
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers, rotation=45, ha="right")
    ax1.set_ylabel("Information Capacity (bits)")
    ax1.set_title(f"{run_name}: Layer Capacity for p={bijectivity['p']}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Effective rank per layer
    ax2 = axes[0, 1]
    ranks = [la["effective_rank"] for la in bijectivity["layer_analysis"]]
    max_ranks = [la["max_rank"] for la in bijectivity["layer_analysis"]]

    x = np.arange(len(layers))
    width = 0.35
    ax2.bar(x - width / 2, ranks, width, label="Effective Rank", color="steelblue")
    ax2.bar(x + width / 2, max_ranks, width, label="Max Rank", color="lightgray")
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers, rotation=45, ha="right")
    ax2.set_ylabel("Rank")
    ax2.set_title(f"{run_name}: Effective vs Max Rank")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Latent space PCA variance
    ax3 = axes[1, 0]
    if "pca_explained_variance" in coverage_analysis:
        cumvar = np.cumsum(coverage_analysis["pca_explained_variance"])
        ax3.plot(range(1, len(cumvar) + 1), cumvar, "o-", color="purple")
        ax3.axhline(y=0.99, color="r", linestyle="--", label="99% variance")
        ax3.set_xlabel("Number of Components")
        ax3.set_ylabel("Cumulative Explained Variance")
        ax3.set_title(f"{run_name}: Latent Space Dimensionality")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Coverage analysis not available", ha="center", va="center", transform=ax3.transAxes)

    # Plot 4: Summary metrics
    ax4 = axes[1, 1]
    ax4.axis("off")

    summary_text = f"""
P-ADIC REPRESENTATION ANALYSIS: {run_name}

Algebraic Structure (p={bijectivity["p"]}):
  Total operations: {bijectivity["n_operations"]:,}
  Required capacity: {bijectivity["required_bits"]:.1f} bits
  Encoder capacity: {bijectivity["total_capacity_bits"]:.1f} bits
  CAN REPRESENT ALL: {"YES" if bijectivity["can_represent_all"] else "NO"}
  Capacity margin: {bijectivity["capacity_margin"]:.1f} bits

"""
    if "n_operations" in coverage_analysis:
        summary_text += f"""Latent Space Coverage:
  Effective dimensions: {coverage_analysis["effective_dim"]} / {coverage_analysis["latent_dim"]}
  Potential collisions: {coverage_analysis["potential_collisions"]} ({coverage_analysis["collision_rate"]:.2%})
  Mean NN distance: {coverage_analysis["mean_nn_distance"]:.4f}
  Min NN distance: {coverage_analysis["min_nn_distance"]:.4f}
"""

    summary_text += f"""
IMPLICATION FOR P-ADIC GENERALIZATION:
  {"Encoder has sufficient capacity for bijective mapping" if bijectivity["can_represent_all"] else "Encoder capacity insufficient - information loss"}
  {"Ready for p-adic generalization" if bijectivity["can_represent_all"] and bijectivity["capacity_margin"] > 10 else "May need architecture changes for larger p"}
"""

    ax4.text(
        0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10, verticalalignment="top", fontfamily="monospace"
    )

    plt.tight_layout()
    plt.savefig(output_dir / f"{run_name}_padic_analysis.png", dpi=150)
    plt.close()

    print(f"Saved p-adic analysis for {run_name}")


def main():
    parser = argparse.ArgumentParser(description="Analyze p-adic structure in encoder")
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINTS_DIR), help="Root checkpoint directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR / "epsilon_vae_analysis" / "padic_analysis"),
        help="Output directory",
    )
    parser.add_argument(
        "--runs", nargs="+", default=["progressive_tiny_lr", "v5_11_progressive"], help="Run names to analyze"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"

    all_results = {}

    for run_name in args.runs:
        print(f"\n{'=' * 70}")
        print(f"P-ADIC ANALYSIS: {run_name}")
        print(f"{'=' * 70}")

        run_dir = checkpoint_dir / run_name
        if not run_dir.exists():
            print(f"  Run not found: {run_dir}")
            continue

        # Use best checkpoint or epoch_0 (whichever has higher coverage)
        best_path = run_dir / "best.pt"
        epoch0_path = run_dir / "epoch_0.pt"

        if best_path.exists():
            ckpt_path = best_path
        elif epoch0_path.exists():
            ckpt_path = epoch0_path
        else:
            print("  No suitable checkpoint found")
            continue

        print(f"  Analyzing: {ckpt_path.name}")

        # Load encoder weights
        weights = load_encoder_weights(ckpt_path)
        print(f"  Coverage: {weights['metrics'].get('coverage', 'N/A')}")

        # Analyze bijectivity for p=3
        print("\n  Bijectivity Analysis (p=3):")
        bijectivity = analyze_encoder_bijectivity(weights["encoder_A"], p=3)
        print(f"    Required bits: {bijectivity['required_bits']:.1f}")
        print(f"    Encoder capacity: {bijectivity['total_capacity_bits']:.1f} bits")
        print(f"    Can represent all: {bijectivity['can_represent_all']}")

        for layer in bijectivity["layer_analysis"]:
            print(
                f"    {layer['layer']}: rank={layer['effective_rank']}/{layer['max_rank']} "
                f"({layer['rank_ratio']:.0%}), capacity={layer['layer_capacity_bits']:.1f} bits"
            )

        # Analyze for p=5 (pentary) - future generalization
        print("\n  Bijectivity Analysis (p=5, hypothetical):")
        bijectivity_p5 = analyze_encoder_bijectivity(weights["encoder_A"], p=5)
        print(f"    Required bits: {bijectivity_p5['required_bits']:.1f}")
        print(f"    Can represent all: {bijectivity_p5['can_represent_all']}")
        print(f"    Capacity margin: {bijectivity_p5['capacity_margin']:.1f} bits")

        # Analyze latent coverage
        print("\n  Latent Space Coverage:")
        coverage = analyze_latent_coverage(ckpt_path, device)
        if "error" not in coverage:
            print(f"    Effective dimensions: {coverage['effective_dim']} / {coverage['latent_dim']}")
            print(f"    Potential collisions: {coverage['potential_collisions']} ({coverage['collision_rate']:.2%})")
            print(f"    Mean NN distance: {coverage['mean_nn_distance']:.4f}")
        else:
            print(f"    Error: {coverage['error']}")
            coverage = {}

        # Visualize
        visualize_padic_analysis(bijectivity, coverage, output_dir, run_name)

        all_results[run_name] = {
            "bijectivity_p3": bijectivity,
            "bijectivity_p5": bijectivity_p5,
            "coverage": coverage,
        }

    # Compare high vs low coverage runs
    if len(args.runs) >= 2:
        print(f"\n{'=' * 70}")
        print("HIGH vs LOW COVERAGE COMPARISON")
        print(f"{'=' * 70}")

        # Find highest and lowest coverage
        coverages = []
        for run_name in args.runs:
            run_dir = checkpoint_dir / run_name
            best_path = run_dir / "best.pt"
            if best_path.exists():
                weights = load_encoder_weights(best_path)
                cov = weights["metrics"].get("coverage", 0)
                coverages.append((run_name, cov, best_path))

        if len(coverages) >= 2:
            coverages.sort(key=lambda x: x[1], reverse=True)
            high_run, high_cov, high_path = coverages[0]
            low_run, low_cov, low_path = coverages[-1]

            print(f"\n  High coverage: {high_run} ({high_cov:.3f})")
            print(f"  Low coverage: {low_run} ({low_cov:.3f})")

            comparison = compare_high_vs_low_coverage(high_path, low_path)

            print("\n  Weight changes (high -> low coverage):")
            for change in comparison["weight_changes"][:5]:
                print(
                    f"    {change['layer']}: "
                    f"norm {change['norm_high']:.2f} -> {change['norm_low']:.2f} "
                    f"(change: {change['relative_change']:.1%})"
                )

    # Save results
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(v) for v in obj]
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_dir / "padic_analysis_results.json", "w") as f:
        json.dump(convert_numpy(all_results), f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
