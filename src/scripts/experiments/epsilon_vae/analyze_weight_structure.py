# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Analyze checkpoint weight structure for p-adic representation learning.

This script examines:
1. Encoder weight structure (frozen vs learned)
2. Projection weight evolution
3. What weight configurations preserve 100% coverage
4. The algebraic structure encoded in weights (3-adic -> p-adic generalization)

Key insight: If the frozen encoder naturally learns 100% coverage, the system
learns 3-adic structure without geometrical tradeoffs, enabling:
- Deeper semantics (pure algebraic representation)
- Future generalization to p-adic spaces (p=5, 7, etc.)

Usage:
    python src/scripts/epsilon_vae/analyze_weight_structure.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR, OUTPUT_DIR


def load_checkpoint_weights(checkpoint_path: Path, device: str = "cpu") -> dict:
    """Load and categorize weights from checkpoint.

    Returns:
        Dict with categorized weight tensors and metadata
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_state = (
        ckpt.get("model_state_dict") or ckpt.get("model_state") or ckpt.get("state_dict") or ckpt.get("model") or {}
    )

    metrics = ckpt.get("metrics", {})
    epoch = ckpt.get("epoch", -1)

    # Categorize weights
    encoder_weights = {}
    projection_weights = {}
    decoder_weights = {}
    other_weights = {}

    for key, value in model_state.items():
        if "encoder" in key:
            encoder_weights[key] = value.cpu().numpy()
        elif "projection" in key or "proj_" in key:
            projection_weights[key] = value.cpu().numpy()
        elif "decoder" in key:
            decoder_weights[key] = value.cpu().numpy()
        else:
            other_weights[key] = value.cpu().numpy()

    return {
        "encoder": encoder_weights,
        "projection": projection_weights,
        "decoder": decoder_weights,
        "other": other_weights,
        "metrics": metrics,
        "epoch": epoch,
        "path": str(checkpoint_path),
    }


def compute_weight_statistics(weights: dict) -> dict:
    """Compute statistical properties of weight matrices.

    Args:
        weights: Dict of weight name -> numpy array

    Returns:
        Dict with statistics for each weight
    """
    stats = {}

    for name, w in weights.items():
        if w.ndim < 1:
            continue

        flat = w.flatten()

        # Basic statistics
        stats[name] = {
            "shape": list(w.shape),
            "n_params": w.size,
            "mean": float(np.mean(flat)),
            "std": float(np.std(flat)),
            "min": float(np.min(flat)),
            "max": float(np.max(flat)),
            "l2_norm": float(np.linalg.norm(flat)),
            "sparsity": float(np.mean(np.abs(flat) < 0.01)),  # fraction near zero
        }

        # For 2D weights, compute spectral properties
        if w.ndim == 2:
            try:
                # Singular values
                U, S, Vh = np.linalg.svd(w, full_matrices=False)
                stats[name]["singular_values"] = S[:10].tolist()  # top 10
                stats[name]["condition_number"] = float(S[0] / (S[-1] + 1e-10))
                stats[name]["effective_rank"] = float(np.sum(0.01 * S[0] < S))

                # Frobenius norm (alternative to L2)
                stats[name]["frobenius_norm"] = float(np.linalg.norm(w, "fro"))
            except:
                pass

    return stats


def analyze_encoder_structure(encoder_weights: dict) -> dict:
    """Analyze encoder weight structure for p-adic properties.

    The encoder maps ternary operations (9-dim) to latent space.
    For clean 3-adic representation:
    - Weights should preserve algebraic structure
    - The mapping should be nearly bijective (full coverage)

    Args:
        encoder_weights: Dict of encoder weight tensors

    Returns:
        Analysis results
    """
    analysis = {
        "layer_sizes": [],
        "total_params": 0,
        "weight_flow": [],
    }

    # Track information flow through layers
    layer_names = sorted([k for k in encoder_weights if "weight" in k])

    for name in layer_names:
        w = encoder_weights[name]
        if w.ndim == 2:
            in_dim, out_dim = w.shape[1], w.shape[0]
            analysis["layer_sizes"].append((name.split(".")[-2], in_dim, out_dim))
            analysis["total_params"] += w.size

            # Compute how much information is preserved
            # High effective rank = information preserved
            U, S, Vh = np.linalg.svd(w, full_matrices=False)
            effective_rank = np.sum(0.01 * S[0] < S)
            max_rank = min(in_dim, out_dim)
            info_preservation = effective_rank / max_rank

            analysis["weight_flow"].append(
                {
                    "layer": name,
                    "in_dim": in_dim,
                    "out_dim": out_dim,
                    "effective_rank": int(effective_rank),
                    "max_rank": max_rank,
                    "info_preservation": float(info_preservation),
                    "top_singular": float(S[0]),
                }
            )

    return analysis


def analyze_projection_evolution(checkpoints: list) -> dict:
    """Analyze how projection weights evolve during training.

    The projection maps Euclidean latent to hyperbolic space.
    Key questions:
    - Does the projection converge to a stable structure?
    - What geometric transformation does it learn?

    Args:
        checkpoints: List of (epoch, checkpoint_data) tuples

    Returns:
        Evolution analysis
    """
    evolution = {
        "epochs": [],
        "projection_norms": [],
        "projection_ranks": [],
        "direction_changes": [],
        "coverage": [],
        "distance_corr": [],
    }

    prev_weights = None

    for epoch, ckpt_data in checkpoints:
        proj_weights = ckpt_data["projection"]
        metrics = ckpt_data["metrics"]

        # Aggregate projection statistics
        total_norm = 0
        total_rank = 0
        for name, w in proj_weights.items():
            if w.ndim >= 1:
                total_norm += np.linalg.norm(w.flatten())
                if w.ndim == 2:
                    U, S, Vh = np.linalg.svd(w, full_matrices=False)
                    total_rank += np.sum(0.01 * S[0] < S)

        evolution["epochs"].append(epoch)
        evolution["projection_norms"].append(float(total_norm))
        evolution["projection_ranks"].append(float(total_rank))
        evolution["coverage"].append(float(metrics.get("coverage", 0)))
        evolution["distance_corr"].append(float(metrics.get("distance_corr_A", 0)))

        # Compute direction change from previous
        if prev_weights is not None:
            direction_change = 0
            n_weights = 0
            for name in proj_weights:
                if name in prev_weights:
                    curr = proj_weights[name].flatten()
                    prev = prev_weights[name].flatten()
                    if len(curr) == len(prev):
                        # Cosine similarity
                        cos_sim = np.dot(curr, prev) / (np.linalg.norm(curr) * np.linalg.norm(prev) + 1e-10)
                        direction_change += 1 - cos_sim
                        n_weights += 1
            if n_weights > 0:
                direction_change /= n_weights
            evolution["direction_changes"].append(float(direction_change))
        else:
            evolution["direction_changes"].append(0.0)

        prev_weights = proj_weights

    return evolution


def analyze_coverage_weight_correlation(checkpoints: list) -> dict:
    """Find which weight properties correlate with 100% coverage.

    This is crucial for understanding what enables clean p-adic representation.

    Args:
        checkpoints: List of (epoch, checkpoint_data) tuples

    Returns:
        Correlation analysis
    """
    # Collect features and targets
    features = []
    coverages = []
    feature_names = []

    for _epoch, ckpt_data in checkpoints:
        metrics = ckpt_data["metrics"]
        coverage = metrics.get("coverage", 0)

        # Extract weight features
        feat_dict = {}

        # Encoder features
        for name, w in ckpt_data["encoder"].items():
            if w.ndim >= 1:
                prefix = name.replace(".", "_")
                feat_dict[f"{prefix}_norm"] = np.linalg.norm(w.flatten())
                feat_dict[f"{prefix}_std"] = np.std(w.flatten())
                if w.ndim == 2:
                    U, S, Vh = np.linalg.svd(w, full_matrices=False)
                    feat_dict[f"{prefix}_rank"] = np.sum(0.01 * S[0] < S)
                    feat_dict[f"{prefix}_top_sv"] = S[0]

        # Projection features
        for name, w in ckpt_data["projection"].items():
            if w.ndim >= 1:
                prefix = name.replace(".", "_")
                feat_dict[f"{prefix}_norm"] = np.linalg.norm(w.flatten())
                feat_dict[f"{prefix}_std"] = np.std(w.flatten())

        if not feature_names:
            feature_names = sorted(feat_dict.keys())

        features.append([feat_dict.get(fn, 0) for fn in feature_names])
        coverages.append(coverage)

    features = np.array(features)
    coverages = np.array(coverages)

    # Compute correlations
    correlations = {}
    for i, name in enumerate(feature_names):
        if np.std(features[:, i]) > 1e-10 and np.std(coverages) > 1e-10:
            corr, pval = spearmanr(features[:, i], coverages)
            correlations[name] = {
                "spearman_corr": float(corr),
                "p_value": float(pval),
            }

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]["spearman_corr"]), reverse=True)

    return {
        "correlations": dict(sorted_corrs[:20]),  # top 20
        "n_checkpoints": len(checkpoints),
        "coverage_range": [float(coverages.min()), float(coverages.max())],
    }


def visualize_weight_analysis(
    checkpoints: list,
    evolution: dict,
    output_dir: Path,
    run_name: str,
):
    """Create visualizations for weight analysis.

    Args:
        checkpoints: List of (epoch, checkpoint_data) tuples
        evolution: Evolution analysis dict
        output_dir: Output directory
        run_name: Name of the run
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = evolution["epochs"]

    # Plot 1: Projection norm and coverage
    ax1 = axes[0, 0]
    ax1.plot(epochs, evolution["projection_norms"], "b-o", label="Projection Norm")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Projection L2 Norm", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs, evolution["coverage"], "r-s", label="Coverage")
    ax1_twin.set_ylabel("Coverage", color="r")
    ax1_twin.tick_params(axis="y", labelcolor="r")
    ax1.set_title(f"{run_name}: Projection Norm vs Coverage")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Direction changes (stability)
    ax2 = axes[0, 1]
    ax2.plot(epochs[1:], evolution["direction_changes"][1:], "g-o")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Direction Change (1 - cosine sim)")
    ax2.set_title(f"{run_name}: Weight Direction Stability")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Coverage vs Distance Correlation tradeoff
    ax3 = axes[1, 0]
    scatter = ax3.scatter(
        evolution["coverage"], evolution["distance_corr"], c=epochs, cmap="viridis", s=100, edgecolors="black"
    )
    plt.colorbar(scatter, ax=ax3, label="Epoch")
    ax3.set_xlabel("Coverage")
    ax3.set_ylabel("Distance Correlation")
    ax3.set_title(f"{run_name}: Coverage vs Dist Corr Tradeoff")
    ax3.grid(True, alpha=0.3)

    # Mark the "ideal" zone (high coverage, reasonable dist_corr)
    ax3.axvline(x=0.95, color="r", linestyle="--", alpha=0.5, label="95% coverage")
    ax3.axhline(y=0.5, color="b", linestyle="--", alpha=0.5, label="50% dist_corr")
    ax3.legend()

    # Plot 4: Effective rank evolution
    ax4 = axes[1, 1]
    ax4.plot(epochs, evolution["projection_ranks"], "m-o")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Total Effective Rank")
    ax4.set_title(f"{run_name}: Projection Effective Rank")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"{run_name}_weight_analysis.png", dpi=150)
    plt.close()

    print(f"Saved weight analysis for {run_name}")


def analyze_full_coverage_checkpoints(checkpoints: list) -> dict:
    """Find and analyze checkpoints with coverage >= 0.99 (near 100%).

    These are the checkpoints that achieve clean p-adic representation.

    Args:
        checkpoints: List of (epoch, checkpoint_data) tuples

    Returns:
        Analysis of high-coverage checkpoints
    """
    high_coverage = []

    for epoch, ckpt_data in checkpoints:
        coverage = ckpt_data["metrics"].get("coverage", 0)
        if coverage >= 0.99:
            high_coverage.append((epoch, ckpt_data, coverage))

    if not high_coverage:
        return {"message": "No checkpoints with coverage >= 99%", "count": 0}

    # Analyze common properties
    analysis = {
        "count": len(high_coverage),
        "epochs": [ep for ep, _, _ in high_coverage],
        "coverages": [cov for _, _, cov in high_coverage],
        "common_properties": {},
    }

    # Find weight properties that are consistent across high-coverage checkpoints
    all_stats = []
    for epoch, ckpt_data, _cov in high_coverage:
        stats = compute_weight_statistics(ckpt_data["encoder"])
        stats.update(compute_weight_statistics(ckpt_data["projection"]))
        all_stats.append(stats)

    # Find properties with low variance (consistent across high-coverage)
    if len(all_stats) > 1:
        property_variances = {}
        for key in all_stats[0]:
            if isinstance(all_stats[0][key], dict) and "l2_norm" in all_stats[0][key]:
                norms = [s[key]["l2_norm"] for s in all_stats if key in s]
                if len(norms) > 1:
                    property_variances[key] = {
                        "mean_norm": float(np.mean(norms)),
                        "std_norm": float(np.std(norms)),
                        "cv": float(np.std(norms) / (np.mean(norms) + 1e-10)),
                    }

        # Sort by coefficient of variation (lowest = most consistent)
        sorted_props = sorted(property_variances.items(), key=lambda x: x[1]["cv"])
        analysis["most_consistent_weights"] = dict(sorted_props[:10])

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Analyze checkpoint weight structure")
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINTS_DIR), help="Root checkpoint directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR / "epsilon_vae_analysis" / "weight_analysis"),
        help="Output directory",
    )
    parser.add_argument(
        "--runs", nargs="+", default=["progressive_tiny_lr", "progressive_conservative"], help="Run names to analyze"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for run_name in args.runs:
        print(f"\n{'=' * 70}")
        print(f"ANALYZING: {run_name}")
        print(f"{'=' * 70}")

        run_dir = checkpoint_dir / run_name
        if not run_dir.exists():
            print(f"  Run directory not found: {run_dir}")
            continue

        # Load all checkpoints
        checkpoints = []
        for ckpt_path in sorted(run_dir.glob("epoch_*.pt")):
            epoch = int(ckpt_path.stem.split("_")[1])
            ckpt_data = load_checkpoint_weights(ckpt_path)
            checkpoints.append((epoch, ckpt_data))

        # Add best checkpoint
        best_path = run_dir / "best.pt"
        if best_path.exists():
            ckpt_data = load_checkpoint_weights(best_path)
            checkpoints.append((-1, ckpt_data))

        checkpoints.sort(key=lambda x: x[0])

        print(f"  Loaded {len(checkpoints)} checkpoints")

        if len(checkpoints) < 2:
            print("  Skipping - need at least 2 checkpoints")
            continue

        # Analyze encoder structure (first checkpoint)
        print("\n  Encoder Structure:")
        encoder_analysis = analyze_encoder_structure(checkpoints[0][1]["encoder"])
        for layer_info in encoder_analysis["weight_flow"]:
            print(
                f"    {layer_info['layer'].split('.')[-2]}: "
                f"{layer_info['in_dim']} -> {layer_info['out_dim']} "
                f"(rank={layer_info['effective_rank']}/{layer_info['max_rank']}, "
                f"info={layer_info['info_preservation']:.1%})"
            )

        # Analyze projection evolution
        print("\n  Projection Evolution:")
        evolution = analyze_projection_evolution(checkpoints)

        # Coverage stability
        cov_start = evolution["coverage"][0]
        cov_end = evolution["coverage"][-1]
        print(f"    Coverage: {cov_start:.3f} -> {cov_end:.3f}")

        # Direction stability
        avg_direction_change = np.mean(evolution["direction_changes"][1:])
        print(f"    Avg direction change: {avg_direction_change:.4f}")

        # Find checkpoints with 100% coverage
        print("\n  High-Coverage Checkpoints:")
        high_cov_analysis = analyze_full_coverage_checkpoints(checkpoints)
        print(f"    Count with coverage >= 99%: {high_cov_analysis['count']}")
        if high_cov_analysis["count"] > 0:
            print(f"    Epochs: {high_cov_analysis['epochs']}")

        # Coverage-weight correlations
        print("\n  Coverage-Weight Correlations (top 5):")
        corr_analysis = analyze_coverage_weight_correlation(checkpoints)
        for _i, (name, vals) in enumerate(list(corr_analysis["correlations"].items())[:5]):
            print(f"    {name}: r={vals['spearman_corr']:.3f} (p={vals['p_value']:.3e})")

        # Visualize
        visualize_weight_analysis(checkpoints, evolution, output_dir, run_name)

        # Store results
        all_results[run_name] = {
            "encoder_analysis": encoder_analysis,
            "evolution": evolution,
            "high_coverage": high_cov_analysis,
            "correlations": corr_analysis,
        }

    # Cross-run comparison
    print(f"\n{'=' * 70}")
    print("CROSS-RUN COMPARISON")
    print(f"{'=' * 70}")

    print("\n  Coverage Preservation:")
    for run_name, results in all_results.items():
        evolution = results["evolution"]
        cov_start = evolution["coverage"][0] if evolution["coverage"] else 0
        cov_end = evolution["coverage"][-1] if evolution["coverage"] else 0
        high_cov = results["high_coverage"]["count"]
        print(f"    {run_name}: {cov_start:.3f} -> {cov_end:.3f} (epochs with 100%: {high_cov})")

    print("\n  Weight Stability (lower = more stable):")
    for run_name, results in all_results.items():
        evolution = results["evolution"]
        avg_change = np.mean(evolution["direction_changes"][1:]) if len(evolution["direction_changes"]) > 1 else 0
        print(f"    {run_name}: {avg_change:.4f}")

    # Save all results
    # Convert numpy types for JSON serialization
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

    with open(output_dir / "weight_analysis_results.json", "w") as f:
        json.dump(convert_numpy(all_results), f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
