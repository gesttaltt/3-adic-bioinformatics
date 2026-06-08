# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Analyze progressive training checkpoints using Epsilon-VAE.

This script uses the trained Hybrid Epsilon-VAE to:
1. Encode checkpoint weights into the latent space
2. Predict embedding geometry for each checkpoint
3. Track how the model evolves during training
4. Visualize training trajectories in latent space

Usage:
    python src/scripts/epsilon_vae/analyze_progressive_checkpoints.py
    python src/scripts/epsilon_vae/analyze_progressive_checkpoints.py --runs progressive_tiny_lr progressive_conservative
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.epsilon_vae.extract_embeddings import (
    select_anchor_operations,
)
from scripts.epsilon_vae.train_epsilon_vae_hybrid import HybridEpsilonVAE

from src.config.paths import CHECKPOINTS_DIR, OUTPUT_DIR
from src.models.epsilon_vae import extract_key_weights


def load_epsilon_vae(model_path: Path, device: str = "cpu"):
    """Load trained Hybrid Epsilon-VAE model.

    Args:
        model_path: Path to model checkpoint
        device: Device to load model to

    Returns:
        Loaded model ready for inference
    """
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt["config"]

    model = HybridEpsilonVAE(
        weight_dim=config["weight_dim"],
        n_anchors=config["n_anchors"],
        embed_dim=config["embed_dim"],
        latent_dim=config["latent_dim"],
        hidden_dim=config["hidden_dim"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Loaded Epsilon-VAE from {model_path.name}")
    print(f"  Weight dim: {config['weight_dim']}")
    print(f"  Latent dim: {config['latent_dim']}")
    print(f"  N anchors: {config['n_anchors']}")

    return model, config


def extract_weights_from_checkpoint(checkpoint_path: Path, device: str = "cpu"):
    """Extract flattened weights from a TernaryVAE checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to use

    Returns:
        Flattened weight tensor or None if extraction fails
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model_state = (
            ckpt.get("model_state_dict") or ckpt.get("model_state") or ckpt.get("state_dict") or ckpt.get("model") or {}
        )

        if not model_state:
            return None, None

        weights = extract_key_weights(model_state)
        if not weights:
            return None, None

        # Flatten weights
        flattened = np.concatenate([w.cpu().numpy().flatten() for w in weights])

        # Also extract metrics if available
        metrics = ckpt.get("metrics", {})

        return flattened, metrics

    except Exception as e:
        print(f"  Error extracting weights: {e}")
        return None, None


def analyze_run(
    run_name: str,
    epsilon_vae,
    config: dict,
    checkpoint_dir: Path,
    anchor_ops: torch.Tensor,
    device: str = "cpu",
):
    """Analyze all checkpoints from a single training run.

    Args:
        run_name: Name of the run directory
        epsilon_vae: Trained Epsilon-VAE model
        config: Model config
        checkpoint_dir: Root checkpoint directory
        anchor_ops: Anchor operations for embedding extraction
        device: Device to use

    Returns:
        Dict with trajectory data
    """
    run_dir = checkpoint_dir / run_name
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return None

    # Find all epoch checkpoints
    checkpoints = []
    for ckpt_path in run_dir.glob("epoch_*.pt"):
        epoch = int(ckpt_path.stem.split("_")[1])
        checkpoints.append((epoch, ckpt_path))

    # Add best checkpoint
    best_path = run_dir / "best.pt"
    if best_path.exists():
        checkpoints.append((-1, best_path))  # -1 marks it as "best"

    # Sort by epoch
    checkpoints.sort(key=lambda x: x[0])

    print(f"\nAnalyzing run: {run_name}")
    print(f"  Found {len(checkpoints)} checkpoints")

    trajectory = {
        "run_name": run_name,
        "epochs": [],
        "latent_z": [],
        "predicted_embeddings": [],
        "predicted_metrics": [],
        "actual_metrics": [],
        "checkpoint_paths": [],
    }

    for epoch, ckpt_path in checkpoints:
        epoch_label = "best" if epoch == -1 else epoch
        print(f"  Processing epoch {epoch_label}...")

        # Extract weights
        weights, actual_metrics = extract_weights_from_checkpoint(ckpt_path, device)

        if weights is None:
            print("    Skipping - failed to extract weights")
            continue

        # Check weight dimension
        if len(weights) != config["weight_dim"]:
            print(f"    Skipping - weight dim mismatch: {len(weights)} vs {config['weight_dim']}")
            continue

        # Encode in Epsilon-VAE
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            mu, logvar = epsilon_vae.encode(weights_tensor)
            z = mu  # Use mean for deterministic analysis

            # Predict embeddings and metrics
            pred_embeddings = epsilon_vae.decode_embeddings(z)
            pred_metrics = epsilon_vae.predict_metrics(z)

        trajectory["epochs"].append(epoch)
        trajectory["latent_z"].append(z.cpu().numpy().squeeze())
        trajectory["predicted_embeddings"].append(pred_embeddings.cpu().numpy().squeeze())
        trajectory["predicted_metrics"].append(pred_metrics.cpu().numpy().squeeze())
        trajectory["actual_metrics"].append(
            {
                "coverage": actual_metrics.get("coverage", 0.0),
                "distance_corr_A": actual_metrics.get("distance_corr_A", 0.0),
                "radial_corr_A": actual_metrics.get("radial_corr_A", 0.0),
            }
        )
        trajectory["checkpoint_paths"].append(str(ckpt_path))

    # Convert to arrays
    trajectory["latent_z"] = np.array(trajectory["latent_z"])
    trajectory["predicted_metrics"] = np.array(trajectory["predicted_metrics"])

    return trajectory


def compute_trajectory_metrics(trajectory: dict) -> dict:
    """Compute summary metrics for a training trajectory.

    Args:
        trajectory: Dict from analyze_run

    Returns:
        Dict with trajectory statistics
    """
    latent_z = trajectory["latent_z"]
    pred_metrics = trajectory["predicted_metrics"]

    # Trajectory length (total distance in latent space)
    diffs = np.diff(latent_z, axis=0)
    trajectory_length = np.sum(np.linalg.norm(diffs, axis=1))

    # Average step size
    avg_step = np.mean(np.linalg.norm(diffs, axis=1))

    # Total displacement (start to end)
    total_displacement = np.linalg.norm(latent_z[-1] - latent_z[0])

    # Efficiency (displacement / length) - 1 means straight line
    efficiency = total_displacement / trajectory_length if trajectory_length > 0 else 0

    # Metric evolution
    coverage_start = pred_metrics[0, 0]
    coverage_end = pred_metrics[-1, 0]
    dist_corr_start = pred_metrics[0, 1]
    dist_corr_end = pred_metrics[-1, 1]

    return {
        "trajectory_length": trajectory_length,
        "avg_step_size": avg_step,
        "total_displacement": total_displacement,
        "efficiency": efficiency,
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "coverage_delta": coverage_end - coverage_start,
        "dist_corr_start": dist_corr_start,
        "dist_corr_end": dist_corr_end,
        "dist_corr_delta": dist_corr_end - dist_corr_start,
    }


def visualize_trajectories(
    trajectories: list,
    output_dir: Path,
    method: str = "pca",
):
    """Visualize training trajectories in 2D latent space.

    Args:
        trajectories: List of trajectory dicts
        output_dir: Directory to save plots
        method: Dimensionality reduction method ('pca' or 'tsne')
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all latent points
    all_points = []
    labels = []
    for traj in trajectories:
        all_points.extend(traj["latent_z"])
        labels.extend([traj["run_name"]] * len(traj["latent_z"]))

    all_points = np.array(all_points)

    # Reduce to 2D
    if method == "pca":
        reducer = PCA(n_components=2)
        points_2d = reducer.fit_transform(all_points)
        explained_var = reducer.explained_variance_ratio_.sum()
        print(f"PCA explained variance: {explained_var:.1%}")
    else:
        reducer = TSNE(n_components=2, random_state=42)
        points_2d = reducer.fit_transform(all_points)
        explained_var = None

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Trajectories with arrows
    ax1 = axes[0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

    start_idx = 0
    for i, traj in enumerate(trajectories):
        n_points = len(traj["latent_z"])
        end_idx = start_idx + n_points

        points = points_2d[start_idx:end_idx]
        epochs = traj["epochs"]

        # Plot trajectory
        ax1.plot(
            points[:, 0],
            points[:, 1],
            "o-",
            color=colors[i],
            label=traj["run_name"],
            markersize=6,
            linewidth=2,
            alpha=0.8,
        )

        # Mark start and end
        ax1.scatter(
            points[0, 0], points[0, 1], s=150, c=[colors[i]], marker="s", edgecolors="black", linewidths=2, zorder=5
        )
        ax1.scatter(
            points[-1, 0], points[-1, 1], s=150, c=[colors[i]], marker="*", edgecolors="black", linewidths=2, zorder=5
        )

        # Add epoch labels
        for j, (x, y) in enumerate(points):
            epoch_label = "best" if epochs[j] == -1 else str(epochs[j])
            ax1.annotate(epoch_label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.7)

        start_idx = end_idx

    ax1.set_xlabel("Latent Dimension 1")
    ax1.set_ylabel("Latent Dimension 2")
    ax1.set_title(f"Training Trajectories in Epsilon-VAE Latent Space\n({method.upper()})")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Predicted metrics evolution
    ax2 = axes[1]

    for i, traj in enumerate(trajectories):
        pred_metrics = traj["predicted_metrics"]
        epochs = np.array(traj["epochs"])

        # Replace -1 (best) with max epoch + 10 for plotting
        plot_epochs = np.where(epochs == -1, epochs.max() + 10, epochs)

        ax2.plot(
            plot_epochs,
            pred_metrics[:, 0],
            "o-",
            color=colors[i],
            label=f"{traj['run_name']} coverage",
            linewidth=2,
            markersize=6,
        )
        ax2.plot(
            plot_epochs,
            pred_metrics[:, 1],
            "s--",
            color=colors[i],
            alpha=0.6,
            label=f"{traj['run_name']} dist_corr",
            linewidth=2,
            markersize=6,
        )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Predicted Metric Value")
    ax2.set_title("Predicted Metrics During Training")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_analysis.png", dpi=150)
    plt.close()

    print(f"\nSaved trajectory visualization to {output_dir / 'trajectory_analysis.png'}")

    # Plot 3: Compare predicted vs actual metrics (if available)
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, traj in enumerate(trajectories):
        pred_coverage = traj["predicted_metrics"][:, 0]
        actual_coverage = [m["coverage"] for m in traj["actual_metrics"]]
        epochs = np.array(traj["epochs"])
        plot_epochs = np.where(epochs == -1, epochs.max() + 10, epochs)

        ax.plot(
            plot_epochs,
            pred_coverage,
            "o-",
            color=colors[i],
            label=f"{traj['run_name']} predicted",
            linewidth=2,
            markersize=6,
        )
        ax.plot(
            plot_epochs,
            actual_coverage,
            "s--",
            color=colors[i],
            alpha=0.6,
            label=f"{traj['run_name']} actual",
            linewidth=2,
            markersize=4,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Coverage")
    ax.set_title("Predicted vs Actual Coverage")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "predicted_vs_actual.png", dpi=150)
    plt.close()

    print(f"Saved predicted vs actual to {output_dir / 'predicted_vs_actual.png'}")


def visualize_embedding_evolution(trajectory: dict, output_dir: Path):
    """Visualize how predicted embeddings evolve during training.

    Args:
        trajectory: Single trajectory dict
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.array(trajectory["predicted_embeddings"])
    epochs = trajectory["epochs"]
    run_name = trajectory["run_name"]

    if len(epochs) < 2:
        print(f"  Skipping {run_name} - not enough epochs for visualization")
        return

    # Select a few key epochs to visualize
    key_indices = [0, len(epochs) // 2, -1]
    key_embeddings = [embeddings[i] for i in key_indices]
    key_epochs = [epochs[i] for i in key_indices]

    # Reduce all embeddings for context
    all_flat = embeddings.reshape(len(embeddings), -1)
    pca = PCA(n_components=2)
    pca.fit_transform(all_flat)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, emb, epoch in zip(axes, key_embeddings, key_epochs, strict=False):
        # Each embedding is (256, 16) - show as image
        im = ax.imshow(emb.T, aspect="auto", cmap="RdBu_r")
        epoch_label = "best" if epoch == -1 else f"epoch {epoch}"
        ax.set_title(f"{run_name}\n{epoch_label}")
        ax.set_xlabel("Anchor Index")
        ax.set_ylabel("Embedding Dimension")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_dir / f"{run_name}_embedding_evolution.png", dpi=150)
    plt.close()

    print(f"Saved embedding evolution for {run_name}")


def main():
    parser = argparse.ArgumentParser(description="Analyze progressive checkpoints with Epsilon-VAE")
    parser.add_argument(
        "--epsilon_vae_path",
        type=str,
        default=str(OUTPUT_DIR / "epsilon_vae_hybrid_models" / "best.pt"),
        help="Path to trained Epsilon-VAE",
    )
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINTS_DIR), help="Root checkpoint directory")
    parser.add_argument(
        "--output_dir", type=str, default=str(OUTPUT_DIR / "epsilon_vae_analysis"), help="Output directory for results"
    )
    parser.add_argument(
        "--runs", nargs="+", default=["progressive_tiny_lr", "progressive_conservative"], help="Run names to analyze"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()

    epsilon_vae_path = Path(args.epsilon_vae_path)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Epsilon-VAE
    print(f"\n{'=' * 70}")
    print("LOADING EPSILON-VAE")
    print(f"{'=' * 70}")
    epsilon_vae, config = load_epsilon_vae(epsilon_vae_path, device)

    # Load anchor operations
    anchor_ops = select_anchor_operations(config["n_anchors"])

    # Analyze each run
    print(f"\n{'=' * 70}")
    print("ANALYZING TRAINING RUNS")
    print(f"{'=' * 70}")

    trajectories = []
    for run_name in args.runs:
        traj = analyze_run(
            run_name=run_name,
            epsilon_vae=epsilon_vae,
            config=config,
            checkpoint_dir=checkpoint_dir,
            anchor_ops=anchor_ops,
            device=device,
        )
        if traj and len(traj["epochs"]) > 0:
            trajectories.append(traj)

    if not trajectories:
        print("No valid trajectories found!")
        return

    # Compute trajectory metrics
    print(f"\n{'=' * 70}")
    print("TRAJECTORY METRICS")
    print(f"{'=' * 70}")

    all_metrics = []
    for traj in trajectories:
        metrics = compute_trajectory_metrics(traj)
        metrics["run_name"] = traj["run_name"]
        all_metrics.append(metrics)

        print(f"\n{traj['run_name']}:")
        print(f"  Trajectory length: {metrics['trajectory_length']:.3f}")
        print(f"  Efficiency: {metrics['efficiency']:.3f}")
        print(
            f"  Coverage: {metrics['coverage_start']:.3f} -> {metrics['coverage_end']:.3f} (delta: {metrics['coverage_delta']:+.3f})"
        )
        print(
            f"  Dist Corr: {metrics['dist_corr_start']:.3f} -> {metrics['dist_corr_end']:.3f} (delta: {metrics['dist_corr_delta']:+.3f})"
        )

    # Save metrics (convert numpy types to Python types)
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_dir / "trajectory_metrics.json", "w") as f:
        json.dump([convert_numpy(m) for m in all_metrics], f, indent=2)

    # Visualize
    print(f"\n{'=' * 70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'=' * 70}")

    visualize_trajectories(trajectories, output_dir)

    for traj in trajectories:
        visualize_embedding_evolution(traj, output_dir)

    # Print sample predictions
    print(f"\n{'=' * 70}")
    print("SAMPLE PREDICTIONS")
    print(f"{'=' * 70}")

    for traj in trajectories:
        print(f"\n{traj['run_name']}:")
        for _i, (epoch, pred, actual) in enumerate(
            zip(traj["epochs"], traj["predicted_metrics"], traj["actual_metrics"], strict=False)
        ):
            epoch_label = "best" if epoch == -1 else f"epoch_{epoch}"
            print(f"\n  [{epoch_label}]")
            print(f"    Predicted: cov={pred[0]:.3f}, dist={pred[1]:.3f}, rad={pred[2]:.3f}")
            print(
                f"    Actual:    cov={actual['coverage']:.3f}, dist={actual['distance_corr_A']:.3f}, rad={actual['radial_corr_A']:.3f}"
            )

    print(f"\n{'=' * 70}")
    print(f"Analysis complete! Results saved to {output_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
