# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Analyze a training run using trained Epsilon-VAE.

This script:
1. Loads a trained Epsilon-VAE model
2. Analyzes all checkpoints from a specified run
3. Compares predicted vs actual metrics
4. Identifies improvement opportunities
5. Suggests optimal directions in latent space

Usage:
    python src/scripts/epsilon_vae/analyze_run.py --run progressive_tiny_lr
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR, OUTPUT_DIR
from src.models.epsilon_vae import extract_key_weights


class EpsilonVAEFlat(nn.Module):
    """Epsilon-VAE with flat weight encoder (must match training)."""

    def __init__(self, weight_dim: int, latent_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder_flat = nn.Sequential(
            nn.Linear(weight_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )

        self.metric_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def encode(self, weights_flat):
        params = self.encoder_flat(weights_flat)
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar

    def predict_metrics(self, z):
        return self.metric_predictor(z)


def load_checkpoint_data(ckpt_path: Path) -> dict:
    """Load checkpoint and extract weights + metrics."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Extract state dict
    state_dict = (
        ckpt.get("model_state_dict") or ckpt.get("model_state") or ckpt.get("state_dict") or ckpt.get("model") or {}
    )

    # Extract key weights
    weights = extract_key_weights(state_dict)
    if not weights:
        return None

    flat_weights = torch.cat([w.flatten() for w in weights])

    # Extract metrics
    metrics = ckpt.get("metrics", {})

    return {
        "weights": flat_weights,
        "epoch": ckpt.get("epoch", -1),
        "coverage": metrics.get("coverage", 0.0),
        "dist_corr": metrics.get("distance_corr_A", 0.0),
        "rad_hier": metrics.get("radial_corr_A", 0.0),
        "all_metrics": metrics,
    }


def analyze_run(run_dir: Path, model: nn.Module, device: str) -> list:
    """Analyze all checkpoints in a run."""
    results = []

    # Find all checkpoints
    ckpt_files = sorted(run_dir.glob("*.pt"))

    print(f"\n{'=' * 80}")
    print(f"ANALYZING RUN: {run_dir.name}")
    print(f"{'=' * 80}")
    print(f"Found {len(ckpt_files)} checkpoints\n")

    for ckpt_path in ckpt_files:
        data = load_checkpoint_data(ckpt_path)
        if data is None:
            print(f"  Skipping {ckpt_path.name} (no weights)")
            continue

        # Check weight dimension compatibility
        weight_dim = data["weights"].shape[0]
        expected_dim = model.encoder_flat[0].in_features

        if weight_dim != expected_dim:
            print(f"  Skipping {ckpt_path.name} (dim {weight_dim} != expected {expected_dim})")
            continue

        # Get prediction
        with torch.no_grad():
            weights = data["weights"].unsqueeze(0).to(device)
            mu, logvar = model.encode(weights)
            pred_metrics = model.predict_metrics(mu)

        pred = pred_metrics[0].cpu().numpy()
        actual = np.array([data["coverage"], data["dist_corr"], data["rad_hier"]])

        results.append(
            {
                "checkpoint": ckpt_path.name,
                "epoch": data["epoch"],
                "latent_mu": mu[0].cpu().numpy(),
                "latent_logvar": logvar[0].cpu().numpy(),
                "predicted": {
                    "coverage": float(pred[0]),
                    "dist_corr": float(pred[1]),
                    "rad_hier": float(pred[2]),
                },
                "actual": {
                    "coverage": float(actual[0]),
                    "dist_corr": float(actual[1]),
                    "rad_hier": float(actual[2]),
                },
                "error": {
                    "coverage": float(abs(pred[0] - actual[0])),
                    "dist_corr": float(abs(pred[1] - actual[1])),
                    "rad_hier": float(abs(pred[2] - actual[2])),
                },
                "all_metrics": data["all_metrics"],
            }
        )

    return results


def print_detailed_analysis(results: list):
    """Print detailed analysis of results."""

    print(f"\n{'=' * 80}")
    print("CHECKPOINT-BY-CHECKPOINT ANALYSIS")
    print(f"{'=' * 80}\n")

    # Header
    print(f"{'Checkpoint':<20} {'Epoch':>6} | {'ACTUAL':^30} | {'PREDICTED':^30} | {'ERROR':^20}")
    print(
        f"{'':<20} {'':>6} | {'Cov':>8} {'Dist':>10} {'Rad':>10} | {'Cov':>8} {'Dist':>10} {'Rad':>10} | {'Cov':>6} {'Dist':>6} {'Rad':>6}"
    )
    print("-" * 120)

    for r in results:
        a = r["actual"]
        p = r["predicted"]
        e = r["error"]

        print(
            f"{r['checkpoint']:<20} {r['epoch']:>6} | "
            f"{a['coverage']:>8.3f} {a['dist_corr']:>10.3f} {a['rad_hier']:>10.3f} | "
            f"{p['coverage']:>8.3f} {p['dist_corr']:>10.3f} {p['rad_hier']:>10.3f} | "
            f"{e['coverage']:>6.3f} {e['dist_corr']:>6.3f} {e['rad_hier']:>6.3f}"
        )

    # Summary statistics
    print(f"\n{'=' * 80}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 80}\n")

    errors = {
        "coverage": [r["error"]["coverage"] for r in results],
        "dist_corr": [r["error"]["dist_corr"] for r in results],
        "rad_hier": [r["error"]["rad_hier"] for r in results],
    }

    for metric, errs in errors.items():
        print(f"{metric:>12}: MAE={np.mean(errs):.4f}, Max={np.max(errs):.4f}, Min={np.min(errs):.4f}")

    # Find best and worst predictions
    print(f"\n{'=' * 80}")
    print("PREDICTION QUALITY ANALYSIS")
    print(f"{'=' * 80}\n")

    total_errors = [sum(r["error"].values()) for r in results]
    best_idx = np.argmin(total_errors)
    worst_idx = np.argmax(total_errors)

    print(f"BEST PREDICTED:  {results[best_idx]['checkpoint']} (total error: {total_errors[best_idx]:.4f})")
    print(f"WORST PREDICTED: {results[worst_idx]['checkpoint']} (total error: {total_errors[worst_idx]:.4f})")

    # Trajectory analysis
    print(f"\n{'=' * 80}")
    print("TRAINING TRAJECTORY ANALYSIS")
    print(f"{'=' * 80}\n")

    # Sort by epoch
    sorted_results = sorted([r for r in results if r["epoch"] >= 0], key=lambda x: x["epoch"])

    if len(sorted_results) > 1:
        print("Coverage evolution:")
        for r in sorted_results:
            bar_len = int(r["actual"]["coverage"] * 50)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            print(f"  Epoch {r['epoch']:>3}: {r['actual']['coverage']:.3f} |{bar}|")

        print("\nDistance correlation evolution:")
        for r in sorted_results:
            bar_len = int(r["actual"]["dist_corr"] * 50)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            print(f"  Epoch {r['epoch']:>3}: {r['actual']['dist_corr']:.3f} |{bar}|")

    # Pareto analysis
    print(f"\n{'=' * 80}")
    print("PARETO FRONTIER ANALYSIS")
    print(f"{'=' * 80}\n")

    # Find Pareto-optimal checkpoints (maximize both coverage and dist_corr)
    pareto_optimal = []
    for i, r in enumerate(sorted_results):
        is_dominated = False
        for j, other in enumerate(sorted_results):
            if i != j:
                # Check if other dominates r
                if (
                    other["actual"]["coverage"] >= r["actual"]["coverage"]
                    and other["actual"]["dist_corr"] >= r["actual"]["dist_corr"]
                    and (
                        other["actual"]["coverage"] > r["actual"]["coverage"]
                        or other["actual"]["dist_corr"] > r["actual"]["dist_corr"]
                    )
                ):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_optimal.append(r)

    print(f"Pareto-optimal checkpoints ({len(pareto_optimal)}):")
    for r in pareto_optimal:
        print(
            f"  Epoch {r['epoch']:>3}: coverage={r['actual']['coverage']:.3f}, dist_corr={r['actual']['dist_corr']:.3f}, rad_hier={r['actual']['rad_hier']:.3f}"
        )

    # Improvement opportunities
    print(f"\n{'=' * 80}")
    print("IMPROVEMENT OPPORTUNITIES")
    print(f"{'=' * 80}\n")

    # Find checkpoint with best balance (closest to ideal 1.0, 1.0, -1.0)
    ideal = np.array([1.0, 1.0, -1.0])
    distances = []
    for r in sorted_results:
        actual = np.array([r["actual"]["coverage"], r["actual"]["dist_corr"], r["actual"]["rad_hier"]])
        dist = np.linalg.norm(actual - ideal)
        distances.append((r, dist))

    distances.sort(key=lambda x: x[1])

    print("Checkpoints closest to ideal (coverage=1.0, dist_corr=1.0, rad_hier=-1.0):")
    for r, dist in distances[:3]:
        print(f"  Epoch {r['epoch']:>3}: distance={dist:.4f}")
        print(
            f"    Actual:    cov={r['actual']['coverage']:.3f}, dist={r['actual']['dist_corr']:.3f}, rad={r['actual']['rad_hier']:.3f}"
        )
        print(
            f"    Gap to ideal: cov={1.0 - r['actual']['coverage']:.3f}, dist={1.0 - r['actual']['dist_corr']:.3f}, rad={-1.0 - r['actual']['rad_hier']:.3f}"
        )

    # Latent space analysis
    print(f"\n{'=' * 80}")
    print("LATENT SPACE ANALYSIS")
    print(f"{'=' * 80}\n")

    latent_mus = np.array([r["latent_mu"] for r in sorted_results])

    # Compute trajectory length in latent space
    if len(latent_mus) > 1:
        trajectory_length = sum(np.linalg.norm(latent_mus[i + 1] - latent_mus[i]) for i in range(len(latent_mus) - 1))
        print(f"Total trajectory length in latent space: {trajectory_length:.4f}")

        # Compute direction of improvement
        if len(pareto_optimal) >= 2:
            best_pareto = max(pareto_optimal, key=lambda r: r["actual"]["coverage"] + r["actual"]["dist_corr"])
            sorted_results[0]

            best_idx = next(i for i, r in enumerate(sorted_results) if r["epoch"] == best_pareto["epoch"])
            improvement_direction = latent_mus[best_idx] - latent_mus[0]

            print("\nImprovement direction (epoch 0 → best):")
            print(f"  Latent dim with largest change: {np.argmax(np.abs(improvement_direction))}")
            print(f"  Direction magnitude: {np.linalg.norm(improvement_direction):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze training run with Epsilon-VAE")
    parser.add_argument("--run", type=str, default="progressive_tiny_lr", help="Run directory name")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(OUTPUT_DIR / "epsilon_vae_models" / "best.pt"),
        help="Path to trained Epsilon-VAE",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model_path = Path(args.model_path)
    print(f"Loading Epsilon-VAE from {model_path}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})

    # Get weight dimension from config or infer
    weight_dim = 95808  # From training
    latent_dim = config.get("latent_dim", 32)
    hidden_dim = config.get("hidden_dim", 256)

    model = EpsilonVAEFlat(
        weight_dim=weight_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Model loaded (latent_dim={latent_dim}, val_mae={ckpt.get('val_mae', 'N/A')})")

    # Analyze run
    run_dir = CHECKPOINTS_DIR / args.run

    if not run_dir.exists():
        print(f"ERROR: Run directory not found: {run_dir}")
        return

    results = analyze_run(run_dir, model, device)

    if not results:
        print("No compatible checkpoints found!")
        return

    # Print detailed analysis
    print_detailed_analysis(results)

    # Save results
    output_path = OUTPUT_DIR / "epsilon_vae_data" / f"analysis_{args.run}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def convert_to_serializable(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    # Convert numpy arrays to lists for JSON
    json_results = []
    for r in results:
        jr = {k: convert_to_serializable(v) for k, v in r.items() if k not in ["latent_mu", "latent_logvar"]}
        jr["latent_mu"] = r["latent_mu"].tolist()
        jr["latent_logvar"] = r["latent_logvar"].tolist()
        json_results.append(jr)

    with open(output_path, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
