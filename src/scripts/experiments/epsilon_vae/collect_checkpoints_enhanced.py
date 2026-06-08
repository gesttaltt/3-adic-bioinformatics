# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Enhanced checkpoint collection for Epsilon-VAE training.

Extracts comprehensive features from checkpoints:
1. All 16+ metrics (not just 3)
2. Training hyperparameters from config
3. Weight statistics (mean, std, norm, sparsity) per layer
4. Optimizer state statistics (momentum magnitudes)
5. Epoch and training progress information

Usage:
    python src/scripts/epsilon_vae/collect_checkpoints_enhanced.py
    python src/scripts/epsilon_vae/collect_checkpoints_enhanced.py --cutoff 2025-12-26
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR, OUTPUT_DIR
from src.models.epsilon_vae import extract_key_weights

# Key hyperparameters that affect training dynamics
KEY_HYPERPARAMS = [
    "lr",
    "batch_size",
    "epochs",
    "weight_decay",
    "curvature",
    "max_radius",
    "radial_weight",
    "margin_weight",
    "option_c",
    "dual_projection",
    "progressive_unfreeze",
    "encoder_a_lr_scale",
    "encoder_b_lr_scale",
    "unfreeze_start_epoch",
    "unfreeze_warmup_epochs",
    "projection_hidden_dim",
    "projection_layers",
    "projection_dropout",
    "rank_loss_weight",
    "n_pairs",
    "hierarchy_threshold",
    "coverage_floor",
    "enable_annealing",
    "annealing_step",
]

# All metrics to extract
ALL_METRICS = [
    "coverage",
    "geo_loss_A",
    "geo_loss_B",
    "rad_loss_A",
    "rad_loss_B",
    "radial_corr_A",
    "radial_corr_B",
    "mean_radius_A",
    "mean_radius_B",
    "radius_min_A",
    "radius_max_A",
    "radius_range_A",
    "radius_v0",
    "radius_v9",
    "distance_corr_A",
    "distance_corr_B",
]


def get_checkpoint_date(ckpt_path: Path) -> datetime:
    """Extract date from checkpoint path or file modification time."""
    dir_name = ckpt_path.parent.name
    for part in dir_name.split("_"):
        if len(part) == 8 and part.isdigit():
            try:
                return datetime.strptime(part, "%Y%m%d")
            except ValueError:
                pass
    return datetime.fromtimestamp(ckpt_path.stat().st_mtime)


def compute_weight_statistics(state_dict: dict) -> dict:
    """Compute comprehensive statistics for model weights."""
    stats = {
        "total_params": 0,
        "total_layers": 0,
        "weight_means": [],
        "weight_stds": [],
        "weight_norms": [],
        "weight_sparsity": [],  # fraction of near-zero weights
        "bias_means": [],
        "bias_stds": [],
        # Per-component statistics
        "encoder_a_norm": 0.0,
        "encoder_b_norm": 0.0,
        "projection_norm": 0.0,
        "decoder_norm": 0.0,
    }

    encoder_a_params = []
    encoder_b_params = []
    projection_params = []
    decoder_params = []

    for name, param in state_dict.items():
        if not isinstance(param, torch.Tensor):
            continue

        stats["total_params"] += param.numel()

        if "weight" in name:
            stats["total_layers"] += 1
            stats["weight_means"].append(float(param.mean()))
            stats["weight_stds"].append(float(param.std()))
            stats["weight_norms"].append(float(param.norm()))
            # Sparsity: fraction of weights with |w| < 0.01
            stats["weight_sparsity"].append(float((param.abs() < 0.01).float().mean()))

        elif "bias" in name:
            stats["bias_means"].append(float(param.mean()))
            stats["bias_stds"].append(float(param.std()))

        # Categorize by component
        if "encoder_a" in name.lower() or "encoder.0" in name or "codon_embed" in name:
            encoder_a_params.append(param.flatten())
        elif "encoder_b" in name.lower() or "encoder.1" in name:
            encoder_b_params.append(param.flatten())
        elif "projection" in name.lower() or "hyper" in name.lower():
            projection_params.append(param.flatten())
        elif "decoder" in name.lower():
            decoder_params.append(param.flatten())

    # Compute component norms
    if encoder_a_params:
        stats["encoder_a_norm"] = float(torch.cat(encoder_a_params).norm())
    if encoder_b_params:
        stats["encoder_b_norm"] = float(torch.cat(encoder_b_params).norm())
    if projection_params:
        stats["projection_norm"] = float(torch.cat(projection_params).norm())
    if decoder_params:
        stats["decoder_norm"] = float(torch.cat(decoder_params).norm())

    # Aggregate statistics
    stats["mean_weight_norm"] = np.mean(stats["weight_norms"]) if stats["weight_norms"] else 0
    stats["std_weight_norm"] = np.std(stats["weight_norms"]) if stats["weight_norms"] else 0
    stats["mean_sparsity"] = np.mean(stats["weight_sparsity"]) if stats["weight_sparsity"] else 0
    stats["mean_weight_std"] = np.mean(stats["weight_stds"]) if stats["weight_stds"] else 0

    return stats


def compute_optimizer_statistics(optimizer_state: dict) -> dict:
    """Extract statistics from optimizer state (momentum, etc.)."""
    stats = {
        "has_momentum": False,
        "mean_momentum_norm": 0.0,
        "mean_exp_avg_sq_norm": 0.0,  # Adam's second moment
        "n_params_with_state": 0,
    }

    if not optimizer_state or "state" not in optimizer_state:
        return stats

    state = optimizer_state["state"]
    momentum_norms = []
    exp_avg_sq_norms = []

    for _param_id, param_state in state.items():
        stats["n_params_with_state"] += 1

        # Adam optimizer has exp_avg (momentum) and exp_avg_sq
        if "exp_avg" in param_state:
            stats["has_momentum"] = True
            momentum_norms.append(float(param_state["exp_avg"].norm()))

        if "exp_avg_sq" in param_state:
            exp_avg_sq_norms.append(float(param_state["exp_avg_sq"].norm()))

    if momentum_norms:
        stats["mean_momentum_norm"] = np.mean(momentum_norms)
    if exp_avg_sq_norms:
        stats["mean_exp_avg_sq_norm"] = np.mean(exp_avg_sq_norms)

    return stats


def extract_hyperparameters(config: dict) -> dict:
    """Extract key hyperparameters from config."""
    hyperparams = {}

    for key in KEY_HYPERPARAMS:
        value = config.get(key)
        if value is not None:
            # Convert booleans to float for neural network
            if isinstance(value, bool):
                hyperparams[key] = 1.0 if value else 0.0
            elif isinstance(value, (int, float)):
                hyperparams[key] = float(value)
            else:
                hyperparams[key] = 0.0  # Skip non-numeric
        else:
            hyperparams[key] = 0.0  # Default for missing

    return hyperparams


def extract_all_metrics(metrics: dict) -> dict:
    """Extract all metrics from checkpoint."""
    all_metrics = {}

    for key in ALL_METRICS:
        value = metrics.get(key, 0.0)
        all_metrics[key] = float(value) if value is not None else 0.0

    return all_metrics


def collect_enhanced_checkpoint_data(checkpoint_dir: Path) -> tuple:
    """Collect comprehensive checkpoint data."""
    dataset = []
    errors = []

    print(f"Scanning {checkpoint_dir}...")

    for run_dir in sorted(checkpoint_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        for ckpt_path in run_dir.glob("*.pt"):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

                # Extract state dict
                state_dict = (
                    ckpt.get("model_state_dict")
                    or ckpt.get("model_state")
                    or ckpt.get("state_dict")
                    or ckpt.get("model")
                    or {}
                )
                if not state_dict:
                    continue

                # Extract key weights for VAE input
                weights = extract_key_weights(state_dict)
                if not weights:
                    continue

                flat_weights = torch.cat([w.flatten() for w in weights])

                # Get all components
                epoch = ckpt.get("epoch", -1)
                date = get_checkpoint_date(ckpt_path)

                # Extract comprehensive features
                metrics = extract_all_metrics(ckpt.get("metrics", {}))
                config = ckpt.get("config", {})
                hyperparams = extract_hyperparameters(config) if config else {}
                weight_stats = compute_weight_statistics(state_dict)
                optimizer_stats = compute_optimizer_statistics(ckpt.get("optimizer_state", {}))

                # Compute training progress features
                total_epochs = config.get("epochs", 150) if config else 150
                progress = epoch / total_epochs if total_epochs > 0 else 0.0

                dataset.append(
                    {
                        "path": str(ckpt_path),
                        "run_name": run_dir.name,
                        "epoch": epoch,
                        "date": date.isoformat(),
                        "date_obj": date,
                        # Weights for VAE encoder
                        "weight_dim": flat_weights.shape[0],
                        "weights_path": str(ckpt_path),  # Load on demand
                        # All metrics (16+)
                        "metrics": metrics,
                        # Training hyperparameters
                        "hyperparams": hyperparams,
                        # Weight statistics
                        "weight_stats": {
                            "total_params": weight_stats["total_params"],
                            "total_layers": weight_stats["total_layers"],
                            "mean_weight_norm": weight_stats["mean_weight_norm"],
                            "std_weight_norm": weight_stats["std_weight_norm"],
                            "mean_sparsity": weight_stats["mean_sparsity"],
                            "mean_weight_std": weight_stats["mean_weight_std"],
                            "encoder_a_norm": weight_stats["encoder_a_norm"],
                            "encoder_b_norm": weight_stats["encoder_b_norm"],
                            "projection_norm": weight_stats["projection_norm"],
                        },
                        # Optimizer statistics
                        "optimizer_stats": optimizer_stats,
                        # Training progress
                        "progress": progress,
                        "total_epochs": total_epochs,
                    }
                )

            except Exception as e:
                errors.append((str(ckpt_path), str(e)))
                continue

    print(f"Collected {len(dataset)} checkpoints, {len(errors)} errors")
    return dataset, errors


def create_feature_vector(item: dict) -> list:
    """Create a flat feature vector from all extracted data."""
    features = []

    # Epoch and progress (2 features)
    features.append(item["epoch"])
    features.append(item["progress"])

    # All metrics (16 features)
    for key in ALL_METRICS:
        features.append(item["metrics"].get(key, 0.0))

    # Hyperparameters (25 features)
    for key in KEY_HYPERPARAMS:
        features.append(item["hyperparams"].get(key, 0.0))

    # Weight statistics (9 features)
    ws = item["weight_stats"]
    features.extend(
        [
            ws["total_params"] / 1e6,  # Normalize to millions
            ws["total_layers"],
            ws["mean_weight_norm"],
            ws["std_weight_norm"],
            ws["mean_sparsity"],
            ws["mean_weight_std"],
            ws["encoder_a_norm"],
            ws["encoder_b_norm"],
            ws["projection_norm"],
        ]
    )

    # Optimizer statistics (3 features)
    os = item["optimizer_stats"]
    features.extend(
        [
            1.0 if os["has_momentum"] else 0.0,
            os["mean_momentum_norm"],
            os["mean_exp_avg_sq_norm"],
        ]
    )

    return features


def analyze_enhanced_dataset(dataset: list) -> dict:
    """Analyze the enhanced dataset."""
    if not dataset:
        return {"error": "No data"}

    # Count feature dimensions
    sample_features = create_feature_vector(dataset[0])

    # Gather statistics
    all_coverages = [d["metrics"]["coverage"] for d in dataset]
    all_dist_corrs = [d["metrics"]["distance_corr_A"] for d in dataset]

    # Count unique configs
    unique_lrs = set(d["hyperparams"].get("lr", 0) for d in dataset)
    unique_architectures = set(
        (d["hyperparams"].get("projection_hidden_dim", 0), d["hyperparams"].get("projection_layers", 0))
        for d in dataset
    )

    return {
        "n_checkpoints": len(dataset),
        "n_runs": len(set(d["run_name"] for d in dataset)),
        "feature_dim": len(sample_features),
        "weight_dims": list(set(d["weight_dim"] for d in dataset)),
        "metrics_range": {
            "coverage": {"min": min(all_coverages), "max": max(all_coverages)},
            "distance_corr_A": {"min": min(all_dist_corrs), "max": max(all_dist_corrs)},
        },
        "hyperparams_diversity": {
            "unique_lrs": len(unique_lrs),
            "unique_architectures": len(unique_architectures),
        },
        "feature_breakdown": {
            "epoch_progress": 2,
            "metrics": len(ALL_METRICS),
            "hyperparams": len(KEY_HYPERPARAMS),
            "weight_stats": 9,
            "optimizer_stats": 3,
            "total": len(sample_features),
        },
    }


def split_by_date(dataset: list, cutoff_date: str) -> tuple:
    """Split dataset temporally."""
    cutoff = datetime.fromisoformat(cutoff_date)
    train_data = [d for d in dataset if d["date_obj"] < cutoff]
    val_data = [d for d in dataset if d["date_obj"] >= cutoff]
    return train_data, val_data


def main():
    parser = argparse.ArgumentParser(description="Enhanced checkpoint collection")
    parser.add_argument("--checkpoint_dir", type=str, default=str(CHECKPOINTS_DIR))
    parser.add_argument("--cutoff", type=str, default="2025-12-26")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR / "epsilon_vae_data_enhanced"))
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    print("=" * 70)
    print("ENHANCED EPSILON-VAE: Checkpoint Collection")
    print("=" * 70)

    dataset, errors = collect_enhanced_checkpoint_data(checkpoint_dir)

    if not dataset:
        print("ERROR: No checkpoints collected!")
        return

    # Analyze
    print("\n" + "=" * 70)
    print("ENHANCED DATASET ANALYSIS")
    print("=" * 70)

    analysis = analyze_enhanced_dataset(dataset)
    print(json.dumps(analysis, indent=2, default=str))

    # Split
    print("\n" + "=" * 70)
    print(f"TEMPORAL SPLIT (cutoff: {args.cutoff})")
    print("=" * 70)

    train_data, val_data = split_by_date(dataset, args.cutoff)

    print(f"Training set: {len(train_data)} checkpoints")
    print(f"Validation set: {len(val_data)} checkpoints")

    # Create feature tensors for quick loading
    print("\n" + "=" * 70)
    print("CREATING FEATURE TENSORS")
    print("=" * 70)

    # Pre-compute feature vectors
    for d in dataset:
        d["features"] = create_feature_vector(d)
        # Remove non-serializable date_obj
        del d["date_obj"]

    for d in train_data:
        if "date_obj" in d:
            del d["date_obj"]
    for d in val_data:
        if "date_obj" in d:
            del d["date_obj"]

    # Save
    print("\n" + "=" * 70)
    print("SAVING ENHANCED DATASET")
    print("=" * 70)

    with open(output_dir / "full_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    with open(output_dir / "train_dataset.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open(output_dir / "val_dataset.json", "w") as f:
        json.dump(val_data, f, indent=2)

    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    # Save feature metadata
    feature_meta = {
        "feature_names": (
            ["epoch", "progress"]
            + ALL_METRICS
            + KEY_HYPERPARAMS
            + [
                "total_params_M",
                "total_layers",
                "mean_weight_norm",
                "std_weight_norm",
                "mean_sparsity",
                "mean_weight_std",
                "encoder_a_norm",
                "encoder_b_norm",
                "projection_norm",
            ]
            + ["has_momentum", "mean_momentum_norm", "mean_exp_avg_sq_norm"]
        ),
        "feature_dim": analysis["feature_dim"],
        "target_metrics": ["coverage", "distance_corr_A", "radial_corr_A"],
    }

    with open(output_dir / "feature_meta.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    print(f"Saved to {output_dir}")
    print(f"  - full_dataset.json ({len(dataset)} checkpoints)")
    print(f"  - train_dataset.json ({len(train_data)} checkpoints)")
    print(f"  - val_dataset.json ({len(val_data)} checkpoints)")
    print("  - analysis.json")
    print("  - feature_meta.json")
    print(f"\nFeature dimension: {analysis['feature_dim']}")

    if errors:
        print(f"\n{len(errors)} errors (see errors.json)")
        with open(output_dir / "errors.json", "w") as f:
            json.dump(errors, f, indent=2)


if __name__ == "__main__":
    main()
