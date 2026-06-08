# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Collect checkpoint dataset for Epsilon-VAE training.

This script:
1. Scans all checkpoint directories
2. Extracts key weights and metrics from each checkpoint
3. Splits data temporally: older for training, recent for validation
4. Saves the dataset for Epsilon-VAE training

Usage:
    python src/scripts/epsilon_vae/collect_checkpoints.py
    python src/scripts/epsilon_vae/collect_checkpoints.py --cutoff 2025-12-26
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR, OUTPUT_DIR
from src.models.epsilon_vae import extract_key_weights


def get_checkpoint_date(ckpt_path: Path) -> datetime:
    """Extract date from checkpoint path or file modification time."""
    # Try to parse from directory name (e.g., ternary_option_c_dual_20251224_033546)
    dir_name = ckpt_path.parent.name

    # Look for date pattern YYYYMMDD
    for part in dir_name.split("_"):
        if len(part) == 8 and part.isdigit():
            try:
                return datetime.strptime(part, "%Y%m%d")
            except ValueError:
                pass

    # Fallback to file modification time
    return datetime.fromtimestamp(ckpt_path.stat().st_mtime)


def collect_checkpoint_data(checkpoint_dir: Path) -> list[dict]:
    """Collect all checkpoint data with weights and metrics."""
    dataset = []
    errors = []

    print(f"Scanning {checkpoint_dir}...")

    for run_dir in sorted(checkpoint_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        for ckpt_path in run_dir.glob("*.pt"):
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

                # Extract state dict (handle different checkpoint formats)
                state_dict = (
                    ckpt.get("model_state_dict")
                    or ckpt.get("model_state")
                    or ckpt.get("state_dict")
                    or ckpt.get("model")
                    or {}
                )
                if not state_dict:
                    continue

                # Handle nested state dict
                if isinstance(state_dict, dict) and not any(
                    k.endswith(".weight") or k.endswith(".bias") for k in list(state_dict.keys())[:10]
                ):
                    # Might be a model object, try to get state_dict
                    if hasattr(state_dict, "state_dict"):
                        state_dict = state_dict.state_dict()

                # Extract key weights
                weights = extract_key_weights(state_dict)
                if not weights:
                    continue

                # Extract metrics
                metrics = ckpt.get("metrics", {})

                # Get epoch
                epoch = ckpt.get("epoch", -1)

                # Get date
                date = get_checkpoint_date(ckpt_path)

                # Compute weight stats for quick reference
                weight_sizes = [w.numel() for w in weights]
                total_params = sum(weight_sizes)

                dataset.append(
                    {
                        "path": str(ckpt_path),
                        "run_name": run_dir.name,
                        "epoch": epoch,
                        "date": date.isoformat(),
                        "date_obj": date,
                        "coverage": metrics.get("coverage", 0.0),
                        "dist_corr": metrics.get("distance_corr_A", 0.0),
                        "rad_hier": metrics.get("radial_corr_A", 0.0),
                        "weight_sizes": weight_sizes,
                        "total_params": total_params,
                        "n_weight_blocks": len(weights),
                    }
                )

            except Exception as e:
                errors.append((str(ckpt_path), str(e)))
                continue

    print(f"Collected {len(dataset)} checkpoints, {len(errors)} errors")
    return dataset, errors


def analyze_dataset(dataset: list[dict]) -> dict:
    """Analyze the collected dataset."""
    if not dataset:
        return {"error": "No data"}

    dates = [d["date_obj"] for d in dataset]
    coverages = [d["coverage"] for d in dataset if d["coverage"] > 0]
    dist_corrs = [d["dist_corr"] for d in dataset if d["dist_corr"] != 0]
    rad_hiers = [d["rad_hier"] for d in dataset if d["rad_hier"] != 0]

    runs = set(d["run_name"] for d in dataset)

    return {
        "n_checkpoints": len(dataset),
        "n_runs": len(runs),
        "date_range": {
            "earliest": min(dates).isoformat(),
            "latest": max(dates).isoformat(),
        },
        "metrics": {
            "coverage": {
                "min": min(coverages) if coverages else 0,
                "max": max(coverages) if coverages else 0,
                "n_with_data": len(coverages),
            },
            "dist_corr": {
                "min": min(dist_corrs) if dist_corrs else 0,
                "max": max(dist_corrs) if dist_corrs else 0,
                "n_with_data": len(dist_corrs),
            },
            "rad_hier": {
                "min": min(rad_hiers) if rad_hiers else 0,
                "max": max(rad_hiers) if rad_hiers else 0,
                "n_with_data": len(rad_hiers),
            },
        },
        "weight_stats": {
            "typical_n_blocks": dataset[0]["n_weight_blocks"] if dataset else 0,
            "typical_total_params": dataset[0]["total_params"] if dataset else 0,
        },
    }


def split_by_date(dataset: list[dict], cutoff_date: str) -> tuple[list, list]:
    """Split dataset temporally."""
    cutoff = datetime.fromisoformat(cutoff_date)

    train_data = [d for d in dataset if d["date_obj"] < cutoff]
    val_data = [d for d in dataset if d["date_obj"] >= cutoff]

    return train_data, val_data


def main():
    parser = argparse.ArgumentParser(description="Collect checkpoint dataset")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(CHECKPOINTS_DIR),
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default="2025-12-26",
        help="Date cutoff for train/val split (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR / "epsilon_vae_data"),
        help="Output directory for dataset",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    print("=" * 60)
    print("EPSILON-VAE: Checkpoint Collection")
    print("=" * 60)

    dataset, errors = collect_checkpoint_data(checkpoint_dir)

    # Analyze
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)

    analysis = analyze_dataset(dataset)
    print(json.dumps(analysis, indent=2, default=str))

    # Split
    print("\n" + "=" * 60)
    print(f"TEMPORAL SPLIT (cutoff: {args.cutoff})")
    print("=" * 60)

    train_data, val_data = split_by_date(dataset, args.cutoff)

    print(f"Training set: {len(train_data)} checkpoints")
    print(f"Validation set: {len(val_data)} checkpoints")

    if train_data:
        train_runs = set(d["run_name"] for d in train_data)
        print(f"  Training runs: {len(train_runs)}")

    if val_data:
        val_runs = set(d["run_name"] for d in val_data)
        print(f"  Validation runs: {len(val_runs)}")
        for d in val_data[:5]:
            print(f"    - {d['run_name']}/{Path(d['path']).name}: cov={d['coverage']:.3f}, dist={d['dist_corr']:.3f}")

    # Save metadata (not weights - too large)
    print("\n" + "=" * 60)
    print("SAVING DATASET METADATA")
    print("=" * 60)

    # Remove non-serializable date_obj
    for d in dataset:
        del d["date_obj"]
    for d in train_data:
        if "date_obj" in d:
            del d["date_obj"]
    for d in val_data:
        if "date_obj" in d:
            del d["date_obj"]

    with open(output_dir / "full_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    with open(output_dir / "train_dataset.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open(output_dir / "val_dataset.json", "w") as f:
        json.dump(val_data, f, indent=2)

    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"Saved to {output_dir}")
    print(f"  - full_dataset.json ({len(dataset)} checkpoints)")
    print(f"  - train_dataset.json ({len(train_data)} checkpoints)")
    print(f"  - val_dataset.json ({len(val_data)} checkpoints)")
    print("  - analysis.json")

    if errors:
        print(f"\n{len(errors)} errors (see errors.json)")
        with open(output_dir / "errors.json", "w") as f:
            json.dump(errors, f, indent=2)


if __name__ == "__main__":
    main()
