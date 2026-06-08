# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Extract embeddings from checkpoints for hybrid Epsilon-VAE training.

This script:
1. Loads each checkpoint
2. Runs inference on fixed anchor ternary operations
3. Extracts hyperbolic embeddings z_A_hyp (the actual embedding space)
4. Saves embeddings + metrics for each checkpoint

The key insight: Instead of training Epsilon-VAE on summary metrics (lossy),
we train it on the actual embedding space (lossless). This captures the full
geometric structure of what each checkpoint produces.

Usage:
    python src/scripts/epsilon_vae/extract_embeddings.py
    python src/scripts/epsilon_vae/extract_embeddings.py --n_anchors 256 --output_dir outputs/epsilon_vae_hybrid
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
from src.data.generation import generate_all_ternary_operations
from src.models.epsilon_vae import extract_key_weights


def select_anchor_operations(n_anchors: int = 256, seed: int = 42) -> torch.Tensor:
    """Select representative anchor ternary operations.

    We use stratified selection to ensure good coverage of the operation space.

    Args:
        n_anchors: Number of anchor operations to select
        seed: Random seed for reproducibility

    Returns:
        Tensor of shape (n_anchors, 9)
    """
    np.random.seed(seed)

    all_ops = generate_all_ternary_operations()  # (19683, 9)

    # Stratified selection based on operation "complexity"
    # Complexity = number of non-zero elements
    complexities = np.abs(all_ops).sum(axis=1)

    # Group by complexity and sample from each group
    anchors = []
    unique_complexities = np.unique(complexities)
    samples_per_group = n_anchors // len(unique_complexities)

    for complexity in unique_complexities:
        mask = complexities == complexity
        group_indices = np.where(mask)[0]
        n_samples = min(samples_per_group, len(group_indices))
        selected = np.random.choice(group_indices, size=n_samples, replace=False)
        anchors.extend(selected)

    # Fill remaining quota randomly
    remaining = n_anchors - len(anchors)
    if remaining > 0:
        available = set(range(len(all_ops))) - set(anchors)
        extra = np.random.choice(list(available), size=remaining, replace=False)
        anchors.extend(extra)

    anchors = np.array(anchors[:n_anchors])
    return torch.tensor(all_ops[anchors], dtype=torch.float32)


def load_model_from_checkpoint(checkpoint_path: Path, device: str = "cpu"):
    """Load TernaryVAE model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to

    Returns:
        Loaded model ready for inference, or None if loading fails
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Get model state dict
        model_state = (
            ckpt.get("model_state_dict") or ckpt.get("model_state") or ckpt.get("state_dict") or ckpt.get("model") or {}
        )

        if not model_state:
            return None

        # Detect model type from state dict keys
        any("projection" in k for k in model_state)
        has_dual = any("proj_A" in k or "proj_B" in k for k in model_state)
        has_controller = any("controller" in k for k in model_state)

        # Import model class
        from src.models.ternary_vae import TernaryVAEV5_11

        # Infer latent_dim from encoder weights
        latent_dim = 16  # default
        for key, value in model_state.items():
            if "encoder_A.fc_mu.weight" in key:
                latent_dim = value.shape[0]
                break

        # Infer hidden_dim and other params from projection weights if available
        hidden_dim = 64  # default
        for key, value in model_state.items():
            if "projection" in key and "direction_net.0.weight" in key:
                hidden_dim = value.shape[0]
                break

        # Create model with matching architecture
        model = TernaryVAEV5_11(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            use_dual_projection=has_dual,
            use_controller=has_controller,
        )

        # Load state dict with strict=False to handle missing/extra keys
        model.load_state_dict(model_state, strict=False)
        model.to(device)
        model.eval()

        return model

    except Exception as e:
        print(f"  Error loading model: {e}")
        import traceback

        traceback.print_exc()
        return None


def extract_embeddings_from_checkpoint(
    checkpoint_path: Path,
    anchor_ops: torch.Tensor,
    device: str = "cpu",
) -> dict:
    """Extract embeddings and metrics from a single checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        anchor_ops: Anchor ternary operations (n_anchors, 9)
        device: Device to run on

    Returns:
        Dict with embeddings and metrics, or None if extraction fails
    """
    model = load_model_from_checkpoint(checkpoint_path, device)
    if model is None:
        return None

    try:
        # Load checkpoint for metrics
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        metrics = ckpt.get("metrics", {})

        # Extract weights for Epsilon-VAE input
        model_state = (
            ckpt.get("model_state_dict") or ckpt.get("model_state") or ckpt.get("state_dict") or ckpt.get("model") or {}
        )
        weights = extract_key_weights(model_state)
        if not weights:
            return None

        # Ensure weights are on CPU
        weights = [w.cpu() if hasattr(w, "cpu") else w for w in weights]

        # Run inference on anchors
        anchor_ops_device = anchor_ops.to(device)

        with torch.no_grad():
            outputs = model(anchor_ops_device, compute_control=False)

            # Extract hyperbolic embeddings (the key output!)
            z_A_hyp = outputs["z_A_hyp"].cpu()  # (n_anchors, latent_dim)
            z_B_hyp = outputs["z_B_hyp"].cpu()  # (n_anchors, latent_dim)

            # Also get Euclidean latents for comparison
            z_A_euc = outputs["z_A_euc"].cpu()
            z_B_euc = outputs["z_B_euc"].cpu()

        return {
            "z_A_hyp": z_A_hyp.numpy(),
            "z_B_hyp": z_B_hyp.numpy(),
            "z_A_euc": z_A_euc.numpy(),
            "z_B_euc": z_B_euc.numpy(),
            "weights": [w.numpy() for w in weights],
            "weight_shapes": [list(w.shape) for w in weights],
            "metrics": {
                "coverage": float(metrics.get("coverage", 0.0)),
                "distance_corr_A": float(metrics.get("distance_corr_A", 0.0)),
                "radial_corr_A": float(metrics.get("radial_corr_A", 0.0)),
            },
            "epoch": ckpt.get("epoch", -1),
        }

    except Exception as e:
        print(f"  Error extracting embeddings: {e}")
        return None


def find_all_checkpoints(checkpoint_dir: Path) -> list:
    """Find all checkpoint files in directory tree.

    Args:
        checkpoint_dir: Root directory to search

    Returns:
        List of checkpoint paths sorted by modification time
    """
    checkpoints = []

    for run_dir in checkpoint_dir.iterdir():
        if not run_dir.is_dir():
            continue

        for ckpt_path in run_dir.glob("*.pt"):
            mtime = ckpt_path.stat().st_mtime
            checkpoints.append((ckpt_path, mtime, run_dir.name))

    # Sort by modification time
    checkpoints.sort(key=lambda x: x[1])

    return checkpoints


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from checkpoints")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(CHECKPOINTS_DIR),
        help="Directory containing checkpoint subdirectories",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR / "epsilon_vae_hybrid"),
        help="Output directory for extracted data",
    )
    parser.add_argument("--n_anchors", type=int, default=256, help="Number of anchor operations")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on")
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of data for validation (temporal split)")
    parser.add_argument("--max_checkpoints", type=int, default=2000, help="Maximum number of checkpoints to process")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Generate anchor operations
    print(f"\n{'=' * 70}")
    print("GENERATING ANCHOR OPERATIONS")
    print(f"{'=' * 70}")
    anchor_ops = select_anchor_operations(args.n_anchors)
    print(f"Selected {len(anchor_ops)} anchor operations")
    print(f"Anchor shapes: {anchor_ops.shape}")

    # Save anchor operations for reproducibility
    np.save(output_dir / "anchor_operations.npy", anchor_ops.numpy())

    # Find all checkpoints
    print(f"\n{'=' * 70}")
    print("FINDING CHECKPOINTS")
    print(f"{'=' * 70}")
    checkpoints = find_all_checkpoints(checkpoint_dir)
    print(f"Found {len(checkpoints)} checkpoints")

    if len(checkpoints) > args.max_checkpoints:
        print(f"Limiting to {args.max_checkpoints} checkpoints")
        checkpoints = checkpoints[: args.max_checkpoints]

    # Extract embeddings from each checkpoint
    print(f"\n{'=' * 70}")
    print("EXTRACTING EMBEDDINGS")
    print(f"{'=' * 70}")

    all_data = []
    failed = 0

    for i, (ckpt_path, mtime, run_name) in enumerate(checkpoints):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"Processing {i + 1}/{len(checkpoints)}: {run_name}/{ckpt_path.name}")

        data = extract_embeddings_from_checkpoint(ckpt_path, anchor_ops, device)

        if data is not None:
            data["path"] = str(ckpt_path)
            data["run_name"] = run_name
            data["mtime"] = mtime
            all_data.append(data)
        else:
            failed += 1

    print(f"\nSuccessfully extracted: {len(all_data)}")
    print(f"Failed: {failed}")

    if len(all_data) == 0:
        print("ERROR: No data extracted!")
        return

    # Temporal split for train/val
    print(f"\n{'=' * 70}")
    print("CREATING TRAIN/VAL SPLIT (TEMPORAL)")
    print(f"{'=' * 70}")

    # Sort by modification time (already sorted, but ensure)
    all_data.sort(key=lambda x: x["mtime"])

    split_idx = int(len(all_data) * (1 - args.val_split))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # Analyze data
    print(f"\n{'=' * 70}")
    print("DATA STATISTICS")
    print(f"{'=' * 70}")

    # Weight dimensions
    weight_dims = [sum(np.prod(s) for s in d["weight_shapes"]) for d in all_data]
    print("\nWeight dimensions:")
    print(f"  Min: {min(weight_dims)}, Max: {max(weight_dims)}")

    # Filter to consistent weight dimensions
    from collections import Counter

    dim_counts = Counter(weight_dims)
    most_common_dim = dim_counts.most_common(1)[0][0]

    train_data = [d for d in train_data if sum(np.prod(s) for s in d["weight_shapes"]) == most_common_dim]
    val_data = [d for d in val_data if sum(np.prod(s) for s in d["weight_shapes"]) == most_common_dim]

    print(f"After filtering to weight_dim={most_common_dim}:")
    print(f"  Training: {len(train_data)}")
    print(f"  Validation: {len(val_data)}")

    # Embedding dimensions
    if train_data:
        embed_dim = train_data[0]["z_A_hyp"].shape[1]
        print(f"\nEmbedding dimension: {embed_dim}")
        print(f"Anchor count: {train_data[0]['z_A_hyp'].shape[0]}")

    # Metric distributions
    train_coverages = [d["metrics"]["coverage"] for d in train_data]
    val_coverages = [d["metrics"]["coverage"] for d in val_data]

    print("\nCoverage distribution:")
    print(f"  Train: mean={np.mean(train_coverages):.3f}, std={np.std(train_coverages):.3f}")
    print(f"  Val:   mean={np.mean(val_coverages):.3f}, std={np.std(val_coverages):.3f}")

    # Save processed data
    print(f"\n{'=' * 70}")
    print("SAVING DATA")
    print(f"{'=' * 70}")

    # Save as numpy arrays for efficient loading
    def save_dataset(data, prefix):
        """Save dataset as numpy arrays + JSON metadata."""
        # Stack embeddings
        z_A_hyp = np.stack([d["z_A_hyp"] for d in data])  # (N, n_anchors, embed_dim)
        z_B_hyp = np.stack([d["z_B_hyp"] for d in data])
        z_A_euc = np.stack([d["z_A_euc"] for d in data])
        z_B_euc = np.stack([d["z_B_euc"] for d in data])

        # Flatten weights
        weights = np.stack([np.concatenate([w.flatten() for w in d["weights"]]) for d in data])

        # Metrics
        metrics = np.array(
            [
                [
                    d["metrics"]["coverage"],
                    d["metrics"]["distance_corr_A"],
                    d["metrics"]["radial_corr_A"],
                ]
                for d in data
            ]
        )

        # Save arrays
        np.save(output_dir / f"{prefix}_z_A_hyp.npy", z_A_hyp)
        np.save(output_dir / f"{prefix}_z_B_hyp.npy", z_B_hyp)
        np.save(output_dir / f"{prefix}_z_A_euc.npy", z_A_euc)
        np.save(output_dir / f"{prefix}_z_B_euc.npy", z_B_euc)
        np.save(output_dir / f"{prefix}_weights.npy", weights)
        np.save(output_dir / f"{prefix}_metrics.npy", metrics)

        # Save metadata
        metadata = [
            {
                "path": d["path"],
                "run_name": d["run_name"],
                "epoch": d["epoch"],
                "metrics": d["metrics"],
            }
            for d in data
        ]

        with open(output_dir / f"{prefix}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved {prefix}:")
        print(f"  z_A_hyp: {z_A_hyp.shape}")
        print(f"  weights: {weights.shape}")
        print(f"  metrics: {metrics.shape}")

    save_dataset(train_data, "train")
    save_dataset(val_data, "val")

    # Save config
    config = {
        "n_anchors": args.n_anchors,
        "weight_dim": most_common_dim,
        "embed_dim": train_data[0]["z_A_hyp"].shape[1] if train_data else 16,
        "n_train": len(train_data),
        "n_val": len(val_data),
        "created": datetime.now().isoformat(),
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nData saved to {output_dir}")
    print(f"Config: {config}")


if __name__ == "__main__":
    main()
