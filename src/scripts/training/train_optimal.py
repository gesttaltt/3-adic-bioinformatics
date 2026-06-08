#!/usr/bin/env python
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Train Optimal VAE - Uses best configuration from ablation study.

This script trains the VAE with the optimal configuration discovered
through systematic ablation testing:

- Hyperbolic prior: +4.3% correlation
- P-adic ranking loss: +6.9% correlation
- Combined: +12% correlation (synergistic!)

Usage:
    # Train with default optimal settings (100 epochs)
    python src/scripts/training/train_optimal.py

    # Quick test (20 epochs)
    python src/scripts/training/train_optimal.py --epochs 20

    # Full training with checkpoints
    python src/scripts/training/train_optimal.py --epochs 200 --save-checkpoints

    # Evaluate only (load existing model)
    python src/scripts/training/train_optimal.py --evaluate --checkpoint outputs/optimal/best.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.generation import generate_all_ternary_operations
from src.losses.dual_vae_loss import KLDivergenceLoss, ReconstructionLoss
from src.models.optimal_vae import OptimalVAE, OptimalVAEConfig
from src.training import EpochMetrics, GrokDetector, TernaryDataset


def set_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_dataloaders(batch_size: int, val_split: float = 0.1, seed: int = 42):
    """Create train and validation dataloaders."""
    # Generate all ternary operations
    all_operations = generate_all_ternary_operations()
    operations_tensor = torch.tensor(all_operations, dtype=torch.float32)
    indices = torch.arange(len(all_operations))

    dataset = TernaryDataset(operations_tensor, indices)

    # Split
    n_total = len(dataset)
    n_val = int(val_split * n_total)
    n_train = n_total - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, n_train, n_val


def train_epoch(model, train_loader, optimizer, recon_loss_fn, kl_loss_fn, device, beta=0.01):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_padic = 0.0
    n_batches = 0

    for batch in train_loader:
        x = batch["operation"].to(device)
        batch_indices = batch["index"].to(device)

        optimizer.zero_grad()

        loss, loss_dict = model.compute_loss(x, batch_indices, recon_loss_fn, kl_loss_fn, beta)

        loss.backward()
        optimizer.step()

        total_loss += loss_dict["total"]
        total_recon += loss_dict["recon"]
        total_kl += loss_dict["kl"]
        if "padic" in loss_dict:
            total_padic += loss_dict["padic"]
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon": total_recon / n_batches,
        "kl": total_kl / n_batches,
        "padic": total_padic / n_batches if total_padic > 0 else 0.0,
    }


def validate(model, val_loader, recon_loss_fn, kl_loss_fn, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_z = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["operation"].to(device)
            batch_indices = batch["index"].to(device)

            loss, _ = model.compute_loss(x, batch_indices, recon_loss_fn, kl_loss_fn, beta=1.0)

            outputs = model(x)
            z = outputs["z"]
            logits = outputs["logits"]

            total_loss += loss.item()
            all_z.append(z.cpu())

            # Accuracy
            pred = torch.argmax(logits, dim=-1)  # (batch, 9)
            target = (x + 1).long()  # Convert {-1,0,1} to {0,1,2}
            correct += (pred == target).float().mean().item() * x.size(0)
            total += x.size(0)

    all_z = torch.cat(all_z, dim=0)
    accuracy = correct / total

    return {
        "loss": total_loss / len(val_loader),
        "accuracy": accuracy,
        "z_mean_norm": all_z.norm(dim=-1).mean().item(),
        "z_std": all_z.std().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train Optimal VAE")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/optimal"))
    parser.add_argument("--save-checkpoints", action="store_true", help="Save checkpoints")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint to load")
    parser.add_argument("--no-hyperbolic", action="store_true", help="Disable hyperbolic")
    parser.add_argument("--no-padic", action="store_true", help="Disable p-adic ranking")

    args = parser.parse_args()

    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 70)
    print("OPTIMAL VAE TRAINING")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hyperbolic: {not args.no_hyperbolic}")
    print(f"P-adic ranking: {not args.no_padic}")
    print("=" * 70)

    # Set seeds
    set_seeds(args.seed)

    # Create config
    config = OptimalVAEConfig(
        enable_hyperbolic=not args.no_hyperbolic,
        enable_padic_ranking=not args.no_padic,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    # Create model
    model = OptimalVAE(config).to(device)
    params = model.count_parameters()
    print(f"\nModel parameters: {params['trainable']:,}")

    # Load checkpoint if specified
    if args.checkpoint and args.checkpoint.exists():
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    # Create dataloaders
    train_loader, val_loader, n_train, n_val = create_dataloaders(args.batch_size, seed=args.seed)
    print(f"Training samples: {n_train:,}")
    print(f"Validation samples: {n_val:,}")

    # Loss functions
    recon_loss_fn = ReconstructionLoss()
    kl_loss_fn = KLDivergenceLoss()

    # Evaluate only
    if args.evaluate:
        val_metrics = validate(model, val_loader, recon_loss_fn, kl_loss_fn, device)
        print("\nValidation Results:")
        print(f"  Loss: {val_metrics['loss']:.4f}")
        print(f"  Accuracy: {val_metrics['accuracy']:.1%}")
        return

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\nStarting training...")
    detector = GrokDetector()
    best_val_loss = float("inf")
    history = []

    start_time = time.time()

    for epoch in range(args.epochs):
        # Train (use configured beta)
        train_metrics = train_epoch(model, train_loader, optimizer, recon_loss_fn, kl_loss_fn, device, beta=config.beta)

        # Validate
        val_metrics = validate(model, val_loader, recon_loss_fn, kl_loss_fn, device)

        # Record
        epoch_data = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "accuracy": val_metrics["accuracy"],
            "padic_loss": train_metrics["padic"],
        }
        history.append(epoch_data)

        # Update grokking detector
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            val_loss=val_metrics["loss"],
            correlation=val_metrics["accuracy"],  # Use accuracy as proxy
            coverage=val_metrics["accuracy"],
        )
        detector.update(metrics)

        # Progress
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(
                f"Epoch {epoch:3d}: "
                f"train={train_metrics['loss']:.4f}, "
                f"val={val_metrics['loss']:.4f}, "
                f"acc={val_metrics['accuracy']:.1%}, "
                f"padic={train_metrics['padic']:.4f}"
            )

        # Save best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            if args.save_checkpoints:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                        "config": config.__dict__,
                    },
                    args.output_dir / "best.pt",
                )

    elapsed = time.time() - start_time

    # Final results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {elapsed:.1f}s ({elapsed / args.epochs:.2f}s/epoch)")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Final accuracy: {val_metrics['accuracy']:.1%}")

    # Save history
    history_path = args.output_dir / f"history_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory saved to: {history_path}")

    # Save final model
    if args.save_checkpoints:
        torch.save(
            {
                "epoch": args.epochs - 1,
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
            },
            args.output_dir / "final.pt",
        )
        print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
