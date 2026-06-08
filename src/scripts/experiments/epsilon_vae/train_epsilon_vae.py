# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Train Epsilon-VAE on checkpoint dataset.

This script:
1. Loads checkpoint metadata and weights
2. Trains Epsilon-VAE to predict metrics from weights
3. Validates on held-out (future) checkpoints
4. Saves trained model for Pareto exploration

Usage:
    python src/scripts/epsilon_vae/train_epsilon_vae.py
    python src/scripts/epsilon_vae/train_epsilon_vae.py --epochs 100 --latent_dim 32
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import OUTPUT_DIR
from src.models.epsilon_vae import (
    epsilon_vae_loss,
    extract_key_weights,
)
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat


class CheckpointDataset(Dataset):
    """Dataset of checkpoints with weights and metrics."""

    def __init__(self, metadata_path: Path, max_samples: int = None):
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        if max_samples:
            self.metadata = self.metadata[:max_samples]

        # Preload all weights
        print(f"Loading {len(self.metadata)} checkpoints...")
        self.samples = []
        for i, item in enumerate(self.metadata):
            try:
                ckpt = load_checkpoint_compat(item["path"], map_location="cpu")
                state_dict = get_model_state_dict(ckpt)
                weights = extract_key_weights(state_dict)

                if weights:
                    # Flatten weights for consistent handling
                    flat_weights = torch.cat([w.flatten() for w in weights])

                    metrics = torch.tensor(
                        [
                            item["coverage"],
                            item["dist_corr"],
                            item["rad_hier"],
                        ],
                        dtype=torch.float32,
                    )

                    self.samples.append(
                        {
                            "weights": flat_weights,
                            "weight_blocks": weights,
                            "metrics": metrics,
                            "path": item["path"],
                        }
                    )

                if (i + 1) % 100 == 0:
                    print(f"  Loaded {i + 1}/{len(self.metadata)}")

            except Exception as e:
                print(f"  Error loading {item['path']}: {e}")
                continue

        print(f"Successfully loaded {len(self.samples)} samples")

        # Filter to consistent weight dimensions
        if self.samples:
            self._filter_consistent_weights()

    def _filter_consistent_weights(self):
        """Filter samples to only those with the most common weight dimension."""
        # Count weight dimensions
        dim_counts = {}
        for s in self.samples:
            dim = s["weights"].shape[0]
            dim_counts[dim] = dim_counts.get(dim, 0) + 1

        # Find most common dimension
        most_common_dim = max(dim_counts, key=dim_counts.get)
        original_count = len(self.samples)

        # Filter to only matching dimensions
        self.samples = [s for s in self.samples if s["weights"].shape[0] == most_common_dim]

        if len(self.samples) < original_count:
            print(f"  Filtered to {len(self.samples)}/{original_count} samples with weight_dim={most_common_dim}")
            print(f"  Dimension distribution: {dim_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Custom collate for variable-sized weight blocks."""
    # For now, use flat weights which are fixed size
    weights = torch.stack([s["weights"] for s in batch])
    metrics = torch.stack([s["metrics"] for s in batch])
    return {"weights": weights, "metrics": metrics}


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_metric_loss = 0
    total_kl_loss = 0

    for batch in dataloader:
        weights = batch["weights"].to(device)
        metrics = batch["metrics"].to(device)

        optimizer.zero_grad()

        # Forward pass - use flat weights
        mu, logvar = model.encode(weights)
        z = model._reparameterize(mu, logvar)
        metrics_pred = model.predict_metrics(z)

        # Compute loss
        losses = epsilon_vae_loss(
            metrics_pred=metrics_pred,
            metrics_true=metrics,
            mu=mu,
            logvar=logvar,
            beta=0.01,  # Low KL weight - focus on metric prediction
        )

        losses["total"].backward()
        optimizer.step()

        total_loss += losses["total"].item()
        total_metric_loss += losses["metric_loss"].item()
        total_kl_loss += losses["kl_loss"].item()

    n_batches = len(dataloader)
    return {
        "loss": total_loss / n_batches,
        "metric_loss": total_metric_loss / n_batches,
        "kl_loss": total_kl_loss / n_batches,
    }


def validate(model, dataloader, device):
    """Validate on held-out data."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            weights = batch["weights"].to(device)
            metrics = batch["metrics"].to(device)

            mu, _ = model.encode(weights)
            metrics_pred = model.predict_metrics(mu)

            all_preds.append(metrics_pred.cpu())
            all_targets.append(metrics.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    # Compute per-metric errors
    errors = (preds - targets).abs()

    return {
        "coverage_mae": errors[:, 0].mean().item(),
        "dist_corr_mae": errors[:, 1].mean().item(),
        "rad_hier_mae": errors[:, 2].mean().item(),
        "total_mae": errors.mean().item(),
        "preds": preds,
        "targets": targets,
    }


class EpsilonVAEFlat(nn.Module):
    """Epsilon-VAE with flat weight encoder for training."""

    def __init__(self, weight_dim: int, latent_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.latent_dim = latent_dim

        # Flat encoder (for training efficiency)
        self.encoder_flat = nn.Sequential(
            nn.Linear(weight_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mu, logvar
        )

        # Metric predictor
        self.metric_predictor = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # [coverage, dist_corr, rad_hier]
        )

        # Create reparameterize as standalone
        self.encoder = type("Encoder", (), {"reparameterize": self._reparameterize})()

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, weights_flat):
        params = self.encoder_flat(weights_flat)
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar

    def predict_metrics(self, z):
        return self.metric_predictor(z)

    def forward(self, weights_flat):
        mu, logvar = self.encode(weights_flat)
        z = self._reparameterize(mu, logvar)
        metrics = self.predict_metrics(z)
        return mu, logvar, metrics


def main():
    parser = argparse.ArgumentParser(description="Train Epsilon-VAE")
    parser.add_argument("--data_dir", type=str, default=str(OUTPUT_DIR / "epsilon_vae_data"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default=str(OUTPUT_DIR / "epsilon_vae_models"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load datasets
    print("\n" + "=" * 60)
    print("LOADING DATASETS")
    print("=" * 60)

    train_dataset = CheckpointDataset(data_dir / "train_dataset.json")
    val_dataset = CheckpointDataset(data_dir / "val_dataset.json")

    if len(train_dataset) == 0:
        print("ERROR: No training samples loaded!")
        return

    # Get weight dimension from first sample
    weight_dim = train_dataset[0]["weights"].shape[0]
    print(f"\nWeight dimension: {weight_dim}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        if len(val_dataset) > 0
        else None
    )

    # Create model
    print("\n" + "=" * 60)
    print("CREATING EPSILON-VAE")
    print("=" * 60)

    model = EpsilonVAEFlat(
        weight_dim=weight_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    best_val_mae = float("inf")
    history = []

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        # Validate
        val_metrics = validate(model, val_loader, device) if val_loader else None

        # Log
        log = f"Epoch {epoch:3d} | Loss: {train_metrics['loss']:.4f} | Metric: {train_metrics['metric_loss']:.4f}"
        if val_metrics:
            log += f" | Val MAE: {val_metrics['total_mae']:.4f}"
            log += f" (cov: {val_metrics['coverage_mae']:.3f}, dist: {val_metrics['dist_corr_mae']:.3f}, rad: {val_metrics['rad_hier_mae']:.3f})"

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(log)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_metric_loss": train_metrics["metric_loss"],
                "val_mae": val_metrics["total_mae"] if val_metrics else None,
            }
        )

        # Save best
        if val_metrics and val_metrics["total_mae"] < best_val_mae:
            best_val_mae = val_metrics["total_mae"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mae": best_val_mae,
                    "config": vars(args),
                },
                save_dir / "best.pt",
            )

    # Save final
    torch.save(
        {
            "epoch": args.epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": vars(args),
        },
        save_dir / "final.pt",
    )

    # Final validation report
    if val_loader:
        print("\n" + "=" * 60)
        print("FINAL VALIDATION (on unseen future checkpoints)")
        print("=" * 60)

        val_metrics = validate(model, val_loader, device)
        print(f"Total MAE: {val_metrics['total_mae']:.4f}")
        print(f"  Coverage MAE: {val_metrics['coverage_mae']:.4f}")
        print(f"  Dist Corr MAE: {val_metrics['dist_corr_mae']:.4f}")
        print(f"  Rad Hier MAE: {val_metrics['rad_hier_mae']:.4f}")

        # Show predictions vs targets
        print("\nPredictions vs Targets:")
        preds = val_metrics["preds"]
        targets = val_metrics["targets"]
        for i in range(min(10, len(preds))):
            print(f"  [{i}] Pred: cov={preds[i, 0]:.3f}, dist={preds[i, 1]:.3f}, rad={preds[i, 2]:.3f}")
            print(f"       True: cov={targets[i, 0]:.3f}, dist={targets[i, 1]:.3f}, rad={targets[i, 2]:.3f}")

    print(f"\nModels saved to {save_dir}")


if __name__ == "__main__":
    main()
