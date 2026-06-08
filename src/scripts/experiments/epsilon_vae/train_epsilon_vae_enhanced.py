# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Train Enhanced Epsilon-VAE with comprehensive features.

This script trains an Epsilon-VAE that uses:
1. Flattened model weights (main input)
2. 54 additional features (context): epoch, progress, hyperparameters, weight stats

The enhanced model conditions on all available information to make
better metric predictions.

Usage:
    python src/scripts/epsilon_vae/train_epsilon_vae_enhanced.py
    python src/scripts/epsilon_vae/train_epsilon_vae_enhanced.py --epochs 100
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
from src.models.epsilon_vae import extract_key_weights
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat


class EnhancedCheckpointDataset(Dataset):
    """Dataset with weights + additional features."""

    def __init__(self, metadata_path: Path, max_samples: int = None):
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        if max_samples:
            self.metadata = self.metadata[:max_samples]

        print(f"Loading {len(self.metadata)} checkpoints...")
        self.samples = []

        for i, item in enumerate(self.metadata):
            try:
                # Load checkpoint for weights
                ckpt = load_checkpoint_compat(item["weights_path"], map_location="cpu")
                state_dict = get_model_state_dict(ckpt)
                weights = extract_key_weights(state_dict)

                if weights:
                    flat_weights = torch.cat([w.flatten() for w in weights])

                    # Get pre-computed features
                    features = torch.tensor(item["features"], dtype=torch.float32)

                    # Target metrics
                    metrics = torch.tensor(
                        [
                            item["metrics"]["coverage"],
                            item["metrics"]["distance_corr_A"],
                            item["metrics"]["radial_corr_A"],
                        ],
                        dtype=torch.float32,
                    )

                    self.samples.append(
                        {
                            "weights": flat_weights,
                            "features": features,
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
        dim_counts = {}
        for s in self.samples:
            dim = s["weights"].shape[0]
            dim_counts[dim] = dim_counts.get(dim, 0) + 1

        most_common_dim = max(dim_counts, key=dim_counts.get)
        original_count = len(self.samples)
        self.samples = [s for s in self.samples if s["weights"].shape[0] == most_common_dim]

        if len(self.samples) < original_count:
            print(f"  Filtered to {len(self.samples)}/{original_count} samples with weight_dim={most_common_dim}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Collate function for dataloader."""
    weights = torch.stack([s["weights"] for s in batch])
    features = torch.stack([s["features"] for s in batch])
    metrics = torch.stack([s["metrics"] for s in batch])
    return {"weights": weights, "features": features, "metrics": metrics}


class EnhancedEpsilonVAE(nn.Module):
    """Enhanced Epsilon-VAE with feature conditioning.

    Architecture:
    - Weight encoder: weights -> hidden representation
    - Feature encoder: 54 features -> feature embedding
    - Fusion: concatenate weight repr + feature embedding
    - Latent: fused -> mu, logvar -> z
    - Metric predictor: z + features -> predicted metrics
    """

    def __init__(
        self,
        weight_dim: int,
        feature_dim: int = 54,
        latent_dim: int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim

        # Weight encoder
        self.weight_encoder = nn.Sequential(
            nn.Linear(weight_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Feature encoder (for the 54 additional features)
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # Fusion layer
        fusion_dim = hidden_dim + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mu, logvar
        )

        # Metric predictor (conditioned on z AND features)
        self.metric_predictor = nn.Sequential(
            nn.Linear(latent_dim + 64, 128),  # z + feature embedding
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 3),  # [coverage, dist_corr, rad_hier]
        )

    def encode(self, weights, features):
        """Encode weights + features to latent distribution."""
        weight_repr = self.weight_encoder(weights)
        feature_repr = self.feature_encoder(features)

        fused = torch.cat([weight_repr, feature_repr], dim=-1)
        params = self.fusion(fused)
        mu, logvar = params.chunk(2, dim=-1)

        return mu, logvar, feature_repr

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def predict_metrics(self, z, feature_repr):
        """Predict metrics from latent z and feature representation."""
        combined = torch.cat([z, feature_repr], dim=-1)
        return self.metric_predictor(combined)

    def forward(self, weights, features):
        """Full forward pass."""
        mu, logvar, feature_repr = self.encode(weights, features)
        z = self.reparameterize(mu, logvar)
        metrics = self.predict_metrics(z, feature_repr)
        return mu, logvar, metrics


def epsilon_vae_loss(metrics_pred, metrics_true, mu, logvar, beta=0.01):
    """Compute loss for Enhanced Epsilon-VAE."""
    # Metric prediction loss (MSE)
    metric_loss = nn.functional.mse_loss(metrics_pred, metrics_true)

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    total = metric_loss + beta * kl_loss

    return {
        "total": total,
        "metric_loss": metric_loss,
        "kl_loss": kl_loss,
    }


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_metric_loss = 0
    total_kl_loss = 0

    for batch in dataloader:
        weights = batch["weights"].to(device)
        features = batch["features"].to(device)
        metrics = batch["metrics"].to(device)

        optimizer.zero_grad()

        mu, logvar, metrics_pred = model(weights, features)

        losses = epsilon_vae_loss(
            metrics_pred=metrics_pred,
            metrics_true=metrics,
            mu=mu,
            logvar=logvar,
            beta=0.01,
        )

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            features = batch["features"].to(device)
            metrics = batch["metrics"].to(device)

            mu, logvar, feature_repr = model.encode(weights, features)
            metrics_pred = model.predict_metrics(mu, feature_repr)

            all_preds.append(metrics_pred.cpu())
            all_targets.append(metrics.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)

    errors = (preds - targets).abs()

    return {
        "coverage_mae": errors[:, 0].mean().item(),
        "dist_corr_mae": errors[:, 1].mean().item(),
        "rad_hier_mae": errors[:, 2].mean().item(),
        "total_mae": errors.mean().item(),
        "preds": preds,
        "targets": targets,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced Epsilon-VAE")
    parser.add_argument("--data_dir", type=str, default=str(OUTPUT_DIR / "epsilon_vae_data_enhanced"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_dir", type=str, default=str(OUTPUT_DIR / "epsilon_vae_models_enhanced"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load feature metadata
    with open(data_dir / "feature_meta.json") as f:
        feature_meta = json.load(f)
    feature_dim = feature_meta["feature_dim"]
    print(f"Feature dimension: {feature_dim}")
    print(f"Feature names: {feature_meta['feature_names'][:10]}...")

    # Load datasets
    print("\n" + "=" * 70)
    print("LOADING ENHANCED DATASETS")
    print("=" * 70)

    train_dataset = EnhancedCheckpointDataset(data_dir / "train_dataset.json")
    val_dataset = EnhancedCheckpointDataset(data_dir / "val_dataset.json")

    if len(train_dataset) == 0:
        print("ERROR: No training samples loaded!")
        return

    # Get dimensions
    weight_dim = train_dataset[0]["weights"].shape[0]
    print(f"\nWeight dimension: {weight_dim}")
    print(f"Feature dimension: {feature_dim}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        if len(val_dataset) > 0
        else None
    )

    # Create model
    print("\n" + "=" * 70)
    print("CREATING ENHANCED EPSILON-VAE")
    print("=" * 70)

    model = EnhancedEpsilonVAE(
        weight_dim=weight_dim,
        feature_dim=feature_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Architecture:")
    print(f"  Weight encoder: {weight_dim} -> {args.hidden_dim}")
    print(f"  Feature encoder: {feature_dim} -> 64")
    print(f"  Fusion: {args.hidden_dim + 64} -> {args.latent_dim}")
    print(f"  Metric predictor: {args.latent_dim + 64} -> 3")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    best_val_mae = float("inf")
    history = []

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        val_metrics = validate(model, val_loader, device) if val_loader else None

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
                    "config": {
                        "weight_dim": weight_dim,
                        "feature_dim": feature_dim,
                        "latent_dim": args.latent_dim,
                        "hidden_dim": args.hidden_dim,
                    },
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
            "config": {
                "weight_dim": weight_dim,
                "feature_dim": feature_dim,
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
            },
        },
        save_dir / "final.pt",
    )

    # Final validation report
    if val_loader:
        print("\n" + "=" * 70)
        print("FINAL VALIDATION (on unseen future checkpoints)")
        print("=" * 70)

        val_metrics = validate(model, val_loader, device)
        print(f"Total MAE: {val_metrics['total_mae']:.4f}")
        print(f"  Coverage MAE: {val_metrics['coverage_mae']:.4f}")
        print(f"  Dist Corr MAE: {val_metrics['dist_corr_mae']:.4f}")
        print(f"  Rad Hier MAE: {val_metrics['rad_hier_mae']:.4f}")

        print(f"\nBest Val MAE achieved: {best_val_mae:.4f}")

        # Show predictions vs targets
        print("\nPredictions vs Targets:")
        preds = val_metrics["preds"]
        targets = val_metrics["targets"]
        for i in range(min(10, len(preds))):
            print(f"  [{i}] Pred: cov={preds[i, 0]:.3f}, dist={preds[i, 1]:.3f}, rad={preds[i, 2]:.3f}")
            print(f"       True: cov={targets[i, 0]:.3f}, dist={targets[i, 1]:.3f}, rad={targets[i, 2]:.3f}")
            print(
                f"       Error: cov={abs(preds[i, 0] - targets[i, 0]):.3f}, dist={abs(preds[i, 1] - targets[i, 1]):.3f}, rad={abs(preds[i, 2] - targets[i, 2]):.3f}"
            )

    print(f"\nModels saved to {save_dir}")


if __name__ == "__main__":
    main()
