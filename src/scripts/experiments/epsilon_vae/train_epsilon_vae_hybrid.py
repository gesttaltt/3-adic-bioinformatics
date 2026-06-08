# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Train Hybrid Epsilon-VAE on embedding space + metrics.

This script trains an Epsilon-VAE with two objectives:
1. PRIMARY: Reconstruct the actual embedding space (z_A_hyp) from checkpoint weights
2. AUXILIARY: Predict summary metrics (coverage, dist_corr, rad_hier)

The key insight: Training on the full embedding space captures the complete
geometric structure, not just 3 summary statistics. This prevents the model
from collapsing to predicting the mean.

Architecture:
    checkpoint_weights ──► [Weight Encoder] ──► latent z (32-dim)
                                                  │
                              ┌───────────────────┴──────────────────┐
                              ↓                                      ↓
                   [Embedding Decoder]                      [Metric Predictor]
                              ↓                                      ↓
                Predicted anchor embeddings              3 metrics (auxiliary)
                       (N_anchors × 16)                (coverage, dist, rad)

Loss:
    L = L_embed + λ_metric * L_metric + β * L_kl

    where L_embed compares predicted vs actual anchor embeddings.

Usage:
    python src/scripts/epsilon_vae/train_epsilon_vae_hybrid.py
    python src/scripts/epsilon_vae/train_epsilon_vae_hybrid.py --epochs 200 --embed_weight 1.0 --metric_weight 0.1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import OUTPUT_DIR


class HybridCheckpointDataset(Dataset):
    """Dataset with checkpoint weights, anchor embeddings, and metrics."""

    def __init__(self, data_dir: Path, split: str = "train"):
        """Load preprocessed data.

        Args:
            data_dir: Directory with numpy arrays from extract_embeddings.py
            split: "train" or "val"
        """
        self.data_dir = Path(data_dir)

        # Load arrays
        self.z_A_hyp = np.load(self.data_dir / f"{split}_z_A_hyp.npy")  # (N, n_anchors, embed_dim)
        self.weights = np.load(self.data_dir / f"{split}_weights.npy")  # (N, weight_dim)
        self.metrics = np.load(self.data_dir / f"{split}_metrics.npy")  # (N, 3)

        # Load metadata
        with open(self.data_dir / f"{split}_metadata.json") as f:
            self.metadata = json.load(f)

        print(f"Loaded {split} dataset:")
        print(f"  Samples: {len(self.weights)}")
        print(f"  Weight dim: {self.weights.shape[1]}")
        print(f"  Embedding shape: {self.z_A_hyp.shape[1:]} (n_anchors, embed_dim)")

    def __len__(self):
        return len(self.weights)

    def __getitem__(self, idx):
        return {
            "weights": torch.tensor(self.weights[idx], dtype=torch.float32),
            "z_A_hyp": torch.tensor(self.z_A_hyp[idx], dtype=torch.float32),
            "metrics": torch.tensor(self.metrics[idx], dtype=torch.float32),
        }


class HybridEpsilonVAE(nn.Module):
    """Hybrid Epsilon-VAE that predicts both embeddings and metrics.

    Architecture:
    - Weight encoder: checkpoint weights -> latent z
    - Embedding decoder: z -> predicted anchor embeddings
    - Metric predictor: z -> 3 metrics (auxiliary)
    """

    def __init__(
        self,
        weight_dim: int,
        n_anchors: int,
        embed_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_anchors = n_anchors
        self.embed_dim = embed_dim
        self.output_dim = n_anchors * embed_dim

        # Weight encoder (checkpoint weights -> latent distribution)
        self.encoder = nn.Sequential(
            nn.Linear(weight_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim * 2),  # mu, logvar
        )

        # Embedding decoder (latent -> anchor embeddings)
        # This is the KEY component: learns the mapping from latent to full embedding space
        self.embedding_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.output_dim),  # n_anchors * embed_dim
        )

        # Metric predictor (auxiliary task)
        self.metric_predictor = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 3),  # coverage, dist_corr, rad_hier
        )

    def encode(self, weights: torch.Tensor) -> tuple:
        """Encode weights to latent distribution."""
        params = self.encoder(weights)
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode_embeddings(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to anchor embeddings."""
        flat = self.embedding_decoder(z)
        return flat.view(-1, self.n_anchors, self.embed_dim)

    def predict_metrics(self, z: torch.Tensor) -> torch.Tensor:
        """Predict metrics from latent."""
        return self.metric_predictor(z)

    def forward(self, weights: torch.Tensor) -> dict:
        """Full forward pass."""
        mu, logvar = self.encode(weights)
        z = self.reparameterize(mu, logvar)

        embeddings_pred = self.decode_embeddings(z)
        metrics_pred = self.predict_metrics(z)

        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "embeddings_pred": embeddings_pred,
            "metrics_pred": metrics_pred,
        }


def hybrid_loss(
    embeddings_pred: torch.Tensor,
    embeddings_true: torch.Tensor,
    metrics_pred: torch.Tensor,
    metrics_true: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    embed_weight: float = 1.0,
    metric_weight: float = 0.1,
    beta: float = 0.001,
) -> dict:
    """Compute hybrid loss.

    Args:
        embeddings_pred: Predicted anchor embeddings (batch, n_anchors, embed_dim)
        embeddings_true: True anchor embeddings
        metrics_pred: Predicted metrics (batch, 3)
        metrics_true: True metrics
        mu, logvar: Latent distribution parameters
        embed_weight: Weight for embedding reconstruction loss
        metric_weight: Weight for metric prediction loss
        beta: Weight for KL divergence

    Returns:
        Dict with loss components
    """
    # Embedding reconstruction loss (MSE on hyperbolic embeddings)
    # This is the PRIMARY loss - captures full geometric structure
    embed_loss = nn.functional.mse_loss(embeddings_pred, embeddings_true)

    # Metric prediction loss (auxiliary)
    metric_loss = nn.functional.mse_loss(metrics_pred, metrics_true)

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss
    total = embed_weight * embed_loss + metric_weight * metric_loss + beta * kl_loss

    return {
        "total": total,
        "embed_loss": embed_loss,
        "metric_loss": metric_loss,
        "kl_loss": kl_loss,
    }


def train_epoch(model, dataloader, optimizer, device, embed_weight, metric_weight, beta):
    """Train for one epoch."""
    model.train()
    total_losses = {"total": 0, "embed_loss": 0, "metric_loss": 0, "kl_loss": 0}

    for batch in dataloader:
        weights = batch["weights"].to(device)
        z_A_hyp = batch["z_A_hyp"].to(device)
        metrics = batch["metrics"].to(device)

        optimizer.zero_grad()

        outputs = model(weights)

        losses = hybrid_loss(
            embeddings_pred=outputs["embeddings_pred"],
            embeddings_true=z_A_hyp,
            metrics_pred=outputs["metrics_pred"],
            metrics_true=metrics,
            mu=outputs["mu"],
            logvar=outputs["logvar"],
            embed_weight=embed_weight,
            metric_weight=metric_weight,
            beta=beta,
        )

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k, v in losses.items():
            total_losses[k] += v.item()

    n_batches = len(dataloader)
    return {k: v / n_batches for k, v in total_losses.items()}


def validate(model, dataloader, device, embed_weight, metric_weight, beta):
    """Validate on held-out data."""
    model.eval()

    all_embed_preds = []
    all_embed_targets = []
    all_metric_preds = []
    all_metric_targets = []

    with torch.no_grad():
        for batch in dataloader:
            weights = batch["weights"].to(device)
            z_A_hyp = batch["z_A_hyp"].to(device)
            metrics = batch["metrics"].to(device)

            outputs = model(weights)

            all_embed_preds.append(outputs["embeddings_pred"].cpu())
            all_embed_targets.append(z_A_hyp.cpu())
            all_metric_preds.append(outputs["metrics_pred"].cpu())
            all_metric_targets.append(metrics.cpu())

    # Concatenate
    embed_preds = torch.cat(all_embed_preds)
    embed_targets = torch.cat(all_embed_targets)
    metric_preds = torch.cat(all_metric_preds)
    metric_targets = torch.cat(all_metric_targets)

    # Compute losses
    embed_loss = nn.functional.mse_loss(embed_preds, embed_targets).item()
    metric_loss = nn.functional.mse_loss(metric_preds, metric_targets).item()

    # Per-metric MAE
    metric_errors = (metric_preds - metric_targets).abs()

    # Embedding cosine similarity (geometric quality)
    embed_preds_flat = embed_preds.view(embed_preds.shape[0], -1)
    embed_targets_flat = embed_targets.view(embed_targets.shape[0], -1)
    cosine_sim = nn.functional.cosine_similarity(embed_preds_flat, embed_targets_flat, dim=1).mean().item()

    return {
        "embed_loss": embed_loss,
        "metric_loss": metric_loss,
        "total_loss": embed_weight * embed_loss + metric_weight * metric_loss,
        "coverage_mae": metric_errors[:, 0].mean().item(),
        "dist_corr_mae": metric_errors[:, 1].mean().item(),
        "rad_hier_mae": metric_errors[:, 2].mean().item(),
        "total_metric_mae": metric_errors.mean().item(),
        "embed_cosine_sim": cosine_sim,
        "embed_preds": embed_preds,
        "embed_targets": embed_targets,
        "metric_preds": metric_preds,
        "metric_targets": metric_targets,
    }


def main():
    parser = argparse.ArgumentParser(description="Train Hybrid Epsilon-VAE")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(OUTPUT_DIR / "epsilon_vae_hybrid"),
        help="Directory with extracted embeddings",
    )
    parser.add_argument(
        "--save_dir", type=str, default=str(OUTPUT_DIR / "epsilon_vae_hybrid_models"), help="Directory to save models"
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--embed_weight", type=float, default=1.0, help="Weight for embedding reconstruction loss")
    parser.add_argument("--metric_weight", type=float, default=0.1, help="Weight for metric prediction loss")
    parser.add_argument("--beta", type=float, default=0.001, help="Weight for KL divergence")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_dir = PROJECT_ROOT / args.data_dir
    save_dir = PROJECT_ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load config
    with open(data_dir / "config.json") as f:
        config = json.load(f)

    print(f"\n{'=' * 70}")
    print("DATA CONFIG")
    print(f"{'=' * 70}")
    print(f"Weight dim: {config['weight_dim']}")
    print(f"Embed dim: {config['embed_dim']}")
    print(f"N anchors: {config['n_anchors']}")
    print(f"N train: {config['n_train']}")
    print(f"N val: {config['n_val']}")

    # Load datasets
    print(f"\n{'=' * 70}")
    print("LOADING DATASETS")
    print(f"{'=' * 70}")

    train_dataset = HybridCheckpointDataset(data_dir, "train")
    val_dataset = HybridCheckpointDataset(data_dir, "val")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    print(f"\n{'=' * 70}")
    print("CREATING HYBRID EPSILON-VAE")
    print(f"{'=' * 70}")

    model = HybridEpsilonVAE(
        weight_dim=config["weight_dim"],
        n_anchors=config["n_anchors"],
        embed_dim=config["embed_dim"],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print("Architecture:")
    print(f"  Weight encoder: {config['weight_dim']} -> {args.hidden_dim} -> {args.latent_dim}")
    print(f"  Embedding decoder: {args.latent_dim} -> {args.hidden_dim} -> {config['n_anchors'] * config['embed_dim']}")
    print(f"  Metric predictor: {args.latent_dim} -> 128 -> 3")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    print(f"\n{'=' * 70}")
    print("TRAINING")
    print(f"{'=' * 70}")
    print(f"Embed weight: {args.embed_weight}")
    print(f"Metric weight: {args.metric_weight}")
    print(f"Beta (KL): {args.beta}")
    print()

    best_val_loss = float("inf")
    history = []

    for epoch in range(args.epochs):
        train_losses = train_epoch(
            model, train_loader, optimizer, device, args.embed_weight, args.metric_weight, args.beta
        )
        scheduler.step()

        val_metrics = validate(model, val_loader, device, args.embed_weight, args.metric_weight, args.beta)

        log = (
            f"Epoch {epoch:3d} | "
            f"Train: {train_losses['total']:.4f} (E:{train_losses['embed_loss']:.4f} M:{train_losses['metric_loss']:.4f}) | "
            f"Val: {val_metrics['total_loss']:.4f} | "
            f"Cos: {val_metrics['embed_cosine_sim']:.3f} | "
            f"MAE: cov={val_metrics['coverage_mae']:.3f} dist={val_metrics['dist_corr_mae']:.3f} rad={val_metrics['rad_hier_mae']:.3f}"
        )

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(log)

        history.append(
            {
                "epoch": epoch,
                "train_total": train_losses["total"],
                "train_embed": train_losses["embed_loss"],
                "train_metric": train_losses["metric_loss"],
                "val_total": val_metrics["total_loss"],
                "val_embed": val_metrics["embed_loss"],
                "val_cosine_sim": val_metrics["embed_cosine_sim"],
                "val_coverage_mae": val_metrics["coverage_mae"],
                "val_dist_corr_mae": val_metrics["dist_corr_mae"],
                "val_rad_hier_mae": val_metrics["rad_hier_mae"],
            }
        )

        # Save best
        if val_metrics["total_loss"] < best_val_loss:
            best_val_loss = val_metrics["total_loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_metrics": {k: v for k, v in val_metrics.items() if not isinstance(v, torch.Tensor)},
                    "config": {
                        "weight_dim": config["weight_dim"],
                        "n_anchors": config["n_anchors"],
                        "embed_dim": config["embed_dim"],
                        "latent_dim": args.latent_dim,
                        "hidden_dim": args.hidden_dim,
                    },
                    "args": vars(args),
                },
                save_dir / "best.pt",
            )

    # Save final
    torch.save(
        {
            "epoch": args.epochs - 1,
            "model_state_dict": model.state_dict(),
            "history": history,
            "config": {
                "weight_dim": config["weight_dim"],
                "n_anchors": config["n_anchors"],
                "embed_dim": config["embed_dim"],
                "latent_dim": args.latent_dim,
                "hidden_dim": args.hidden_dim,
            },
            "args": vars(args),
        },
        save_dir / "final.pt",
    )

    # Final validation
    print(f"\n{'=' * 70}")
    print("FINAL VALIDATION")
    print(f"{'=' * 70}")

    val_metrics = validate(model, val_loader, device, args.embed_weight, args.metric_weight, args.beta)

    print("\nEmbedding Reconstruction:")
    print(f"  MSE: {val_metrics['embed_loss']:.6f}")
    print(f"  Cosine Similarity: {val_metrics['embed_cosine_sim']:.4f}")

    print("\nMetric Prediction (MAE):")
    print(f"  Coverage:  {val_metrics['coverage_mae']:.4f}")
    print(f"  Dist Corr: {val_metrics['dist_corr_mae']:.4f}")
    print(f"  Rad Hier:  {val_metrics['rad_hier_mae']:.4f}")
    print(f"  Total:     {val_metrics['total_metric_mae']:.4f}")

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Models saved to {save_dir}")

    # Show sample predictions
    print(f"\n{'=' * 70}")
    print("SAMPLE PREDICTIONS vs TARGETS")
    print(f"{'=' * 70}")

    metric_preds = val_metrics["metric_preds"]
    metric_targets = val_metrics["metric_targets"]

    for i in range(min(10, len(metric_preds))):
        print(
            f"\n[{i}] Predicted: cov={metric_preds[i, 0]:.3f}, dist={metric_preds[i, 1]:.3f}, rad={metric_preds[i, 2]:.3f}"
        )
        print(
            f"    Actual:    cov={metric_targets[i, 0]:.3f}, dist={metric_targets[i, 1]:.3f}, rad={metric_targets[i, 2]:.3f}"
        )
        error = (metric_preds[i] - metric_targets[i]).abs()
        print(f"    Error:     cov={error[0]:.3f}, dist={error[1]:.3f}, rad={error[2]:.3f}")


if __name__ == "__main__":
    main()
