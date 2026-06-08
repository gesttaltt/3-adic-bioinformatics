#!/usr/bin/env python3
"""Ensemble Training - Combine VAE + MLP Refiner + Gradient Features.

Takes the best performing approach (MLP Refiner at 0.78) and enhances it
with gradient-based features learned from the latent space structure.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parents[3]))

from src.bioinformatics.data.preprocessing import compute_features
from src.bioinformatics.data.protherm_loader import ProThermLoader
from src.bioinformatics.models.ddg_vae import DDGVAE


@dataclass
class EnsembleConfig:
    """Configuration for ensemble training."""

    epochs: int = 200
    batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    patience: int = 50
    dropout: float = 0.15

    vae_checkpoint: str = "outputs/ddg_vae_training_20260129_212316/vae_protherm/best.pt"
    mlp_checkpoint: str = "outputs/refiners_20260129_230857/mlp_refiner/best.pt"


class EnsemblePredictor(nn.Module):
    """Ensemble predictor combining VAE embeddings with learned corrections."""

    def __init__(self, vae: DDGVAE, dropout: float = 0.15):
        super().__init__()

        # Freeze VAE encoder
        self.encoder = vae.encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # MLP branch (similar to working MLP Refiner)
        self.mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.SiLU(),
        )

        # Gradient learner - learns the optimal DDG direction
        self.gradient_learner = nn.Linear(32, 1, bias=False)

        # Residual learner - learns corrections to gradient prediction
        self.residual_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )

        # Combine gradient and residual with learned weights
        self.combination_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass."""
        # Get VAE embedding
        with torch.no_grad():
            mu, _ = self.encoder(x)

        # MLP processing
        h = self.mlp(mu)

        # Gradient-based prediction (learned optimal direction)
        gradient_pred = self.gradient_learner(mu).squeeze(-1)

        # Residual correction
        residual = self.residual_head(h).squeeze(-1)

        # Combine
        w = torch.sigmoid(self.combination_weight)
        ddg = gradient_pred + w * residual

        return {
            "ddg": ddg,
            "gradient_pred": gradient_pred,
            "residual": residual,
            "weight": w,
            "embedding": mu,
            "hidden": h,
        }


class EnsembleDataset(Dataset):
    """Dataset for ensemble prediction."""

    def __init__(self, records: list):
        self.features = []
        self.ddg_values = []
        self.mutation_ids = []

        for record in records:
            feat = compute_features(record.wild_type, record.mutant)
            arr = feat.to_array(include_hyperbolic=False)
            arr = np.pad(arr, (0, 6), mode="constant")

            self.features.append(arr)
            self.ddg_values.append(record.ddg)
            self.mutation_ids.append(f"{record.pdb_id}_{record.chain}_{record.mutation_string}")

        self.features = np.array(self.features, dtype=np.float32)
        self.ddg_values = np.array(self.ddg_values, dtype=np.float32)

    def __len__(self):
        return len(self.ddg_values)

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.features[idx]),
            "ddg": torch.tensor(self.ddg_values[idx], dtype=torch.float32),
        }


def train_ensemble():
    """Train ensemble predictor."""
    print("=" * 70)
    print("Ensemble DDG Prediction Training")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = EnsembleConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/ensemble_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Load VAE
    print("\n[1] Loading VAE-ProTherm...")
    vae = DDGVAE.create_protherm_variant(use_hyperbolic=False)
    ckpt = torch.load(config.vae_checkpoint, map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae = vae.to(device)
    print("  VAE loaded")

    # Create ensemble model
    print("\n[2] Creating ensemble predictor...")
    model = EnsemblePredictor(vae, config.dropout)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")

    # Load data
    print("\n[3] Loading ProTherm data...")
    loader = ProThermLoader()
    db = loader.load_curated()
    records = db.records
    print(f"  Loaded {len(records)} mutations")

    # Train/val split (same as MLP Refiner for fair comparison)
    np.random.seed(42)
    indices = np.random.permutation(len(records))
    split = int(0.8 * len(records))
    train_records = [records[i] for i in indices[:split]]
    val_records = [records[i] for i in indices[split:]]
    print(f"  Train: {len(train_records)}, Val: {len(val_records)}")

    train_dataset = EnsembleDataset(train_records)
    val_dataset = EnsembleDataset(val_records)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-7)

    # Training
    print("\n" + "=" * 70)
    print("[4] Training Ensemble")
    print("=" * 70)

    best_spearman = -1
    patience_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_spearman": [],
        "val_pearson": [],
        "gradient_spearman": [],
        "weight": [],
    }

    for epoch in range(config.epochs):
        # Train
        model.train()
        train_losses = []

        for batch in train_loader:
            x = batch["x"].to(device)
            ddg = batch["ddg"].to(device)

            optimizer.zero_grad()
            out = model(x)

            # MSE loss
            loss = F.mse_loss(out["ddg"], ddg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        all_gradient_preds = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                ddg = batch["ddg"].to(device)

                out = model(x)
                loss = F.mse_loss(out["ddg"], ddg)

                val_losses.append(loss.item())
                all_preds.extend(out["ddg"].cpu().numpy())
                all_targets.extend(ddg.cpu().numpy())
                all_gradient_preds.extend(out["gradient_pred"].cpu().numpy())

        # Metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_spearman = spearmanr(all_targets, all_preds)[0]
        val_pearson = pearsonr(all_targets, all_preds)[0]
        gradient_spearman = spearmanr(all_targets, all_gradient_preds)[0]
        weight = torch.sigmoid(model.combination_weight).item()

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_spearman"].append(float(val_spearman))
        history["val_pearson"].append(float(val_pearson))
        history["gradient_spearman"].append(float(gradient_spearman))
        history["weight"].append(float(weight))

        scheduler.step()

        # Logging
        if epoch % 10 == 0 or val_spearman > best_spearman:
            print(
                f"Epoch {epoch:3d}: loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"spearman={val_spearman:.4f} pearson={val_pearson:.4f} "
                f"grad_spear={gradient_spearman:.4f} w={weight:.3f}"
            )

        # Best checkpoint
        if val_spearman > best_spearman:
            best_spearman = val_spearman
            best_pearson = val_pearson
            best_gradient_spearman = gradient_spearman
            patience_counter = 0

            # Extract learned gradient direction
            learned_gradient = model.gradient_learner.weight.data.cpu().numpy().flatten()

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_spearman": val_spearman,
                    "val_pearson": val_pearson,
                    "gradient_spearman": gradient_spearman,
                    "learned_gradient": learned_gradient.tolist(),
                    "config": config.__dict__,
                },
                output_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Extract final learned gradient
    learned_gradient = model.gradient_learner.weight.data.cpu().numpy().flatten()

    print("\n" + "=" * 70)
    print("ENSEMBLE TRAINING COMPLETE")
    print("=" * 70)
    print("\nBest Results:")
    print(f"  Spearman (ensemble):  {best_spearman:.4f}")
    print(f"  Pearson (ensemble):   {best_pearson:.4f}")
    print(f"  Spearman (gradient):  {best_gradient_spearman:.4f}")
    print(f"\nCheckpoint: {output_dir / 'best.pt'}")

    print("\n" + "=" * 70)
    print("Results Comparison")
    print("=" * 70)
    print("  VAE-ProTherm alone:   0.64")
    print("  MLP Refiner:          0.78")
    print(f"  Ensemble:             {best_spearman:.2f}")

    improvement = (best_spearman - 0.64) / 0.64 * 100
    print(f"\n  Improvement over VAE: {improvement:+.1f}%")

    if best_spearman >= 0.80:
        print("\n  TARGET ACHIEVED: Spearman >= 0.80")

    # Analyze learned gradient
    print("\n" + "=" * 70)
    print("Learned Gradient Analysis")
    print("=" * 70)
    top_dims = np.argsort(np.abs(learned_gradient))[::-1][:5]
    print("  Top 5 dimensions by importance:")
    for rank, dim in enumerate(top_dims, 1):
        print(f"    {rank}. Dim {dim}: {learned_gradient[dim]:.4f}")

    return model, best_spearman


if __name__ == "__main__":
    train_ensemble()
