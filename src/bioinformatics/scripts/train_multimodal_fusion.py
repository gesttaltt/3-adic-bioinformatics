#!/usr/bin/env python3
"""Multimodal Fusion Training - Combine Three Specialist VAEs.

This script trains a cross-modal attention fusion layer that combines
embeddings from VAE-S669, VAE-ProTherm, and VAE-Wide to achieve
superior DDG prediction.

Target: Spearman > 0.80 on ProTherm validation set.
"""

from __future__ import annotations

import json

# Add project root to path
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parents[3]))

from src.bioinformatics.data.protherm_loader import ProThermLoader
from src.bioinformatics.models.ddg_vae import DDGVAE


@dataclass
class FusionConfig:
    """Configuration for multimodal fusion."""

    # VAE latent dimensions
    s669_dim: int = 16
    protherm_dim: int = 32
    wide_dim: int = 64

    # Fusion architecture - simplified for small dataset
    fusion_dim: int = 64
    n_heads: int = 4
    n_fusion_layers: int = 1
    dropout: float = 0.3  # Higher dropout for regularization

    # Training
    epochs: int = 300
    batch_size: int = 8  # Smaller batch for more updates
    learning_rate: float = 5e-5  # Lower LR
    weight_decay: float = 1e-3  # Stronger regularization
    patience: int = 50

    # Paths
    s669_checkpoint: str = "outputs/ddg_vae_training_20260129_212316/vae_s669/best.pt"
    protherm_checkpoint: str = "outputs/ddg_vae_training_20260129_212316/vae_protherm/best.pt"
    wide_checkpoint: str = "outputs/vae_wide_filtered_20260129_220019/best.pt"


class CrossModalAttention(nn.Module):
    """Cross-modal attention layer for fusing VAE embeddings."""

    def __init__(self, dims: list[int], fusion_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.n_modalities = len(dims)
        self.fusion_dim = fusion_dim

        # Project each modality to fusion dimension
        self.projections = nn.ModuleList([nn.Linear(d, fusion_dim) for d in dims])

        # Learnable modality embeddings
        self.modality_embeddings = nn.Parameter(torch.randn(self.n_modalities, fusion_dim) * 0.02)

        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        # Layer norm
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple modality embeddings via attention.

        Args:
            embeddings: List of [batch, dim_i] tensors for each modality

        Returns:
            Fused embedding [batch, fusion_dim]
        """
        embeddings[0].shape[0]

        # Project each modality and add modality embedding
        projected = []
        for i, (proj, emb) in enumerate(zip(self.projections, embeddings, strict=False)):
            x = proj(emb)  # [batch, fusion_dim]
            x = x + self.modality_embeddings[i]  # Add modality identity
            projected.append(x)

        # Stack as sequence: [batch, n_modalities, fusion_dim]
        x = torch.stack(projected, dim=1)

        # Self-attention across modalities
        attn_out, attn_weights = self.cross_attention(x, x, x)

        # Residual + norm
        x = self.norm(x + attn_out)

        # Pool across modalities (attention-weighted mean)
        # Use attention weights from first position as importance
        weights = attn_weights[:, 0, :].unsqueeze(-1)  # [batch, n_mod, 1]
        fused = (x * weights).sum(dim=1)  # [batch, fusion_dim]

        return fused


class SimpleFusion(nn.Module):
    """Simple concatenation-based fusion with gating."""

    def __init__(self, dims: list[int], fusion_dim: int, dropout: float = 0.3):
        super().__init__()
        total_dim = sum(dims)

        # Learned importance weights per modality
        self.modality_gates = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in dims])

        # Compression to fusion dimension
        self.compress = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.SiLU(),
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
        )

    def forward(self, embeddings: list[torch.Tensor]) -> torch.Tensor:
        # Apply learned gates
        gated = []
        for emb, gate in zip(embeddings, self.modality_gates, strict=False):
            gated.append(emb * torch.sigmoid(gate))

        # Concatenate and compress
        concat = torch.cat(gated, dim=-1)
        return self.compress(concat)


class MultimodalFusionVAE(nn.Module):
    """Multimodal VAE that fuses three specialist encoders."""

    def __init__(
        self,
        vae_s669: DDGVAE,
        vae_protherm: DDGVAE,
        vae_wide: DDGVAE,
        config: FusionConfig,
        use_attention: bool = False,  # Default to simple fusion
    ):
        super().__init__()

        # Freeze specialist encoders
        self.encoder_s669 = vae_s669.encoder
        self.encoder_protherm = vae_protherm.encoder
        self.encoder_wide = vae_wide.encoder

        for param in self.encoder_s669.parameters():
            param.requires_grad = False
        for param in self.encoder_protherm.parameters():
            param.requires_grad = False
        for param in self.encoder_wide.parameters():
            param.requires_grad = False

        # Choose fusion type
        dims = [config.s669_dim, config.protherm_dim, config.wide_dim]
        if use_attention:
            self.fusion = CrossModalAttention(
                dims=dims,
                fusion_dim=config.fusion_dim,
                n_heads=config.n_heads,
                dropout=config.dropout,
            )
        else:
            self.fusion = SimpleFusion(
                dims=dims,
                fusion_dim=config.fusion_dim,
                dropout=config.dropout,
            )

        # Single fusion layer with residual
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim),
            nn.SiLU(),
            nn.LayerNorm(config.fusion_dim),
            nn.Dropout(config.dropout),
        )

        # Simple DDG prediction head
        self.ddg_head = nn.Sequential(
            nn.Linear(config.fusion_dim, 32),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, 1),
        )

        # Uncertainty head (for fuzzy output)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(config.fusion_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
            nn.Softplus(),
        )

    def encode_all(self, x_s669: torch.Tensor, x_protherm: torch.Tensor, x_wide: torch.Tensor):
        """Get embeddings from all three encoders."""
        with torch.no_grad():
            mu_s669, _ = self.encoder_s669(x_s669)
            mu_protherm, _ = self.encoder_protherm(x_protherm)
            mu_wide, _ = self.encoder_wide(x_wide)

        return [mu_s669, mu_protherm, mu_wide]

    def forward(self, x_s669: torch.Tensor, x_protherm: torch.Tensor, x_wide: torch.Tensor):
        """Forward pass through fusion network."""
        # Get frozen embeddings
        embeddings = self.encode_all(x_s669, x_protherm, x_wide)

        # Fusion
        fused = self.fusion(embeddings)

        # Residual fusion layer
        fused = fused + self.fusion_layer(fused)

        # Predictions
        ddg = self.ddg_head(fused).squeeze(-1)
        uncertainty = self.uncertainty_head(fused).squeeze(-1)

        return {
            "ddg": ddg,
            "uncertainty": uncertainty,
            "fused_embedding": fused,
            "modality_embeddings": embeddings,
        }


class MultimodalDataset(Dataset):
    """Dataset providing features for all three VAE encoders."""

    def __init__(self, records: list, include_hyperbolic: bool = False):
        """Initialize with ProTherm records."""
        from src.bioinformatics.data.preprocessing import add_hyperbolic_features, compute_features

        self.records = records
        self.features_s669 = []  # 14-dim (basic)
        self.features_protherm = []  # 20-dim (with hyperbolic)
        self.features_wide = []  # 14-dim (basic)
        self.ddg_values = []

        for record in records:
            # Basic features (14-dim)
            basic = compute_features(record.wild_type, record.mutant)
            basic_array = basic.to_array(include_hyperbolic=False)

            # Extended features with hyperbolic (20-dim)
            if include_hyperbolic:
                extended = add_hyperbolic_features(basic, record.wild_type, record.mutant, {}, 1.0)
                extended_array = extended.to_array(include_hyperbolic=True)
            else:
                # Pad to 20 dims for ProTherm encoder
                extended_array = np.pad(basic_array, (0, 6), mode="constant")

            self.features_s669.append(basic_array)
            self.features_protherm.append(extended_array)
            self.features_wide.append(basic_array)
            self.ddg_values.append(record.ddg)

        self.features_s669 = np.array(self.features_s669, dtype=np.float32)
        self.features_protherm = np.array(self.features_protherm, dtype=np.float32)
        self.features_wide = np.array(self.features_wide, dtype=np.float32)
        self.ddg_values = np.array(self.ddg_values, dtype=np.float32)

    def __len__(self):
        return len(self.ddg_values)

    def __getitem__(self, idx):
        return {
            "x_s669": torch.from_numpy(self.features_s669[idx]),
            "x_protherm": torch.from_numpy(self.features_protherm[idx]),
            "x_wide": torch.from_numpy(self.features_wide[idx]),
            "ddg": torch.tensor(self.ddg_values[idx], dtype=torch.float32),
        }


def gaussian_nll_loss(pred: torch.Tensor, target: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
    """Gaussian negative log-likelihood loss with learned uncertainty."""
    uncertainty = uncertainty.clamp(min=0.01, max=10.0)
    loss = 0.5 * (torch.log(uncertainty**2) + ((pred - target) ** 2) / (uncertainty**2))
    return loss.mean()


def ranking_loss(pred: torch.Tensor, target: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    """Pairwise ranking loss to optimize correlation directly.

    For pairs where target[i] > target[j], we want pred[i] > pred[j].
    """
    n = pred.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=pred.device)

    # Create all pairs
    pred_i = pred.unsqueeze(1)  # [n, 1]
    pred_j = pred.unsqueeze(0)  # [1, n]
    target_i = target.unsqueeze(1)
    target_j = target.unsqueeze(0)

    # Target ordering: 1 if target[i] > target[j], -1 otherwise
    target_diff = target_i - target_j
    ordering = torch.sign(target_diff)

    # Predicted difference
    pred_diff = pred_i - pred_j

    # Margin ranking loss: max(0, -ordering * pred_diff + margin)
    losses = F.relu(-ordering * pred_diff + margin)

    # Only consider upper triangle (avoid duplicate pairs)
    mask = torch.triu(torch.ones(n, n, device=pred.device), diagonal=1)
    losses = losses * mask

    return losses.sum() / mask.sum().clamp(min=1)


def train_multimodal_fusion():
    """Train the multimodal fusion model."""
    print("=" * 70)
    print("Multimodal Fusion Training")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = FusionConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/multimodal_fusion_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Load specialist VAEs
    print("\n[1] Loading specialist VAEs...")

    vae_s669 = DDGVAE.create_s669_variant(use_hyperbolic=False)
    ckpt = torch.load(config.s669_checkpoint, map_location=device, weights_only=False)
    vae_s669.load_state_dict(ckpt["model_state_dict"])
    print("  Loaded VAE-S669")

    vae_protherm = DDGVAE.create_protherm_variant(use_hyperbolic=False)
    ckpt = torch.load(config.protherm_checkpoint, map_location=device, weights_only=False)
    vae_protherm.load_state_dict(ckpt["model_state_dict"])
    print("  Loaded VAE-ProTherm")

    vae_wide = DDGVAE.create_wide_variant(use_hyperbolic=False)
    ckpt = torch.load(config.wide_checkpoint, map_location=device, weights_only=False)
    vae_wide.load_state_dict(ckpt["model_state_dict"])
    print("  Loaded VAE-Wide")

    # Create fusion model
    print("\n[2] Creating multimodal fusion model...")
    model = MultimodalFusionVAE(vae_s669, vae_protherm, vae_wide, config)
    model = model.to(device)

    # Count parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Frozen parameters: {frozen:,}")

    # Load ProTherm data
    print("\n[3] Loading ProTherm data...")
    loader = ProThermLoader()
    db = loader.load_curated()
    records = db.records
    print(f"  Loaded {len(records)} mutations")

    # Train/val split
    np.random.seed(42)
    indices = np.random.permutation(len(records))
    split = int(0.8 * len(records))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    print(f"  Train: {len(train_records)}, Val: {len(val_records)}")

    # Create datasets
    train_dataset = MultimodalDataset(train_records)
    val_dataset = MultimodalDataset(val_records)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

    # Training loop
    print("\n" + "=" * 70)
    print("[4] Training Multimodal Fusion")
    print("=" * 70)

    best_spearman = -1
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_spearman": [], "val_uncertainty": []}

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_losses = []

        for batch in train_loader:
            x_s669 = batch["x_s669"].to(device)
            x_protherm = batch["x_protherm"].to(device)
            x_wide = batch["x_wide"].to(device)
            ddg = batch["ddg"].to(device)

            optimizer.zero_grad()
            out = model(x_s669, x_protherm, x_wide)

            # Combined loss: MSE + ranking loss (directly optimizes correlation)
            mse_loss = F.mse_loss(out["ddg"], ddg)
            rank_loss = ranking_loss(out["ddg"], ddg, margin=0.1)

            # Weight ranking loss higher to focus on correlation
            loss = 0.3 * mse_loss + 0.7 * rank_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        # Validation
        model.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        all_uncertainties = []

        with torch.no_grad():
            for batch in val_loader:
                x_s669 = batch["x_s669"].to(device)
                x_protherm = batch["x_protherm"].to(device)
                x_wide = batch["x_wide"].to(device)
                ddg = batch["ddg"].to(device)

                out = model(x_s669, x_protherm, x_wide)

                mse_loss = F.mse_loss(out["ddg"], ddg)
                rank_loss = ranking_loss(out["ddg"], ddg, margin=0.1)
                loss = 0.3 * mse_loss + 0.7 * rank_loss

                val_losses.append(loss.item())
                all_preds.extend(out["ddg"].cpu().numpy())
                all_targets.extend(ddg.cpu().numpy())
                all_uncertainties.extend(out["uncertainty"].cpu().numpy())

        # Compute metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_spearman = spearmanr(all_targets, all_preds)[0]
        val_uncertainty = np.mean(all_uncertainties)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_spearman"].append(val_spearman)
        history["val_uncertainty"].append(val_uncertainty)

        scheduler.step()

        # Logging
        if epoch % 10 == 0 or val_spearman > best_spearman:
            print(
                f"Epoch {epoch:3d}: loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"spearman={val_spearman:.4f} uncertainty={val_uncertainty:.3f}"
            )

        # Best model checkpoint
        if val_spearman > best_spearman:
            best_spearman = val_spearman
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_spearman": val_spearman,
                    "val_loss": val_loss,
                    "config": config.__dict__,
                },
                output_dir / "best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # Final checkpoint
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_spearman": val_spearman,
        },
        output_dir / "final.pt",
    )

    # Save history (convert numpy types)
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history_serializable, f, indent=2)

    print("\n" + "=" * 70)
    print("MULTIMODAL FUSION TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nBest Spearman: {best_spearman:.4f}")
    print(f"Checkpoint: {output_dir / 'best.pt'}")

    # Compare with baselines
    print("\n" + "=" * 70)
    print("Results Comparison")
    print("=" * 70)
    print("  VAE-ProTherm alone: 0.64")
    print("  MLP Refiner:        0.78")
    print(f"  Multimodal Fusion:  {best_spearman:.2f}")

    improvement = (best_spearman - 0.64) / 0.64 * 100
    print(f"\n  Improvement over VAE baseline: {improvement:+.1f}%")

    if best_spearman >= 0.80:
        print("\n  TARGET ACHIEVED: Spearman >= 0.80")

    return model, best_spearman


if __name__ == "__main__":
    train_multimodal_fusion()
