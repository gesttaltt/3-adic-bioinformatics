#!/usr/bin/env python3
"""Step 1: Meta-VAE over Three Specialist VAE Embeddings.

Takes the frozen embeddings from VAE-S669, VAE-ProTherm, and VAE-Wide,
trains a new VAE on their concatenated embeddings, then applies MLP refiner.

This tests whether a VAE can learn useful structure from the combined
latent spaces of specialist models.
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
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parents[3]))

from src.bioinformatics.data.preprocessing import compute_features
from src.bioinformatics.data.protherm_loader import ProThermLoader
from src.bioinformatics.models.ddg_mlp_refiner import DDGMLPRefiner, RefinerConfig
from src.bioinformatics.models.ddg_vae import DDGVAE
from src.bioinformatics.training.deterministic import set_deterministic_mode


@dataclass
class MetaVAEConfig:
    """Configuration for Meta-VAE."""

    # Input dimensions from specialist VAEs
    s669_dim: int = 16
    protherm_dim: int = 32
    wide_dim: int = 64

    # Meta-VAE architecture
    hidden_dim: int = 128
    latent_dim: int = 32

    # Training
    epochs: int = 200
    batch_size: int = 16
    learning_rate: float = 1e-4
    kl_weight: float = 0.1
    patience: int = 30


class MetaVAE(nn.Module):
    """VAE that learns from concatenated specialist embeddings."""

    def __init__(self, config: MetaVAEConfig):
        super().__init__()
        self.config = config

        input_dim = config.s669_dim + config.protherm_dim + config.wide_dim  # 112

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
        )

        self.fc_mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim, config.latent_dim)

        # Decoder (reconstruct embeddings + predict DDG)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )

        # Reconstruction head
        self.recon_head = nn.Linear(config.hidden_dim, input_dim)

        # DDG prediction head
        self.ddg_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-10, 2)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.decoder(z)
        recon = self.recon_head(h)
        ddg = self.ddg_head(h)
        return recon, ddg

    def forward(self, x: torch.Tensor) -> dict:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon, ddg = self.decode(z)

        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "recon": recon,
            "ddg_pred": ddg.squeeze(-1),
        }

    def loss(self, x: torch.Tensor, ddg_target: torch.Tensor, kl_weight: float = 0.1) -> dict:
        out = self.forward(x)

        # Reconstruction loss
        recon_loss = F.mse_loss(out["recon"], x)

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())

        # DDG prediction loss
        ddg_loss = F.mse_loss(out["ddg_pred"], ddg_target)

        total = recon_loss + kl_weight * kl_loss + ddg_loss

        return {
            "loss": total,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "ddg_loss": ddg_loss,
        }


class TripleEmbeddingDataset(Dataset):
    """Dataset with embeddings from all three specialist VAEs."""

    def __init__(
        self,
        records: list,
        vae_s669: DDGVAE,
        vae_protherm: DDGVAE,
        vae_wide: DDGVAE,
        device: str = "cuda",
    ):
        self.embeddings_concat = []
        self.embeddings_s669 = []
        self.embeddings_protherm = []
        self.embeddings_wide = []
        self.ddg_values = []
        self.vae_preds = []  # From ProTherm VAE (best baseline)

        vae_s669 = vae_s669.to(device).eval()
        vae_protherm = vae_protherm.to(device).eval()
        vae_wide = vae_wide.to(device).eval()

        with torch.no_grad():
            for record in records:
                # Compute features
                feat = compute_features(record.wild_type, record.mutant)
                basic_arr = feat.to_array(include_hyperbolic=False)

                # S669 features (14-dim)
                x_s669 = torch.tensor(basic_arr, dtype=torch.float32).unsqueeze(0).to(device)

                # ProTherm features (20-dim, padded)
                protherm_arr = np.pad(basic_arr, (0, 6), mode="constant")
                x_protherm = torch.tensor(protherm_arr, dtype=torch.float32).unsqueeze(0).to(device)

                # Wide features (14-dim)
                x_wide = torch.tensor(basic_arr, dtype=torch.float32).unsqueeze(0).to(device)

                # Get embeddings
                out_s669 = vae_s669(x_s669)
                out_protherm = vae_protherm(x_protherm)
                out_wide = vae_wide(x_wide)

                emb_s669 = out_s669["mu"].cpu().squeeze(0)
                emb_protherm = out_protherm["mu"].cpu().squeeze(0)
                emb_wide = out_wide["mu"].cpu().squeeze(0)

                # Concatenate
                concat = torch.cat([emb_s669, emb_protherm, emb_wide], dim=0)

                self.embeddings_concat.append(concat)
                self.embeddings_s669.append(emb_s669)
                self.embeddings_protherm.append(emb_protherm)
                self.embeddings_wide.append(emb_wide)
                self.ddg_values.append(record.ddg)
                self.vae_preds.append(out_protherm["ddg_pred"].cpu().item())

        self.embeddings_concat = torch.stack(self.embeddings_concat)
        self.embeddings_s669 = torch.stack(self.embeddings_s669)
        self.embeddings_protherm = torch.stack(self.embeddings_protherm)
        self.embeddings_wide = torch.stack(self.embeddings_wide)
        self.ddg_values = torch.tensor(self.ddg_values, dtype=torch.float32)
        self.vae_preds = torch.tensor(self.vae_preds, dtype=torch.float32)

    def __len__(self):
        return len(self.ddg_values)

    def __getitem__(self, idx):
        return {
            "concat": self.embeddings_concat[idx],
            "s669": self.embeddings_s669[idx],
            "protherm": self.embeddings_protherm[idx],
            "wide": self.embeddings_wide[idx],
            "ddg": self.ddg_values[idx],
            "vae_pred": self.vae_preds[idx],
        }


def train_step1():
    """Train Meta-VAE on specialist embeddings, then apply MLP refiner."""
    print("=" * 70)
    print("STEP 1: Meta-VAE over Specialist Embeddings")
    print("=" * 70)

    set_deterministic_mode(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = MetaVAEConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/multimodal_step1_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Load specialist VAEs
    print("\n[1] Loading specialist VAEs...")

    vae_s669 = DDGVAE.create_s669_variant(use_hyperbolic=False)
    ckpt = torch.load(
        "outputs/ddg_vae_training_20260129_212316/vae_s669/best.pt", map_location=device, weights_only=False
    )
    vae_s669.load_state_dict(ckpt["model_state_dict"])
    print("  Loaded VAE-S669 (latent_dim=16)")

    vae_protherm = DDGVAE.create_protherm_variant(use_hyperbolic=False)
    ckpt = torch.load(
        "outputs/ddg_vae_training_20260129_212316/vae_protherm/best.pt", map_location=device, weights_only=False
    )
    vae_protherm.load_state_dict(ckpt["model_state_dict"])
    print("  Loaded VAE-ProTherm (latent_dim=32)")

    vae_wide = DDGVAE.create_wide_variant(use_hyperbolic=False)
    ckpt = torch.load("outputs/vae_wide_filtered_20260129_220019/best.pt", map_location=device, weights_only=False)
    vae_wide.load_state_dict(ckpt["model_state_dict"])
    print("  Loaded VAE-Wide (latent_dim=64)")

    # Load ProTherm data
    print("\n[2] Creating triple embedding dataset...")
    loader = ProThermLoader()
    db = loader.load_curated()
    records = db.records
    print(f"  Loaded {len(records)} mutations")

    dataset = TripleEmbeddingDataset(records, vae_s669, vae_protherm, vae_wide, device)
    print(f"  Concatenated embedding dim: {dataset.embeddings_concat.shape[1]}")

    # Split
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    print(f"  Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # Create Meta-VAE
    print("\n[3] Creating Meta-VAE...")
    meta_vae = MetaVAE(config).to(device)
    trainable = sum(p.numel() for p in meta_vae.parameters())
    print(f"  Parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(meta_vae.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Train Meta-VAE
    print("\n" + "=" * 70)
    print("[4] Training Meta-VAE")
    print("=" * 70)

    best_spearman = -1
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_spearman": [], "kl_loss": [], "recon_loss": []}

    for epoch in range(config.epochs):
        # Train
        meta_vae.train()
        train_losses = []

        for batch in train_loader:
            x = batch["concat"].to(device)
            ddg = batch["ddg"].to(device)

            optimizer.zero_grad()
            loss_dict = meta_vae.loss(x, ddg, config.kl_weight)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(meta_vae.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss_dict["loss"].item())

        # Validate
        meta_vae.eval()
        val_losses = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["concat"].to(device)
                ddg = batch["ddg"].to(device)

                loss_dict = meta_vae.loss(x, ddg, config.kl_weight)
                val_losses.append(loss_dict["loss"].item())

                out = meta_vae(x)
                all_preds.extend(out["ddg_pred"].cpu().numpy())
                all_targets.extend(ddg.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_spearman = spearmanr(all_targets, all_preds)[0]

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_spearman"].append(float(val_spearman))

        scheduler.step()

        if epoch % 20 == 0 or val_spearman > best_spearman:
            print(f"Epoch {epoch:3d}: loss={train_loss:.4f} val_loss={val_loss:.4f} spearman={val_spearman:.4f}")

        if val_spearman > best_spearman:
            best_spearman = val_spearman
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": meta_vae.state_dict(),
                    "epoch": epoch,
                    "val_spearman": val_spearman,
                    "config": config.__dict__,
                },
                output_dir / "meta_vae_best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nMeta-VAE Best Spearman: {best_spearman:.4f}")

    # ========================================
    # Now train MLP Refiner on Meta-VAE embeddings
    # ========================================
    print("\n" + "=" * 70)
    print("[5] Training MLP Refiner on Meta-VAE Embeddings")
    print("=" * 70)

    # Load best Meta-VAE
    ckpt = torch.load(output_dir / "meta_vae_best.pt", map_location=device, weights_only=False)
    meta_vae.load_state_dict(ckpt["model_state_dict"])
    meta_vae.eval()

    # Extract Meta-VAE embeddings
    class MetaVAEEmbeddingDataset(Dataset):
        def __init__(self, base_dataset, meta_vae, device):
            self.embeddings = []
            self.vae_preds = []
            self.labels = []

            meta_vae.eval()
            with torch.no_grad():
                for i in range(len(base_dataset)):
                    item = base_dataset[i]
                    x = item["concat"].unsqueeze(0).to(device)

                    out = meta_vae(x)
                    self.embeddings.append(out["mu"].cpu().squeeze(0))
                    # Ensure vae_pred is a scalar
                    vae_pred = out["ddg_pred"].cpu()
                    if vae_pred.dim() > 0:
                        vae_pred = vae_pred.squeeze()
                    self.vae_preds.append(vae_pred.item() if vae_pred.numel() == 1 else vae_pred[0].item())
                    self.labels.append(item["ddg"].item() if hasattr(item["ddg"], "item") else item["ddg"])

            self.embeddings = torch.stack(self.embeddings)
            self.vae_preds = torch.tensor(self.vae_preds, dtype=torch.float32)
            self.labels = torch.tensor(self.labels, dtype=torch.float32)

        @property
        def embedding_dim(self):
            return self.embeddings.shape[1]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "embedding": self.embeddings[idx],
                "vae_pred": self.vae_preds[idx],
                "label": self.labels[idx],
            }

    # Create embedding datasets
    train_emb_ds = MetaVAEEmbeddingDataset(train_ds, meta_vae, device)
    val_emb_ds = MetaVAEEmbeddingDataset(val_ds, meta_vae, device)

    print(f"  Meta-VAE embedding dim: {train_emb_ds.embedding_dim}")

    # Train MLP Refiner
    refiner_config = RefinerConfig(
        latent_dim=config.latent_dim,
        hidden_dims=[64, 64, 32],
        dropout=0.1,
        use_residual=True,
        initial_residual_weight=0.3,
    )

    refiner = DDGMLPRefiner(config=refiner_config).to(device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    train_loader = DataLoader(train_emb_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_emb_ds, batch_size=16, shuffle=False)

    best_refiner_spearman = -1
    refiner_history = {"train_loss": [], "val_loss": [], "val_spearman": []}

    for epoch in range(100):
        # Train
        refiner.train()
        train_losses = []

        for batch in train_loader:
            z = batch["embedding"].to(device)
            vae_pred = batch["vae_pred"].to(device).unsqueeze(-1)
            y = batch["label"].to(device).unsqueeze(-1)

            optimizer.zero_grad()
            loss_dict = refiner.loss(z, y, vae_pred)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss_dict["loss"].item())

        # Validate
        refiner.eval()
        val_losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                z = batch["embedding"].to(device)
                vae_pred = batch["vae_pred"].to(device).unsqueeze(-1)
                y = batch["label"].to(device).unsqueeze(-1)

                loss_dict = refiner.loss(z, y, vae_pred)
                val_losses.append(loss_dict["loss"].item())

                out = refiner(z, vae_pred)
                all_preds.extend(out["ddg_pred"].cpu().numpy().flatten())
                all_labels.extend(y.cpu().numpy().flatten())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_spearman = spearmanr(all_preds, all_labels)[0]

        refiner_history["train_loss"].append(float(train_loss))
        refiner_history["val_loss"].append(float(val_loss))
        refiner_history["val_spearman"].append(float(val_spearman))

        scheduler.step()

        if epoch % 20 == 0 or val_spearman > best_refiner_spearman:
            rw = torch.sigmoid(refiner.residual_weight).item()
            print(
                f"Epoch {epoch:3d}: loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"spearman={val_spearman:.4f} res_w={rw:.3f}"
            )

        if val_spearman > best_refiner_spearman:
            best_refiner_spearman = val_spearman
            torch.save(
                {
                    "model_state_dict": refiner.state_dict(),
                    "epoch": epoch,
                    "val_spearman": val_spearman,
                },
                output_dir / "mlp_refiner_best.pt",
            )

    # Save histories
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(
            {
                "meta_vae": history,
                "mlp_refiner": refiner_history,
            },
            f,
            indent=2,
        )

    # ========================================
    # Correlation Analysis
    # ========================================
    print("\n" + "=" * 70)
    print("[6] Multimodality Correlation Analysis")
    print("=" * 70)

    # Load best models
    ckpt = torch.load(output_dir / "meta_vae_best.pt", map_location=device, weights_only=False)
    meta_vae.load_state_dict(ckpt["model_state_dict"])
    ckpt = torch.load(output_dir / "mlp_refiner_best.pt", map_location=device, weights_only=False)
    refiner.load_state_dict(ckpt["model_state_dict"])

    meta_vae.eval()
    refiner.eval()

    # Get predictions on full validation set
    all_preds = {"meta_vae": [], "refiner": [], "s669": [], "protherm": [], "wide": []}
    all_targets = []

    with torch.no_grad():
        for item in val_ds:
            x_concat = item["concat"].unsqueeze(0).to(device)
            ddg = item["ddg"].item()

            # Meta-VAE prediction
            meta_out = meta_vae(x_concat)
            all_preds["meta_vae"].append(meta_out["ddg_pred"].cpu().item())

            # Refiner prediction
            refiner_out = refiner(meta_out["mu"], meta_out["ddg_pred"].unsqueeze(-1))
            all_preds["refiner"].append(refiner_out["ddg_pred"].cpu().item())

            # Individual VAE predictions (from stored data)
            # Get features and run through individual VAEs
            all_targets.append(ddg)

    # Compute correlations
    results = {
        "meta_vae_spearman": float(spearmanr(all_targets, all_preds["meta_vae"])[0]),
        "meta_vae_pearson": float(pearsonr(all_targets, all_preds["meta_vae"])[0]),
        "refiner_spearman": float(spearmanr(all_targets, all_preds["refiner"])[0]),
        "refiner_pearson": float(pearsonr(all_targets, all_preds["refiner"])[0]),
    }

    print(
        f"\n  Meta-VAE alone:      Spearman={results['meta_vae_spearman']:.4f}, Pearson={results['meta_vae_pearson']:.4f}"
    )
    print(
        f"  Meta-VAE + Refiner:  Spearman={results['refiner_spearman']:.4f}, Pearson={results['refiner_pearson']:.4f}"
    )

    # Save results
    with open(output_dir / "correlation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 1 COMPLETE")
    print("=" * 70)
    print("\nResults:")
    print(f"  Meta-VAE Spearman:           {best_spearman:.4f}")
    print(f"  Meta-VAE + Refiner Spearman: {best_refiner_spearman:.4f}")
    print("\nBaselines:")
    print("  VAE-ProTherm alone:          0.64")
    print("  VAE-ProTherm + MLP Refiner:  0.78")
    print(f"\nCheckpoints: {output_dir}")

    return {
        "meta_vae_spearman": best_spearman,
        "refiner_spearman": best_refiner_spearman,
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":
    train_step1()
