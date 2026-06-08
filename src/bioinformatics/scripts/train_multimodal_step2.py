#!/usr/bin/env python3
"""Step 2: Single VAE on Combined Raw Datasets.

Trains a single VAE on combined data from S669, ProTherm, and ProteinGym,
then evaluates correlations to verify multimodality.

This tests whether a single VAE can learn from heterogeneous data sources.
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
from src.bioinformatics.data.proteingym_loader import ProteinGymLoader
from src.bioinformatics.data.protherm_loader import ProThermLoader
from src.bioinformatics.data.s669_loader import S669Loader
from src.bioinformatics.models.ddg_mlp_refiner import DDGMLPRefiner, RefinerConfig
from src.bioinformatics.training.deterministic import set_deterministic_mode


@dataclass
class CombinedVAEConfig:
    """Configuration for combined dataset VAE."""

    input_dim: int = 14
    hidden_dim: int = 128
    latent_dim: int = 32

    # Training
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-4
    kl_weight: float = 0.1
    patience: int = 20

    # Dataset sampling
    max_protherm: int = 177
    max_s669: int = 52
    max_proteingym: int = 10000


class CombinedVAE(nn.Module):
    """VAE trained on combined heterogeneous datasets."""

    def __init__(self, config: CombinedVAEConfig):
        super().__init__()
        self.config = config

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
        )

        self.fc_mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim, config.latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
        )

        # Reconstruction head
        self.recon_head = nn.Linear(config.hidden_dim, config.input_dim)

        # DDG/fitness prediction head
        self.ddg_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

        # Source-specific heads (for multi-task learning)
        self.source_heads = nn.ModuleDict(
            {
                "protherm": nn.Linear(config.hidden_dim, 1),
                "s669": nn.Linear(config.hidden_dim, 1),
                "proteingym": nn.Linear(config.hidden_dim, 1),
            }
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

    def decode(self, z: torch.Tensor, source: str = None) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.decoder(z)
        recon = self.recon_head(h)

        # Use source-specific head if available
        ddg = self.source_heads[source](h) if source is not None and source in self.source_heads else self.ddg_head(h)

        return recon, ddg

    def forward(self, x: torch.Tensor, source: str = None) -> dict:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon, ddg = self.decode(z, source)

        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "recon": recon,
            "ddg_pred": ddg.squeeze(-1),
        }

    def loss(self, x: torch.Tensor, target: torch.Tensor, source: str = None, kl_weight: float = 0.1) -> dict:
        out = self.forward(x, source)

        # Reconstruction loss
        recon_loss = F.mse_loss(out["recon"], x)

        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())

        # Target prediction loss
        target_loss = F.mse_loss(out["ddg_pred"], target)

        total = recon_loss + kl_weight * kl_loss + target_loss

        return {
            "loss": total,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "target_loss": target_loss,
        }


class CombinedDataset(Dataset):
    """Combined dataset from multiple sources."""

    def __init__(
        self,
        protherm_records: list,
        s669_records: list,
        proteingym_records: list,
    ):
        self.features = []
        self.targets = []
        self.sources = []

        # Process ProTherm (DDG values)
        for record in protherm_records:
            feat = compute_features(record.wild_type, record.mutant)
            self.features.append(feat.to_array(include_hyperbolic=False))
            self.targets.append(record.ddg)
            self.sources.append("protherm")

        # Process S669 (DDG values)
        for record in s669_records:
            feat = compute_features(record.wild_type, record.mutant)
            self.features.append(feat.to_array(include_hyperbolic=False))
            self.targets.append(record.ddg)
            self.sources.append("s669")

        # Process ProteinGym (fitness values - different scale!)
        for record in proteingym_records:
            feat = compute_features(record.wild_type, record.mutant)
            self.features.append(feat.to_array(include_hyperbolic=False))
            # Note: fitness is NOT DDG, but we'll use it for learning representations
            self.targets.append(record.fitness)
            self.sources.append("proteingym")

        self.features = np.array(self.features, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

        print(f"  Combined dataset: {len(self.features)} samples")
        print(f"    ProTherm: {sum(1 for s in self.sources if s == 'protherm')}")
        print(f"    S669: {sum(1 for s in self.sources if s == 's669')}")
        print(f"    ProteinGym: {sum(1 for s in self.sources if s == 'proteingym')}")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "x": torch.from_numpy(self.features[idx]),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32),
            "source": self.sources[idx],
        }


class SourceSubset(Dataset):
    """Subset of combined dataset filtered by source."""

    def __init__(self, combined_dataset: CombinedDataset, source: str):
        self.indices = [i for i, s in enumerate(combined_dataset.sources) if s == source]
        self.combined = combined_dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.combined[self.indices[idx]]


def train_step2():
    """Train VAE on combined raw datasets."""
    print("=" * 70)
    print("STEP 2: Single VAE on Combined Raw Datasets")
    print("=" * 70)

    set_deterministic_mode(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = CombinedVAEConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/multimodal_step2_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # Load datasets
    print("\n[1] Loading datasets...")

    # ProTherm
    protherm_loader = ProThermLoader()
    protherm_db = protherm_loader.load_curated()
    protherm_records = protherm_db.records[: config.max_protherm]
    print(f"  ProTherm: {len(protherm_records)} mutations")

    # S669
    s669_loader = S669Loader()
    try:
        s669_records = s669_loader.load_from_csv()[: config.max_s669]
    except Exception as e:
        print(f"  Warning: Could not load S669 full data: {e}")
        s669_records = []
    print(f"  S669: {len(s669_records)} mutations")

    # ProteinGym
    proteingym_loader = ProteinGymLoader()
    proteingym_records = proteingym_loader.load_all(max_per_protein=500)
    np.random.shuffle(proteingym_records)
    proteingym_records = proteingym_records[: config.max_proteingym]
    print(f"  ProteinGym: {len(proteingym_records)} mutations")

    # Create combined dataset
    print("\n[2] Creating combined dataset...")
    combined_dataset = CombinedDataset(protherm_records, s669_records, proteingym_records)

    # Split - keep ProTherm separate for evaluation
    protherm_subset = SourceSubset(combined_dataset, "protherm")
    n_protherm_val = int(len(protherm_subset) * 0.2)
    len(protherm_subset) - n_protherm_val

    # For training, use all sources
    # For validation, only use ProTherm (our target task)
    train_indices = []
    val_indices = []

    for i, source in enumerate(combined_dataset.sources):
        if source == "protherm":
            if len(val_indices) < n_protherm_val:
                val_indices.append(i)
            else:
                train_indices.append(i)
        else:
            train_indices.append(i)

    from torch.utils.data import Subset

    train_ds = Subset(combined_dataset, train_indices)
    val_ds = Subset(combined_dataset, val_indices)

    print(f"  Train: {len(train_ds)} (includes all sources)")
    print(f"  Val: {len(val_ds)} (ProTherm only, for DDG evaluation)")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # Create VAE
    print("\n[3] Creating Combined VAE...")
    vae = CombinedVAE(config).to(device)
    trainable = sum(p.numel() for p in vae.parameters())
    print(f"  Parameters: {trainable:,}")

    optimizer = torch.optim.AdamW(vae.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Train
    print("\n" + "=" * 70)
    print("[4] Training Combined VAE")
    print("=" * 70)

    best_spearman = -1
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_spearman": []}

    for epoch in range(config.epochs):
        # Train
        vae.train()
        train_losses = []

        for batch in train_loader:
            x = batch["x"].to(device)
            target = batch["target"].to(device)
            sources = batch["source"]

            # Multi-task training: use source-specific heads
            optimizer.zero_grad()

            # Process each source type separately for proper head selection
            total_loss = 0
            for source in set(sources):
                mask = [s == source for s in sources]
                if sum(mask) > 0:
                    x_source = x[mask]
                    target_source = target[mask]
                    loss_dict = vae.loss(x_source, target_source, source, config.kl_weight)
                    total_loss = total_loss + loss_dict["loss"] * sum(mask)

            total_loss = total_loss / len(sources)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()

            train_losses.append(total_loss.item())

        # Validate (ProTherm only)
        vae.eval()
        val_losses = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                target = batch["target"].to(device)

                loss_dict = vae.loss(x, target, "protherm", config.kl_weight)
                val_losses.append(loss_dict["loss"].item())

                out = vae(x, "protherm")
                all_preds.extend(out["ddg_pred"].cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_spearman = spearmanr(all_targets, all_preds)[0]

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_spearman"].append(float(val_spearman))

        scheduler.step()

        if epoch % 10 == 0 or val_spearman > best_spearman:
            print(f"Epoch {epoch:3d}: loss={train_loss:.4f} val_loss={val_loss:.4f} spearman={val_spearman:.4f}")

        if val_spearman > best_spearman:
            best_spearman = val_spearman
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": vae.state_dict(),
                    "epoch": epoch,
                    "val_spearman": val_spearman,
                    "config": config.__dict__,
                },
                output_dir / "combined_vae_best.pt",
            )
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\nCombined VAE Best Spearman: {best_spearman:.4f}")

    # ========================================
    # Train MLP Refiner
    # ========================================
    print("\n" + "=" * 70)
    print("[5] Training MLP Refiner on Combined VAE Embeddings")
    print("=" * 70)

    # Load best VAE
    ckpt = torch.load(output_dir / "combined_vae_best.pt", map_location=device, weights_only=False)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.eval()

    # Extract embeddings for ProTherm validation set
    class VAEEmbeddingDataset(Dataset):
        def __init__(self, base_dataset, vae, device, source="protherm"):
            self.embeddings = []
            self.vae_preds = []
            self.labels = []

            vae.eval()
            with torch.no_grad():
                for i in range(len(base_dataset)):
                    item = base_dataset[i]
                    x = item["x"].unsqueeze(0).to(device)

                    out = vae(x, source)
                    self.embeddings.append(out["mu"].cpu().squeeze(0))
                    self.vae_preds.append(out["ddg_pred"].cpu().item())
                    self.labels.append(item["target"].item() if hasattr(item["target"], "item") else item["target"])

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

    # Create ProTherm-only datasets for refiner
    protherm_full = SourceSubset(combined_dataset, "protherm")
    n_val = int(len(protherm_full) * 0.2)
    n_train = len(protherm_full) - n_val
    train_protherm, val_protherm = random_split(
        protherm_full, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_emb_ds = VAEEmbeddingDataset(train_protherm, vae, device)
    val_emb_ds = VAEEmbeddingDataset(val_protherm, vae, device)

    print(f"  Embedding dim: {train_emb_ds.embedding_dim}")
    print(f"  Train: {len(train_emb_ds)}, Val: {len(val_emb_ds)}")

    # Train refiner
    refiner_config = RefinerConfig(
        latent_dim=config.latent_dim,
        hidden_dims=[64, 64, 32],
        dropout=0.1,
        use_residual=True,
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
                "combined_vae": history,
                "mlp_refiner": refiner_history,
            },
            f,
            indent=2,
        )

    # ========================================
    # Cross-source correlation analysis
    # ========================================
    print("\n" + "=" * 70)
    print("[6] Cross-Source Correlation Analysis")
    print("=" * 70)

    # Evaluate on each source
    vae.eval()
    results = {}

    for source in ["protherm", "s669", "proteingym"]:
        source_subset = SourceSubset(combined_dataset, source)
        if len(source_subset) == 0:
            continue

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for i in range(len(source_subset)):
                item = source_subset[i]
                x = item["x"].unsqueeze(0).to(device)
                target = item["target"].item()

                out = vae(x, source)
                all_preds.append(out["ddg_pred"].cpu().item())
                all_targets.append(target)

        spearman = spearmanr(all_targets, all_preds)[0]
        pearson = pearsonr(all_targets, all_preds)[0]

        results[source] = {
            "n_samples": len(source_subset),
            "spearman": float(spearman),
            "pearson": float(pearson),
        }

        print(f"\n  {source.upper()}:")
        print(f"    N={len(source_subset)}, Spearman={spearman:.4f}, Pearson={pearson:.4f}")

    # Save results
    results["combined_vae_best_spearman"] = float(best_spearman)
    results["refiner_best_spearman"] = float(best_refiner_spearman)

    with open(output_dir / "correlation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("STEP 2 COMPLETE")
    print("=" * 70)
    print("\nResults:")
    print(f"  Combined VAE Spearman (ProTherm): {best_spearman:.4f}")
    print(f"  + MLP Refiner Spearman:           {best_refiner_spearman:.4f}")
    print("\nBaselines:")
    print("  VAE-ProTherm alone:               0.64")
    print("  VAE-ProTherm + MLP Refiner:       0.78")
    print("  Step 1 Meta-VAE + Refiner:        0.63")
    print(f"\nCheckpoints: {output_dir}")

    return {
        "combined_vae_spearman": best_spearman,
        "refiner_spearman": best_refiner_spearman,
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":
    train_step2()
