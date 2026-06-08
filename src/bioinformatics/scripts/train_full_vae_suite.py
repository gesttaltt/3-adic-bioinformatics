#!/usr/bin/env python3
"""Train Complete VAE Suite on FULL Datasets.

This script ensures all VAEs and MLP refiners are trained on their FULL datasets:
1. VAE-S669: Full 669 mutations (NOT the 52 curated subset)
2. VAE-ProTherm: Full 177 curated mutations
3. VAE-Wide: ProteinGym (up to 100K samples)
4. MLP Refiners for each VAE

After training, a transformer can attend over all activations.
"""

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from src.bioinformatics.data.preprocessing import compute_features
from src.bioinformatics.data.protherm_loader import ProThermLoader
from src.bioinformatics.data.s669_loader import S669Loader


@dataclass
class SuiteConfig:
    """Configuration for full VAE suite training."""

    # VAE architecture
    vae_hidden_dim: int = 128
    vae_latent_dim: int = 32
    vae_dropout: float = 0.1

    # MLP Refiner architecture
    refiner_hidden_dims: list[int] = None

    # Training
    vae_epochs: int = 200
    refiner_epochs: int = 150
    batch_size: int = 32
    vae_lr: float = 1e-4
    refiner_lr: float = 5e-5
    weight_decay: float = 1e-4
    patience: int = 30

    def __post_init__(self):
        if self.refiner_hidden_dims is None:
            self.refiner_hidden_dims = [64, 64, 32]


class DDGVAE(nn.Module):
    """VAE for DDG prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 2)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        pred = self.decoder(z).squeeze(-1)
        return {"mu": mu, "logvar": logvar, "z": z, "pred": pred}


class MLPRefiner(nn.Module):
    """MLP that refines VAE predictions using latent embeddings."""

    def __init__(self, latent_dim: int, hidden_dims: list[int] = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64, 32]

        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.SiLU(),
                    nn.LayerNorm(h_dim),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = h_dim

        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, 1)
        self.residual_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, mu: torch.Tensor, vae_pred: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.mlp(mu)
        delta = self.head(h).squeeze(-1)
        w = torch.sigmoid(self.residual_weight)
        refined = vae_pred + w * delta
        return {"pred": refined, "delta": delta, "weight": w}


def load_s669_full() -> tuple[np.ndarray, np.ndarray]:
    """Load FULL S669 dataset (669 mutations)."""
    loader = S669Loader()
    records = loader.load_from_csv()  # Uses s669_full.csv

    X, y = [], []
    for record in records:
        feat = compute_features(record.wild_type, record.mutant)
        X.append(feat.to_array(include_hyperbolic=False))
        y.append(record.ddg)

    print(f"  Loaded S669 FULL: {len(X)} samples")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_protherm_full() -> tuple[np.ndarray, np.ndarray]:
    """Load FULL ProTherm dataset (177 mutations)."""
    loader = ProThermLoader()
    db = loader.load_curated()

    X, y = [], []
    for record in db.records:
        feat = compute_features(record.wild_type, record.mutant)
        X.append(feat.to_array(include_hyperbolic=False))
        y.append(record.ddg)

    print(f"  Loaded ProTherm FULL: {len(X)} samples")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_proteingym(max_samples: int = 100000) -> tuple[np.ndarray, np.ndarray]:
    """Load ProteinGym dataset."""
    try:
        from src.bioinformatics.data.proteingym_loader import ProteinGymLoader

        loader = ProteinGymLoader(
            data_dir=project_root / "data" / "bioinformatics" / "ddg" / "proteingym" / "DMS_ProteinGym_substitutions"
        )
        dataset = loader.create_dataset(max_records=max_samples, use_fitness_as_label=True)
        X, y = dataset.get_arrays()
        print(f"  Loaded ProteinGym: {len(X)} samples")
        return X, y
    except Exception as e:
        print(f"  ProteinGym loading failed: {e}")
        return None, None


def train_vae(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    name: str,
    config: SuiteConfig,
) -> tuple[DDGVAE, dict]:
    """Train a VAE on dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DDGVAE(
        input_dim=X_train.shape[1],
        hidden_dim=config.vae_hidden_dim,
        latent_dim=config.vae_latent_dim,
        dropout=config.vae_dropout,
    ).to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.vae_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.vae_epochs)

    best_val_corr = -1
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_spearman": []}

    for epoch in range(config.vae_epochs):
        model.train()
        epoch_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)

            # Reconstruction + KL loss
            recon_loss = F.mse_loss(out["pred"], batch_y)
            kl_loss = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
            loss = recon_loss + 0.01 * kl_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_loss = F.mse_loss(val_out["pred"], y_val_t).item()
            val_corr = spearmanr(y_val_t.cpu().numpy(), val_out["pred"].cpu().numpy())[0]

        history["train_loss"].append(epoch_loss / len(loader))
        history["val_loss"].append(val_loss)
        history["val_spearman"].append(val_corr)

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"    {name}: Early stopping at epoch {epoch}")
                break

        if (epoch + 1) % 25 == 0:
            print(f"    {name} epoch {epoch + 1}: val_spearman={val_corr:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    print(f"  {name}: Best Spearman = {best_val_corr:.4f}")
    return model, {"best_spearman": best_val_corr, "history": history}


def train_refiner(
    vae: DDGVAE,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    name: str,
    config: SuiteConfig,
) -> tuple[MLPRefiner, dict]:
    """Train MLP refiner for a VAE."""
    device = next(vae.parameters()).device
    vae.eval()

    # Get VAE embeddings
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    with torch.no_grad():
        train_vae_out = vae(X_train_t)
        val_vae_out = vae(X_val_t)

    train_mu = train_vae_out["mu"]
    train_vae_pred = train_vae_out["pred"]
    val_mu = val_vae_out["mu"]
    val_vae_pred = val_vae_out["pred"]

    # Create refiner
    refiner = MLPRefiner(
        latent_dim=config.vae_latent_dim,
        hidden_dims=config.refiner_hidden_dims,
        dropout=config.vae_dropout,
    ).to(device)

    dataset = TensorDataset(train_mu, train_vae_pred, y_train_t)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(refiner.parameters(), lr=config.refiner_lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.refiner_epochs)

    best_val_corr = -1
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_spearman": []}

    for epoch in range(config.refiner_epochs):
        refiner.train()
        epoch_loss = 0
        for batch_mu, batch_vae_pred, batch_y in loader:
            optimizer.zero_grad()
            out = refiner(batch_mu, batch_vae_pred)
            loss = F.mse_loss(out["pred"], batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Evaluate
        refiner.eval()
        with torch.no_grad():
            val_out = refiner(val_mu, val_vae_pred)
            val_corr = spearmanr(y_val_t.cpu().numpy(), val_out["pred"].cpu().numpy())[0]

        history["train_loss"].append(epoch_loss / len(loader))
        history["val_spearman"].append(val_corr)

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_state = {k: v.cpu().clone() for k, v in refiner.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"    {name} Refiner: Early stopping at epoch {epoch}")
                break

        if (epoch + 1) % 25 == 0:
            print(f"    {name} Refiner epoch {epoch + 1}: val_spearman={val_corr:.4f}")

    if best_state:
        refiner.load_state_dict(best_state)

    print(f"  {name} Refiner: Best Spearman = {best_val_corr:.4f}")
    return refiner, {"best_spearman": best_val_corr, "history": history}


def main():
    """Train complete VAE suite on full datasets."""
    print("=" * 70)
    print("TRAINING COMPLETE VAE SUITE ON FULL DATASETS")
    print("=" * 70)

    config = SuiteConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/full_vae_suite_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    np.random.seed(42)
    torch.manual_seed(42)

    results = {}

    # ========================================
    # 1. S669 FULL (669 samples)
    # ========================================
    print("\n" + "=" * 60)
    print("[1/3] S669 FULL (669 mutations)")
    print("=" * 60)

    X_s669, y_s669 = load_s669_full()
    s669_idx = np.random.permutation(len(X_s669))
    s669_split = int(0.8 * len(s669_idx))
    X_s669_train, y_s669_train = X_s669[s669_idx[:s669_split]], y_s669[s669_idx[:s669_split]]
    X_s669_val, y_s669_val = X_s669[s669_idx[s669_split:]], y_s669[s669_idx[s669_split:]]
    print(f"  Train: {len(X_s669_train)}, Val: {len(X_s669_val)}")

    print("\n  Training VAE-S669...")
    vae_s669, vae_s669_hist = train_vae(X_s669_train, y_s669_train, X_s669_val, y_s669_val, "VAE-S669", config)

    print("\n  Training MLP Refiner for S669...")
    refiner_s669, refiner_s669_hist = train_refiner(
        vae_s669, X_s669_train, y_s669_train, X_s669_val, y_s669_val, "S669", config
    )

    results["s669"] = {
        "n_samples": len(X_s669),
        "vae_spearman": vae_s669_hist["best_spearman"],
        "refiner_spearman": refiner_s669_hist["best_spearman"],
    }

    # Save S669 models
    (output_dir / "s669").mkdir(exist_ok=True)
    torch.save({"model_state_dict": vae_s669.state_dict(), **results["s669"]}, output_dir / "s669" / "vae.pt")
    torch.save({"model_state_dict": refiner_s669.state_dict(), **results["s669"]}, output_dir / "s669" / "refiner.pt")

    # ========================================
    # 2. ProTherm FULL (177 samples)
    # ========================================
    print("\n" + "=" * 60)
    print("[2/3] ProTherm FULL (177 mutations)")
    print("=" * 60)

    X_protherm, y_protherm = load_protherm_full()
    protherm_idx = np.random.permutation(len(X_protherm))
    protherm_split = int(0.8 * len(protherm_idx))
    X_protherm_train, y_protherm_train = (
        X_protherm[protherm_idx[:protherm_split]],
        y_protherm[protherm_idx[:protherm_split]],
    )
    X_protherm_val, y_protherm_val = (
        X_protherm[protherm_idx[protherm_split:]],
        y_protherm[protherm_idx[protherm_split:]],
    )
    print(f"  Train: {len(X_protherm_train)}, Val: {len(X_protherm_val)}")

    print("\n  Training VAE-ProTherm...")
    vae_protherm, vae_protherm_hist = train_vae(
        X_protherm_train, y_protherm_train, X_protherm_val, y_protherm_val, "VAE-ProTherm", config
    )

    print("\n  Training MLP Refiner for ProTherm...")
    refiner_protherm, refiner_protherm_hist = train_refiner(
        vae_protherm, X_protherm_train, y_protherm_train, X_protherm_val, y_protherm_val, "ProTherm", config
    )

    results["protherm"] = {
        "n_samples": len(X_protherm),
        "vae_spearman": vae_protherm_hist["best_spearman"],
        "refiner_spearman": refiner_protherm_hist["best_spearman"],
    }

    # Save ProTherm models
    (output_dir / "protherm").mkdir(exist_ok=True)
    torch.save(
        {"model_state_dict": vae_protherm.state_dict(), **results["protherm"]}, output_dir / "protherm" / "vae.pt"
    )
    torch.save(
        {"model_state_dict": refiner_protherm.state_dict(), **results["protherm"]},
        output_dir / "protherm" / "refiner.pt",
    )

    # ========================================
    # 3. ProteinGym (Wide)
    # ========================================
    print("\n" + "=" * 60)
    print("[3/3] ProteinGym (Wide - up to 100K)")
    print("=" * 60)

    X_wide, y_wide = load_proteingym(max_samples=100000)

    if X_wide is not None:
        wide_idx = np.random.permutation(len(X_wide))
        wide_split = int(0.8 * len(wide_idx))
        X_wide_train, y_wide_train = X_wide[wide_idx[:wide_split]], y_wide[wide_idx[:wide_split]]
        X_wide_val, y_wide_val = X_wide[wide_idx[wide_split:]], y_wide[wide_idx[wide_split:]]
        print(f"  Train: {len(X_wide_train)}, Val: {len(X_wide_val)}")

        print("\n  Training VAE-Wide...")
        vae_wide, vae_wide_hist = train_vae(X_wide_train, y_wide_train, X_wide_val, y_wide_val, "VAE-Wide", config)

        print("\n  Training MLP Refiner for Wide...")
        refiner_wide, refiner_wide_hist = train_refiner(
            vae_wide, X_wide_train, y_wide_train, X_wide_val, y_wide_val, "Wide", config
        )

        results["wide"] = {
            "n_samples": len(X_wide),
            "vae_spearman": vae_wide_hist["best_spearman"],
            "refiner_spearman": refiner_wide_hist["best_spearman"],
        }

        # Save Wide models
        (output_dir / "wide").mkdir(exist_ok=True)
        torch.save({"model_state_dict": vae_wide.state_dict(), **results["wide"]}, output_dir / "wide" / "vae.pt")
        torch.save(
            {"model_state_dict": refiner_wide.state_dict(), **results["wide"]}, output_dir / "wide" / "refiner.pt"
        )
    else:
        results["wide"] = {"status": "skipped"}

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)

    print(f"""
| Dataset   | Samples | VAE Spearman | Refiner Spearman |
|-----------|--------:|:------------:|:----------------:|
| S669      | {results["s669"]["n_samples"]:>6} | {results["s669"]["vae_spearman"]:.4f}       | {results["s669"]["refiner_spearman"]:.4f}           |
| ProTherm  | {results["protherm"]["n_samples"]:>6} | {results["protherm"]["vae_spearman"]:.4f}       | {results["protherm"]["refiner_spearman"]:.4f}           |
| Wide      | {results.get("wide", {}).get("n_samples", "N/A"):>6} | {results.get("wide", {}).get("vae_spearman", "N/A")}       | {results.get("wide", {}).get("refiner_spearman", "N/A")}           |
""")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nAll models saved to: {output_dir}")
    print("\nNext step: Run transformer attention over all VAE+Refiner activations")

    return results


if __name__ == "__main__":
    main()
