"""Run best VAE configurations on real HIV data.

Uses Stanford HIVDB drug resistance data:
- PI: 2,171 records (Protease Inhibitors)
- NRTI: 1,867 records
- NNRTI: 2,270 records
- INI: 846 records (Integrase Inhibitors)

Total: ~7,154 records with real drug resistance phenotypes.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class Config:
    input_dim: int = 99  # Will be set based on data
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64, 32])
    batch_size: int = 32
    epochs: int = 100
    lr: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    use_rank: bool = True
    use_contrast: bool = True
    ranking_weight: float = 0.3
    contrastive_weight: float = 0.1


class VAE(nn.Module):
    """VAE with ranking and contrastive losses."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Encoder
        layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.1)])
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, cfg.latent_dim)

        # Decoder
        layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.1)])
            in_dim = h
        layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        x_recon = self.decoder(z)
        return {"x_recon": x_recon, "mu": mu, "logvar": logvar, "z": z}


def compute_loss(cfg: Config, out: dict, x: torch.Tensor, fitness: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute all losses."""
    losses = {}

    # Reconstruction
    losses["recon"] = F.mse_loss(out["x_recon"], x)

    # KL divergence
    kl = -0.5 * torch.sum(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
    losses["kl"] = 0.001 * kl / x.size(0)

    z = out["z"]

    # Ranking loss (main driver of correlation)
    if cfg.use_rank:
        z_proj = z[:, 0]
        z_c = z_proj - z_proj.mean()
        f_c = fitness - fitness.mean()
        z_std = torch.sqrt(torch.sum(z_c**2) + 1e-8)
        f_std = torch.sqrt(torch.sum(f_c**2) + 1e-8)
        corr = torch.sum(z_c * f_c) / (z_std * f_std)
        losses["rank"] = cfg.ranking_weight * (-corr)

    # Contrastive loss
    if cfg.use_contrast:
        z_norm = F.normalize(z, dim=-1)
        sim = torch.mm(z_norm, z_norm.t()) / 0.1
        labels = torch.arange(z.size(0), device=z.device)
        losses["contrast"] = cfg.contrastive_weight * F.cross_entropy(sim, labels)

    losses["total"] = sum(losses.values())
    return losses


def load_stanford_data(drug_class: str = "pi") -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load Stanford HIVDB data."""
    data_dir = project_root / "data" / "research"

    file_mapping = {
        "pi": "stanford_hivdb_pi.txt",
        "nrti": "stanford_hivdb_nrti.txt",
        "nnrti": "stanford_hivdb_nnrti.txt",
        "ini": "stanford_hivdb_ini.txt",
    }

    drug_columns = {
        "pi": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"],
        "nrti": ["ABC", "AZT", "D4T", "DDI", "FTC", "3TC", "TDF"],
        "nnrti": ["DOR", "EFV", "ETR", "NVP", "RPV"],
        "ini": ["BIC", "CAB", "DTG", "EVG", "RAL"],
    }

    filepath = data_dir / file_mapping[drug_class]
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath, sep="\t", low_memory=False)

    # Get position columns
    # All Stanford HIVDB files use "P" prefix for position columns
    prefix = "P"

    position_cols = [col for col in df.columns if col.startswith(prefix) and col[len(prefix) :].isdigit()]
    position_cols = sorted(position_cols, key=lambda x: int(x[len(prefix) :]))

    return df, position_cols, drug_columns[drug_class]


def encode_amino_acids(df: pd.DataFrame, position_cols: list[str]) -> np.ndarray:
    """One-hot encode amino acid sequences."""
    aa_alphabet = "ACDEFGHIKLMNPQRSTVWY*-"
    aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}

    n_samples = len(df)
    n_positions = len(position_cols)
    n_aa = len(aa_alphabet)

    encoded = np.zeros((n_samples, n_positions * n_aa), dtype=np.float32)

    for idx, (_, row) in enumerate(df.iterrows()):
        for j, col in enumerate(position_cols):
            aa = str(row[col]).upper() if pd.notna(row[col]) else "-"
            if aa in aa_to_idx:
                encoded[idx, j * n_aa + aa_to_idx[aa]] = 1.0
            else:
                # Unknown AA - use gap
                encoded[idx, j * n_aa + aa_to_idx["-"]] = 1.0

    return encoded


def prepare_data(
    drug_class: str, target_drug: str, test_size: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Prepare data for training."""
    df, position_cols, drugs = load_stanford_data(drug_class)

    if target_drug not in drugs:
        raise ValueError(f"Drug {target_drug} not in {drug_class}. Available: {drugs}")

    # Filter rows with valid drug resistance values
    df_valid = df[df[target_drug].notna() & (df[target_drug] > 0)].copy()
    print(f"  Valid samples for {target_drug}: {len(df_valid)}")

    if len(df_valid) < 100:
        raise ValueError(f"Not enough samples for {target_drug}: {len(df_valid)}")

    # Encode sequences
    X = encode_amino_acids(df_valid, position_cols)

    # Get resistance values (log transform for better distribution)
    y = np.log10(df_valid[target_drug].values + 1).astype(np.float32)

    # Normalize y to [0, 1]
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return (
        torch.tensor(X_train),
        torch.tensor(y_train),
        torch.tensor(X_test),
        torch.tensor(y_test),
        X.shape[1],
    )


def train_and_evaluate(
    cfg: Config,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
) -> dict[str, float]:
    """Train model and evaluate."""
    device = torch.device(cfg.device)
    model = VAE(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    best_test_corr = -1.0
    history = {"train_corr": [], "test_corr": []}

    for epoch in range(cfg.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            losses = compute_loss(cfg, out, x, y)
            losses["total"].backward()
            optimizer.step()

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Train correlation
                out_train = model(train_x.to(device))
                z_train = out_train["z"][:, 0].cpu().numpy()
                train_corr = np.corrcoef(z_train, train_y.numpy())[0, 1]

                # Test correlation
                out_test = model(test_x.to(device))
                z_test = out_test["z"][:, 0].cpu().numpy()
                test_corr = np.corrcoef(z_test, test_y.numpy())[0, 1]

            history["train_corr"].append(train_corr)
            history["test_corr"].append(test_corr)

            if test_corr > best_test_corr:
                best_test_corr = test_corr

            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch + 1}: train_corr={train_corr:+.4f}, test_corr={test_corr:+.4f}")

    return {
        "best_test_corr": best_test_corr,
        "final_train_corr": history["train_corr"][-1] if history["train_corr"] else 0,
        "final_test_corr": history["test_corr"][-1] if history["test_corr"] else 0,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run VAE on real HIV data")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--drug-class", type=str, default="all", choices=["pi", "nrti", "nnrti", "ini", "all"])
    args = parser.parse_args()

    print("=" * 80)
    print("RUNNING BEST VAE CONFIGURATION ON REAL HIV DATA")
    print("=" * 80)
    print()

    drug_classes = {
        "pi": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"],
        "nrti": ["ABC", "AZT", "D4T", "DDI", "FTC", "3TC", "TDF"],
        "nnrti": ["DOR", "EFV", "ETR", "NVP", "RPV"],
        "ini": ["BIC", "CAB", "DTG", "EVG", "RAL"],
    }

    classes_to_test = list(drug_classes.keys()) if args.drug_class == "all" else [args.drug_class]

    all_results = []

    for drug_class in classes_to_test:
        print(f"\n{'=' * 80}")
        print(f"DRUG CLASS: {drug_class.upper()}")
        print("=" * 80)

        for drug in drug_classes[drug_class]:
            print(f"\nDrug: {drug}")
            try:
                # Prepare data
                train_x, train_y, test_x, test_y, input_dim = prepare_data(drug_class, drug)

                # Create config
                cfg = Config(
                    input_dim=input_dim,
                    epochs=args.epochs,
                    use_rank=True,
                    use_contrast=True,
                )

                # Train and evaluate
                results = train_and_evaluate(cfg, train_x, train_y, test_x, test_y)

                all_results.append(
                    {
                        "drug_class": drug_class,
                        "drug": drug,
                        "n_train": len(train_x),
                        "n_test": len(test_x),
                        **results,
                    }
                )

                print(f"  Best test correlation: {results['best_test_corr']:+.4f}")

            except Exception as e:
                print(f"  Error: {e}")
                all_results.append(
                    {
                        "drug_class": drug_class,
                        "drug": drug,
                        "error": str(e),
                    }
                )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: ALL DRUGS")
    print("=" * 80)
    print()
    print(f"{'Drug Class':<12} {'Drug':<8} {'N Train':<10} {'N Test':<10} {'Best Test Corr':<15}")
    print("-" * 80)

    successful = [r for r in all_results if "best_test_corr" in r]
    for r in sorted(successful, key=lambda x: -x["best_test_corr"]):
        print(
            f"{r['drug_class'].upper():<12} {r['drug']:<8} {r['n_train']:<10} {r['n_test']:<10} {r['best_test_corr']:+.4f}"
        )

    if successful:
        avg_corr = np.mean([r["best_test_corr"] for r in successful])
        print("-" * 80)
        print(f"{'AVERAGE':<12} {'':<8} {'':<10} {'':<10} {avg_corr:+.4f}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = project_root / "results" / "real_hiv_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
