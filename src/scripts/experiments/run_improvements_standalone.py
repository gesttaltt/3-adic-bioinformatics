"""Run all improved models on real HIV data - standalone version.

Tests all new improvements without relying on src/__init__.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).parent.parent.parent


# Load gene-specific VAE
exec(open(project_root / "src/models/gene_specific_vae.py").read())


class BaselineVAE(nn.Module):
    """Baseline VAE for comparison."""

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        x_recon = self.decoder(z)
        return {"x_recon": x_recon, "mu": mu, "logvar": logvar, "z": z, "prediction": z[:, 0]}


class SimpleTransformer(nn.Module):
    """Simple transformer for drug resistance prediction."""

    def __init__(self, n_positions: int, n_aa: int = 22, d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.n_positions = n_positions
        self.n_aa = n_aa

        self.aa_embed = nn.Linear(n_aa, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fc_mu = nn.Linear(d_model, 16)
        self.fc_logvar = nn.Linear(d_model, 16)
        self.decoder = nn.Linear(16, n_positions * n_aa)
        self.predictor = nn.Linear(16, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.n_positions, self.n_aa)
        x = self.aa_embed(x)
        x = self.transformer(x)
        x = x.mean(dim=1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        x_recon = self.decoder(z)
        prediction = self.predictor(z).squeeze(-1)

        return {"x_recon": x_recon, "mu": mu, "logvar": logvar, "z": z, "prediction": prediction}


def load_stanford_data(drug_class: str) -> tuple[pd.DataFrame, list[str], list[str]]:
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
        "nrti": ["ABC", "AZT", "D4T", "DDI", "3TC", "TDF"],
        "nnrti": ["DOR", "EFV", "ETR", "NVP", "RPV"],
        "ini": ["BIC", "DTG", "EVG", "RAL"],
    }

    filepath = data_dir / file_mapping[drug_class]
    df = pd.read_csv(filepath, sep="\t", low_memory=False)

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
                encoded[idx, j * n_aa + aa_to_idx["-"]] = 1.0

    return encoded


def prepare_data(drug_class: str, target_drug: str) -> tuple:
    """Prepare data for training."""
    df, position_cols, drugs = load_stanford_data(drug_class)
    df_valid = df[df[target_drug].notna() & (df[target_drug] > 0)].copy()

    if len(df_valid) < 100:
        raise ValueError(f"Not enough samples: {len(df_valid)}")

    X = encode_amino_acids(df_valid, position_cols)
    y = np.log10(df_valid[target_drug].values + 1).astype(np.float32)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return (
        torch.tensor(X_train),
        torch.tensor(y_train),
        torch.tensor(X_test),
        torch.tensor(y_test),
        X.shape[1],
        len(position_cols),
    )


def compute_loss(out: dict, x: torch.Tensor, y: torch.Tensor, ranking_weight: float = 0.3) -> torch.Tensor:
    """Compute training loss."""
    recon = F.mse_loss(out["x_recon"], x)
    kl = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())

    pred = out.get("prediction", out["z"][:, 0])
    p_c = pred - pred.mean()
    y_c = y - y.mean()
    p_std = torch.sqrt(torch.sum(p_c**2) + 1e-8)
    y_std = torch.sqrt(torch.sum(y_c**2) + 1e-8)
    corr = torch.sum(p_c * y_c) / (p_std * y_std)
    rank = ranking_weight * (-corr)

    return recon + 0.001 * kl + rank


def train_model(model: nn.Module, train_x, train_y, test_x, test_y, epochs=50, device="cpu") -> float:
    """Train model and return best test correlation."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)

    best_corr = -1.0
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = compute_loss(out, x, y)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(test_x.to(device))
                pred = out.get("prediction", out["z"][:, 0]).cpu().numpy()
                corr = np.corrcoef(pred, test_y.numpy())[0, 1]
                if corr > best_corr:
                    best_corr = corr

    return best_corr


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--drug-class", type=str, default="pi")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("=" * 80)
    print(f"EXPERIMENTS FOR {args.drug_class.upper()}")
    print("=" * 80)

    _, position_cols, drugs = load_stanford_data(args.drug_class)
    results = []

    for drug in drugs:
        print(f"\n--- {drug} ---")
        try:
            train_x, train_y, test_x, test_y, input_dim, n_pos = prepare_data(args.drug_class, drug)
            print(f"  Data: {len(train_x)} train, {len(test_x)} test")

            # Baseline
            model = BaselineVAE(input_dim)
            baseline = train_model(model, train_x, train_y, test_x, test_y, args.epochs, device)
            print(f"  Baseline:     {baseline:+.4f}")

            # Gene-Specific
            if args.drug_class == "pi":
                cfg = GeneConfig.for_protease()
            elif args.drug_class in ["nrti", "nnrti"]:
                cfg = GeneConfig.for_reverse_transcriptase()
                cfg.input_dim = input_dim
                cfg.n_positions = n_pos
            else:
                cfg = GeneConfig.for_integrase()
                cfg.input_dim = input_dim
                cfg.n_positions = n_pos

            cfg.input_dim = input_dim
            model = GeneSpecificVAE(cfg)
            gene_spec = train_model(model, train_x, train_y, test_x, test_y, args.epochs, device)
            print(f"  Gene-Specific:{gene_spec:+.4f}")

            # Transformer
            model = SimpleTransformer(n_pos)
            trans = train_model(model, train_x, train_y, test_x, test_y, args.epochs, device)
            print(f"  Transformer:  {trans:+.4f}")

            results.append(
                {
                    "drug": drug,
                    "baseline": baseline,
                    "gene_specific": gene_spec,
                    "transformer": trans,
                    "best_improvement": max(gene_spec, trans) - baseline,
                }
            )

        except Exception as e:
            print(f"  Error: {e}")

    # Summary
    if results:
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"{'Drug':<8} {'Baseline':>10} {'Gene-Spec':>12} {'Transformer':>12} {'Improve':>10}")
        print("-" * 60)
        for r in sorted(results, key=lambda x: -x["baseline"]):
            print(
                f"{r['drug']:<8} {r['baseline']:>+10.4f} {r['gene_specific']:>+12.4f} "
                f"{r['transformer']:>+12.4f} {r['best_improvement']:>+10.4f}"
            )

        avg_b = np.mean([r["baseline"] for r in results])
        avg_g = np.mean([r["gene_specific"] for r in results])
        avg_t = np.mean([r["transformer"] for r in results])
        print("-" * 60)
        print(f"{'AVERAGE':<8} {avg_b:>+10.4f} {avg_g:>+12.4f} {avg_t:>+12.4f}")


if __name__ == "__main__":
    main()
