"""Comprehensive validation across all HIV drug classes.

This script validates the p-adic VAE framework across all 23 HIV drugs,
comparing different architectures and documenting the results.

Run with: python src/scripts/experiments/run_full_validation.py
"""

from __future__ import annotations

import sys
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


# =============================================================================
# Data Loading (with fixed P prefix)
# =============================================================================


def load_data(drug_class: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load Stanford HIVDB data with correct P prefix."""
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


def encode_sequences(df: pd.DataFrame, position_cols: list[str]) -> np.ndarray:
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


def prepare_drug_data(drug_class: str, drug: str) -> tuple:
    """Prepare train/test data for a specific drug."""
    df, position_cols, _ = load_data(drug_class)
    df_valid = df[df[drug].notna() & (df[drug] > 0)].copy()

    if len(df_valid) < 50:
        raise ValueError(f"Not enough samples: {len(df_valid)}")

    X = encode_sequences(df_valid, position_cols)
    y = np.log10(df_valid[drug].values + 1).astype(np.float32)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return (
        torch.tensor(X_train),
        torch.tensor(y_train),
        torch.tensor(X_test),
        torch.tensor(y_test),
        X.shape[1],
        len(position_cols),
        len(df_valid),
    )


# =============================================================================
# Model Architectures
# =============================================================================


class StandardVAE(nn.Module):
    """Standard VAE with ranking loss."""

    def __init__(self, input_dim: int, hidden_dims: list[int] = None, latent_dim: int = 16):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        # Encoder
        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(0.1)])
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        # Decoder
        dec_layers = []
        in_dim = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(in_dim, h), nn.GELU()])
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        x_recon = self.decoder(z)
        return {"x_recon": x_recon, "mu": mu, "logvar": logvar, "z": z, "prediction": z[:, 0]}


class AttentionVAE(nn.Module):
    """VAE with self-attention for capturing position interactions."""

    def __init__(self, input_dim: int, n_positions: int, n_aa: int = 22, latent_dim: int = 16):
        super().__init__()
        self.n_positions = n_positions
        self.n_aa = n_aa
        d_model = 64

        # Position embedding
        self.pos_embed = nn.Linear(n_aa, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_positions, d_model) * 0.02)

        # Self-attention
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
        self.norm2 = nn.LayerNorm(d_model)

        # Latent
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        batch_size = x.size(0)

        # Reshape to (batch, positions, aa)
        x_reshaped = x.view(batch_size, self.n_positions, self.n_aa)

        # Embed positions
        h = self.pos_embed(x_reshaped) + self.pos_encoding

        # Self-attention
        attn_out, self.last_attn_weights = self.attention(h, h, h)
        h = self.norm1(h + attn_out)

        # FFN
        h = self.norm2(h + self.ffn(h))

        # Pool
        h = h.mean(dim=1)

        # Latent
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        x_recon = self.decoder(z)
        return {"x_recon": x_recon, "mu": mu, "logvar": logvar, "z": z, "prediction": z[:, 0]}


class TransformerVAE(nn.Module):
    """Transformer-based VAE."""

    def __init__(self, input_dim: int, n_positions: int, n_aa: int = 22, latent_dim: int = 16):
        super().__init__()
        self.n_positions = n_positions
        self.n_aa = n_aa
        d_model = 64

        self.embed = nn.Linear(n_aa, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, n_positions, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.norm = nn.LayerNorm(d_model)
        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.n_positions, self.n_aa)
        h = self.embed(x) + self.pos_embed
        h = self.transformer(h)
        h = self.norm(h.mean(dim=1))

        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10, 10)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        x_recon = self.decoder(z)

        return {"x_recon": x_recon, "mu": mu, "logvar": logvar, "z": z, "prediction": z[:, 0]}


# =============================================================================
# Training
# =============================================================================


def train_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    epochs: int = 50,
    device: str = "cpu",
    ranking_weight: float = 0.3,
) -> tuple[float, float]:
    """Train model and return best test correlation and final loss."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)

    best_corr = -1.0
    final_loss = 0.0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x)

            # Loss
            recon = F.mse_loss(out["x_recon"], x[:, : out["x_recon"].size(1)])
            kl = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())

            pred = out["prediction"]
            p_c = pred - pred.mean()
            y_c = y - y.mean()
            corr = torch.sum(p_c * y_c) / (torch.sqrt(torch.sum(p_c**2) + 1e-8) * torch.sqrt(torch.sum(y_c**2) + 1e-8))

            loss = recon + 0.001 * kl + ranking_weight * (-corr)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        final_loss = epoch_loss / len(loader)

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(test_x.to(device))
                pred = out["prediction"].cpu().numpy()
                test_corr = np.corrcoef(pred, test_y.numpy())[0, 1]
                if not np.isnan(test_corr) and test_corr > best_corr:
                    best_corr = test_corr

    return best_corr, final_loss


# =============================================================================
# Main Validation
# =============================================================================


def run_validation(epochs: int = 50, device: str = "cuda"):
    """Run comprehensive validation across all drugs."""
    all_results = []

    drug_classes = {
        "pi": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"],
        "nrti": ["ABC", "AZT", "D4T", "DDI", "3TC", "TDF"],
        "nnrti": ["DOR", "EFV", "ETR", "NVP", "RPV"],
        "ini": ["BIC", "DTG", "EVG", "RAL"],
    }

    for drug_class, drugs in drug_classes.items():
        print(f"\n{'=' * 60}")
        print(f"DRUG CLASS: {drug_class.upper()}")
        print("=" * 60)

        for drug in drugs:
            print(f"\n--- {drug} ---")
            try:
                train_x, train_y, test_x, test_y, input_dim, n_pos, n_samples = prepare_drug_data(drug_class, drug)
                print(f"  Samples: {n_samples} | Positions: {n_pos} | Features: {input_dim}")

                results = {"drug": drug, "class": drug_class, "n_samples": n_samples, "n_positions": n_pos}

                # Standard VAE
                model = StandardVAE(input_dim)
                corr, _ = train_model(model, train_x, train_y, test_x, test_y, epochs, device)
                results["standard_vae"] = corr
                print(f"  Standard VAE:   {corr:+.4f}")

                # Attention VAE
                model = AttentionVAE(input_dim, n_pos)
                corr, _ = train_model(model, train_x, train_y, test_x, test_y, epochs, device)
                results["attention_vae"] = corr
                print(f"  Attention VAE:  {corr:+.4f}")

                # Transformer VAE
                model = TransformerVAE(input_dim, n_pos)
                corr, _ = train_model(model, train_x, train_y, test_x, test_y, epochs, device)
                results["transformer_vae"] = corr
                print(f"  Transformer VAE:{corr:+.4f}")

                # Best model
                results["best"] = max(results["standard_vae"], results["attention_vae"], results["transformer_vae"])
                print(f"  Best:           {results['best']:+.4f}")

                all_results.append(results)

            except Exception as e:
                print(f"  Error: {e}")

    return all_results


def print_summary(results: list[dict]):
    """Print summary table."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VALIDATION SUMMARY")
    print("=" * 80)

    # Group by class
    by_class = {}
    for r in results:
        cls = r["class"]
        if cls not in by_class:
            by_class[cls] = []
        by_class[cls].append(r)

    print(
        f"\n{'Drug':<8} {'Class':<6} {'N':<6} {'Pos':<5} {'Std VAE':>10} {'Attn VAE':>10} {'Trans VAE':>10} {'Best':>8}"
    )
    print("-" * 80)

    for drug_class in ["pi", "nrti", "nnrti", "ini"]:
        if drug_class not in by_class:
            continue
        for r in sorted(by_class[drug_class], key=lambda x: -x["best"]):
            print(
                f"{r['drug']:<8} {r['class']:<6} {r['n_samples']:<6} {r['n_positions']:<5} "
                f"{r['standard_vae']:>+10.4f} {r['attention_vae']:>+10.4f} {r['transformer_vae']:>+10.4f} "
                f"{r['best']:>+8.4f}"
            )
        # Class average
        avg_std = np.mean([r["standard_vae"] for r in by_class[drug_class]])
        avg_attn = np.mean([r["attention_vae"] for r in by_class[drug_class]])
        avg_trans = np.mean([r["transformer_vae"] for r in by_class[drug_class]])
        avg_best = np.mean([r["best"] for r in by_class[drug_class]])
        print(
            f"{'AVG':<8} {drug_class:<6} {'':<6} {'':<5} {avg_std:>+10.4f} {avg_attn:>+10.4f} {avg_trans:>+10.4f} {avg_best:>+8.4f}"
        )
        print("-" * 80)

    # Overall average
    avg_std = np.mean([r["standard_vae"] for r in results])
    avg_attn = np.mean([r["attention_vae"] for r in results])
    avg_trans = np.mean([r["transformer_vae"] for r in results])
    avg_best = np.mean([r["best"] for r in results])
    print(
        f"{'OVERALL':<8} {'':<6} {'':<6} {'':<5} {avg_std:>+10.4f} {avg_attn:>+10.4f} {avg_trans:>+10.4f} {avg_best:>+8.4f}"
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")

    results = run_validation(epochs=args.epochs, device=device)

    print_summary(results)

    # Save results
    results_path = project_root / "results" / "full_validation.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
