#!/usr/bin/env python3
"""
Optimal Configuration: Combining Best Findings

Based on experiment results:
- Loss: ListMLE (#158) - best ranking loss
- Encoding: OneHot (#201) - best encoding
- Optimizer: AdamW (#353) - best optimizer

Tests this combination across all drug classes.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


def listmle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ListMLE loss - maximum likelihood estimation for rankings."""
    n = pred.size(0)
    if n < 2:
        return torch.tensor(0.0, device=pred.device)

    sorted_idx = torch.argsort(target, descending=True)
    sorted_pred = pred[sorted_idx]

    total_loss = 0.0
    for i in range(n):
        remaining = sorted_pred[i:]
        log_prob = sorted_pred[i] - torch.logsumexp(remaining, dim=0)
        total_loss -= log_prob

    return total_loss / n


class OptimalVAE(nn.Module):
    """VAE with optimal configuration based on experiments."""

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()

        # Encoder: 3 layers with BatchNorm + ReLU + Dropout
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        # Decoder: symmetric
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        x_recon = self.decoder(z)
        return {"x_recon": x_recon, "mu": mu, "logvar": logvar, "z": z}


class OptimalTrainer:
    """Trainer with optimal configuration."""

    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device

    def load_stanford_raw(self, drug_class: str = "pi") -> tuple[pd.DataFrame, list[str], list[str]]:
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
        prefix = "P"
        position_cols = [col for col in df.columns if col.startswith(prefix) and col[len(prefix) :].isdigit()]
        position_cols = sorted(position_cols, key=lambda x: int(x[len(prefix) :]))

        return df, position_cols, drug_columns[drug_class]

    def encode_amino_acids(self, df: pd.DataFrame, position_cols: list[str]) -> np.ndarray:
        """OneHot encode amino acid sequences (best encoding)."""
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

    def train(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> dict:
        """Train with optimal configuration: ListMLE + AdamW."""
        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        # AdamW optimizer (best optimizer)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

        best_corr = -1.0
        history = []

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            out = model(X_train_t)
            pred = out["z"][:, 0]  # First latent dimension for ranking

            # Reconstruction loss
            recon_loss = F.mse_loss(out["x_recon"], X_train_t)

            # KL divergence
            kl_loss = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())

            # ListMLE ranking loss (best ranking loss)
            rank_loss = listmle_loss(pred, y_train_t)

            # Contrastive loss
            z_norm = F.normalize(out["z"], dim=-1)
            sim = torch.mm(z_norm, z_norm.t()) / 0.1
            labels = torch.arange(out["z"].size(0), device=self.device)
            contrast_loss = F.cross_entropy(sim, labels)

            # Total loss
            total_loss = recon_loss + 0.001 * kl_loss + 0.3 * rank_loss + 0.1 * contrast_loss

            if torch.isnan(total_loss):
                print(f"    NaN at epoch {epoch}")
                break

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(total_loss.detach())

            # Evaluate
            if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
                model.eval()
                with torch.no_grad():
                    test_out = model(X_test_t)
                    test_pred = test_out["z"][:, 0].cpu().numpy()
                    corr, _ = stats.spearmanr(test_pred, y_test)
                    if not np.isnan(corr):
                        best_corr = max(best_corr, corr)
                        history.append({"epoch": epoch + 1, "corr": corr})

        return {"best_corr": best_corr, "history": history}

    def run_all(self) -> pd.DataFrame:
        """Run on all drug classes."""
        results = []

        for drug_class in ["pi", "nrti", "nnrti", "ini"]:
            print(f"\n{'=' * 70}")
            print(f"OPTIMAL CONFIGURATION - {drug_class.upper()}")
            print(f"{'=' * 70}\n")

            try:
                df, position_cols, drugs = self.load_stanford_raw(drug_class)
            except FileNotFoundError as e:
                print(f"  {e}")
                continue

            for drug in drugs:
                try:
                    df_valid = df[df[drug].notna() & (df[drug] > 0)].copy()
                    if len(df_valid) < 50:
                        print(f"  {drug}: Too few samples ({len(df_valid)})")
                        continue

                    X = self.encode_amino_acids(df_valid, position_cols)
                    y = np.log10(df_valid[drug].values + 1).astype(np.float32)
                    y = (y - y.min()) / (y.max() - y.min() + 1e-8)

                    # Train/test split
                    n = len(X)
                    split_idx = int(0.8 * n)
                    indices = np.random.permutation(n)
                    train_idx, test_idx = indices[:split_idx], indices[split_idx:]

                    X_train, X_test = X[train_idx], X[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]

                    # Create model
                    model = OptimalVAE(X.shape[1]).to(self.device)

                    # Train
                    print(f"  {drug}: Training ({len(X)} samples)...", end=" ")
                    result = self.train(model, X_train, y_train, X_test, y_test)
                    print(f"corr = {result['best_corr']:+.3f}")

                    results.append(
                        {
                            "drug_class": drug_class,
                            "drug": drug,
                            "n_samples": len(X),
                            "best_corr": result["best_corr"],
                        }
                    )

                except Exception as e:
                    print(f"  {drug}: FAILED - {e}")
                    results.append(
                        {
                            "drug_class": drug_class,
                            "drug": drug,
                            "best_corr": np.nan,
                            "error": str(e),
                        }
                    )

        return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("OPTIMAL CONFIGURATION TEST")
    print("Based on experiment findings:")
    print("  - Loss: ListMLE (#158)")
    print("  - Encoding: OneHot (#201)")
    print("  - Optimizer: AdamW (#353)")
    print("=" * 70)

    trainer = OptimalTrainer()
    results = trainer.run_all()

    # Save results
    output_path = project_root / "results" / "optimal_configuration_results.csv"
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Summary
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY - Optimal Configuration Results")
    print(f"{'=' * 70}\n")

    for drug_class in results["drug_class"].unique():
        class_results = results[results["drug_class"] == drug_class].dropna(subset=["best_corr"])
        if len(class_results) > 0:
            avg = class_results["best_corr"].mean()
            print(f"\n{drug_class.upper()} (avg: {avg:+.3f}):")
            for _, row in class_results.iterrows():
                print(f"  {row['drug']}: {row['best_corr']:+.3f}")

    # Overall
    overall_avg = results["best_corr"].mean()
    print(f"\n{'=' * 70}")
    print(f"OVERALL AVERAGE: {overall_avg:+.3f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
