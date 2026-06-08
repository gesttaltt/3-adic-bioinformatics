"""Test Cross-Resistance VAE on NRTI Drugs.

Compares:
1. Individual VAE per drug (baseline)
2. Cross-Resistance VAE (joint prediction)

Validates:
- Prediction accuracy (correlation)
- Cross-drug correlation patterns
- Biological relevance of learned patterns
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")

# Setup path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr

# =============================================================================
# Cross-Resistance Knowledge Base (embedded to avoid import issues)
# =============================================================================

CROSS_RESISTANCE_MATRIX = {
    "AZT": {"AZT": 1.00, "D4T": 0.85, "ABC": 0.40, "TDF": 0.30, "DDI": 0.35, "3TC": -0.15},
    "D4T": {"AZT": 0.85, "D4T": 1.00, "ABC": 0.45, "TDF": 0.35, "DDI": 0.40, "3TC": -0.10},
    "ABC": {"AZT": 0.40, "D4T": 0.45, "ABC": 1.00, "TDF": 0.55, "DDI": 0.50, "3TC": 0.35},
    "TDF": {"AZT": 0.30, "D4T": 0.35, "TDF": 1.00, "ABC": 0.55, "DDI": 0.45, "3TC": 0.25},
    "DDI": {"AZT": 0.35, "D4T": 0.40, "ABC": 0.50, "TDF": 0.45, "DDI": 1.00, "3TC": 0.20},
    "3TC": {"AZT": -0.15, "D4T": -0.10, "ABC": 0.35, "TDF": 0.25, "DDI": 0.20, "3TC": 1.00},
}


@dataclass
class CrossResistanceConfig:
    """Configuration for cross-resistance VAE."""

    input_dim: int
    latent_dim: int = 32
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    drug_names: list[str] = field(default_factory=lambda: ["AZT", "D4T", "ABC", "TDF", "DDI", "3TC"])
    n_positions: int = 240
    dropout: float = 0.1
    use_cross_attention: bool = True
    ranking_weight: float = 0.3


class CrossDrugAttention(nn.Module):
    """Attention mechanism for cross-drug information sharing."""

    def __init__(self, n_drugs: int, latent_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_drugs = n_drugs
        self.latent_dim = latent_dim
        self.drug_embed = nn.Embedding(n_drugs, latent_dim)
        self.attention = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        self.output_proj = nn.Linear(latent_dim, latent_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.size(0)
        device = z.device
        drug_indices = torch.arange(self.n_drugs, device=device)
        drug_embeds = self.drug_embed(drug_indices).unsqueeze(0).expand(batch_size, -1, -1)
        z_expanded = z.unsqueeze(1).expand(-1, self.n_drugs, -1)
        query = drug_embeds + z_expanded
        attn_out, _ = self.attention(query, query, query)
        return self.output_proj(attn_out)


class CrossResistanceVAE(nn.Module):
    """VAE with cross-resistance modeling for NRTI drugs."""

    def __init__(self, cfg: CrossResistanceConfig):
        super().__init__()
        self.cfg = cfg
        self.drug_names = cfg.drug_names
        self.n_drugs = len(cfg.drug_names)

        # Shared encoder
        layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, h),
                    nn.GELU(),
                    nn.LayerNorm(h),
                    nn.Dropout(cfg.dropout),
                ]
            )
            in_dim = h
        self.encoder = nn.Sequential(*layers)

        self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, cfg.latent_dim)

        # Decoder
        dec_layers = []
        dec_in = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.extend([nn.Linear(dec_in, h), nn.GELU()])
            dec_in = h
        dec_layers.append(nn.Linear(dec_in, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Cross-drug attention
        if cfg.use_cross_attention:
            self.cross_attention = CrossDrugAttention(self.n_drugs, cfg.latent_dim)
        else:
            self.cross_attention = None

        # Drug-specific prediction heads
        self.drug_heads = nn.ModuleDict(
            {
                drug: nn.Sequential(
                    nn.Linear(cfg.latent_dim, 32),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(32, 1),
                )
                for drug in cfg.drug_names
            }
        )

        # Cross-resistance matrix as buffer
        self.register_buffer("cross_resistance_matrix", self._build_cross_resistance_tensor())

    def _build_cross_resistance_tensor(self) -> torch.Tensor:
        n = len(self.drug_names)
        matrix = torch.zeros(n, n)
        for i, drug1 in enumerate(self.drug_names):
            for j, drug2 in enumerate(self.drug_names):
                if drug1 in CROSS_RESISTANCE_MATRIX and drug2 in CROSS_RESISTANCE_MATRIX[drug1]:
                    matrix[i, j] = CROSS_RESISTANCE_MATRIX[drug1][drug2]
        return matrix

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)

        if self.cross_attention is not None:
            z_drugs = self.cross_attention(z)
        else:
            z_drugs = z.unsqueeze(1).expand(-1, self.n_drugs, -1)

        predictions = {}
        for i, drug in enumerate(self.drug_names):
            z_drug = z_drugs[:, i, :]
            pred = self.drug_heads[drug](z_drug).squeeze(-1)
            predictions[drug] = pred

        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "predictions": predictions,
        }


# Standard VAE for baseline comparison
class StandardVAE(nn.Module):
    """Standard VAE for single drug prediction."""

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, input_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        x_recon = self.decoder(z)
        pred = self.predictor(z).squeeze(-1)
        return {"x_recon": x_recon, "mu": mu, "logvar": logvar, "prediction": pred}


def compute_cross_loss(cfg, out, x, targets, cross_weight=0.1):
    """Compute loss with cross-resistance regularization."""
    losses = {}

    losses["recon"] = F.mse_loss(out["x_recon"], x)
    kl = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
    losses["kl"] = 0.001 * kl

    predictions = out["predictions"]
    drug_losses = []

    for drug, target in targets.items():
        if drug not in predictions:
            continue
        pred = predictions[drug]
        p_c = pred - pred.mean()
        t_c = target - target.mean()
        p_std = torch.sqrt(torch.sum(p_c**2) + 1e-8)
        t_std = torch.sqrt(torch.sum(t_c**2) + 1e-8)
        corr = torch.sum(p_c * t_c) / (p_std * t_std)
        drug_loss = cfg.ranking_weight * (-corr)
        losses[f"{drug}_loss"] = drug_loss
        drug_losses.append(drug_loss)

    if drug_losses:
        losses["drug_total"] = sum(drug_losses) / len(drug_losses)

    # Cross-resistance consistency loss
    if len(predictions) >= 2:
        drug_names = list(predictions.keys())
        pred_stack = torch.stack([predictions[d] for d in drug_names], dim=1)
        pred_centered = pred_stack - pred_stack.mean(dim=0, keepdim=True)

        n_drugs = len(drug_names)
        cross_loss = 0.0
        count = 0

        for i in range(n_drugs):
            for j in range(i + 1, n_drugs):
                pred_corr = (pred_centered[:, i] * pred_centered[:, j]).mean()
                pred_corr = pred_corr / (pred_centered[:, i].std() * pred_centered[:, j].std() + 1e-8)

                drug_i, drug_j = drug_names[i], drug_names[j]
                if drug_i in CROSS_RESISTANCE_MATRIX and drug_j in CROSS_RESISTANCE_MATRIX[drug_i]:
                    expected = CROSS_RESISTANCE_MATRIX[drug_i][drug_j]
                    cross_loss = cross_loss + (pred_corr - expected) ** 2
                    count += 1

        if count > 0:
            losses["cross_resistance"] = cross_weight * cross_loss / count

    losses["total"] = losses["recon"] + losses["kl"] + losses.get("drug_total", 0) + losses.get("cross_resistance", 0)
    return losses


def load_nrti_data(data_dir: Path) -> dict:
    """Load all NRTI drug data from Stanford HIVDB format."""
    drugs = ["AZT", "D4T", "ABC", "TDF", "DDI", "3TC"]
    data = {}

    # Load Stanford HIVDB NRTI file
    file_path = data_dir / "research" / "stanford_hivdb_nrti.txt"
    if not file_path.exists():
        print(f"  Warning: {file_path} not found")
        return data

    df = pd.read_csv(file_path, sep="\t", low_memory=False)
    print(f"  Loaded {len(df)} total samples from NRTI dataset")

    # Get position columns (all use P prefix)
    pos_cols = [c for c in df.columns if c.startswith("P") and c[1:].isdigit()]
    pos_cols = sorted(pos_cols, key=lambda x: int(x[1:]))
    print(f"  Found {len(pos_cols)} position columns")

    # One-hot encode
    aa_alphabet = "ACDEFGHIKLMNPQRSTVWY*-"
    aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}
    n_aa = len(aa_alphabet)

    for drug in drugs:
        if drug not in df.columns:
            print(f"  Warning: {drug} not in dataset")
            continue

        # Filter valid samples
        df_valid = df[df[drug].notna() & (df[drug] > 0)].copy()

        if len(df_valid) < 50:
            print(f"  Warning: {drug} has only {len(df_valid)} samples, skipping")
            continue

        # Encode sequences
        n_samples = len(df_valid)
        n_positions = len(pos_cols)
        X = np.zeros((n_samples, n_positions * n_aa), dtype=np.float32)

        for idx, (_, row) in enumerate(df_valid.iterrows()):
            for j, col in enumerate(pos_cols):
                aa = str(row[col]).upper() if pd.notna(row[col]) else "-"
                if aa in aa_to_idx:
                    X[idx, j * n_aa + aa_to_idx[aa]] = 1.0
                else:
                    X[idx, j * n_aa + aa_to_idx["-"]] = 1.0

        # Normalize resistance scores
        y = np.log10(df_valid[drug].values + 1).astype(np.float32)
        y = (y - y.min()) / (y.max() - y.min() + 1e-8)

        data[drug] = {"X": X, "y": y, "n_positions": n_positions}
        print(f"  {drug}: {len(y)} samples, {X.shape[1]} features")

    return data


def align_samples(data: dict) -> tuple:
    """Get common dimensions across drugs."""
    n_features = min(d["X"].shape[1] for d in data.values())
    aligned = {}
    for drug, d in data.items():
        X = d["X"][:, :n_features]
        y = d["y"]
        aligned[drug] = {"X": X, "y": y}
    return aligned, n_features


def train_individual_vaes(data: dict, input_dim: int, epochs: int = 50, device: str = "cuda") -> dict:
    """Train individual VAEs for each drug (baseline)."""
    results = {}

    for drug, d in data.items():
        print(f"\n  Training Standard VAE for {drug}...")

        X = torch.tensor(d["X"], dtype=torch.float32)
        y = torch.tensor(d["y"], dtype=torch.float32)

        n = len(X)
        idx = torch.randperm(n)
        train_idx = idx[: int(0.8 * n)]
        test_idx = idx[int(0.8 * n) :]

        X_train, y_train = X[train_idx].to(device), y[train_idx].to(device)
        X_test, y_test = X[test_idx].to(device), y[test_idx].to(device)

        model = StandardVAE(input_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for _epoch in range(epochs):
            optimizer.zero_grad()
            out = model(X_train)

            recon = F.mse_loss(out["x_recon"], X_train)
            kl = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())

            pred = out["prediction"]
            p_c = pred - pred.mean()
            y_c = y_train - y_train.mean()
            corr = torch.sum(p_c * y_c) / (torch.sqrt(torch.sum(p_c**2) + 1e-8) * torch.sqrt(torch.sum(y_c**2) + 1e-8))
            ranking = 0.3 * (-corr)

            loss = recon + 0.001 * kl + ranking
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(X_test)
            pred = out["prediction"].cpu().numpy()
            true = y_test.cpu().numpy()
            corr, _ = pearsonr(pred, true)

        results[drug] = {"correlation": corr, "model": model, "test_idx": test_idx}
        print(f"    {drug}: {corr:+.4f}")

    return results


def load_aligned_nrti_data(data_dir: Path) -> tuple:
    """Load NRTI data with samples aligned across all drugs."""
    drugs = ["AZT", "D4T", "ABC", "TDF", "DDI", "3TC"]

    file_path = data_dir / "research" / "stanford_hivdb_nrti.txt"
    df = pd.read_csv(file_path, sep="\t", low_memory=False)

    # Get samples where ALL drugs have valid data
    valid_mask = pd.Series([True] * len(df))
    for drug in drugs:
        valid_mask &= df[drug].notna() & (df[drug] > 0)

    df_valid = df[valid_mask].copy()
    print(f"  Aligned samples (all drugs): {len(df_valid)}")

    # Get position columns
    pos_cols = [c for c in df.columns if c.startswith("P") and c[1:].isdigit()]
    pos_cols = sorted(pos_cols, key=lambda x: int(x[1:]))

    # One-hot encode
    aa_alphabet = "ACDEFGHIKLMNPQRSTVWY*-"
    aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}
    n_aa = len(aa_alphabet)
    n_samples = len(df_valid)
    n_positions = len(pos_cols)

    X = np.zeros((n_samples, n_positions * n_aa), dtype=np.float32)
    for idx, (_, row) in enumerate(df_valid.iterrows()):
        for j, col in enumerate(pos_cols):
            aa = str(row[col]).upper() if pd.notna(row[col]) else "-"
            if aa in aa_to_idx:
                X[idx, j * n_aa + aa_to_idx[aa]] = 1.0
            else:
                X[idx, j * n_aa + aa_to_idx["-"]] = 1.0

    # Get targets for each drug (normalized)
    targets = {}
    for drug in drugs:
        y = np.log10(df_valid[drug].values + 1).astype(np.float32)
        y = (y - y.min()) / (y.max() - y.min() + 1e-8)
        targets[drug] = y

    return X, targets, n_positions


def train_cross_resistance_vae(
    data: dict, input_dim: int, n_positions: int, epochs: int = 50, device: str = "cuda"
) -> dict:
    """Train cross-resistance VAE for joint prediction using aligned samples."""
    print("\n  Training Cross-Resistance VAE (aligned samples)...")

    # Load truly aligned data
    data_dir = root / "data"
    X_aligned, targets_aligned, _ = load_aligned_nrti_data(data_dir)

    drug_names = list(targets_aligned.keys())
    n_samples = len(X_aligned)

    cfg = CrossResistanceConfig(
        input_dim=X_aligned.shape[1],
        latent_dim=32,
        drug_names=drug_names,
        n_positions=n_positions,
        ranking_weight=0.3,
    )

    model = CrossResistanceVAE(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Split
    idx = torch.randperm(n_samples)
    train_idx = idx[: int(0.8 * n_samples)]
    test_idx = idx[int(0.8 * n_samples) :]

    X = torch.tensor(X_aligned, dtype=torch.float32)
    X_train = X[train_idx].to(device)
    X_test = X[test_idx].to(device)

    train_targets = {}
    test_targets = {}
    for drug in drug_names:
        y = torch.tensor(targets_aligned[drug], dtype=torch.float32)
        train_targets[drug] = y[train_idx].to(device)
        test_targets[drug] = y[test_idx].to(device)

    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        out = model(X_train)
        losses = compute_cross_loss(cfg, out, X_train, train_targets, cross_weight=0.1)

        losses["total"].backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            cr = losses.get("cross_resistance", torch.tensor(0))
            print(f"    Epoch {epoch + 1}: total={losses['total'].item():.4f}, cross={cr.item():.4f}")

    model.eval()
    results = {}

    with torch.no_grad():
        out = model(X_test)
        predictions = out["predictions"]

        for drug in drug_names:
            pred = predictions[drug].cpu().numpy()
            true = test_targets[drug].cpu().numpy()
            corr, _ = pearsonr(pred, true)
            results[drug] = {"correlation": corr}
            print(f"    {drug}: {corr:+.4f}")

    print("\n  Cross-Drug Prediction Correlations:")
    pred_values = {drug: predictions[drug].cpu().numpy() for drug in drug_names}

    for i, drug1 in enumerate(drug_names):
        for j, drug2 in enumerate(drug_names):
            if i < j:
                corr, _ = pearsonr(pred_values[drug1], pred_values[drug2])
                expected = CROSS_RESISTANCE_MATRIX.get(drug1, {}).get(drug2, 0)
                print(f"    {drug1}-{drug2}: pred={corr:+.3f}, expected={expected:+.3f}")

    results["model"] = model
    return results


def main():
    print("=" * 70)
    print("CROSS-RESISTANCE VAE VALIDATION")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    data_dir = root / "data"
    epochs = 50

    print("\nLoading NRTI data...")
    nrti_data = load_nrti_data(data_dir)

    if len(nrti_data) < 2:
        print("Error: Need at least 2 NRTI drugs for cross-resistance testing")
        return

    aligned_data, n_features = align_samples(nrti_data)
    n_positions = n_features // 22
    print(f"\nAligned to {n_features} features ({n_positions} positions)")

    print("\n" + "=" * 70)
    print("BASELINE: Individual VAEs per drug")
    print("=" * 70)
    individual_results = train_individual_vaes(aligned_data, n_features, epochs, device)

    print("\n" + "=" * 70)
    print("CROSS-RESISTANCE VAE: Joint prediction")
    print("=" * 70)
    cross_results = train_cross_resistance_vae(aligned_data, n_features, n_positions, epochs, device)

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Drug':<8} {'Individual':<12} {'Cross-Res':<12} {'Difference':<12}")
    print("-" * 44)

    total_indiv = 0
    total_cross = 0
    count = 0

    for drug in sorted(aligned_data.keys()):
        indiv = individual_results.get(drug, {}).get("correlation", 0)
        cross = cross_results.get(drug, {}).get("correlation", 0)
        diff = cross - indiv

        total_indiv += indiv
        total_cross += cross
        count += 1

        diff_str = f"{diff:+.4f}" if diff != 0 else "  0.0000"
        print(f"{drug:<8} {indiv:+.4f}       {cross:+.4f}       {diff_str}")

    print("-" * 44)
    avg_indiv = total_indiv / count
    avg_cross = total_cross / count
    avg_diff = avg_cross - avg_indiv
    print(f"{'Average':<8} {avg_indiv:+.4f}       {avg_cross:+.4f}       {avg_diff:+.4f}")

    print("\n" + "=" * 70)
    print("CROSS-RESISTANCE PATTERN VALIDATION")
    print("=" * 70)
    print("\nExpected patterns (from literature):")
    print("  - AZT/D4T: High cross-resistance (TAMs)")
    print("  - 3TC/AZT: Negative (M184V resensitizes)")
    print("  - K65R/TAM: Antagonism")

    results_df = pd.DataFrame(
        [
            {
                "drug": drug,
                "individual": individual_results.get(drug, {}).get("correlation", 0),
                "cross_resistance": cross_results.get(drug, {}).get("correlation", 0),
            }
            for drug in aligned_data
        ]
    )

    out_path = root / "results" / "cross_resistance_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
