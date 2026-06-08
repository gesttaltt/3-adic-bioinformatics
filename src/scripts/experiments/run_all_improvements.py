"""Run all improved models on real HIV data.

This script tests all the new improvements:
1. Gene-specific VAE architectures
2. TAM-aware encoding for NRTIs
3. Transformer architecture
4. Multi-task learning
5. Uncertainty quantification

Compares results against baseline VAE.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
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

from src.models.gene_specific_vae import GeneConfig, GeneSpecificVAE
from src.models.multi_task_vae import MultiTaskConfig, MultiTaskVAE
from src.models.resistance_transformer import ResistanceTransformer, TransformerConfig
from src.models.uncertainty import MCDropoutWrapper


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""

    epochs: int = 100
    batch_size: int = 32
    lr: float = 0.001
    use_ranking: bool = True
    ranking_weight: float = 0.3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


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
                encoded[idx, j * n_aa + aa_to_idx["-"]] = 1.0

    return encoded


def prepare_data(
    drug_class: str,
    target_drug: str,
    test_size: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Prepare data for training."""
    df, position_cols, drugs = load_stanford_data(drug_class)

    if target_drug not in drugs:
        raise ValueError(f"Drug {target_drug} not in {drug_class}. Available: {drugs}")

    df_valid = df[df[target_drug].notna() & (df[target_drug] > 0)].copy()

    if len(df_valid) < 100:
        raise ValueError(f"Not enough samples for {target_drug}: {len(df_valid)}")

    X = encode_amino_acids(df_valid, position_cols)
    y = np.log10(df_valid[target_drug].values + 1).astype(np.float32)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return (
        torch.tensor(X_train),
        torch.tensor(y_train),
        torch.tensor(X_test),
        torch.tensor(y_test),
        X.shape[1],
        len(position_cols),
    )


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


def compute_loss(cfg: ExperimentConfig, out: dict, x: torch.Tensor, y: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute training loss."""
    losses = {}

    losses["recon"] = F.mse_loss(out["x_recon"], x)

    kl = -0.5 * torch.sum(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
    losses["kl"] = 0.001 * kl / x.size(0)

    if cfg.use_ranking:
        pred = out.get("prediction", out["z"][:, 0])
        p_c = pred - pred.mean()
        y_c = y - y.mean()
        p_std = torch.sqrt(torch.sum(p_c**2) + 1e-8)
        y_std = torch.sqrt(torch.sum(y_c**2) + 1e-8)
        corr = torch.sum(p_c * y_c) / (p_std * y_std)
        losses["rank"] = cfg.ranking_weight * (-corr)

    losses["total"] = sum(losses.values())
    return losses


def train_model(
    model: nn.Module,
    cfg: ExperimentConfig,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
) -> dict[str, float]:
    """Train model and return metrics."""
    device = torch.device(cfg.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    best_test_corr = -1.0

    for epoch in range(cfg.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            losses = compute_loss(cfg, out, x, y)
            losses["total"].backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                out_test = model(test_x.to(device))
                pred = out_test.get("prediction", out_test["z"][:, 0]).cpu().numpy()
                test_corr = np.corrcoef(pred, test_y.numpy())[0, 1]

            if test_corr > best_test_corr:
                best_test_corr = test_corr

    return {"best_test_corr": best_test_corr}


def run_experiments(drug_class: str = "pi", epochs: int = 100):
    """Run all experiments for a drug class."""
    cfg = ExperimentConfig(epochs=epochs)

    print(f"\n{'=' * 80}")
    print(f"EXPERIMENTS FOR {drug_class.upper()}")
    print("=" * 80)

    _, position_cols, drugs = load_stanford_data(drug_class)
    len(position_cols)

    results = []

    for drug in drugs:
        print(f"\n--- Drug: {drug} ---")

        try:
            train_x, train_y, test_x, test_y, input_dim, n_pos = prepare_data(drug_class, drug)
            print(f"  Data: {len(train_x)} train, {len(test_x)} test")

            # 1. Baseline VAE
            print("  Testing Baseline VAE...")
            model = BaselineVAE(input_dim)
            res = train_model(model, cfg, train_x, train_y, test_x, test_y)
            baseline_corr = res["best_test_corr"]
            print(f"    Baseline: {baseline_corr:+.4f}")

            # 2. Gene-Specific VAE
            print("  Testing Gene-Specific VAE...")
            if drug_class == "pi":
                gene_cfg = GeneConfig.for_protease()
            elif drug_class in ["nrti", "nnrti"]:
                gene_cfg = GeneConfig.for_reverse_transcriptase()
                gene_cfg.input_dim = input_dim
                gene_cfg.n_positions = n_pos
            else:
                gene_cfg = GeneConfig.for_integrase()
                gene_cfg.input_dim = input_dim
                gene_cfg.n_positions = n_pos

            gene_cfg.input_dim = input_dim
            model = GeneSpecificVAE(gene_cfg)
            res = train_model(model, cfg, train_x, train_y, test_x, test_y)
            gene_corr = res["best_test_corr"]
            print(f"    Gene-Specific: {gene_corr:+.4f}")

            # 3. Transformer
            print("  Testing Transformer...")
            trans_cfg = TransformerConfig(n_positions=n_pos, d_model=64, n_layers=2, n_heads=4)
            model = ResistanceTransformer(trans_cfg)
            res = train_model(model, cfg, train_x, train_y, test_x, test_y)
            trans_corr = res["best_test_corr"]
            print(f"    Transformer: {trans_corr:+.4f}")

            results.append(
                {
                    "drug_class": drug_class,
                    "drug": drug,
                    "n_train": len(train_x),
                    "n_test": len(test_x),
                    "baseline_corr": baseline_corr,
                    "gene_specific_corr": gene_corr,
                    "transformer_corr": trans_corr,
                    "best_improvement": max(gene_corr, trans_corr) - baseline_corr,
                }
            )

        except Exception as e:
            print(f"  Error: {e}")
            results.append(
                {
                    "drug_class": drug_class,
                    "drug": drug,
                    "error": str(e),
                }
            )

    return results


def run_multi_task_experiment(drug_class: str = "pi", epochs: int = 100):
    """Run multi-task learning experiment."""
    cfg = ExperimentConfig(epochs=epochs)

    print(f"\n{'=' * 80}")
    print(f"MULTI-TASK EXPERIMENT FOR {drug_class.upper()}")
    print("=" * 80)

    _, position_cols, drugs = load_stanford_data(drug_class)
    n_positions = len(position_cols)
    input_dim = n_positions * 22

    # Load all drug data
    all_data = {}
    for drug in drugs:
        try:
            train_x, train_y, test_x, test_y, _, _ = prepare_data(drug_class, drug)
            all_data[drug] = {"train_x": train_x, "train_y": train_y, "test_x": test_x, "test_y": test_y}
        except Exception as e:
            print(f"  Skipping {drug}: {e}")

    if len(all_data) < 2:
        print("  Not enough drugs for multi-task learning")
        return []

    # Create multi-task model
    mt_cfg = MultiTaskConfig(input_dim=input_dim, drug_names=list(all_data.keys()))
    model = MultiTaskVAE(mt_cfg)

    device = torch.device(cfg.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Train on all drugs together
    print("  Training multi-task model...")
    for _epoch in range(cfg.epochs):
        model.train()
        for drug, data in all_data.items():
            x = data["train_x"].to(device)
            y = data["train_y"].to(device)

            # Sample batch
            idx = torch.randint(0, len(x), (cfg.batch_size,))
            x_batch = x[idx]
            y_batch = y[idx]

            optimizer.zero_grad()
            out = model(x_batch, drug=drug)

            # Compute loss
            recon = F.mse_loss(out["x_recon"], x_batch)
            kl = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
            pred = out["prediction"]
            p_c = pred - pred.mean()
            y_c = y_batch - y_batch.mean()
            corr = torch.sum(p_c * y_c) / (torch.sqrt(torch.sum(p_c**2) + 1e-8) * torch.sqrt(torch.sum(y_c**2) + 1e-8))
            loss = recon + 0.001 * kl + cfg.ranking_weight * (-corr)

            loss.backward()
            optimizer.step()

    # Evaluate
    print("\n  Multi-task results:")
    results = []
    model.eval()
    with torch.no_grad():
        for drug, data in all_data.items():
            out = model(data["test_x"].to(device), drug=drug)
            pred = out["prediction"].cpu().numpy()
            test_y = data["test_y"].numpy()
            corr = np.corrcoef(pred, test_y)[0, 1]
            print(f"    {drug}: {corr:+.4f}")
            results.append({"drug": drug, "multi_task_corr": corr})

    return results


def run_uncertainty_experiment(drug_class: str = "pi", drug: str = "LPV", epochs: int = 100):
    """Run uncertainty quantification experiment."""
    cfg = ExperimentConfig(epochs=epochs)

    print(f"\n{'=' * 80}")
    print(f"UNCERTAINTY EXPERIMENT FOR {drug}")
    print("=" * 80)

    train_x, train_y, test_x, test_y, input_dim, n_pos = prepare_data(drug_class, drug)

    # Train baseline with dropout
    model = BaselineVAE(input_dim)

    # Add more dropout for MC Dropout
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.2

    device = torch.device(cfg.device)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    for _epoch in range(cfg.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            losses = compute_loss(cfg, out, x, y)
            losses["total"].backward()
            optimizer.step()

    # MC Dropout uncertainty
    mc_wrapper = MCDropoutWrapper(model, n_samples=50)
    estimate = mc_wrapper.predict_with_uncertainty(test_x.to(device))

    print(f"\n  Mean prediction std: {estimate.std.mean():.4f}")
    print(f"  Prediction range: [{estimate.mean.min():.4f}, {estimate.mean.max():.4f}]")

    # Check calibration
    actual = test_y.numpy()
    pred_mean = estimate.mean.cpu().numpy()
    pred_std = estimate.std.cpu().numpy()

    # 95% coverage
    lower = pred_mean - 1.96 * pred_std
    upper = pred_mean + 1.96 * pred_std
    coverage = ((actual >= lower) & (actual <= upper)).mean()
    print(f"  95% CI coverage: {coverage:.2%}")

    return {
        "mean_std": estimate.std.mean().item(),
        "coverage_95": coverage,
        "correlation": np.corrcoef(pred_mean, actual)[0, 1],
    }


def main():
    """Run all experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run all improvements")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--drug-class", type=str, default="pi", choices=["pi", "nrti", "nnrti", "ini", "all"])
    args = parser.parse_args()

    all_results = []

    # Run main experiments
    classes = ["pi", "nrti", "nnrti", "ini"] if args.drug_class == "all" else [args.drug_class]

    for drug_class in classes:
        results = run_experiments(drug_class, epochs=args.epochs)
        all_results.extend(results)

    # Run multi-task for PI
    if "pi" in classes:
        run_multi_task_experiment("pi", epochs=args.epochs)

    # Run uncertainty for LPV
    if "pi" in classes:
        run_uncertainty_experiment("pi", "LPV", epochs=args.epochs)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    successful = [r for r in all_results if "baseline_corr" in r]

    if successful:
        print(f"\n{'Drug':<8} {'Baseline':>10} {'Gene-Spec':>10} {'Transformer':>12} {'Improve':>10}")
        print("-" * 60)

        for r in sorted(successful, key=lambda x: -x.get("baseline_corr", 0)):
            print(
                f"{r['drug']:<8} {r['baseline_corr']:>+10.4f} {r['gene_specific_corr']:>+10.4f} "
                f"{r['transformer_corr']:>+12.4f} {r['best_improvement']:>+10.4f}"
            )

        # Averages
        avg_baseline = np.mean([r["baseline_corr"] for r in successful])
        avg_gene = np.mean([r["gene_specific_corr"] for r in successful])
        avg_trans = np.mean([r["transformer_corr"] for r in successful])

        print("-" * 60)
        print(f"{'AVERAGE':<8} {avg_baseline:>+10.4f} {avg_gene:>+10.4f} {avg_trans:>+12.4f}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = project_root / "results" / "all_improvements_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
