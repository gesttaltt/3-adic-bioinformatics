#!/usr/bin/env python3
"""Improved Training Script with All Quick Wins Implemented.

Improvements over baseline:
1. TAM-aware encoding for NRTI/NNRTI drugs
2. Position-weighted reconstruction loss
3. Drug-specific interaction features (RPV, TDF)
4. Ensemble training for robustness
5. Stable architecture (BatchNorm + ReLU)

Copyright 2024-2025 AI Whisperers
Licensed under PolyForm Noncommercial License 1.0.0
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# Key resistance positions by drug class (from Stanford HIVDB)
KEY_POSITIONS = {
    "pi": [30, 32, 33, 46, 47, 48, 50, 54, 76, 82, 84, 88, 90],
    "nrti": [41, 62, 65, 67, 69, 70, 74, 75, 115, 151, 184, 210, 215, 219],
    "nnrti": [100, 101, 103, 106, 181, 188, 190, 225, 230],
    "ini": [66, 92, 118, 140, 143, 147, 148, 155, 263],
}

# Drug-specific interaction patterns
DRUG_INTERACTIONS = {
    "RPV": {
        "positions": [138, 184, 181, 101, 221, 227],
        "interactions": [
            (138, 184),  # E138K + M184V/I
            (138, 181),  # E138K + Y181C
            (181, 101),  # Y181C + K101E
        ],
    },
    "TDF": {
        "positions": [65, 70, 41, 210, 215],
        "interactions": [
            (65, 184),  # K65R + M184V
            (41, 215),  # M41L + T215Y (TAM1)
            (70, 219),  # K70R + K219Q (TAM2)
        ],
    },
    "ETR": {
        "positions": [100, 101, 181, 190, 230],
        "interactions": [
            (181, 101),  # Y181C + K101E
            (100, 181),  # L100I + Y181C
        ],
    },
}


@dataclass
class ImprovedConfig:
    """Configuration for improved training."""

    input_dim: int = 99
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64, 32])
    batch_size: int = 32
    epochs: int = 100
    lr: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss weights
    use_rank: bool = True
    use_contrast: bool = True
    ranking_weight: float = 0.3
    contrastive_weight: float = 0.1

    # Improvements
    use_tam: bool = False
    use_position_weights: bool = True
    position_weight_factor: float = 2.0
    use_interactions: bool = True

    # Drug info
    drug_class: str = "pi"
    target_drug: str = ""

    # For ensemble
    seed: int = 42
    n_ensemble: int = 1


class ImprovedVAE(nn.Module):
    """VAE with stable architecture (BatchNorm + ReLU)."""

    def __init__(self, cfg: ImprovedConfig, onehot_dim: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.onehot_dim = onehot_dim or cfg.input_dim

        # Encoder - BatchNorm + ReLU (stable)
        layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.1)])
            in_dim = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, cfg.latent_dim)

        # Decoder - only reconstructs one-hot portion
        layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            layers.extend([nn.Linear(in_dim, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.1)])
            in_dim = h
        layers.append(nn.Linear(in_dim, self.onehot_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        # Reconstruct only one-hot portion
        x_recon = self.decoder(z)

        return {"x_recon": x_recon, "mu": mu, "logvar": logvar, "z": z}


def create_position_weights(n_positions: int, n_aa: int, drug_class: str, weight_factor: float = 2.0) -> torch.Tensor:
    """Create position weight tensor for reconstruction loss."""
    weights = torch.ones(n_positions * n_aa)

    key_pos = KEY_POSITIONS.get(drug_class, [])
    for pos in key_pos:
        if pos < n_positions:
            start = pos * n_aa
            end = start + n_aa
            weights[start:end] = weight_factor

    return weights


def compute_improved_loss(
    cfg: ImprovedConfig,
    out: dict[str, torch.Tensor],
    x: torch.Tensor,
    fitness: torch.Tensor,
    position_weights: torch.Tensor | None = None,
    onehot_dim: int | None = None,
) -> dict[str, torch.Tensor]:
    """Compute losses with position weighting."""
    losses = {}
    onehot_dim = onehot_dim or x.shape[1]

    # Reconstruction (only one-hot portion, with position weights)
    x_onehot = x[:, :onehot_dim]
    if position_weights is not None and cfg.use_position_weights:
        weights = position_weights.to(x.device)
        diff = (out["x_recon"] - x_onehot) ** 2
        losses["recon"] = (diff * weights).mean()
    else:
        losses["recon"] = F.mse_loss(out["x_recon"], x_onehot)

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
    prefix = "P"
    position_cols = [col for col in df.columns if col.startswith(prefix) and col[len(prefix) :].isdigit()]
    position_cols = sorted(position_cols, key=lambda x: int(x[len(prefix) :]))

    return df, position_cols, drug_columns[drug_class]


def encode_amino_acids(df: pd.DataFrame, position_cols: list[str]) -> np.ndarray:
    """Standard one-hot encoding."""
    aa_alphabet = "ACDEFGHIKLMNPQRSTVWY*-"
    aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}

    n_positions = len(position_cols)
    n_aa = len(aa_alphabet)
    X = np.zeros((len(df), n_positions * n_aa), dtype=np.float32)

    for idx, (_, row) in enumerate(df.iterrows()):
        for j, col in enumerate(position_cols):
            aa = str(row[col]).upper() if pd.notna(row[col]) else "-"
            if aa in aa_to_idx:
                X[idx, j * n_aa + aa_to_idx[aa]] = 1.0
            else:
                X[idx, j * n_aa + aa_to_idx["-"]] = 1.0

    return X


def encode_with_tam(df: pd.DataFrame, position_cols: list[str]) -> tuple[np.ndarray, int]:
    """Encode with TAM features for NRTI/NNRTI."""
    from src.encoders.tam_aware_encoder import TAMAwareEncoder

    tam_encoder = TAMAwareEncoder(position_cols)
    X = tam_encoder.encode_dataframe(df)
    return X, tam_encoder.onehot_dim


def compute_drug_interactions(df: pd.DataFrame, position_cols: list[str], drug: str) -> np.ndarray:
    """Compute drug-specific interaction features."""
    if drug not in DRUG_INTERACTIONS:
        return np.zeros((len(df), 0), dtype=np.float32)

    config = DRUG_INTERACTIONS[drug]
    n_features = len(config["positions"]) + len(config["interactions"])
    features = np.zeros((len(df), n_features), dtype=np.float32)

    # Create position lookup
    pos_to_col = {}
    for col in position_cols:
        pos = int(col[1:])  # Remove 'P' prefix
        pos_to_col[pos] = col

    for idx, (_, row) in enumerate(df.iterrows()):
        feat_idx = 0

        # Individual position features
        for pos in config["positions"]:
            if pos in pos_to_col:
                aa = str(row[pos_to_col[pos]]).upper() if pd.notna(row[pos_to_col[pos]]) else "-"
                # Check if mutated (not wildtype)
                features[idx, feat_idx] = 1.0 if aa not in ["-", "*"] and len(aa) == 1 else 0.0
            feat_idx += 1

        # Interaction features
        for pos1, pos2 in config["interactions"]:
            mut1 = 0.0
            mut2 = 0.0
            if pos1 in pos_to_col:
                aa1 = str(row[pos_to_col[pos1]]).upper() if pd.notna(row[pos_to_col[pos1]]) else "-"
                mut1 = 1.0 if aa1 not in ["-", "*"] and len(aa1) == 1 else 0.0
            if pos2 in pos_to_col:
                aa2 = str(row[pos_to_col[pos2]]).upper() if pd.notna(row[pos_to_col[pos2]]) else "-"
                mut2 = 1.0 if aa2 not in ["-", "*"] and len(aa2) == 1 else 0.0
            features[idx, feat_idx] = mut1 * mut2  # Interaction = both mutated
            feat_idx += 1

    return features


def prepare_data(
    drug_class: str,
    target_drug: str,
    use_tam: bool = False,
    use_interactions: bool = True,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Prepare data with all improvements."""
    df, position_cols, drugs = load_stanford_data(drug_class)

    if target_drug not in drugs:
        raise ValueError(f"Drug {target_drug} not in {drug_class}. Available: {drugs}")

    # Filter valid samples
    df_valid = df[df[target_drug].notna() & (df[target_drug] > 0)].copy()
    print(f"  Valid samples for {target_drug}: {len(df_valid)}")

    if len(df_valid) < 100:
        raise ValueError(f"Not enough samples for {target_drug}: {len(df_valid)}")

    # Encode sequences
    if use_tam and drug_class in ["nrti", "nnrti"]:
        print("  Using TAM-aware encoding...")
        X, onehot_dim = encode_with_tam(df_valid, position_cols)
    else:
        X = encode_amino_acids(df_valid, position_cols)
        onehot_dim = X.shape[1]

    # Add interaction features for specific drugs
    if use_interactions and target_drug in DRUG_INTERACTIONS:
        print(f"  Adding interaction features for {target_drug}...")
        interactions = compute_drug_interactions(df_valid, position_cols, target_drug)
        X = np.concatenate([X, interactions], axis=1)

    # Get resistance values (log transform and normalize)
    y = np.log10(df_valid[target_drug].values + 1).astype(np.float32)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        X.shape[1],  # input_dim
        onehot_dim,  # for reconstruction
    )


def train_single_model(
    cfg: ImprovedConfig,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    onehot_dim: int,
) -> tuple[nn.Module, float, dict]:
    """Train a single model."""
    device = torch.device(cfg.device)

    # Set seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Create model
    model = ImprovedVAE(cfg, onehot_dim=onehot_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # Position weights
    n_positions = onehot_dim // 22  # 22 AA alphabet
    position_weights = create_position_weights(n_positions, 22, cfg.drug_class, cfg.position_weight_factor)

    # Data loader
    dataset = TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    best_test_corr = -1.0
    history = {"train_corr": [], "test_corr": [], "loss": []}

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            losses = compute_improved_loss(cfg, out, x, y, position_weights, onehot_dim)
            losses["total"].backward()
            optimizer.step()

            epoch_loss += losses["total"].item()

        history["loss"].append(epoch_loss / len(loader))

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Train correlation
                out_train = model(train_x.to(device))
                pred_train = out_train["z"][:, 0].cpu().numpy()
                train_corr = stats.spearmanr(pred_train, train_y.numpy())[0]

                # Test correlation
                out_test = model(test_x.to(device))
                pred_test = out_test["z"][:, 0].cpu().numpy()
                test_corr = stats.spearmanr(pred_test, test_y.numpy())[0]

            history["train_corr"].append(train_corr)
            history["test_corr"].append(test_corr)

            if test_corr > best_test_corr:
                best_test_corr = test_corr

            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch + 1}: train={train_corr:+.4f}, test={test_corr:+.4f}")

    return model, best_test_corr, history


def train_ensemble(
    cfg: ImprovedConfig,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    onehot_dim: int,
) -> tuple[list[nn.Module], float, float]:
    """Train ensemble of models."""
    models = []
    correlations = []

    print(f"  Training ensemble of {cfg.n_ensemble} models...")

    for i in range(cfg.n_ensemble):
        cfg.seed = 42 + i
        model, best_corr, _ = train_single_model(cfg, train_x, train_y, test_x, test_y, onehot_dim)
        models.append(model)
        correlations.append(best_corr)
        print(f"    Model {i + 1}: {best_corr:+.4f}")

    # Ensemble prediction
    device = torch.device(cfg.device)
    predictions = []

    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(test_x.to(device))
            pred = out["z"][:, 0].cpu().numpy()
            predictions.append(pred)

    # Mean ensemble
    ensemble_pred = np.mean(predictions, axis=0)
    ensemble_corr = stats.spearmanr(ensemble_pred, test_y.numpy())[0]

    print(f"  Ensemble correlation: {ensemble_corr:+.4f}")

    return models, ensemble_corr, np.std(correlations)


def run_drug_class(
    drug_class: str,
    drugs: list[str],
    cfg: ImprovedConfig,
) -> list[dict]:
    """Run training for a drug class."""
    results = []

    for drug in drugs:
        print(f"\nDrug: {drug}")
        try:
            # Prepare data
            train_x, train_y, test_x, test_y, input_dim, onehot_dim = prepare_data(
                drug_class,
                drug,
                use_tam=cfg.use_tam,
                use_interactions=cfg.use_interactions,
                seed=cfg.seed,
            )

            # Update config
            cfg.input_dim = input_dim
            cfg.drug_class = drug_class
            cfg.target_drug = drug

            # Train
            if cfg.n_ensemble > 1:
                models, best_corr, std = train_ensemble(cfg, train_x, train_y, test_x, test_y, onehot_dim)
            else:
                model, best_corr, history = train_single_model(cfg, train_x, train_y, test_x, test_y, onehot_dim)
                std = 0.0

            print(f"  Best test correlation: {best_corr:+.4f}")

            results.append(
                {
                    "drug_class": drug_class,
                    "drug": drug,
                    "n_train": len(train_x),
                    "n_test": len(test_x),
                    "best_test_corr": best_corr,
                    "std": std,
                    "use_tam": cfg.use_tam,
                    "use_interactions": cfg.use_interactions,
                    "n_ensemble": cfg.n_ensemble,
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


def main():
    parser = argparse.ArgumentParser(description="Improved VAE Training")
    parser.add_argument("--drug-class", type=str, default="all", choices=["pi", "nrti", "nnrti", "ini", "all"])
    parser.add_argument("--drug", type=str, default=None, help="Specific drug to train")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--use-tam", action="store_true", help="Use TAM encoding for NRTI/NNRTI")
    parser.add_argument("--use-interactions", action="store_true", help="Use drug-specific interactions")
    parser.add_argument("--n-ensemble", type=int, default=1, help="Number of ensemble models")
    parser.add_argument("--no-position-weights", action="store_true", help="Disable position weights")
    parser.add_argument("--compare-baseline", action="store_true", help="Compare with baseline")
    args = parser.parse_args()

    print("=" * 80)
    print("IMPROVED VAE TRAINING")
    print("=" * 80)
    print("\nSettings:")
    print(f"  epochs={args.epochs}")
    print(f"  use_tam={args.use_tam}")
    print(f"  use_interactions={args.use_interactions}")
    print(f"  n_ensemble={args.n_ensemble}")
    print(f"  position_weights={not args.no_position_weights}")

    cfg = ImprovedConfig(
        epochs=args.epochs,
        use_tam=args.use_tam,
        use_interactions=args.use_interactions,
        use_position_weights=not args.no_position_weights,
        n_ensemble=args.n_ensemble,
    )

    drug_classes = {
        "pi": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"],
        "nrti": ["ABC", "AZT", "D4T", "DDI", "3TC", "TDF"],
        "nnrti": ["DOR", "EFV", "ETR", "NVP", "RPV"],
        "ini": ["BIC", "DTG", "EVG", "RAL"],
    }

    all_results = []

    if args.drug:
        # Single drug mode
        for dc, drugs in drug_classes.items():
            if args.drug in drugs:
                results = run_drug_class(dc, [args.drug], cfg)
                all_results.extend(results)
                break
    elif args.drug_class == "all":
        # All drug classes
        for dc, drugs in drug_classes.items():
            print(f"\n{'=' * 80}")
            print(f"DRUG CLASS: {dc.upper()}")
            print("=" * 80)
            results = run_drug_class(dc, drugs, cfg)
            all_results.extend(results)
    else:
        # Single drug class
        print(f"\n{'=' * 80}")
        print(f"DRUG CLASS: {args.drug_class.upper()}")
        print("=" * 80)
        results = run_drug_class(args.drug_class, drug_classes[args.drug_class], cfg)
        all_results.extend(results)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Drug':<8} {'Class':<8} {'N Train':<10} {'Correlation':<12}")
    print("-" * 50)

    valid_results = [r for r in all_results if "error" not in r]
    for r in sorted(valid_results, key=lambda x: x["best_test_corr"], reverse=True):
        print(f"{r['drug']:<8} {r['drug_class']:<8} {r['n_train']:<10} {r['best_test_corr']:+.4f}")

    if valid_results:
        avg_corr = np.mean([r["best_test_corr"] for r in valid_results])
        print("-" * 50)
        print(f"{'AVERAGE':<8} {'':<8} {'':<10} {avg_corr:+.4f}")

    # Save results
    results_path = project_root / "results" / "improved_training_results.csv"
    pd.DataFrame(all_results).to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
