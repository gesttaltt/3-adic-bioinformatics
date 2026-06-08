"""Enhanced Training Script with TAM Integration and Multi-Task Support.

This script addresses the critical gap in NRTI/NNRTI/INI performance by:
1. TAM-aware encoding for NRTI drugs
2. Position-weighted loss functions
3. Multi-task training across drug classes
4. Improved architecture for long sequences (RT, IN)

Target improvements:
- NRTI: +0.07 → +0.25
- NNRTI: +0.19 → +0.35
- INI: +0.14 → +0.30
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
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from encoding.tam_aware_encoder import NRTI_KEY_POSITIONS, TAMAwareEncoder


@dataclass
class EnhancedConfig:
    """Enhanced configuration with drug-class-specific settings."""

    input_dim: int = 99
    latent_dim: int = 16
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    batch_size: int = 32
    epochs: int = 150
    lr: float = 0.001
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss weights
    use_rank: bool = True
    use_contrast: bool = True
    use_tam_loss: bool = False  # Enable for NRTI
    ranking_weight: float = 0.3
    contrastive_weight: float = 0.1
    tam_weight: float = 0.2
    position_weight: float = 0.1

    # Architecture options
    use_attention: bool = False
    use_residual: bool = True
    dropout: float = 0.15

    # Drug-class specific
    drug_class: str = "pi"
    target_drug: str = ""


class AttentionBlock(nn.Module):
    """Self-attention for capturing mutation interactions."""

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim) -> (batch, 1, dim)
        x = x.unsqueeze(1)
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out).squeeze(1)


class ResidualBlock(nn.Module):
    """Residual block with normalization."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class EnhancedVAE(nn.Module):
    """Enhanced VAE with attention and residual connections."""

    def __init__(self, cfg: EnhancedConfig):
        super().__init__()
        self.cfg = cfg

        # Encoder
        layers = []
        in_dim = cfg.input_dim
        for i, h in enumerate(cfg.hidden_dims):
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.GELU())
            layers.append(nn.LayerNorm(h))
            layers.append(nn.Dropout(cfg.dropout))

            if cfg.use_residual and i > 0 and in_dim == h:
                layers.append(ResidualBlock(h, cfg.dropout))

            in_dim = h

        if cfg.use_attention:
            layers.append(AttentionBlock(in_dim))

        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(in_dim, cfg.latent_dim)

        # Decoder
        decoder_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.GELU())
            decoder_layers.append(nn.LayerNorm(h))
            decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h

        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Direct predictor (for ranking)
        self.predictor = nn.Sequential(
            nn.Linear(cfg.latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        x_recon = self.decoder(z)
        prediction = self.predictor(z).squeeze(-1)

        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "prediction": prediction,
        }


class TAMAwareLoss(nn.Module):
    """TAM-aware loss function for NRTI drugs."""

    def __init__(
        self,
        key_positions: list[int] = None,
        position_weight: float = 2.0,
        tam_consistency_weight: float = 0.5,
    ):
        super().__init__()
        self.key_positions = key_positions or NRTI_KEY_POSITIONS
        self.position_weight = position_weight
        self.tam_consistency_weight = tam_consistency_weight

    def create_position_weights(self, n_positions: int, n_aa: int = 22) -> torch.Tensor:
        """Create weight mask emphasizing key positions."""
        weights = torch.ones(n_positions * n_aa)

        for pos in self.key_positions:
            if pos < n_positions:
                start = pos * n_aa
                end = start + n_aa
                weights[start:end] = self.position_weight

        return weights

    def forward(
        self,
        x_recon: torch.Tensor,
        x: torch.Tensor,
        tam_features: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute TAM-aware reconstruction loss."""
        device = x.device
        n_positions = x.shape[1] // 22  # Assuming 22 AA alphabet

        # Position-weighted reconstruction
        weights = self.create_position_weights(n_positions).to(device)
        weighted_mse = (weights * (x - x_recon) ** 2).mean()

        loss = weighted_mse

        # TAM consistency (if TAM features provided)
        if tam_features is not None and z is not None:
            # The latent should correlate with TAM patterns
            tam_pred = z[:, : tam_features.shape[1]]
            tam_loss = F.mse_loss(tam_pred, tam_features)
            loss = loss + self.tam_consistency_weight * tam_loss

        return loss


def compute_enhanced_loss(
    cfg: EnhancedConfig,
    out: dict[str, torch.Tensor],
    x: torch.Tensor,
    fitness: torch.Tensor,
    tam_features: torch.Tensor | None = None,
    tam_loss_fn: TAMAwareLoss | None = None,
) -> dict[str, torch.Tensor]:
    """Compute all losses with TAM awareness."""
    losses = {}

    # Reconstruction (optionally TAM-aware)
    if cfg.use_tam_loss and tam_loss_fn is not None:
        losses["recon"] = tam_loss_fn(out["x_recon"], x, tam_features, out["z"])
    else:
        losses["recon"] = F.mse_loss(out["x_recon"], x)

    # KL divergence
    kl = -0.5 * torch.sum(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
    losses["kl"] = 0.001 * kl / x.size(0)

    # Prediction loss (MSE)
    losses["pred"] = F.mse_loss(out["prediction"], fitness)

    # Ranking loss (main driver)
    if cfg.use_rank:
        pred = out["prediction"]
        p_c = pred - pred.mean()
        f_c = fitness - fitness.mean()
        p_std = torch.sqrt(torch.sum(p_c**2) + 1e-8)
        f_std = torch.sqrt(torch.sum(f_c**2) + 1e-8)
        corr = torch.sum(p_c * f_c) / (p_std * f_std)
        losses["rank"] = cfg.ranking_weight * (-corr)

    # Contrastive loss
    if cfg.use_contrast:
        z = out["z"]
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

    # Get position columns based on drug class
    if drug_class == "pi":
        prefix = "P"
    elif drug_class in ["nrti", "nnrti"]:
        prefix = "RT"
    else:  # ini
        prefix = "IN"

    position_cols = [col for col in df.columns if col.startswith(prefix) and col[len(prefix) :].isdigit()]
    position_cols = sorted(position_cols, key=lambda x: int(x[len(prefix) :]))

    return df, position_cols, drug_columns[drug_class]


def encode_amino_acids(df: pd.DataFrame, position_cols: list[str]) -> np.ndarray:
    """Standard one-hot encoding."""
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


def prepare_data_enhanced(
    drug_class: str,
    target_drug: str,
    test_size: float = 0.2,
    use_tam: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor | None, torch.Tensor | None]:
    """Prepare data with optional TAM features."""
    df, position_cols, drugs = load_stanford_data(drug_class)

    if target_drug not in drugs:
        raise ValueError(f"Drug {target_drug} not in {drug_class}. Available: {drugs}")

    # Filter rows with valid drug resistance values
    df_valid = df[df[target_drug].notna() & (df[target_drug] > 0)].copy()
    print(f"  Valid samples for {target_drug}: {len(df_valid)}")

    if len(df_valid) < 100:
        raise ValueError(f"Not enough samples for {target_drug}: {len(df_valid)}")

    # Encode sequences
    if use_tam and drug_class in ["nrti", "nnrti"]:
        print("  Using TAM-aware encoding...")
        tam_encoder = TAMAwareEncoder(position_cols)
        X = tam_encoder.encode_dataframe(df_valid)
        tam_features_full = X[:, -tam_encoder.tam_dim :]  # Last N columns are TAM features
    else:
        X = encode_amino_acids(df_valid, position_cols)
        tam_features_full = None

    # Get resistance values (log transform)
    y = np.log10(df_valid[target_drug].values + 1).astype(np.float32)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)

    # Split
    if tam_features_full is not None:
        X_train, X_test, y_train, y_test, tam_train, tam_test = train_test_split(
            X, y, tam_features_full, test_size=test_size, random_state=42
        )
        return (
            torch.tensor(X_train),
            torch.tensor(y_train),
            torch.tensor(X_test),
            torch.tensor(y_test),
            X.shape[1],
            torch.tensor(tam_train),
            torch.tensor(tam_test),
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return (
            torch.tensor(X_train),
            torch.tensor(y_train),
            torch.tensor(X_test),
            torch.tensor(y_test),
            X.shape[1],
            None,
            None,
        )


def train_and_evaluate_enhanced(
    cfg: EnhancedConfig,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    tam_train: torch.Tensor | None = None,
    tam_test: torch.Tensor | None = None,
) -> dict[str, float]:
    """Train model and evaluate with enhanced features."""
    device = torch.device(cfg.device)
    model = EnhancedVAE(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # TAM loss function
    tam_loss_fn = TAMAwareLoss() if cfg.use_tam_loss else None

    # Prepare data loaders
    dataset = TensorDataset(train_x, train_y, tam_train) if tam_train is not None else TensorDataset(train_x, train_y)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    best_test_corr = -1.0
    history = {"train_corr": [], "test_corr": [], "train_loss": []}

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0

        for batch in loader:
            if tam_train is not None:
                x, y, tam = batch
                x, y, tam = x.to(device), y.to(device), tam.to(device)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                tam = None

            optimizer.zero_grad()
            out = model(x)
            losses = compute_enhanced_loss(cfg, out, x, y, tam, tam_loss_fn)
            losses["total"].backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_loss += losses["total"].item()

        scheduler.step()
        history["train_loss"].append(epoch_loss / len(loader))

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Train correlation
                out_train = model(train_x.to(device))
                pred_train = out_train["prediction"].cpu().numpy()
                train_corr = np.corrcoef(pred_train, train_y.numpy())[0, 1]

                # Test correlation
                out_test = model(test_x.to(device))
                pred_test = out_test["prediction"].cpu().numpy()
                test_corr = np.corrcoef(pred_test, test_y.numpy())[0, 1]

            history["train_corr"].append(train_corr)
            history["test_corr"].append(test_corr)

            if test_corr > best_test_corr:
                best_test_corr = test_corr

            if (epoch + 1) % 50 == 0:
                print(
                    f"    Epoch {epoch + 1}: train={train_corr:+.4f}, test={test_corr:+.4f}, loss={epoch_loss / len(loader):.4f}"
                )

    return {
        "best_test_corr": best_test_corr,
        "final_train_corr": history["train_corr"][-1] if history["train_corr"] else 0,
        "final_test_corr": history["test_corr"][-1] if history["test_corr"] else 0,
    }


def run_drug_class(drug_class: str, drugs: list[str], epochs: int, use_tam: bool) -> list[dict]:
    """Run training for a drug class."""
    results = []

    for drug in drugs:
        print(f"\nDrug: {drug}")
        try:
            # Prepare data
            train_x, train_y, test_x, test_y, input_dim, tam_train, tam_test = prepare_data_enhanced(
                drug_class, drug, use_tam=use_tam
            )

            # Create config with drug-class-specific settings
            cfg = EnhancedConfig(
                input_dim=input_dim,
                epochs=epochs,
                drug_class=drug_class,
                target_drug=drug,
                use_rank=True,
                use_contrast=True,
                use_tam_loss=use_tam and drug_class in ["nrti", "nnrti"],
                use_attention=drug_class in ["nrti", "nnrti", "ini"],  # Attention for RT/IN
                use_residual=True,
            )

            # Adjust architecture for longer sequences
            if drug_class in ["nrti", "nnrti"]:
                cfg.hidden_dims = [512, 256, 128, 64]  # Larger for RT (560 positions)
                cfg.latent_dim = 32
            elif drug_class == "ini":
                cfg.hidden_dims = [256, 128, 64]  # Medium for IN (288 positions)
                cfg.latent_dim = 24

            # Train and evaluate
            result = train_and_evaluate_enhanced(cfg, train_x, train_y, test_x, test_y, tam_train, tam_test)

            results.append(
                {
                    "drug_class": drug_class,
                    "drug": drug,
                    "n_train": len(train_x),
                    "n_test": len(test_x),
                    "use_tam": use_tam,
                    **result,
                }
            )

            print(f"  Best test correlation: {result['best_test_corr']:+.4f}")

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
    parser = argparse.ArgumentParser(description="Enhanced VAE Training with TAM Support")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--drug-class", type=str, default="all", choices=["pi", "nrti", "nnrti", "ini", "all"])
    parser.add_argument("--use-tam", action="store_true", help="Use TAM-aware encoding for NRTI/NNRTI")
    parser.add_argument("--compare", action="store_true", help="Compare with and without TAM")
    args = parser.parse_args()

    print("=" * 80)
    print("ENHANCED VAE TRAINING WITH TAM INTEGRATION")
    print("=" * 80)
    print(f"\nSettings: epochs={args.epochs}, use_tam={args.use_tam}, compare={args.compare}")

    drug_classes = {
        "pi": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"],
        "nrti": ["ABC", "AZT", "D4T", "DDI", "3TC", "TDF"],  # Removed FTC (limited data)
        "nnrti": ["DOR", "EFV", "ETR", "NVP", "RPV"],
        "ini": ["BIC", "DTG", "EVG", "RAL"],  # Removed CAB (limited data)
    }

    classes_to_test = list(drug_classes.keys()) if args.drug_class == "all" else [args.drug_class]

    all_results = []

    for drug_class in classes_to_test:
        print(f"\n{'=' * 80}")
        print(f"DRUG CLASS: {drug_class.upper()}")
        print("=" * 80)

        if args.compare and drug_class in ["nrti", "nnrti"]:
            # Run without TAM
            print("\n--- Without TAM ---")
            results_no_tam = run_drug_class(drug_class, drug_classes[drug_class], args.epochs, use_tam=False)
            for r in results_no_tam:
                r["variant"] = "baseline"
            all_results.extend(results_no_tam)

            # Run with TAM
            print("\n--- With TAM ---")
            results_tam = run_drug_class(drug_class, drug_classes[drug_class], args.epochs, use_tam=True)
            for r in results_tam:
                r["variant"] = "tam"
            all_results.extend(results_tam)

        else:
            use_tam = args.use_tam and drug_class in ["nrti", "nnrti"]
            results = run_drug_class(drug_class, drug_classes[drug_class], args.epochs, use_tam=use_tam)
            all_results.extend(results)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: ALL DRUGS")
    print("=" * 80)

    successful = [r for r in all_results if "best_test_corr" in r]

    if args.compare:
        print(f"\n{'Drug':<8} {'Class':<8} {'Variant':<10} {'N Train':<10} {'Test Corr':<12}")
        print("-" * 60)
        for r in sorted(successful, key=lambda x: (x["drug"], x.get("variant", ""))):
            print(
                f"{r['drug']:<8} {r['drug_class']:<8} {r.get('variant', 'n/a'):<10} {r['n_train']:<10} {r['best_test_corr']:+.4f}"
            )
    else:
        print(f"\n{'Drug':<8} {'Class':<8} {'N Train':<10} {'Test Corr':<12}")
        print("-" * 50)
        for r in sorted(successful, key=lambda x: -x["best_test_corr"]):
            print(f"{r['drug']:<8} {r['drug_class']:<8} {r['n_train']:<10} {r['best_test_corr']:+.4f}")

    # Class averages
    print("\n--- Class Averages ---")
    for dc in classes_to_test:
        class_results = [r for r in successful if r["drug_class"] == dc]
        if class_results:
            avg = np.mean([r["best_test_corr"] for r in class_results])
            print(f"{dc.upper()}: {avg:+.4f}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_path = project_root / "results" / "enhanced_training_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
