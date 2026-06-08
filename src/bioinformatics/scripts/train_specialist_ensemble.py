#!/usr/bin/env python3
"""Specialist Ensemble: Combine best specialists with learned weights.

Instead of trying to train a single cross-dataset model (which suffers from
negative transfer), we:
1. Train specialist transformers on each dataset until optimal
2. Learn an ensemble that combines their predictions adaptively

This preserves the strength of each specialist while enabling cross-dataset use.
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
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
project_root = Path(__file__).parents[3]
sys.path.insert(0, str(project_root))

from src.bioinformatics.data.preprocessing import compute_features
from src.bioinformatics.data.protherm_loader import ProThermLoader
from src.bioinformatics.data.s669_loader import S669Loader


@dataclass
class EnsembleConfig:
    """Configuration for ensemble training."""

    # Specialist architecture
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dropout: float = 0.1

    # Training
    specialist_epochs: int = 200
    ensemble_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 30


class DDGTransformer(nn.Module):
    """Transformer for DDG prediction."""

    def __init__(self, input_dim: int, d_model: int = 64, nhead: int = 4, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, input_dim + 1, d_model) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.size(0)
        x = x.unsqueeze(-1)
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_enc
        x = self.transformer(x)
        cls_out = x[:, 0]
        pred = self.head(cls_out).squeeze(-1)
        return {"pred": pred, "embedding": cls_out}


class LearnedEnsemble(nn.Module):
    """Ensemble with learned weights based on input features."""

    def __init__(self, specialists: list[nn.Module], input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.specialists = nn.ModuleList(specialists)
        self.n_specialists = len(specialists)

        # Freeze specialists
        for specialist in self.specialists:
            for param in specialist.parameters():
                param.requires_grad = False

        # Weight prediction network
        self.weight_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, self.n_specialists),
        )

        # Bias predictions (learned correction for each specialist)
        self.bias_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim // 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
                for _ in range(self.n_specialists)
            ]
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Get specialist predictions
        specialist_preds = []
        with torch.no_grad():
            for specialist in self.specialists:
                out = specialist(x)
                specialist_preds.append(out["pred"])

        preds = torch.stack(specialist_preds, dim=1)  # [batch, n_specialists]

        # Compute adaptive weights based on input
        weights = F.softmax(self.weight_net(x), dim=-1)  # [batch, n_specialists]

        # Compute bias corrections
        biases = torch.stack([net(x).squeeze(-1) for net in self.bias_nets], dim=1)

        # Corrected predictions
        corrected_preds = preds + biases

        # Weighted ensemble prediction
        ensemble_pred = (corrected_preds * weights).sum(dim=1)

        # Also compute simple average for comparison
        simple_avg = preds.mean(dim=1)

        return {
            "ensemble_pred": ensemble_pred,
            "simple_avg": simple_avg,
            "specialist_preds": preds,
            "weights": weights,
            "biases": biases,
        }


def load_data():
    """Load both datasets."""
    # ProTherm
    protherm_loader = ProThermLoader()
    db = protherm_loader.load_curated()

    X_protherm, y_protherm = [], []
    for record in db.records:
        feat = compute_features(record.wild_type, record.mutant)
        X_protherm.append(feat.to_array(include_hyperbolic=False))
        y_protherm.append(record.ddg)
    X_protherm = np.array(X_protherm, dtype=np.float32)
    y_protherm = np.array(y_protherm, dtype=np.float32)

    # S669
    s669_loader = S669Loader()
    records = s669_loader.load_from_csv()

    X_s669, y_s669 = [], []
    for record in records:
        feat = compute_features(record.wild_type, record.mutant)
        X_s669.append(feat.to_array(include_hyperbolic=False))
        y_s669.append(record.ddg)
    X_s669 = np.array(X_s669, dtype=np.float32)
    y_s669 = np.array(y_s669, dtype=np.float32)

    return X_protherm, y_protherm, X_s669, y_s669


def train_specialist(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    name: str,
    config: EnsembleConfig,
) -> tuple[DDGTransformer, float]:
    """Train a specialist transformer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DDGTransformer(
        input_dim=X_train.shape[1],
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.specialist_epochs)

    best_val_corr = -1
    best_state = None
    patience_counter = 0

    for epoch in range(config.specialist_epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_x)
            loss = F.mse_loss(out["pred"], batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_val_t)
            val_corr = spearmanr(y_val_t.cpu().numpy(), val_out["pred"].cpu().numpy())[0]

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                break

        if (epoch + 1) % 25 == 0:
            print(f"    {name} epoch {epoch + 1}: val_spearman={val_corr:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    print(f"  {name}: Best Spearman = {best_val_corr:.4f}")
    return model, best_val_corr


def train_ensemble(
    ensemble: LearnedEnsemble,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: EnsembleConfig,
) -> tuple[LearnedEnsemble, dict]:
    """Train the ensemble layer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble = ensemble.to(device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Only train ensemble parameters
    optimizer = torch.optim.AdamW(
        [p for p in ensemble.parameters() if p.requires_grad], lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.ensemble_epochs)

    best_val_corr = -1
    best_state = None
    patience_counter = 0
    history = {"train_loss": [], "val_corr": [], "weights": []}

    for epoch in range(config.ensemble_epochs):
        ensemble.train()
        epoch_loss = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            out = ensemble(batch_x)

            # Primary loss
            loss = F.mse_loss(out["ensemble_pred"], batch_y)

            # Ranking loss
            if len(batch_y) > 1:
                idx1 = torch.randperm(len(batch_y))[: min(32, len(batch_y))]
                idx2 = torch.randperm(len(batch_y))[: min(32, len(batch_y))]
                diff_pred = out["ensemble_pred"][idx1] - out["ensemble_pred"][idx2]
                diff_true = batch_y[idx1] - batch_y[idx2]
                ranking_loss = F.relu(0.1 - diff_pred * torch.sign(diff_true)).mean()
                loss = loss + 0.2 * ranking_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        ensemble.eval()
        with torch.no_grad():
            val_out = ensemble(X_val_t)
            val_corr = spearmanr(y_val_t.cpu().numpy(), val_out["ensemble_pred"].cpu().numpy())[0]
            avg_weights = val_out["weights"].mean(dim=0).cpu().numpy()

        history["train_loss"].append(epoch_loss / len(loader))
        history["val_corr"].append(val_corr)
        history["weights"].append(avg_weights.tolist())

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_state = {k: v.cpu().clone() for k, v in ensemble.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"    Early stopping at epoch {epoch}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"    Ensemble epoch {epoch + 1}: val_corr={val_corr:.4f}, weights={avg_weights}")

    if best_state:
        ensemble.load_state_dict(best_state)

    return ensemble, {"best_val_corr": best_val_corr, "history": history}


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, is_ensemble: bool = False) -> dict:
    """Evaluate model."""
    device = next(model.parameters()).device
    X_t = torch.tensor(X, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        out = model(X_t)
        if is_ensemble:
            preds = out["ensemble_pred"].cpu().numpy()
            simple_avg = out["simple_avg"].cpu().numpy()
            weights = out["weights"].mean(dim=0).cpu().numpy()
        else:
            preds = out["pred"].cpu().numpy()
            simple_avg = None
            weights = None

    result = {
        "spearman": spearmanr(y, preds)[0],
        "pearson": pearsonr(y, preds)[0],
        "mae": np.mean(np.abs(y - preds)),
        "n": len(y),
    }

    if simple_avg is not None:
        result["simple_avg_spearman"] = spearmanr(y, simple_avg)[0]
        result["weights"] = weights.tolist()

    return result


def main():
    """Run ensemble training."""
    print("=" * 70)
    print("SPECIALIST ENSEMBLE: Combining Transformers with Adaptive Weights")
    print("=" * 70)

    config = EnsembleConfig()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/specialist_ensemble_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("\n[1] Loading data...")
    X_protherm, y_protherm, X_s669, y_s669 = load_data()
    print(f"  ProTherm: {len(X_protherm)} samples")
    print(f"  S669: {len(X_s669)} samples")

    # Splits
    np.random.seed(42)

    protherm_idx = np.random.permutation(len(X_protherm))
    protherm_split = int(0.8 * len(protherm_idx))
    X_protherm_train = X_protherm[protherm_idx[:protherm_split]]
    y_protherm_train = y_protherm[protherm_idx[:protherm_split]]
    X_protherm_val = X_protherm[protherm_idx[protherm_split:]]
    y_protherm_val = y_protherm[protherm_idx[protherm_split:]]

    s669_idx = np.random.permutation(len(X_s669))
    s669_split = int(0.8 * len(s669_idx))
    X_s669_train = X_s669[s669_idx[:s669_split]]
    y_s669_train = y_s669[s669_idx[:s669_split]]
    X_s669_val = X_s669[s669_idx[s669_split:]]
    y_s669_val = y_s669[s669_idx[s669_split:]]

    # Train specialists
    print("\n[2] Training SPECIALISTS...")

    print("\n  Training ProTherm Specialist...")
    specialist_protherm, protherm_corr = train_specialist(
        X_protherm_train,
        y_protherm_train,
        X_protherm_val,
        y_protherm_val,
        "ProTherm",
        config,
    )

    print("\n  Training S669 Specialist...")
    specialist_s669, s669_corr = train_specialist(
        X_s669_train,
        y_s669_train,
        X_s669_val,
        y_s669_val,
        "S669",
        config,
    )

    # Evaluate specialists individually
    print("\n[3] Specialist Performance...")
    print("\n  ProTherm Specialist:")
    protherm_on_protherm = evaluate(specialist_protherm, X_protherm_val, y_protherm_val)
    protherm_on_s669 = evaluate(specialist_protherm, X_s669_val, y_s669_val)
    print(f"    On ProTherm: Spearman = {protherm_on_protherm['spearman']:.4f}")
    print(f"    On S669: Spearman = {protherm_on_s669['spearman']:.4f}")

    print("\n  S669 Specialist:")
    s669_on_protherm = evaluate(specialist_s669, X_protherm_val, y_protherm_val)
    s669_on_s669 = evaluate(specialist_s669, X_s669_val, y_s669_val)
    print(f"    On ProTherm: Spearman = {s669_on_protherm['spearman']:.4f}")
    print(f"    On S669: Spearman = {s669_on_s669['spearman']:.4f}")

    # Create ensemble
    print("\n[4] Training ENSEMBLE...")
    specialists = [specialist_protherm.to(device), specialist_s669.to(device)]
    ensemble = LearnedEnsemble(specialists, input_dim=X_protherm.shape[1])

    # Train on combined data
    X_combined_train = np.vstack([X_protherm_train, X_s669_train])
    y_combined_train = np.concatenate([y_protherm_train, y_s669_train])
    X_combined_val = np.vstack([X_protherm_val, X_s669_val])
    y_combined_val = np.concatenate([y_protherm_val, y_s669_val])

    print(f"  Training on {len(X_combined_train)} combined samples...")
    ensemble, ensemble_history = train_ensemble(
        ensemble,
        X_combined_train,
        y_combined_train,
        X_combined_val,
        y_combined_val,
        config,
    )

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    ensemble_protherm = evaluate(ensemble, X_protherm_val, y_protherm_val, is_ensemble=True)
    ensemble_s669 = evaluate(ensemble, X_s669_val, y_s669_val, is_ensemble=True)
    ensemble_combined = evaluate(ensemble, X_combined_val, y_combined_val, is_ensemble=True)

    print(f"""
| Model                 | ProTherm | S669   | Combined |
|-----------------------|----------|--------|----------|
| ProTherm Specialist   | {protherm_on_protherm["spearman"]:.4f}   | {protherm_on_s669["spearman"]:.4f} | -        |
| S669 Specialist       | {s669_on_protherm["spearman"]:.4f}   | {s669_on_s669["spearman"]:.4f} | -        |
| Simple Average        | {ensemble_protherm["simple_avg_spearman"]:.4f}   | {ensemble_s669["simple_avg_spearman"]:.4f} | {ensemble_combined["simple_avg_spearman"]:.4f}   |
| Learned Ensemble      | {ensemble_protherm["spearman"]:.4f}   | {ensemble_s669["spearman"]:.4f} | {ensemble_combined["spearman"]:.4f}   |

Learned Weights (avg): {ensemble_combined["weights"]}
  - ProTherm specialist: {ensemble_combined["weights"][0]:.2f}
  - S669 specialist: {ensemble_combined["weights"][1]:.2f}
""")

    # Save results
    results = {
        "specialists": {
            "protherm": {"on_protherm": protherm_on_protherm, "on_s669": protherm_on_s669},
            "s669": {"on_protherm": s669_on_protherm, "on_s669": s669_on_s669},
        },
        "ensemble": {
            "protherm": ensemble_protherm,
            "s669": ensemble_s669,
            "combined": ensemble_combined,
        },
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    torch.save({"model_state_dict": specialist_protherm.state_dict()}, output_dir / "specialist_protherm.pt")
    torch.save({"model_state_dict": specialist_s669.state_dict()}, output_dir / "specialist_s669.pt")
    torch.save({"model_state_dict": ensemble.state_dict()}, output_dir / "ensemble.pt")

    print(f"\nResults saved to: {output_dir}")

    return ensemble, results


if __name__ == "__main__":
    main()
