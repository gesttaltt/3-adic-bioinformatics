"""Phase 1 Improvements: TAM Integration, Stable Transformer, MAML Evaluation.

This script implements the immediate priority improvements:
1. TAM encoding integration for NRTI drugs
2. Numerically stable transformer for long sequences
3. MAML few-shot evaluation

Run with: python src/scripts/experiments/run_phase1_improvements.py
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
# TAM-Aware Encoding (from src/encoding/tam_aware_encoder.py)
# =============================================================================

TAM_PATHWAYS = {
    "TAM1": {"mutations": ["M41L", "L210W", "T215Y"], "positions": [41, 210, 215]},
    "TAM2": {"mutations": ["D67N", "K70R", "T215F", "K219Q", "K219E"], "positions": [67, 70, 215, 219]},
    "Q151M": {"mutations": ["A62V", "V75I", "F77L", "F116Y", "Q151M"], "positions": [62, 75, 77, 116, 151]},
    "K65R": {"mutations": ["K65R"], "positions": [65]},
    "M184": {"mutations": ["M184V", "M184I"], "positions": [184]},
    "L74V": {"mutations": ["L74V", "L74I"], "positions": [74]},
}

NRTI_KEY_POSITIONS = [41, 44, 62, 65, 67, 69, 70, 74, 75, 77, 115, 116, 118, 151, 184, 210, 215, 219]

REFERENCE_AA = {
    41: "M",
    44: "E",
    62: "A",
    65: "K",
    67: "D",
    69: "T",
    70: "K",
    74: "L",
    75: "V",
    77: "F",
    115: "Y",
    116: "F",
    118: "V",
    151: "Q",
    184: "M",
    210: "L",
    215: "T",
    219: "K",
}


def detect_mutations(row: pd.Series, position_cols: list[str]) -> set[str]:
    """Detect mutations from a data row."""
    mutations = set()
    for col in position_cols:
        # Position columns use P prefix (e.g., P41, P184)
        if col.startswith("P"):
            try:
                pos = int(col[1:])
                aa = str(row[col]).upper() if pd.notna(row[col]) else ""
                if aa and pos in REFERENCE_AA and aa != REFERENCE_AA[pos] and aa != "-":
                    mutations.add(f"{REFERENCE_AA[pos]}{pos}{aa}")
            except (ValueError, KeyError):
                continue
    return mutations


def extract_tam_features(row: pd.Series, position_cols: list[str]) -> np.ndarray:
    """Extract TAM pattern features from a row."""
    mutations = detect_mutations(row, position_cols)

    features = []

    # Pattern scores (6 patterns)
    for _pattern_name, pattern_info in TAM_PATHWAYS.items():
        present = sum(1 for mut in pattern_info["mutations"] if mut in mutations)
        score = present / len(pattern_info["mutations"]) if pattern_info["mutations"] else 0
        features.append(score)

    # Total TAM count (normalized)
    tam_muts = {"M41L", "D67N", "K70R", "L210W", "T215Y", "T215F", "K219Q", "K219E"}
    tam_count = len(mutations & tam_muts) / len(tam_muts)
    features.append(tam_count)

    # Key position indicators (18 positions)
    for pos in NRTI_KEY_POSITIONS:
        col = f"P{pos}"  # Use P prefix matching actual data format
        if col in position_cols:
            try:
                aa = str(row.get(col, "")).upper()
                ref = REFERENCE_AA.get(pos, "")
                is_mutated = 1.0 if aa and aa != ref and aa != "-" else 0.0
            except:
                is_mutated = 0.0
            features.append(is_mutated)
        else:
            features.append(0.0)

    # Pathway interactions (4 features)
    tam1 = features[0]  # TAM1 score
    tam2 = features[1]  # TAM2 score
    q151m = features[2]  # Q151M score
    m184 = features[4]  # M184 score
    k65r = features[3]  # K65R score

    features.append(tam1 * tam2)  # TAM1+TAM2 interaction
    features.append(q151m * (tam1 + tam2) / 2)  # Q151M + TAMs
    features.append(m184 * tam1)  # M184V + TAM1
    features.append(k65r * m184)  # K65R + M184V antagonism

    return np.array(features, dtype=np.float32)


def encode_with_tam(df: pd.DataFrame, position_cols: list[str]) -> np.ndarray:
    """Encode sequences with both one-hot and TAM features."""
    aa_alphabet = "ACDEFGHIKLMNPQRSTVWY*-"
    aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}

    n_samples = len(df)
    n_positions = len(position_cols)
    n_aa = len(aa_alphabet)

    # One-hot encoding
    onehot = np.zeros((n_samples, n_positions * n_aa), dtype=np.float32)

    # TAM features (6 patterns + 1 count + 18 positions + 4 interactions = 29)
    n_tam_features = 29
    tam_features = np.zeros((n_samples, n_tam_features), dtype=np.float32)

    for idx, (_, row) in enumerate(df.iterrows()):
        # One-hot
        for j, col in enumerate(position_cols):
            aa = str(row[col]).upper() if pd.notna(row[col]) else "-"
            if aa in aa_to_idx:
                onehot[idx, j * n_aa + aa_to_idx[aa]] = 1.0
            else:
                onehot[idx, j * n_aa + aa_to_idx["-"]] = 1.0

        # TAM features
        tam_features[idx] = extract_tam_features(row, position_cols)

    # Concatenate
    return np.concatenate([onehot, tam_features], axis=1)


# =============================================================================
# Stable Transformer for Long Sequences
# =============================================================================


class StableTransformer(nn.Module):
    """Numerically stable transformer for long sequences."""

    def __init__(
        self,
        n_positions: int,
        n_aa: int = 22,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_sparse_attention: bool = False,
        window_size: int = 64,
    ):
        super().__init__()
        self.n_positions = n_positions
        self.n_aa = n_aa
        self.d_model = d_model
        self.use_sparse_attention = use_sparse_attention
        self.window_size = window_size

        # Embedding with proper initialization
        self.aa_embed = nn.Linear(n_aa, d_model)
        nn.init.xavier_uniform_(self.aa_embed.weight)

        # Positional encoding (learned, more stable than sinusoidal for long seqs)
        self.pos_embed = nn.Parameter(torch.randn(1, n_positions, d_model) * 0.02)

        # Pre-norm transformer (more stable)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,  # Smaller FF for stability
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output layers
        self.norm = nn.LayerNorm(d_model)
        self.fc_mu = nn.Linear(d_model, 16)
        self.fc_logvar = nn.Linear(d_model, 16)
        self.decoder = nn.Linear(16, n_positions * n_aa)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = x.size(0)

        # Handle TAM-enhanced input
        if x.size(1) > self.n_positions * self.n_aa:
            # Split off TAM features
            onehot = x[:, : self.n_positions * self.n_aa]
            tam = x[:, self.n_positions * self.n_aa :]
        else:
            onehot = x
            tam = None

        # Reshape to (batch, positions, aa)
        onehot = onehot.view(batch_size, self.n_positions, self.n_aa)

        # Embed
        h = self.aa_embed(onehot)
        h = h + self.pos_embed[:, : self.n_positions]

        # Apply transformer with gradient checkpointing for memory
        if self.training and self.n_positions > 200:
            h = torch.utils.checkpoint.checkpoint(self.transformer, h, use_reentrant=False)
        else:
            h = self.transformer(h)

        # Pool
        h = h.mean(dim=1)
        h = self.norm(h)

        # If TAM features exist, incorporate them
        if tam is not None:
            tam_proj = nn.functional.linear(tam, torch.randn(self.d_model, tam.size(1), device=x.device) * 0.01)
            h = h + 0.1 * tam_proj

        # Latent
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10, 10)  # Clamp for stability

        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        # Decode (only one-hot part)
        x_recon = self.decoder(z)

        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "prediction": z[:, 0],
        }


# =============================================================================
# MAML Implementation for Few-Shot
# =============================================================================


class MAMLVAE(nn.Module):
    """VAE for MAML meta-learning."""

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()

        # Use LayerNorm instead of BatchNorm for MAML
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.GELU(),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
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


def maml_inner_loop(
    model: nn.Module,
    support_x: torch.Tensor,
    support_y: torch.Tensor,
    inner_lr: float = 0.01,
    inner_steps: int = 5,
) -> nn.Module:
    """MAML inner loop: adapt model to task."""
    import copy

    adapted = copy.deepcopy(model)

    for _ in range(inner_steps):
        out = adapted(support_x)
        loss = compute_maml_loss(out, support_x, support_y)

        grads = torch.autograd.grad(loss, adapted.parameters(), create_graph=False)

        with torch.no_grad():
            for param, grad in zip(adapted.parameters(), grads, strict=False):
                param.sub_(inner_lr * grad)

    return adapted


def compute_maml_loss(out: dict, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute loss for MAML."""
    recon = F.mse_loss(out["x_recon"], x)
    kl = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())

    pred = out["prediction"]
    p_c = pred - pred.mean()
    y_c = y - y.mean()
    corr = torch.sum(p_c * y_c) / (torch.sqrt(torch.sum(p_c**2) + 1e-8) * torch.sqrt(torch.sum(y_c**2) + 1e-8))

    return recon + 0.001 * kl + 0.3 * (-corr)


# =============================================================================
# TAM-Specific Loss Function
# =============================================================================


class TAMSpecificLoss(nn.Module):
    """Loss function that incorporates TAM pathway knowledge."""

    def __init__(self, n_positions: int = 560, ranking_weight: float = 0.3):
        super().__init__()
        self.n_positions = n_positions
        self.ranking_weight = ranking_weight

        # Position importance weights (higher for key NRTI positions)
        self.position_weights = torch.ones(n_positions)
        for pos in NRTI_KEY_POSITIONS:
            if pos < n_positions:
                self.position_weights[pos] = 3.0  # 3x weight for key positions

        # TAM pattern predictor
        self.tam_predictor = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, len(TAM_PATHWAYS)),
            nn.Sigmoid(),
        )

    def forward(
        self,
        out: dict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        tam_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        losses = {}

        # Weighted reconstruction (emphasize key positions)
        x_onehot = x[:, : self.n_positions * 22].view(-1, self.n_positions, 22)
        recon = out["x_recon"].view(-1, self.n_positions, 22)

        weights = self.position_weights.to(x.device).unsqueeze(0).unsqueeze(-1)
        weighted_recon = (weights * (x_onehot - recon) ** 2).mean()
        losses["recon"] = weighted_recon

        # KL divergence
        kl = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
        losses["kl"] = 0.001 * kl

        # Ranking loss
        pred = out["prediction"]
        p_c = pred - pred.mean()
        y_c = y - y.mean()
        p_std = torch.sqrt(torch.sum(p_c**2) + 1e-8)
        y_std = torch.sqrt(torch.sum(y_c**2) + 1e-8)
        corr = torch.sum(p_c * y_c) / (p_std * y_std)
        losses["rank"] = self.ranking_weight * (-corr)

        # TAM consistency loss (if TAM features provided)
        if tam_features is not None:
            tam_pred = self.tam_predictor(out["z"])
            tam_target = tam_features[:, : len(TAM_PATHWAYS)]  # First 6 features are pattern scores
            losses["tam"] = 0.1 * F.mse_loss(tam_pred, tam_target)

        losses["total"] = sum(losses.values())
        return losses


# =============================================================================
# Data Loading
# =============================================================================


def load_data(drug_class: str) -> tuple[pd.DataFrame, list[str], list[str]]:
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

    # Verify positions are reasonable for the drug class
    expected_positions = {"pi": 99, "nrti": 240, "nnrti": 240, "ini": 288}
    if len(position_cols) == 0:
        raise ValueError(f"No position columns found for {drug_class}")
    print(
        f"  Found {len(position_cols)} positions for {drug_class} (expected ~{expected_positions.get(drug_class, 'unknown')})"
    )

    return df, position_cols, drug_columns[drug_class]


def prepare_data(
    drug_class: str,
    drug: str,
    use_tam: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Prepare data with optional TAM encoding."""
    df, position_cols, drugs = load_data(drug_class)
    df_valid = df[df[drug].notna() & (df[drug] > 0)].copy()

    if len(df_valid) < 100:
        raise ValueError(f"Not enough samples: {len(df_valid)}")

    # Encode
    if use_tam and drug_class in ["nrti", "nnrti"]:
        X = encode_with_tam(df_valid, position_cols)
    else:
        # Standard one-hot
        aa_alphabet = "ACDEFGHIKLMNPQRSTVWY*-"
        aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}
        n_positions = len(position_cols)
        X = np.zeros((len(df_valid), n_positions * 22), dtype=np.float32)
        for idx, (_, row) in enumerate(df_valid.iterrows()):
            for j, col in enumerate(position_cols):
                aa = str(row[col]).upper() if pd.notna(row[col]) else "-"
                if aa in aa_to_idx:
                    X[idx, j * 22 + aa_to_idx[aa]] = 1.0
                else:
                    X[idx, j * 22 + aa_to_idx["-"]] = 1.0

    # Target
    y = np.log10(df_valid[drug].values + 1).astype(np.float32)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return (
        torch.tensor(X_train),
        torch.tensor(y_train),
        torch.tensor(X_test),
        torch.tensor(y_test),
        X.shape[1],
        len(position_cols),
    )


# =============================================================================
# Training Functions
# =============================================================================


def train_with_tam_loss(
    model: nn.Module,
    loss_fn: TAMSpecificLoss,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    epochs: int = 50,
    device: str = "cpu",
) -> tuple[float, list[float]]:
    """Train with TAM-specific loss."""
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(loss_fn.parameters()), lr=0.001)

    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)

    best_corr = -1.0
    history = []

    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            # Extract TAM features if present
            n_onehot = loss_fn.n_positions * 22
            tam_features = x[:, n_onehot:] if x.size(1) > n_onehot else None

            optimizer.zero_grad()
            out = model(x)
            losses = loss_fn(out, x, y, tam_features)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Evaluate
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(test_x.to(device))
                pred = out["prediction"].cpu().numpy()
                corr = np.corrcoef(pred, test_y.numpy())[0, 1]
                if not np.isnan(corr) and corr > best_corr:
                    best_corr = corr
                history.append(corr)

    return best_corr, history


def train_standard(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    epochs: int = 50,
    device: str = "cpu",
) -> float:
    """Standard training loop."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)
    best_corr = -1.0

    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            out = model(x)

            # Standard loss
            recon = F.mse_loss(out["x_recon"], x[:, : out["x_recon"].size(1)])
            kl = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())

            pred = out["prediction"]
            p_c = pred - pred.mean()
            y_c = y - y.mean()
            corr = torch.sum(p_c * y_c) / (torch.sqrt(torch.sum(p_c**2) + 1e-8) * torch.sqrt(torch.sum(y_c**2) + 1e-8))

            loss = recon + 0.001 * kl + 0.3 * (-corr)

            if torch.isnan(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(test_x.to(device))
                pred = out["prediction"].cpu().numpy()
                corr = np.corrcoef(pred, test_y.numpy())[0, 1]
                if not np.isnan(corr) and corr > best_corr:
                    best_corr = corr

    return best_corr


def run_maml_evaluation(
    train_drugs: list[str],
    test_drugs: list[str],
    drug_class: str = "pi",
    support_sizes: list[int] = None,
    epochs: int = 30,
    device: str = "cpu",
) -> dict:
    """Run MAML few-shot evaluation."""
    if support_sizes is None:
        support_sizes = [5, 10, 20, 50]
    print("\n" + "=" * 60)
    print("MAML FEW-SHOT EVALUATION")
    print("=" * 60)

    # Load all drug data
    all_data = {}
    input_dim = None
    for drug in train_drugs + test_drugs:
        try:
            train_x, train_y, test_x, test_y, dim, _ = prepare_data(drug_class, drug)
            all_data[drug] = {
                "train_x": train_x,
                "train_y": train_y,
                "test_x": test_x,
                "test_y": test_y,
            }
            input_dim = dim
        except Exception as e:
            print(f"  Skipping {drug}: {e}")

    if input_dim is None:
        return {"error": "No valid drugs"}

    # Create meta-model
    meta_model = MAMLVAE(input_dim).to(device)
    meta_optimizer = torch.optim.Adam(meta_model.parameters(), lr=0.001)

    # Meta-training on train_drugs
    print(f"\nMeta-training on: {train_drugs}")
    for epoch in range(epochs):
        meta_optimizer.zero_grad()
        meta_loss = 0.0

        for drug in train_drugs:
            if drug not in all_data:
                continue

            data = all_data[drug]

            # Sample support and query sets
            n = len(data["train_x"])
            idx = torch.randperm(n)
            n_support = min(20, n // 2)

            support_x = data["train_x"][idx[:n_support]].to(device)
            support_y = data["train_y"][idx[:n_support]].to(device)
            query_x = data["train_x"][idx[n_support : n_support + 20]].to(device)
            query_y = data["train_y"][idx[n_support : n_support + 20]].to(device)

            # Inner loop
            adapted = maml_inner_loop(meta_model, support_x, support_y)

            # Compute query loss
            out = adapted(query_x)
            query_loss = compute_maml_loss(out, query_x, query_y)
            meta_loss += query_loss

        meta_loss = meta_loss / len(train_drugs)
        meta_loss.backward()
        meta_optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}: meta_loss = {meta_loss.item():.4f}")

    # Evaluate on test_drugs with different support sizes
    results = {}
    print(f"\nEvaluating on: {test_drugs}")

    for drug in test_drugs:
        if drug not in all_data:
            continue

        data = all_data[drug]
        results[drug] = {}

        for n_support in support_sizes:
            if n_support > len(data["train_x"]):
                continue

            # Sample support set
            idx = torch.randperm(len(data["train_x"]))[:n_support]
            support_x = data["train_x"][idx].to(device)
            support_y = data["train_y"][idx].to(device)

            # Adapt
            adapted = maml_inner_loop(meta_model, support_x, support_y, inner_steps=10)

            # Evaluate on test set
            adapted.eval()
            with torch.no_grad():
                out = adapted(data["test_x"].to(device))
                pred = out["prediction"].cpu().numpy()
                corr = np.corrcoef(pred, data["test_y"].numpy())[0, 1]

            results[drug][n_support] = corr
            print(f"  {drug} (n={n_support}): {corr:+.4f}")

    return results


# =============================================================================
# Main Experiment Runner
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--experiment", type=str, default="all", choices=["tam", "transformer", "maml", "tam_loss", "multitask", "all"]
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results_all = {}

    # ==========================================================================
    # Experiment 1: TAM Encoding for NRTIs
    # ==========================================================================
    if args.experiment in ["tam", "all"]:
        print("\n" + "=" * 80)
        print("EXPERIMENT 1: TAM ENCODING FOR NRTI DRUGS")
        print("=" * 80)

        nrti_drugs = ["ABC", "AZT", "D4T", "DDI", "3TC", "TDF"]
        results_tam = []

        for drug in nrti_drugs:
            print(f"\n--- {drug} ---")
            try:
                # Without TAM
                train_x, train_y, test_x, test_y, input_dim, n_pos = prepare_data("nrti", drug, use_tam=False)

                class SimpleVAE(nn.Module):
                    def __init__(self, dim):
                        super().__init__()
                        self.enc = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU())
                        self.fc_mu = nn.Linear(64, 16)
                        self.fc_logvar = nn.Linear(64, 16)
                        self.dec = nn.Linear(16, dim)

                    def forward(self, x):
                        h = self.enc(x)
                        mu = self.fc_mu(h)
                        logvar = self.fc_logvar(h)
                        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                        return {"x_recon": self.dec(z), "mu": mu, "logvar": logvar, "z": z, "prediction": z[:, 0]}

                model_no_tam = SimpleVAE(input_dim)
                corr_no_tam = train_standard(model_no_tam, train_x, train_y, test_x, test_y, args.epochs, device)
                print(f"  Without TAM: {corr_no_tam:+.4f}")

                # With TAM
                train_x_tam, train_y_tam, test_x_tam, test_y_tam, input_dim_tam, _ = prepare_data(
                    "nrti", drug, use_tam=True
                )
                model_with_tam = SimpleVAE(input_dim_tam)
                corr_with_tam = train_standard(
                    model_with_tam, train_x_tam, train_y_tam, test_x_tam, test_y_tam, args.epochs, device
                )
                print(f"  With TAM:    {corr_with_tam:+.4f}")
                print(f"  Improvement: {corr_with_tam - corr_no_tam:+.4f}")

                results_tam.append(
                    {
                        "drug": drug,
                        "no_tam": corr_no_tam,
                        "with_tam": corr_with_tam,
                        "improvement": corr_with_tam - corr_no_tam,
                    }
                )

            except Exception as e:
                print(f"  Error: {e}")

        results_all["tam"] = results_tam

        # Summary
        if results_tam:
            print("\n--- TAM Encoding Summary ---")
            avg_no_tam = np.mean([r["no_tam"] for r in results_tam])
            avg_with_tam = np.mean([r["with_tam"] for r in results_tam])
            print(f"Average without TAM: {avg_no_tam:+.4f}")
            print(f"Average with TAM:    {avg_with_tam:+.4f}")
            print(f"Average improvement: {avg_with_tam - avg_no_tam:+.4f}")

    # ==========================================================================
    # Experiment 2: Stable Transformer
    # ==========================================================================
    if args.experiment in ["transformer", "all"]:
        print("\n" + "=" * 80)
        print("EXPERIMENT 2: STABLE TRANSFORMER FOR LONG SEQUENCES")
        print("=" * 80)

        results_trans = []

        for drug_class, drugs in [("pi", ["LPV", "DRV"]), ("nrti", ["AZT", "3TC"])]:
            for drug in drugs:
                print(f"\n--- {drug} ({drug_class.upper()}) ---")
                try:
                    train_x, train_y, test_x, test_y, input_dim, n_pos = prepare_data(drug_class, drug)
                    print(f"  Positions: {n_pos}, Input dim: {input_dim}")

                    model = StableTransformer(n_positions=n_pos, d_model=64, n_layers=2)
                    corr = train_standard(model, train_x, train_y, test_x, test_y, args.epochs, device)
                    print(f"  Stable Transformer: {corr:+.4f}")

                    results_trans.append({"drug": drug, "class": drug_class, "correlation": corr})

                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback

                    traceback.print_exc()

        results_all["transformer"] = results_trans

    # ==========================================================================
    # Experiment 3: MAML Few-Shot
    # ==========================================================================
    if args.experiment in ["maml", "all"]:
        print("\n" + "=" * 80)
        print("EXPERIMENT 3: MAML FEW-SHOT LEARNING")
        print("=" * 80)

        # Meta-train on 6 PI drugs, meta-test on 2
        train_drugs = ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV"]
        test_drugs = ["TPV", "DRV"]

        maml_results = run_maml_evaluation(
            train_drugs, test_drugs, "pi", support_sizes=[5, 10, 20, 50], epochs=30, device=device
        )
        results_all["maml"] = maml_results

    # ==========================================================================
    # Experiment 4: TAM-Specific Loss
    # ==========================================================================
    if args.experiment in ["tam_loss", "all"]:
        print("\n" + "=" * 80)
        print("EXPERIMENT 4: TAM-SPECIFIC LOSS FUNCTION")
        print("=" * 80)

        results_tam_loss = []

        for drug in ["AZT", "3TC", "TDF"]:
            print(f"\n--- {drug} ---")
            try:
                train_x, train_y, test_x, test_y, input_dim, n_pos = prepare_data("nrti", drug, use_tam=True)

                # Standard loss
                class SimpleVAE(nn.Module):
                    def __init__(self, dim, n_pos):
                        super().__init__()
                        self.n_pos = n_pos
                        self.enc = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU())
                        self.fc_mu = nn.Linear(64, 16)
                        self.fc_logvar = nn.Linear(64, 16)
                        self.dec = nn.Linear(16, n_pos * 22)

                    def forward(self, x):
                        h = self.enc(x)
                        mu = self.fc_mu(h)
                        logvar = self.fc_logvar(h)
                        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                        return {"x_recon": self.dec(z), "mu": mu, "logvar": logvar, "z": z, "prediction": z[:, 0]}

                model_std = SimpleVAE(input_dim, n_pos)
                corr_std = train_standard(model_std, train_x, train_y, test_x, test_y, args.epochs, device)
                print(f"  Standard loss: {corr_std:+.4f}")

                # TAM-specific loss
                model_tam = SimpleVAE(input_dim, n_pos)
                tam_loss = TAMSpecificLoss(n_positions=n_pos)
                corr_tam, _ = train_with_tam_loss(
                    model_tam, tam_loss, train_x, train_y, test_x, test_y, args.epochs, device
                )
                print(f"  TAM loss:      {corr_tam:+.4f}")
                print(f"  Improvement:   {corr_tam - corr_std:+.4f}")

                results_tam_loss.append(
                    {
                        "drug": drug,
                        "standard": corr_std,
                        "tam_loss": corr_tam,
                        "improvement": corr_tam - corr_std,
                    }
                )

            except Exception as e:
                print(f"  Error: {e}")
                import traceback

                traceback.print_exc()

        results_all["tam_loss"] = results_tam_loss

    # ==========================================================================
    # Experiment 5: Multi-Task Training for PI Drugs
    # ==========================================================================
    if args.experiment in ["multitask", "all"]:
        print("\n" + "=" * 80)
        print("EXPERIMENT 5: MULTI-TASK TRAINING FOR ALL PI DRUGS")
        print("=" * 80)

        pi_drugs = ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"]

        # Load data for all PI drugs
        print("\nLoading data for all PI drugs...")
        all_drug_data = {}
        df, position_cols, _ = load_data("pi")
        input_dim = len(position_cols) * 22

        for drug in pi_drugs:
            df_valid = df[df[drug].notna() & (df[drug] > 0)].copy()
            if len(df_valid) < 100:
                print(f"  Skipping {drug}: only {len(df_valid)} samples")
                continue

            # Encode
            aa_alphabet = "ACDEFGHIKLMNPQRSTVWY*-"
            aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}
            X = np.zeros((len(df_valid), input_dim), dtype=np.float32)
            for idx, (_, row) in enumerate(df_valid.iterrows()):
                for j, col in enumerate(position_cols):
                    aa = str(row[col]).upper() if pd.notna(row[col]) else "-"
                    if aa in aa_to_idx:
                        X[idx, j * 22 + aa_to_idx[aa]] = 1.0
                    else:
                        X[idx, j * 22 + aa_to_idx["-"]] = 1.0

            y = np.log10(df_valid[drug].values + 1).astype(np.float32)
            y = (y - y.min()) / (y.max() - y.min() + 1e-8)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            all_drug_data[drug] = {
                "train_x": torch.tensor(X_train),
                "train_y": torch.tensor(y_train),
                "test_x": torch.tensor(X_test),
                "test_y": torch.tensor(y_test),
            }
            print(f"  {drug}: {len(X_train)} train, {len(X_test)} test")

        # Create multi-task model
        class MultiTaskVAE(nn.Module):
            """Multi-task VAE with shared encoder and drug-specific heads."""

            def __init__(self, input_dim, drug_names):
                super().__init__()
                self.drug_names = drug_names

                # Shared encoder (use LayerNorm for variable batch sizes)
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
                self.fc_mu = nn.Linear(64, 16)
                self.fc_logvar = nn.Linear(64, 16)

                # Shared decoder
                self.decoder = nn.Sequential(
                    nn.Linear(16, 64),
                    nn.GELU(),
                    nn.Linear(64, 128),
                    nn.GELU(),
                    nn.Linear(128, 256),
                    nn.GELU(),
                    nn.Linear(256, input_dim),
                )

                # Drug-specific prediction heads
                self.drug_heads = nn.ModuleDict(
                    {
                        drug: nn.Sequential(
                            nn.Linear(16, 32),
                            nn.GELU(),
                            nn.Dropout(0.1),
                            nn.Linear(32, 1),
                        )
                        for drug in drug_names
                    }
                )

            def forward(self, x, drug=None):
                h = self.encoder(x)
                mu = self.fc_mu(h)
                logvar = self.fc_logvar(h)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                x_recon = self.decoder(z)

                result = {"x_recon": x_recon, "mu": mu, "logvar": logvar, "z": z}

                if drug is not None:
                    result["prediction"] = self.drug_heads[drug](z).squeeze(-1)
                else:
                    result["predictions"] = {name: self.drug_heads[name](z).squeeze(-1) for name in self.drug_names}

                return result

        # Train single-task baselines
        print("\n--- Single-Task Baselines ---")
        single_task_results = {}

        class SimpleVAE(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.enc = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU())
                self.fc_mu = nn.Linear(64, 16)
                self.fc_logvar = nn.Linear(64, 16)
                self.dec = nn.Linear(16, dim)

            def forward(self, x):
                h = self.enc(x)
                mu = self.fc_mu(h)
                logvar = self.fc_logvar(h)
                z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
                return {"x_recon": self.dec(z), "mu": mu, "logvar": logvar, "z": z, "prediction": z[:, 0]}

        for drug in all_drug_data:
            data = all_drug_data[drug]
            model = SimpleVAE(input_dim)
            corr = train_standard(
                model, data["train_x"], data["train_y"], data["test_x"], data["test_y"], args.epochs, device
            )
            single_task_results[drug] = corr
            print(f"  {drug}: {corr:+.4f}")

        # Train multi-task model
        print("\n--- Multi-Task Training ---")
        model = MultiTaskVAE(input_dim, list(all_drug_data.keys())).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        # Create unified training batches
        train_data = []
        for drug, data in all_drug_data.items():
            for i in range(len(data["train_x"])):
                train_data.append((data["train_x"][i], data["train_y"][i], drug))

        np.random.shuffle(train_data)

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0

            # Mini-batch training
            batch_size = 32
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i : i + batch_size]

                # Group by drug for efficient processing
                drug_batches = {}
                for x, y, drug in batch:
                    if drug not in drug_batches:
                        drug_batches[drug] = {"x": [], "y": []}
                    drug_batches[drug]["x"].append(x)
                    drug_batches[drug]["y"].append(y)

                optimizer.zero_grad()
                batch_loss = 0.0

                for drug, drug_data in drug_batches.items():
                    x = torch.stack(drug_data["x"]).to(device)
                    y = torch.tensor(drug_data["y"]).to(device)

                    out = model(x, drug=drug)

                    # Loss
                    recon = F.mse_loss(out["x_recon"], x)
                    kl = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())

                    pred = out["prediction"]
                    p_c = pred - pred.mean()
                    y_c = y - y.mean()
                    corr = torch.sum(p_c * y_c) / (
                        torch.sqrt(torch.sum(p_c**2) + 1e-8) * torch.sqrt(torch.sum(y_c**2) + 1e-8)
                    )

                    loss = recon + 0.001 * kl + 0.3 * (-corr)
                    batch_loss = batch_loss + loss

                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += batch_loss.item()

            scheduler.step()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}: loss = {total_loss / len(train_data) * batch_size:.4f}")

        # Evaluate multi-task model
        print("\n--- Multi-Task Evaluation ---")
        model.eval()
        multi_task_results = {}

        with torch.no_grad():
            for drug, data in all_drug_data.items():
                out = model(data["test_x"].to(device), drug=drug)
                pred = out["prediction"].cpu().numpy()
                corr = np.corrcoef(pred, data["test_y"].numpy())[0, 1]
                multi_task_results[drug] = corr

        # Summary
        print("\n--- Comparison: Single-Task vs Multi-Task ---")
        print(f"{'Drug':<8} {'Single-Task':>12} {'Multi-Task':>12} {'Difference':>12}")
        print("-" * 50)

        results_multitask = []
        for drug in sorted(all_drug_data.keys()):
            single = single_task_results.get(drug, 0)
            multi = multi_task_results.get(drug, 0)
            diff = multi - single
            print(f"{drug:<8} {single:>+12.4f} {multi:>+12.4f} {diff:>+12.4f}")
            results_multitask.append(
                {
                    "drug": drug,
                    "single_task": single,
                    "multi_task": multi,
                    "improvement": diff,
                }
            )

        avg_single = np.mean(list(single_task_results.values()))
        avg_multi = np.mean(list(multi_task_results.values()))
        print("-" * 50)
        print(f"{'AVERAGE':<8} {avg_single:>+12.4f} {avg_multi:>+12.4f} {avg_multi - avg_single:>+12.4f}")

        results_all["multitask"] = results_multitask

    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    if "tam" in results_all and results_all["tam"]:
        print("\nTAM Encoding Results:")
        for r in results_all["tam"]:
            print(f"  {r['drug']}: {r['no_tam']:+.4f} -> {r['with_tam']:+.4f} ({r['improvement']:+.4f})")

    if "transformer" in results_all and results_all["transformer"]:
        print("\nStable Transformer Results:")
        for r in results_all["transformer"]:
            print(f"  {r['drug']} ({r['class']}): {r['correlation']:+.4f}")

    if "maml" in results_all and results_all["maml"]:
        print("\nMAML Few-Shot Results:")
        for drug, sizes in results_all["maml"].items():
            if isinstance(sizes, dict):
                print(f"  {drug}:", end=" ")
                for n, corr in sizes.items():
                    print(f"n={n}: {corr:+.4f}", end=" ")
                print()

    if "tam_loss" in results_all and results_all["tam_loss"]:
        print("\nTAM-Specific Loss Results:")
        for r in results_all["tam_loss"]:
            print(f"  {r['drug']}: {r['standard']:+.4f} -> {r['tam_loss']:+.4f} ({r['improvement']:+.4f})")

    if "multitask" in results_all and results_all["multitask"]:
        print("\nMulti-Task Training Results:")
        for r in results_all["multitask"]:
            print(f"  {r['drug']}: {r['single_task']:+.4f} -> {r['multi_task']:+.4f} ({r['improvement']:+.4f})")

    # Save results
    results_path = project_root / "results" / "phase1_improvements.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for exp_name, exp_results in results_all.items():
        if isinstance(exp_results, list):
            for r in exp_results:
                r["experiment"] = exp_name
                all_rows.append(r)

    if all_rows:
        pd.DataFrame(all_rows).to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
