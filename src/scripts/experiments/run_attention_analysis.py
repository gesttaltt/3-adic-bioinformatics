"""Attention Analysis for HIV Drug Resistance Models.

This script analyzes which positions the trained models attend to and
compares them with known resistance mutations from the literature.

Run with: python src/scripts/experiments/run_attention_analysis.py
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
# Known Resistance Mutations (from Stanford HIVDB)
# =============================================================================

KNOWN_MUTATIONS = {
    # PI (Protease) - positions 1-99
    "pi": {
        "major": [23, 24, 30, 32, 33, 46, 47, 48, 50, 53, 54, 73, 76, 82, 84, 88, 90],
        "accessory": [10, 11, 13, 16, 20, 34, 35, 36, 43, 58, 60, 62, 63, 64, 69, 71, 74, 77, 83, 85, 89, 93],
        "drug_specific": {
            "LPV": [32, 47, 50, 54, 76, 82, 84],
            "DRV": [32, 47, 50, 54, 76, 84],
            "ATV": [32, 48, 50, 54, 82, 84, 88],
            "FPV": [32, 47, 50, 54, 76, 82, 84],
            "IDV": [32, 46, 54, 76, 82, 84],
            "NFV": [30, 46, 54, 82, 84, 88, 90],
            "SQV": [48, 54, 82, 84, 88, 90],
            "TPV": [33, 47, 58, 74, 82, 83, 84],
        },
    },
    # NRTI (Reverse Transcriptase) - positions 1-240 (RT domain)
    "nrti": {
        "major": [41, 62, 65, 67, 69, 70, 74, 75, 77, 115, 116, 151, 184, 210, 215, 219],
        "tam": [41, 67, 70, 210, 215, 219],  # Thymidine Analog Mutations
        "drug_specific": {
            "AZT": [41, 67, 70, 210, 215, 219],  # TAMs
            "3TC": [65, 184],
            "TDF": [65, 70],
            "ABC": [65, 74, 115, 184],
            "D4T": [41, 67, 70, 75, 210, 215, 219],
            "DDI": [65, 74],
        },
    },
    # NNRTI (Reverse Transcriptase) - positions 1-240
    "nnrti": {
        "major": [100, 101, 103, 106, 138, 179, 181, 188, 190, 225, 227, 230],
        "drug_specific": {
            "EFV": [100, 101, 103, 106, 188, 190, 225],
            "NVP": [100, 101, 103, 106, 181, 188, 190],
            "ETR": [100, 101, 138, 179, 181],
            "RPV": [100, 101, 138, 179, 181, 227],
        },
    },
    # INI (Integrase) - positions 1-288
    "ini": {
        "major": [66, 92, 118, 121, 140, 143, 147, 148, 155, 263],
        "drug_specific": {
            "RAL": [66, 92, 140, 143, 148, 155],
            "EVG": [66, 92, 118, 121, 140, 143, 147, 148, 155],
            "DTG": [118, 140, 148, 263],
        },
    },
}


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


# =============================================================================
# Attention Model
# =============================================================================


class AttentionVAE(nn.Module):
    """VAE with extractable attention weights."""

    def __init__(self, input_dim: int, n_positions: int, n_aa: int = 22, latent_dim: int = 16):
        super().__init__()
        self.n_positions = n_positions
        self.n_aa = n_aa
        d_model = 64

        self.pos_embed = nn.Linear(n_aa, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_positions, d_model) * 0.02)

        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True, dropout=0.1)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
        self.norm2 = nn.LayerNorm(d_model)

        self.fc_mu = nn.Linear(d_model, latent_dim)
        self.fc_logvar = nn.Linear(d_model, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

        self.last_attn_weights = None

    def forward(self, x, return_attention: bool = False):
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, self.n_positions, self.n_aa)
        h = self.pos_embed(x_reshaped) + self.pos_encoding

        attn_out, attn_weights = self.attention(h, h, h, average_attn_weights=True)
        self.last_attn_weights = attn_weights
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ffn(h))
        h = h.mean(dim=1)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        x_recon = self.decoder(z)

        result = {"x_recon": x_recon, "mu": mu, "logvar": logvar, "z": z, "prediction": z[:, 0]}
        if return_attention:
            result["attention"] = attn_weights
        return result


# =============================================================================
# Analysis Functions
# =============================================================================


def train_attention_model(
    drug_class: str, drug: str, epochs: int = 50, device: str = "cuda"
) -> tuple[AttentionVAE, float]:
    """Train attention model and return it with test correlation."""
    df, position_cols, _ = load_data(drug_class)
    df_valid = df[df[drug].notna() & (df[drug] > 0)].copy()

    if len(df_valid) < 50:
        raise ValueError(f"Not enough samples: {len(df_valid)}")

    X = encode_sequences(df_valid, position_cols)
    y = np.log10(df_valid[drug].values + 1).astype(np.float32)
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_dim = X.shape[1]
    n_positions = len(position_cols)

    model = AttentionVAE(input_dim, n_positions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_x = torch.tensor(X_train)
    train_y = torch.tensor(y_train)
    test_x = torch.tensor(X_test)
    torch.tensor(y_test)

    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=32, shuffle=True)

    best_corr = -1.0
    for epoch in range(epochs):
        model.train()
        for x, y_batch in loader:
            x, y_batch = x.to(device), y_batch.to(device)
            optimizer.zero_grad()

            out = model(x)
            recon = F.mse_loss(out["x_recon"], x)
            kl = -0.5 * torch.mean(1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp())
            pred = out["prediction"]
            p_c = pred - pred.mean()
            y_c = y_batch - y_batch.mean()
            corr = torch.sum(p_c * y_c) / (torch.sqrt(torch.sum(p_c**2) + 1e-8) * torch.sqrt(torch.sum(y_c**2) + 1e-8))
            loss = recon + 0.001 * kl + 0.3 * (-corr)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                out = model(test_x.to(device))
                pred = out["prediction"].cpu().numpy()
                test_corr = np.corrcoef(pred, y_test)[0, 1]
                if not np.isnan(test_corr) and test_corr > best_corr:
                    best_corr = test_corr

    return model, best_corr, test_x, position_cols


def extract_attention_importance(model: AttentionVAE, test_x: torch.Tensor, device: str = "cuda") -> np.ndarray:
    """Extract average attention-based position importance."""
    model.eval()
    model.to(device)

    with torch.no_grad():
        _ = model(test_x.to(device), return_attention=True)
        attn = model.last_attn_weights.cpu().numpy()  # (batch, n_positions, n_positions)

    # Average attention weights across batch and heads
    avg_attn = attn.mean(axis=0)  # (n_positions, n_positions)

    # Position importance = how much attention each position receives
    position_importance = avg_attn.sum(axis=0)  # Sum of attention FROM all positions TO this position

    # Normalize
    position_importance = position_importance / position_importance.max()

    return position_importance


def compute_gradient_importance(model: AttentionVAE, test_x: torch.Tensor, device: str = "cuda") -> np.ndarray:
    """Compute gradient-based position importance."""
    model.eval()
    model.to(device)

    test_x = test_x.to(device)
    test_x.requires_grad = True

    out = model(test_x)
    prediction = out["prediction"]

    # Gradient of prediction w.r.t. input
    grad = torch.autograd.grad(prediction.sum(), test_x, create_graph=False)[0]
    grad = grad.abs().cpu().numpy()  # (batch, n_positions * n_aa)

    # Reshape to (batch, n_positions, n_aa)
    n_aa = 22
    n_positions = model.n_positions
    grad = grad.reshape(-1, n_positions, n_aa)

    # Sum across amino acids and batch
    position_importance = grad.sum(axis=(0, 2))  # (n_positions,)

    # Normalize
    position_importance = position_importance / position_importance.max()

    return position_importance


def compare_with_known_mutations(
    position_importance: np.ndarray,
    position_cols: list[str],
    drug_class: str,
    drug: str,
    top_k: int = 20,
) -> dict:
    """Compare model attention with known resistance mutations."""
    # Get top attended positions
    top_indices = np.argsort(position_importance)[::-1][:top_k]
    top_positions = [int(position_cols[i][1:]) for i in top_indices]  # Remove 'P' prefix

    # Known mutations
    known = KNOWN_MUTATIONS.get(drug_class, {})
    major_positions = set(known.get("major", []))
    drug_specific = set(known.get("drug_specific", {}).get(drug, []))
    all_known = major_positions | drug_specific

    # Compute overlap
    top_positions_set = set(top_positions)
    overlap = top_positions_set & all_known

    precision = len(overlap) / len(top_positions_set) if top_positions_set else 0
    recall = len(overlap) / len(all_known) if all_known else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "top_positions": top_positions,
        "known_major": sorted(major_positions),
        "known_drug_specific": sorted(drug_specific),
        "overlap": sorted(overlap),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# =============================================================================
# Main
# =============================================================================


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--drug-class", type=str, default="pi")
    parser.add_argument("--drug", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    drug_classes = {
        "pi": ["LPV", "DRV", "ATV"],
        "nrti": ["AZT", "3TC", "TDF"],
        "nnrti": ["EFV", "NVP"],
        "ini": ["RAL", "EVG"],
    }

    if args.drug:
        drugs_to_analyze = [(args.drug_class, args.drug)]
    else:
        drugs_to_analyze = [(args.drug_class, d) for d in drug_classes.get(args.drug_class, [])]

    all_results = []

    for drug_class, drug in drugs_to_analyze:
        print(f"\n{'=' * 60}")
        print(f"ATTENTION ANALYSIS: {drug} ({drug_class.upper()})")
        print("=" * 60)

        try:
            # Train model
            print("\nTraining attention model...")
            model, corr, test_x, position_cols = train_attention_model(drug_class, drug, args.epochs, device)
            print(f"Test correlation: {corr:+.4f}")

            # Extract attention importance
            print("\nExtracting attention importance...")
            attn_importance = extract_attention_importance(model, test_x, device)

            # Extract gradient importance
            print("Computing gradient importance...")
            grad_importance = compute_gradient_importance(model, test_x, device)

            # Compare with known mutations
            print("\nComparing with known resistance mutations...")
            attn_results = compare_with_known_mutations(attn_importance, position_cols, drug_class, drug)
            grad_results = compare_with_known_mutations(grad_importance, position_cols, drug_class, drug)

            print("\n--- Attention-based Analysis ---")
            print(f"Top 20 positions: {attn_results['top_positions']}")
            print(f"Known major:      {attn_results['known_major']}")
            print(f"Known drug-spec:  {attn_results['known_drug_specific']}")
            print(f"Overlap:          {attn_results['overlap']}")
            print(
                f"Precision: {attn_results['precision']:.2%} | Recall: {attn_results['recall']:.2%} | F1: {attn_results['f1']:.2%}"
            )

            print("\n--- Gradient-based Analysis ---")
            print(f"Top 20 positions: {grad_results['top_positions']}")
            print(f"Overlap:          {grad_results['overlap']}")
            print(
                f"Precision: {grad_results['precision']:.2%} | Recall: {grad_results['recall']:.2%} | F1: {grad_results['f1']:.2%}"
            )

            all_results.append(
                {
                    "drug": drug,
                    "class": drug_class,
                    "correlation": corr,
                    "attn_precision": attn_results["precision"],
                    "attn_recall": attn_results["recall"],
                    "attn_f1": attn_results["f1"],
                    "grad_precision": grad_results["precision"],
                    "grad_recall": grad_results["recall"],
                    "grad_f1": grad_results["f1"],
                    "n_known": len(attn_results["known_major"]) + len(attn_results["known_drug_specific"]),
                }
            )

        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    if all_results:
        print("\n" + "=" * 60)
        print("ATTENTION ANALYSIS SUMMARY")
        print("=" * 60)

        print(
            f"\n{'Drug':<8} {'Corr':>8} {'Attn P':>8} {'Attn R':>8} {'Attn F1':>8} {'Grad P':>8} {'Grad R':>8} {'Grad F1':>8}"
        )
        print("-" * 72)
        for r in all_results:
            print(
                f"{r['drug']:<8} {r['correlation']:>+8.4f} "
                f"{r['attn_precision']:>8.2%} {r['attn_recall']:>8.2%} {r['attn_f1']:>8.2%} "
                f"{r['grad_precision']:>8.2%} {r['grad_recall']:>8.2%} {r['grad_f1']:>8.2%}"
            )

        avg_attn_f1 = np.mean([r["attn_f1"] for r in all_results])
        avg_grad_f1 = np.mean([r["grad_f1"] for r in all_results])
        print("-" * 72)
        print(f"Average F1 - Attention: {avg_attn_f1:.2%} | Gradient: {avg_grad_f1:.2%}")

        # Save results
        results_path = project_root / "results" / "attention_analysis.csv"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_results).to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
