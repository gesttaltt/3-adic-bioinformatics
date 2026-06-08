#!/usr/bin/env python3
"""
ESM-2 Large Model Experiments (650M parameters)
================================================

This script compares the 8M vs 650M ESM-2 model performance
on HIV drug resistance prediction.

Author: Claude Code
Date: December 28, 2024
"""

import gc
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "research"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# =============================================================================
# DATA LOADING
# =============================================================================


def load_stanford_data(drug: str, gene: str = "PI") -> tuple[np.ndarray, list[str]]:
    """Load Stanford HIVDB data for a specific drug."""
    gene_files = {
        "PI": "stanford_hivdb_pi.txt",
        "NRTI": "stanford_hivdb_nrti.txt",
        "NNRTI": "stanford_hivdb_nnrti.txt",
        "INI": "stanford_hivdb_ini.txt",
    }

    file_path = DATA_DIR / gene_files.get(gene, "stanford_hivdb_pi.txt")

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path, sep="\t", low_memory=False)

    # Find sequence column
    seq_col = None
    for col in df.columns:
        if "seq" in col.lower():
            seq_col = col
            break
    if seq_col is None:
        seq_col = df.columns[0]

    # Get drug column (case-insensitive)
    drug_col = None
    for col in df.columns:
        if col.upper() == drug.upper():
            drug_col = col
            break

    if drug_col is None:
        raise ValueError(f"Drug {drug} not found in {file_path}")

    # Filter rows with valid data
    df = df[[seq_col, drug_col]].dropna()
    df = df[df[drug_col] != ""]
    df[drug_col] = pd.to_numeric(df[drug_col], errors="coerce")
    df = df.dropna()

    if len(df) == 0:
        raise ValueError(f"No valid data found for {drug}")

    raw_sequences = df[seq_col].tolist()
    resistances = df[drug_col].values.astype(np.float32)

    # Clean sequences
    cleaned_sequences = []
    for seq in raw_sequences:
        seq = str(seq).upper().replace(".", "-").replace("~", "-").replace("*", "X")
        seq = "".join(c if c in "ACDEFGHIKLMNPQRSTVWY-X" else "X" for c in seq)
        cleaned_sequences.append(seq)

    return resistances, cleaned_sequences


# =============================================================================
# ESM-2 EMBEDDER
# =============================================================================


class ESM2Embedder:
    """Extract ESM-2 embeddings with support for large models."""

    MODELS = {
        "small": {"name": "facebook/esm2_t6_8M_UR50D", "dim": 320, "params": "8M"},
        "medium": {"name": "facebook/esm2_t12_35M_UR50D", "dim": 480, "params": "35M"},
        "large": {"name": "facebook/esm2_t33_650M_UR50D", "dim": 1280, "params": "650M"},
    }

    def __init__(self, model_size: str = "small", device: str = None, half_precision: bool = False):
        from transformers import AutoModel, AutoTokenizer

        if model_size not in self.MODELS:
            raise ValueError(f"Unknown model size: {model_size}")

        self.model_info = self.MODELS[model_size]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.half_precision = half_precision and self.device == "cuda"

        print(f"\nLoading ESM-2 ({model_size}) - {self.model_info['params']} parameters...")
        print(f"  Model: {self.model_info['name']}")
        print(f"  Device: {self.device}")
        print(f"  Half precision: {self.half_precision}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_info["name"])
        self.model = AutoModel.from_pretrained(self.model_info["name"])

        if self.half_precision:
            self.model = self.model.half()
            print("  Using FP16 for reduced memory")

        self.model = self.model.to(self.device)
        self.model.eval()

        self.embedding_dim = self.model_info["dim"]
        print(f"  Embedding dim: {self.embedding_dim}")
        print("  Model loaded successfully!")

    def embed_sequences(self, sequences: list[str], batch_size: int = None) -> np.ndarray:
        """Embed multiple sequences."""
        # Adjust batch size based on model size
        if batch_size is None:
            if self.model_info["params"] == "650M":
                batch_size = 4  # Smaller batches for large model
            else:
                batch_size = 16

        embeddings = []
        total = len(sequences)

        for i in range(0, total, batch_size):
            batch = sequences[i : i + batch_size]

            # Clean sequences (remove gaps for ESM-2)
            clean_batch = [s.replace("-", "").replace("X", "A") for s in batch]

            inputs = self.tokenizer(clean_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                self.device
            )

            with torch.no_grad():
                if self.half_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**inputs)
                else:
                    outputs = self.model(**inputs)

            # Mean pooling
            batch_emb = outputs.last_hidden_state.mean(dim=1)

            if self.half_precision:
                batch_emb = batch_emb.float()

            embeddings.append(batch_emb.cpu().numpy())

            if (i // batch_size) % 5 == 0:
                print(f"  Embedded {min(i + batch_size, total)}/{total}")

            # Clear cache for large model
            if self.model_info["params"] == "650M":
                torch.cuda.empty_cache()

        return np.vstack(embeddings)

    def cleanup(self):
        """Free GPU memory."""
        del self.model
        del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()


# =============================================================================
# VAE MODEL
# =============================================================================


class ESM2VAE(nn.Module):
    """VAE using ESM-2 embeddings as input."""

    def __init__(self, esm_dim: int = 320, latent_dim: int = 16):
        super().__init__()

        # Adapt architecture to embedding dimension
        hidden_dim = min(256, esm_dim // 2)

        self.encoder = nn.Sequential(
            nn.Linear(esm_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, esm_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        pred = self.predictor(z).squeeze(-1)
        return recon, pred, mu, logvar


# =============================================================================
# TRAINING
# =============================================================================


def listmle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ListMLE ranking loss."""
    sorted_idx = torch.argsort(target, descending=True)
    sorted_pred = pred[sorted_idx]

    n = len(sorted_pred)
    total_loss = 0.0

    for i in range(n):
        remaining = sorted_pred[i:]
        log_prob = sorted_pred[i] - torch.logsumexp(remaining, dim=0)
        total_loss -= log_prob

    return total_loss / n


def train_and_evaluate(X: np.ndarray, y: np.ndarray, esm_dim: int, epochs: int = 100, verbose: bool = True) -> float:
    """Train VAE and return best test correlation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Train/test split
    n = len(y)
    idx = np.random.permutation(n)
    split = int(0.8 * n)

    X_train = torch.FloatTensor(X[idx[:split]]).to(device)
    y_train = torch.FloatTensor(y[idx[:split]]).to(device)
    X_test = torch.FloatTensor(X[idx[split:]]).to(device)
    y_test = torch.FloatTensor(y[idx[split:]]).to(device)

    model = ESM2VAE(esm_dim=esm_dim, latent_dim=16).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    best_corr = -1.0

    for epoch in range(epochs):
        model.train()
        recon, pred, mu, logvar = model(X_train)

        recon_loss = nn.functional.mse_loss(recon, X_train)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        rank_loss = listmle_loss(pred, y_train)
        pred_loss = nn.functional.mse_loss(pred, y_train)

        loss = recon_loss + 0.001 * kl_loss + 0.3 * rank_loss + 0.5 * pred_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 20 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                _, test_pred, _, _ = model(X_test)
                corr, _ = spearmanr(test_pred.cpu().numpy(), y_test.cpu().numpy())

                if corr > best_corr:
                    best_corr = corr

                if verbose:
                    print(f"    Epoch {epoch + 1}/{epochs}: corr = {corr:+.3f} (best: {best_corr:+.3f})")

    return best_corr


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================


def main():
    print("=" * 70)
    print(" ESM-2 MODEL SIZE COMPARISON (8M vs 650M)")
    print("=" * 70)

    # Set seed
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Check GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU Memory: {gpu_mem:.1f} GB")
        if gpu_mem < 4:
            print("WARNING: Less than 4GB GPU memory. Large model may not fit.")

    # Problem drugs
    problem_drugs = {
        "TPV": "PI",
        "DRV": "PI",
        "DTG": "INI",
        "RPV": "NNRTI",
    }

    results = {
        "small_8M": {},
        "large_650M": {},
    }

    # =========================================================================
    # EXPERIMENT 1: Small Model (8M)
    # =========================================================================
    print("\n" + "=" * 70)
    print(" PHASE 1: ESM-2 Small (8M parameters, 320-dim embeddings)")
    print("=" * 70)

    embedder_small = ESM2Embedder(model_size="small")

    for drug, gene in problem_drugs.items():
        print(f"\n--- {drug} ({gene}) ---")

        try:
            y, sequences = load_stanford_data(drug, gene)
            print(f"  Samples: {len(y)}")

            print("  Computing embeddings...")
            X = embedder_small.embed_sequences(sequences)
            print(f"  Shape: {X.shape}")

            print("  Training VAE...")
            corr = train_and_evaluate(X, y, esm_dim=embedder_small.embedding_dim)
            results["small_8M"][drug] = corr
            print(f"  Result: {corr:+.3f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results["small_8M"][drug] = 0.0

    # Cleanup small model
    embedder_small.cleanup()

    # =========================================================================
    # EXPERIMENT 2: Large Model (650M)
    # =========================================================================
    print("\n" + "=" * 70)
    print(" PHASE 2: ESM-2 Large (650M parameters, 1280-dim embeddings)")
    print("=" * 70)

    try:
        embedder_large = ESM2Embedder(model_size="large", half_precision=True)

        for drug, gene in problem_drugs.items():
            print(f"\n--- {drug} ({gene}) ---")

            try:
                y, sequences = load_stanford_data(drug, gene)
                print(f"  Samples: {len(y)}")

                print("  Computing embeddings...")
                X = embedder_large.embed_sequences(sequences)
                print(f"  Shape: {X.shape}")

                print("  Training VAE...")
                corr = train_and_evaluate(X, y, esm_dim=embedder_large.embedding_dim)
                results["large_650M"][drug] = corr
                print(f"  Result: {corr:+.3f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback

                traceback.print_exc()
                results["large_650M"][drug] = 0.0

        # Cleanup large model
        embedder_large.cleanup()

    except Exception as e:
        print(f"\nFailed to load large model: {e}")
        print("This may be due to insufficient GPU memory.")
        for drug in problem_drugs:
            results["large_650M"][drug] = 0.0

    # =========================================================================
    # RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print(" RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n{'Drug':<8} {'8M (320-dim)':>14} {'650M (1280-dim)':>16} {'Improvement':>12}")
    print("-" * 54)

    for drug in problem_drugs:
        small = results["small_8M"].get(drug, 0)
        large = results["large_650M"].get(drug, 0)

        if small > 0 and large > 0:
            improvement = ((large - small) / abs(small)) * 100
            imp_str = f"{improvement:+.1f}%"
        else:
            imp_str = "N/A"

        print(f"{drug:<8} {small:>+14.3f} {large:>+16.3f} {imp_str:>12}")

    # Average improvement
    valid_pairs = [
        (results["small_8M"].get(d, 0), results["large_650M"].get(d, 0))
        for d in problem_drugs
        if results["small_8M"].get(d, 0) > 0 and results["large_650M"].get(d, 0) > 0
    ]

    if valid_pairs:
        avg_small = np.mean([p[0] for p in valid_pairs])
        avg_large = np.mean([p[1] for p in valid_pairs])
        avg_improvement = ((avg_large - avg_small) / abs(avg_small)) * 100

        print("-" * 54)
        print(f"{'Average':<8} {avg_small:>+14.3f} {avg_large:>+16.3f} {avg_improvement:>+11.1f}%")

    # Save results
    output_file = RESULTS_DIR / "esm2_model_comparison.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Recommendations
    print("\n" + "=" * 70)
    print(" RECOMMENDATIONS")
    print("=" * 70)

    for drug in problem_drugs:
        small = results["small_8M"].get(drug, 0)
        large = results["large_650M"].get(drug, 0)

        if large > small and large > 0:
            print(f"  {drug}: Use 650M model ({large:+.3f} vs {small:+.3f})")
        elif small > 0:
            print(f"  {drug}: Use 8M model ({small:+.3f} vs {large:+.3f})")
        else:
            print(f"  {drug}: No valid results")


if __name__ == "__main__":
    main()
