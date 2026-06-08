#!/usr/bin/env python3
"""
Hybrid ESM-2 + Transfer Learning Experiments
=============================================

This script combines the best of both approaches:
1. ESM-2 embeddings (rich protein representations)
2. Transfer learning (cross-drug knowledge)

The goal is to leverage:
- ESM-2's understanding of protein evolutionary constraints
- Transfer learning's ability to share knowledge across drugs

Author: Claude Code
Date: December 28, 2024
"""

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


def load_all_drug_data(gene: str) -> dict[str, tuple[np.ndarray, list[str]]]:
    """Load data for all drugs in a gene class."""
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

    # Get all drug columns (numeric columns that aren't the sequence)
    drug_data = {}

    for col in df.columns:
        if col == seq_col:
            continue

        # Check if column has numeric resistance values
        temp = df[[seq_col, col]].dropna()
        temp = temp[temp[col] != ""]
        temp[col] = pd.to_numeric(temp[col], errors="coerce")
        temp = temp.dropna()

        if len(temp) < 50:  # Skip drugs with too few samples
            continue

        # Clean sequences
        sequences = []
        for seq in temp[seq_col].tolist():
            seq = str(seq).upper().replace(".", "-").replace("~", "-").replace("*", "X")
            seq = "".join(c if c in "ACDEFGHIKLMNPQRSTVWY-X" else "X" for c in seq)
            sequences.append(seq)

        drug_data[col] = (temp[col].values.astype(np.float32), sequences)

    return drug_data


# =============================================================================
# ESM-2 EMBEDDER
# =============================================================================


class ESM2Embedder:
    """Extract ESM-2 embeddings."""

    def __init__(self, model_size: str = "small", device: str = None):
        from transformers import AutoModel, AutoTokenizer

        models = {
            "small": ("facebook/esm2_t6_8M_UR50D", 320),
        }

        self.model_name, self.embedding_dim = models.get(model_size, models["small"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"\nLoading ESM-2 ({model_size})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        print(f"  Embedding dim: {self.embedding_dim}")

    def embed_sequences(self, sequences: list[str], batch_size: int = 16) -> np.ndarray:
        """Embed multiple sequences."""
        embeddings = []
        total = len(sequences)

        for i in range(0, total, batch_size):
            batch = sequences[i : i + batch_size]
            clean_batch = [s.replace("-", "").replace("X", "A") for s in batch]

            inputs = self.tokenizer(clean_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                self.device
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_emb)

        return np.vstack(embeddings)


# =============================================================================
# HYBRID MODEL
# =============================================================================


class HybridTransferVAE(nn.Module):
    """
    Hybrid VAE combining ESM-2 embeddings with transfer learning.

    Architecture:
    - Shared encoder learns from ESM-2 embeddings across all drugs
    - Drug-specific prediction heads specialize for each drug
    - Pre-training on all drugs, then fine-tuning on target
    """

    def __init__(self, esm_dim: int = 320, latent_dim: int = 16, n_drugs: int = 1):
        super().__init__()

        # Shared encoder (learns cross-drug representations)
        self.shared_encoder = nn.Sequential(
            nn.Linear(esm_dim, 256),
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

        # Shared decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, esm_dim),
        )

        # Drug-specific prediction heads
        self.drug_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(latent_dim, 32),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                )
                for _ in range(n_drugs)
            ]
        )

        self.n_drugs = n_drugs
        self.latent_dim = latent_dim

    def encode(self, x):
        h = self.shared_encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, drug_idx: int = 0):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        pred = self.drug_heads[drug_idx](z).squeeze(-1)
        return recon, pred, mu, logvar, z


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================


def listmle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ListMLE ranking loss."""
    if len(pred) < 2:
        return torch.tensor(0.0, device=pred.device)

    sorted_idx = torch.argsort(target, descending=True)
    sorted_pred = pred[sorted_idx]

    n = len(sorted_pred)
    total_loss = 0.0

    for i in range(n):
        remaining = sorted_pred[i:]
        log_prob = sorted_pred[i] - torch.logsumexp(remaining, dim=0)
        total_loss -= log_prob

    return total_loss / n


def contrastive_loss(z: torch.Tensor, y: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Contrastive loss to push apart samples with different resistance levels.
    Encourages latent space to respect resistance ordering.
    """
    n = len(z)
    if n < 2:
        return torch.tensor(0.0, device=z.device)

    # Compute pairwise distances
    dists = torch.cdist(z, z)

    # Compute resistance differences
    y_diff = torch.abs(y.unsqueeze(0) - y.unsqueeze(1))

    # Normalize resistance differences
    y_norm = y_diff / (y_diff.max() + 1e-6)

    # Loss: high resistance difference should have high distance
    loss = torch.mean((1 - y_norm) * dists.pow(2) + y_norm * torch.clamp(margin - dists, min=0).pow(2))

    return loss


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================


def pretrain_model(
    model: HybridTransferVAE,
    all_data: dict[str, tuple[np.ndarray, torch.Tensor]],
    drug_to_idx: dict[str, int],
    epochs: int = 50,
    lr: float = 1e-3,
) -> None:
    """Pre-train model on all drugs in the class."""
    device = next(model.parameters()).device
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    print("  Pre-training on all drugs...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for drug, (X, y) in all_data.items():
            drug_idx = drug_to_idx[drug]

            X_t = torch.FloatTensor(X).to(device)
            y_t = torch.FloatTensor(y).to(device)

            recon, pred, mu, logvar, z = model(X_t, drug_idx)

            # Multi-task loss
            recon_loss = nn.functional.mse_loss(recon, X_t)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            pred_loss = nn.functional.mse_loss(pred, y_t)
            rank_loss = listmle_loss(pred, y_t)

            loss = recon_loss + 0.001 * kl_loss + 0.5 * pred_loss + 0.3 * rank_loss
            total_loss += loss.item()
            n_batches += 1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / n_batches
            print(f"    Epoch {epoch + 1}/{epochs}: avg_loss = {avg_loss:.4f}")


def finetune_model(
    model: HybridTransferVAE,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    drug_idx: int,
    epochs: int = 100,
    freeze_epochs: int = 20,
) -> float:
    """Fine-tune model on target drug with gradual unfreezing."""
    next(model.parameters()).device

    # Phase 1: Train only the drug head (frozen encoder)
    print(f"  Fine-tuning Phase 1: Training head only ({freeze_epochs} epochs)...")

    for param in model.shared_encoder.parameters():
        param.requires_grad = False
    for param in model.fc_mu.parameters():
        param.requires_grad = False
    for param in model.fc_logvar.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.01)

    for epoch in range(freeze_epochs):
        model.train()
        recon, pred, mu, logvar, z = model(X_train, drug_idx)
        loss = nn.functional.mse_loss(pred, y_train) + 0.3 * listmle_loss(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Phase 2: Unfreeze and fine-tune all
    print(f"  Fine-tuning Phase 2: Full model ({epochs - freeze_epochs} epochs)...")

    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - freeze_epochs)

    best_corr = -1.0

    for epoch in range(epochs - freeze_epochs):
        model.train()
        recon, pred, mu, logvar, z = model(X_train, drug_idx)

        recon_loss = nn.functional.mse_loss(recon, X_train)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        pred_loss = nn.functional.mse_loss(pred, y_train)
        rank_loss = listmle_loss(pred, y_train)
        contrast_loss = contrastive_loss(z, y_train)

        loss = recon_loss + 0.001 * kl_loss + 0.5 * pred_loss + 0.3 * rank_loss + 0.1 * contrast_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Evaluate
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                _, test_pred, _, _, _ = model(X_test, drug_idx)
                corr, _ = spearmanr(test_pred.cpu().numpy(), y_test.cpu().numpy())

                if corr > best_corr:
                    best_corr = corr

                print(f"    Epoch {freeze_epochs + epoch + 1}/{epochs}: corr = {corr:+.3f} (best: {best_corr:+.3f})")

    return best_corr


# =============================================================================
# BASELINE METHODS
# =============================================================================


class SimpleESM2VAE(nn.Module):
    """Simple ESM-2 VAE without transfer learning (baseline)."""

    def __init__(self, esm_dim: int = 320, latent_dim: int = 16):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(esm_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, esm_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        pred = self.predictor(z).squeeze(-1)
        return recon, pred, mu, logvar


def train_baseline_esm2(X: np.ndarray, y: np.ndarray, esm_dim: int, epochs: int = 100) -> float:
    """Train baseline ESM-2 VAE without transfer learning."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n = len(y)
    idx = np.random.permutation(n)
    split = int(0.8 * n)

    X_train = torch.FloatTensor(X[idx[:split]]).to(device)
    y_train = torch.FloatTensor(y[idx[:split]]).to(device)
    X_test = torch.FloatTensor(X[idx[split:]]).to(device)
    y_test = torch.FloatTensor(y[idx[split:]]).to(device)

    model = SimpleESM2VAE(esm_dim=esm_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    best_corr = -1.0

    for epoch in range(epochs):
        model.train()
        recon, pred, mu, logvar = model(X_train)

        recon_loss = nn.functional.mse_loss(recon, X_train)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        pred_loss = nn.functional.mse_loss(pred, y_train)
        rank_loss = listmle_loss(pred, y_train)

        loss = recon_loss + 0.001 * kl_loss + 0.5 * pred_loss + 0.3 * rank_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                _, test_pred, _, _ = model(X_test)
                corr, _ = spearmanr(test_pred.cpu().numpy(), y_test.cpu().numpy())
                if corr > best_corr:
                    best_corr = corr

    return best_corr


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================


def run_hybrid_experiment(target_drug: str, gene: str, embedder: ESM2Embedder) -> dict[str, float]:
    """Run hybrid ESM-2 + Transfer Learning experiment."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'=' * 60}")
    print(f" HYBRID EXPERIMENT: {target_drug} ({gene})")
    print("=" * 60)

    # Load all drug data for this gene class
    print("\n1. Loading all drug data...")
    all_drug_data = load_all_drug_data(gene)
    print(f"   Found {len(all_drug_data)} drugs in {gene} class")

    for drug, (y, _) in all_drug_data.items():
        print(f"     {drug}: {len(y)} samples")

    if target_drug not in all_drug_data:
        print(f"   ERROR: {target_drug} not found!")
        return {}

    # Compute ESM-2 embeddings for all sequences
    print("\n2. Computing ESM-2 embeddings...")

    all_embeddings = {}
    for drug, (y, sequences) in all_drug_data.items():
        print(f"   Embedding {drug}...")
        X = embedder.embed_sequences(sequences)
        all_embeddings[drug] = (X, y)
        print(f"     Shape: {X.shape}")

    # Setup
    drugs = list(all_embeddings.keys())
    drug_to_idx = {d: i for i, d in enumerate(drugs)}
    n_drugs = len(drugs)

    # Get target data
    X_target, y_target = all_embeddings[target_drug]

    # Train/test split for target
    n = len(y_target)
    idx = np.random.permutation(n)
    split = int(0.8 * n)

    X_train = torch.FloatTensor(X_target[idx[:split]]).to(device)
    y_train = torch.FloatTensor(y_target[idx[:split]]).to(device)
    X_test = torch.FloatTensor(X_target[idx[split:]]).to(device)
    y_test = torch.FloatTensor(y_target[idx[split:]]).to(device)

    results = {}

    # Experiment 1: Baseline ESM-2 (no transfer)
    print("\n3. Baseline ESM-2 VAE (no transfer learning)...")
    baseline_corr = train_baseline_esm2(X_target, y_target, embedder.embedding_dim)
    results["esm2_baseline"] = baseline_corr
    print(f"   Result: {baseline_corr:+.3f}")

    # Experiment 2: Hybrid (ESM-2 + Transfer)
    print("\n4. Hybrid ESM-2 + Transfer Learning...")

    model = HybridTransferVAE(esm_dim=embedder.embedding_dim, latent_dim=16, n_drugs=n_drugs).to(device)

    # Pre-train on all drugs
    pretrain_model(model, all_embeddings, drug_to_idx, epochs=50)

    # Fine-tune on target drug
    target_idx = drug_to_idx[target_drug]
    hybrid_corr = finetune_model(model, X_train, y_train, X_test, y_test, target_idx, epochs=100)
    results["hybrid_transfer"] = hybrid_corr
    print(f"   Result: {hybrid_corr:+.3f}")

    # Calculate improvement
    improvement = ((hybrid_corr - baseline_corr) / abs(baseline_corr)) * 100 if baseline_corr != 0 else 0
    results["improvement"] = improvement

    print(f"\n   Improvement: {improvement:+.1f}%")

    return results


def main():
    print("=" * 70)
    print(" HYBRID ESM-2 + TRANSFER LEARNING EXPERIMENTS")
    print("=" * 70)
    print("\nThis approach combines:")
    print("  - ESM-2 embeddings (evolutionary protein representations)")
    print("  - Transfer learning (cross-drug knowledge sharing)")

    # Set seed
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Problem drugs
    problem_drugs = {
        "TPV": "PI",
        "DRV": "PI",
        "DTG": "INI",
        "RPV": "NNRTI",
    }

    # Initialize ESM-2 embedder
    embedder = ESM2Embedder(model_size="small")

    # Run experiments
    all_results = {}

    for drug, gene in problem_drugs.items():
        results = run_hybrid_experiment(drug, gene, embedder)
        all_results[drug] = results

    # Summary
    print("\n" + "=" * 70)
    print(" FINAL RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Drug':<8} {'ESM-2 Only':>12} {'Hybrid':>12} {'Improvement':>14}")
    print("-" * 50)

    for drug in problem_drugs:
        if drug in all_results:
            baseline = all_results[drug].get("esm2_baseline", 0)
            hybrid = all_results[drug].get("hybrid_transfer", 0)
            improvement = all_results[drug].get("improvement", 0)
            print(f"{drug:<8} {baseline:>+12.3f} {hybrid:>+12.3f} {improvement:>+13.1f}%")

    # Save results
    output_file = RESULTS_DIR / "hybrid_esm2_transfer_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Recommendations
    print("\n" + "=" * 70)
    print(" RECOMMENDATIONS")
    print("=" * 70)

    for drug in problem_drugs:
        if drug in all_results:
            baseline = all_results[drug].get("esm2_baseline", 0)
            hybrid = all_results[drug].get("hybrid_transfer", 0)

            if hybrid > baseline:
                print(f"  {drug}: Use Hybrid approach ({hybrid:+.3f} vs {baseline:+.3f})")
            else:
                print(f"  {drug}: Use ESM-2 only ({baseline:+.3f} vs {hybrid:+.3f})")


if __name__ == "__main__":
    main()
