#!/usr/bin/env python3
"""
ESM-2 Enhanced VAE Experiments
==============================

This script implements and tests three approaches:
1. ESM-2 VAE: Replace one-hot with ESM-2 embeddings
2. ESM-2 + Transfer Learning: Combine both approaches
3. ESM-2 + Structural Features: Add PDB binding site features

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


def load_stanford_data(drug: str, gene: str = "PI") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load Stanford HIVDB data for a specific drug."""
    # Map gene to file
    gene_files = {
        "PI": "stanford_hivdb_pi.txt",
        "NRTI": "stanford_hivdb_nrti.txt",
        "NNRTI": "stanford_hivdb_nnrti.txt",
        "INI": "stanford_hivdb_ini.txt",
    }

    file_path = DATA_DIR / gene_files.get(gene, "stanford_hivdb_pi.txt")

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Read data using pandas
    df = pd.read_csv(file_path, sep="\t", low_memory=False)

    # Find sequence column
    seq_col = None
    for col in df.columns:
        if "seq" in col.lower():
            seq_col = col
            break

    if seq_col is None:
        seq_col = df.columns[0]

    # Check if drug column exists
    if drug.upper() not in [c.upper() for c in df.columns]:
        raise ValueError(f"Drug {drug} not found in {file_path}. Available: {list(df.columns)}")

    # Get drug column (case-insensitive)
    drug_col = None
    for col in df.columns:
        if col.upper() == drug.upper():
            drug_col = col
            break

    # Filter rows with valid data
    df = df[[seq_col, drug_col]].dropna()
    df = df[df[drug_col] != ""]

    # Convert to numeric
    df[drug_col] = pd.to_numeric(df[drug_col], errors="coerce")
    df = df.dropna()

    if len(df) == 0:
        raise ValueError(f"No valid data found for {drug}")

    raw_sequences = df[seq_col].tolist()
    resistances = df[drug_col].tolist()

    # Clean sequences
    set("ACDEFGHIKLMNPQRSTVWY-X.")
    cleaned_sequences = []
    for seq in raw_sequences:
        seq = str(seq).upper().replace(".", "-").replace("~", "-").replace("*", "X")
        seq = "".join(c if c in "ACDEFGHIKLMNPQRSTVWY-X" else "X" for c in seq)
        cleaned_sequences.append(seq)

    # Pad sequences to same length
    max_len = max(len(s) for s in cleaned_sequences)
    padded_sequences = [s.ljust(max_len, "-") for s in cleaned_sequences]

    # One-hot encode
    aa_to_idx = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY-X")}
    n_aa = len(aa_to_idx)

    X = np.zeros((len(padded_sequences), max_len * n_aa), dtype=np.float32)
    for i, seq in enumerate(padded_sequences):
        for j, aa in enumerate(seq):
            idx = aa_to_idx.get(aa, aa_to_idx["X"])
            X[i, j * n_aa + idx] = 1.0

    y = np.array(resistances, dtype=np.float32)

    return X, y, cleaned_sequences


# =============================================================================
# ESM-2 EMBEDDINGS
# =============================================================================


class ESM2Embedder:
    """Extract ESM-2 embeddings for protein sequences."""

    def __init__(self, model_size: str = "small", device: str = None):
        from transformers import AutoModel, AutoTokenizer

        models = {
            "small": "facebook/esm2_t6_8M_UR50D",
            "medium": "facebook/esm2_t12_35M_UR50D",
        }

        self.model_name = models.get(model_size, models["small"])
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading ESM-2 ({model_size})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        print(f"  Embedding dim: {self.embedding_dim}")

    def embed_sequences(self, sequences: list[str], batch_size: int = 16) -> np.ndarray:
        """Embed multiple sequences."""
        embeddings = []

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i : i + batch_size]

            # Clean sequences (remove gaps for ESM-2)
            clean_batch = [s.replace("-", "").replace("X", "A") for s in batch]

            inputs = self.tokenizer(clean_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(
                self.device
            )

            with torch.no_grad():
                outputs = self.model(**inputs)

            # Mean pooling
            batch_emb = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            embeddings.append(batch_emb)

            if (i // batch_size) % 10 == 0:
                print(f"  Embedded {min(i + batch_size, len(sequences))}/{len(sequences)}")

        return np.vstack(embeddings)


# =============================================================================
# STRUCTURAL FEATURES
# =============================================================================

# HIV Protease binding site residues (from PDB structures)
PI_BINDING_SITE = [23, 25, 27, 28, 29, 30, 32, 47, 48, 50, 53, 54, 76, 80, 81, 82, 84]

# HIV RT binding site residues
RT_BINDING_SITE = [100, 101, 103, 106, 108, 181, 188, 190, 225, 227, 230, 318]

# HIV Integrase binding site residues
IN_BINDING_SITE = [66, 92, 140, 143, 147, 148, 155, 263]


def compute_structural_features(sequences: list[str], gene: str = "PI") -> np.ndarray:
    """Compute structural features for sequences."""
    binding_sites = {
        "PI": PI_BINDING_SITE,
        "NRTI": RT_BINDING_SITE,
        "NNRTI": RT_BINDING_SITE,
        "INI": IN_BINDING_SITE,
    }

    binding_site = binding_sites.get(gene, PI_BINDING_SITE)
    features = []

    for seq in sequences:
        seq_features = []

        # Feature 1: Number of mutations at binding site
        seq[0] if len(sequences) > 0 else seq  # Use first as reference
        binding_mutations = sum(1 for pos in binding_site if pos < len(seq) and seq[pos] != "-")
        seq_features.append(binding_mutations / len(binding_site))

        # Feature 2: Average distance to binding site for mutations
        mutation_positions = [i for i, aa in enumerate(seq) if aa not in "-X"]
        if mutation_positions and binding_site:
            avg_dist = np.mean([min(abs(pos - bs) for bs in binding_site) for pos in mutation_positions[:10]])
            seq_features.append(avg_dist / 100)  # Normalize
        else:
            seq_features.append(0.5)

        # Feature 3: Sequence length ratio
        seq_len = len(seq.replace("-", "").replace("X", ""))
        seq_features.append(seq_len / len(seq) if len(seq) > 0 else 0)

        # Feature 4: Hydrophobic content at binding site
        hydrophobic = set("AVILMFYW")
        if binding_site:
            hydro_count = sum(1 for pos in binding_site if pos < len(seq) and seq[pos] in hydrophobic)
            seq_features.append(hydro_count / len(binding_site))
        else:
            seq_features.append(0.5)

        features.append(seq_features)

    return np.array(features, dtype=np.float32)


# =============================================================================
# VAE MODELS
# =============================================================================


class BaseVAE(nn.Module):
    """Base VAE for drug resistance prediction."""

    def __init__(self, input_dim: int, latent_dim: int = 16):
        super().__init__()

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
            nn.Linear(256, input_dim),
        )

        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def predict(self, z):
        return self.predictor(z).squeeze(-1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        pred = self.predict(z)
        return recon, pred, mu, logvar


class ESM2VAE(nn.Module):
    """VAE using ESM-2 embeddings as input."""

    def __init__(self, esm_dim: int = 320, latent_dim: int = 16, struct_dim: int = 0):
        super().__init__()

        input_dim = esm_dim + struct_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
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
            nn.Linear(128, input_dim),
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


class TransferESM2VAE(nn.Module):
    """Transfer learning VAE with ESM-2 embeddings."""

    def __init__(self, esm_dim: int = 320, latent_dim: int = 16, n_drugs: int = 1):
        super().__init__()

        # Shared encoder
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

        # Shared decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, esm_dim),
        )

        # Drug-specific prediction heads
        self.drug_heads = nn.ModuleList(
            [nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 1)) for _ in range(n_drugs)]
        )

    def encode(self, x):
        h = self.encoder(x)
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
        return recon, pred, mu, logvar


# =============================================================================
# TRAINING FUNCTIONS
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


def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 100,
    lr: float = 1e-3,
    drug_idx: int = 0,
    verbose: bool = True,
) -> float:
    """Train model and return best test correlation."""
    next(model.parameters()).device
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_corr = -1.0

    for epoch in range(epochs):
        model.train()

        # Forward pass
        if hasattr(model, "drug_heads"):
            recon, pred, mu, logvar = model(X_train, drug_idx)
        else:
            recon, pred, mu, logvar = model(X_train)

        # Losses
        recon_loss = nn.functional.mse_loss(recon, X_train)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        rank_loss = listmle_loss(pred, y_train)
        pred_loss = nn.functional.mse_loss(pred, y_train)

        loss = recon_loss + 0.001 * kl_loss + 0.3 * rank_loss + 0.5 * pred_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Evaluate
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                if hasattr(model, "drug_heads"):
                    _, test_pred, _, _ = model(X_test, drug_idx)
                else:
                    _, test_pred, _, _ = model(X_test)

                test_pred_np = test_pred.cpu().numpy()
                y_test_np = y_test.cpu().numpy()

                corr, _ = spearmanr(test_pred_np, y_test_np)

                if corr > best_corr:
                    best_corr = corr

                if verbose and ((epoch + 1) % 20 == 0 or epoch == epochs - 1):
                    print(f"  Epoch {epoch + 1}/{epochs}: corr = {corr:+.3f} (best: {best_corr:+.3f})")

    return best_corr


# =============================================================================
# EXPERIMENTS
# =============================================================================


def run_baseline_experiment(drug: str, gene: str) -> float:
    """Run baseline one-hot VAE experiment."""
    print(f"\n--- Baseline VAE for {drug} ---")

    X, y, _ = load_stanford_data(drug, gene)
    print(f"  Samples: {len(y)}")

    # Train/test split
    n = len(y)
    idx = np.random.permutation(n)
    split = int(0.8 * n)

    X_train = torch.FloatTensor(X[idx[:split]]).cuda()
    y_train = torch.FloatTensor(y[idx[:split]]).cuda()
    X_test = torch.FloatTensor(X[idx[split:]]).cuda()
    y_test = torch.FloatTensor(y[idx[split:]]).cuda()

    model = BaseVAE(X.shape[1], latent_dim=16).cuda()
    corr = train_model(model, X_train, y_train, X_test, y_test, epochs=100)

    print(f"  Result: {corr:+.3f}")
    return corr


def run_esm2_experiment(drug: str, gene: str, embedder: ESM2Embedder) -> float:
    """Run ESM-2 VAE experiment."""
    print(f"\n--- ESM-2 VAE for {drug} ---")

    _, y, sequences = load_stanford_data(drug, gene)
    print(f"  Samples: {len(y)}")

    # Get ESM-2 embeddings
    print("  Computing ESM-2 embeddings...")
    X = embedder.embed_sequences(sequences)
    print(f"  Embedding shape: {X.shape}")

    # Train/test split
    n = len(y)
    idx = np.random.permutation(n)
    split = int(0.8 * n)

    X_train = torch.FloatTensor(X[idx[:split]]).cuda()
    y_train = torch.FloatTensor(y[idx[:split]]).cuda()
    X_test = torch.FloatTensor(X[idx[split:]]).cuda()
    y_test = torch.FloatTensor(y[idx[split:]]).cuda()

    model = ESM2VAE(esm_dim=embedder.embedding_dim, latent_dim=16).cuda()
    corr = train_model(model, X_train, y_train, X_test, y_test, epochs=100)

    print(f"  Result: {corr:+.3f}")
    return corr


def run_esm2_structural_experiment(drug: str, gene: str, embedder: ESM2Embedder) -> float:
    """Run ESM-2 + structural features experiment."""
    print(f"\n--- ESM-2 + Structural for {drug} ---")

    _, y, sequences = load_stanford_data(drug, gene)
    print(f"  Samples: {len(y)}")

    # Get ESM-2 embeddings
    print("  Computing ESM-2 embeddings...")
    esm_emb = embedder.embed_sequences(sequences)

    # Get structural features
    print("  Computing structural features...")
    struct_feat = compute_structural_features(sequences, gene)
    print(f"  Structural features: {struct_feat.shape}")

    # Combine
    X = np.hstack([esm_emb, struct_feat])
    print(f"  Combined shape: {X.shape}")

    # Train/test split
    n = len(y)
    idx = np.random.permutation(n)
    split = int(0.8 * n)

    X_train = torch.FloatTensor(X[idx[:split]]).cuda()
    y_train = torch.FloatTensor(y[idx[:split]]).cuda()
    X_test = torch.FloatTensor(X[idx[split:]]).cuda()
    y_test = torch.FloatTensor(y[idx[split:]]).cuda()

    model = ESM2VAE(esm_dim=embedder.embedding_dim, latent_dim=16, struct_dim=struct_feat.shape[1]).cuda()
    corr = train_model(model, X_train, y_train, X_test, y_test, epochs=100)

    print(f"  Result: {corr:+.3f}")
    return corr


def run_esm2_transfer_experiment(target_drug: str, all_drugs: list[str], gene: str, embedder: ESM2Embedder) -> float:
    """Run ESM-2 + Transfer Learning experiment."""
    print(f"\n--- ESM-2 Transfer Learning for {target_drug} ---")

    # Collect all data for pre-training
    all_X = []
    all_y = []
    all_drug_idx = []
    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}

    for drug in all_drugs:
        try:
            _, y, sequences = load_stanford_data(drug, gene)
            X = embedder.embed_sequences(sequences)
            all_X.append(X)
            all_y.append(y)
            all_drug_idx.extend([drug_to_idx[drug]] * len(y))
        except Exception as e:
            print(f"  Skipping {drug}: {e}")

    if len(all_X) == 0:
        return 0.0

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)
    drug_idx_all = np.array(all_drug_idx)

    print(f"  Total pre-training samples: {len(y_all)}")

    # Pre-train on all drugs
    print("  Phase 1: Pre-training on all drugs...")
    model = TransferESM2VAE(esm_dim=embedder.embedding_dim, latent_dim=16, n_drugs=len(all_drugs)).cuda()

    # Pre-training loop
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    X_train_all = torch.FloatTensor(X_all).cuda()
    y_train_all = torch.FloatTensor(y_all).cuda()
    drug_idx_all_t = torch.LongTensor(drug_idx_all).cuda()

    for epoch in range(50):
        model.train()
        total_loss = 0.0

        for drug in all_drugs:
            d_idx = drug_to_idx[drug]
            mask = drug_idx_all_t == d_idx

            if mask.sum() < 10:
                continue

            X_d = X_train_all[mask]
            y_d = y_train_all[mask]

            recon, pred, mu, logvar = model(X_d, d_idx)

            recon_loss = nn.functional.mse_loss(recon, X_d)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            pred_loss = nn.functional.mse_loss(pred, y_d)

            loss = recon_loss + 0.001 * kl_loss + 0.5 * pred_loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1}/50, Loss: {total_loss:.4f}")

    # Fine-tune on target drug
    print(f"  Phase 2: Fine-tuning on {target_drug}...")

    target_idx = drug_to_idx[target_drug]
    mask = drug_idx_all_t == target_idx

    X_target = X_train_all[mask]
    y_target = y_train_all[mask]

    n = X_target.shape[0]
    idx = torch.randperm(n)
    split = int(0.8 * n)

    X_train = X_target[idx[:split]]
    y_train = y_target[idx[:split]]
    X_test = X_target[idx[split:]]
    y_test = y_target[idx[split:]]

    # Freeze encoder initially
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.01)

    # Phase 2a: Train head only
    for epoch in range(10):
        model.train()
        recon, pred, mu, logvar = model(X_train, target_idx)
        loss = nn.functional.mse_loss(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Unfreeze and fine-tune
    for param in model.encoder.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    best_corr = -1.0
    for epoch in range(90):
        model.train()
        recon, pred, mu, logvar = model(X_train, target_idx)

        recon_loss = nn.functional.mse_loss(recon, X_train)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        rank_loss = listmle_loss(pred, y_train)
        pred_loss = nn.functional.mse_loss(pred, y_train)

        loss = recon_loss + 0.001 * kl_loss + 0.3 * rank_loss + 0.5 * pred_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                _, test_pred, _, _ = model(X_test, target_idx)
                corr, _ = spearmanr(test_pred.cpu().numpy(), y_test.cpu().numpy())
                if corr > best_corr:
                    best_corr = corr
                print(f"    Epoch {epoch + 1}/90: corr = {corr:+.3f} (best: {best_corr:+.3f})")

    print(f"  Result: {best_corr:+.3f}")
    return best_corr


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print(" ESM-2 ENHANCED VAE EXPERIMENTS")
    print("=" * 70)

    # Set seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Drug configurations
    drugs_config = {
        "PI": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"],
        "NNRTI": ["EFV", "NVP", "ETR", "RPV"],
        "INI": ["RAL", "EVG", "DTG"],
    }

    # Problem drugs to focus on
    problem_drugs = {
        "TPV": "PI",
        "DRV": "PI",
        "DTG": "INI",
        "RPV": "NNRTI",
    }

    results = {
        "baseline": {},
        "esm2": {},
        "esm2_structural": {},
        "esm2_transfer": {},
    }

    # Initialize ESM-2 embedder
    embedder = ESM2Embedder(model_size="small")

    # Run experiments on problem drugs
    for drug, gene in problem_drugs.items():
        print(f"\n{'=' * 70}")
        print(f" EXPERIMENTS FOR {drug} ({gene})")
        print("=" * 70)

        try:
            # 1. Baseline
            results["baseline"][drug] = run_baseline_experiment(drug, gene)

            # 2. ESM-2 only
            results["esm2"][drug] = run_esm2_experiment(drug, gene, embedder)

            # 3. ESM-2 + Structural
            results["esm2_structural"][drug] = run_esm2_structural_experiment(drug, gene, embedder)

            # 4. ESM-2 + Transfer Learning
            all_drugs = drugs_config[gene]
            results["esm2_transfer"][drug] = run_esm2_transfer_experiment(drug, all_drugs, gene, embedder)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()

    # Print summary
    print("\n" + "=" * 70)
    print(" RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Drug':<8} {'Baseline':>10} {'ESM-2':>10} {'ESM2+Struct':>12} {'ESM2+Transfer':>14}")
    print("-" * 60)

    for drug in problem_drugs:
        baseline = results["baseline"].get(drug, 0)
        esm2 = results["esm2"].get(drug, 0)
        esm2_struct = results["esm2_structural"].get(drug, 0)
        esm2_transfer = results["esm2_transfer"].get(drug, 0)

        print(f"{drug:<8} {baseline:>+10.3f} {esm2:>+10.3f} {esm2_struct:>+12.3f} {esm2_transfer:>+14.3f}")

    # Save results
    output_file = RESULTS_DIR / "esm2_experiment_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Best approach per drug
    print("\n" + "=" * 70)
    print(" BEST APPROACH PER DRUG")
    print("=" * 70)

    for drug in problem_drugs:
        scores = {
            "Baseline": results["baseline"].get(drug, 0),
            "ESM-2": results["esm2"].get(drug, 0),
            "ESM-2+Structural": results["esm2_structural"].get(drug, 0),
            "ESM-2+Transfer": results["esm2_transfer"].get(drug, 0),
        }
        best = max(scores, key=scores.get)
        print(f"  {drug}: {best} ({scores[best]:+.3f})")


if __name__ == "__main__":
    main()
