#!/usr/bin/env python3
"""Comprehensive Analysis of All Available Datasets.

This script analyzes ALL datasets in the project:
1. Sequence data (FASTA/FNA files)
2. HuggingFace datasets (parquet)
3. Epidemiological data (CSV)
4. Protein interaction data
5. Mapping results

Outputs a detailed report and trains models on combined data.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.config.paths import OUTPUT_DIR

# Try to import optional dependencies
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, some analyses will be skipped")

try:
    import pyarrow.parquet as pq

    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    print("Warning: pyarrow not available, parquet files will be skipped")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# =============================================================================
# GENETIC CODE (imported from centralized biology module)
# =============================================================================
from src.biology.codons import CODON_TO_INDEX, GENETIC_CODE
from src.geometry import poincare_distance

# RNA to DNA conversion
RNA_TO_DNA = {"U": "T", "A": "A", "C": "C", "G": "G"}

CODONS = list(GENETIC_CODE.keys())

# Amino acid encoding
AA_TO_IDX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY*")}


# =============================================================================
# DATA LOADING UTILITIES
# =============================================================================


def parse_fasta(filepath: Path) -> list[tuple[str, str]]:
    """Parse FASTA file."""
    sequences = []
    current_name = None
    current_seq = []

    with open(filepath, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name:
                    sequences.append((current_name, "".join(current_seq)))
                current_name = line[1:]
                current_seq = []
            elif line:
                current_seq.append(line.upper())

        if current_name:
            sequences.append((current_name, "".join(current_seq)))

    return sequences


def rna_to_dna(seq: str) -> str:
    """Convert RNA sequence to DNA."""
    return "".join(RNA_TO_DNA.get(c, c) for c in seq.upper())


def sequence_to_codons(seq: str, is_rna: bool = False) -> list[int]:
    """Convert DNA/RNA sequence to codon indices."""
    if is_rna:
        seq = rna_to_dna(seq)

    codons = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        if codon in CODON_TO_INDEX:
            codons.append(CODON_TO_INDEX[codon])
    return codons


def sequence_to_amino_acids(seq: str) -> list[int]:
    """Convert amino acid sequence to indices."""
    return [AA_TO_IDX.get(aa, 20) for aa in seq.upper() if aa in AA_TO_IDX]


def compute_padic_distance(codon1: str, codon2: str, p: int = 3) -> float:
    """Compute p-adic distance between codons."""
    if codon1 == codon2:
        return 0.0
    for i, (b1, b2) in enumerate(zip(codon1, codon2, strict=False)):
        if b1 != b2:
            return p ** (-(i + 1))
    return 0.0


def build_padic_distance_matrix() -> np.ndarray:
    """Build 64x64 p-adic distance matrix."""
    n_codons = len(CODONS)
    dist_matrix = np.zeros((n_codons, n_codons))
    for i, c1 in enumerate(CODONS):
        for j, c2 in enumerate(CODONS):
            dist_matrix[i, j] = compute_padic_distance(c1, c2)
    return dist_matrix


# =============================================================================
# DATASET CLASSES
# =============================================================================


class CodonDataset(Dataset):
    """Dataset for codon sequences."""

    def __init__(self, sequences: list[list[int]], max_len: int = 100, source: str = "unknown"):
        self.sequences = []
        self.max_len = max_len
        self.source = source

        for seq in sequences:
            if len(seq) >= 10:
                self.sequences.append(seq[:max_len])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        codons = self.sequences[idx]
        if len(codons) < self.max_len:
            codons = codons + [0] * (self.max_len - len(codons))
        return torch.tensor(codons[: self.max_len], dtype=torch.long)


class AminoAcidDataset(Dataset):
    """Dataset for amino acid sequences."""

    def __init__(self, sequences: list[list[int]], max_len: int = 100, source: str = "unknown"):
        self.sequences = []
        self.max_len = max_len
        self.source = source

        for seq in sequences:
            if len(seq) >= 5:
                self.sequences.append(seq[:max_len])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        aas = self.sequences[idx]
        if len(aas) < self.max_len:
            aas = aas + [20] * (self.max_len - len(aas))  # Pad with stop codon index
        return torch.tensor(aas[: self.max_len], dtype=torch.long)


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================


class PAdicEmbedding(nn.Module):
    """Embedding initialized from p-adic distances."""

    def __init__(self, n_tokens: int, embed_dim: int, use_padic: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(n_tokens, embed_dim)

        if use_padic and n_tokens == 64:
            # Initialize from p-adic MDS for codons
            dist_matrix = build_padic_distance_matrix()
            embeddings = self._mds_from_distances(dist_matrix, embed_dim)
            with torch.no_grad():
                self.embedding.weight.copy_(torch.tensor(embeddings, dtype=torch.float32))

    def _mds_from_distances(self, D: np.ndarray, dim: int) -> np.ndarray:
        """MDS embedding from distance matrix."""
        n = D.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (D**2) @ H
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        pos_idx = eigenvalues > 1e-10
        eigenvalues = eigenvalues[pos_idx][:dim]
        eigenvectors = eigenvectors[:, pos_idx][:, :dim]
        if len(eigenvalues) < dim:
            pad = dim - len(eigenvalues)
            eigenvalues = np.concatenate([eigenvalues, np.zeros(pad)])
            eigenvectors = np.concatenate([eigenvectors, np.zeros((n, pad))], axis=1)
        return eigenvectors * np.sqrt(np.abs(eigenvalues))

    def forward(self, x):
        return self.embedding(x)


class UnifiedVAE(nn.Module):
    """Unified VAE for both codon and amino acid sequences."""

    def __init__(
        self,
        n_tokens: int = 64,
        embed_dim: int = 16,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        max_len: int = 100,
        use_padic: bool = True,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.latent_dim = latent_dim
        self.max_len = max_len

        self.embed = PAdicEmbedding(n_tokens, embed_dim, use_padic and n_tokens == 64)

        self.encoder = nn.Sequential(
            nn.Linear(max_len * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_len * n_tokens),
        )

    def encode(self, x):
        emb = self.embed(x).view(x.size(0), -1)
        h = self.encoder(emb)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder(z)
        return h.view(h.size(0), self.max_len, self.n_tokens)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar, z


def vae_loss(logits, targets, mu, logvar, beta=0.1):
    """VAE loss function."""
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = targets.view(-1)
    recon_loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================


def analyze_sequence_data(data_dir: Path) -> dict:
    """Analyze all sequence files."""
    results = {"fasta": [], "fna": [], "total_sequences": 0, "total_codons": 0}

    # FASTA files
    fasta_files = list(data_dir.rglob("*.fasta"))
    for fasta_path in fasta_files:
        try:
            seqs = parse_fasta(fasta_path)
            codon_counts = [len(sequence_to_codons(s)) for _, s in seqs]
            results["fasta"].append(
                {
                    "file": str(fasta_path.relative_to(data_dir)),
                    "n_sequences": len(seqs),
                    "mean_codons": np.mean(codon_counts) if codon_counts else 0,
                    "total_codons": sum(codon_counts),
                }
            )
            results["total_sequences"] += len(seqs)
            results["total_codons"] += sum(codon_counts)
        except Exception as e:
            print(f"  Error reading {fasta_path.name}: {e}")

    # FNA (RNA) files
    fna_files = list(data_dir.rglob("*.fna"))
    for fna_path in fna_files:
        try:
            seqs = parse_fasta(fna_path)
            codon_counts = [len(sequence_to_codons(s, is_rna=True)) for _, s in seqs]
            results["fna"].append(
                {
                    "file": str(fna_path.relative_to(data_dir)),
                    "n_sequences": len(seqs),
                    "mean_codons": np.mean(codon_counts) if codon_counts else 0,
                    "total_codons": sum(codon_counts),
                }
            )
            results["total_sequences"] += len(seqs)
            results["total_codons"] += sum(codon_counts)
        except Exception as e:
            print(f"  Error reading {fna_path.name}: {e}")

    return results


def analyze_csv_data(data_dir: Path) -> dict:
    """Analyze all CSV files."""
    if not HAS_PANDAS:
        return {"error": "pandas not available"}

    results = {"files": [], "total_rows": 0}

    csv_files = list(data_dir.rglob("*.csv"))
    for csv_path in csv_files:
        try:
            pd.read_csv(csv_path, nrows=5)  # Just read header + few rows
            full_df = pd.read_csv(csv_path)
            results["files"].append(
                {
                    "file": str(csv_path.relative_to(data_dir)),
                    "n_rows": len(full_df),
                    "n_columns": len(full_df.columns),
                    "columns": list(full_df.columns)[:10],  # First 10 columns
                }
            )
            results["total_rows"] += len(full_df)
        except Exception as e:
            print(f"  Error reading {csv_path.name}: {e}")

    return results


def analyze_parquet_data(data_dir: Path) -> dict:
    """Analyze parquet files."""
    if not HAS_PARQUET:
        return {"error": "pyarrow not available"}

    results = {"files": []}

    parquet_files = list(data_dir.rglob("*.parquet"))
    for pq_path in parquet_files:
        try:
            table = pq.read_table(pq_path)
            df = table.to_pandas()
            results["files"].append(
                {
                    "file": str(pq_path.relative_to(data_dir)),
                    "n_rows": len(df),
                    "n_columns": len(df.columns),
                    "columns": list(df.columns),
                }
            )
        except Exception as e:
            print(f"  Error reading {pq_path.name}: {e}")

    return results


def load_all_sequences(data_dir: Path, max_len: int = 100) -> tuple[list, list]:
    """Load all sequences from all sources."""
    codon_sequences = []
    aa_sequences = []

    # Load FASTA (DNA)
    for fasta_path in data_dir.rglob("*.fasta"):
        try:
            seqs = parse_fasta(fasta_path)
            for _name, seq in seqs:
                codons = sequence_to_codons(seq)
                if len(codons) >= 10:
                    codon_sequences.append(codons[:max_len])
        except Exception:
            pass

    # Load FNA (RNA)
    for fna_path in data_dir.rglob("*.fna"):
        try:
            seqs = parse_fasta(fna_path)
            for _name, seq in seqs:
                codons = sequence_to_codons(seq, is_rna=True)
                if len(codons) >= 10:
                    codon_sequences.append(codons[:max_len])
        except Exception:
            pass

    # Load parquet protein sequences
    if HAS_PARQUET and HAS_PANDAS:
        for pq_path in data_dir.rglob("*.parquet"):
            try:
                df = pq.read_table(pq_path).to_pandas()
                # Look for sequence columns
                seq_cols = [c for c in df.columns if "seq" in c.lower() or "sequence" in c.lower()]
                for col in seq_cols:
                    for seq in df[col].dropna():
                        if isinstance(seq, str) and len(seq) >= 5:
                            # Check if it's amino acid sequence (no T/U)
                            if all(c in "ACDEFGHIKLMNPQRSTVWY*" for c in seq.upper()):
                                aa_seq = sequence_to_amino_acids(seq)
                                if len(aa_seq) >= 5:
                                    aa_sequences.append(aa_seq[:max_len])
            except Exception:
                pass

    return codon_sequences, aa_sequences


def train_model(model, train_loader, val_loader, device, n_epochs=30, model_name="model"):
    """Train a VAE model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, n_epochs + 1):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits, mu, logvar, z = model(batch)
            loss, _, _ = vae_loss(logits, batch, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                logits, mu, logvar, z = model(batch)
                loss, _, _ = vae_loss(logits, batch, mu, logvar)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}: Train={train_loss:.4f}, Val={val_loss:.4f}")

    return best_val_loss, history


def analyze_latent_space(model, dataloader, device, curvature: float = 1.0) -> dict:
    """Analyze the learned latent space.

    Args:
        model: VAE model with encode method
        dataloader: DataLoader for the dataset
        device: Device to use
        curvature: Curvature for hyperbolic distance (V5.12.2)
    """
    model.eval()
    all_z = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            all_z.append(mu.cpu())

    all_z_tensor = torch.cat(all_z, dim=0)
    all_z = all_z_tensor.numpy()

    # V5.12.2: Compute hyperbolic radii (distance from origin)
    origin = torch.zeros_like(all_z_tensor)
    hyp_radii = poincare_distance(all_z_tensor, origin, c=curvature).numpy()

    # Compute statistics using hyperbolic distance
    results = {
        "n_samples": len(all_z),
        "latent_dim": all_z.shape[1],
        "mean_norm": float(np.mean(hyp_radii)),
        "std_norm": float(np.std(hyp_radii)),
        "mean_per_dim": [float(x) for x in np.mean(all_z, axis=0)],
        "std_per_dim": [float(x) for x in np.std(all_z, axis=0)],
    }

    # V5.12.2: Compute pairwise hyperbolic distances (sample)
    n_sample = min(200, len(all_z))
    sample_z = all_z_tensor[:n_sample]
    distances = []
    for i in range(n_sample):
        for j in range(i + 1, n_sample):
            dist = poincare_distance(sample_z[i : i + 1], sample_z[j : j + 1], c=curvature).item()
            distances.append(dist)

    results["mean_pairwise_dist"] = float(np.mean(distances))
    results["std_pairwise_dist"] = float(np.std(distances))

    return results


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("=" * 70)
    print("COMPREHENSIVE DATASET ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_dir = PROJECT_ROOT / "data" / "external"
    results_dir = PROJECT_ROOT / "results" / "comprehensive_analysis"
    results_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "datasets": {},
        "models": {},
        "summary": {},
    }

    # =========================================================================
    # PHASE 1: Dataset Discovery
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Dataset Discovery")
    print("=" * 70)

    print("\n[1.1] Analyzing sequence files (FASTA/FNA)...")
    seq_results = analyze_sequence_data(data_dir)
    report["datasets"]["sequences"] = seq_results
    print(f"  Found {len(seq_results['fasta'])} FASTA files, {len(seq_results['fna'])} FNA files")
    print(f"  Total: {seq_results['total_sequences']} sequences, {seq_results['total_codons']:,} codons")

    print("\n[1.2] Analyzing CSV files...")
    csv_results = analyze_csv_data(data_dir)
    report["datasets"]["csv"] = csv_results
    if "error" not in csv_results:
        print(f"  Found {len(csv_results['files'])} CSV files, {csv_results['total_rows']:,} total rows")

    print("\n[1.3] Analyzing Parquet files...")
    pq_results = analyze_parquet_data(data_dir)
    report["datasets"]["parquet"] = pq_results
    if "error" not in pq_results:
        print(f"  Found {len(pq_results['files'])} Parquet files")
        for pf in pq_results["files"]:
            print(f"    - {pf['file']}: {pf['n_rows']:,} rows, {pf['n_columns']} columns")

    # =========================================================================
    # PHASE 2: Data Loading
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Data Loading")
    print("=" * 70)

    max_len = 100
    print(f"\nLoading all sequences (max_len={max_len})...")
    codon_seqs, aa_seqs = load_all_sequences(data_dir, max_len)
    print(f"  Loaded {len(codon_seqs)} codon sequences")
    print(f"  Loaded {len(aa_seqs)} amino acid sequences")

    report["summary"]["codon_sequences"] = len(codon_seqs)
    report["summary"]["aa_sequences"] = len(aa_seqs)

    # =========================================================================
    # PHASE 3: Model Training
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Model Training")
    print("=" * 70)

    # Train Codon VAE
    if len(codon_seqs) >= 20:
        print("\n[3.1] Training Codon VAE...")
        codon_dataset = CodonDataset(codon_seqs, max_len=max_len, source="all")

        train_size = int(0.8 * len(codon_dataset))
        val_size = len(codon_dataset) - train_size
        train_data, val_data = torch.utils.data.random_split(codon_dataset, [train_size, val_size])

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32)

        codon_model = UnifiedVAE(
            n_tokens=64, embed_dim=16, hidden_dim=128, latent_dim=16, max_len=max_len, use_padic=True
        ).to(device)

        n_params = sum(p.numel() for p in codon_model.parameters() if p.requires_grad)
        print(f"  Model parameters: {n_params:,}")
        print(f"  Training on {train_size} samples, validating on {val_size}")

        best_loss, history = train_model(codon_model, train_loader, val_loader, device, n_epochs=30, model_name="codon")

        # Save model
        save_path = OUTPUT_DIR / "models" / "codon_vae_all.pt"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(codon_model.state_dict(), save_path)

        # Analyze latent space
        latent_analysis = analyze_latent_space(codon_model, DataLoader(codon_dataset, batch_size=64), device)

        report["models"]["codon_vae"] = {
            "n_sequences": len(codon_dataset),
            "best_val_loss": best_loss,
            "latent_analysis": latent_analysis,
            "saved_to": str(save_path),
        }

        print(f"  Best validation loss: {best_loss:.4f}")
        print(
            f"  Latent space: mean_norm={latent_analysis['mean_norm']:.4f}, mean_dist={latent_analysis['mean_pairwise_dist']:.4f}"
        )
    else:
        print("\n[3.1] Skipping Codon VAE (insufficient sequences)")

    # Train Amino Acid VAE
    if len(aa_seqs) >= 20:
        print("\n[3.2] Training Amino Acid VAE...")
        aa_dataset = AminoAcidDataset(aa_seqs, max_len=max_len, source="all")

        train_size = int(0.8 * len(aa_dataset))
        val_size = len(aa_dataset) - train_size
        train_data, val_data = torch.utils.data.random_split(aa_dataset, [train_size, val_size])

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=32)

        aa_model = UnifiedVAE(
            n_tokens=21, embed_dim=16, hidden_dim=128, latent_dim=16, max_len=max_len, use_padic=False
        ).to(device)

        n_params = sum(p.numel() for p in aa_model.parameters() if p.requires_grad)
        print(f"  Model parameters: {n_params:,}")
        print(f"  Training on {train_size} samples, validating on {val_size}")

        best_loss, history = train_model(aa_model, train_loader, val_loader, device, n_epochs=30, model_name="aa")

        # Save model
        save_path = OUTPUT_DIR / "models" / "aa_vae_all.pt"
        torch.save(aa_model.state_dict(), save_path)

        # Analyze latent space
        latent_analysis = analyze_latent_space(aa_model, DataLoader(aa_dataset, batch_size=64), device)

        report["models"]["aa_vae"] = {
            "n_sequences": len(aa_dataset),
            "best_val_loss": best_loss,
            "latent_analysis": latent_analysis,
            "saved_to": str(save_path),
        }

        print(f"  Best validation loss: {best_loss:.4f}")
        print(
            f"  Latent space: mean_norm={latent_analysis['mean_norm']:.4f}, mean_dist={latent_analysis['mean_pairwise_dist']:.4f}"
        )
    else:
        print("\n[3.2] Skipping Amino Acid VAE (insufficient sequences)")

    # =========================================================================
    # PHASE 4: Epidemiological Analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Epidemiological Data Analysis")
    print("=" * 70)

    if HAS_PANDAS:
        epi_dir = data_dir / "kaggle" / "hiv-aids-dataset"
        if epi_dir.exists():
            print("\n[4.1] Analyzing HIV/AIDS epidemiological data...")
            epi_results = {}

            for csv_path in epi_dir.glob("*.csv"):
                try:
                    df = pd.read_csv(csv_path)
                    name = csv_path.stem

                    # Get summary stats
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        summary = {
                            "n_rows": len(df),
                            "n_countries": df.iloc[:, 0].nunique() if len(df.columns) > 0 else 0,
                            "years": sorted([c for c in df.columns if c.isdigit()])[:5],
                            "latest_year_mean": float(df[numeric_cols[-1]].mean()) if len(numeric_cols) > 0 else None,
                        }
                        epi_results[name] = summary
                        print(f"  {name}: {summary['n_rows']} rows, {summary['n_countries']} countries")
                except Exception as e:
                    print(f"  Error with {csv_path.name}: {e}")

            report["datasets"]["epidemiological"] = epi_results
    else:
        print("\nSkipping epidemiological analysis (pandas not available)")

    # =========================================================================
    # PHASE 5: Generate Report
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: Final Report")
    print("=" * 70)

    # Summary
    report["summary"]["total_datasets"] = (
        len(seq_results.get("fasta", []))
        + len(seq_results.get("fna", []))
        + len(csv_results.get("files", []))
        + len(pq_results.get("files", []))
    )

    # Save report
    report_path = results_dir / "comprehensive_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")

    # Print summary
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"Total datasets analyzed: {report['summary']['total_datasets']}")
    print(f"Total codon sequences: {report['summary'].get('codon_sequences', 0):,}")
    print(f"Total amino acid sequences: {report['summary'].get('aa_sequences', 0):,}")

    if "codon_vae" in report["models"]:
        m = report["models"]["codon_vae"]
        print("\nCodon VAE:")
        print(f"  - Trained on: {m['n_sequences']} sequences")
        print(f"  - Best loss: {m['best_val_loss']:.4f}")
        print(f"  - Saved to: {m['saved_to']}")

    if "aa_vae" in report["models"]:
        m = report["models"]["aa_vae"]
        print("\nAmino Acid VAE:")
        print(f"  - Trained on: {m['n_sequences']} sequences")
        print(f"  - Best loss: {m['best_val_loss']:.4f}")
        print(f"  - Saved to: {m['saved_to']}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
