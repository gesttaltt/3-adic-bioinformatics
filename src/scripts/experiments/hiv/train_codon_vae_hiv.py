#!/usr/bin/env python3
"""Train Codon VAE on HIV Sequences from Scratch.

This script trains a simple codon VAE directly on HIV FASTA sequences
without requiring any pretrained checkpoints. Uses p-adic geometry
to capture the hierarchical structure of the genetic code.

Usage:
    python src/scripts/train_codon_vae_hiv.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Genetic code mapping (imported from centralized biology module)
from src.biology.codons import (
    CODON_TO_INDEX,
    GENETIC_CODE,
)
from src.config.paths import OUTPUT_DIR

CODONS = list(GENETIC_CODE.keys())


def parse_fasta(filepath: Path) -> list[tuple[str, str]]:
    """Parse FASTA file into list of (name, sequence) tuples."""
    sequences = []
    current_name = None
    current_seq = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name:
                    sequences.append((current_name, "".join(current_seq)))
                current_name = line[1:]
                current_seq = []
            else:
                current_seq.append(line.upper())

        if current_name:
            sequences.append((current_name, "".join(current_seq)))

    return sequences


def sequence_to_codons(seq: str) -> list[int]:
    """Convert DNA sequence to list of codon indices."""
    codons = []
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        if codon in CODON_TO_INDEX:
            codons.append(CODON_TO_INDEX[codon])
    return codons


def compute_padic_distance(codon1: str, codon2: str, p: int = 3) -> float:
    """Compute p-adic distance between two codons.

    The p-adic distance measures how similar codons are based on
    their nucleotide positions, with earlier positions weighted more heavily.
    """
    if codon1 == codon2:
        return 0.0

    # Find first differing position
    for i, (b1, b2) in enumerate(zip(codon1, codon2, strict=False)):
        if b1 != b2:
            return p ** (-(i + 1))  # 1/p^k for difference at position k

    return 0.0


def build_padic_distance_matrix() -> np.ndarray:
    """Build 64x64 p-adic distance matrix for all codons."""
    n_codons = len(CODONS)
    dist_matrix = np.zeros((n_codons, n_codons))

    for i, c1 in enumerate(CODONS):
        for j, c2 in enumerate(CODONS):
            dist_matrix[i, j] = compute_padic_distance(c1, c2)

    return dist_matrix


class HIVCodonDataset(Dataset):
    """Dataset of HIV sequences as codon indices."""

    def __init__(self, fasta_files: list[Path], max_len: int = 100):
        self.sequences = []
        self.max_len = max_len

        for fasta_path in fasta_files:
            if fasta_path.exists():
                seqs = parse_fasta(fasta_path)
                for _name, seq in seqs:
                    codons = sequence_to_codons(seq)
                    if len(codons) >= 10:  # Minimum length
                        self.sequences.append(codons[:max_len])

        print(f"Loaded {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        codons = self.sequences[idx]
        # Pad or truncate to max_len
        if len(codons) < self.max_len:
            codons = codons + [0] * (self.max_len - len(codons))
        return torch.tensor(codons[: self.max_len], dtype=torch.long)


class PAdicCodonEmbedding(nn.Module):
    """Codon embedding initialized from p-adic distance structure."""

    def __init__(self, embed_dim: int = 16, n_codons: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_codons = n_codons

        # Initialize embeddings using MDS on p-adic distances
        dist_matrix = build_padic_distance_matrix()
        embeddings = self._mds_from_distances(dist_matrix, embed_dim)

        self.embedding = nn.Embedding(n_codons, embed_dim)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.tensor(embeddings, dtype=torch.float32))

    def _mds_from_distances(self, D: np.ndarray, dim: int) -> np.ndarray:
        """Classical MDS to get embedding from distance matrix."""
        n = D.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * H @ (D**2) @ H

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(B)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Take top dim positive eigenvalues
        pos_idx = eigenvalues > 1e-10
        eigenvalues = eigenvalues[pos_idx][:dim]
        eigenvectors = eigenvectors[:, pos_idx][:, :dim]

        if len(eigenvalues) < dim:
            # Pad with zeros if not enough dimensions
            pad = dim - len(eigenvalues)
            eigenvalues = np.concatenate([eigenvalues, np.zeros(pad)])
            eigenvectors = np.concatenate([eigenvectors, np.zeros((n, pad))], axis=1)

        return eigenvectors * np.sqrt(eigenvalues)

    def forward(self, x):
        return self.embedding(x)


class CodonVAE(nn.Module):
    """Simple VAE for codon sequences with p-adic structure."""

    def __init__(
        self,
        embed_dim: int = 16,
        hidden_dim: int = 128,
        latent_dim: int = 16,
        max_len: int = 100,
        n_codons: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.max_len = max_len
        self.n_codons = n_codons

        # P-adic initialized codon embedding
        self.codon_embed = PAdicCodonEmbedding(embed_dim, n_codons)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(max_len * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_len * n_codons),
        )

    def encode(self, x):
        """Encode sequence to latent distribution."""
        # x: (batch, seq_len) of codon indices
        emb = self.codon_embed(x)  # (batch, seq_len, embed_dim)
        emb = emb.view(emb.size(0), -1)  # (batch, seq_len * embed_dim)

        h = self.encoder(emb)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """Decode latent to sequence logits."""
        h = self.decoder(z)  # (batch, seq_len * n_codons)
        logits = h.view(h.size(0), self.max_len, self.n_codons)
        return logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar, z


def poincare_project(z: torch.Tensor, max_radius: float = 0.95) -> torch.Tensor:
    """Project points to Poincare ball."""
    norm = torch.norm(z, dim=-1, keepdim=True)
    scale = torch.clamp(norm, min=1e-10)
    return z / scale * torch.tanh(norm) * max_radius


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Compute Poincare ball distance."""
    norm_x_sq = torch.sum(x**2, dim=-1, keepdim=True)
    norm_y_sq = torch.sum(y**2, dim=-1, keepdim=True)
    diff_sq = torch.sum((x - y) ** 2, dim=-1, keepdim=True)

    denom = (1 - c * norm_x_sq) * (1 - c * norm_y_sq)
    denom = torch.clamp(denom, min=1e-10)

    arg = 1 + 2 * c * diff_sq / denom
    arg = torch.clamp(arg, min=1.0 + 1e-10)

    return (1 / np.sqrt(c)) * torch.acosh(arg).squeeze(-1)


def vae_loss(logits, targets, mu, logvar, z, beta=0.1, gamma=0.01):
    """VAE loss with p-adic regularization."""
    logits.size(0)

    # Reconstruction loss (cross-entropy)
    logits_flat = logits.view(-1, logits.size(-1))  # (batch*seq, n_codons)
    targets_flat = targets.view(-1)  # (batch*seq,)
    recon_loss = F.cross_entropy(logits_flat, targets_flat, reduction="mean")

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # Hyperbolic regularization: encourage hierarchical structure
    z_hyp = poincare_project(z)
    radius = torch.norm(z_hyp, dim=-1)
    radius_var = torch.var(radius)  # Encourage spread

    total_loss = recon_loss + beta * kl_loss + gamma * (1 / (radius_var + 1e-6))

    return total_loss, recon_loss, kl_loss


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0

    for _batch_idx, batch in enumerate(dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()

        logits, mu, logvar, z = model(batch)
        loss, recon, kl = vae_loss(logits, batch, mu, logvar, z)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()

    n_batches = len(dataloader)
    return total_loss / n_batches, total_recon / n_batches, total_kl / n_batches


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            logits, mu, logvar, z = model(batch)
            loss, _, _ = vae_loss(logits, batch, mu, logvar, z)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    print("=" * 60)
    print("Codon VAE Training on HIV Sequences")
    print("=" * 60)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Find FASTA files
    data_dir = PROJECT_ROOT / "data" / "external" / "github" / "HIV-1_Paper"
    fasta_dir = data_dir / "Individual_Representative_Sequences_Used_for_Subtyping"

    fasta_files = list(fasta_dir.glob("*.fasta"))
    print(f"Found {len(fasta_files)} FASTA files")

    if not fasta_files:
        print("No FASTA files found. Please check the data directory.")
        return

    # Create dataset
    max_len = 100  # Max codons per sequence
    dataset = HIVCodonDataset(fasta_files, max_len=max_len)

    if len(dataset) == 0:
        print("No valid sequences found in FASTA files.")
        return

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    model = CodonVAE(
        embed_dim=16,
        hidden_dim=128,
        latent_dim=16,
        max_len=max_len,
        n_codons=64,
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Training
    n_epochs = 50
    best_val_loss = float("inf")

    print("\nTraining:")
    print("-" * 50)

    for epoch in range(1, n_epochs + 1):
        train_loss, recon_loss, kl_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = evaluate(model, val_loader, device)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            save_path = OUTPUT_DIR / "models" / "codon_vae_hiv.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                save_path,
            )
            marker = " *"
        else:
            marker = ""

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train: {train_loss:.4f} (Recon: {recon_loss:.4f}, KL: {kl_loss:.4f}) | "
                f"Val: {val_loss:.4f}{marker}"
            )

    print("-" * 50)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {OUTPUT_DIR / 'models' / 'codon_vae_hiv.pt'}")

    # Final analysis
    print("\n" + "=" * 60)
    print("Latent Space Analysis")
    print("=" * 60)

    model.eval()
    all_z = []
    all_seqs = []

    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=64, shuffle=False):
            batch = batch.to(device)
            mu, _ = model.encode(batch)
            z_hyp = poincare_project(mu)
            all_z.append(z_hyp.cpu())
            all_seqs.append(batch.cpu())

    all_z = torch.cat(all_z, dim=0)

    # Analyze structure
    radii = torch.norm(all_z, dim=-1)
    print(f"Latent radius: mean={radii.mean():.4f}, std={radii.std():.4f}")
    print(f"Latent radius: min={radii.min():.4f}, max={radii.max():.4f}")

    # Compute pairwise distances in hyperbolic space
    n_samples = min(100, len(all_z))
    sample_z = all_z[:n_samples]

    distances = []
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = poincare_distance(sample_z[i].unsqueeze(0), sample_z[j].unsqueeze(0))
            distances.append(d.item())

    distances = np.array(distances)
    print(f"Pairwise distances: mean={distances.mean():.4f}, std={distances.std():.4f}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
