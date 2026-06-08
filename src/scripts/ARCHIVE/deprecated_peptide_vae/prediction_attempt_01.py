#!/usr/bin/env python3
"""PeptideVAE Training - Prediction Attempt 01

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!! IMPORTANT DISCLAIMER - DEPRECATED                                        !!
!!                                                                          !!
!! This script implements a NAIVE LSTM VAE from scratch and is DEPRECATED.  !!
!! It completely ignores the existing infrastructure:                       !!
!!   - src/encoders/peptide_encoder.py (full PeptideVAE with Transformer)   !!
!!   - src/encoders/trainable_codon_encoder.py (p-adic codon embeddings)    !!
!!   - src/geometry/ (proper hyperbolic math)                               !!
!!   - TernaryVAE v5.11/v5.12 architecture                                  !!
!!                                                                          !!
!! USE INSTEAD: src/encoders/peptide_encoder.PeptideVAE                     !!
!! This file is kept for historical reference only.                         !!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Evidence-based training following PEPTIDE_VAE_TRAINING_PLAN.md

Validated findings incorporated:
- C3: Cluster signal is REAL (length, charge, hydrophobicity matter)
- C5: Pathogen metadata HURTS (NO pathogen labels in model)
- Colbes: DDG benchmark ρ=0.585 is ground truth validation

Architecture:
- Hyperbolic latent space with poincare_distance (NOT .norm())
- Physicochemical auxiliary heads (C3 validated)
- NO pathogen conditioning (C5 falsified)

Phases:
1. Coverage + Physicochemical grounding
2. Radial hierarchy from activity
3. Cluster structure enhancement
4. DDG validation head (CRITICAL: must achieve ρ ≥ 0.585)

Usage:
    python src/scripts/peptide_vae/prediction_attempt_01.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from scipy.stats import pearsonr, spearmanr

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Amino acid vocabulary
AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
PAD_IDX = len(AA_VOCAB)  # 20 = padding

# Physicochemical properties (from Colbes validation)
AA_PROPERTIES = {
    "A": {"hydro": 0.62, "charge": 0.0, "size": -0.77},
    "C": {"hydro": 0.29, "charge": 0.0, "size": -0.41},
    "D": {"hydro": -0.90, "charge": -1.0, "size": -0.32},
    "E": {"hydro": -0.74, "charge": -1.0, "size": 0.40},
    "F": {"hydro": 1.19, "charge": 0.0, "size": 0.81},
    "G": {"hydro": 0.48, "charge": 0.0, "size": -1.00},
    "H": {"hydro": -0.40, "charge": 0.5, "size": 0.18},
    "I": {"hydro": 1.38, "charge": 0.0, "size": 0.55},
    "K": {"hydro": -1.50, "charge": 1.0, "size": 0.46},
    "L": {"hydro": 1.06, "charge": 0.0, "size": 0.55},
    "M": {"hydro": 0.64, "charge": 0.0, "size": 0.46},
    "N": {"hydro": -0.78, "charge": 0.0, "size": -0.18},
    "P": {"hydro": 0.12, "charge": 0.0, "size": -0.45},
    "Q": {"hydro": -0.85, "charge": 0.0, "size": 0.18},
    "R": {"hydro": -2.53, "charge": 1.0, "size": 0.64},
    "S": {"hydro": -0.18, "charge": 0.0, "size": -0.59},
    "T": {"hydro": -0.05, "charge": 0.0, "size": -0.32},
    "V": {"hydro": 1.08, "charge": 0.0, "size": 0.14},
    "W": {"hydro": 0.81, "charge": 0.0, "size": 1.00},
    "Y": {"hydro": 0.26, "charge": 0.0, "size": 0.73},
}


def compute_physicochemical(sequence: str) -> dict:
    """Compute physicochemical properties for a peptide sequence."""
    length = len(sequence)

    total_hydro = 0.0
    total_charge = 0.0

    for aa in sequence:
        if aa in AA_PROPERTIES:
            total_hydro += AA_PROPERTIES[aa]["hydro"]
            total_charge += AA_PROPERTIES[aa]["charge"]

    return {
        "length": length,
        "hydrophobicity": total_hydro / max(length, 1),
        "net_charge": total_charge,
    }


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Compute hyperbolic distance in Poincare ball.

    CRITICAL: Use this instead of torch.norm() for radial computations!
    V5.12.2 audit showed .norm() breaks hierarchy.
    """
    sqrt_c = c**0.5

    # Mobius addition: -x + y
    x_sq = (x * x).sum(dim=-1, keepdim=True)
    y_sq = (y * y).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)

    num = (1 + 2 * c * xy + c * y_sq) * (-x) + (1 - c * x_sq) * y
    denom = 1 + 2 * c * (-xy) + c**2 * x_sq * y_sq
    diff = num / torch.clamp(denom, min=1e-8)

    diff_norm = torch.clamp(torch.norm(diff, dim=-1), min=1e-8, max=1 - 1e-5)
    return 2.0 / sqrt_c * torch.atanh(sqrt_c * diff_norm)


def exp_map_zero(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Exponential map from tangent space at origin to Poincare ball."""
    sqrt_c = c**0.5
    v_norm = torch.clamp(torch.norm(v, dim=-1, keepdim=True), min=1e-8)
    scale = torch.tanh(sqrt_c * v_norm / 2.0) / (sqrt_c * v_norm)
    return v * scale


@dataclass
class PeptideVAEConfig:
    """Configuration for PeptideVAE."""

    vocab_size: int = 21  # 20 AAs + padding
    max_seq_len: int = 50
    embed_dim: int = 64
    hidden_dim: int = 128
    latent_dim: int = 16
    n_layers: int = 2
    dropout: float = 0.1
    curvature: float = 1.0

    # Loss weights (from training plan)
    recon_weight: float = 1.0
    kl_weight: float = 0.1
    phys_weight: float = 1.0  # C3 validated: physicochemical features matter


class PeptideDataset(Dataset):
    """Dataset for peptide sequences with physicochemical properties."""

    def __init__(self, sequences: list[str], max_len: int = 50):
        self.sequences = sequences
        self.max_len = max_len

        # Precompute physicochemical properties
        self.properties = [compute_physicochemical(seq) for seq in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        props = self.properties[idx]

        # Encode sequence
        encoded = [AA_TO_IDX.get(aa, PAD_IDX) for aa in seq[: self.max_len]]

        # Pad to max_len
        padding = [PAD_IDX] * (self.max_len - len(encoded))
        encoded = encoded + padding

        return {
            "sequence": torch.tensor(encoded, dtype=torch.long),
            "length": torch.tensor(props["length"], dtype=torch.float),
            "hydrophobicity": torch.tensor(props["hydrophobicity"], dtype=torch.float),
            "net_charge": torch.tensor(props["net_charge"], dtype=torch.float),
            "raw_sequence": seq,
        }


class PeptideEncoder(nn.Module):
    """Encoder for peptide sequences to hyperbolic latent space."""

    def __init__(self, config: PeptideVAEConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=PAD_IDX)

        self.lstm = nn.LSTM(
            config.embed_dim,
            config.hidden_dim,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0,
            bidirectional=True,
        )

        # Output: mu and logvar for reparameterization
        self.fc_mu = nn.Linear(config.hidden_dim * 2, config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim * 2, config.latent_dim)

        # LayerNorm for stability (V5.12.4 improvement)
        self.layer_norm = nn.LayerNorm(config.hidden_dim * 2)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)

        # LSTM encoding
        output, (h_n, _) = self.lstm(embedded)

        # Concatenate forward and backward final states
        h_forward = h_n[-2]  # (batch, hidden_dim)
        h_backward = h_n[-1]  # (batch, hidden_dim)
        h = torch.cat([h_forward, h_backward], dim=-1)  # (batch, hidden_dim * 2)

        h = self.layer_norm(h)
        h = self.dropout(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Clamp logvar to prevent KL collapse/explosion (V5.12.4)
        logvar = torch.clamp(logvar, min=-10, max=2)

        return mu, logvar


class PeptideDecoder(nn.Module):
    """Decoder from hyperbolic latent space to peptide sequences."""

    def __init__(self, config: PeptideVAEConfig):
        super().__init__()
        self.config = config

        self.fc = nn.Linear(config.latent_dim, config.hidden_dim)

        self.lstm = nn.LSTM(
            config.hidden_dim + config.embed_dim,
            config.hidden_dim,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0,
        )

        self.output = nn.Linear(config.hidden_dim, config.vocab_size)
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        self.layer_norm = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        z: torch.Tensor,
        target: torch.Tensor | None = None,
        max_len: int = 50,
    ) -> torch.Tensor:
        batch_size = z.size(0)
        device = z.device

        # Project latent to hidden
        h = self.fc(z)  # (batch, hidden_dim)
        h = self.layer_norm(h)

        # Initialize LSTM state
        h_0 = h.unsqueeze(0).repeat(self.config.n_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)

        # Autoregressive decoding
        outputs = []

        if target is not None:
            # Teacher forcing during training
            max_len = target.size(1)
            embedded_target = self.embedding(target)  # (batch, seq_len, embed_dim)

            # Expand latent for each position
            z_expanded = h.unsqueeze(1).expand(-1, max_len, -1)

            # Concatenate
            lstm_input = torch.cat([z_expanded, embedded_target], dim=-1)

            output, _ = self.lstm(lstm_input, (h_0, c_0))
            logits = self.output(output)  # (batch, seq_len, vocab_size)
        else:
            # Autoregressive decoding during inference
            current_token = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            hidden = (h_0, c_0)

            for _ in range(max_len):
                embedded = self.embedding(current_token)
                lstm_input = torch.cat([h.unsqueeze(1), embedded], dim=-1)
                output, hidden = self.lstm(lstm_input, hidden)
                logit = self.output(output)
                outputs.append(logit)
                current_token = logit.argmax(dim=-1)

            logits = torch.cat(outputs, dim=1)

        return logits


class PhysicochemicalHead(nn.Module):
    """Auxiliary head for physicochemical property prediction.

    C3 VALIDATED: These properties ARE the signal sources.
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),  # length, hydrophobicity, charge
        )

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.net(z)
        return {
            "length_pred": out[:, 0],
            "hydro_pred": out[:, 1],
            "charge_pred": out[:, 2],
        }


class PeptideVAE(nn.Module):
    """PeptideVAE with hyperbolic latent space and physicochemical heads.

    Architecture follows PEPTIDE_VAE_TRAINING_PLAN.md:
    - Hyperbolic latent space (poincare_distance, NOT .norm())
    - Physicochemical auxiliary heads (C3 validated)
    - NO pathogen conditioning (C5 falsified)
    """

    def __init__(self, config: PeptideVAEConfig):
        super().__init__()
        self.config = config

        self.encoder = PeptideEncoder(config)
        self.decoder = PeptideDecoder(config)
        self.phys_head = PhysicochemicalHead(config.latent_dim)

        self.curvature = config.curvature

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_euc = mu + eps * std
        else:
            z_euc = mu

        # Project to hyperbolic space
        z_hyp = exp_map_zero(z_euc, self.curvature)
        return z_hyp

    def forward(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        # Encode
        mu, logvar = self.encoder(x)

        # Reparameterize to hyperbolic space
        z_hyp = self.reparameterize(mu, logvar)

        # Decode (from hyperbolic latent)
        logits = self.decoder(z_hyp, target)

        # Physicochemical predictions (C3 validated)
        phys_preds = self.phys_head(z_hyp)

        # Compute radii using HYPERBOLIC distance (NOT .norm()!)
        origin = torch.zeros_like(z_hyp)
        radii = poincare_distance(z_hyp, origin, self.curvature)

        return {
            "logits": logits,
            "mu": mu,
            "logvar": logvar,
            "z_hyp": z_hyp,
            "radii": radii,
            **phys_preds,
        }

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute VAE loss with physicochemical auxiliary loss.

        Loss = reconstruction + β * KL + λ_phys * physicochemical
        """
        config = self.config

        # Reconstruction loss (cross-entropy)
        logits = outputs["logits"]
        target = batch["sequence"]

        # Reshape for cross-entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        target_flat = target.view(-1)

        # Mask padding
        mask = target_flat != PAD_IDX
        recon_loss = F.cross_entropy(logits_flat[mask], target_flat[mask], reduction="mean")

        # KL divergence
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Physicochemical loss (C3 validated - these ARE the signal sources)
        length_true = batch["length"]
        hydro_true = batch["hydrophobicity"]
        charge_true = batch["net_charge"]

        # Normalize length for loss computation
        length_pred = outputs["length_pred"] * 10  # Scale up
        length_target = length_true

        phys_loss = (
            F.mse_loss(length_pred, length_target)
            + F.mse_loss(outputs["hydro_pred"], hydro_true)
            + F.mse_loss(outputs["charge_pred"], charge_true)
        )

        # Total loss
        total_loss = config.recon_weight * recon_loss + config.kl_weight * kl_loss + config.phys_weight * phys_loss

        return {
            "total": total_loss,
            "recon": recon_loss,
            "kl": kl_loss,
            "phys": phys_loss,
        }


def load_peptide_data(data_dir: Path) -> list[str]:
    """Load peptide sequences from validation batch CSVs."""
    sequences = []

    csv_files = list(data_dir.glob("*.csv"))

    for csv_file in csv_files:
        with open(csv_file) as f:
            lines = f.readlines()

        # Skip header
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) > 0:
                seq = parts[0].strip()
                # Validate sequence
                if seq and all(aa in AA_VOCAB for aa in seq):
                    sequences.append(seq)

    # Remove duplicates
    sequences = list(set(sequences))

    return sequences


def train_epoch(
    model: PeptideVAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_phys = 0.0
    n_batches = 0

    for batch in dataloader:
        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch["sequence"], target=batch["sequence"])

        # Compute loss
        losses = model.compute_loss(outputs, batch)

        # Backward pass
        losses["total"].backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += losses["total"].item()
        total_recon += losses["recon"].item()
        total_kl += losses["kl"].item()
        total_phys += losses["phys"].item()
        n_batches += 1

    return {
        "total": total_loss / n_batches,
        "recon": total_recon / n_batches,
        "kl": total_kl / n_batches,
        "phys": total_phys / n_batches,
    }


@torch.no_grad()
def evaluate(
    model: PeptideVAE,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model."""
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_phys = 0.0
    n_batches = 0

    # For physicochemical correlation
    all_length_pred = []
    all_length_true = []
    all_hydro_pred = []
    all_hydro_true = []
    all_charge_pred = []
    all_charge_true = []

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        outputs = model(batch["sequence"], target=batch["sequence"])
        losses = model.compute_loss(outputs, batch)

        total_loss += losses["total"].item()
        total_recon += losses["recon"].item()
        total_kl += losses["kl"].item()
        total_phys += losses["phys"].item()
        n_batches += 1

        # Collect predictions for correlation
        all_length_pred.extend((outputs["length_pred"] * 10).cpu().numpy())
        all_length_true.extend(batch["length"].cpu().numpy())
        all_hydro_pred.extend(outputs["hydro_pred"].cpu().numpy())
        all_hydro_true.extend(batch["hydrophobicity"].cpu().numpy())
        all_charge_pred.extend(outputs["charge_pred"].cpu().numpy())
        all_charge_true.extend(batch["net_charge"].cpu().numpy())

    # Compute correlations (C3 validation)
    results = {
        "total": total_loss / n_batches,
        "recon": total_recon / n_batches,
        "kl": total_kl / n_batches,
        "phys": total_phys / n_batches,
    }

    if HAS_SCIPY and len(all_length_pred) > 10:
        length_r, _ = pearsonr(all_length_pred, all_length_true)
        hydro_r, _ = pearsonr(all_hydro_pred, all_hydro_true)
        charge_r, _ = pearsonr(all_charge_pred, all_charge_true)

        results["length_r"] = length_r
        results["hydro_r"] = hydro_r
        results["charge_r"] = charge_r

    return results


def compute_reconstruction_accuracy(
    model: PeptideVAE,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Compute sequence reconstruction accuracy."""
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(batch["sequence"], target=batch["sequence"])

            # Get predictions
            pred = outputs["logits"].argmax(dim=-1)
            target = batch["sequence"]

            # Count correct (excluding padding)
            mask = target != PAD_IDX
            correct += ((pred == target) & mask).sum().item()
            total += mask.sum().item()

    return correct / total if total > 0 else 0.0


def main():
    print("=" * 70)
    print("PeptideVAE Training - Prediction Attempt 01")
    print("Evidence-based training following validated findings")
    print("=" * 70)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Configuration
    config = PeptideVAEConfig(
        vocab_size=21,
        max_seq_len=50,
        embed_dim=64,
        hidden_dim=128,
        latent_dim=16,
        n_layers=2,
        dropout=0.1,
        curvature=1.0,
        recon_weight=1.0,
        kl_weight=0.1,
        phys_weight=1.0,  # C3 validated
    )

    print(f"\nConfig: latent_dim={config.latent_dim}, hidden_dim={config.hidden_dim}")

    # Load data
    data_dir = PROJECT_ROOT / "deliverables" / "partners" / "carlos_brizuela" / "results" / "validation_batch"

    print(f"\nLoading data from: {data_dir}")
    sequences = load_peptide_data(data_dir)
    print(f"Loaded {len(sequences)} unique peptide sequences")

    if len(sequences) < 10:
        print("\nERROR: Not enough sequences for training!")
        return 1

    # Show sample sequences
    print("\nSample sequences:")
    for seq in sequences[:5]:
        props = compute_physicochemical(seq)
        print(f"  {seq}: len={props['length']}, hydro={props['hydrophobicity']:.2f}, charge={props['net_charge']:.1f}")

    # Split data
    np.random.seed(42)
    indices = np.random.permutation(len(sequences))
    n_train = int(0.8 * len(sequences))

    train_seqs = [sequences[i] for i in indices[:n_train]]
    val_seqs = [sequences[i] for i in indices[n_train:]]

    print(f"\nTrain: {len(train_seqs)}, Val: {len(val_seqs)}")

    # Create datasets
    train_dataset = PeptideDataset(train_seqs, max_len=config.max_seq_len)
    val_dataset = PeptideDataset(val_seqs, max_len=config.max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Create model
    model = PeptideVAE(config).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Training loop
    n_epochs = 100
    best_val_loss = float("inf")

    print("\n" + "=" * 70)
    print("PHASE 1: Coverage + Physicochemical Grounding")
    print("=" * 70)

    for epoch in range(n_epochs):
        train_losses = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        # Log every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            print(
                f"  Train: total={train_losses['total']:.4f}, recon={train_losses['recon']:.4f}, "
                f"kl={train_losses['kl']:.4f}, phys={train_losses['phys']:.4f}"
            )
            print(f"  Val:   total={val_metrics['total']:.4f}, recon={val_metrics['recon']:.4f}")

            if "length_r" in val_metrics:
                print(
                    f"  Physicochemical R: length={val_metrics['length_r']:.3f}, "
                    f"hydro={val_metrics['hydro_r']:.3f}, charge={val_metrics['charge_r']:.3f}"
                )

        # Save best model
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "val_loss": best_val_loss,
                },
                PROJECT_ROOT / "checkpoints" / "peptide_vae_attempt_01.pt",
            )

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    # Reconstruction accuracy
    recon_acc = compute_reconstruction_accuracy(model, val_loader, device)
    print(f"\nReconstruction Accuracy: {recon_acc:.1%}")

    # Final metrics
    final_metrics = evaluate(model, val_loader, device)
    print("\nFinal Validation Metrics:")
    print(f"  Total Loss: {final_metrics['total']:.4f}")
    print(f"  Reconstruction: {final_metrics['recon']:.4f}")

    if "length_r" in final_metrics:
        print("\nPhysicochemical Encoding (C3 Validation):")
        print(f"  Length R: {final_metrics['length_r']:.3f}")
        print(f"  Hydrophobicity R: {final_metrics['hydro_r']:.3f}")
        print(f"  Charge R: {final_metrics['charge_r']:.3f}")

        # Check validation criteria
        mean_r = np.mean([final_metrics["length_r"], final_metrics["hydro_r"], final_metrics["charge_r"]])

        if mean_r >= 0.8:
            print(f"\n✓ PHASE 1 PASSED: Mean physicochemical R = {mean_r:.3f} ≥ 0.8")
        else:
            print(f"\n✗ PHASE 1 NEEDS WORK: Mean physicochemical R = {mean_r:.3f} < 0.8")

    # Save results
    results = {
        "attempt": "prediction_attempt_01",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "latent_dim": config.latent_dim,
            "hidden_dim": config.hidden_dim,
            "n_epochs": n_epochs,
        },
        "results": {
            "reconstruction_accuracy": float(recon_acc),
            "final_val_loss": float(final_metrics["total"]),
            "physicochemical_r": {
                "length": float(final_metrics.get("length_r", 0)),
                "hydrophobicity": float(final_metrics.get("hydro_r", 0)),
                "charge": float(final_metrics.get("charge_r", 0)),
            },
        },
        "validation_status": {
            "phase_1_coverage": recon_acc >= 0.95,
            "phase_1_physicochemical": final_metrics.get("length_r", 0) >= 0.8,
        },
    }

    results_path = PROJECT_ROOT / "results" / "peptide_vae_attempt_01.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
