#!/usr/bin/env python3
"""PeptideVAE Training - Prediction Attempt 02

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

Fixes from Attempt 01:
- Hydrophobicity R was only 0.083 (needs to be primary predictor per Colbes)
- Added z-score normalization for physicochemical targets
- Increased weight on hydrophobicity (3x) since it's the PRIMARY signal

Key findings from Colbes:
- Hydrophobicity importance: 0.633 (HIGHEST)
- delta_hydro coefficient: 0.31
- This MUST be learned well for DDG validation

Usage:
    python src/scripts/peptide_vae/prediction_attempt_02.py
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
PAD_IDX = len(AA_VOCAB)

# Physicochemical properties (Kyte-Doolittle hydrophobicity scale)
AA_PROPERTIES = {
    "A": {"hydro": 1.8, "charge": 0.0, "size": 89},
    "C": {"hydro": 2.5, "charge": 0.0, "size": 121},
    "D": {"hydro": -3.5, "charge": -1.0, "size": 133},
    "E": {"hydro": -3.5, "charge": -1.0, "size": 147},
    "F": {"hydro": 2.8, "charge": 0.0, "size": 165},
    "G": {"hydro": -0.4, "charge": 0.0, "size": 75},
    "H": {"hydro": -3.2, "charge": 0.5, "size": 155},
    "I": {"hydro": 4.5, "charge": 0.0, "size": 131},
    "K": {"hydro": -3.9, "charge": 1.0, "size": 146},
    "L": {"hydro": 3.8, "charge": 0.0, "size": 131},
    "M": {"hydro": 1.9, "charge": 0.0, "size": 149},
    "N": {"hydro": -3.5, "charge": 0.0, "size": 132},
    "P": {"hydro": -1.6, "charge": 0.0, "size": 115},
    "Q": {"hydro": -3.5, "charge": 0.0, "size": 146},
    "R": {"hydro": -4.5, "charge": 1.0, "size": 174},
    "S": {"hydro": -0.8, "charge": 0.0, "size": 105},
    "T": {"hydro": -0.7, "charge": 0.0, "size": 119},
    "V": {"hydro": 4.2, "charge": 0.0, "size": 117},
    "W": {"hydro": -0.9, "charge": 0.0, "size": 204},
    "Y": {"hydro": -1.3, "charge": 0.0, "size": 181},
}


def compute_physicochemical(sequence: str) -> dict:
    """Compute physicochemical properties for a peptide."""
    length = len(sequence)
    total_hydro = 0.0
    total_charge = 0.0
    total_size = 0.0

    for aa in sequence:
        if aa in AA_PROPERTIES:
            total_hydro += AA_PROPERTIES[aa]["hydro"]
            total_charge += AA_PROPERTIES[aa]["charge"]
            total_size += AA_PROPERTIES[aa]["size"]

    return {
        "length": length,
        "hydrophobicity": total_hydro / max(length, 1),  # Average hydropathy
        "net_charge": total_charge,
        "avg_size": total_size / max(length, 1),
    }


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Hyperbolic distance in Poincare ball. Use this, NOT .norm()!"""
    sqrt_c = c**0.5
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
    vocab_size: int = 21
    max_seq_len: int = 50
    embed_dim: int = 64
    hidden_dim: int = 128
    latent_dim: int = 16
    n_layers: int = 2
    dropout: float = 0.1
    curvature: float = 1.0
    recon_weight: float = 1.0
    kl_weight: float = 0.1
    # CRITICAL: Hydrophobicity is PRIMARY predictor (Colbes 0.633)
    hydro_weight: float = 3.0  # 3x weight on hydrophobicity
    length_weight: float = 1.0
    charge_weight: float = 1.0


class PeptideDataset(Dataset):
    """Dataset with z-score normalized physicochemical targets."""

    def __init__(self, sequences: list[str], max_len: int = 50, stats: dict = None):
        self.sequences = sequences
        self.max_len = max_len
        self.properties = [compute_physicochemical(seq) for seq in sequences]

        # Compute or use provided statistics for normalization
        if stats is None:
            lengths = [p["length"] for p in self.properties]
            hydros = [p["hydrophobicity"] for p in self.properties]
            charges = [p["net_charge"] for p in self.properties]

            self.stats = {
                "length_mean": np.mean(lengths),
                "length_std": np.std(lengths) + 1e-8,
                "hydro_mean": np.mean(hydros),
                "hydro_std": np.std(hydros) + 1e-8,
                "charge_mean": np.mean(charges),
                "charge_std": np.std(charges) + 1e-8,
            }
        else:
            self.stats = stats

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        props = self.properties[idx]

        encoded = [AA_TO_IDX.get(aa, PAD_IDX) for aa in seq[: self.max_len]]
        padding = [PAD_IDX] * (self.max_len - len(encoded))
        encoded = encoded + padding

        # Z-score normalize targets for better learning
        length_norm = (props["length"] - self.stats["length_mean"]) / self.stats["length_std"]
        hydro_norm = (props["hydrophobicity"] - self.stats["hydro_mean"]) / self.stats["hydro_std"]
        charge_norm = (props["net_charge"] - self.stats["charge_mean"]) / self.stats["charge_std"]

        return {
            "sequence": torch.tensor(encoded, dtype=torch.long),
            "length": torch.tensor(props["length"], dtype=torch.float),
            "length_norm": torch.tensor(length_norm, dtype=torch.float),
            "hydrophobicity": torch.tensor(props["hydrophobicity"], dtype=torch.float),
            "hydro_norm": torch.tensor(hydro_norm, dtype=torch.float),
            "net_charge": torch.tensor(props["net_charge"], dtype=torch.float),
            "charge_norm": torch.tensor(charge_norm, dtype=torch.float),
            "raw_sequence": seq,
        }


class PeptideEncoder(nn.Module):
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
        self.fc_mu = nn.Linear(config.hidden_dim * 2, config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim * 2, config.latent_dim)
        self.layer_norm = nn.LayerNorm(config.hidden_dim * 2)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (h_n, _) = self.lstm(embedded)
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        h = self.layer_norm(h)
        h = self.dropout(h)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=2)
        return mu, logvar


class PeptideDecoder(nn.Module):
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

    def forward(self, z, target=None, max_len=50):
        batch_size = z.size(0)
        h = self.layer_norm(self.fc(z))
        h_0 = h.unsqueeze(0).repeat(self.config.n_layers, 1, 1)
        c_0 = torch.zeros_like(h_0)

        if target is not None:
            max_len = target.size(1)
            embedded_target = self.embedding(target)
            z_expanded = h.unsqueeze(1).expand(-1, max_len, -1)
            lstm_input = torch.cat([z_expanded, embedded_target], dim=-1)
            output, _ = self.lstm(lstm_input, (h_0, c_0))
            logits = self.output(output)
        else:
            outputs = []
            current_token = torch.zeros(batch_size, 1, dtype=torch.long, device=z.device)
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
    """Improved head with dedicated hydrophobicity branch.

    Hydrophobicity is the PRIMARY predictor (Colbes 0.633 importance).
    """

    def __init__(self, latent_dim: int, hidden_dim: int = 64):
        super().__init__()

        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

        # Dedicated branches for each property
        # Hydrophobicity gets more capacity (it's the PRIMARY predictor)
        self.hydro_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.length_branch = nn.Linear(hidden_dim, 1)
        self.charge_branch = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        shared = self.shared(z)
        return {
            "length_pred": self.length_branch(shared).squeeze(-1),
            "hydro_pred": self.hydro_branch(shared).squeeze(-1),
            "charge_pred": self.charge_branch(shared).squeeze(-1),
        }


class PeptideVAE(nn.Module):
    def __init__(self, config: PeptideVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = PeptideEncoder(config)
        self.decoder = PeptideDecoder(config)
        self.phys_head = PhysicochemicalHead(config.latent_dim)
        self.curvature = config.curvature

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_euc = mu + eps * std
        else:
            z_euc = mu
        return exp_map_zero(z_euc, self.curvature)

    def forward(self, x, target=None):
        mu, logvar = self.encoder(x)
        z_hyp = self.reparameterize(mu, logvar)
        logits = self.decoder(z_hyp, target)
        phys_preds = self.phys_head(z_hyp)

        origin = torch.zeros_like(z_hyp)
        radii = poincare_distance(z_hyp, origin, self.curvature)

        return {"logits": logits, "mu": mu, "logvar": logvar, "z_hyp": z_hyp, "radii": radii, **phys_preds}

    def compute_loss(self, outputs, batch):
        config = self.config

        # Reconstruction loss
        logits = outputs["logits"]
        target = batch["sequence"]
        logits_flat = logits.view(-1, config.vocab_size)
        target_flat = target.view(-1)
        mask = target_flat != PAD_IDX
        recon_loss = F.cross_entropy(logits_flat[mask], target_flat[mask])

        # KL divergence
        mu, logvar = outputs["mu"], outputs["logvar"]
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        # Physicochemical loss with SEPARATE weights
        # Use NORMALIZED targets for training
        length_loss = F.mse_loss(outputs["length_pred"], batch["length_norm"])
        hydro_loss = F.mse_loss(outputs["hydro_pred"], batch["hydro_norm"])
        charge_loss = F.mse_loss(outputs["charge_pred"], batch["charge_norm"])

        # CRITICAL: Weight hydrophobicity more heavily (primary predictor)
        phys_loss = (
            config.length_weight * length_loss
            + config.hydro_weight * hydro_loss  # 3x weight
            + config.charge_weight * charge_loss
        )

        total_loss = config.recon_weight * recon_loss + config.kl_weight * kl_loss + phys_loss

        return {
            "total": total_loss,
            "recon": recon_loss,
            "kl": kl_loss,
            "phys": phys_loss,
            "length_loss": length_loss,
            "hydro_loss": hydro_loss,
            "charge_loss": charge_loss,
        }


def load_peptide_data(data_dir: Path) -> list[str]:
    sequences = []
    for csv_file in data_dir.glob("*.csv"):
        with open(csv_file) as f:
            lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(",")
            if parts:
                seq = parts[0].strip()
                if seq and all(aa in AA_VOCAB for aa in seq):
                    sequences.append(seq)
    return list(set(sequences))


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    totals = {"total": 0, "recon": 0, "kl": 0, "phys": 0, "hydro_loss": 0, "length_loss": 0, "charge_loss": 0}
    n = 0

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(batch["sequence"], target=batch["sequence"])
        losses = model.compute_loss(outputs, batch)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        for k in totals:
            if k in losses:
                totals[k] += losses[k].item()
        n += 1

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(model, dataloader, device, stats):
    model.eval()
    totals = {"total": 0, "recon": 0, "kl": 0, "phys": 0}
    n = 0

    all_preds = {"length": [], "hydro": [], "charge": []}
    all_true = {"length": [], "hydro": [], "charge": []}

    for batch in dataloader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(batch["sequence"], target=batch["sequence"])
        losses = model.compute_loss(outputs, batch)

        for k in totals:
            if k in losses:
                totals[k] += losses[k].item()
        n += 1

        # Denormalize predictions for correlation
        length_pred = outputs["length_pred"].cpu().numpy() * stats["length_std"] + stats["length_mean"]
        hydro_pred = outputs["hydro_pred"].cpu().numpy() * stats["hydro_std"] + stats["hydro_mean"]
        charge_pred = outputs["charge_pred"].cpu().numpy() * stats["charge_std"] + stats["charge_mean"]

        all_preds["length"].extend(length_pred)
        all_preds["hydro"].extend(hydro_pred)
        all_preds["charge"].extend(charge_pred)

        all_true["length"].extend(batch["length"].cpu().numpy())
        all_true["hydro"].extend(batch["hydrophobicity"].cpu().numpy())
        all_true["charge"].extend(batch["net_charge"].cpu().numpy())

    results = {k: v / n for k, v in totals.items()}

    if HAS_SCIPY and len(all_preds["length"]) > 10:
        results["length_r"], _ = pearsonr(all_preds["length"], all_true["length"])
        results["hydro_r"], _ = pearsonr(all_preds["hydro"], all_true["hydro"])
        results["charge_r"], _ = pearsonr(all_preds["charge"], all_true["charge"])

    return results


def compute_reconstruction_accuracy(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch["sequence"], target=batch["sequence"])
            pred = outputs["logits"].argmax(dim=-1)
            target = batch["sequence"]
            mask = target != PAD_IDX
            correct += ((pred == target) & mask).sum().item()
            total += mask.sum().item()
    return correct / total if total > 0 else 0.0


def main():
    print("=" * 70)
    print("PeptideVAE Training - Prediction Attempt 02")
    print("FIX: Hydrophobicity weighting (3x) + z-score normalization")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    config = PeptideVAEConfig(
        hydro_weight=3.0,  # CRITICAL: 3x weight on hydrophobicity
        length_weight=1.0,
        charge_weight=1.0,
    )

    data_dir = PROJECT_ROOT / "deliverables" / "partners" / "carlos_brizuela" / "results" / "validation_batch"
    print(f"\nLoading data from: {data_dir}")
    sequences = load_peptide_data(data_dir)
    print(f"Loaded {len(sequences)} unique peptide sequences")

    # Show data statistics
    props = [compute_physicochemical(seq) for seq in sequences]
    hydros = [p["hydrophobicity"] for p in props]
    print("\nHydrophobicity distribution:")
    print(f"  Mean: {np.mean(hydros):.3f}, Std: {np.std(hydros):.3f}")
    print(f"  Min: {np.min(hydros):.3f}, Max: {np.max(hydros):.3f}")

    # Split data
    np.random.seed(42)
    indices = np.random.permutation(len(sequences))
    n_train = int(0.8 * len(sequences))
    train_seqs = [sequences[i] for i in indices[:n_train]]
    val_seqs = [sequences[i] for i in indices[n_train:]]

    # Create datasets with shared stats
    train_dataset = PeptideDataset(train_seqs, max_len=config.max_seq_len)
    stats = train_dataset.stats  # Use train stats for val
    val_dataset = PeptideDataset(val_seqs, max_len=config.max_seq_len, stats=stats)

    print("\nNormalization stats (from train):")
    print(f"  Length: mean={stats['length_mean']:.2f}, std={stats['length_std']:.2f}")
    print(f"  Hydro: mean={stats['hydro_mean']:.3f}, std={stats['hydro_std']:.3f}")
    print(f"  Charge: mean={stats['charge_mean']:.2f}, std={stats['charge_std']:.2f}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = PeptideVAE(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    n_epochs = 100
    best_val_loss = float("inf")

    print("\n" + "=" * 70)
    print("PHASE 1: Coverage + Physicochemical Grounding (Hydro-Weighted)")
    print("=" * 70)

    for epoch in range(n_epochs):
        train_losses = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, stats)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"\nEpoch {epoch + 1}/{n_epochs}")
            print(
                f"  Train: total={train_losses['total']:.4f}, recon={train_losses['recon']:.4f}, "
                f"hydro_loss={train_losses['hydro_loss']:.4f}"
            )
            print(f"  Val: total={val_metrics['total']:.4f}")

            if "hydro_r" in val_metrics:
                print(
                    f"  Physicochemical R: length={val_metrics['length_r']:.3f}, "
                    f"HYDRO={val_metrics['hydro_r']:.3f}, charge={val_metrics['charge_r']:.3f}"
                )

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "stats": stats,
                },
                PROJECT_ROOT / "checkpoints" / "peptide_vae_attempt_02.pt",
            )

    # Final evaluation
    print("\n" + "=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    recon_acc = compute_reconstruction_accuracy(model, val_loader, device)
    print(f"\nReconstruction Accuracy: {recon_acc:.1%}")

    final_metrics = evaluate(model, val_loader, device, stats)

    if "hydro_r" in final_metrics:
        print("\nPhysicochemical Encoding (C3 Validation):")
        print(f"  Length R:        {final_metrics['length_r']:.3f}")
        print(f"  HYDROPHOBICITY R: {final_metrics['hydro_r']:.3f}  <-- PRIMARY PREDICTOR")
        print(f"  Charge R:        {final_metrics['charge_r']:.3f}")

        mean_r = np.mean([final_metrics["length_r"], final_metrics["hydro_r"], final_metrics["charge_r"]])

        # Phase 1 success criteria
        if final_metrics["hydro_r"] >= 0.7 and mean_r >= 0.7:
            print(f"\n✓ PHASE 1 PASSED: Hydro R = {final_metrics['hydro_r']:.3f} ≥ 0.7")
        elif final_metrics["hydro_r"] >= 0.5:
            print(f"\n~ PHASE 1 PARTIAL: Hydro R = {final_metrics['hydro_r']:.3f} (acceptable)")
        else:
            print(f"\n✗ PHASE 1 NEEDS WORK: Hydro R = {final_metrics['hydro_r']:.3f} < 0.5")

    # Save results
    results = {
        "attempt": "prediction_attempt_02",
        "fixes": ["3x hydrophobicity weight", "z-score normalization", "dedicated hydro branch"],
        "timestamp": datetime.now().isoformat(),
        "results": {
            "reconstruction_accuracy": float(recon_acc),
            "physicochemical_r": {
                "length": float(final_metrics.get("length_r", 0)),
                "hydrophobicity": float(final_metrics.get("hydro_r", 0)),
                "charge": float(final_metrics.get("charge_r", 0)),
            },
        },
    }

    results_path = PROJECT_ROOT / "results" / "peptide_vae_attempt_02.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
