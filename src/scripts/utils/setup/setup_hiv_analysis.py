#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
"""
Setup Script for HIV Analysis Pipeline

This script prepares all prerequisites for running HIV bioinformatics analysis:
1. Verifies trained VAE checkpoints exist
2. Extracts V5.11 hyperbolic embeddings
3. Trains the codon encoder (codon_encoder_3adic.pt)

Run this once before using HIV analysis scripts.

Usage:
    python src/scripts/setup/setup_hiv_analysis.py
    python src/scripts/setup/setup_hiv_analysis.py --force  # Regenerate all
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


# Fix for numpy version compatibility in torch checkpoints
# Checkpoints saved with numpy 2.x use numpy._core, but numpy 1.x uses numpy.core
class NumpyBackwardsCompatUnpickler(pickle.Unpickler):
    """Custom unpickler to handle numpy._core -> numpy.core renaming."""

    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)


def load_checkpoint_compat(path, map_location="cpu"):
    """Load a checkpoint with numpy version compatibility."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e):
            # Fall back to custom unpickler
            with open(path, "rb") as f:
                unpickler = NumpyBackwardsCompatUnpickler(f)
                return unpickler.load()
        raise
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import spearmanr

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.geometry import poincare_distance
from src.models.ternary_vae import TernaryVAEV5_11_PartialFreeze


# =============================================================================
# CONFIGURATION
# =============================================================================

# Checkpoint paths (relative to PROJECT_ROOT)
CHECKPOINT_PATHS = {
    "v5_5": "results/checkpoints/v5_5/best.pt",
    "v5_11": "results/checkpoints/v5_11/best.pt",
    "v5_11_overnight": "results/checkpoints/v5_11_overnight/best.pt",
}

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "research" / "bioinformatics" / "genetic_code" / "data"
HIV_DATA_DIR = PROJECT_ROOT / "research" / "bioinformatics" / "codon_encoder_research" / "hiv" / "data"

# Genetic code
GENETIC_CODE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

CLUSTER_SIZES = [6, 6, 6, 4, 4, 4, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]


# =============================================================================
# POINCARE DISTANCE
# =============================================================================


def poincare_distance(x, y, c=1.0, eps=1e-10):
    """Compute Poincare ball distance."""
    norm_x_sq = torch.sum(x**2, dim=-1, keepdim=True)
    norm_y_sq = torch.sum(y**2, dim=-1, keepdim=True)
    diff_sq = torch.sum((x - y) ** 2, dim=-1, keepdim=True)

    denom = (1 - c * norm_x_sq) * (1 - c * norm_y_sq)
    denom = torch.clamp(denom, min=eps)

    arg = 1 + 2 * c * diff_sq / denom
    arg = torch.clamp(arg, min=1.0 + eps)

    dist = (1 / np.sqrt(c)) * torch.acosh(arg)
    return dist.squeeze(-1)


# =============================================================================
# STEP 1: VERIFY CHECKPOINTS
# =============================================================================


def verify_checkpoints():
    """Verify that required checkpoints exist."""
    print("\n[1/4] Verifying checkpoints...")

    available = {}
    for name, path in CHECKPOINT_PATHS.items():
        full_path = PROJECT_ROOT / path
        exists = full_path.exists()
        available[name] = exists
        status = "OK" if exists else "MISSING"
        print(f"  {name}: {status} ({full_path})")

    # We need v5_5 as base and one of the v5_11 variants
    if not available["v5_5"]:
        print("\nERROR: v5_5 checkpoint is required but missing!")
        print("Please train the base model first with:")
        print("  python src/scripts/train/train.py --epochs 100")
        return None

    # Find best available v5_11 checkpoint
    v5_11_path = None
    for variant in ["v5_11_overnight", "v5_11"]:
        if available.get(variant):
            v5_11_path = PROJECT_ROOT / CHECKPOINT_PATHS[variant]
            print(f"\n  Using {variant} for V5.11 embeddings")
            break

    if v5_11_path is None:
        print("\nWARNING: No V5.11 checkpoint found. Using v5_5 only.")
        print("For best results, train V5.11 with:")
        print("  python src/scripts/train/train.py --option_c --dual_projection")
        v5_11_path = PROJECT_ROOT / CHECKPOINT_PATHS["v5_5"]

    return {
        "v5_5": PROJECT_ROOT / CHECKPOINT_PATHS["v5_5"],
        "v5_11": v5_11_path,
    }


# =============================================================================
# STEP 2: EXTRACT EMBEDDINGS
# =============================================================================


def extract_embeddings(checkpoints, device="cpu", force=False):
    """Extract hyperbolic embeddings from trained model."""
    print("\n[2/4] Extracting embeddings...")

    embeddings_path = OUTPUT_DIR / "v5_11_3_embeddings.pt"

    if embeddings_path.exists() and not force:
        print(f"  Embeddings already exist: {embeddings_path}")
        print("  Use --force to regenerate")
        return load_checkpoint_compat(embeddings_path, map_location=device)

    # Load model - use default hidden_dim=64 to match checkpoint
    print("  Loading model...")
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=64,  # Must match checkpoint
        max_radius=0.95,
        curvature=1.0,
        use_dual_projection=True,
        n_projection_layers=1,  # Must match checkpoint
        projection_dropout=0.0,  # Must match checkpoint
        freeze_encoder_b=False,
        encoder_b_lr_scale=0.1,
    )

    # Load v5.5 base
    model.load_v5_5_checkpoint(checkpoints["v5_5"], device=device)

    # Load v5.11 weights if different from v5.5
    if checkpoints["v5_11"] != checkpoints["v5_5"]:
        checkpoint = load_checkpoint_compat(checkpoints["v5_11"], map_location=device)
        state = checkpoint.get("model_state", checkpoint)
        trainable_keys = [k for k in state.keys() if "projection" in k or "encoder_B" in k]
        trainable_state = {k: v for k, v in state.items() if k in trainable_keys}
        model.load_state_dict(trainable_state, strict=False)

    model.to(device)
    model.eval()

    # Generate all operations
    print("  Generating embeddings for 19,683 operations...")
    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, dtype=torch.float32, device=device)
    indices = torch.arange(len(operations), device=device)
    valuations = TERNARY.valuation(indices)

    with torch.no_grad():
        outputs = model(x, compute_control=False)

    z_A_hyp = outputs["z_A_hyp"].cpu()
    z_B_hyp = outputs["z_B_hyp"].cpu()
    z_A_euc = outputs["z_A_euc"].cpu()
    z_B_euc = outputs["z_B_euc"].cpu()

    # V5.12.2: Use hyperbolic distance for radii
    origin = torch.zeros_like(z_A_hyp)
    radii_A = poincare_distance(z_A_hyp, origin, c=1.0)
    radii_B = poincare_distance(z_B_hyp, origin, c=1.0)

    # Compute hierarchy correlation
    corr_B, _ = spearmanr(valuations.cpu().numpy(), radii_B.numpy())

    print(f"  Hierarchy correlation: {corr_B:.4f}")
    print(f"  Radii range: [{radii_B.min():.4f}, {radii_B.max():.4f}]")

    embeddings = {
        "z_A_hyp": z_A_hyp,
        "z_B_hyp": z_B_hyp,
        "z_A_euc": z_A_euc,
        "z_B_euc": z_B_euc,
        "valuations": valuations.cpu(),
        "radii_A": radii_A,
        "radii_B": radii_B,
        "operations": torch.tensor(operations),
        "indices": indices.cpu(),
        "metadata": {
            "model_version": "V5.11",
            "hierarchy_correlation": float(corr_B),
            "n_operations": 19683,
            "latent_dim": 16,
            "max_radius": 0.95,
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, embeddings_path)
    print(f"  Saved: {embeddings_path}")

    return embeddings


# =============================================================================
# STEP 3: FIND NATURAL POSITIONS
# =============================================================================


def find_natural_positions(embeddings, force=False):
    """Find natural positions for 21 amino acid clusters."""
    print("\n[3/4] Finding natural positions...")

    positions_path = OUTPUT_DIR / "natural_positions_v5_11_3.json"

    if positions_path.exists() and not force:
        print(f"  Positions already exist: {positions_path}")
        with open(positions_path) as f:
            return json.load(f)

    z_B_hyp = embeddings["z_B_hyp"]
    valuations = embeddings["valuations"].numpy()
    n_clusters = 21

    # Find positions matching genetic code degeneracy pattern
    positions = []
    labels = []

    for cluster_id, size in enumerate(CLUSTER_SIZES):
        # Find positions at appropriate valuation level
        # Higher degeneracy -> more common -> higher valuation
        target_val = 9 - (cluster_id * 9 // n_clusters)

        mask = valuations == target_val
        if mask.sum() < size:
            # Fall back to nearby valuations
            for delta in range(1, 10):
                mask = np.abs(valuations - target_val) <= delta
                if mask.sum() >= size:
                    break

        # Select positions with good spread
        candidates = np.where(mask)[0]
        if len(candidates) >= size:
            # Use k-means-like selection for spread
            selected = []
            z_candidates = z_B_hyp[candidates]

            # Start with centroid (Euclidean mean as initialization point)
            centroid = z_candidates.mean(dim=0)
            dists = poincare_distance(z_candidates, centroid.unsqueeze(0).expand_as(z_candidates))
            selected.append(candidates[dists.argmin().item()])

            # Add diverse points
            while len(selected) < size:
                remaining = [c for c in candidates if c not in selected]
                if not remaining:
                    break
                # Pick point furthest from selected (hyperbolic distance)
                min_dists = []
                for c in remaining:
                    d = min(poincare_distance(z_B_hyp[c], z_B_hyp[s]).item() for s in selected)
                    min_dists.append(d)
                best_idx = np.argmax(min_dists)
                selected.append(remaining[best_idx])

            positions.extend(selected)
            labels.extend([cluster_id] * len(selected))

    # Compute cluster statistics
    cluster_to_positions = defaultdict(list)
    for pos, label in zip(positions, labels):
        cluster_to_positions[label].append(pos)

    # Compute separation ratio
    within_dists = []
    between_dists = []

    for i, pos_i in enumerate(positions):
        for j, pos_j in enumerate(positions[i+1:], i+1):
            d = poincare_distance(z_B_hyp[pos_i], z_B_hyp[pos_j]).item()
            if labels[i] == labels[j]:
                within_dists.append(d)
            else:
                between_dists.append(d)

    mean_within = np.mean(within_dists) if within_dists else 1.0
    mean_between = np.mean(between_dists) if between_dists else 1.0
    separation_ratio = mean_between / mean_within

    result = {
        "metadata": {
            "source": "V5.11 embeddings",
            "n_positions": len(positions),
            "n_clusters": n_clusters,
            "separation_ratio": float(separation_ratio),
            "timestamp": datetime.now().isoformat(),
        },
        "positions": [int(p) for p in positions],  # Convert to native Python int
        "labels": [int(l) for l in labels],
        "clusters": {str(k): [int(p) for p in v] for k, v in cluster_to_positions.items()},
        "cluster_sizes": CLUSTER_SIZES,
        "degeneracy_pattern": CLUSTER_SIZES,
    }

    with open(positions_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Found {len(positions)} positions in {n_clusters} clusters")
    print(f"  Separation ratio: {separation_ratio:.2f}x")
    print(f"  Saved: {positions_path}")

    return result


# =============================================================================
# STEP 4: TRAIN CODON ENCODER
# =============================================================================


class CodonEncoder3Adic(nn.Module):
    """Encode codons into hyperbolic embedding space."""

    def __init__(self, input_dim=12, hidden_dim=32, embed_dim=16, n_clusters=21):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_clusters = n_clusters

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.cluster_head = nn.Linear(embed_dim, n_clusters)
        self.cluster_centers = nn.Parameter(torch.randn(n_clusters, embed_dim) * 0.1)

    def forward(self, x):
        embedding = self.encoder(x)
        cluster_logits = self.cluster_head(embedding)
        return embedding, cluster_logits

    def encode(self, x):
        return self.encoder(x)


def train_codon_encoder(embeddings, positions_data, force=False):
    """Train the codon encoder."""
    print("\n[4/4] Training codon encoder...")

    encoder_path = OUTPUT_DIR / "codon_encoder_3adic.pt"

    if encoder_path.exists() and not force:
        print(f"  Encoder already exists: {encoder_path}")
        print("  Use --force to retrain")
        return True

    # Prepare codon data
    codons = list(GENETIC_CODE.keys())
    amino_acids = [GENETIC_CODE[c] for c in codons]

    aa_counts = defaultdict(int)
    for aa in amino_acids:
        aa_counts[aa] += 1

    aa_sorted = sorted(aa_counts.keys(), key=lambda x: (-aa_counts[x], x))
    aa_to_cluster = {aa: i for i, aa in enumerate(aa_sorted)}

    codon_clusters = [aa_to_cluster[GENETIC_CODE[c]] for c in codons]

    # One-hot encoding
    nuc_to_idx = {"A": 0, "C": 1, "G": 2, "T": 3}
    codon_features = []
    for codon in codons:
        features = []
        for nuc in codon:
            one_hot = [0, 0, 0, 0]
            one_hot[nuc_to_idx[nuc]] = 1
            features.extend(one_hot)
        codon_features.append(features)

    codon_features = torch.tensor(codon_features, dtype=torch.float32)
    codon_clusters = torch.tensor(codon_clusters, dtype=torch.long)

    # Create pairs
    positive_pairs = []
    negative_pairs = []
    for i in range(64):
        for j in range(i + 1, 64):
            if amino_acids[i] == amino_acids[j]:
                positive_pairs.append((i, j))
            else:
                negative_pairs.append((i, j))

    # Create model
    model = CodonEncoder3Adic(input_dim=12, hidden_dim=32, embed_dim=16, n_clusters=21)

    # Initialize cluster centers
    z_B_hyp = embeddings["z_B_hyp"]
    cluster_to_positions = {int(k): v for k, v in positions_data["clusters"].items()}

    with torch.no_grad():
        for cluster_id in range(21):
            pos_list = cluster_to_positions.get(cluster_id, [])
            if pos_list:
                center = z_B_hyp[pos_list].mean(dim=0)
                model.cluster_centers[cluster_id] = center

    # Train with gradient clipping and lower LR
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)

    print("  Training for 300 epochs...")
    best_acc = 0.0
    best_state = None

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()

        embeddings_out, cluster_logits = model(codon_features)

        # Classification loss
        loss_cluster = F.cross_entropy(cluster_logits, codon_clusters)

        # Simplified contrastive loss with clamping for stability
        loss_contrastive = torch.tensor(0.0)
        for i, j in positive_pairs:
            d = poincare_distance(embeddings_out[i:i+1], embeddings_out[j:j+1])
            d = torch.clamp(d, 0, 10)  # Clamp for stability
            loss_contrastive = loss_contrastive + d.pow(2)

        n_neg = min(len(negative_pairs), len(positive_pairs) * 2)
        for i, j in negative_pairs[:n_neg]:
            d = poincare_distance(embeddings_out[i:i+1], embeddings_out[j:j+1])
            d = torch.clamp(d, 0, 10)  # Clamp for stability
            loss_contrastive = loss_contrastive + F.relu(2.0 - d).pow(2)

        loss_contrastive = loss_contrastive / (len(positive_pairs) + n_neg)

        # Center alignment
        loss_center = torch.tensor(0.0)
        for i, cluster_id in enumerate(codon_clusters):
            center = model.cluster_centers[cluster_id]
            d = poincare_distance(embeddings_out[i:i+1], center.unsqueeze(0))
            loss_center = loss_center + d.pow(2)
        loss_center = loss_center / len(codon_clusters)

        # Total loss with NaN check
        loss = loss_cluster + 0.5 * loss_contrastive + 0.3 * loss_center

        if torch.isnan(loss):
            print(f"    Epoch {epoch}: NaN detected, stopping early")
            if best_state is not None:
                model.load_state_dict(best_state)
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Track best model
        with torch.no_grad():
            preds = cluster_logits.argmax(dim=1)
            acc = (preds == codon_clusters).float().mean().item()

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == 299:
            print(f"    Epoch {epoch:3d}: loss={loss.item():.4f}, acc={acc*100:.1f}%")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"  Restored best model with accuracy: {best_acc*100:.1f}%")

    # Evaluate
    model.eval()
    with torch.no_grad():
        embeddings_out, cluster_logits = model(codon_features)
        preds = cluster_logits.argmax(dim=1)
        cluster_acc = (preds == codon_clusters).float().mean().item()

        synonymous_correct = sum(1 for i, j in positive_pairs if preds[i] == preds[j])
        synonymous_acc = synonymous_correct / len(positive_pairs)

    # Create codon to position mapping
    codon_to_position = {}
    for i, codon in enumerate(codons):
        cluster_id = preds[i].item()
        cluster_positions = cluster_to_positions.get(cluster_id, [])
        if cluster_positions:
            # Find nearest position
            codon_emb = embeddings_out[i]
            min_dist = float("inf")
            best_pos = cluster_positions[0]
            for pos in cluster_positions:
                pos_emb = z_B_hyp[pos]
                d = torch.norm(codon_emb - pos_emb).item()
                if d < min_dist:
                    min_dist = d
                    best_pos = pos
            codon_to_position[codon] = best_pos

    # Save
    output = {
        "model_state": model.state_dict(),
        "codon_to_position": codon_to_position,
        "aa_to_cluster": aa_to_cluster,
        "aa_sorted": aa_sorted,
        "metadata": {
            "version": "3-adic",
            "source_embeddings": "V5.11",
            "cluster_accuracy": cluster_acc,
            "synonymous_accuracy": synonymous_acc,
            "timestamp": datetime.now().isoformat(),
        },
    }

    torch.save(output, encoder_path)
    print(f"\n  Cluster accuracy: {cluster_acc*100:.1f}%")
    print(f"  Synonymous accuracy: {synonymous_acc*100:.1f}%")
    print(f"  Saved: {encoder_path}")

    # Also save to HIV data directory
    HIV_DATA_DIR.mkdir(parents=True, exist_ok=True)
    hiv_encoder_path = HIV_DATA_DIR / "codon_encoder_3adic.pt"
    torch.save(output, hiv_encoder_path)
    print(f"  Also saved to: {hiv_encoder_path}")

    return True


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Setup HIV Analysis Pipeline")
    parser.add_argument("--force", action="store_true", help="Force regeneration of all files")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()

    print("=" * 70)
    print("HIV ANALYSIS SETUP")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Force regeneration: {args.force}")

    # Step 1: Verify checkpoints
    checkpoints = verify_checkpoints()
    if checkpoints is None:
        return 1

    # Step 2: Extract embeddings
    embeddings = extract_embeddings(checkpoints, args.device, args.force)

    # Step 3: Find natural positions
    positions_data = find_natural_positions(embeddings, args.force)

    # Step 4: Train codon encoder
    success = train_codon_encoder(embeddings, positions_data, args.force)

    if success:
        print("\n" + "=" * 70)
        print("SETUP COMPLETE")
        print("=" * 70)
        print("\nYou can now run HIV analysis scripts:")
        print("  python src/scripts/run_hiv_analysis.py")
        print("  python research/bioinformatics/codon_encoder_research/hiv/scripts/01_hiv_escape_analysis.py")
        print("=" * 70)
        return 0
    else:
        print("\nSetup failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
