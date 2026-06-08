# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""V5.5 Manifold Quality Analysis.

Analyzes the v5.5 checkpoint to understand:
1. Reconstruction accuracy across all operations
2. 3-adic ranking correlation
3. Radial distribution vs 3-adic valuation
4. Per-region quality metrics

Usage:
    python src/scripts/visualization/analyze_v5_5_quality.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR
from src.data.generation import generate_all_ternary_operations
from src.geometry import poincare_distance


def load_v5_5_checkpoint(device: str = "cpu"):
    """Load v5.5 checkpoint."""
    checkpoint_path = CHECKPOINTS_DIR / "v5_5" / "latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint


def build_encoder_decoder(checkpoint, device):
    """Build encoder/decoder from checkpoint state dict."""
    import torch.nn as nn

    class SimpleEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(9, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.fc_mu = nn.Linear(64, 16)
            self.fc_logvar = nn.Linear(64, 16)

        def forward(self, x):
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

    class SimpleDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 27),
            )

        def forward(self, z):
            logits = self.decoder(z)
            return logits.view(-1, 9, 3)

    # Create models
    encoder_A = SimpleEncoder().to(device)
    encoder_B = SimpleEncoder().to(device)
    decoder_A = SimpleDecoder().to(device)

    # Load state
    model_state = checkpoint["model"]

    # Filter and load encoder_A
    enc_A_state = {k.replace("encoder_A.", ""): v for k, v in model_state.items() if k.startswith("encoder_A.")}
    encoder_A.load_state_dict(enc_A_state)

    # Filter and load encoder_B
    enc_B_state = {k.replace("encoder_B.", ""): v for k, v in model_state.items() if k.startswith("encoder_B.")}
    encoder_B.load_state_dict(enc_B_state)

    # Filter and load decoder_A
    dec_A_state = {k.replace("decoder_A.", ""): v for k, v in model_state.items() if k.startswith("decoder_A.")}
    decoder_A.load_state_dict(dec_A_state)

    return encoder_A, encoder_B, decoder_A


def compute_3adic_valuation(n: int) -> int:
    """Compute v_3(n) - how many times 3 divides n."""
    if n == 0:
        return 9  # Maximum valuation for 0
    v = 0
    while n % 3 == 0:
        n //= 3
        v += 1
    return v


def analyze_manifold_quality(encoder_A, encoder_B, decoder_A, device="cpu"):
    """Comprehensive quality analysis of the v5.5 manifold."""

    print("=" * 70)
    print("V5.5 MANIFOLD QUALITY ANALYSIS")
    print("=" * 70)

    # Generate all operations
    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, dtype=torch.float32, device=device)
    n_ops = len(operations)

    print(f"\nTotal operations: {n_ops}")

    # Encode all operations
    encoder_A.eval()
    encoder_B.eval()
    decoder_A.eval()

    with torch.no_grad():
        mu_A, logvar_A = encoder_A(x)
        mu_B, logvar_B = encoder_B(x)

        # Use means for analysis (deterministic)
        z_A = mu_A
        z_B = mu_B

        # Decode
        logits_A = decoder_A(z_A)

    # 1. RECONSTRUCTION ACCURACY
    print("\n" + "-" * 70)
    print("1. RECONSTRUCTION ACCURACY")
    print("-" * 70)

    # Convert logits to predictions
    preds = torch.argmax(logits_A, dim=-1) - 1  # Map {0,1,2} -> {-1,0,1}
    (x + 1).long()  # Map {-1,0,1} -> {0,1,2}

    # Per-position accuracy
    correct = (preds == x).float()
    per_position_acc = correct.mean(dim=0)

    print("\nPer-position accuracy:")
    for i, acc in enumerate(per_position_acc):
        print(f"  Position {i}: {acc.item()*100:.2f}%")

    # Overall accuracy
    sample_acc = correct.mean(dim=1)
    perfect_recon = (sample_acc == 1.0).sum().item()
    overall_acc = correct.mean().item()

    print(f"\nOverall element accuracy: {overall_acc*100:.2f}%")
    print(f"Perfect reconstructions: {perfect_recon}/{n_ops} ({perfect_recon/n_ops*100:.2f}%)")
    print(f"Coverage (>0 correct): {(sample_acc > 0).sum().item()}/{n_ops} ({(sample_acc > 0).sum().item()/n_ops*100:.2f}%)")

    # 2. RADIAL DISTRIBUTION
    print("\n" + "-" * 70)
    print("2. RADIAL DISTRIBUTION (Hyperbolic Structure)")
    print("-" * 70)

    # V5.12.2: Use hyperbolic distance for radii
    origin = torch.zeros_like(z_A)
    radii_A = poincare_distance(z_A, origin, c=1.0).cpu().numpy()
    radii_B = poincare_distance(z_B, origin, c=1.0).cpu().numpy()

    print(f"\nVAE-A radii: mean={radii_A.mean():.4f}, std={radii_A.std():.4f}, min={radii_A.min():.4f}, max={radii_A.max():.4f}")
    print(f"VAE-B radii: mean={radii_B.mean():.4f}, std={radii_B.std():.4f}, min={radii_B.min():.4f}, max={radii_B.max():.4f}")

    # Analyze radius vs 3-adic valuation
    valuations = np.array([compute_3adic_valuation(i) for i in range(n_ops)])

    corr_A_rad, p_A_rad = spearmanr(valuations, radii_A)
    corr_B_rad, p_B_rad = spearmanr(valuations, radii_B)

    print("\nRadius vs 3-adic valuation correlation (expect NEGATIVE for proper hierarchy):")
    print(f"  VAE-A: r={corr_A_rad:.4f} (p={p_A_rad:.2e})")
    print(f"  VAE-B: r={corr_B_rad:.4f} (p={p_B_rad:.2e})")

    # Mean radius by valuation level
    print("\nMean radius by 3-adic valuation (higher v = more divisible by 3):")
    print("  Valuation | Count | VAE-A radius | VAE-B radius")
    print("  " + "-" * 50)
    for v in range(10):
        mask = valuations == v
        count = mask.sum()
        if count > 0:
            mean_A = radii_A[mask].mean()
            mean_B = radii_B[mask].mean()
            print(f"  v={v:6d} | {count:5d} | {mean_A:11.4f} | {mean_B:11.4f}")

    # 3. 3-ADIC RANKING CORRELATION
    print("\n" + "-" * 70)
    print("3. 3-ADIC RANKING CORRELATION")
    print("-" * 70)

    # Sample pairs and compute ranking correlation
    np.random.seed(42)
    n_samples = 50000
    pairs = np.random.choice(n_ops, size=(n_samples, 2), replace=True)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    z_A_np = z_A.cpu().numpy()
    z_B_np = z_B.cpu().numpy()

    # Compute 3-adic distances
    def compute_3adic_distance(idx1, idx2):
        """Count differing 3-adic digits."""
        diff = 0
        i1, i2 = idx1, idx2
        for _ in range(9):
            if (i1 % 3) != (i2 % 3):
                diff += 1
            i1 //= 3
            i2 //= 3
        return diff

    adic_dists = np.array([compute_3adic_distance(i, j) for i, j in pairs])

    # Compute latent distances using proper hyperbolic (Poincaré) distance
    with torch.no_grad():
        latent_dists_A = poincare_distance(z_A[pairs[:, 0]], z_A[pairs[:, 1]]).cpu().numpy()
        latent_dists_B = poincare_distance(z_B[pairs[:, 0]], z_B[pairs[:, 1]]).cpu().numpy()

    corr_A, p_A = spearmanr(adic_dists, latent_dists_A)
    corr_B, p_B = spearmanr(adic_dists, latent_dists_B)

    print("\n3-adic digit distance vs latent distance correlation:")
    print(f"  VAE-A: r={corr_A:.4f} (p={p_A:.2e})")
    print(f"  VAE-B: r={corr_B:.4f} (p={p_B:.2e})")

    # 4. QUALITY BY REGION (valuation-based)
    print("\n" + "-" * 70)
    print("4. QUALITY BY REGION (3-adic valuation)")
    print("-" * 70)

    print("\nReconstruction accuracy by 3-adic valuation:")
    print("  Valuation | Count | Accuracy | Perfect")
    print("  " + "-" * 45)

    for v in range(10):
        mask = valuations == v
        count = mask.sum()
        if count > 0:
            region_correct = correct[mask].mean().item()
            region_perfect = (sample_acc[mask] == 1.0).sum().item()
            print(f"  v={v:6d} | {count:5d} | {region_correct*100:7.2f}% | {region_perfect:5d}")

    # 5. SPECIAL OPERATIONS ANALYSIS
    print("\n" + "-" * 70)
    print("5. SPECIAL OPERATIONS ANALYSIS")
    print("-" * 70)

    # Find special operations
    special_ops = {}

    # Zero operation (all zeros)
    zero_op = np.zeros(9)
    for idx, op in enumerate(operations):
        if np.array_equal(op, zero_op):
            special_ops["zero"] = idx

    # Constant operations
    const_ops = [np.full(9, -1), np.full(9, 0), np.full(9, 1)]
    const_names = ["const_-1", "const_0", "const_1"]
    for name, const_op in zip(const_names, const_ops):
        for idx, op in enumerate(operations):
            if np.array_equal(op, const_op):
                special_ops[name] = idx

    # Identity-like
    proj1 = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])  # op(a,b) = a
    proj2 = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])  # op(a,b) = b
    for idx, op in enumerate(operations):
        if np.array_equal(op, proj1):
            special_ops["proj_a"] = idx
        if np.array_equal(op, proj2):
            special_ops["proj_b"] = idx

    print("\nSpecial operations:")
    for name, idx in special_ops.items():
        acc = sample_acc[idx].item()
        radius_A = radii_A[idx]
        radius_B = radii_B[idx]
        v = compute_3adic_valuation(idx)
        print(f"  {name:10s}: idx={idx:5d}, v_3={v}, acc={acc*100:.0f}%, r_A={radius_A:.3f}, r_B={radius_B:.3f}")

    # 6. SUMMARY METRICS
    print("\n" + "=" * 70)
    print("QUALITY SUMMARY")
    print("=" * 70)

    print(
        f"""
COVERAGE:
  - Perfect reconstructions: {perfect_recon/n_ops*100:.1f}%
  - Overall element accuracy: {overall_acc*100:.1f}%

3-ADIC STRUCTURE:
  - Distance correlation: VAE-A r={corr_A:.3f}, VAE-B r={corr_B:.3f}
  - Radial hierarchy:     VAE-A r={corr_A_rad:.3f}, VAE-B r={corr_B_rad:.3f}

RADIAL DISTRIBUTION:
  - VAE-A: {radii_A.mean():.2f} +/- {radii_A.std():.2f} (range: {radii_A.min():.2f} - {radii_A.max():.2f})
  - VAE-B: {radii_B.mean():.2f} +/- {radii_B.std():.2f} (range: {radii_B.min():.2f} - {radii_B.max():.2f})
"""
    )

    return {
        "coverage": perfect_recon / n_ops,
        "accuracy": overall_acc,
        "ranking_corr_A": corr_A,
        "ranking_corr_B": corr_B,
        "radial_corr_A": corr_A_rad,
        "radial_corr_B": corr_B_rad,
        "radii_A": radii_A,
        "radii_B": radii_B,
        "z_A": z_A_np,
        "z_B": z_B_np,
        "valuations": valuations,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load checkpoint
    checkpoint = load_v5_5_checkpoint(device)
    print("\nCheckpoint info:")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Keys: {list(checkpoint.keys())}")

    # Build models from checkpoint
    encoder_A, encoder_B, decoder_A = build_encoder_decoder(checkpoint, device)

    # Run analysis
    results = analyze_manifold_quality(encoder_A, encoder_B, decoder_A, device)

    return results


if __name__ == "__main__":
    main()
