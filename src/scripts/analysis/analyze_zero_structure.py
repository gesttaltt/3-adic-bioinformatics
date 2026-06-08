# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Zero-trit structure analysis for Ternary VAE.

Analyzes whether the model exploits zeros structurally or treats them as random categorical values.

Three diagnostic questions:
1. Zero-frequency by trit position - do zeros cluster at certain valuation levels?
2. Correlation between zero-count and latent properties (radius, structure)
3. P-adic valuation (trailing zeros) relationship to radial position
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import json

import numpy as np
import torch
from scipy import stats

from src.config.paths import CHECKPOINTS_DIR
from src.data.generation import generate_all_ternary_operations
from src.geometry import poincare_distance
from src.models import TernaryVAE_PartialFreeze


def compute_padic_valuation(op: np.ndarray) -> int:
    """Compute 3-adic valuation (number of trailing zeros in base-3 representation).

    For a ternary operation, we count trailing zeros from position 0.
    """
    valuation = 0
    for val in op:
        if val == 0:
            valuation += 1
        else:
            break
    return valuation


def compute_zero_count(op: np.ndarray) -> int:
    """Count total zeros in operation."""
    return np.sum(op == 0)


def compute_zero_pattern(op: np.ndarray) -> str:
    """Get binary pattern of zeros (1=zero, 0=non-zero)."""
    return "".join(["1" if v == 0 else "0" for v in op])


def analyze_checkpoint(checkpoint_path: str, device: str = "cuda"):
    """Run full zero-structure analysis on a checkpoint."""

    print(f"\n{'=' * 60}")
    print("ZERO-TRIT STRUCTURE ANALYSIS")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"{'=' * 60}\n")

    # Load model
    model = TernaryVAE_PartialFreeze(
        latent_dim=16,
        hidden_dim=64,
        use_dual_projection=True,
        use_controller=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    state_dict = checkpoint.get("model", checkpoint)

    # Try to load, handling missing keys gracefully
    try:
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Warning loading state dict: {e}")

    model.to(device)
    model.eval()

    # Generate all operations
    all_ops = generate_all_ternary_operations()
    all_ops_tensor = torch.tensor(all_ops, dtype=torch.float32, device=device)

    # Encode all operations
    print("Encoding all 19,683 ternary operations...")
    with torch.no_grad():
        outputs = model(all_ops_tensor)
        z_A_hyp = outputs["z_A_hyp"].cpu()
        z_B_hyp = outputs["z_B_hyp"].cpu()

    # V5.12.2: Compute latent radii using hyperbolic distance
    origin_A = torch.zeros_like(z_A_hyp)
    origin_B = torch.zeros_like(z_B_hyp)
    radius_A = poincare_distance(z_A_hyp, origin_A, c=1.0).numpy()
    radius_B = poincare_distance(z_B_hyp, origin_B, c=1.0).numpy()
    z_A_hyp = z_A_hyp.numpy()
    z_B_hyp = z_B_hyp.numpy()

    # Compute input properties for each operation
    zero_counts = np.array([compute_zero_count(op) for op in all_ops])
    valuations = np.array([compute_padic_valuation(op) for op in all_ops])

    # Position-wise zero frequency
    zero_by_position = np.mean(all_ops == 0, axis=0)

    print("\n" + "=" * 60)
    print("1. ZERO FREQUENCY BY POSITION")
    print("=" * 60)
    print("\n(In a uniform random distribution, each position would have ~33.3% zeros)")
    print("\nPosition | Zero Freq | Deviation from Expected")
    print("-" * 50)
    for i, freq in enumerate(zero_by_position):
        deviation = (freq - 1 / 3) * 100
        print(f"   {i}     |   {freq:.4f}  |  {deviation:+.2f}%")

    print(f"\nUniformity test: {'UNIFORM' if np.std(zero_by_position) < 0.01 else 'NON-UNIFORM'}")
    print(f"  Std of position frequencies: {np.std(zero_by_position):.6f}")

    # Zero count distribution
    print("\n" + "=" * 60)
    print("2. ZERO COUNT VS LATENT RADIUS")
    print("=" * 60)

    # Group by zero count
    unique_counts = np.unique(zero_counts)
    print("\nZero Count | Mean Radius A | Mean Radius B | Count")
    print("-" * 55)
    for count in unique_counts:
        mask = zero_counts == count
        mean_r_A = radius_A[mask].mean()
        mean_r_B = radius_B[mask].mean()
        n = mask.sum()
        print(f"    {count}      |    {mean_r_A:.4f}    |    {mean_r_B:.4f}    | {n:5d}")

    # Correlation
    corr_A, p_A = stats.pearsonr(zero_counts, radius_A)
    corr_B, p_B = stats.pearsonr(zero_counts, radius_B)
    print(f"\nCorrelation (zero_count vs radius_A): r={corr_A:.4f}, p={p_A:.2e}")
    print(f"Correlation (zero_count vs radius_B): r={corr_B:.4f}, p={p_B:.2e}")

    if abs(corr_A) > 0.1 or abs(corr_B) > 0.1:
        print("\n>>> FINDING: Model shows radius-zero correlation!")
        print("    Zero-heavy operations cluster at different radii than non-zero operations.")
    else:
        print("\n>>> FINDING: Model treats zeros uniformly (no radius-zero correlation)")

    # P-adic valuation analysis
    print("\n" + "=" * 60)
    print("3. P-ADIC VALUATION (TRAILING ZEROS) VS RADIUS")
    print("=" * 60)

    unique_vals = np.unique(valuations)
    print("\nValuation | Mean Radius A | Mean Radius B | Count")
    print("-" * 55)
    for val in unique_vals:
        mask = valuations == val
        mean_r_A = radius_A[mask].mean()
        mean_r_B = radius_B[mask].mean()
        n = mask.sum()
        print(f"    {val}      |    {mean_r_A:.4f}    |    {mean_r_B:.4f}    | {n:5d}")

    # Correlation
    corr_val_A, p_val_A = stats.pearsonr(valuations, radius_A)
    corr_val_B, p_val_B = stats.pearsonr(valuations, radius_B)
    print(f"\nCorrelation (valuation vs radius_A): r={corr_val_A:.4f}, p={p_val_A:.2e}")
    print(f"Correlation (valuation vs radius_B): r={corr_val_B:.4f}, p={p_val_B:.2e}")

    # Expected p-adic behavior: higher valuation = closer to origin (smaller radius)
    if corr_val_A < -0.1 or corr_val_B < -0.1:
        print("\n>>> FINDING: Model shows p-adic hierarchy structure!")
        print("    Higher valuation (more trailing zeros) = smaller radius")
        print("    This is the expected p-adic distance relationship: d(a,0) = 3^{-v(a)}")
    elif corr_val_A > 0.1 or corr_val_B > 0.1:
        print("\n>>> FINDING: Model shows INVERTED p-adic structure")
        print("    Higher valuation = LARGER radius (opposite of expected)")
    else:
        print("\n>>> FINDING: No p-adic valuation-radius relationship detected")

    # Reconstruction quality by zero count
    print("\n" + "=" * 60)
    print("4. RECONSTRUCTION ANALYSIS BY ZERO COUNT")
    print("=" * 60)

    with torch.no_grad():
        x_recon = outputs.get("x_recon_A")
        if x_recon is not None:
            x_recon = x_recon.cpu().numpy()

            # Discretize reconstruction
            x_recon_discrete = np.sign(x_recon)
            x_recon_discrete[np.abs(x_recon) < 0.33] = 0

            # Compute accuracy by zero count
            print("\nZero Count | Accuracy | Zero-Position Accuracy | Non-Zero Accuracy")
            print("-" * 70)
            for count in unique_counts:
                mask = zero_counts == count
                ops_subset = all_ops[mask]
                recon_subset = x_recon_discrete[mask]

                # Overall accuracy
                acc = np.mean(ops_subset == recon_subset)

                # Zero-position accuracy (how well are zeros reconstructed as zeros?)
                zero_mask = ops_subset == 0
                zero_acc = np.mean(recon_subset[zero_mask] == 0) if zero_mask.sum() > 0 else float("nan")

                # Non-zero accuracy
                nonzero_mask = ops_subset != 0
                if nonzero_mask.sum() > 0:
                    nonzero_acc = np.mean(recon_subset[nonzero_mask] == ops_subset[nonzero_mask])
                else:
                    nonzero_acc = float("nan")

                print(f"    {count}      |  {acc:.4f}  |        {zero_acc:.4f}        |      {nonzero_acc:.4f}")
        else:
            print("No reconstruction output available in model")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    results = {
        "checkpoint": str(checkpoint_path),
        "zero_count_radius_corr_A": float(corr_A),
        "zero_count_radius_corr_B": float(corr_B),
        "valuation_radius_corr_A": float(corr_val_A),
        "valuation_radius_corr_B": float(corr_val_B),
        "zero_structure_detected": bool(abs(corr_A) > 0.1 or abs(corr_B) > 0.1),
        "padic_hierarchy_detected": bool(corr_val_A < -0.1 or corr_val_B < -0.1),
        "inverted_padic": bool(corr_val_A > 0.1 or corr_val_B > 0.1),
    }

    if results["padic_hierarchy_detected"]:
        print("\n[+] Model HAS learned p-adic valuation structure")
        print("    Zeros are being exploited for hierarchical encoding")
    elif results["inverted_padic"]:
        print("\n[~] Model shows inverted p-adic structure")
        print("    May benefit from explicit valuation-weighted loss")
    else:
        print("\n[-] Model treats zeros as categorical, not structural")
        print("    Opportunity: Add sparsity incentive or valuation-weighted loss")

    if results["zero_structure_detected"]:
        print("\n[+] Zero-count affects latent radius")
        print("    Model is implicitly using zero-density for encoding")
    else:
        print("\n[-] Zero-count is independent of latent structure")
        print("    Zeros are not being exploited for efficiency")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze zero-trit structure in Ternary VAE")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "v5_11" / "best.pt"),
        help="Path to checkpoint",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    results = analyze_checkpoint(args.checkpoint, args.device)

    # Save results
    output_path = Path(args.checkpoint).parent / "zero_structure_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
