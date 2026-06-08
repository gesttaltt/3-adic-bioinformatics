# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Compare frozen encoder vs unfrozen encoder configurations.

This script compares three configurations:
1. FROZEN encoder (v5_11_progressive): 100% coverage, p-adic generalizable
2. UNFROZEN encoder: Deeper hierarchical resolution but <100% coverage
3. The trade-off between coverage and hierarchy

Key insight from user conjecture:
- Frozen encoder achieves 100% coverage → p-adic generalization possible
- Unfrozen encoder achieves deeper hierarchical resolution → better semantics
- These are fundamentally different optimization targets

This comparison helps understand the trade-off and potentially find
interpolation strategies between the two.

Usage:
    python src/scripts/epsilon_vae/compare_frozen_vs_unfrozen.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR, OUTPUT_DIR
from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.models.ternary_vae import TernaryVAEV5_11


def load_checkpoint(checkpoint_path: Path, device: str = "cpu"):
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_state = ckpt.get("model_state_dict") or ckpt.get("model_state") or {}

    latent_dim = 16
    for key, value in model_state.items():
        if "encoder_A.fc_mu.weight" in key:
            latent_dim = value.shape[0]
            break

    model = TernaryVAEV5_11(
        latent_dim=latent_dim,
        hidden_dim=64,
        use_dual_projection=True,
        use_controller=False,
    )

    model.load_state_dict(model_state, strict=False)
    model.to(device)

    metrics = ckpt.get("metrics", {})

    return model, metrics


def compute_comprehensive_metrics(model, all_ops: torch.Tensor, indices: torch.Tensor, device: str) -> dict:
    """Compute comprehensive metrics for a model configuration.

    Args:
        model: TernaryVAE model
        all_ops: All 19683 ternary operations
        indices: Operation indices
        device: Device

    Returns:
        Dict with all metrics
    """
    model.eval()
    all_ops = all_ops.to(device)
    indices = indices.to(device)

    with torch.no_grad():
        outputs = model(all_ops, compute_control=False)

        z_A_hyp = outputs["z_A_hyp"]
        z_B_hyp = outputs["z_B_hyp"]
        z_A_euc = outputs["z_A_euc"]
        z_B_euc = outputs["z_B_euc"]
        mu_A = outputs["mu_A"]

        # Coverage (using mean, not sampled)
        logits_A = model.decoder_A(mu_A)
        preds = torch.argmax(logits_A, dim=-1) - 1
        targets = all_ops.long()
        correct = (preds == targets).float().mean(dim=1)
        coverage = (correct == 1.0).sum().item() / len(all_ops)

        # Radii
        radii_A = z_A_hyp.norm(dim=-1).cpu().numpy()
        radii_B = z_B_hyp.norm(dim=-1).cpu().numpy()

        # Valuations
        valuations = TERNARY.valuation(indices).cpu().numpy()

        # Radial hierarchy correlation (should be NEGATIVE for proper hierarchy)
        radial_corr_A = spearmanr(valuations, radii_A)[0]
        radial_corr_B = spearmanr(valuations, radii_B)[0]

        # Radius statistics by valuation level
        radius_by_valuation = {}
        for v in range(10):
            mask = valuations == v
            if mask.any():
                radius_by_valuation[v] = {
                    "mean_A": float(radii_A[mask].mean()),
                    "std_A": float(radii_A[mask].std()),
                    "mean_B": float(radii_B[mask].mean()),
                    "std_B": float(radii_B[mask].std()),
                    "count": int(mask.sum()),
                }

        # Hierarchical separation: ratio of high-v to low-v radius
        # Perfect hierarchy: v=0 at outer edge (high radius), v=9 at center (low radius)
        r_v0 = radii_A[valuations == 0].mean() if (valuations == 0).any() else 0
        r_v9 = radii_A[valuations == 9].mean() if (valuations == 9).any() else 0
        hierarchy_separation = r_v0 / (r_v9 + 1e-6)

        # Latent space utilization
        latent_std_A = z_A_euc.std(dim=0).cpu().numpy()
        z_B_euc.std(dim=0).cpu().numpy()

        # Effective dimensionality (PCA)
        z_A_centered = z_A_euc - z_A_euc.mean(dim=0)
        cov_A = (z_A_centered.T @ z_A_centered) / len(z_A_euc)
        eigenvalues_A = torch.linalg.eigvalsh(cov_A).cpu().numpy()
        eigenvalues_A = eigenvalues_A[::-1]  # Descending order
        explained_var_A = eigenvalues_A / eigenvalues_A.sum()
        effective_dim_A = np.sum(np.cumsum(explained_var_A) < 0.99) + 1

        # Check for p-adic generalization
        # Use interpolated operations to test
        base_ops = all_ops.cpu().numpy()
        rng = np.random.RandomState(42)

        # Test p=3.5
        n_additional = int(3.5**9) - len(base_ops)
        idx1 = rng.randint(0, len(base_ops), size=min(n_additional, 50000))
        idx2 = rng.randint(0, len(base_ops), size=min(n_additional, 50000))
        alpha = rng.beta(2, 2, size=(len(idx1), 1))
        interpolated = (1 - alpha) * base_ops[idx1] + alpha * base_ops[idx2]
        extended = np.vstack([base_ops, interpolated]).astype(np.float32)
        extended_tensor = torch.tensor(extended, device=device)

        outputs_ext = model(extended_tensor, compute_control=False)
        z_ext = outputs_ext["z_A_hyp"]
        norms_ext = z_ext.norm(dim=-1)
        padic_coverage = ((norms_ext > 0.1) & (norms_ext < 0.99)).float().mean().item()

    return {
        "coverage": coverage,
        "radial_corr_A": radial_corr_A,
        "radial_corr_B": radial_corr_B,
        "hierarchy_separation": hierarchy_separation,
        "radius_v0": r_v0,
        "radius_v9": r_v9,
        "radius_range_A": radii_A.max() - radii_A.min(),
        "mean_radius_A": radii_A.mean(),
        "std_radius_A": radii_A.std(),
        "effective_dim_A": effective_dim_A,
        "latent_std_mean_A": latent_std_A.mean(),
        "padic_coverage_3_5": padic_coverage,
        "radius_by_valuation": radius_by_valuation,
    }


def visualize_comparison(frozen_metrics: dict, unfrozen_metrics: dict, output_dir: Path):
    """Create comparison visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Coverage comparison
    ax1 = axes[0, 0]
    configs = ["Frozen\n(p-adic)", "Unfrozen\n(hierarchy)"]
    coverages = [frozen_metrics["coverage"], unfrozen_metrics["coverage"]]
    colors = ["green" if c >= 0.99 else "orange" if c >= 0.9 else "red" for c in coverages]
    bars = ax1.bar(configs, coverages, color=colors, alpha=0.7, edgecolor="black")
    ax1.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="100% target")
    ax1.axhline(y=0.95, color="orange", linestyle="--", alpha=0.5, label="95% threshold")
    ax1.set_ylabel("Coverage")
    ax1.set_title("Coverage Comparison")
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    for bar, cov in zip(bars, coverages, strict=False):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{cov:.2%}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # Plot 2: Radial hierarchy comparison
    ax2 = axes[0, 1]
    hierarchy_A = [frozen_metrics["radial_corr_A"], unfrozen_metrics["radial_corr_A"]]
    hierarchy_B = [frozen_metrics["radial_corr_B"], unfrozen_metrics["radial_corr_B"]]
    x = np.arange(2)
    width = 0.35
    ax2.bar(x - width / 2, hierarchy_A, width, label="VAE-A", color="steelblue")
    ax2.bar(x + width / 2, hierarchy_B, width, label="VAE-B", color="coral")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs)
    ax2.set_ylabel("Radial Correlation (more negative = better)")
    ax2.set_title("Hierarchical Resolution")
    ax2.legend()
    ax2.axhline(y=-0.7, color="green", linestyle="--", alpha=0.5, label="Good threshold")

    # Plot 3: Radius by valuation
    ax3 = axes[0, 2]
    val_levels = range(10)
    frozen_radii = [frozen_metrics["radius_by_valuation"].get(v, {}).get("mean_A", 0) for v in val_levels]
    unfrozen_radii = [unfrozen_metrics["radius_by_valuation"].get(v, {}).get("mean_A", 0) for v in val_levels]
    ax3.plot(val_levels, frozen_radii, "go-", label="Frozen", linewidth=2, markersize=8)
    ax3.plot(val_levels, unfrozen_radii, "ro-", label="Unfrozen", linewidth=2, markersize=8)
    ax3.set_xlabel("Valuation Level")
    ax3.set_ylabel("Mean Radius")
    ax3.set_title("Radius by Valuation (ideal: monotonic decrease)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: P-adic generalization
    ax4 = axes[1, 0]
    padic_cov = [frozen_metrics["padic_coverage_3_5"], unfrozen_metrics["padic_coverage_3_5"]]
    colors = ["green" if c >= 0.95 else "orange" if c >= 0.8 else "red" for c in padic_cov]
    bars = ax4.bar(configs, padic_cov, color=colors, alpha=0.7, edgecolor="black")
    ax4.axhline(y=1.0, color="green", linestyle="--", alpha=0.5)
    ax4.set_ylabel("Coverage at p=3.5")
    ax4.set_title("P-adic Generalization (70K+ ops)")
    ax4.set_ylim(0, 1.1)
    for bar, cov in zip(bars, padic_cov, strict=False):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{cov:.2%}",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    # Plot 5: Trade-off scatter
    ax5 = axes[1, 1]
    ax5.scatter(
        frozen_metrics["coverage"],
        -frozen_metrics["radial_corr_A"],
        s=200,
        c="green",
        marker="o",
        label="Frozen",
        edgecolors="black",
        linewidths=2,
    )
    ax5.scatter(
        unfrozen_metrics["coverage"],
        -unfrozen_metrics["radial_corr_A"],
        s=200,
        c="red",
        marker="s",
        label="Unfrozen",
        edgecolors="black",
        linewidths=2,
    )
    ax5.set_xlabel("Coverage")
    ax5.set_ylabel("-Radial Correlation (higher = better hierarchy)")
    ax5.set_title("Coverage vs Hierarchy Trade-off")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axvline(x=0.95, color="orange", linestyle="--", alpha=0.5)
    ax5.axhline(y=0.7, color="blue", linestyle="--", alpha=0.5)

    # Plot 6: Summary table
    ax6 = axes[1, 2]
    ax6.axis("off")

    summary_text = f"""
FROZEN ENCODER vs UNFROZEN ENCODER COMPARISON

                          FROZEN        UNFROZEN
                         (p-adic)      (hierarchy)
─────────────────────────────────────────────────
Coverage                  {frozen_metrics["coverage"]:.2%}         {unfrozen_metrics["coverage"]:.2%}
Radial Corr A             {frozen_metrics["radial_corr_A"]:.3f}         {unfrozen_metrics["radial_corr_A"]:.3f}
Radial Corr B             {frozen_metrics["radial_corr_B"]:.3f}         {unfrozen_metrics["radial_corr_B"]:.3f}
Hierarchy Sep             {frozen_metrics["hierarchy_separation"]:.2f}x          {unfrozen_metrics["hierarchy_separation"]:.2f}x
P-adic Coverage (3.5)     {frozen_metrics["padic_coverage_3_5"]:.2%}         {unfrozen_metrics["padic_coverage_3_5"]:.2%}
Radius Range              {frozen_metrics["radius_range_A"]:.3f}         {unfrozen_metrics["radius_range_A"]:.3f}
Effective Dim             {frozen_metrics["effective_dim_A"]}             {unfrozen_metrics["effective_dim_A"]}

KEY INSIGHTS:
─────────────────────────────────────────────────
Frozen: {"GOOD" if frozen_metrics["coverage"] >= 0.99 else "POOR"} for p-adic generalization
        (100% coverage enables algebraic completeness)

Unfrozen: {"GOOD" if unfrozen_metrics["radial_corr_A"] < -0.5 else "POOR"} for hierarchical semantics
          (Deep hierarchy enables structured reasoning)

TRADE-OFF: {"COMPLEMENTARY" if frozen_metrics["coverage"] >= 0.99 and unfrozen_metrics["radial_corr_A"] < frozen_metrics["radial_corr_A"] else "OVERLAPPING"}
"""

    ax6.text(
        0.05,
        0.95,
        summary_text,
        transform=ax6.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_dir / "frozen_vs_unfrozen_comparison.png", dpi=150)
    plt.close()

    print(f"Saved comparison visualization to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare frozen vs unfrozen encoder")
    parser.add_argument(
        "--frozen_checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "v5_11_progressive" / "best.pt"),
        help="Path to frozen encoder checkpoint",
    )
    parser.add_argument(
        "--unfrozen_checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "progressive_tiny_lr" / "best.pt"),
        help="Path to unfrozen encoder checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR / "epsilon_vae_analysis" / "frozen_vs_unfrozen"),
        help="Output directory",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("FROZEN vs UNFROZEN ENCODER COMPARISON")
    print(f"{'=' * 70}")

    # Load checkpoints
    frozen_path = PROJECT_ROOT / args.frozen_checkpoint
    unfrozen_path = PROJECT_ROOT / args.unfrozen_checkpoint

    print(f"\nLoading frozen encoder: {frozen_path}")
    frozen_model, _ = load_checkpoint(frozen_path, device)

    print(f"Loading unfrozen encoder: {unfrozen_path}")
    unfrozen_model, _ = load_checkpoint(unfrozen_path, device)

    # Generate all operations
    all_ops = generate_all_ternary_operations()
    all_ops_tensor = torch.tensor(all_ops, dtype=torch.float32)
    indices = torch.arange(len(all_ops))

    # Compute metrics
    print(f"\n{'=' * 70}")
    print("COMPUTING METRICS")
    print(f"{'=' * 70}")

    print("\nFrozen encoder metrics:")
    frozen_metrics = compute_comprehensive_metrics(frozen_model, all_ops_tensor, indices, device)
    print(f"  Coverage: {frozen_metrics['coverage']:.2%}")
    print(f"  Radial Corr A: {frozen_metrics['radial_corr_A']:.3f}")
    print(f"  P-adic Coverage (3.5): {frozen_metrics['padic_coverage_3_5']:.2%}")

    print("\nUnfrozen encoder metrics:")
    unfrozen_metrics = compute_comprehensive_metrics(unfrozen_model, all_ops_tensor, indices, device)
    print(f"  Coverage: {unfrozen_metrics['coverage']:.2%}")
    print(f"  Radial Corr A: {unfrozen_metrics['radial_corr_A']:.3f}")
    print(f"  P-adic Coverage (3.5): {unfrozen_metrics['padic_coverage_3_5']:.2%}")

    # Visualize
    print(f"\n{'=' * 70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'=' * 70}")

    visualize_comparison(frozen_metrics, unfrozen_metrics, output_dir)

    # Save results
    results = {
        "frozen": frozen_metrics,
        "unfrozen": unfrozen_metrics,
        "frozen_checkpoint": str(frozen_path),
        "unfrozen_checkpoint": str(unfrozen_path),
    }

    # Convert numpy types
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(convert_numpy(results), f, indent=2)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    print("\nFROZEN ENCODER:")
    print(f"  Coverage: {frozen_metrics['coverage']:.2%} {'✓' if frozen_metrics['coverage'] >= 0.99 else '✗'}")
    print(f"  Hierarchy: {frozen_metrics['radial_corr_A']:.3f}")
    print(f"  P-adic generalization: {'ENABLED' if frozen_metrics['padic_coverage_3_5'] >= 0.95 else 'LIMITED'}")

    print("\nUNFROZEN ENCODER:")
    print(f"  Coverage: {unfrozen_metrics['coverage']:.2%}")
    print(
        f"  Hierarchy: {unfrozen_metrics['radial_corr_A']:.3f} {'✓ deeper' if unfrozen_metrics['radial_corr_A'] < frozen_metrics['radial_corr_A'] else ''}"
    )
    print(f"  P-adic generalization: {'ENABLED' if unfrozen_metrics['padic_coverage_3_5'] >= 0.95 else 'LIMITED'}")

    hierarchy_diff = unfrozen_metrics["radial_corr_A"] - frozen_metrics["radial_corr_A"]
    coverage_diff = frozen_metrics["coverage"] - unfrozen_metrics["coverage"]

    print("\nTRADE-OFF:")
    print(f"  Hierarchy gain (unfrozen): {hierarchy_diff:.3f}")
    print(f"  Coverage cost (unfrozen): {coverage_diff:.2%}")

    if coverage_diff > 0 and hierarchy_diff < 0:
        print("\n  → FUNDAMENTAL TRADE-OFF CONFIRMED")
        print(f"     Unfrozen gains {-hierarchy_diff:.3f} hierarchy at cost of {coverage_diff:.2%} coverage")
    elif coverage_diff == 0:
        print("\n  → NO COVERAGE TRADE-OFF")
        print(f"     Both achieve same coverage, unfrozen has {'better' if hierarchy_diff < 0 else 'worse'} hierarchy")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
