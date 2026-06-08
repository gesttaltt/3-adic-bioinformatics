# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Compare V5.11 Option A vs Option C in detail.

Analyzes:
1. Radial distribution by valuation level
2. Pairwise distance correlations
3. Latent space structure metrics
4. Per-valuation statistics
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR
from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.losses import poincare_distance
from src.models import TernaryVAEV5_11, TernaryVAEV5_11_PartialFreeze


def load_model_a(checkpoint_path, v5_5_path, device):
    """Load Option A model."""
    model = TernaryVAEV5_11(
        latent_dim=16,
        hidden_dim=64,
        max_radius=0.95,
        curvature=1.0,
        use_controller=False,
    )
    model.load_v5_5_checkpoint(v5_5_path, device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def load_model_c(checkpoint_path, v5_5_path, device):
    """Load PartialFreeze model (formerly Option C)."""
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=64,
        max_radius=0.95,
        curvature=1.0,
        use_controller=False,
        freeze_encoder_b=False,
        encoder_b_lr_scale=0.1,
    )
    model.load_v5_5_checkpoint(v5_5_path, device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def compute_detailed_metrics(model, x, indices, device, name="Model"):
    """Compute detailed metrics for a model."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {name}")
    print("=" * 60)

    with torch.no_grad():
        outputs = model(x, compute_control=False)
        z_A_hyp = outputs["z_A_hyp"]
        z_B_hyp = outputs["z_B_hyp"]
        mu_A = outputs["mu_A"]
        mu_B = outputs["mu_B"]

    # V5.12.2: Convert to numpy using hyperbolic distance
    origin = torch.zeros_like(z_A_hyp)
    radii_A = poincare_distance(z_A_hyp, origin, c=1.0).cpu().numpy()
    radii_B = poincare_distance(z_B_hyp, origin, c=1.0).cpu().numpy()
    valuations = TERNARY.valuation(indices).cpu().numpy()

    # 1. Overall radial correlations
    radial_corr_A = spearmanr(valuations, radii_A)[0]
    radial_corr_B = spearmanr(valuations, radii_B)[0]

    print("\n1. RADIAL HIERARCHY CORRELATION")
    print(f"   VAE-A: {radial_corr_A:.4f}")
    print(f"   VAE-B: {radial_corr_B:.4f}")

    # 2. Per-valuation statistics
    print("\n2. RADIUS BY VALUATION LEVEL")
    print(f"   {'Val':>3} | {'Count':>6} | {'Mean_A':>7} | {'Std_A':>6} | {'Mean_B':>7} | {'Std_B':>6} | {'Target':>7}")
    print(f"   {'-' * 3}-+-{'-' * 6}-+-{'-' * 7}-+-{'-' * 6}-+-{'-' * 7}-+-{'-' * 6}-+-{'-' * 7}")

    val_stats = {}
    for v in range(10):
        mask = valuations == v
        count = mask.sum()
        if count > 0:
            mean_A = radii_A[mask].mean()
            std_A = radii_A[mask].std()
            mean_B = radii_B[mask].mean()
            std_B = radii_B[mask].std()
            target = 0.85 - v * (0.85 - 0.1) / 9

            val_stats[v] = {
                "count": count,
                "mean_A": mean_A,
                "std_A": std_A,
                "mean_B": mean_B,
                "std_B": std_B,
                "target": target,
            }

            print(
                f"   {v:>3} | {count:>6} | {mean_A:>7.4f} | {std_A:>6.4f} | {mean_B:>7.4f} | {std_B:>6.4f} | {target:>7.4f}"
            )

    # 3. Poincare distance analysis
    print("\n3. POINCARE DISTANCE BY VALUATION DIFFERENCE")

    # Sample pairs
    n_pairs = 5000
    i_idx = torch.randint(0, len(x), (n_pairs,), device=device)
    j_idx = torch.randint(0, len(x), (n_pairs,), device=device)
    same = i_idx == j_idx
    j_idx[same] = (j_idx[same] + 1) % len(x)

    with torch.no_grad():
        d_A = poincare_distance(z_A_hyp[i_idx], z_A_hyp[j_idx]).cpu().numpy()
        d_B = poincare_distance(z_B_hyp[i_idx], z_B_hyp[j_idx]).cpu().numpy()

    diff = torch.abs(indices[i_idx] - indices[j_idx])
    pair_vals = TERNARY.valuation(diff).cpu().numpy()

    # Target distance
    3.0 * np.exp(-pair_vals / 3.0)

    dist_corr_A = spearmanr(pair_vals, -d_A)[0]  # Negative because high val = small dist
    dist_corr_B = spearmanr(pair_vals, -d_B)[0]

    print("   Distance-Valuation Correlation:")
    print(f"   VAE-A: {dist_corr_A:.4f}")
    print(f"   VAE-B: {dist_corr_B:.4f}")

    print("\n   Mean Distance by Pair Valuation:")
    print(f"   {'PairVal':>7} | {'Mean_A':>7} | {'Mean_B':>7} | {'Target':>7}")
    print(f"   {'-' * 7}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}")
    for pv in range(7):
        mask = pair_vals == pv
        if mask.sum() > 0:
            mean_A = d_A[mask].mean()
            mean_B = d_B[mask].mean()
            target = 3.0 * np.exp(-pv / 3.0)
            print(f"   {pv:>7} | {mean_A:>7.4f} | {mean_B:>7.4f} | {target:>7.4f}")

    # 4. Latent space spread
    print("\n4. LATENT SPACE STATISTICS")
    print(
        f"   VAE-A radius: min={radii_A.min():.4f}, max={radii_A.max():.4f}, range={radii_A.max() - radii_A.min():.4f}"
    )
    print(
        f"   VAE-B radius: min={radii_B.min():.4f}, max={radii_B.max():.4f}, range={radii_B.max() - radii_B.min():.4f}"
    )

    # Euclidean spread
    mu_A_np = mu_A.cpu().numpy()
    mu_B_np = mu_B.cpu().numpy()
    print(f"   VAE-A mu norm: mean={np.linalg.norm(mu_A_np, axis=1).mean():.4f}")
    print(f"   VAE-B mu norm: mean={np.linalg.norm(mu_B_np, axis=1).mean():.4f}")

    # 5. Coverage check
    with torch.no_grad():
        logits_A = model.decoder_A(mu_A)
        preds = torch.argmax(logits_A, dim=-1) - 1
        targets = x.long()
        correct = (preds == targets).float().mean(dim=1)
        coverage = (correct == 1.0).sum().item() / len(x)

    print(f"\n5. COVERAGE: {coverage * 100:.2f}%")

    return {
        "radial_corr_A": radial_corr_A,
        "radial_corr_B": radial_corr_B,
        "dist_corr_A": dist_corr_A,
        "dist_corr_B": dist_corr_B,
        "val_stats": val_stats,
        "radii_A": radii_A,
        "radii_B": radii_B,
        "valuations": valuations,
        "coverage": coverage,
    }


def plot_comparison(metrics_a, metrics_c, output_path):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Radial distribution by valuation - Option A
    ax = axes[0, 0]
    for v in range(10):
        mask = metrics_a["valuations"] == v
        if mask.sum() > 0:
            radii = metrics_a["radii_A"][mask]
            ax.scatter([v] * len(radii), radii, alpha=0.3, s=5)
    ax.set_xlabel("Valuation")
    ax.set_ylabel("Radius")
    ax.set_title(f"Option A - VAE-A (corr={metrics_a['radial_corr_A']:.3f})")
    ax.set_xlim(-0.5, 9.5)

    # 2. Radial distribution by valuation - Option A VAE-B
    ax = axes[0, 1]
    for v in range(10):
        mask = metrics_a["valuations"] == v
        if mask.sum() > 0:
            radii = metrics_a["radii_B"][mask]
            ax.scatter([v] * len(radii), radii, alpha=0.3, s=5)
    ax.set_xlabel("Valuation")
    ax.set_ylabel("Radius")
    ax.set_title(f"Option A - VAE-B (corr={metrics_a['radial_corr_B']:.3f})")
    ax.set_xlim(-0.5, 9.5)

    # 3. Mean radius comparison
    ax = axes[0, 2]
    vals = sorted(metrics_a["val_stats"].keys())
    mean_A_a = [metrics_a["val_stats"][v]["mean_A"] for v in vals]
    mean_B_a = [metrics_a["val_stats"][v]["mean_B"] for v in vals]
    mean_A_c = [metrics_c["val_stats"][v]["mean_A"] for v in vals]
    mean_B_c = [metrics_c["val_stats"][v]["mean_B"] for v in vals]
    targets = [metrics_a["val_stats"][v]["target"] for v in vals]

    ax.plot(vals, mean_A_a, "b-o", label="Opt A - VAE-A", markersize=4)
    ax.plot(vals, mean_B_a, "b--s", label="Opt A - VAE-B", markersize=4)
    ax.plot(vals, mean_A_c, "r-o", label="Opt C - VAE-A", markersize=4)
    ax.plot(vals, mean_B_c, "r--s", label="Opt C - VAE-B", markersize=4)
    ax.plot(vals, targets, "k:", label="Target", linewidth=2)
    ax.set_xlabel("Valuation")
    ax.set_ylabel("Mean Radius")
    ax.set_title("Mean Radius by Valuation")
    ax.legend(fontsize=8)

    # 4. Radial distribution - Option C VAE-A
    ax = axes[1, 0]
    for v in range(10):
        mask = metrics_c["valuations"] == v
        if mask.sum() > 0:
            radii = metrics_c["radii_A"][mask]
            ax.scatter([v] * len(radii), radii, alpha=0.3, s=5)
    ax.set_xlabel("Valuation")
    ax.set_ylabel("Radius")
    ax.set_title(f"Option C - VAE-A (corr={metrics_c['radial_corr_A']:.3f})")
    ax.set_xlim(-0.5, 9.5)

    # 5. Radial distribution - Option C VAE-B
    ax = axes[1, 1]
    for v in range(10):
        mask = metrics_c["valuations"] == v
        if mask.sum() > 0:
            radii = metrics_c["radii_B"][mask]
            ax.scatter([v] * len(radii), radii, alpha=0.3, s=5)
    ax.set_xlabel("Valuation")
    ax.set_ylabel("Radius")
    ax.set_title(f"Option C - VAE-B (corr={metrics_c['radial_corr_B']:.3f})")
    ax.set_xlim(-0.5, 9.5)

    # 6. Summary bar chart
    ax = axes[1, 2]
    metrics_names = ["Radial A", "Radial B", "Dist A", "Dist B"]
    opt_a_vals = [
        -metrics_a["radial_corr_A"],  # Negate to show positive bars
        -metrics_a["radial_corr_B"],
        metrics_a["dist_corr_A"],
        metrics_a["dist_corr_B"],
    ]
    opt_c_vals = [
        -metrics_c["radial_corr_A"],
        -metrics_c["radial_corr_B"],
        metrics_c["dist_corr_A"],
        metrics_c["dist_corr_B"],
    ]

    x_pos = np.arange(len(metrics_names))
    width = 0.35
    ax.bar(
        x_pos - width / 2,
        opt_a_vals,
        width,
        label="Option A",
        color="blue",
        alpha=0.7,
    )
    ax.bar(
        x_pos + width / 2,
        opt_c_vals,
        width,
        label="Option C",
        color="red",
        alpha=0.7,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel("Correlation (|value|)")
    ax.set_title("Correlation Comparison")
    ax.legend()
    ax.axhline(y=0.3, color="green", linestyle="--", label="Target")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nPlot saved to: {output_path}")
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Paths
    v5_5_path = CHECKPOINTS_DIR / "v5_5" / "latest.pt"
    # Option A: smaller checkpoints (441KB) - only projection trainable
    # Option C: larger checkpoints (815KB) - encoder_B also trainable
    option_a_path = CHECKPOINTS_DIR / "v5_11" / "epoch_180.pt"  # Option A
    option_c_path = CHECKPOINTS_DIR / "v5_11" / "best.pt"  # Option C (latest)

    # Load dataset
    print("\nLoading dataset...")
    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, dtype=torch.float32, device=device)
    indices = torch.arange(len(operations), device=device)
    print(f"Dataset size: {len(x)}")

    # Load and analyze Option A
    print("\n" + "=" * 60)
    print("LOADING OPTION A MODEL")
    print("=" * 60)
    model_a = load_model_a(option_a_path, v5_5_path, device)
    metrics_a = compute_detailed_metrics(model_a, x, indices, device, "Option A (Both Frozen)")

    # Load and analyze Option C
    print("\n" + "=" * 60)
    print("LOADING OPTION C MODEL")
    print("=" * 60)
    model_c = load_model_c(option_c_path, v5_5_path, device)
    metrics_c = compute_detailed_metrics(model_c, x, indices, device, "Option C (Encoder-B Trainable)")

    # Summary comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric':<25} | {'Option A':>12} | {'Option C':>12} | {'Winner':>10}")
    print(f"{'-' * 25}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}")

    comparisons = [
        (
            "Radial Corr A",
            metrics_a["radial_corr_A"],
            metrics_c["radial_corr_A"],
            "min",
        ),
        (
            "Radial Corr B",
            metrics_a["radial_corr_B"],
            metrics_c["radial_corr_B"],
            "min",
        ),
        (
            "Distance Corr A",
            metrics_a["dist_corr_A"],
            metrics_c["dist_corr_A"],
            "max",
        ),
        (
            "Distance Corr B",
            metrics_a["dist_corr_B"],
            metrics_c["dist_corr_B"],
            "max",
        ),
        ("Coverage", metrics_a["coverage"], metrics_c["coverage"], "max"),
    ]

    for name, val_a, val_c, mode in comparisons:
        if mode == "min":
            winner = "A" if val_a < val_c else ("C" if val_c < val_a else "TIE")
        else:
            winner = "A" if val_a > val_c else ("C" if val_c > val_a else "TIE")
        print(f"{name:<25} | {val_a:>12.4f} | {val_c:>12.4f} | {winner:>10}")

    # Create plots
    output_dir = Path("local-reports")
    output_dir.mkdir(exist_ok=True)
    plot_comparison(metrics_a, metrics_c, output_dir / "option_a_vs_c_comparison.png")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
