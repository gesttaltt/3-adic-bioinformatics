#!/usr/bin/env python3
"""Analyze all training checkpoints to identify best hierarchy+richness balance.

This script loads stored metrics from checkpoints and evaluates actual model
performance to find the optimal balance point.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.models import TernaryVAEV5_11_PartialFreeze


def evaluate_checkpoint(ckpt_path, device):
    """Load checkpoint and compute actual metrics."""
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except Exception as e:
        return {"error": str(e)}

    # Extract stored metrics
    stored = ckpt.get("metrics", {})
    config = ckpt.get("config", {})
    epoch = ckpt.get("epoch", "N/A")

    result = {
        "epoch": epoch,
        "stored_coverage": stored.get("coverage", stored.get("accuracy", "N/A")),
        "stored_hierarchy": stored.get("hierarchy", stored.get("radial_corr_A", "N/A")),
        "stored_hierarchy_B": stored.get("hierarchy_B", stored.get("radial_corr_B", "N/A")),
        "stored_richness": stored.get("richness", stored.get("within_var", "N/A")),
        "stored_Q": stored.get("Q", "N/A"),
        "stored_dist_corr": stored.get("dist_corr", stored.get("dist_corr_A", "N/A")),
        "config_use_controller": config.get("use_controller", "N/A"),
    }

    # Try to load and evaluate model
    try:
        use_controller = config.get("use_controller", False)
        model = TernaryVAEV5_11_PartialFreeze(
            latent_dim=16,
            hidden_dim=64,
            max_radius=0.99,
            curvature=1.0,
            use_controller=use_controller,
            use_dual_projection=True,
        )

        state_dict = ckpt.get("model_state_dict") or ckpt.get("model_state", {})
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()

        # Generate data
        all_ops_np = generate_all_ternary_operations()
        all_ops = torch.tensor(all_ops_np, dtype=torch.float32)
        indices = torch.arange(len(all_ops))
        valuations = TERNARY.valuation(indices).numpy()

        # Evaluate
        all_radii_A = []
        all_radii_B = []
        all_correct = []

        with torch.no_grad():
            for i in range(0, len(all_ops), 4096):
                batch = all_ops[i : i + 4096].to(device)
                out = model(batch, compute_control=False)

                all_radii_A.append(out["z_A_hyp"].norm(dim=-1).cpu().numpy())
                all_radii_B.append(out["z_B_hyp"].norm(dim=-1).cpu().numpy())

                logits = model.decoder_A(out["mu_A"])
                preds = torch.argmax(logits, dim=-1) - 1
                correct = (preds == batch.long()).float().mean(dim=1).cpu().numpy()
                all_correct.append(correct)

        all_radii_A = np.concatenate(all_radii_A)
        all_radii_B = np.concatenate(all_radii_B)
        all_correct = np.concatenate(all_correct)

        # Compute metrics
        coverage = (all_correct == 1.0).mean()
        hierarchy_A = spearmanr(valuations, all_radii_A)[0]
        hierarchy_B = spearmanr(valuations, all_radii_B)[0]

        # Richness: within-level variance
        richness = 0
        for v in range(10):
            mask = valuations == v
            if mask.sum() > 1:
                richness += all_radii_A[mask].var()
        richness /= 10

        result["actual_coverage"] = coverage
        result["actual_hierarchy_A"] = hierarchy_A
        result["actual_hierarchy_B"] = hierarchy_B
        result["actual_richness"] = richness
        result["r_v0"] = all_radii_A[valuations == 0].mean()
        result["r_v9"] = all_radii_A[valuations >= 8].mean() if (valuations >= 8).any() else np.nan

    except Exception as e:
        result["eval_error"] = str(e)

    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoints_dir = PROJECT_ROOT / "checkpoints"

    # Focus on most relevant checkpoints for hierarchy+richness analysis
    focus_checkpoints = [
        "v5_11_homeostasis",
        "v5_11_progressive",
        "v5_11_progressive_50ep",
        "v5_11_annealing",
        "v5_11_annealing_long",
        "v5_11_11_production",
        "hierarchy_focused",
        "hierarchy_extreme",
        "radial_target",
        "radial_collapse",
        "soft_radial",
        "balanced_radial",
        "homeostatic_rich",
        "max_hierarchy",
        "v5_11_overnight",  # Include for comparison (known illusion)
        "v5_11_learnable",
        "v5_11_learnable_qreg",
    ]

    results = []

    for name in focus_checkpoints:
        ckpt_path = checkpoints_dir / name / "best.pt"
        if ckpt_path.exists():
            print(f"\nAnalyzing: {name}")
            result = evaluate_checkpoint(ckpt_path, device)
            result["name"] = name
            results.append(result)
        else:
            print(f"  [NOT FOUND: {name}]")

    # Print summary table
    print("\n" + "=" * 120)
    print("CHECKPOINT COMPARISON - Focus on Hierarchy + Richness Balance")
    print("=" * 120)
    print(f"{'Name':<30} {'Cov%':>8} {'Hier_A':>10} {'Hier_B':>10} {'Richness':>12} {'r_v0':>8} {'r_v9':>8} {'Q':>8}")
    print("-" * 120)

    for r in results:
        if "eval_error" in r:
            print(f"{r['name']:<30} [ERROR: {r['eval_error'][:50]}]")
            continue

        cov = r.get("actual_coverage", r.get("stored_coverage", "N/A"))
        hier_a = r.get("actual_hierarchy_A", r.get("stored_hierarchy", "N/A"))
        hier_b = r.get("actual_hierarchy_B", r.get("stored_hierarchy_B", "N/A"))
        rich = r.get("actual_richness", r.get("stored_richness", "N/A"))
        r_v0 = r.get("r_v0", "N/A")
        r_v9 = r.get("r_v9", "N/A")
        Q = r.get("stored_Q", "N/A")

        def fmt(v):
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)[:8]

        cov_pct = f"{cov * 100:.1f}" if isinstance(cov, float) else str(cov)

        print(
            f"{r['name']:<30} {cov_pct:>8} {fmt(hier_a):>10} {fmt(hier_b):>10} {fmt(rich):>12} {fmt(r_v0):>8} {fmt(r_v9):>8} {fmt(Q):>8}"
        )

    print("=" * 120)

    # Identify best candidates
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    # Filter valid results
    valid = [r for r in results if "actual_coverage" in r]

    # Sort by best hierarchy (most negative VAE-B)
    by_hier_b = sorted(valid, key=lambda x: x.get("actual_hierarchy_B", 0))
    print("\nBest Hierarchy (VAE-B, most negative):")
    for r in by_hier_b[:3]:
        print(
            f"  {r['name']}: hier_B={r['actual_hierarchy_B']:.4f}, cov={r['actual_coverage'] * 100:.1f}%, rich={r.get('actual_richness', 0):.6f}"
        )

    # Sort by best richness (highest variance)
    by_rich = sorted(valid, key=lambda x: -x.get("actual_richness", 0))
    print("\nBest Richness (highest within-level variance):")
    for r in by_rich[:3]:
        print(
            f"  {r['name']}: rich={r['actual_richness']:.6f}, hier_B={r.get('actual_hierarchy_B', 0):.4f}, cov={r['actual_coverage'] * 100:.1f}%"
        )

    # Best balance: high coverage, good hierarchy, preserved richness
    def balance_score(r):
        cov = r.get("actual_coverage", 0)
        hier = abs(r.get("actual_hierarchy_B", 0))  # Want more negative = higher abs
        rich = r.get("actual_richness", 0)
        # Normalize: cov (0-1), hier (0-1), rich scaled
        if cov < 0.95:  # Coverage must be high
            return 0
        return hier * 0.5 + min(rich * 1000, 0.5)  # Weight hierarchy and richness equally

    by_balance = sorted(valid, key=lambda x: -balance_score(x))
    print("\nBest Balance (coverage>95%, hierarchy + richness):")
    for r in by_balance[:5]:
        score = balance_score(r)
        print(
            f"  {r['name']}: score={score:.4f}, hier_B={r.get('actual_hierarchy_B', 0):.4f}, rich={r.get('actual_richness', 0):.6f}, cov={r['actual_coverage'] * 100:.1f}%"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
