#!/usr/bin/env python3
"""Stability verification: Multiple long runs with optimal config.

Optimal config from sweeps:
- curvature = 2.0
- lr = 3e-4
- early_stopping_patience = 5
- hierarchy_weight = 5.0

This script runs 10 independent experiments to verify:
1. Consistency of peak hierarchy
2. Variance across runs
3. Epoch at which peak occurs
4. Relationship between initial state and final quality

Usage:
    python src/scripts/experiments/epsilon_vae/stability_verification.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.models import TernaryVAEV5_11_PartialFreeze
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat


class OptimalLoss(nn.Module):
    """Optimal loss configuration from sweeps."""

    def __init__(self):
        super().__init__()
        self.hierarchy_weight = 5.0
        self.coverage_weight = 1.0
        self.richness_weight = 2.0
        self.separation_weight = 3.0

        # Target radii: v0 at outer edge, v9 at center
        self.register_buffer("target_radii", torch.tensor([0.9 - (v / 9) * 0.8 for v in range(10)]))

    def forward(self, z_hyp, indices, logits, targets, original_radii=None):
        device = z_hyp.device
        radii = z_hyp.norm(dim=-1)
        valuations = TERNARY.valuation(indices).long().to(device)

        # Hierarchy loss
        hierarchy_loss = torch.tensor(0.0, device=device)
        unique_vals = torch.unique(valuations)
        for v in unique_vals:
            mask = valuations == v
            if mask.sum() > 0:
                hierarchy_loss += (radii[mask].mean() - self.target_radii[v]) ** 2
        hierarchy_loss /= len(unique_vals)

        # Coverage loss
        coverage_loss = nn.functional.cross_entropy(logits.view(-1, 3), (targets + 1).long().view(-1))

        # Richness preservation
        richness_loss = torch.tensor(0.0, device=device)
        if original_radii is not None:
            for v in unique_vals:
                mask = valuations == v
                if mask.sum() > 1:
                    ratio = radii[mask].var() / (original_radii[mask].var() + 1e-8)
                    if ratio < 0.5:
                        richness_loss += (0.5 - ratio) ** 2
            richness_loss /= max(len(unique_vals), 1)

        # Separation loss
        separation_loss = torch.tensor(0.0, device=device)
        mean_radii = [
            (v, radii[valuations == v].mean()) for v in sorted(unique_vals.tolist()) if (valuations == v).sum() > 0
        ]
        for i in range(len(mean_radii) - 1):
            separation_loss += torch.relu(mean_radii[i + 1][1] - mean_radii[i][1] + 0.01)

        total = (
            self.hierarchy_weight * hierarchy_loss
            + self.coverage_weight * coverage_loss
            + self.richness_weight * richness_loss
            + self.separation_weight * separation_loss
        )

        return {
            "total": total,
            "hierarchy": hierarchy_loss.item(),
            "coverage": coverage_loss.item(),
            "richness": richness_loss.item(),
        }


def compute_metrics(model, all_ops, indices, device):
    """Compute all metrics."""
    model.eval()
    all_radii, all_correct = [], []

    with torch.no_grad():
        for i in range(0, len(all_ops), 4096):
            batch = all_ops[i : i + 4096].to(device)
            out = model(batch, compute_control=False)
            all_radii.append(out["z_A_hyp"].norm(dim=-1).cpu().numpy())
            logits = model.decoder_A(out["mu_A"])
            preds = torch.argmax(logits, dim=-1) - 1
            all_correct.append((preds == batch.long()).float().mean(dim=1).cpu().numpy())

    all_radii = np.concatenate(all_radii)
    all_correct = np.concatenate(all_correct)
    valuations = TERNARY.valuation(indices).numpy()

    coverage = (all_correct == 1.0).mean()
    hierarchy = spearmanr(valuations, all_radii)[0]

    # Richness per level
    richness_per_level = {}
    total_richness = 0
    for v in range(10):
        mask = valuations == v
        if mask.sum() > 1:
            var = all_radii[mask].var()
            richness_per_level[v] = float(var)
            total_richness += var
    richness = total_richness / 10

    # Mean radii per level
    mean_radii_per_level = {}
    for v in range(10):
        mask = valuations == v
        if mask.sum() > 0:
            mean_radii_per_level[v] = float(all_radii[mask].mean())

    model.train()
    return {
        "coverage": float(coverage),
        "hierarchy": float(hierarchy),
        "richness": float(richness),
        "r_v0": mean_radii_per_level.get(0, np.nan),
        "r_v9": mean_radii_per_level.get(9, np.nan),
        "mean_radii": mean_radii_per_level,
        "richness_per_level": richness_per_level,
    }


def run_single_experiment(run_id, all_ops, indices, device, base_ckpt_path, epochs=50, patience=5, lr=3e-4):
    """Run a single stability experiment."""
    print(f"\n{'=' * 60}")
    print(f"RUN {run_id}")
    print(f"{'=' * 60}")

    save_dir = PROJECT_ROOT / f"checkpoints/stability_run_{run_id}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create model with optimal config
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=64,
        max_radius=0.99,
        curvature=2.0,  # Optimal from Phase 1
        use_controller=False,
        use_dual_projection=True,
        freeze_encoder_b=False,
        encoder_b_lr_scale=0.1,
        encoder_a_lr_scale=0.05,
    )

    # Load base checkpoint
    if base_ckpt_path.exists():
        ckpt = load_checkpoint_compat(base_ckpt_path, map_location=device)
        model.load_state_dict(get_model_state_dict(ckpt), strict=False)

    model = model.to(device)
    model.set_encoder_a_frozen(True)
    model.set_encoder_b_frozen(False)

    # Initial metrics
    init_metrics = compute_metrics(model, all_ops, indices, device)
    print(f"  Init: hier={init_metrics['hierarchy']:.4f}, cov={init_metrics['coverage'] * 100:.1f}%")

    # Setup training
    dataset = TensorDataset(all_ops, indices)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    # Get original radii for richness reference
    with torch.no_grad():
        model.eval()
        original_radii = torch.cat(
            [
                model(all_ops[i : i + 4096].to(device), compute_control=False)["z_A_hyp"].norm(dim=-1)
                for i in range(0, len(all_ops), 4096)
            ]
        )
        model.train()

    loss_fn = OptimalLoss().to(device)
    param_groups = model.get_param_groups(lr)
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

    # Training loop
    history = []
    best_hier = 0.0
    best_epoch = 0
    best_metrics = None
    no_improve = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_losses = {"total": 0, "hierarchy": 0, "coverage": 0, "richness": 0}
        n_batches = 0

        for batch_ops, batch_idx in dataloader:
            batch_ops = batch_ops.to(device)
            batch_idx = batch_idx.to(device)
            orig_radii_batch = original_radii[batch_idx]

            out = model(batch_ops, compute_control=False)
            z_A = out["z_A_hyp"]
            logits = model.decoder_A(out["mu_A"])

            losses = loss_fn(z_A, batch_idx, logits, batch_ops, orig_radii_batch)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in epoch_losses:
                epoch_losses[k] += losses.get(k, losses["total"].item() if k == "total" else 0)
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        # Evaluate every epoch
        metrics = compute_metrics(model, all_ops, indices, device)
        elapsed = time.time() - start_time

        history.append({"epoch": epoch, "elapsed": elapsed, "losses": epoch_losses, **metrics})

        # Print progress
        print(
            f"  Epoch {epoch:2d} | hier={metrics['hierarchy']:.4f} "
            f"rich={metrics['richness']:.6f} cov={metrics['coverage'] * 100:.1f}% "
            f"| {elapsed:.1f}s"
        )

        # Track best
        if metrics["hierarchy"] < best_hier and metrics["coverage"] > 0.99:
            best_hier = metrics["hierarchy"]
            best_epoch = epoch
            best_metrics = metrics.copy()
            no_improve = 0

            # Save best model
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "run_id": run_id,
                },
                save_dir / "best.pt",
            )
            print(f"    *** New best: {best_hier:.4f} ***")
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Final metrics
    final_metrics = compute_metrics(model, all_ops, indices, device)
    elapsed_total = time.time() - start_time

    result = {
        "run_id": run_id,
        "init_metrics": init_metrics,
        "best_hierarchy": best_hier,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "history": history,
        "elapsed_seconds": elapsed_total,
        "stopped_at_epoch": len(history) - 1,
    }

    # Save results
    with open(save_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2, default=float)

    print(
        f"\n  RESULT: best={best_hier:.4f} @ epoch {best_epoch}, "
        f"final={final_metrics['hierarchy']:.4f}, elapsed={elapsed_total:.1f}s"
    )

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs per run")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/v5_11_homeostasis/best.pt")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: {args.runs} runs, {args.epochs} epochs, patience={args.patience}, lr={args.lr}")

    # Load data
    print("\n=== Loading Dataset ===")
    all_ops_np = generate_all_ternary_operations()
    all_ops = torch.tensor(all_ops_np, dtype=torch.float32)
    indices = torch.arange(len(all_ops))
    print(f"Loaded {len(all_ops)} operations")

    base_ckpt = PROJECT_ROOT / args.base_checkpoint

    # Run experiments
    results = []
    total_start = time.time()

    for run_id in range(1, args.runs + 1):
        try:
            result = run_single_experiment(
                run_id, all_ops, indices, device, base_ckpt, epochs=args.epochs, patience=args.patience, lr=args.lr
            )
            results.append(result)
        except Exception as e:
            print(f"  ERROR in run {run_id}: {e}")
            import traceback

            traceback.print_exc()
            results.append({"run_id": run_id, "error": str(e)})

        torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start

    # Analysis
    print("\n" + "=" * 70)
    print("STABILITY ANALYSIS")
    print("=" * 70)

    valid_results = [r for r in results if "error" not in r]

    if valid_results:
        best_hiers = [r["best_hierarchy"] for r in valid_results]
        best_epochs = [r["best_epoch"] for r in valid_results]
        [r["final_metrics"]["hierarchy"] for r in valid_results]
        init_hiers = [r["init_metrics"]["hierarchy"] for r in valid_results]

        print(f"\n{'Run':<6} {'Init':>10} {'Best':>10} {'@ Epoch':>8} {'Final':>10}")
        print("-" * 50)
        for r in valid_results:
            print(
                f"{r['run_id']:<6} {r['init_metrics']['hierarchy']:>10.4f} "
                f"{r['best_hierarchy']:>10.4f} {r['best_epoch']:>8} "
                f"{r['final_metrics']['hierarchy']:>10.4f}"
            )

        print("-" * 50)
        print("\nBest Hierarchy:")
        print(f"  Mean: {np.mean(best_hiers):.4f}")
        print(f"  Std:  {np.std(best_hiers):.4f}")
        print(f"  Min:  {np.min(best_hiers):.4f}")
        print(f"  Max:  {np.max(best_hiers):.4f}")

        print("\nBest Epoch:")
        print(f"  Mean: {np.mean(best_epochs):.1f}")
        print(f"  Std:  {np.std(best_epochs):.1f}")
        print(f"  Range: {np.min(best_epochs)} - {np.max(best_epochs)}")

        print("\nInit Hierarchy:")
        print(f"  Mean: {np.mean(init_hiers):.4f}")
        print(f"  Std:  {np.std(init_hiers):.4f}")

        # Correlation between init and best
        if len(init_hiers) > 2:
            corr = np.corrcoef(init_hiers, best_hiers)[0, 1]
            print(f"\nCorrelation (init → best): {corr:.3f}")

    print(f"\nTotal time: {total_elapsed / 60:.1f} minutes")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "runs": args.runs,
            "epochs": args.epochs,
            "patience": args.patience,
            "lr": args.lr,
        },
        "total_elapsed_minutes": total_elapsed / 60,
        "results": results,
        "analysis": {
            "best_hier_mean": float(np.mean(best_hiers)) if valid_results else None,
            "best_hier_std": float(np.std(best_hiers)) if valid_results else None,
            "best_hier_min": float(np.min(best_hiers)) if valid_results else None,
            "best_hier_max": float(np.max(best_hiers)) if valid_results else None,
            "best_epoch_mean": float(np.mean(best_epochs)) if valid_results else None,
        },
    }

    summary_path = PROJECT_ROOT / "checkpoints/stability_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
