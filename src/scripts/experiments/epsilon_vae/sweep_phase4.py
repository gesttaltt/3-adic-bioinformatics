#!/usr/bin/env python3
"""Phase 4: LR optimization with winning early stopping strategy.

Phase 3 found:
- Early stopping (patience=5) achieves -0.7920 hierarchy
- Best hierarchy peaks at epoch 3

This phase tests:
1. Different learning rates
2. Multiple runs to measure variance
3. Trying to push past -0.8 hierarchy

Usage:
    python src/scripts/experiments/epsilon_vae/sweep_phase4.py
"""

import argparse
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


class RichHierarchyLoss(nn.Module):
    def __init__(self, hierarchy_weight=5.0, coverage_weight=1.0, richness_weight=2.0, separation_weight=3.0):
        super().__init__()
        self.hierarchy_weight = hierarchy_weight
        self.coverage_weight = coverage_weight
        self.richness_weight = richness_weight
        self.separation_weight = separation_weight

        self.register_buffer("target_radii", torch.tensor([0.9 - (v / 9) * 0.8 for v in range(10)]))

    def forward(self, z_hyp, indices, logits, targets, original_radii=None):
        device = z_hyp.device
        radii = z_hyp.norm(dim=-1)
        valuations = TERNARY.valuation(indices).long().to(device)

        hierarchy_loss = torch.tensor(0.0, device=device)
        unique_vals = torch.unique(valuations)
        for v in unique_vals:
            mask = valuations == v
            if mask.sum() > 0:
                mean_r = radii[mask].mean()
                hierarchy_loss += (mean_r - self.target_radii[v]) ** 2
        hierarchy_loss /= len(unique_vals)

        coverage_loss = nn.functional.cross_entropy(logits.view(-1, 3), (targets + 1).long().view(-1))

        richness_loss = torch.tensor(0.0, device=device)
        if original_radii is not None:
            for v in unique_vals:
                mask = valuations == v
                if mask.sum() > 1:
                    ratio = radii[mask].var() / (original_radii[mask].var() + 1e-8)
                    if ratio < 0.5:
                        richness_loss += (0.5 - ratio) ** 2
            richness_loss /= max(len(unique_vals), 1)

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
        return {"total": total}


def compute_metrics(model, all_ops, indices, device):
    model.eval()
    all_radii, all_correct = [], []

    with torch.no_grad():
        for i in range(0, len(all_ops), 4096):
            batch_ops = all_ops[i : i + 4096].to(device)
            out = model(batch_ops, compute_control=False)
            all_radii.append(out["z_A_hyp"].norm(dim=-1).cpu().numpy())
            logits = model.decoder_A(out["mu_A"])
            preds = torch.argmax(logits, dim=-1) - 1
            all_correct.append((preds == batch_ops.long()).float().mean(dim=1).cpu().numpy())

    all_radii = np.concatenate(all_radii)
    all_correct = np.concatenate(all_correct)
    valuations = TERNARY.valuation(indices).numpy()

    coverage = (all_correct == 1.0).mean()
    hierarchy = spearmanr(valuations, all_radii)[0]
    richness = sum(all_radii[valuations == v].var() for v in range(10) if (valuations == v).sum() > 1) / 10

    model.train()
    return {
        "coverage": coverage,
        "hierarchy": hierarchy,
        "richness": richness,
        "r_v0": float(all_radii[valuations == 0].mean()),
        "r_v9": float(all_radii[valuations == 9].mean()) if (valuations == 9).any() else np.nan,
    }


def run_experiment(config, all_ops, indices, device, base_ckpt_path):
    exp_name = config["name"]
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {exp_name} (lr={config['lr']:.1e})")
    print(f"{'=' * 60}")

    save_dir = PROJECT_ROOT / f"checkpoints/sweep4_{exp_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=64,
        max_radius=0.99,
        curvature=2.0,
        use_controller=False,
        use_dual_projection=True,
        freeze_encoder_b=False,
        encoder_b_lr_scale=config.get("encoder_b_lr_scale", 0.1),
        encoder_a_lr_scale=0.05,
    )

    if base_ckpt_path.exists():
        try:
            ckpt = load_checkpoint_compat(base_ckpt_path, map_location=device)
            model.load_state_dict(get_model_state_dict(ckpt), strict=False)
        except Exception as e:
            print(f"  Warning: {e}")

    model = model.to(device)
    model.set_encoder_a_frozen(True)
    model.set_encoder_b_frozen(False)

    dataset = TensorDataset(all_ops, indices)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    with torch.no_grad():
        model.eval()
        original_radii = torch.cat(
            [
                model(all_ops[i : i + 4096].to(device), compute_control=False)["z_A_hyp"].norm(dim=-1)
                for i in range(0, len(all_ops), 4096)
            ]
        )
        model.train()

    init_metrics = compute_metrics(model, all_ops, indices, device)
    print(f"  Initial: hier={init_metrics['hierarchy']:.4f}")

    loss_fn = RichHierarchyLoss(
        hierarchy_weight=config.get("hierarchy_weight", 5.0),
        richness_weight=config.get("richness_weight", 2.0),
    ).to(device)

    epochs = config.get("epochs", 30)
    lr = config["lr"]
    patience = config.get("patience", 5)

    param_groups = model.get_param_groups(lr)
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

    history = []
    best_hier = 0.0
    best_epoch = 0
    no_improve_count = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        for batch_ops, batch_idx in dataloader:
            batch_ops, batch_idx = batch_ops.to(device), batch_idx.to(device)
            out = model(batch_ops, compute_control=False)
            losses = loss_fn(
                out["z_A_hyp"], batch_idx, model.decoder_A(out["mu_A"]), batch_ops, original_radii[batch_idx]
            )
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Evaluate every epoch for fine-grained tracking
        metrics = compute_metrics(model, all_ops, indices, device)
        time.time() - start_time

        print(
            f"  Epoch {epoch:2d} | hier={metrics['hierarchy']:.4f} rich={metrics['richness']:.6f} cov={metrics['coverage'] * 100:.1f}%"
        )
        history.append({"epoch": epoch, **metrics})

        if metrics["hierarchy"] < best_hier and metrics["coverage"] > 0.99:
            best_hier = metrics["hierarchy"]
            best_epoch = epoch
            no_improve_count = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics, "config": config},
                save_dir / "best.pt",
            )
            print(f"    *** New best: {best_hier:.4f} ***")
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    final_metrics = compute_metrics(model, all_ops, indices, device)

    result = {
        "name": exp_name,
        "config": config,
        "init_metrics": init_metrics,
        "final_metrics": final_metrics,
        "best_hierarchy": best_hier,
        "best_epoch": best_epoch,
        "history": history,
        "elapsed_minutes": (time.time() - start_time) / 60,
    }

    with open(save_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2, default=float)

    print(f"\n  RESULT: best={best_hier:.4f} @ epoch {best_epoch}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/v5_11_homeostasis/best.pt")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Learning rate experiments
    experiments = {
        "lr_1e4": {"lr": 1e-4, "patience": 5, "epochs": args.epochs},
        "lr_3e4": {"lr": 3e-4, "patience": 5, "epochs": args.epochs},
        "lr_5e4": {"lr": 5e-4, "patience": 5, "epochs": args.epochs},
        "lr_1e3": {"lr": 1e-3, "patience": 5, "epochs": args.epochs},
        # Slower encoder B learning
        "lr_5e4_slowB": {"lr": 5e-4, "encoder_b_lr_scale": 0.05, "patience": 5, "epochs": args.epochs},
        # Higher hierarchy weight
        "lr_5e4_highH": {"lr": 5e-4, "hierarchy_weight": 8.0, "patience": 5, "epochs": args.epochs},
        # Multiple runs for variance
        "lr_5e4_run2": {"lr": 5e-4, "patience": 5, "epochs": args.epochs},
        "lr_5e4_run3": {"lr": 5e-4, "patience": 5, "epochs": args.epochs},
    }

    print(f"\nRunning {len(experiments)} experiments")

    all_ops_np = generate_all_ternary_operations()
    all_ops = torch.tensor(all_ops_np, dtype=torch.float32)
    indices = torch.arange(len(all_ops))

    base_ckpt = PROJECT_ROOT / args.base_checkpoint

    results = []
    total_start = time.time()

    for name, config in experiments.items():
        config["name"] = name
        try:
            result = run_experiment(config, all_ops, indices, device, base_ckpt)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"name": name, "error": str(e)})
        torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("PHASE 4 SUMMARY: Learning Rate Optimization")
    print("=" * 70)
    print(f"{'Experiment':<18} {'LR':>10} {'Best Hier':>10} {'@ Ep':>6} {'Cov':>6}")
    print("-" * 70)

    for r in results:
        if "error" not in r:
            print(
                f"{r['name']:<18} {r['config']['lr']:>10.1e} {r['best_hierarchy']:>10.4f} "
                f"{r['best_epoch']:>6} {r['final_metrics']['coverage'] * 100:>5.1f}%"
            )

    print("-" * 70)
    print(f"Total: {total_elapsed / 60:.1f} min")

    valid = [r for r in results if "error" not in r]
    if valid:
        best = min(valid, key=lambda x: x["best_hierarchy"])
        print(f"\nBEST: {best['name']} -> {best['best_hierarchy']:.4f} @ epoch {best['best_epoch']}")

        # Variance for lr_5e4 runs
        lr5e4_runs = [r for r in valid if "lr_5e4" in r["name"] and "slow" not in r["name"] and "high" not in r["name"]]
        if len(lr5e4_runs) >= 2:
            hiers = [r["best_hierarchy"] for r in lr5e4_runs]
            print(f"LR=5e-4 variance: mean={np.mean(hiers):.4f}, std={np.std(hiers):.4f}")

    summary_path = PROJECT_ROOT / "checkpoints/sweep4_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {"timestamp": datetime.now().isoformat(), "total_elapsed_minutes": total_elapsed / 60, "results": results},
            f,
            indent=2,
            default=float,
        )


if __name__ == "__main__":
    main()
