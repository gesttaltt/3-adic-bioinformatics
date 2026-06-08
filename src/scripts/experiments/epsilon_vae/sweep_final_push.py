#!/usr/bin/env python3
"""Final push: Try to break -0.8 hierarchy barrier.

Best findings so far:
- LR=3e-4 with early stopping is optimal
- lr_1e3 achieves -0.7902 at epoch 0

This script tries:
1. Starting from homeostatic_rich (already has -0.83 hierarchy)
2. Very careful fine-tuning with low LR
3. Multiple random seeds

Usage:
    python src/scripts/experiments/epsilon_vae/sweep_final_push.py
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


class HierarchyLoss(nn.Module):
    """Simplified loss focused purely on hierarchy."""

    def __init__(self, hierarchy_weight=8.0, coverage_weight=1.0):
        super().__init__()
        self.hierarchy_weight = hierarchy_weight
        self.coverage_weight = coverage_weight
        self.register_buffer(
            "target_radii",
            torch.tensor(
                [
                    0.95 - (v / 9) * 0.85
                    for v in range(10)  # More aggressive targets
                ]
            ),
        )

    def forward(self, z_hyp, indices, logits, targets):
        device = z_hyp.device
        radii = z_hyp.norm(dim=-1)
        valuations = TERNARY.valuation(indices).long().to(device)

        hierarchy_loss = torch.tensor(0.0, device=device)
        unique_vals = torch.unique(valuations)
        for v in unique_vals:
            mask = valuations == v
            if mask.sum() > 0:
                hierarchy_loss += (radii[mask].mean() - self.target_radii[v]) ** 2
        hierarchy_loss /= len(unique_vals)

        coverage_loss = nn.functional.cross_entropy(logits.view(-1, 3), (targets + 1).long().view(-1))

        return self.hierarchy_weight * hierarchy_loss + self.coverage_weight * coverage_loss


def compute_metrics(model, all_ops, indices, device):
    model.eval()
    all_radii, all_correct = [], []
    with torch.no_grad():
        for i in range(0, len(all_ops), 4096):
            batch = all_ops[i : i + 4096].to(device)
            out = model(batch, compute_control=False)
            all_radii.append(out["z_A_hyp"].norm(dim=-1).cpu().numpy())
            logits = model.decoder_A(out["mu_A"])
            all_correct.append((torch.argmax(logits, dim=-1) - 1 == batch.long()).float().mean(dim=1).cpu().numpy())
    all_radii = np.concatenate(all_radii)
    all_correct = np.concatenate(all_correct)
    valuations = TERNARY.valuation(indices).numpy()
    model.train()
    return {
        "coverage": (all_correct == 1.0).mean(),
        "hierarchy": spearmanr(valuations, all_radii)[0],
        "richness": sum(all_radii[valuations == v].var() for v in range(10) if (valuations == v).sum() > 1) / 10,
        "r_v0": float(all_radii[valuations == 0].mean()),
        "r_v9": float(all_radii[valuations == 9].mean()) if (valuations == 9).any() else np.nan,
    }


def run_experiment(name, ckpt_path, lr, all_ops, indices, device, epochs=20, patience=5):
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {name}")
    print(f"Checkpoint: {ckpt_path.name}, LR: {lr:.1e}")
    print(f"{'=' * 60}")

    save_dir = PROJECT_ROOT / f"checkpoints/final_{name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=64,
        max_radius=0.99,
        curvature=2.0,
        use_controller=False,
        use_dual_projection=True,
        freeze_encoder_b=False,
        encoder_b_lr_scale=0.1,
        encoder_a_lr_scale=0.05,
    )

    if ckpt_path.exists():
        ckpt = load_checkpoint_compat(ckpt_path, map_location=device)
        model.load_state_dict(get_model_state_dict(ckpt), strict=False)
        print("  Loaded checkpoint")

    model = model.to(device)
    model.set_encoder_a_frozen(True)
    model.set_encoder_b_frozen(False)

    init_metrics = compute_metrics(model, all_ops, indices, device)
    print(f"  Initial: hier={init_metrics['hierarchy']:.4f}, cov={init_metrics['coverage'] * 100:.1f}%")

    dataset = TensorDataset(all_ops, indices)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)
    loss_fn = HierarchyLoss().to(device)
    param_groups = model.get_param_groups(lr)
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

    best_hier = 0.0
    best_epoch = 0
    no_improve = 0
    start = time.time()

    for epoch in range(epochs):
        model.train()
        for batch_ops, batch_idx in dataloader:
            batch_ops, batch_idx = batch_ops.to(device), batch_idx.to(device)
            out = model(batch_ops, compute_control=False)
            loss = loss_fn(out["z_A_hyp"], batch_idx, model.decoder_A(out["mu_A"]), batch_ops)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        metrics = compute_metrics(model, all_ops, indices, device)
        print(
            f"  Epoch {epoch:2d} | hier={metrics['hierarchy']:.4f} rich={metrics['richness']:.6f} cov={metrics['coverage'] * 100:.1f}%"
        )

        if metrics["hierarchy"] < best_hier and metrics["coverage"] > 0.99:
            best_hier = metrics["hierarchy"]
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics}, save_dir / "best.pt"
            )
            print(f"    *** New best: {best_hier:.4f} ***")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start
    final = compute_metrics(model, all_ops, indices, device)

    print(f"\n  RESULT: best={best_hier:.4f} @ epoch {best_epoch}, elapsed={elapsed / 60:.1f}min")
    return {
        "name": name,
        "best_hierarchy": best_hier,
        "best_epoch": best_epoch,
        "init_metrics": init_metrics,
        "final_metrics": final,
        "elapsed": elapsed / 60,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_ops = torch.tensor(generate_all_ternary_operations(), dtype=torch.float32)
    indices = torch.arange(len(all_ops))

    ckpt_homeostatic = PROJECT_ROOT / "checkpoints/v5_11_homeostasis/best.pt"
    ckpt_rich = PROJECT_ROOT / "checkpoints/homeostatic_rich/best.pt"

    experiments = [
        # From homeostatic checkpoint (already good hierarchy)
        ("homeo_lr3e4", ckpt_homeostatic, 3e-4),
        ("homeo_lr1e3", ckpt_homeostatic, 1e-3),
        ("homeo_lr5e4", ckpt_homeostatic, 5e-4),
        # From rich checkpoint (ceiling hierarchy + high richness)
        ("rich_lr3e4", ckpt_rich, 3e-4),
        ("rich_lr1e4", ckpt_rich, 1e-4),
        ("rich_lr5e5", ckpt_rich, 5e-5),  # Very low LR to preserve structure
    ]

    results = []
    total_start = time.time()

    for name, ckpt, lr in experiments:
        result = run_experiment(name, ckpt, lr, all_ops, indices, device)
        results.append(result)
        torch.cuda.empty_cache()

    total = time.time() - total_start

    print("\n" + "=" * 70)
    print("FINAL PUSH SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<18} {'Init Hier':>10} {'Best Hier':>10} {'@ Ep':>6}")
    print("-" * 70)

    for r in results:
        print(
            f"{r['name']:<18} {r['init_metrics']['hierarchy']:>10.4f} {r['best_hierarchy']:>10.4f} {r['best_epoch']:>6}"
        )

    print("-" * 70)
    print(f"Total: {total / 60:.1f} min")

    best = min(results, key=lambda x: x["best_hierarchy"])
    print(f"\nBEST OVERALL: {best['name']} -> {best['best_hierarchy']:.4f}")

    if best["best_hierarchy"] < -0.8:
        print("*** BROKE THE -0.8 BARRIER! ***")

    summary_path = PROJECT_ROOT / "checkpoints/final_push_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {"timestamp": datetime.now().isoformat(), "results": results, "total_minutes": total / 60},
            f,
            indent=2,
            default=float,
        )


if __name__ == "__main__":
    main()
