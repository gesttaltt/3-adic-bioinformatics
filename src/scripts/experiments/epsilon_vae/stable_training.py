#!/usr/bin/env python3
"""Stable training with CORRECT checkpoint loading.

The v5_11_homeostasis checkpoint was saved with:
- use_controller=True
- curvature=1.0
- max_radius=0.95
- projection_layers=1

We MUST match this architecture for correct loading!
"""

import json
import sys
import time
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
    def __init__(self):
        super().__init__()
        self.hierarchy_weight = 5.0
        self.coverage_weight = 1.0
        self.richness_weight = 2.0
        self.separation_weight = 3.0
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
                hierarchy_loss += (radii[mask].mean() - self.target_radii[v]) ** 2
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
            batch = all_ops[i : i + 4096].to(device)
            out = model(batch, compute_control=False)
            all_radii.append(out["z_A_hyp"].norm(dim=-1).cpu().numpy())
            logits = model.decoder_A(out["mu_A"])
            all_correct.append((torch.argmax(logits, dim=-1) - 1 == batch.long()).float().mean(dim=1).cpu().numpy())
    all_radii = np.concatenate(all_radii)
    all_correct = np.concatenate(all_correct)
    valuations = TERNARY.valuation(indices).numpy()
    richness = sum(all_radii[valuations == v].var() for v in range(10) if (valuations == v).sum() > 1) / 10
    model.train()
    return {
        "coverage": float((all_correct == 1.0).mean()),
        "hierarchy": float(spearmanr(valuations, all_radii)[0]),
        "richness": float(richness),
        "r_v0": float(all_radii[valuations == 0].mean()),
        "r_v9": float(all_radii[valuations == 9].mean()) if (valuations == 9).any() else np.nan,
    }


def run_stable_experiment(run_id, all_ops, indices, device, ckpt_path, epochs=30, patience=5, lr=3e-4):
    print(f"\n{'=' * 60}")
    print(f"STABLE RUN {run_id}")
    print(f"{'=' * 60}")

    save_dir = PROJECT_ROOT / f"checkpoints/stable_run_{run_id}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get checkpoint config for key architecture params
    ckpt_preview = load_checkpoint_compat(ckpt_path, map_location="cpu")
    ckpt_config = ckpt_preview.get("config", {})
    print(
        f"  Checkpoint config: curv={ckpt_config.get('curvature', 1.0)}, "
        f"controller={ckpt_config.get('use_controller', False)}, "
        f"proj_hidden={ckpt_config.get('projection_hidden_dim', 64)}"
    )

    # CRITICAL: Use projection_hidden_dim (64) as hidden_dim to match projection layers
    # The encoder weights will partially load via strict=False
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=ckpt_config.get("projection_hidden_dim", 64),  # Match projection!
        max_radius=ckpt_config.get("max_radius", 0.95),
        curvature=ckpt_config.get("curvature", 1.0),
        use_controller=ckpt_config.get("use_controller", True),
        use_dual_projection=ckpt_config.get("dual_projection", True),
        freeze_encoder_b=False,
        encoder_b_lr_scale=0.1,
        encoder_a_lr_scale=0.05,
    )

    # Load checkpoint
    ckpt = load_checkpoint_compat(ckpt_path, map_location=device)
    model_state = get_model_state_dict(ckpt)

    # Try strict load first
    try:
        model.load_state_dict(model_state, strict=True)
        print("  Strict load succeeded!")
    except Exception as e:
        print(f"  Strict load failed, using partial: {str(e)[:100]}")
        model.load_state_dict(model_state, strict=False)

    model = model.to(device)
    model.set_encoder_a_frozen(True)
    model.set_encoder_b_frozen(False)

    # Verify consistent loading
    init_metrics = compute_metrics(model, all_ops, indices, device)
    print(f"  Init: hier={init_metrics['hierarchy']:.4f}, cov={init_metrics['coverage'] * 100:.1f}%")

    # Setup training
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

    loss_fn = OptimalLoss().to(device)
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
            losses = loss_fn(
                out["z_A_hyp"], batch_idx, model.decoder_A(out["mu_A"]), batch_ops, original_radii[batch_idx]
            )
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        metrics = compute_metrics(model, all_ops, indices, device)
        print(
            f"  Epoch {epoch:2d} | hier={metrics['hierarchy']:.4f} "
            f"rich={metrics['richness']:.6f} cov={metrics['coverage'] * 100:.1f}%"
        )

        if metrics["hierarchy"] < best_hier and metrics["coverage"] > 0.99:
            best_hier = metrics["hierarchy"]
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics, "run_id": run_id},
                save_dir / "best.pt",
            )
            print(f"    *** New best: {best_hier:.4f} ***")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start
    final = compute_metrics(model, all_ops, indices, device)

    print(f"\n  RESULT: best={best_hier:.4f} @ epoch {best_epoch}, elapsed={elapsed:.1f}s")
    return {
        "run_id": run_id,
        "init_metrics": init_metrics,
        "best_hierarchy": best_hier,
        "best_epoch": best_epoch,
        "final_metrics": final,
        "elapsed": elapsed,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_ops = torch.tensor(generate_all_ternary_operations(), dtype=torch.float32)
    indices = torch.arange(len(all_ops))

    ckpt_path = PROJECT_ROOT / "checkpoints/v5_11_homeostasis/best.pt"

    results = []
    total_start = time.time()

    for run_id in range(1, args.runs + 1):
        result = run_stable_experiment(run_id, all_ops, indices, device, ckpt_path, args.epochs)
        results.append(result)
        torch.cuda.empty_cache()

    total = time.time() - total_start

    # Analysis
    print("\n" + "=" * 60)
    print("STABLE TRAINING RESULTS")
    print("=" * 60)

    init_hiers = [r["init_metrics"]["hierarchy"] for r in results]
    best_hiers = [r["best_hierarchy"] for r in results]

    print(f"\n{'Run':<6} {'Init':>10} {'Best':>10} {'@ Ep':>6}")
    print("-" * 40)
    for r in results:
        print(
            f"{r['run_id']:<6} {r['init_metrics']['hierarchy']:>10.4f} "
            f"{r['best_hierarchy']:>10.4f} {r['best_epoch']:>6}"
        )

    print(f"\nInit Hierarchy: mean={np.mean(init_hiers):.4f}, std={np.std(init_hiers):.4f}")
    print(f"Best Hierarchy: mean={np.mean(best_hiers):.4f}, std={np.std(best_hiers):.4f}")
    print(f"\nTotal time: {total / 60:.1f} min")

    summary_path = PROJECT_ROOT / "checkpoints/stable_training_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"results": results, "total_minutes": total / 60}, f, indent=2, default=float)


if __name__ == "__main__":
    main()
