#!/usr/bin/env python3
"""Train TernaryVAE from scratch with optimal parameters.

Based on sweep findings (SWEEP_SUMMARY.md):
- curvature = 2.0 (marginally better)
- lr = 3e-4 (optimal sweet spot)
- early_stopping_patience = 5 (hierarchy peaks at epoch 3-10)
- hierarchy_weight = 5.0, richness_weight = 2.0

This script trains WITHOUT loading any checkpoint using a TWO-PHASE approach:
1. Phase 1: Train for coverage (reconstruction accuracy)
2. Phase 2: Train for hierarchy (radial ordering by valuation)
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
from src.utils.checkpoint import save_checkpoint

# Optimal config from sweep experiments
# Note: existing checkpoints use hidden_dim=256, so we match that for comparable results
OPTIMAL_CONFIG = {
    "latent_dim": 16,
    "hidden_dim": 256,  # Match existing checkpoints (64 is too small)
    "max_radius": 0.99,
    "curvature": 2.0,  # From Phase 1: 2.0 is marginally better
    "use_controller": False,  # Simpler architecture for from-scratch
    "use_dual_projection": True,
    "freeze_encoder_b": False,
    "encoder_b_lr_scale": 0.1,
    "encoder_a_lr_scale": 1.0,  # No scaling when training from scratch
}


class CoverageLoss(nn.Module):
    """Phase 1: Focus on reconstruction accuracy."""

    def forward(self, logits, targets):
        return nn.functional.cross_entropy(logits.view(-1, 3), (targets + 1).long().view(-1))


class HierarchyLoss(nn.Module):
    """Phase 2: Focus on radial ordering by valuation."""

    def __init__(self, hierarchy_weight=5.0, separation_weight=3.0):
        super().__init__()
        self.hierarchy_weight = hierarchy_weight
        self.separation_weight = separation_weight
        self.register_buffer("target_radii", torch.tensor([0.9 - (v / 9) * 0.8 for v in range(10)]))

    def forward(self, z_hyp, indices, logits, targets):
        device = z_hyp.device
        radii = z_hyp.norm(dim=-1)
        valuations = TERNARY.valuation(indices).long().to(device)

        # Coverage loss (still need to maintain reconstruction)
        coverage_loss = nn.functional.cross_entropy(logits.view(-1, 3), (targets + 1).long().view(-1))

        # Hierarchy loss
        hierarchy_loss = torch.tensor(0.0, device=device)
        unique_vals = torch.unique(valuations)
        for v in unique_vals:
            mask = valuations == v
            if mask.sum() > 0:
                hierarchy_loss += (radii[mask].mean() - self.target_radii[v]) ** 2
        hierarchy_loss /= len(unique_vals)

        # Separation loss
        separation_loss = torch.tensor(0.0, device=device)
        mean_radii = [
            (v, radii[valuations == v].mean()) for v in sorted(unique_vals.tolist()) if (valuations == v).sum() > 0
        ]
        for i in range(len(mean_radii) - 1):
            separation_loss += torch.relu(mean_radii[i + 1][1] - mean_radii[i][1] + 0.01)

        total = coverage_loss + self.hierarchy_weight * hierarchy_loss + self.separation_weight * separation_loss

        return {"total": total, "coverage": coverage_loss, "hierarchy": hierarchy_loss, "separation": separation_loss}


def compute_metrics(model, all_ops, indices, device):
    """Compute evaluation metrics."""
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


def train_phase1_coverage(
    model: nn.Module,
    all_ops: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-3,
    target_coverage: float = 0.99,
):
    """Phase 1: Train for coverage."""
    print("\n--- PHASE 1: COVERAGE ---")

    dataset = TensorDataset(all_ops, indices)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    loss_fn = CoverageLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_coverage = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch_ops, _ in dataloader:
            batch_ops = batch_ops.to(device)

            # Get model outputs and compute logits WITH gradients
            # Note: model's logits_A is computed in no_grad for verification only
            out = model(batch_ops, compute_control=False)
            logits = model.decoder_A(out["mu_A"])  # Compute with gradients

            loss = loss_fn(logits, batch_ops)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)

        metrics = compute_metrics(model, all_ops, indices, device)
        print(
            f"  Epoch {epoch:2d} | cov={metrics['coverage'] * 100:.1f}% "
            f"hier={metrics['hierarchy']:.4f} loss={avg_loss:.4f}"
        )

        if metrics["coverage"] > best_coverage:
            best_coverage = metrics["coverage"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if metrics["coverage"] >= target_coverage:
            print(f"  Target coverage {target_coverage * 100:.0f}% reached!")
            break

        if no_improve >= 10:
            print(f"  Coverage plateau at {best_coverage * 100:.1f}%")
            break

    # Restore best coverage state
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_coverage


def train_phase2_hierarchy(
    model: nn.Module,
    all_ops: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    save_dir: Path,
    epochs: int = 30,
    lr: float = 3e-4,
    patience: int = 5,
):
    """Phase 2: Train for hierarchy while maintaining coverage."""
    print("\n--- PHASE 2: HIERARCHY ---")

    dataset = TensorDataset(all_ops, indices)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    loss_fn = HierarchyLoss(hierarchy_weight=5.0, separation_weight=3.0).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_hier = 0.0
    best_epoch = 0
    best_metrics = None
    no_improve = 0
    history = []

    for epoch in range(epochs):
        model.train()

        for batch_ops, batch_idx in dataloader:
            batch_ops, batch_idx = batch_ops.to(device), batch_idx.to(device)
            out = model(batch_ops, compute_control=False)
            # Compute logits with gradients (model's logits_A is in no_grad)
            logits = model.decoder_A(out["mu_A"])
            losses = loss_fn(out["z_A_hyp"], batch_idx, logits, batch_ops)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        metrics = compute_metrics(model, all_ops, indices, device)
        history.append({"epoch": epoch, **metrics})

        print(
            f"  Epoch {epoch:2d} | hier={metrics['hierarchy']:.4f} "
            f"rich={metrics['richness']:.6f} cov={metrics['coverage'] * 100:.1f}%"
        )

        # Track best (negative hierarchy, maintain coverage)
        if metrics["hierarchy"] < best_hier and metrics["coverage"] > 0.95:
            best_hier = metrics["hierarchy"]
            best_epoch = epoch
            best_metrics = metrics.copy()
            no_improve = 0

            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "metrics": metrics,
                "config": OPTIMAL_CONFIG,
                "training_type": "from_scratch_two_phase",
            }
            save_checkpoint(checkpoint, save_dir / "best.pt")
            print(f"    *** New best: {best_hier:.4f} ***")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    return best_hier, best_epoch, best_metrics, history


def train_from_scratch(
    run_id: int,
    all_ops: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
    save_dir: Path,
    phase1_epochs: int = 50,
    phase2_epochs: int = 30,
):
    """Train a model from scratch using two-phase approach."""
    print(f"\n{'=' * 60}")
    print(f"FROM-SCRATCH RUN {run_id}")
    print(f"{'=' * 60}")

    run_dir = save_dir / f"scratch_run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Create model from scratch
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=OPTIMAL_CONFIG["latent_dim"],
        hidden_dim=OPTIMAL_CONFIG["hidden_dim"],
        max_radius=OPTIMAL_CONFIG["max_radius"],
        curvature=OPTIMAL_CONFIG["curvature"],
        use_controller=OPTIMAL_CONFIG["use_controller"],
        use_dual_projection=OPTIMAL_CONFIG["use_dual_projection"],
        freeze_encoder_b=OPTIMAL_CONFIG["freeze_encoder_b"],
    )
    model = model.to(device)

    # CRITICAL: Unfreeze both encoders for from-scratch training
    model.set_encoder_a_frozen(False)
    model.set_encoder_b_frozen(False)

    init_metrics = compute_metrics(model, all_ops, indices, device)
    print(f"  Random init: hier={init_metrics['hierarchy']:.4f}, cov={init_metrics['coverage'] * 100:.1f}%")

    start = time.time()

    # Phase 1: Coverage
    phase1_coverage = train_phase1_coverage(
        model, all_ops, indices, device, epochs=phase1_epochs, lr=1e-3, target_coverage=0.99
    )

    phase1_metrics = compute_metrics(model, all_ops, indices, device)
    print(f"\n  After Phase 1: cov={phase1_metrics['coverage'] * 100:.1f}%, hier={phase1_metrics['hierarchy']:.4f}")

    # Phase 2: Hierarchy
    best_hier, best_epoch, best_metrics, history = train_phase2_hierarchy(
        model, all_ops, indices, device, run_dir, epochs=phase2_epochs, lr=3e-4, patience=5
    )

    elapsed = time.time() - start
    final_metrics = compute_metrics(model, all_ops, indices, device)

    print(f"\n  RESULT: best_hier={best_hier:.4f} @ epoch {best_epoch}, elapsed={elapsed:.1f}s")

    return {
        "run_id": run_id,
        "init_metrics": init_metrics,
        "phase1_coverage": phase1_coverage,
        "phase1_metrics": phase1_metrics,
        "best_hierarchy": best_hier,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "final_metrics": final_metrics,
        "elapsed": elapsed,
        "history": history,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train TernaryVAE from scratch (two-phase)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--phase1-epochs", type=int, default=50, help="Phase 1 (coverage) epochs")
    parser.add_argument("--phase2-epochs", type=int, default=30, help="Phase 2 (hierarchy) epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (optional)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("Two-Phase Training from Scratch")
    print(f"  Phase 1: {args.phase1_epochs} epochs for coverage")
    print(f"  Phase 2: {args.phase2_epochs} epochs for hierarchy")
    print(f"  Curvature: {OPTIMAL_CONFIG['curvature']}")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"  Seed: {args.seed}")

    # Load data
    all_ops = torch.tensor(generate_all_ternary_operations(), dtype=torch.float32)
    indices = torch.arange(len(all_ops))
    print(f"\nData: {len(all_ops)} operations")

    save_dir = PROJECT_ROOT / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_start = time.time()

    for run_id in range(1, args.runs + 1):
        result = train_from_scratch(
            run_id=run_id,
            all_ops=all_ops,
            indices=indices,
            device=device,
            save_dir=save_dir,
            phase1_epochs=args.phase1_epochs,
            phase2_epochs=args.phase2_epochs,
        )
        results.append(result)
        torch.cuda.empty_cache()

    total = time.time() - total_start

    # Summary
    print("\n" + "=" * 60)
    print("FROM-SCRATCH TRAINING SUMMARY")
    print("=" * 60)

    best_hiers = [r["best_hierarchy"] for r in results]
    coverages = [r["best_metrics"]["coverage"] if r["best_metrics"] else 0 for r in results]

    print(f"\n{'Run':<6} {'P1 Cov':>10} {'Best Hier':>12} {'Final Hier':>12} {'@ Ep':>6}")
    print("-" * 55)
    for r in results:
        print(
            f"{r['run_id']:<6} {r['phase1_coverage'] * 100:>9.1f}% "
            f"{r['best_hierarchy']:>12.4f} {r['final_metrics']['hierarchy']:>12.4f} "
            f"{r['best_epoch']:>6}"
        )

    print(f"\nBest Hierarchy: mean={np.mean(best_hiers):.4f}, std={np.std(best_hiers):.4f}")
    print(f"Coverage: mean={np.mean(coverages) * 100:.1f}%")
    print(f"\nTotal time: {total / 60:.1f} min")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "config": OPTIMAL_CONFIG,
        "results": [{k: v for k, v in r.items() if k != "history"} for r in results],
        "summary": {
            "best_hier_mean": float(np.mean(best_hiers)),
            "best_hier_std": float(np.std(best_hiers)),
            "coverage_mean": float(np.mean(coverages)),
            "total_minutes": total / 60,
        },
    }

    summary_path = save_dir / "from_scratch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
