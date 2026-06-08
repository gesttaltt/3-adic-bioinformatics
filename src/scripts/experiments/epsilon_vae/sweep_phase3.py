#!/usr/bin/env python3
"""Phase 3: Early stopping and LR scheduling experiments.

Phase 2 found that hierarchy PEAKS at epochs 10-20 then DEGRADES.
This phase tests strategies to preserve peak performance:

1. Early stopping with patience
2. LR decay after peak detection
3. Cosine annealing
4. Step decay schedule

Usage:
    python src/scripts/experiments/epsilon_vae/sweep_phase3.py
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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.models import TernaryVAEV5_11_PartialFreeze
from src.models.homeostasis import compute_Q
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat


class RichHierarchyLoss(nn.Module):
    """Loss function balancing hierarchy, coverage, and richness."""

    def __init__(
        self,
        inner_radius: float = 0.1,
        outer_radius: float = 0.9,
        hierarchy_weight: float = 5.0,
        coverage_weight: float = 1.0,
        richness_weight: float = 2.0,
        separation_weight: float = 3.0,
    ):
        super().__init__()
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.hierarchy_weight = hierarchy_weight
        self.coverage_weight = coverage_weight
        self.richness_weight = richness_weight
        self.separation_weight = separation_weight
        self.max_valuation = 9

        self.register_buffer(
            "target_radii",
            torch.tensor([outer_radius - (v / self.max_valuation) * (outer_radius - inner_radius) for v in range(10)]),
        )

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
                target_r = self.target_radii[v]
                hierarchy_loss = hierarchy_loss + (mean_r - target_r) ** 2
        hierarchy_loss = hierarchy_loss / len(unique_vals)

        coverage_loss = nn.functional.cross_entropy(
            logits.view(-1, 3),
            (targets + 1).long().view(-1),
        )

        richness_loss = torch.tensor(0.0, device=device)
        if original_radii is not None:
            for v in unique_vals:
                mask = valuations == v
                if mask.sum() > 1:
                    new_var = radii[mask].var()
                    orig_var = original_radii[mask].var() + 1e-8
                    ratio = new_var / orig_var
                    if ratio < 0.5:
                        richness_loss = richness_loss + (0.5 - ratio) ** 2
            richness_loss = richness_loss / max(len(unique_vals), 1)

        separation_loss = torch.tensor(0.0, device=device)
        mean_radii_list = []
        for v in sorted(unique_vals.tolist()):
            mask = valuations == v
            if mask.sum() > 0:
                mean_radii_list.append((v, radii[mask].mean()))
        for i in range(len(mean_radii_list) - 1):
            v1, r1 = mean_radii_list[i]
            v2, r2 = mean_radii_list[i + 1]
            violation = torch.relu(r2 - r1 + 0.01)
            separation_loss = separation_loss + violation

        total = (
            self.hierarchy_weight * hierarchy_loss
            + self.coverage_weight * coverage_loss
            + self.richness_weight * richness_loss
            + self.separation_weight * separation_loss
        )

        return {
            "total": total,
            "hierarchy_loss": hierarchy_loss,
            "coverage_loss": coverage_loss,
            "richness_loss": richness_loss,
        }


def compute_metrics(model, all_ops, indices, device):
    """Compute metrics."""
    model.eval()
    batch_size = 4096
    all_radii, all_correct, all_z = [], [], []

    with torch.no_grad():
        for i in range(0, len(all_ops), batch_size):
            batch_ops = all_ops[i : i + batch_size].to(device)
            out = model(batch_ops, compute_control=False)
            z_A = out["z_A_hyp"]
            all_radii.append(z_A.norm(dim=-1).cpu().numpy())
            all_z.append(z_A.cpu().numpy())
            logits = model.decoder_A(out["mu_A"])
            preds = torch.argmax(logits, dim=-1) - 1
            correct = (preds == batch_ops.long()).float().mean(dim=1).cpu().numpy()
            all_correct.append(correct)

    all_radii = np.concatenate(all_radii)
    all_z = np.concatenate(all_z)
    all_correct = np.concatenate(all_correct)
    valuations = TERNARY.valuation(indices).numpy()

    coverage = (all_correct == 1.0).mean()
    hierarchy = spearmanr(valuations, all_radii)[0]

    richness = 0
    for v in range(10):
        mask = valuations == v
        if mask.sum() > 1:
            richness += all_radii[mask].var()
    richness /= 10

    sample_idx = np.random.choice(len(all_z), min(1000, len(all_z)), replace=False)
    z_sample = all_z[sample_idx]
    val_sample = valuations[sample_idx]
    z_dists = np.sqrt(((z_sample[:, None] - z_sample[None, :]) ** 2).sum(-1))
    val_dists = np.abs(val_sample[:, None] - val_sample[None, :]).astype(float)
    triu_idx = np.triu_indices(len(sample_idx), k=1)
    dist_corr = spearmanr(z_dists[triu_idx], val_dists[triu_idx])[0]

    model.train()
    return {
        "coverage": coverage,
        "hierarchy": hierarchy,
        "richness": richness,
        "dist_corr": dist_corr,
        "Q": compute_Q(dist_corr, hierarchy),
        "r_v0": float(all_radii[valuations == 0].mean()),
        "r_v9": float(all_radii[valuations == 9].mean()) if (valuations == 9).any() else np.nan,
    }


def run_experiment(config, all_ops, indices, device, base_ckpt_path):
    """Run experiment with LR scheduling."""
    exp_name = config["name"]
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"Strategy: {config.get('scheduler', 'none')}, patience={config.get('patience', 'N/A')}")
    print(f"{'=' * 60}")

    save_dir = PROJECT_ROOT / f"checkpoints/sweep3_{exp_name}"
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

    if base_ckpt_path.exists():
        try:
            ckpt = load_checkpoint_compat(base_ckpt_path, map_location=device)
            model_state = get_model_state_dict(ckpt)
            model.load_state_dict(model_state, strict=False)
            print("  Loaded checkpoint")
        except Exception as e:
            print(f"  Warning: {e}")

    model = model.to(device)
    model.set_encoder_a_frozen(True)
    model.set_encoder_b_frozen(False)

    dataset = TensorDataset(all_ops, indices)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    with torch.no_grad():
        model.eval()
        original_radii_list = []
        for i in range(0, len(all_ops), 4096):
            batch = all_ops[i : i + 4096].to(device)
            out = model(batch, compute_control=False)
            original_radii_list.append(out["z_A_hyp"].norm(dim=-1))
        original_radii = torch.cat(original_radii_list)
        model.train()

    init_metrics = compute_metrics(model, all_ops, indices, device)
    print(f"  Initial: cov={init_metrics['coverage'] * 100:.1f}%, hier={init_metrics['hierarchy']:.4f}")

    loss_fn = RichHierarchyLoss(
        hierarchy_weight=config.get("hierarchy_weight", 5.0),
        coverage_weight=1.0,
        richness_weight=config.get("richness_weight", 2.0),
        separation_weight=3.0,
    ).to(device)

    epochs = config.get("epochs", 60)
    lr = config.get("lr", 5e-4)
    patience = config.get("patience", 10)

    param_groups = model.get_param_groups(lr)
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

    # Setup scheduler based on config
    scheduler_type = config.get("scheduler", "none")
    scheduler = None

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr / 100)
    elif scheduler_type == "step":
        scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
    elif scheduler_type == "plateau":
        # Note: We use hierarchy (want to minimize, so negate for max mode)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    # Training
    history = []
    best_hier = 0.0
    best_epoch = 0
    no_improve_count = 0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
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

            epoch_loss += losses["total"].item()
            n_batches += 1

        epoch_loss /= n_batches

        # Step scheduler (for non-plateau types)
        if scheduler and scheduler_type != "plateau":
            scheduler.step()

        # Evaluate every 3 epochs
        if epoch % 3 == 0 or epoch == epochs - 1:
            metrics = compute_metrics(model, all_ops, indices, device)
            elapsed = time.time() - start_time

            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:3d}/{epochs} | lr={current_lr:.2e} | "
                f"hier={metrics['hierarchy']:.4f} rich={metrics['richness']:.6f} | {elapsed / 60:.1f}min"
            )

            history.append({"epoch": epoch, "loss": epoch_loss, "lr": current_lr, **metrics})

            # Step plateau scheduler
            if scheduler and scheduler_type == "plateau":
                scheduler.step(metrics["hierarchy"])

            # Track best and early stopping
            if metrics["hierarchy"] < best_hier and metrics["coverage"] > 0.99:
                best_hier = metrics["hierarchy"]
                best_epoch = epoch
                no_improve_count = 0

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "metrics": metrics,
                        "config": config,
                    },
                    save_dir / "best.pt",
                )
                print(f"    *** New best: {best_hier:.4f} ***")
            else:
                no_improve_count += 1

            # Early stopping check
            if config.get("early_stop", False) and no_improve_count >= patience:
                print(f"  Early stopping at epoch {epoch} (no improvement for {patience} evals)")
                break

    final_metrics = compute_metrics(model, all_ops, indices, device)
    elapsed = time.time() - start_time

    result = {
        "name": exp_name,
        "config": config,
        "init_metrics": init_metrics,
        "final_metrics": final_metrics,
        "best_hierarchy": best_hier,
        "best_epoch": best_epoch,
        "history": history,
        "elapsed_minutes": elapsed / 60,
    }

    with open(save_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2, default=float)

    print(f"\n  RESULT: best_hier={best_hier:.4f} @ epoch {best_epoch}, final={final_metrics['hierarchy']:.4f}")
    print(f"  Elapsed: {elapsed / 60:.1f} min")

    return result


def main():
    parser = argparse.ArgumentParser(description="Phase 3: LR scheduling experiments")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/v5_11_homeostasis/best.pt")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Phase 3 experiments
    experiments = {
        # Baseline (no scheduling, no early stop)
        "no_schedule": {
            "scheduler": "none",
            "early_stop": False,
            "epochs": args.epochs,
        },
        # Early stopping only
        "early_stop_p10": {
            "scheduler": "none",
            "early_stop": True,
            "patience": 10,
            "epochs": args.epochs,
        },
        # Cosine annealing
        "cosine": {
            "scheduler": "cosine",
            "early_stop": False,
            "epochs": args.epochs,
        },
        # Step decay
        "step_decay": {
            "scheduler": "step",
            "early_stop": False,
            "epochs": args.epochs,
        },
        # Reduce on plateau + early stop
        "plateau_early": {
            "scheduler": "plateau",
            "early_stop": True,
            "patience": 8,
            "epochs": args.epochs,
        },
        # Aggressive early stop (patience=5)
        "early_stop_p5": {
            "scheduler": "none",
            "early_stop": True,
            "patience": 5,
            "epochs": args.epochs,
        },
    }

    print(f"\nRunning {len(experiments)} experiments")

    # Load data
    print("\n=== Loading Dataset ===")
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
            import traceback

            traceback.print_exc()
            results.append({"name": name, "error": str(e)})

        torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 80)
    print("PHASE 3 SUMMARY: LR Scheduling & Early Stopping")
    print("=" * 80)
    print(f"{'Experiment':<18} {'Scheduler':<10} {'Best Hier':>10} {'@ Epoch':>8} {'Final':>10} {'Time':>6}")
    print("-" * 80)

    for r in results:
        if "error" in r:
            print(f"{r['name']:<18} ERROR")
        else:
            cfg = r["config"]
            print(
                f"{r['name']:<18} {cfg.get('scheduler', 'none'):<10} "
                f"{r['best_hierarchy']:>10.4f} {r['best_epoch']:>8} "
                f"{r['final_metrics']['hierarchy']:>10.4f} {r['elapsed_minutes']:>5.1f}m"
            )

    print("-" * 80)
    print(f"Total time: {total_elapsed / 60:.1f} minutes")

    # Best result
    valid = [r for r in results if "error" not in r]
    if valid:
        best = min(valid, key=lambda x: x["best_hierarchy"])
        print(f"\nBEST: {best['name']} with hierarchy={best['best_hierarchy']:.4f} @ epoch {best['best_epoch']}")

    # Save summary
    summary_path = PROJECT_ROOT / "checkpoints/sweep3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "total_elapsed_minutes": total_elapsed / 60,
                "results": results,
            },
            f,
            indent=2,
            default=float,
        )


if __name__ == "__main__":
    main()
