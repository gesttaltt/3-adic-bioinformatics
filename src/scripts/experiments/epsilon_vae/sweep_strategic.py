#!/usr/bin/env python3
"""Strategic parameter sweep for 3-4 hour training session.

Runs targeted experiments to answer specific architectural questions:
1. Is curvature=1.0 optimal? (test 0.5, 1.0, 2.0)
2. Is latent_dim=16 optimal? (test 8, 16, 32)

Each experiment runs 40 epochs (~15-20 min) for quick signal.
Total: 6 experiments × 20 min = ~2 hours

Usage:
    python src/scripts/experiments/epsilon_vae/sweep_strategic.py
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
        min_richness_ratio: float = 0.5,
    ):
        super().__init__()
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.hierarchy_weight = hierarchy_weight
        self.coverage_weight = coverage_weight
        self.richness_weight = richness_weight
        self.separation_weight = separation_weight
        self.min_richness_ratio = min_richness_ratio
        self.max_valuation = 9

        self.register_buffer(
            "target_radii",
            torch.tensor([outer_radius - (v / self.max_valuation) * (outer_radius - inner_radius) for v in range(10)]),
        )

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
                mean_r = radii[mask].mean()
                target_r = self.target_radii[v]
                hierarchy_loss = hierarchy_loss + (mean_r - target_r) ** 2
        hierarchy_loss = hierarchy_loss / len(unique_vals)

        # Coverage loss
        coverage_loss = nn.functional.cross_entropy(
            logits.view(-1, 3),
            (targets + 1).long().view(-1),
        )

        # Richness loss
        richness_loss = torch.tensor(0.0, device=device)
        if original_radii is not None:
            for v in unique_vals:
                mask = valuations == v
                if mask.sum() > 1:
                    new_var = radii[mask].var()
                    orig_var = original_radii[mask].var() + 1e-8
                    ratio = new_var / orig_var
                    if ratio < self.min_richness_ratio:
                        richness_loss = richness_loss + (self.min_richness_ratio - ratio) ** 2
            richness_loss = richness_loss / max(len(unique_vals), 1)

        # Separation loss
        separation_loss = torch.tensor(0.0, device=device)
        mean_radii_list = []
        for v in sorted(unique_vals.tolist()):
            mask = valuations == v
            if mask.sum() > 0:
                mean_radii_list.append((v, radii[mask].mean()))
        for i in range(len(mean_radii_list) - 1):
            v1, r1 = mean_radii_list[i]
            v2, r2 = mean_radii_list[i + 1]
            margin = 0.01
            violation = torch.relu(r2 - r1 + margin)
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

    # Richness
    richness = 0
    for v in range(10):
        mask = valuations == v
        if mask.sum() > 1:
            richness += all_radii[mask].var()
    richness /= 10

    # Distance correlation (sampled)
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
    """Run a single experiment with given config."""
    exp_name = config["name"]
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"Config: {config}")
    print(f"{'=' * 60}")

    save_dir = PROJECT_ROOT / f"checkpoints/sweep_{exp_name}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create model with experiment config
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=config.get("latent_dim", 16),
        hidden_dim=config.get("hidden_dim", 64),
        max_radius=0.99,
        curvature=config.get("curvature", 1.0),
        use_controller=False,
        use_dual_projection=True,
        freeze_encoder_b=False,
        encoder_b_lr_scale=0.1,
        encoder_a_lr_scale=0.05,
    )

    # Load base checkpoint (only if dimensions match)
    if base_ckpt_path.exists() and config.get("latent_dim", 16) == 16:
        try:
            ckpt = load_checkpoint_compat(base_ckpt_path, map_location=device)
            model_state = get_model_state_dict(ckpt)
            model.load_state_dict(model_state, strict=False)
            print("  Loaded base checkpoint")
        except Exception as e:
            print(f"  Warning: Could not load checkpoint: {e}")
    else:
        print(f"  Training from scratch (latent_dim={config.get('latent_dim', 16)})")

    model = model.to(device)
    model.set_encoder_a_frozen(True)
    model.set_encoder_b_frozen(False)

    # Dataset
    dataset = TensorDataset(all_ops, indices)
    dataloader = DataLoader(dataset, batch_size=config.get("batch_size", 512), shuffle=True)

    # Get original radii
    with torch.no_grad():
        model.eval()
        original_radii_list = []
        for i in range(0, len(all_ops), 4096):
            batch = all_ops[i : i + 4096].to(device)
            out = model(batch, compute_control=False)
            original_radii_list.append(out["z_A_hyp"].norm(dim=-1))
        original_radii = torch.cat(original_radii_list)
        model.train()

    # Initial metrics
    init_metrics = compute_metrics(model, all_ops, indices, device)
    print(
        f"  Initial: cov={init_metrics['coverage'] * 100:.1f}%, hier={init_metrics['hierarchy']:.4f}, rich={init_metrics['richness']:.6f}"
    )

    # Loss and optimizer
    loss_fn = RichHierarchyLoss(
        hierarchy_weight=config.get("hierarchy_weight", 5.0),
        coverage_weight=config.get("coverage_weight", 1.0),
        richness_weight=config.get("richness_weight", 2.0),
        separation_weight=config.get("separation_weight", 3.0),
    ).to(device)

    lr = config.get("lr", 5e-4)
    epochs = config.get("epochs", 40)

    # Training loop
    history = []
    best_hier = 0.0
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        param_groups = model.get_param_groups(lr)
        optimizer = optim.AdamW(param_groups, weight_decay=1e-4)

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

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == epochs - 1:
            metrics = compute_metrics(model, all_ops, indices, device)
            elapsed = time.time() - start_time

            print(
                f"  Epoch {epoch:3d}/{epochs} | loss={epoch_loss:.4f} | "
                f"cov={metrics['coverage'] * 100:.1f}% hier={metrics['hierarchy']:.4f} "
                f"rich={metrics['richness']:.6f} | {elapsed / 60:.1f}min"
            )

            history.append({"epoch": epoch, "loss": epoch_loss, **metrics})

            if metrics["hierarchy"] < best_hier and metrics["coverage"] > 0.99:
                best_hier = metrics["hierarchy"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "metrics": metrics,
                        "config": config,
                    },
                    save_dir / "best.pt",
                )

    # Final metrics
    final_metrics = compute_metrics(model, all_ops, indices, device)
    elapsed = time.time() - start_time

    result = {
        "name": exp_name,
        "config": config,
        "init_metrics": init_metrics,
        "final_metrics": final_metrics,
        "history": history,
        "elapsed_minutes": elapsed / 60,
        "best_hierarchy": best_hier,
    }

    # Save results
    with open(save_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2, default=float)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "metrics": final_metrics,
            "config": config,
        },
        save_dir / "final.pt",
    )

    print(
        f"\n  RESULT: hier={final_metrics['hierarchy']:.4f}, rich={final_metrics['richness']:.6f}, "
        f"cov={final_metrics['coverage'] * 100:.1f}% ({elapsed / 60:.1f} min)"
    )

    return result


def main():
    parser = argparse.ArgumentParser(description="Strategic parameter sweep")
    parser.add_argument("--epochs", type=int, default=40, help="Epochs per experiment")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/v5_11_homeostasis/best.pt")
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help="Which experiments: all, curvature, latent_dim, or comma-separated list",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Epochs per experiment: {args.epochs}")

    # Define experiments
    all_experiments = {
        # Curvature sweep (fixed latent_dim=16)
        "curv_0.5": {"curvature": 0.5, "latent_dim": 16, "epochs": args.epochs},
        "curv_1.0": {"curvature": 1.0, "latent_dim": 16, "epochs": args.epochs},
        "curv_2.0": {"curvature": 2.0, "latent_dim": 16, "epochs": args.epochs},
        # Latent dimension sweep (fixed curvature=1.0)
        "latent_8": {"latent_dim": 8, "curvature": 1.0, "epochs": args.epochs},
        "latent_16": {"latent_dim": 16, "curvature": 1.0, "epochs": args.epochs},
        "latent_32": {"latent_dim": 32, "curvature": 1.0, "epochs": args.epochs},
    }

    # Select experiments
    if args.experiments == "all":
        experiments = all_experiments
    elif args.experiments == "curvature":
        experiments = {k: v for k, v in all_experiments.items() if k.startswith("curv_")}
    elif args.experiments == "latent_dim":
        experiments = {k: v for k, v in all_experiments.items() if k.startswith("latent_")}
    else:
        exp_names = [e.strip() for e in args.experiments.split(",")]
        experiments = {k: v for k, v in all_experiments.items() if k in exp_names}

    print(f"\nRunning {len(experiments)} experiments: {list(experiments.keys())}")
    print(f"Estimated time: {len(experiments) * 15} - {len(experiments) * 25} minutes")

    # Load data once
    print("\n=== Loading Dataset ===")
    all_ops_np = generate_all_ternary_operations()
    all_ops = torch.tensor(all_ops_np, dtype=torch.float32)
    indices = torch.arange(len(all_ops))
    print(f"Loaded {len(all_ops)} operations")

    base_ckpt = PROJECT_ROOT / args.base_checkpoint

    # Run experiments
    results = []
    total_start = time.time()

    for name, config in experiments.items():
        config["name"] = name
        try:
            result = run_experiment(config, all_ops, indices, device, base_ckpt)
            results.append(result)
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            results.append({"name": name, "error": str(e)})

        # Clear GPU memory between experiments
        torch.cuda.empty_cache()

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "=" * 70)
    print("SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<15} {'Hierarchy':>10} {'Richness':>12} {'Coverage':>10} {'Time':>8}")
    print("-" * 70)

    for r in results:
        if "error" in r:
            print(f"{r['name']:<15} ERROR: {r['error']}")
        else:
            fm = r["final_metrics"]
            print(
                f"{r['name']:<15} {fm['hierarchy']:>10.4f} {fm['richness']:>12.6f} "
                f"{fm['coverage'] * 100:>9.1f}% {r['elapsed_minutes']:>7.1f}m"
            )

    print("-" * 70)
    print(f"Total time: {total_elapsed / 60:.1f} minutes")

    # Save summary
    summary_path = PROJECT_ROOT / "checkpoints/sweep_summary.json"
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
    print(f"\nSummary saved to: {summary_path}")

    # Best result
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        best = min(valid_results, key=lambda x: x["final_metrics"]["hierarchy"])
        print(f"\nBEST: {best['name']} with hierarchy={best['final_metrics']['hierarchy']:.4f}")


if __name__ == "__main__":
    main()
