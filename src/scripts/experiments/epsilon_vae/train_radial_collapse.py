#!/usr/bin/env python3
"""Train for -1.0 hierarchy by collapsing intra-level variance.

Key insight: The -0.8321 ceiling is caused by within-valuation-level variance.
Even though all valuation levels are perfectly ordered (v0 > v1 > ... > v9),
samples within each level have slightly different radii.

Spearman correlation requires ALL samples with same valuation to have
IDENTICAL radii to achieve -1.0.

Solution: Add intra-level variance penalty that collapses each valuation
level to a single radial shell.

Usage:
    python src/scripts/epsilon_vae/train_radial_collapse.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR
from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.models import TernaryVAEV5_11_PartialFreeze
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat


class CollapsedRadialLoss(nn.Module):
    """Loss that collapses each valuation level to a single radial shell."""

    def __init__(
        self,
        inner_radius: float = 0.05,
        outer_radius: float = 0.95,
        radial_weight: float = 10.0,
        variance_weight: float = 50.0,  # High weight to collapse variance
        coverage_weight: float = 0.1,
    ):
        super().__init__()
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.radial_weight = radial_weight
        self.variance_weight = variance_weight
        self.coverage_weight = coverage_weight
        self.max_valuation = 9

        # Precompute target radii for each valuation level
        self.register_buffer(
            "target_radii_by_val",
            torch.tensor([outer_radius - (v / self.max_valuation) * (outer_radius - inner_radius) for v in range(10)]),
        )

    def forward(
        self,
        z_hyp: torch.Tensor,
        indices: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        device = z_hyp.device
        radii = z_hyp.norm(dim=-1)
        valuations = TERNARY.valuation(indices).long().to(device)

        # === 1. Radial Target Loss ===
        # Push each sample toward its valuation's target radius
        target_radii = self.target_radii_by_val[valuations]
        radial_loss = ((radii - target_radii) ** 2).mean()

        # === 2. Intra-Level Variance Loss (KEY TO -1.0) ===
        # Force all samples with same valuation to have identical radius
        variance_loss = torch.tensor(0.0, device=device)
        unique_vals = torch.unique(valuations)

        for v in unique_vals:
            mask = valuations == v
            if mask.sum() > 1:
                radii_v = radii[mask]
                mean_r = radii_v.mean()
                # Variance within this valuation level
                variance_loss = variance_loss + ((radii_v - mean_r) ** 2).mean()

        variance_loss = variance_loss / len(unique_vals)

        # === 3. Coverage Regularization ===
        coverage_loss = nn.functional.cross_entropy(
            logits.view(-1, 3),
            (targets + 1).long().view(-1),
        )

        total = (
            self.radial_weight * radial_loss
            + self.variance_weight * variance_loss
            + self.coverage_weight * coverage_loss
        )

        return {
            "total": total,
            "radial_loss": radial_loss,
            "variance_loss": variance_loss,
            "coverage_loss": coverage_loss,
        }


def compute_metrics(model, all_ops, indices, device):
    """Compute comprehensive metrics including within-level variance."""
    model.eval()
    batch_size = 4096
    n_samples = len(all_ops)

    all_radii = []
    all_correct = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_ops = all_ops[i : i + batch_size].to(device)

            out = model(batch_ops, compute_control=False)
            z_A = out["z_A_hyp"]
            mu_A = out["mu_A"]

            radii = z_A.norm(dim=-1).cpu().numpy()
            all_radii.append(radii)

            logits = model.decoder_A(mu_A)
            preds = torch.argmax(logits, dim=-1) - 1
            correct = (preds == batch_ops.long()).float().mean(dim=1).cpu().numpy()
            all_correct.append(correct)

    all_radii = np.concatenate(all_radii)
    all_correct = np.concatenate(all_correct)
    valuations = TERNARY.valuation(indices).numpy()

    coverage = (all_correct == 1.0).mean()
    hierarchy = spearmanr(valuations, all_radii)[0]

    # Compute within-level variance
    total_variance = 0
    for v in range(10):
        mask = valuations == v
        if mask.sum() > 1:
            total_variance += all_radii[mask].var()
    avg_variance = total_variance / 10

    r_v0 = all_radii[valuations == 0].mean()
    r_v9 = all_radii[valuations >= 8].mean() if (valuations >= 8).any() else 0.5

    model.train()

    return {
        "coverage": coverage,
        "hierarchy": hierarchy,
        "r_v0": r_v0,
        "r_v9": r_v9,
        "avg_intra_variance": avg_variance,
        "mean_radius": all_radii.mean(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train with collapsed radial variance")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--inner_radius", type=float, default=0.05)
    parser.add_argument("--outer_radius", type=float, default=0.95)
    parser.add_argument("--radial_weight", type=float, default=10.0)
    parser.add_argument("--variance_weight", type=float, default=100.0)
    parser.add_argument("--coverage_weight", type=float, default=0.05)
    parser.add_argument("--start_checkpoint", type=str, default=str(CHECKPOINTS_DIR / "radial_target" / "best.pt"))
    parser.add_argument("--save_dir", type=str, default=str(CHECKPOINTS_DIR / "radial_collapse"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # === Create Model ===
    print("\n=== Creating Model ===")
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=64,
        max_radius=0.99,
        curvature=1.0,
        use_controller=False,
        use_dual_projection=True,
        freeze_encoder_b=False,
        encoder_b_lr_scale=0.5,
        encoder_a_lr_scale=0.1,
    )

    start_path = PROJECT_ROOT / args.start_checkpoint
    if start_path.exists():
        print(f"Loading from: {start_path}")
        ckpt = load_checkpoint_compat(start_path, map_location=device)
        model_state = get_model_state_dict(ckpt)
        model.load_state_dict(model_state, strict=False)

    model.set_encoder_a_frozen(False)
    model = model.to(device)
    print(f"Freeze state: {model.get_freeze_state_summary()}")

    # === Dataset ===
    print("\n=== Loading Dataset ===")
    all_ops_np = generate_all_ternary_operations()
    all_ops = torch.tensor(all_ops_np, dtype=torch.float32)
    indices = torch.arange(len(all_ops))
    dataset = TensorDataset(all_ops, indices)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Dataset: {len(all_ops)} operations")

    # === Loss & Optimizer ===
    loss_fn = CollapsedRadialLoss(
        inner_radius=args.inner_radius,
        outer_radius=args.outer_radius,
        radial_weight=args.radial_weight,
        variance_weight=args.variance_weight,
        coverage_weight=args.coverage_weight,
    ).to(device)

    param_groups = model.get_param_groups(args.lr)
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # === Training ===
    print("\n=== Starting Training ===")
    print("Target: collapse intra-level variance to achieve -1.0 hierarchy")
    print(f"Weights: radial={args.radial_weight}, variance={args.variance_weight}, coverage={args.coverage_weight}")

    best_hierarchy = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {"total": 0, "radial": 0, "variance": 0, "coverage": 0}
        n_batches = 0

        for batch_ops, batch_idx in dataloader:
            batch_ops = batch_ops.to(device)
            batch_idx = batch_idx.to(device)

            out = model(batch_ops, compute_control=False)
            z_A = out["z_A_hyp"]
            mu_A = out["mu_A"]
            logits = model.decoder_A(mu_A)

            losses = loss_fn(z_A, batch_idx, logits, batch_ops)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses["total"] += losses["total"].item()
            epoch_losses["radial"] += losses["radial_loss"].item()
            epoch_losses["variance"] += losses["variance_loss"].item()
            epoch_losses["coverage"] += losses["coverage_loss"].item()
            n_batches += 1

        scheduler.step()

        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            metrics = compute_metrics(model, all_ops, indices, device)

            print(f"\nEpoch {epoch}/{args.epochs}")
            print(
                f"  Loss: {epoch_losses['total']:.6f} (rad:{epoch_losses['radial']:.6f}, var:{epoch_losses['variance']:.6f}, cov:{epoch_losses['coverage']:.6f})"
            )
            print(f"  Hierarchy: {metrics['hierarchy']:.4f} (target: -1.0)")
            print(f"  Coverage: {metrics['coverage'] * 100:.2f}%")
            print(f"  Radius: v0={metrics['r_v0']:.4f} -> v9={metrics['r_v9']:.4f}")
            print(f"  Avg Intra-Variance: {metrics['avg_intra_variance']:.8f}")

            is_best = metrics["hierarchy"] < best_hierarchy
            if is_best:
                best_hierarchy = metrics["hierarchy"]
                print(f"  [NEW BEST HIERARCHY: {best_hierarchy:.4f}]")

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "metrics": metrics,
                        "config": vars(args),
                    },
                    save_dir / "best.pt",
                )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                },
                save_dir / "latest.pt",
            )

    print("\n=== Training Complete ===")
    print(f"Best Hierarchy: {best_hierarchy:.4f}")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    main()
