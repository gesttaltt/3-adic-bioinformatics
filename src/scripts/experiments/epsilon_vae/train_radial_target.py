#!/usr/bin/env python3
"""Train for maximum hierarchy using RadialStratificationLoss with explicit radius targets.

This script uses the existing RadialStratificationLoss which provides direct
per-sample gradients toward exact target radii, rather than correlation-based loss.

Key difference from train_hierarchy_focused.py:
- Uses RadialStratificationLoss(inner_radius=0.05, outer_radius=0.95)
- Direct MSE/smooth_l1 toward target radii instead of correlation
- Stronger gradient signal per sample

Target: v0->0.95 (outer), v9->0.05 (inner), linear interpolation between.

Usage:
    python src/scripts/epsilon_vae/train_radial_target.py
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

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR
from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.losses import RadialStratificationLoss
from src.models import TernaryVAEV5_11_PartialFreeze
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat


class CombinedRadialLoss(nn.Module):
    """Combines RadialStratificationLoss with coverage regularization."""

    def __init__(
        self,
        inner_radius: float = 0.05,
        outer_radius: float = 0.95,
        radial_weight: float = 10.0,
        coverage_weight: float = 0.1,
    ):
        super().__init__()
        self.radial_weight = radial_weight
        self.coverage_weight = coverage_weight

        # Use existing RadialStratificationLoss with aggressive targets
        self.radial_loss = RadialStratificationLoss(
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            max_valuation=9,
            valuation_weighting=True,
            loss_type="mse",  # Sharp gradients for precise targeting
        )

    def forward(
        self,
        z_hyp: torch.Tensor,
        indices: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """Compute combined loss."""
        # Radial stratification loss (primary)
        radial_loss, radial_metrics = self.radial_loss(z_hyp, indices, return_metrics=True)

        # Coverage regularization (secondary)
        coverage_loss = nn.functional.cross_entropy(
            logits.view(-1, 3),
            (targets + 1).long().view(-1),
        )

        total = self.radial_weight * radial_loss + self.coverage_weight * coverage_loss

        return {
            "total": total,
            "radial_loss": radial_loss,
            "coverage_loss": coverage_loss,
            "radial_metrics": radial_metrics,
        }


def compute_metrics(model, all_ops, indices, device):
    """Compute comprehensive metrics."""
    model.eval()
    batch_size = 4096
    n_samples = len(all_ops)

    all_radii = []
    all_correct = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_ops = all_ops[i : i + batch_size].to(device)
            indices[i : i + batch_size].to(device)

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

    r_v0 = all_radii[valuations == 0].mean()
    r_v9 = all_radii[valuations >= 8].mean() if (valuations >= 8).any() else 0.5

    model.train()

    return {
        "coverage": coverage,
        "hierarchy": hierarchy,
        "r_v0": r_v0,
        "r_v9": r_v9,
        "mean_radius": all_radii.mean(),
        "std_radius": all_radii.std(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train with RadialStratificationLoss")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--inner_radius", type=float, default=0.05)
    parser.add_argument("--outer_radius", type=float, default=0.95)
    parser.add_argument("--radial_weight", type=float, default=10.0)
    parser.add_argument("--coverage_weight", type=float, default=0.1)
    parser.add_argument("--start_checkpoint", type=str, default=str(CHECKPOINTS_DIR / "hierarchy_extreme" / "best.pt"))
    parser.add_argument("--save_dir", type=str, default=str(CHECKPOINTS_DIR / "radial_target"))
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
        max_radius=0.99,  # Allow full radial range
        curvature=1.0,
        use_controller=False,
        use_dual_projection=True,
        freeze_encoder_b=False,
        encoder_b_lr_scale=0.5,
        encoder_a_lr_scale=0.1,
    )

    # Load starting checkpoint
    start_path = PROJECT_ROOT / args.start_checkpoint
    if start_path.exists():
        print(f"Loading from: {start_path}")
        ckpt = load_checkpoint_compat(start_path, map_location=device)
        model_state = get_model_state_dict(ckpt)
        model.load_state_dict(model_state, strict=False)
        print(f"  Starting metrics: {ckpt.get('metrics', {})}")

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
    loss_fn = CombinedRadialLoss(
        inner_radius=args.inner_radius,
        outer_radius=args.outer_radius,
        radial_weight=args.radial_weight,
        coverage_weight=args.coverage_weight,
    )

    param_groups = model.get_param_groups(args.lr)
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # === Training ===
    print("\n=== Starting Training ===")
    print(f"Target radii: v0->{args.outer_radius}, v9->{args.inner_radius}")
    print(f"Weights: radial={args.radial_weight}, coverage={args.coverage_weight}")

    best_hierarchy = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {"total": 0, "radial": 0, "coverage": 0}
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
            epoch_losses["coverage"] += losses["coverage_loss"].item()
            n_batches += 1

        scheduler.step()

        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        # Compute metrics every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            metrics = compute_metrics(model, all_ops, indices, device)

            print(f"\nEpoch {epoch}/{args.epochs}")
            print(
                f"  Loss: {epoch_losses['total']:.4f} (rad:{epoch_losses['radial']:.4f}, cov:{epoch_losses['coverage']:.4f})"
            )
            print(f"  Hierarchy: {metrics['hierarchy']:.4f} (target: -1.0)")
            print(f"  Coverage: {metrics['coverage'] * 100:.2f}%")
            print(
                f"  Radius: v0={metrics['r_v0']:.4f} (target:{args.outer_radius}) -> v9={metrics['r_v9']:.4f} (target:{args.inner_radius})"
            )

            is_best = metrics["hierarchy"] < best_hierarchy
            if is_best:
                best_hierarchy = metrics["hierarchy"]
                print(f"  [NEW BEST HIERARCHY: {best_hierarchy:.4f}]")

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
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
                    "config": vars(args),
                },
                save_dir / "latest.pt",
            )

        if epoch % 50 == 0 and epoch > 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": compute_metrics(model, all_ops, indices, device) if epoch % 5 != 0 else metrics,
                },
                save_dir / f"epoch_{epoch}.pt",
            )

    print("\n=== Training Complete ===")
    print(f"Best Hierarchy: {best_hierarchy:.4f}")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    main()
