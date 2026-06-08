#!/usr/bin/env python3
"""Train balanced radial model: near-perfect hierarchy WITH geometric richness.

Key insight: The frozen encoder produces perfect ordering but -0.8321 ceiling.
To break this ceiling while preserving richness, we need to:
1. Unfreeze the projection layer (fast adaptation)
2. Keep encoders frozen (preserve learned features)
3. Train with balanced loss: hierarchy + richness + coverage

Target: ~-0.95 hierarchy with ~30% richness preservation

Usage:
    python src/scripts/epsilon_vae/train_balanced_radial.py
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

from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.models import TernaryVAEV5_11_PartialFreeze
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat


class BalancedRadialLoss(nn.Module):
    """Loss balancing hierarchy, coverage, and geometric richness."""

    def __init__(
        self,
        inner_radius: float = 0.05,
        outer_radius: float = 0.95,
        hierarchy_weight: float = 10.0,
        coverage_weight: float = 0.5,
        richness_weight: float = 1.0,
        target_richness_ratio: float = 0.3,  # Keep 30% of original variance
    ):
        super().__init__()
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.hierarchy_weight = hierarchy_weight
        self.coverage_weight = coverage_weight
        self.richness_weight = richness_weight
        self.target_richness_ratio = target_richness_ratio
        self.max_valuation = 9

        # Precompute target radii
        self.register_buffer(
            "target_radii",
            torch.tensor([outer_radius - (v / self.max_valuation) * (outer_radius - inner_radius) for v in range(10)]),
        )

    def forward(
        self,
        z_hyp: torch.Tensor,
        indices: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        original_radii: torch.Tensor,
    ) -> dict:
        device = z_hyp.device
        radii = z_hyp.norm(dim=-1)
        valuations = TERNARY.valuation(indices).long().to(device)

        # === 1. Hierarchy Loss: Push toward target radii ===
        target_r = self.target_radii[valuations]
        hierarchy_loss = ((radii - target_r) ** 2).mean()

        # === 2. Coverage Loss ===
        coverage_loss = nn.functional.cross_entropy(
            logits.view(-1, 3),
            (targets + 1).long().view(-1),
        )

        # === 3. Richness Preservation Loss ===
        # Keep some within-level variance (don't fully collapse)
        richness_loss = torch.tensor(0.0, device=device)
        unique_vals = torch.unique(valuations)

        for v in unique_vals:
            mask = valuations == v
            if mask.sum() > 1:
                radii_v = radii[mask]
                orig_radii_v = original_radii[mask]

                new_var = radii_v.var()
                orig_var = orig_radii_v.var() + 1e-8

                # Target: keep target_richness_ratio of original variance
                # Penalize both too much collapse AND too much variance
                ratio = new_var / orig_var
                target = self.target_richness_ratio

                # Asymmetric penalty: harder on collapse than on excess
                if ratio < target:
                    # Too collapsed - strong penalty
                    richness_loss = richness_loss + (target - ratio) ** 2 * 2
                else:
                    # Too much variance - weaker penalty
                    richness_loss = richness_loss + (ratio - target) ** 2 * 0.5

        richness_loss = richness_loss / max(len(unique_vals), 1)

        total = (
            self.hierarchy_weight * hierarchy_loss
            + self.coverage_weight * coverage_loss
            + self.richness_weight * richness_loss
        )

        return {
            "total": total,
            "hierarchy_loss": hierarchy_loss,
            "coverage_loss": coverage_loss,
            "richness_loss": richness_loss,
        }


def compute_metrics(model, all_ops, indices, device):
    """Compute comprehensive metrics including richness."""
    model.eval()
    batch_size = 4096

    all_radii = []
    all_correct = []

    # First pass to get original radii (frozen encoder output)
    with torch.no_grad():
        # Store original projection state
        model.set_encoder_a_frozen(True)
        model.set_encoder_b_frozen(True)

        for i in range(0, len(all_ops), batch_size):
            batch_ops = all_ops[i : i + batch_size].to(device)

            out = model(batch_ops, compute_control=False)
            radii = out["z_A_hyp"].norm(dim=-1).cpu().numpy()
            all_radii.append(radii)

            logits = model.decoder_A(out["mu_A"])
            preds = torch.argmax(logits, dim=-1) - 1
            correct = (preds == batch_ops.long()).float().mean(dim=1).cpu().numpy()
            all_correct.append(correct)

    all_radii = np.concatenate(all_radii)
    all_correct = np.concatenate(all_correct)
    valuations = TERNARY.valuation(indices).numpy()

    coverage = (all_correct == 1.0).mean()
    hierarchy = spearmanr(valuations, all_radii)[0]

    # Compute richness (within-level variance)
    total_var = 0
    for v in range(10):
        mask = valuations == v
        if mask.sum() > 1:
            total_var += all_radii[mask].var()
    avg_var = total_var / 10

    # Radius by level
    r_v0 = all_radii[valuations == 0].mean()
    r_v9 = all_radii[valuations == 9].mean() if (valuations == 9).any() else np.nan

    model.train()

    return {
        "coverage": coverage,
        "hierarchy": hierarchy,
        "richness": avg_var,
        "r_v0": r_v0,
        "r_v9": r_v9,
        "unique_radii": len(np.unique(np.round(all_radii, 6))),
        "mean_radius": all_radii.mean(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train balanced radial model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hierarchy_weight", type=float, default=15.0)
    parser.add_argument("--coverage_weight", type=float, default=0.5)
    parser.add_argument("--richness_weight", type=float, default=2.0)
    parser.add_argument("--target_richness", type=float, default=0.3)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/radial_target/best.pt")
    parser.add_argument("--save_dir", type=str, default="checkpoints/balanced_radial")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = PROJECT_ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # === Load Model ===
    print("\n=== Loading Model ===")
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=64,
        max_radius=0.99,
        curvature=1.0,
        use_controller=False,
        use_dual_projection=True,
        freeze_encoder_b=True,  # Freeze encoder B
        freeze_encoder_a=True,  # Freeze encoder A - only projection trainable
        encoder_b_lr_scale=0.0,
        encoder_a_lr_scale=0.0,
    )

    ckpt_path = PROJECT_ROOT / args.checkpoint
    if ckpt_path.exists():
        print(f"Loading from: {ckpt_path}")
        ckpt = load_checkpoint_compat(ckpt_path, map_location=device)
        model_state = get_model_state_dict(ckpt)
        model.load_state_dict(model_state, strict=False)
        print(f"  Source metrics: {ckpt.get('metrics', {})}")

    model = model.to(device)
    model.set_encoder_a_frozen(True)
    model.set_encoder_b_frozen(True)

    # Only projection is trainable
    trainable_params = sum(p.numel() for p in model.projection.parameters() if p.requires_grad)
    print(f"Trainable parameters (projection only): {trainable_params:,}")

    # === Get Original Radii (for richness reference) ===
    print("\n=== Computing Original Radii ===")
    all_ops_np = generate_all_ternary_operations()
    all_ops = torch.tensor(all_ops_np, dtype=torch.float32)
    indices = torch.arange(len(all_ops))

    with torch.no_grad():
        original_radii_list = []
        for i in range(0, len(all_ops), 4096):
            batch = all_ops[i : i + 4096].to(device)
            out = model(batch, compute_control=False)
            original_radii_list.append(out["z_A_hyp"].norm(dim=-1))
        original_radii_full = torch.cat(original_radii_list)

    print(f"Original radii range: {original_radii_full.min():.4f} - {original_radii_full.max():.4f}")

    # === Dataset ===
    dataset = TensorDataset(all_ops, indices)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # === Loss & Optimizer ===
    loss_fn = BalancedRadialLoss(
        hierarchy_weight=args.hierarchy_weight,
        coverage_weight=args.coverage_weight,
        richness_weight=args.richness_weight,
        target_richness_ratio=args.target_richness,
    ).to(device)

    # Only optimize projection
    optimizer = optim.AdamW(model.projection.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # === Training ===
    print("\n=== Starting Training ===")
    print(
        f"Weights: hierarchy={args.hierarchy_weight}, coverage={args.coverage_weight}, richness={args.richness_weight}"
    )
    print(f"Target richness ratio: {args.target_richness}")

    best_hierarchy = 0.0
    initial_richness = None

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {"total": 0, "hierarchy": 0, "coverage": 0, "richness": 0}
        n_batches = 0

        for batch_ops, batch_idx in dataloader:
            batch_ops = batch_ops.to(device)
            batch_idx = batch_idx.to(device)

            # Get original radii for this batch
            orig_radii_batch = original_radii_full[batch_idx]

            out = model(batch_ops, compute_control=False)
            z_A = out["z_A_hyp"]
            mu_A = out["mu_A"]
            logits = model.decoder_A(mu_A)

            losses = loss_fn(z_A, batch_idx, logits, batch_ops, orig_radii_batch)

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.projection.parameters(), 1.0)
            optimizer.step()

            epoch_losses["total"] += losses["total"].item()
            epoch_losses["hierarchy"] += losses["hierarchy_loss"].item()
            epoch_losses["coverage"] += losses["coverage_loss"].item()
            epoch_losses["richness"] += losses["richness_loss"].item()
            n_batches += 1

        scheduler.step()

        for k in epoch_losses:
            epoch_losses[k] /= n_batches

        if epoch % 5 == 0 or epoch == args.epochs - 1:
            metrics = compute_metrics(model, all_ops, indices, device)

            if initial_richness is None:
                initial_richness = metrics["richness"]

            richness_ratio = metrics["richness"] / (initial_richness + 1e-10)

            print(f"\nEpoch {epoch}/{args.epochs}")
            print(
                f"  Loss: {epoch_losses['total']:.4f} (h:{epoch_losses['hierarchy']:.4f}, c:{epoch_losses['coverage']:.4f}, r:{epoch_losses['richness']:.4f})"
            )
            print(f"  Hierarchy: {metrics['hierarchy']:.4f} (target: ~-0.95)")
            print(f"  Coverage: {metrics['coverage'] * 100:.2f}%")
            print(f"  Richness ratio: {richness_ratio:.4f} (target: {args.target_richness})")
            print(f"  Radius: v0={metrics['r_v0']:.4f} -> v9={metrics['r_v9']:.4f}")
            print(f"  Unique radii: {metrics['unique_radii']}")

            is_best = metrics["hierarchy"] < best_hierarchy
            if is_best:
                best_hierarchy = metrics["hierarchy"]
                print(f"  [NEW BEST: {best_hierarchy:.4f}]")

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "metrics": metrics,
                        "richness_ratio": richness_ratio,
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
