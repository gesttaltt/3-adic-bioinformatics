#!/usr/bin/env python3
"""Train V5.11 with Epsilon-StateNet Coupled Controller.

This training script couples the Epsilon-VAE with a StateNet controller
to learn dynamics that maintain 100% coverage while pushing hierarchy.

The key innovation: instead of hand-tuned homeostasis, the controller
learns from the Epsilon-VAE's understanding of the weight space to make
optimal training decisions.

Architecture:
    TernaryVAE weights → EpsilonStateNet → Training control signals
                                         ↓
                              [lr_scales, freeze_probs, loss_weights]

Training alternates between:
1. Training TernaryVAE with control signals
2. Updating EpsilonStateNet based on achieved metrics

Usage:
    python src/scripts/epsilon_vae/train_epsilon_coupled.py
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR
from src.core import TERNARY
from src.losses import GlobalRankLoss, PAdicGeodesicLoss, RadialHierarchyLoss
from src.models import TernaryVAEV5_11_PartialFreeze
from src.models.epsilon_statenet import (
    EpsilonStateNetLoss,
    create_epsilon_statenet,
)
from src.models.epsilon_vae import extract_key_weights


def parse_args():
    parser = argparse.ArgumentParser(description="Train V5.11 with Epsilon-StateNet Controller")

    # Basic training
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")

    # Checkpoints
    parser.add_argument(
        "--v5_5_checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "v5_5" / "latest.pt"),
        help="Path to v5.5 base checkpoint",
    )
    parser.add_argument(
        "--epsilon_checkpoint", type=str, default=None, help="Path to pretrained Epsilon-VAE (optional)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(CHECKPOINTS_DIR / "v5_11_epsilon_coupled"),
        help="Directory to save checkpoints",
    )

    # Controller settings
    parser.add_argument("--controller_lr", type=float, default=1e-4, help="Learning rate for EpsilonStateNet")
    parser.add_argument("--controller_update_freq", type=int, default=5, help="Update controller every N batches")
    parser.add_argument("--coverage_target", type=float, default=0.995, help="Target coverage for controller")
    parser.add_argument("--hierarchy_target", type=float, default=-0.85, help="Target hierarchy for controller")

    # Model settings
    parser.add_argument("--dual_projection", action="store_true", default=True, help="Use dual projection")
    parser.add_argument("--projection_hidden_dim", type=int, default=64, help="Projection hidden dimension")
    parser.add_argument("--projection_layers", type=int, default=1, help="Number of projection layers")
    parser.add_argument("--projection_dropout", type=float, default=0.1, help="Projection dropout")

    # Loss weights (base values, controller can adjust)
    parser.add_argument("--radial_weight", type=float, default=2.0, help="Base radial loss weight")
    parser.add_argument("--margin_weight", type=float, default=1.0, help="Base margin weight")
    parser.add_argument("--rank_loss_weight", type=float, default=1.0, help="Base rank loss weight")

    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    return parser.parse_args()


def extract_model_weights(model: nn.Module) -> list:
    """Extract key weights from model for Epsilon encoding."""
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    return extract_key_weights(state_dict)


def compute_metrics(model, all_ops, device) -> dict:
    """Compute training metrics."""
    model.eval()
    with torch.no_grad():
        out = model(all_ops.to(device), compute_control=False)
        z_A = out["z_A_hyp"]
        z_B = out["z_B_hyp"]

        # Convert {0, 1, 2} to {-1, 0, 1} for TERNARY.from_ternary
        ternary_ops = (all_ops - 1).long()
        indices = TERNARY.from_ternary(ternary_ops)
        # Use proper 3-adic valuation: v_3(n) = max k such that 3^k divides n
        valuations = TERNARY.valuation(indices).float().to(device)
        radii_A = torch.norm(z_A, dim=1).clamp(max=0.999)
        radii_B = torch.norm(z_B, dim=1).clamp(max=0.999)

        # Radial hierarchy: compute mean radius per valuation level
        # With 3-adic valuation, distribution is skewed (most have v=0)
        # Use slope of mean radius vs valuation as hierarchy metric
        mean_radii_by_val = []
        val_levels = []
        for v in range(10):  # v_3 can be 0-9
            mask = valuations == v
            if mask.sum() > 0:
                mean_radii_by_val.append(radii_A[mask].mean().item())
                val_levels.append(v)

        # Compute correlation using level means (more robust to skew)
        if len(val_levels) >= 2:
            val_t = torch.tensor(val_levels, dtype=torch.float32)
            rad_t = torch.tensor(mean_radii_by_val, dtype=torch.float32)
            corr_A = torch.corrcoef(torch.stack([rad_t, val_t]))[0, 1].item()
        else:
            corr_A = 0.0

        # Same for B
        mean_radii_by_val_B = []
        for v in range(10):
            mask = valuations == v
            if mask.sum() > 0:
                mean_radii_by_val_B.append(radii_B[mask].mean().item())
        if len(mean_radii_by_val_B) >= 2:
            rad_t_B = torch.tensor(mean_radii_by_val_B, dtype=torch.float32)
            corr_B = torch.corrcoef(torch.stack([rad_t_B, val_t]))[0, 1].item()
        else:
            corr_B = 0.0

        # Coverage (unique embeddings)
        import numpy as np
        from scipy.spatial.distance import pdist

        z_np = z_A.cpu().numpy()
        sample_idx = np.random.choice(len(z_np), min(2000, len(z_np)), replace=False)
        dists = pdist(z_np[sample_idx])
        coverage = (dists > 0.01).mean()

        # Distance correlation
        pairwise_dists = torch.cdist(z_A[:1000], z_A[:1000])
        val_diffs = (valuations[:1000].unsqueeze(1) - valuations[:1000].unsqueeze(0)).abs()
        dist_corr = torch.corrcoef(torch.stack([pairwise_dists.flatten(), val_diffs.float().flatten()]))[0, 1].item()

        # Radius by valuation (3-adic: v=0 for indices not divisible by 3, v=9 for index 0)
        # Low valuation (v=0) should be at outer radius, high valuation (v=9) at inner
        v0_mask = valuations == 0  # Not divisible by 3 → should be outer (large radius)
        v_high_mask = valuations >= 8  # High valuation → should be inner (small radius)
        r_v0 = radii_A[v0_mask].mean().item() if v0_mask.any() else 0.5
        r_v9 = radii_A[v_high_mask].mean().item() if v_high_mask.any() else 0.5

        # Q metric
        Q = dist_corr + 1.5 * abs(corr_A)

    model.train()
    return {
        "coverage": coverage,
        "hierarchy_A": corr_A,
        "hierarchy_B": corr_B,
        "dist_corr_A": dist_corr,
        "r_v0": r_v0,
        "r_v9": r_v9,
        "Q": Q,
    }


def metrics_to_tensor(metrics: dict, device) -> torch.Tensor:
    """Convert metrics dict to tensor for StateNet input."""
    return torch.tensor(
        [
            metrics["coverage"],
            metrics["hierarchy_A"],
            metrics["hierarchy_B"],
            metrics["dist_corr_A"],
            0.0,  # dist_corr_B placeholder
            metrics["r_v0"],
            metrics["r_v9"],
            metrics["Q"],
        ],
        dtype=torch.float32,
        device=device,
    )


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # =====================
    # Create TernaryVAE model
    # =====================
    print("\n=== Creating TernaryVAE V5.11 ===")
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=args.projection_hidden_dim,
        max_radius=0.95,
        curvature=1.0,
        use_controller=False,
        use_dual_projection=args.dual_projection,
        n_projection_layers=args.projection_layers,
        projection_dropout=args.projection_dropout,
        freeze_encoder_b=False,
        encoder_b_lr_scale=0.1,
        encoder_a_lr_scale=0.05,
    )

    # Load v5.5 checkpoint
    if Path(args.v5_5_checkpoint).exists():
        model.load_v5_5_checkpoint(args.v5_5_checkpoint, device=str(device))
        print(f"Loaded base checkpoint: {args.v5_5_checkpoint}")

    model = model.to(device)

    # =====================
    # Create EpsilonStateNet controller
    # =====================
    print("\n=== Creating EpsilonStateNet Controller ===")
    controller = create_epsilon_statenet(
        pretrained_epsilon_path=args.epsilon_checkpoint,
        embed_dim=64,
        epsilon_dim=32,
        metric_dim=8,
        hidden_dim=64,
        n_heads=4,
        n_components=4,
    )
    controller = controller.to(device)

    controller_loss_fn = EpsilonStateNetLoss(
        coverage_target=args.coverage_target,
        hierarchy_target=args.hierarchy_target,
        coverage_weight=10.0,
        hierarchy_weight=1.0,
    )

    # =====================
    # Create dataset
    # =====================
    print("\n=== Loading Dataset ===")
    # Generate ops in {0, 1, 2} directly
    all_ops_list = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    for e in range(3):
                        for f in range(3):
                            for g in range(3):
                                for h in range(3):
                                    for i in range(3):
                                        all_ops_list.append([a, b, c, d, e, f, g, h, i])
    all_ops = torch.tensor(all_ops_list, dtype=torch.float32)
    indices = torch.arange(len(all_ops))
    dataset = TensorDataset(all_ops, indices)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Dataset size: {len(all_ops)}")

    # =====================
    # Create losses
    # =====================
    geo_loss_fn = PAdicGeodesicLoss(curvature=1.0)
    radial_loss_fn = RadialHierarchyLoss(
        inner_radius=0.10,
        outer_radius=0.85,
        max_valuation=9,  # 3-adic valuation: max is 9 for index 0 (3^9 divides 0)
        margin_weight=args.margin_weight,
    )
    rank_loss_fn = GlobalRankLoss(n_pairs=2000, temperature=0.1)

    # =====================
    # Create optimizers
    # =====================
    # Model optimizer with parameter groups
    param_groups = [
        {"params": model.projection.parameters(), "lr": args.lr, "name": "projection"},
        {"params": model.encoder_B.parameters(), "lr": args.lr * 0.1, "name": "encoder_B"},
    ]
    model_optimizer = optim.AdamW(param_groups, weight_decay=0.001)
    model_scheduler = optim.lr_scheduler.CosineAnnealingLR(model_optimizer, T_max=args.epochs)

    # Controller optimizer
    controller_optimizer = optim.Adam(controller.parameters(), lr=args.controller_lr)

    # =====================
    # Training state
    # =====================
    best_composite = float("-inf")
    best_coverage = 0.0
    best_hierarchy = 0.0

    print("\n=== Starting Training ===")
    print(f"Coverage target: {args.coverage_target}")
    print(f"Hierarchy target: {args.hierarchy_target}")

    for epoch in range(args.epochs):
        model.train()
        controller.train()

        epoch_losses = {
            "geo": 0,
            "rad": 0,
            "rank": 0,
            "total": 0,
            "controller": 0,
        }
        n_batches = 0

        for batch_idx, (ops, idx) in enumerate(dataloader):
            ops = ops.to(device)
            idx = idx.to(device)

            # =====================
            # Get control signals from EpsilonStateNet
            # =====================
            if batch_idx % args.controller_update_freq == 0:
                # Extract current weights and metrics
                weight_blocks = extract_model_weights(model)
                weight_blocks = [w.to(device) for w in weight_blocks]

                with torch.no_grad():
                    current_metrics = compute_metrics(model, all_ops, device)
                    metrics_tensor = metrics_to_tensor(current_metrics, device)

                # Get control signals
                controls = controller(weight_blocks, metrics_tensor)

                # Apply control signals
                lr_scales = controls["lr_scales"].detach()
                freeze_probs = controls["freeze_probs"].detach()
                loss_weights = controls["loss_weights"].detach()

                # Update learning rates
                for i, pg in enumerate(model_optimizer.param_groups):
                    if i < len(lr_scales):
                        pg["lr"] = args.lr * lr_scales[i].item()

                # Update freeze states (soft freeze via gradient scaling)
                # encoder_A always frozen, encoder_B controlled
                encoder_b_grad_scale = 1.0 - freeze_probs[1].item()

                # Update loss weights
                radial_weight = args.radial_weight * loss_weights[0].item()
                args.margin_weight * loss_weights[1].item()
                rank_weight = args.rank_loss_weight * loss_weights[2].item()

            # =====================
            # Forward pass
            # =====================
            out = model(ops, compute_control=False)
            z_A = out["z_A_hyp"]
            z_B = out["z_B_hyp"]

            # =====================
            # Compute losses
            # =====================
            # All loss functions take (z_hyp, batch_indices) and compute valuations internally
            # using TERNARY.valuation(batch_indices)

            # PAdicGeodesicLoss takes (z_hyp, batch_indices)
            geo_loss_A, _ = geo_loss_fn(z_A, idx)
            geo_loss_B, _ = geo_loss_fn(z_B, idx)
            geo_loss = geo_loss_A + geo_loss_B

            # RadialHierarchyLoss takes (z_hyp, batch_indices)
            rad_loss_A, _ = radial_loss_fn(z_A, idx)
            rad_loss_B, _ = radial_loss_fn(z_B, idx)
            rad_loss = (rad_loss_A + rad_loss_B) * radial_weight

            # GlobalRankLoss takes (z_hyp, batch_indices)
            rank_loss, _ = rank_loss_fn(z_A, idx)
            rank_loss = rank_loss * rank_weight

            total_loss = geo_loss + rad_loss + rank_loss

            # =====================
            # Backward pass with gradient scaling
            # =====================
            model_optimizer.zero_grad()
            total_loss.backward()

            # Scale encoder_B gradients based on controller decision
            if encoder_b_grad_scale < 1.0:
                for param in model.encoder_B.parameters():
                    if param.grad is not None:
                        param.grad *= encoder_b_grad_scale

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            model_optimizer.step()

            # =====================
            # Update controller periodically
            # =====================
            if batch_idx % args.controller_update_freq == 0 and batch_idx > 0:
                controller_optimizer.zero_grad()

                # Recompute controls with gradients
                controls = controller(weight_blocks, metrics_tensor)

                # Compute controller loss based on achieved metrics
                ctrl_loss = controller_loss_fn(
                    controls,
                    current_metrics["coverage"],
                    current_metrics["hierarchy_A"],
                )

                ctrl_loss["total"].backward()
                controller_optimizer.step()

                epoch_losses["controller"] += ctrl_loss["total"].item()

            # Track losses
            epoch_losses["geo"] += geo_loss.item()
            epoch_losses["rad"] += rad_loss.item()
            epoch_losses["rank"] += rank_loss.item()
            epoch_losses["total"] += total_loss.item()
            n_batches += 1

        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        model_scheduler.step()

        # =====================
        # Compute epoch metrics
        # =====================
        metrics = compute_metrics(model, all_ops, device)

        # Composite score
        composite = metrics["coverage"] + abs(metrics["hierarchy_A"])

        # Print progress
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(
                f"  Loss: {epoch_losses['total']:.4f} (geo: {epoch_losses['geo']:.4f}, "
                f"rad: {epoch_losses['rad']:.4f}, rank: {epoch_losses['rank']:.4f})"
            )
            print(f"  Coverage: {metrics['coverage'] * 100:.2f}%")
            print(f"  Hierarchy: A={metrics['hierarchy_A']:.4f}, B={metrics['hierarchy_B']:.4f}")
            print(f"  Radius: v0={metrics['r_v0']:.4f}, v9={metrics['r_v9']:.4f}")
            print(f"  Q: {metrics['Q']:.4f}")
            print(f"  Controller Loss: {epoch_losses['controller']:.4f}")

            # Show current control signals
            if "lr_scales" in dir():
                print(f"  LR Scales: {lr_scales.cpu().numpy()}")
                print(f"  Freeze Probs: {freeze_probs.cpu().numpy()}")

        # =====================
        # Save checkpoints
        # =====================
        is_best = composite > best_composite
        if is_best:
            best_composite = composite
            best_coverage = metrics["coverage"]
            best_hierarchy = metrics["hierarchy_A"]
            print(
                f"  [NEW BEST] Composite: {composite:.4f} "
                f"(coverage: {best_coverage * 100:.2f}%, hierarchy: {best_hierarchy:.4f})"
            )

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "controller_state": controller.state_dict(),
            "model_optimizer_state": model_optimizer.state_dict(),
            "controller_optimizer_state": controller_optimizer.state_dict(),
            "metrics": metrics,
            "config": vars(args),
            "composite_score": composite,
        }

        if is_best:
            torch.save(checkpoint, save_dir / "best.pt")

        if epoch % 20 == 0:
            torch.save(checkpoint, save_dir / f"epoch_{epoch}.pt")

        torch.save(checkpoint, save_dir / "latest.pt")

    print("\n=== Training Complete ===")
    print(f"Best Composite: {best_composite:.4f}")
    print(f"Best Coverage: {best_coverage * 100:.2f}%")
    print(f"Best Hierarchy: {best_hierarchy:.4f}")
    print(f"Checkpoints saved to: {save_dir}")


if __name__ == "__main__":
    main()
