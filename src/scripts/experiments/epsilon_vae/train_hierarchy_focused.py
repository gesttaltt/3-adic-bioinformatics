#!/usr/bin/env python3
"""Train for maximum hierarchy (-1.0 target) while maintaining coverage connection.

This script trains with unfrozen encoders to push hierarchy as deep as possible
toward the theoretical -1.0 limit, while continuously regularizing for coverage
so the embedding space never fully disconnects from the coverage objective.

Strategy:
1. Start from v5_11_homeostasis (99.9% coverage, -0.75 hierarchy)
2. Unfreeze encoders progressively
3. Primary loss: push radial correlation toward -1.0
4. Secondary loss: coverage regularization (never disappears)
5. Even if coverage drops, the gradient signal keeps the space "connected"

Usage:
    python src/scripts/epsilon_vae/train_hierarchy_focused.py
"""

import argparse
import json
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
from src.models import TernaryVAEV5_11_PartialFreeze
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat


class HierarchyFocusedLoss(nn.Module):
    """Loss function targeting -1.0 hierarchy with coverage regularization."""

    def __init__(
        self,
        hierarchy_target: float = -1.0,
        coverage_weight: float = 0.1,
        hierarchy_weight: float = 10.0,
        radial_separation_weight: float = 5.0,
    ):
        super().__init__()
        self.hierarchy_target = hierarchy_target
        self.coverage_weight = coverage_weight
        self.hierarchy_weight = hierarchy_weight
        self.radial_separation_weight = radial_separation_weight

    def forward(
        self,
        z_hyp: torch.Tensor,
        indices: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        """
        Compute hierarchy-focused loss.

        Args:
            z_hyp: Hyperbolic embeddings [batch, dim]
            indices: Operation indices [batch]
            logits: Decoder logits [batch, 9, 3]
            targets: Target operations [batch, 9]
        """
        device = z_hyp.device

        # Compute radii and valuations
        radii = z_hyp.norm(dim=-1)
        valuations = TERNARY.valuation(indices).float().to(device)

        # === PRIMARY: Hierarchy Loss ===
        # Push correlation toward -1.0
        # Use differentiable approximation of correlation
        radii_centered = radii - radii.mean()
        val_centered = valuations - valuations.mean()

        # Pearson correlation (differentiable)
        cov = (radii_centered * val_centered).mean()
        std_r = radii_centered.std() + 1e-6
        std_v = val_centered.std() + 1e-6
        correlation = cov / (std_r * std_v)

        # Loss: push toward -1.0
        hierarchy_loss = (correlation - self.hierarchy_target).pow(2)

        # === SECONDARY: Radial Separation Loss ===
        # Enforce that high valuation -> low radius, low valuation -> high radius
        # Group by valuation and compute mean radius per group
        separation_loss = torch.tensor(0.0, device=device)

        unique_vals = torch.unique(valuations)
        if len(unique_vals) > 1:
            mean_radii = []
            for v in sorted(unique_vals.tolist()):
                mask = valuations == v
                if mask.sum() > 0:
                    mean_radii.append((v, radii[mask].mean()))

            # Penalize violations of monotonic decrease
            for i in range(len(mean_radii) - 1):
                v1, r1 = mean_radii[i]
                v2, r2 = mean_radii[i + 1]
                # v2 > v1, so r2 should be < r1
                # Penalize if r2 >= r1 (wrong order)
                violation = torch.relu(r2 - r1 + 0.02)  # 0.02 margin
                separation_loss = separation_loss + violation

        # === TERTIARY: Coverage Regularization ===
        # Cross-entropy for reconstruction - keeps embedding connected to coverage
        coverage_loss = nn.functional.cross_entropy(
            logits.view(-1, 3),
            (targets + 1).long().view(-1),  # Convert to {0,1,2}
        )

        # Total loss
        total = (
            self.hierarchy_weight * hierarchy_loss
            + self.radial_separation_weight * separation_loss
            + self.coverage_weight * coverage_loss
        )

        return {
            "total": total,
            "hierarchy_loss": hierarchy_loss,
            "separation_loss": separation_loss,
            "coverage_loss": coverage_loss,
            "correlation": correlation.detach(),
        }


def compute_metrics(model, all_ops, indices, device):
    """Compute comprehensive metrics."""
    model.eval()

    # Process in batches to avoid OOM
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

            # Coverage
            logits = model.decoder_A(mu_A)
            preds = torch.argmax(logits, dim=-1) - 1
            correct = (preds == batch_ops.long()).float().mean(dim=1).cpu().numpy()
            all_correct.append(correct)

    all_radii = np.concatenate(all_radii)
    all_correct = np.concatenate(all_correct)
    valuations = TERNARY.valuation(indices).numpy()

    coverage = (all_correct == 1.0).mean()
    hierarchy = spearmanr(valuations, all_radii)[0]

    # Radius by valuation
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
    parser = argparse.ArgumentParser(description="Train hierarchy-focused model")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hierarchy_weight", type=float, default=10.0)
    parser.add_argument("--coverage_weight", type=float, default=0.5)
    parser.add_argument("--separation_weight", type=float, default=5.0)
    parser.add_argument("--start_checkpoint", type=str, default=str(CHECKPOINTS_DIR / "v5_11_homeostasis" / "best.pt"))
    parser.add_argument("--save_dir", type=str, default=str(CHECKPOINTS_DIR / "hierarchy_focused"))
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
        max_radius=0.95,
        curvature=1.0,
        use_controller=False,
        use_dual_projection=True,
        freeze_encoder_b=False,  # Unfreeze for hierarchy learning
        encoder_b_lr_scale=0.5,
        encoder_a_lr_scale=0.1,  # Also unfreeze encoder_A progressively
    )

    # Load starting checkpoint
    start_path = Path(args.start_checkpoint)
    if start_path.exists():
        print(f"Loading from: {start_path}")
        ckpt = load_checkpoint_compat(start_path, map_location=device)
        model_state = get_model_state_dict(ckpt)
        model.load_state_dict(model_state, strict=False)
        print(f"  Starting epoch: {ckpt.get('epoch', 'N/A')}")
        print(f"  Starting coverage: {ckpt.get('metrics', {}).get('coverage', 'N/A')}")

    # Unfreeze encoder_A for hierarchy learning
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
    loss_fn = HierarchyFocusedLoss(
        hierarchy_target=-1.0,
        hierarchy_weight=args.hierarchy_weight,
        coverage_weight=args.coverage_weight,
        radial_separation_weight=args.separation_weight,
    )

    # Different learning rates for different components
    param_groups = model.get_param_groups(args.lr)
    optimizer = optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # === Training ===
    print("\n=== Starting Training ===")
    print("Target: hierarchy=-1.0, maintain coverage connection")
    print(
        f"Weights: hierarchy={args.hierarchy_weight}, coverage={args.coverage_weight}, separation={args.separation_weight}"
    )

    best_hierarchy = 0.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = {"total": 0, "hierarchy": 0, "separation": 0, "coverage": 0}
        epoch_corr = 0
        n_batches = 0

        for batch_ops, batch_idx in dataloader:
            batch_ops = batch_ops.to(device)
            batch_idx = batch_idx.to(device)

            # Forward
            out = model(batch_ops, compute_control=False)
            z_A = out["z_A_hyp"]
            mu_A = out["mu_A"]

            # Reconstruction for coverage regularization
            logits = model.decoder_A(mu_A)

            # Compute loss
            losses = loss_fn(z_A, batch_idx, logits, batch_ops)

            # Backward
            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track
            epoch_losses["total"] += losses["total"].item()
            epoch_losses["hierarchy"] += losses["hierarchy_loss"].item()
            epoch_losses["separation"] += losses["separation_loss"].item()
            epoch_losses["coverage"] += losses["coverage_loss"].item()
            epoch_corr += losses["correlation"].item()
            n_batches += 1

        scheduler.step()

        # Average
        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        epoch_corr /= n_batches

        # Compute full metrics every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            metrics = compute_metrics(model, all_ops, indices, device)

            # Composite: prioritize hierarchy but track coverage
            composite = abs(metrics["hierarchy"]) + 0.5 * metrics["coverage"]

            print(f"\nEpoch {epoch}/{args.epochs}")
            print(
                f"  Loss: {epoch_losses['total']:.4f} (hier:{epoch_losses['hierarchy']:.4f}, sep:{epoch_losses['separation']:.4f}, cov:{epoch_losses['coverage']:.4f})"
            )
            print(f"  Hierarchy: {metrics['hierarchy']:.4f} (target: -1.0)")
            print(f"  Coverage: {metrics['coverage'] * 100:.2f}%")
            print(f"  Radius: v0={metrics['r_v0']:.4f} -> v9={metrics['r_v9']:.4f}")
            print(f"  Composite: {composite:.4f}")

            history.append(
                {
                    "epoch": epoch,
                    "hierarchy": metrics["hierarchy"],
                    "coverage": metrics["coverage"],
                    "r_v0": metrics["r_v0"],
                    "r_v9": metrics["r_v9"],
                    "losses": epoch_losses,
                }
            )

            # Save best
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

            # Save latest
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "config": vars(args),
                },
                save_dir / "latest.pt",
            )

        # Save periodic checkpoints
        if epoch % 50 == 0 and epoch > 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics if "metrics" in dir() else {},
                },
                save_dir / f"epoch_{epoch}.pt",
            )

    # Save history
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("\n=== Training Complete ===")
    print(f"Best Hierarchy: {best_hierarchy:.4f}")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    main()
