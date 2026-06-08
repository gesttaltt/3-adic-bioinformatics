#!/usr/bin/env python3
"""Train soft radial projection for near-perfect hierarchy with geometric preservation.

This script trains a learnable radial projection that:
1. Pushes radii toward valuation targets (hierarchy)
2. Preserves angular structure (coverage)
3. Maintains meaningful within-level geometric variation (richness)

The goal is to achieve hierarchy close to -1.0 while keeping the rich
geometric manifold structure that encodes operation-specific semantics.

Usage:
    python src/scripts/epsilon_vae/train_soft_radial.py
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


class SoftRadialProjection(nn.Module):
    """Learnable radial projection that balances hierarchy and geometric richness.

    Instead of hard snapping, this layer learns a smooth mapping from
    encoder radii to target radii while preserving angular information.
    """

    def __init__(
        self,
        latent_dim: int = 16,
        inner_radius: float = 0.05,
        outer_radius: float = 0.95,
        max_valuation: int = 9,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.max_valuation = max_valuation

        # Learnable target radii (initialized to linear spacing)
        targets = torch.tensor(
            [outer_radius - (v / max_valuation) * (outer_radius - inner_radius) for v in range(max_valuation + 1)]
        )
        self.target_radii = nn.Parameter(targets)

        # Learnable radius adjustment network
        # Takes: [current_radius, valuation_embedding, angular_features]
        self.radius_net = nn.Sequential(
            nn.Linear(latent_dim + 11, hidden_dim),  # +11 for valuation one-hot + radius
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output in [0, 1] for interpolation
        )

        # Temperature for soft interpolation (learnable)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        z_hyp: torch.Tensor,
        valuations: torch.Tensor,
        alpha: float = 1.0,  # Interpolation strength (0=original, 1=full adjustment)
    ) -> torch.Tensor:
        """Apply soft radial projection.

        Args:
            z_hyp: Hyperbolic embeddings [batch, dim]
            valuations: Valuation levels [batch]
            alpha: Strength of radial adjustment (for curriculum)

        Returns:
            z_adjusted: Embeddings with adjusted radii
        """
        batch_size = z_hyp.shape[0]
        device = z_hyp.device

        # Get current radius and direction
        radius = z_hyp.norm(dim=-1, keepdim=True)
        direction = z_hyp / (radius + 1e-8)

        # Get target radius for each sample
        target_radius = self.target_radii[valuations.long()].unsqueeze(-1)

        # Build input features for radius network
        val_onehot = torch.zeros(batch_size, 10, device=device)
        val_onehot.scatter_(1, valuations.long().unsqueeze(1), 1)

        features = torch.cat(
            [
                direction,  # Angular information [batch, dim]
                radius,  # Current radius [batch, 1]
                val_onehot,  # Valuation context [batch, 10]
            ],
            dim=-1,
        )

        # Compute adjustment factor
        adjustment = self.radius_net(features)  # [batch, 1] in [0, 1]

        # Soft interpolation between current and target radius
        # adjustment=0 -> keep original, adjustment=1 -> use target
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        soft_adjustment = torch.sigmoid((adjustment - 0.5) * temp)

        new_radius = (1 - soft_adjustment * alpha) * radius + (soft_adjustment * alpha) * target_radius

        # Clamp to valid range
        new_radius = torch.clamp(new_radius, min=0.01, max=0.99)

        return direction * new_radius


class GeometricPreservationLoss(nn.Module):
    """Loss that balances hierarchy, coverage, and geometric richness."""

    def __init__(
        self,
        hierarchy_weight: float = 10.0,
        coverage_weight: float = 1.0,
        richness_weight: float = 0.5,
        separation_weight: float = 2.0,
    ):
        super().__init__()
        self.hierarchy_weight = hierarchy_weight
        self.coverage_weight = coverage_weight
        self.richness_weight = richness_weight
        self.separation_weight = separation_weight

    def forward(
        self,
        z_adjusted: torch.Tensor,
        z_original: torch.Tensor,
        valuations: torch.Tensor,
        target_radii: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> dict:
        device = z_adjusted.device
        radii = z_adjusted.norm(dim=-1)
        original_radii = z_original.norm(dim=-1)

        # === 1. Hierarchy Loss: Push radii toward targets ===
        target_r = target_radii[valuations.long()]
        hierarchy_loss = ((radii - target_r) ** 2).mean()

        # === 2. Coverage Loss: Reconstruction accuracy ===
        coverage_loss = nn.functional.cross_entropy(
            logits.view(-1, 3),
            (targets + 1).long().view(-1),
        )

        # === 3. Richness Loss: Preserve within-level variance ===
        # Encourage some variance within each valuation level
        richness_loss = torch.tensor(0.0, device=device)
        unique_vals = torch.unique(valuations)

        for v in unique_vals:
            mask = valuations == v
            if mask.sum() > 1:
                radii_v = radii[mask]
                orig_radii_v = original_radii[mask]

                # Preserve relative ordering within level
                # (don't collapse everything to single point)
                orig_var = orig_radii_v.var() + 1e-8
                new_var = radii_v.var() + 1e-8

                # Penalize if we collapse variance too much
                # Target: keep ~10-50% of original variance
                variance_ratio = new_var / orig_var
                target_ratio = 0.3  # Keep 30% of original variance
                richness_loss = richness_loss + (variance_ratio - target_ratio).pow(2)

        richness_loss = richness_loss / max(len(unique_vals), 1)

        # === 4. Separation Loss: Maintain level ordering ===
        separation_loss = torch.tensor(0.0, device=device)
        mean_radii = []
        for v in sorted(unique_vals.tolist()):
            mask = valuations == v
            if mask.sum() > 0:
                mean_radii.append((v, radii[mask].mean()))

        for i in range(len(mean_radii) - 1):
            v1, r1 = mean_radii[i]
            v2, r2 = mean_radii[i + 1]
            # v2 > v1, so r2 should be < r1
            margin = 0.05  # Minimum gap between levels
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
            "separation_loss": separation_loss,
        }


class SoftRadialModel(nn.Module):
    """Wrapper combining frozen encoder with learnable soft radial projection."""

    def __init__(
        self,
        base_model: TernaryVAEV5_11_PartialFreeze,
        latent_dim: int = 16,
        inner_radius: float = 0.05,
        outer_radius: float = 0.95,
    ):
        super().__init__()
        self.base_model = base_model
        self.soft_proj = SoftRadialProjection(
            latent_dim=latent_dim,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
        )

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, indices: torch.Tensor, alpha: float = 1.0):
        # Get base embeddings
        out = self.base_model(x, compute_control=False)

        # Get valuations
        valuations = TERNARY.valuation(indices).to(x.device)

        # Apply soft radial projection
        z_adjusted = self.soft_proj(out["z_A_hyp"], valuations, alpha=alpha)

        return {
            "z_A_hyp": out["z_A_hyp"],
            "z_adjusted": z_adjusted,
            "mu_A": out["mu_A"],
            "valuations": valuations,
        }


def compute_metrics(model, all_ops, indices, device, alpha=1.0):
    """Compute comprehensive metrics."""
    model.eval()
    batch_size = 4096

    all_radii_orig = []
    all_radii_adj = []
    all_correct = []

    with torch.no_grad():
        for i in range(0, len(all_ops), batch_size):
            batch_ops = all_ops[i : i + batch_size].to(device)
            batch_idx = indices[i : i + batch_size].to(device)

            out = model(batch_ops, batch_idx, alpha=alpha)

            all_radii_orig.append(out["z_A_hyp"].norm(dim=-1).cpu().numpy())
            all_radii_adj.append(out["z_adjusted"].norm(dim=-1).cpu().numpy())

            logits = model.base_model.decoder_A(out["mu_A"])
            preds = torch.argmax(logits, dim=-1) - 1
            correct = (preds == batch_ops.long()).float().mean(dim=1).cpu().numpy()
            all_correct.append(correct)

    all_radii_orig = np.concatenate(all_radii_orig)
    all_radii_adj = np.concatenate(all_radii_adj)
    all_correct = np.concatenate(all_correct)
    valuations = TERNARY.valuation(indices).numpy()

    coverage = (all_correct == 1.0).mean()
    hierarchy_orig = spearmanr(valuations, all_radii_orig)[0]
    hierarchy_adj = spearmanr(valuations, all_radii_adj)[0]

    # Compute geometric richness (average within-level variance)
    richness_orig = 0
    richness_adj = 0
    for v in range(10):
        mask = valuations == v
        if mask.sum() > 1:
            richness_orig += all_radii_orig[mask].var()
            richness_adj += all_radii_adj[mask].var()
    richness_orig /= 10
    richness_adj /= 10

    model.train()

    return {
        "coverage": coverage,
        "hierarchy_original": hierarchy_orig,
        "hierarchy_adjusted": hierarchy_adj,
        "richness_original": richness_orig,
        "richness_adjusted": richness_adj,
        "richness_ratio": richness_adj / (richness_orig + 1e-10),
        "unique_radii": len(np.unique(np.round(all_radii_adj, 6))),
    }


def main():
    parser = argparse.ArgumentParser(description="Train soft radial projection")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hierarchy_weight", type=float, default=10.0)
    parser.add_argument("--coverage_weight", type=float, default=1.0)
    parser.add_argument("--richness_weight", type=float, default=0.5)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/radial_target/best.pt")
    parser.add_argument("--save_dir", type=str, default="checkpoints/soft_radial")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    save_dir = PROJECT_ROOT / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # === Load Base Model ===
    print("\n=== Loading Base Model ===")
    base_model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=64,
        max_radius=0.99,
        curvature=1.0,
        use_controller=False,
        use_dual_projection=True,
    )

    ckpt_path = PROJECT_ROOT / args.checkpoint
    if ckpt_path.exists():
        print(f"Loading from: {ckpt_path}")
        ckpt = load_checkpoint_compat(ckpt_path, map_location=device)
        model_state = get_model_state_dict(ckpt)
        base_model.load_state_dict(model_state, strict=False)

    base_model = base_model.to(device)

    # === Create Soft Radial Model ===
    print("\n=== Creating Soft Radial Model ===")
    model = SoftRadialModel(base_model, latent_dim=16).to(device)

    trainable_params = sum(p.numel() for p in model.soft_proj.parameters())
    print(f"Trainable parameters: {trainable_params:,}")

    # === Dataset ===
    print("\n=== Loading Dataset ===")
    all_ops_np = generate_all_ternary_operations()
    all_ops = torch.tensor(all_ops_np, dtype=torch.float32)
    indices = torch.arange(len(all_ops))
    dataset = TensorDataset(all_ops, indices)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Dataset: {len(all_ops)} operations")

    # === Loss & Optimizer ===
    loss_fn = GeometricPreservationLoss(
        hierarchy_weight=args.hierarchy_weight,
        coverage_weight=args.coverage_weight,
        richness_weight=args.richness_weight,
    )

    optimizer = optim.AdamW(model.soft_proj.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # === Training with Curriculum ===
    print("\n=== Starting Training ===")
    print(
        f"Weights: hierarchy={args.hierarchy_weight}, coverage={args.coverage_weight}, richness={args.richness_weight}"
    )

    best_score = 0.0

    for epoch in range(args.epochs):
        model.train()

        # Curriculum: gradually increase alpha from 0.5 to 1.0
        alpha = min(0.5 + 0.5 * (epoch / args.epochs), 1.0)

        epoch_losses = {"total": 0, "hierarchy": 0, "coverage": 0, "richness": 0}
        n_batches = 0

        for batch_ops, batch_idx in dataloader:
            batch_ops = batch_ops.to(device)
            batch_idx = batch_idx.to(device)

            out = model(batch_ops, batch_idx, alpha=alpha)
            valuations = out["valuations"]

            logits = model.base_model.decoder_A(out["mu_A"])

            losses = loss_fn(
                out["z_adjusted"],
                out["z_A_hyp"],
                valuations,
                model.soft_proj.target_radii,
                logits,
                batch_ops,
            )

            optimizer.zero_grad()
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.soft_proj.parameters(), 1.0)
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
            metrics = compute_metrics(model, all_ops, indices, device, alpha=1.0)

            # Score: hierarchy (negative, want close to -1) + richness preservation
            score = abs(metrics["hierarchy_adjusted"]) + 0.2 * metrics["richness_ratio"]

            print(f"\nEpoch {epoch}/{args.epochs} (alpha={alpha:.2f})")
            print(f"  Loss: {epoch_losses['total']:.4f}")
            print(f"  Hierarchy: {metrics['hierarchy_original']:.4f} -> {metrics['hierarchy_adjusted']:.4f}")
            print(f"  Richness ratio: {metrics['richness_ratio']:.4f} (target: ~0.3)")
            print(f"  Unique radii: {metrics['unique_radii']}")
            print(f"  Coverage: {metrics['coverage'] * 100:.2f}%")

            if score > best_score:
                best_score = score
                print(f"  [NEW BEST: score={score:.4f}]")

                torch.save(
                    {
                        "epoch": epoch,
                        "soft_proj_state_dict": model.soft_proj.state_dict(),
                        "base_checkpoint": args.checkpoint,
                        "metrics": metrics,
                        "config": vars(args),
                    },
                    save_dir / "best.pt",
                )

            torch.save(
                {
                    "epoch": epoch,
                    "soft_proj_state_dict": model.soft_proj.state_dict(),
                    "metrics": metrics,
                },
                save_dir / "latest.pt",
            )

    print("\n=== Training Complete ===")
    print(f"Best score: {best_score:.4f}")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    main()
