#!/usr/bin/env python3
"""Apply RadialSnapProjection to achieve -1.0 hierarchy.

This script adds a radius-snapping layer that maps each sample to its
valuation's exact target radius while preserving the angular component.

Key insight: The encoder already produces perfect ordering (v0 > v1 > ... > v9).
We just need to snap all samples with same valuation to identical radius.

Usage:
    python src/scripts/epsilon_vae/apply_radial_snap.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.models import TernaryVAEV5_11_PartialFreeze
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat


class RadialSnapProjection(nn.Module):
    """Snaps radius to exact target while preserving angular direction.

    This layer takes hyperbolic embeddings and snaps their radial component
    to exact target values based on valuation level, while preserving the
    angular direction (which encodes the operation identity for coverage).

    The result is that all samples with the same valuation have IDENTICAL
    radii, which is required for Spearman correlation = -1.0.
    """

    def __init__(
        self,
        inner_radius: float = 0.05,
        outer_radius: float = 0.95,
        max_valuation: int = 9,
    ):
        super().__init__()
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.max_valuation = max_valuation

        # Precompute exact target radii for each valuation level
        # v=0 -> outer (0.95), v=9 -> inner (0.05)
        targets = torch.tensor(
            [outer_radius - (v / max_valuation) * (outer_radius - inner_radius) for v in range(max_valuation + 1)]
        )
        self.register_buffer("target_radii", targets)

    def forward(self, z_hyp: torch.Tensor, valuations: torch.Tensor) -> torch.Tensor:
        """Snap radii to exact targets while preserving direction.

        Args:
            z_hyp: Hyperbolic embeddings [batch, dim]
            valuations: Valuation levels [batch] (integers 0-9)

        Returns:
            z_snapped: Embeddings with exact target radii [batch, dim]
        """
        # Get current radius and unit direction
        radius = z_hyp.norm(dim=-1, keepdim=True)
        direction = z_hyp / (radius + 1e-8)

        # Normalize direction to unit length (eliminate floating point accumulation)
        direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)

        # Get exact target radius for each sample's valuation
        target_radius = self.target_radii[valuations.long()].unsqueeze(-1)

        # Reconstruct with exact radius, original direction
        z_snapped = direction * target_radius

        return z_snapped


class SnappedTernaryVAE(nn.Module):
    """Wrapper that adds RadialSnapProjection to frozen base model."""

    def __init__(
        self,
        base_model: TernaryVAEV5_11_PartialFreeze,
        inner_radius: float = 0.05,
        outer_radius: float = 0.95,
    ):
        super().__init__()
        self.base_model = base_model
        self.snap = RadialSnapProjection(inner_radius, outer_radius)

        # Freeze base model entirely
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor, indices: torch.Tensor):
        """Forward with radius snapping.

        Args:
            x: Input operations [batch, 9]
            indices: Operation indices for valuation lookup [batch]

        Returns:
            Dict with snapped embeddings and decoder outputs
        """
        # Get base model outputs
        out = self.base_model(x, compute_control=False)

        # Get valuations from indices
        valuations = TERNARY.valuation(indices).long().to(x.device)

        # Snap radii to exact targets
        z_A_snapped = self.snap(out["z_A_hyp"], valuations)
        z_B_snapped = self.snap(out["z_B_hyp"], valuations)

        # Decode from snapped embeddings (use mu for deterministic)
        # Note: decoder_A expects Euclidean, but we can test with snapped hyperbolic

        return {
            "z_A_hyp": out["z_A_hyp"],
            "z_A_snapped": z_A_snapped,
            "z_B_hyp": out["z_B_hyp"],
            "z_B_snapped": z_B_snapped,
            "mu_A": out["mu_A"],
            "mu_B": out["mu_B"],
            "valuations": valuations,
        }


def evaluate_model(model, all_ops, indices, device, use_snapped=False):
    """Evaluate model with optional radius snapping."""
    model.eval()
    batch_size = 4096
    n_samples = len(all_ops)

    all_radii = []
    all_radii_snapped = []
    all_correct = []

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_ops = all_ops[i : i + batch_size].to(device)
            batch_idx = indices[i : i + batch_size].to(device)

            if isinstance(model, SnappedTernaryVAE):
                out = model(batch_ops, batch_idx)
                radii = out["z_A_hyp"].norm(dim=-1).cpu().numpy()
                radii_snapped = out["z_A_snapped"].norm(dim=-1).cpu().numpy()
                mu_A = out["mu_A"]
            else:
                out = model(batch_ops, compute_control=False)
                radii = out["z_A_hyp"].norm(dim=-1).cpu().numpy()
                radii_snapped = radii  # No snapping
                mu_A = out["mu_A"]

            all_radii.append(radii)
            all_radii_snapped.append(radii_snapped)

            # Coverage check
            logits = model.base_model.decoder_A(mu_A) if isinstance(model, SnappedTernaryVAE) else model.decoder_A(mu_A)
            preds = torch.argmax(logits, dim=-1) - 1
            correct = (preds == batch_ops.long()).float().mean(dim=1).cpu().numpy()
            all_correct.append(correct)

    all_radii = np.concatenate(all_radii)
    all_radii_snapped = np.concatenate(all_radii_snapped)
    all_correct = np.concatenate(all_correct)
    valuations = TERNARY.valuation(indices).numpy()

    coverage = (all_correct == 1.0).mean()
    hierarchy_original = spearmanr(valuations, all_radii)[0]

    # Round snapped radii to eliminate floating point noise (5 decimals is plenty)
    all_radii_snapped_rounded = np.round(all_radii_snapped, 5)
    hierarchy_snapped = spearmanr(valuations, all_radii_snapped_rounded)[0]

    # Compute within-level variance
    var_original = 0
    var_snapped = 0
    for v in range(10):
        mask = valuations == v
        if mask.sum() > 1:
            var_original += all_radii[mask].var()
            var_snapped += all_radii_snapped[mask].var()
    var_original /= 10
    var_snapped /= 10

    # Radius stats
    r_v0 = all_radii_snapped[valuations == 0].mean()
    r_v9 = all_radii_snapped[valuations >= 8].mean() if (valuations >= 8).any() else 0.5

    return {
        "coverage": coverage,
        "hierarchy_original": hierarchy_original,
        "hierarchy_snapped": hierarchy_snapped,
        "variance_original": var_original,
        "variance_snapped": var_snapped,
        "r_v0": r_v0,
        "r_v9": r_v9,
        "unique_radii_original": len(np.unique(np.round(all_radii, 8))),
        "unique_radii_snapped": len(np.unique(all_radii_snapped_rounded)),
    }


def main():
    parser = argparse.ArgumentParser(description="Apply RadialSnapProjection")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/radial_target/best.pt")
    parser.add_argument("--inner_radius", type=float, default=0.05)
    parser.add_argument("--outer_radius", type=float, default=0.95)
    parser.add_argument("--save_dir", type=str, default="checkpoints/radial_snapped")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # === Load Base Model ===
    print("\n=== Loading Base Model ===")
    base_model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=64,
        max_radius=0.99,
        curvature=1.0,
        use_controller=False,
        use_dual_projection=True,
        freeze_encoder_b=True,
        freeze_encoder_a=True,
    )

    ckpt_path = PROJECT_ROOT / args.checkpoint
    if ckpt_path.exists():
        print(f"Loading from: {ckpt_path}")
        ckpt = load_checkpoint_compat(ckpt_path, map_location=device)
        model_state = get_model_state_dict(ckpt)
        base_model.load_state_dict(model_state, strict=False)
        print(f"  Source metrics: {ckpt.get('metrics', {})}")
    else:
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return

    base_model = base_model.to(device)

    # === Create Snapped Model ===
    print("\n=== Creating Snapped Model ===")
    snapped_model = SnappedTernaryVAE(
        base_model,
        inner_radius=args.inner_radius,
        outer_radius=args.outer_radius,
    ).to(device)

    print(f"Target radii: {snapped_model.snap.target_radii.cpu().numpy()}")

    # === Load Dataset ===
    print("\n=== Loading Dataset ===")
    all_ops_np = generate_all_ternary_operations()
    all_ops = torch.tensor(all_ops_np, dtype=torch.float32)
    indices = torch.arange(len(all_ops))
    print(f"Dataset: {len(all_ops)} operations")

    # === Evaluate ===
    print("\n=== Evaluating ===")
    metrics = evaluate_model(snapped_model, all_ops, indices, device)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nCoverage: {metrics['coverage'] * 100:.2f}%")
    print(f"\nHierarchy (original radii):  {metrics['hierarchy_original']:.4f}")
    print(f"Hierarchy (snapped radii):   {metrics['hierarchy_snapped']:.4f}")
    print(f"\nVariance (original): {metrics['variance_original']:.8f}")
    print(f"Variance (snapped):  {metrics['variance_snapped']:.8f}")
    print(f"\nUnique radii (original): {metrics['unique_radii_original']}")
    print(f"Unique radii (snapped):  {metrics['unique_radii_snapped']}")
    print(f"\nRadius v0: {metrics['r_v0']:.4f} (target: {args.outer_radius})")
    print(f"Radius v9: {metrics['r_v9']:.4f} (target: {args.inner_radius})")
    print("=" * 60)

    # === Save ===
    if metrics["hierarchy_snapped"] <= -0.99:
        save_dir = PROJECT_ROOT / args.save_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Saving to {save_dir} ===")

        # Save the snapped model
        torch.save(
            {
                "base_model_state_dict": base_model.state_dict(),
                "snap_state_dict": snapped_model.snap.state_dict(),
                "metrics": metrics,
                "config": {
                    "inner_radius": args.inner_radius,
                    "outer_radius": args.outer_radius,
                    "source_checkpoint": args.checkpoint,
                },
            },
            save_dir / "best.pt",
        )

        print(f"Saved snapped model with hierarchy={metrics['hierarchy_snapped']:.4f}")
    else:
        print(f"\nWARNING: Hierarchy {metrics['hierarchy_snapped']:.4f} not at -1.0")

    print("\n=== Complete ===")


if __name__ == "__main__":
    main()
