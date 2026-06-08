#!/usr/bin/env python3
"""Investigate checkpoint loading inconsistency.

The stability test showed huge variance in initial hierarchy (-0.78 to +0.75).
This script investigates why.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.models import TernaryVAEV5_11_PartialFreeze
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat


def compute_hierarchy(model, all_ops, indices, device):
    """Compute hierarchy correlation."""
    model.eval()
    all_radii = []
    with torch.no_grad():
        for i in range(0, len(all_ops), 4096):
            batch = all_ops[i : i + 4096].to(device)
            out = model(batch, compute_control=False)
            all_radii.append(out["z_A_hyp"].norm(dim=-1).cpu().numpy())
    all_radii = np.concatenate(all_radii)
    valuations = TERNARY.valuation(indices).numpy()
    return spearmanr(valuations, all_radii)[0]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    all_ops = torch.tensor(generate_all_ternary_operations(), dtype=torch.float32)
    indices = torch.arange(len(all_ops))

    ckpt_path = PROJECT_ROOT / "checkpoints/v5_11_homeostasis/best.pt"
    print(f"\nCheckpoint: {ckpt_path}")

    # Load checkpoint once and inspect
    ckpt = load_checkpoint_compat(ckpt_path, map_location=device)
    print(f"\nCheckpoint keys: {list(ckpt.keys())}")

    if "metrics" in ckpt:
        print(f"Stored metrics: {ckpt['metrics']}")

    # Test 1: Load same checkpoint multiple times
    print("\n=== Test 1: Load checkpoint 5 times ===")
    hierarchies = []
    for i in range(5):
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

        # Fresh load each time
        ckpt = load_checkpoint_compat(ckpt_path, map_location=device)
        model_state = get_model_state_dict(ckpt)
        model.load_state_dict(model_state, strict=False)
        model = model.to(device)
        model.eval()

        hier = compute_hierarchy(model, all_ops, indices, device)
        hierarchies.append(hier)
        print(f"  Load {i + 1}: hierarchy = {hier:.4f}")

    print(f"\n  Mean: {np.mean(hierarchies):.4f}")
    print(f"  Std:  {np.std(hierarchies):.4f}")

    # Test 2: Check if curvature affects it
    print("\n=== Test 2: Different curvatures ===")
    for curv in [0.5, 1.0, 2.0]:
        model = TernaryVAEV5_11_PartialFreeze(
            latent_dim=16,
            hidden_dim=64,
            max_radius=0.99,
            curvature=curv,
            use_controller=False,
            use_dual_projection=True,
            freeze_encoder_b=False,
        )
        ckpt = load_checkpoint_compat(ckpt_path, map_location=device)
        model.load_state_dict(get_model_state_dict(ckpt), strict=False)
        model = model.to(device)
        model.eval()

        hier = compute_hierarchy(model, all_ops, indices, device)
        print(f"  Curvature {curv}: hierarchy = {hier:.4f}")

    # Test 3: Check if freeze state matters
    print("\n=== Test 3: Freeze states ===")
    for freeze_a, freeze_b in [(True, False), (False, True), (False, False), (True, True)]:
        model = TernaryVAEV5_11_PartialFreeze(
            latent_dim=16,
            hidden_dim=64,
            max_radius=0.99,
            curvature=2.0,
            use_controller=False,
            use_dual_projection=True,
            freeze_encoder_b=freeze_b,
        )
        ckpt = load_checkpoint_compat(ckpt_path, map_location=device)
        model.load_state_dict(get_model_state_dict(ckpt), strict=False)
        model = model.to(device)
        model.set_encoder_a_frozen(freeze_a)
        model.set_encoder_b_frozen(freeze_b)
        model.eval()

        hier = compute_hierarchy(model, all_ops, indices, device)
        print(f"  freeze_a={freeze_a}, freeze_b={freeze_b}: hierarchy = {hier:.4f}")

    # Test 4: Check the checkpoint from our best run
    print("\n=== Test 4: Check best stability run checkpoint ===")
    best_ckpt = PROJECT_ROOT / "checkpoints/stability_run_4/best.pt"
    if best_ckpt.exists():
        model = TernaryVAEV5_11_PartialFreeze(
            latent_dim=16,
            hidden_dim=64,
            max_radius=0.99,
            curvature=2.0,
            use_controller=False,
            use_dual_projection=True,
            freeze_encoder_b=False,
        )
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model = model.to(device)
        model.eval()

        hier = compute_hierarchy(model, all_ops, indices, device)
        print(f"  stability_run_4/best.pt: hierarchy = {hier:.4f}")
        if "metrics" in ckpt:
            print(f"  Stored metrics: {ckpt['metrics']}")

    # Test 5: Check homeostatic_rich
    print("\n=== Test 5: homeostatic_rich checkpoint ===")
    rich_ckpt = PROJECT_ROOT / "checkpoints/homeostatic_rich/best.pt"
    if rich_ckpt.exists():
        model = TernaryVAEV5_11_PartialFreeze(
            latent_dim=16,
            hidden_dim=64,
            max_radius=0.99,
            curvature=2.0,
            use_controller=False,
            use_dual_projection=True,
            freeze_encoder_b=False,
        )
        ckpt = torch.load(rich_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        model = model.to(device)
        model.eval()

        hier = compute_hierarchy(model, all_ops, indices, device)
        print(f"  homeostatic_rich/best.pt: hierarchy = {hier:.4f}")
        if "metrics" in ckpt:
            print(f"  Stored metrics: {ckpt['metrics']}")


if __name__ == "__main__":
    main()
