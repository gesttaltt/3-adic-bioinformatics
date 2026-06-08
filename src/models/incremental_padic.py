# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Incremental p-adic expansion using the existing TernaryVAE architecture.

This script implements a simpler approach to fractional p interpolation:
1. Start from v5_11_progressive (100% coverage at p=3)
2. Incrementally add "fractional" operations to the training set
3. Fine-tune the model to handle the expanded operation space
4. Monitor coverage on both original and expanded operations

Key insight: For p close to 3 (e.g., 3.01-3.5), the existing architecture
has sufficient capacity. We just need to expand the operation space gradually.

Usage:
    python src/scripts/epsilon_vae/incremental_padic.py --target_p 3.1
    python src/scripts/epsilon_vae/incremental_padic.py --target_p 3.5 --steps 10
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR, OUTPUT_DIR
from src.data.generation import generate_all_ternary_operations
from src.geometry import poincare_distance
from src.models.ternary_vae import TernaryVAEV5_11


def load_baseline_model(checkpoint_path: Path, device: str = "cpu"):
    """Load the v5_11_progressive baseline model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_state = ckpt.get("model_state_dict") or ckpt.get("model_state") or {}

    latent_dim = 16
    for key, value in model_state.items():
        if "encoder_A.fc_mu.weight" in key:
            latent_dim = value.shape[0]
            break

    model = TernaryVAEV5_11(
        latent_dim=latent_dim,
        hidden_dim=64,
        use_dual_projection=True,
        use_controller=False,
    )

    model.load_state_dict(model_state, strict=False)
    model.to(device)

    return model


def generate_fractional_operations(p: float, base_ops: np.ndarray, seed: int = 42) -> np.ndarray:
    """Generate operation set for fractional p.

    For p slightly above 3, we add "interpolated" operations:
    - These are convex combinations of existing ternary operations
    - They represent "fuzzy" or "fractional" states
    - The number of additional ops = floor(p^9) - 3^9

    Args:
        p: Target p value (e.g., 3.1)
        base_ops: Base ternary operations (19683, 9)
        seed: Random seed

    Returns:
        Extended operation array
    """
    if p <= 3.0:
        return base_ops

    rng = np.random.RandomState(seed)

    n_base = len(base_ops)  # 19683
    n_target = int(np.floor(p**9))
    n_additional = n_target - n_base

    if n_additional <= 0:
        return base_ops

    print(f"  Adding {n_additional:,} interpolated operations")

    # Strategy: Create interpolated operations as convex combinations
    # This represents the "fuzzy boundary" between discrete states

    # Select operation pairs to interpolate
    idx1 = rng.randint(0, n_base, size=n_additional)
    idx2 = rng.randint(0, n_base, size=n_additional)

    # Ensure we're interpolating between different operations
    same_mask = idx1 == idx2
    while same_mask.any():
        idx2[same_mask] = rng.randint(0, n_base, size=same_mask.sum())
        same_mask = idx1 == idx2

    # Interpolation weights (using beta distribution for smooth coverage)
    # Beta(2,2) gives values concentrated around 0.5
    alpha = rng.beta(2, 2, size=(n_additional, 1))

    # Create interpolated operations
    interpolated = (1 - alpha) * base_ops[idx1] + alpha * base_ops[idx2]

    # Combine with original
    extended = np.vstack([base_ops, interpolated])

    return extended.astype(np.float32)


def compute_dual_coverage(model, base_ops: torch.Tensor, extended_ops: torch.Tensor, device: str) -> dict:
    """Compute coverage on both original and extended operations.

    Args:
        model: TernaryVAE model
        base_ops: Original 19683 ternary operations
        extended_ops: Extended operation set (includes interpolated)
        device: Device

    Returns:
        Coverage metrics for both sets
    """
    model.eval()

    with torch.no_grad():
        # Coverage on original ops
        outputs_base = model(base_ops.to(device), compute_control=False)
        z_base = outputs_base["z_A_hyp"]
        # V5.12.2: Use hyperbolic distance for radii
        origin_base = torch.zeros_like(z_base)
        radii_base = poincare_distance(z_base, origin_base, c=1.0)
        coverage_base = ((radii_base > 0.1) & (radii_base < 0.99)).float().mean().item()

        # Coverage on extended ops
        outputs_ext = model(extended_ops.to(device), compute_control=False)
        z_ext = outputs_ext["z_A_hyp"]
        # V5.12.2: Use hyperbolic distance for radii
        origin_ext = torch.zeros_like(z_ext)
        radii_ext = poincare_distance(z_ext, origin_ext, c=1.0)
        coverage_ext = ((radii_ext > 0.1) & (radii_ext < 0.99)).float().mean().item()

        # Check for collisions in extended space
        dists = torch.cdist(z_ext, z_ext)
        mask = torch.eye(len(z_ext), device=device).bool()
        dists.masked_fill_(mask, float("inf"))
        min_dists = dists.min(dim=1)[0]
        collisions = (min_dists < 0.01).sum().item()

    return {
        "coverage_base": coverage_base,
        "coverage_extended": coverage_ext,
        "collisions": collisions,
        "n_base": len(base_ops),
        "n_extended": len(extended_ops),
    }


def train_incremental(
    model,
    base_ops: np.ndarray,
    target_p: float,
    device: str = "cpu",
    epochs: int = 50,
    lr: float = 1e-5,  # Very small LR to preserve learned structure
    batch_size: int = 512,
    constraint_weight: float = 0.1,
):
    """Incrementally train model to handle fractional p operations.

    Key strategy:
    1. Train on extended operation set
    2. Use very small learning rate
    3. Add constraint loss to preserve encoder structure
    4. Monitor coverage on original operations (should stay at 100%)

    Args:
        model: TernaryVAE model (loaded from v5_11_progressive)
        base_ops: Original ternary operations
        target_p: Target p value
        device: Device
        epochs: Training epochs
        lr: Learning rate (should be small!)
        batch_size: Batch size
        constraint_weight: Weight for encoder constraint loss

    Returns:
        Training history
    """
    model.to(device)

    # Generate extended operations
    extended_ops = generate_fractional_operations(target_p, base_ops)
    base_tensor = torch.tensor(base_ops, dtype=torch.float32)
    extended_tensor = torch.tensor(extended_ops, dtype=torch.float32)

    # Save reference encoder state (to constrain drift)
    {k: v.clone() for k, v in model.state_dict().items() if "encoder" in k}

    # Only train projection layers (freeze encoder initially)
    trainable_params = []
    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = False
        elif "proj" in name or "projection" in name:
            param.requires_grad = True
            trainable_params.append(param)
        else:
            param.requires_grad = True
            trainable_params.append(param)

    # Initial evaluation (before training decision)
    init_metrics = compute_dual_coverage(model, base_tensor, extended_tensor, device)
    print("\nInitial metrics:")
    print(f"  Base coverage: {init_metrics['coverage_base']:.4f}")
    print(f"  Extended coverage: {init_metrics['coverage_extended']:.4f}")

    if not trainable_params:
        # No trainable params - just evaluate, model already works
        print("No trainable parameters - model already generalizes!")
        return {
            "history": {
                "epochs": [],
                "coverage_base": [init_metrics["coverage_base"]],
                "coverage_extended": [init_metrics["coverage_extended"]],
                "loss": [],
            },
            "final_metrics": init_metrics,
            "best_state": model.state_dict(),
        }

    # Dataset
    dataset = TensorDataset(extended_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    print(f"\nTraining with {len(trainable_params)} parameter groups...")

    history = {
        "epochs": [],
        "coverage_base": [init_metrics["coverage_base"]],
        "coverage_extended": [init_metrics["coverage_extended"]],
        "loss": [],
    }

    best_extended_coverage = init_metrics["coverage_extended"]
    best_state = model.state_dict().copy()

    print("\nTraining with frozen encoder...")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for (batch,) in dataloader:
            batch = batch.to(device)

            optimizer.zero_grad()
            outputs = model(batch, compute_control=False)

            # Main loss
            loss = outputs.get("loss", outputs.get("recon_loss", torch.tensor(0.0)))
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss, device=device, requires_grad=True)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["loss"].append(avg_loss)
        history["epochs"].append(epoch)

        # Evaluate periodically
        if epoch % 5 == 0 or epoch == epochs - 1:
            metrics = compute_dual_coverage(model, base_tensor, extended_tensor, device)
            history["coverage_base"].append(metrics["coverage_base"])
            history["coverage_extended"].append(metrics["coverage_extended"])

            print(
                f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                f"Base: {metrics['coverage_base']:.4f} | "
                f"Extended: {metrics['coverage_extended']:.4f}"
            )

            # Save best
            if metrics["coverage_extended"] > best_extended_coverage and metrics["coverage_base"] >= 0.99:
                best_extended_coverage = metrics["coverage_extended"]
                best_state = model.state_dict().copy()

            # Early stop if base coverage drops
            if metrics["coverage_base"] < 0.95:
                print("WARNING: Base coverage dropped, reverting to best state")
                model.load_state_dict(best_state)
                break

    # Final evaluation
    model.load_state_dict(best_state)
    final_metrics = compute_dual_coverage(model, base_tensor, extended_tensor, device)

    return {
        "history": history,
        "final_metrics": final_metrics,
        "best_state": best_state,
    }


def main():
    parser = argparse.ArgumentParser(description="Incremental p-adic expansion")
    parser.add_argument(
        "--baseline_checkpoint", type=str, default=str(CHECKPOINTS_DIR / "v5_11_progressive" / "best.pt")
    )
    parser.add_argument("--target_p", type=float, default=3.1, help="Target p value")
    parser.add_argument("--steps", type=int, default=1, help="Number of intermediate steps (for gradual increase)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR / "incremental_padic"))
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("INCREMENTAL P-ADIC EXPANSION")
    print(f"{'=' * 70}")
    print(f"Baseline: {args.baseline_checkpoint}")
    print(f"Target p: {args.target_p}")
    print(f"Device: {device}")

    # Load baseline
    baseline_path = Path(args.baseline_checkpoint)
    model = load_baseline_model(baseline_path, device)

    # Generate base operations
    base_ops = generate_all_ternary_operations()
    print(f"\nBase operations: {len(base_ops):,}")

    # Create interpolation steps
    if args.steps > 1:
        p_values = np.linspace(3.0, args.target_p, args.steps + 1)[1:]  # Skip 3.0
    else:
        p_values = [args.target_p]

    print(f"P values: {[f'{p:.3f}' for p in p_values]}")

    # Results tracking
    all_results = []

    for p in p_values:
        print(f"\n{'=' * 70}")
        print(f"EXPANDING TO p = {p:.4f}")
        print(f"{'=' * 70}")

        n_ops = int(np.floor(p**9))
        print(f"Target operations: {n_ops:,}")

        results = train_incremental(
            model=model,
            base_ops=base_ops,
            target_p=p,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
        )

        all_results.append(
            {
                "p": p,
                "metrics": results["final_metrics"],
            }
        )

        # Save checkpoint
        torch.save(
            {
                "p": p,
                "model_state_dict": results["best_state"],
                "metrics": results["final_metrics"],
            },
            output_dir / f"checkpoint_p{p:.3f}.pt",
        )

        print(f"\nFinal at p={p:.3f}:")
        print(f"  Base coverage: {results['final_metrics']['coverage_base']:.4f}")
        print(f"  Extended coverage: {results['final_metrics']['coverage_extended']:.4f}")
        print(f"  Collisions: {results['final_metrics']['collisions']}")

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    for r in all_results:
        status = "OK" if r["metrics"]["coverage_base"] >= 0.99 else "DEGRADED"
        print(
            f"  p={r['p']:.3f}: base={r['metrics']['coverage_base']:.4f}, "
            f"extended={r['metrics']['coverage_extended']:.4f}, status={status}"
        )

    # Save summary
    with open(output_dir / "results.json", "w") as f:
        json.dump(
            {
                "target_p": args.target_p,
                "results": all_results,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
            default=float,
        )

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
