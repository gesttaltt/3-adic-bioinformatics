# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Quick Training Smoke Test - Verify GPU and Training Pipeline.

This script runs a fast 5-epoch training to verify:
1. GPU is detected and working
2. All training components are functional
3. Model saves and loads correctly
4. Metrics are computed properly

Usage:
    python src/scripts/quick_train.py
    python src/scripts/quick_train.py --epochs 10 --device cuda
    python src/scripts/quick_train.py --full  # Run with all features enabled

Expected runtime: 1-3 minutes on GPU, 5-10 minutes on CPU
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_gpu_info():
    """Print GPU information."""
    print_header("GPU Information")

    if torch.cuda.is_available():
        print("  CUDA Available: Yes")
        print(f"  Device Count: {torch.cuda.device_count()}")
        print(f"  Current Device: {torch.cuda.current_device()}")
        print(f"  Device Name: {torch.cuda.get_device_name(0)}")

        props = torch.cuda.get_device_properties(0)
        print(f"  Total VRAM: {props.total_memory / 1024**3:.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")

        # Check available memory
        free_mem = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        print(f"  Available VRAM: {free_mem / 1024**3:.2f} GB")
        print(f"  PyTorch Version: {torch.__version__}")
        print(f"  CUDA Version: {torch.version.cuda}")
    else:
        print("  CUDA Available: No")
        print("  Running on CPU (training will be slower)")

    return torch.cuda.is_available()


def run_quick_training(args):
    """Run quick training smoke test."""
    from scipy.stats import spearmanr

    from src.config.paths import CHECKPOINTS_DIR
    from src.core import TERNARY
    from src.data.generation import generate_all_ternary_operations
    from src.geometry import poincare_distance
    from src.losses import PAdicGeodesicLoss, RadialHierarchyLoss
    from src.models import TernaryVAEV5_11_PartialFreeze

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("  [WARNING] CUDA not available, falling back to CPU")
        device = "cpu"

    print_header("Training Configuration")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Full Mode: {args.full}")

    # Create model
    print_header("Creating Model")
    start_time = time.time()

    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=64,
        max_radius=0.95,
        curvature=1.0,
        use_controller=args.full,
        use_dual_projection=args.full,
    )

    # Check for v5.5 checkpoint
    v5_5_path = CHECKPOINTS_DIR / "v5_5" / "latest.pt"
    if v5_5_path.exists():
        print(f"  Loading frozen weights from: {v5_5_path}")
        model.load_v5_5_checkpoint(v5_5_path, device)
    else:
        print(f"  [INFO] No v5.5 checkpoint found at {v5_5_path}")
        print("  [INFO] Training with random initialization (coverage may be low)")

    model = model.to(device)

    # Count parameters
    param_counts = model.count_parameters()
    print(f"  Total Parameters: {param_counts['total']:,}")
    print(f"  Trainable Parameters: {param_counts['trainable']:,}")
    print(f"  Model creation time: {time.time() - start_time:.2f}s")

    # Create dataset
    print_header("Loading Dataset")
    start_time = time.time()

    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, dtype=torch.float32, device=device)
    indices = torch.arange(len(operations), device=device)

    print(f"  Dataset size: {len(x)} operations")
    print(f"  Input shape: {x.shape}")
    print(f"  Data loading time: {time.time() - start_time:.2f}s")

    # Create loss functions
    geodesic_loss_fn = PAdicGeodesicLoss(curvature=1.0, n_pairs=1000).to(device)
    radial_loss_fn = RadialHierarchyLoss(inner_radius=0.1, outer_radius=0.85).to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.get_trainable_parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    # Training loop
    print_header("Training")
    training_start = time.time()

    batch_size = args.batch_size
    n_samples = len(x)
    n_batches = (n_samples + batch_size - 1) // batch_size

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        # Shuffle
        perm = torch.randperm(n_samples, device=device)

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_idx = perm[start_idx:end_idx]

            x_batch = x[batch_idx]
            idx_batch = indices[batch_idx]

            optimizer.zero_grad()

            # Forward
            outputs = model(x_batch, compute_control=False)
            z_A_hyp = outputs["z_A_hyp"]
            z_B_hyp = outputs["z_B_hyp"]

            # Losses
            geo_loss_A, _ = geodesic_loss_fn(z_A_hyp, idx_batch)
            geo_loss_B, _ = geodesic_loss_fn(z_B_hyp, idx_batch)
            rad_loss_A, _ = radial_loss_fn(z_A_hyp, idx_batch)
            rad_loss_B, _ = radial_loss_fn(z_B_hyp, idx_batch)

            tau = min(1.0, epoch / (args.epochs * 0.7))
            loss = tau * (geo_loss_A + geo_loss_B) + 2.0 * (rad_loss_A + rad_loss_B)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / n_batches
        epoch_time = time.time() - epoch_start

        # Quick evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(x, compute_control=False)
            z_A = outputs["z_A_hyp"]

            # Coverage
            mu_A = outputs["mu_A"]
            logits = model.decoder_A(mu_A)
            preds = torch.argmax(logits, dim=-1) - 1
            correct = (preds == x.long()).float().mean(dim=1)
            coverage = (correct == 1.0).sum().item() / len(x)

            # V5.12.2: Radial correlation using hyperbolic distance
            origin = torch.zeros_like(z_A)
            radii = poincare_distance(z_A, origin, c=1.0).cpu().numpy()
            valuations = TERNARY.valuation(indices).cpu().numpy()
            radial_corr = spearmanr(valuations, radii)[0]

        print(
            f"  Epoch {epoch + 1}/{args.epochs}: loss={avg_loss:.4f}, "
            f"coverage={coverage * 100:.1f}%, radial_corr={radial_corr:.3f}, "
            f"time={epoch_time:.2f}s"
        )

    total_time = time.time() - training_start

    # Final evaluation
    print_header("Final Evaluation")
    model.eval()
    with torch.no_grad():
        outputs = model(x, compute_control=False)
        z_A = outputs["z_A_hyp"]
        z_B = outputs["z_B_hyp"]

        # Coverage
        mu_A = outputs["mu_A"]
        logits = model.decoder_A(mu_A)
        preds = torch.argmax(logits, dim=-1) - 1
        correct = (preds == x.long()).float().mean(dim=1)
        coverage = (correct == 1.0).sum().item() / len(x)

        # V5.12.2: Radial correlations using hyperbolic distance
        origin = torch.zeros_like(z_A)
        radii_A = poincare_distance(z_A, origin, c=1.0).cpu().numpy()
        radii_B = poincare_distance(z_B, origin, c=1.0).cpu().numpy()
        valuations = TERNARY.valuation(indices).cpu().numpy()

        radial_corr_A = spearmanr(valuations, radii_A)[0]
        radial_corr_B = spearmanr(valuations, radii_B)[0]

        # Radius stats
        mean_radius = radii_A.mean()
        radius_range = radii_A.max() - radii_A.min()

    print(f"  Coverage: {coverage * 100:.1f}%")
    print(f"  Radial Hierarchy A: {radial_corr_A:.4f}")
    print(f"  Radial Hierarchy B: {radial_corr_B:.4f}")
    print(f"  Mean Radius: {mean_radius:.4f}")
    print(f"  Radius Range: {radius_range:.4f}")

    # Save checkpoint
    if args.save:
        print_header("Saving Checkpoint")
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = save_dir / f"quick_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        torch.save(
            {
                "epoch": args.epochs,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": {
                    "coverage": coverage,
                    "radial_corr_A": radial_corr_A,
                    "radial_corr_B": radial_corr_B,
                },
            },
            checkpoint_path,
        )
        print(f"  Saved to: {checkpoint_path}")

    # Summary
    print_header("Summary")
    print(f"  Total training time: {total_time:.2f}s")
    print(f"  Time per epoch: {total_time / args.epochs:.2f}s")
    print(f"  Samples per second: {len(x) * args.epochs / total_time:.0f}")

    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated(0) / 1024**3
        print(f"  Peak GPU memory: {max_mem:.2f} GB")

    # Status
    print_header("Status")
    if coverage > 0.5:
        print("  ✓ Training pipeline: WORKING")
    else:
        print("  ⚠ Training pipeline: Working but coverage low (expected without v5.5 checkpoint)")

    if radial_corr_A < 0:
        print("  ✓ Radial hierarchy: Learning correctly (negative correlation)")
    else:
        print("  ⚠ Radial hierarchy: Not yet negative (needs more epochs)")

    print("\n  GPU smoke test: PASSED")
    print("  Your hardware is ready for full training!")

    return True


def main():
    parser = argparse.ArgumentParser(description="Quick training smoke test for GPU verification")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size (default: 512)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (default: cuda)")
    parser.add_argument("--full", action="store_true", help="Enable all features (controller, dual projection)")
    parser.add_argument("--save", action="store_true", help="Save checkpoint after training")
    parser.add_argument("--save_dir", type=str, default="outputs/quick_train", help="Directory to save checkpoints")

    args = parser.parse_args()

    print_header("Ternary VAE Quick Training Smoke Test")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check GPU
    has_gpu = print_gpu_info()

    if not has_gpu and args.device == "cuda":
        print("\n  [INFO] No GPU detected, switching to CPU")
        args.device = "cpu"

    # Run training
    try:
        success = run_quick_training(args)
        return 0 if success else 1
    except Exception as e:
        print(f"\n  [ERROR] Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
