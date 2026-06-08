# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Optimize p=3 (ternary) baseline for p-adic generalization.

This script implements Phase 1 of the fractional p-adic interpolation strategy:
1. Analyze what makes v5_11_progressive achieve ~100% coverage
2. Create optimal unfreezing strategy for the encoder
3. Train to achieve the best possible p=3 baseline

Key insight from analysis:
- v5_11_progressive maintains 100% coverage throughout training
- The encoder weights change minimally (5-6% norm change)
- Coverage loss correlates with encoder weight norm increase

Strategy:
1. Start from v5_11_progressive best checkpoint (known good)
2. Use constrained optimization to maintain encoder structure
3. Monitor coverage continuously
4. Only allow changes that preserve algebraic completeness

Usage:
    python src/scripts/epsilon_vae/optimize_p3_baseline.py
    python src/scripts/epsilon_vae/optimize_p3_baseline.py --checkpoint outputs/models/v5_11_progressive/best.pt
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.optim as optim

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR, OUTPUT_DIR
from src.data.generation import generate_all_ternary_operations
from src.models.ternary_vae import TernaryVAEV5_11


def load_reference_checkpoint(checkpoint_path: Path, device: str = "cpu"):
    """Load the reference checkpoint (v5_11_progressive best).

    Returns:
        Model, optimizer state, and reference metrics
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model_state = ckpt.get("model_state_dict") or ckpt.get("model_state") or {}

    # Infer architecture
    latent_dim = 16
    for key, value in model_state.items():
        if "encoder_A.fc_mu.weight" in key:
            latent_dim = value.shape[0]
            break

    # Create model
    model = TernaryVAEV5_11(
        latent_dim=latent_dim,
        hidden_dim=64,
        use_dual_projection=True,
        use_controller=False,
    )

    model.load_state_dict(model_state, strict=False)
    model.to(device)

    metrics = ckpt.get("metrics", {})

    return model, metrics, ckpt


def compute_coverage(model, all_ops: torch.Tensor, device: str = "cpu") -> dict:
    """Compute coverage and related metrics for all operations.

    Args:
        model: TernaryVAE model
        all_ops: All 19683 ternary operations
        device: Device to use

    Returns:
        Dict with coverage metrics
    """
    model.eval()
    all_ops = all_ops.to(device)

    with torch.no_grad():
        outputs = model(all_ops, compute_control=False)

        z_A_hyp = outputs["z_A_hyp"]  # Hyperbolic embeddings

        # Coverage: fraction of operations that are "well-represented"
        # An operation is well-represented if its embedding has reasonable norm
        norms = torch.norm(z_A_hyp, dim=1)

        # Operations with very small or very large norms are problematic
        min_norm = 0.1
        max_norm = 10.0
        well_represented = (norms > min_norm) & (norms < max_norm)
        coverage = well_represented.float().mean().item()

        # Uniqueness: check for collisions in latent space
        # Use pairwise distances
        dists = torch.cdist(z_A_hyp, z_A_hyp)
        # Mask diagonal
        mask = torch.eye(len(z_A_hyp), device=device).bool()
        dists = dists.masked_fill(mask, float("inf"))
        min_dists = dists.min(dim=1)[0]

        # Collision if minimum distance < threshold
        collision_threshold = 0.01
        collisions = (min_dists < collision_threshold).sum().item()

        # Spread: standard deviation of embeddings
        spread = z_A_hyp.std(dim=0).mean().item()

    return {
        "coverage": coverage,
        "collisions": collisions,
        "collision_rate": collisions / len(all_ops),
        "mean_norm": norms.mean().item(),
        "std_norm": norms.std().item(),
        "spread": spread,
        "min_dist_mean": min_dists.mean().item(),
        "min_dist_min": min_dists.min().item(),
    }


def compute_encoder_constraint_loss(
    model,
    reference_state: dict,
    constraint_weight: float = 0.1,
) -> torch.Tensor:
    """Compute loss that constrains encoder to stay close to reference.

    This prevents the encoder from drifting away from the optimal configuration
    that achieves 100% coverage.

    Args:
        model: Current model
        reference_state: Reference state dict (from v5_11_progressive)
        constraint_weight: Weight for constraint loss

    Returns:
        Constraint loss tensor
    """
    constraint_loss = 0.0
    n_constrained = 0

    current_state = model.state_dict()

    for key in reference_state:
        if "encoder" in key and key in current_state:
            ref_weights = reference_state[key]
            curr_weights = current_state[key]

            if ref_weights.shape == curr_weights.shape:
                # L2 distance from reference
                diff = (curr_weights - ref_weights.to(curr_weights.device)).pow(2).sum()
                # Normalize by number of parameters
                diff = diff / ref_weights.numel()
                constraint_loss = constraint_loss + diff
                n_constrained += 1

    if n_constrained > 0:
        constraint_loss = constraint_loss / n_constrained

    return constraint_weight * constraint_loss


class GradualUnfreezer:
    """Gradually unfreeze encoder layers during training.

    Strategy:
    1. Start with encoder fully frozen
    2. Unfreeze from output layers toward input
    3. Use very small learning rates for encoder
    4. Monitor coverage - if it drops, re-freeze

    This preserves the algebraic structure learned by the encoder.
    """

    def __init__(
        self,
        model,
        encoder_lr_scale: float = 0.01,  # Encoder LR is 1% of base
        unfreeze_schedule: dict = None,
    ):
        self.model = model
        self.encoder_lr_scale = encoder_lr_scale

        # Default schedule: start with projection unfrozen, gradually add encoder
        self.unfreeze_schedule = unfreeze_schedule or {
            0: ["projection", "proj_"],  # Start with projection only
            20: ["fc_mu", "fc_logvar"],  # Unfreeze latent projections
            40: ["encoder.4"],  # Unfreeze last hidden layer
            70: ["encoder.2"],  # Unfreeze middle layer
            100: ["encoder.0"],  # Finally unfreeze input layer (careful!)
        }

        self.current_unfrozen = set()

    def get_param_groups(self, base_lr: float, epoch: int) -> list:
        """Get parameter groups with appropriate learning rates.

        Args:
            base_lr: Base learning rate
            epoch: Current epoch

        Returns:
            List of param groups for optimizer
        """
        # Determine what should be unfrozen at this epoch
        for e, layers in sorted(self.unfreeze_schedule.items()):
            if epoch >= e:
                self.current_unfrozen.update(layers)

        encoder_params = []
        projection_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            is_encoder = "encoder" in name

            if is_encoder:
                # Check if this layer is unfrozen
                is_unfrozen = any(layer in name for layer in self.current_unfrozen)
                param.requires_grad = is_unfrozen
                if is_unfrozen:
                    encoder_params.append(param)
            elif "projection" in name or "proj_" in name:
                param.requires_grad = True
                projection_params.append(param)
            else:
                param.requires_grad = True
                other_params.append(param)

        param_groups = []

        if encoder_params:
            param_groups.append(
                {
                    "params": encoder_params,
                    "lr": base_lr * self.encoder_lr_scale,
                    "name": "encoder",
                }
            )

        if projection_params:
            param_groups.append(
                {
                    "params": projection_params,
                    "lr": base_lr,
                    "name": "projection",
                }
            )

        if other_params:
            param_groups.append(
                {
                    "params": other_params,
                    "lr": base_lr,
                    "name": "other",
                }
            )

        return param_groups

    def report_status(self, epoch: int):
        """Report which layers are currently unfrozen."""
        frozen = []
        unfrozen = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                unfrozen.append(name.split(".")[0])
            else:
                frozen.append(name.split(".")[0])

        unfrozen = list(set(unfrozen))
        frozen = list(set(frozen))

        return {
            "epoch": epoch,
            "unfrozen": unfrozen,
            "frozen": frozen,
            "n_unfrozen": sum(p.requires_grad for p in self.model.parameters()),
            "n_total": sum(1 for _ in self.model.parameters()),
        }


def train_optimized_p3(
    model,
    reference_state: dict,
    all_ops: torch.Tensor,
    device: str = "cpu",
    epochs: int = 100,
    base_lr: float = 1e-4,
    constraint_weight: float = 0.1,
    output_dir: Path = None,
):
    """Train to optimize p=3 performance while preserving coverage.

    Args:
        model: TernaryVAE model
        reference_state: Reference state dict
        all_ops: All ternary operations
        device: Device
        epochs: Number of epochs
        base_lr: Base learning rate
        constraint_weight: Weight for encoder constraint
        output_dir: Output directory

    Returns:
        Training history
    """
    output_dir = output_dir or OUTPUT_DIR / "p3_optimized"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize gradual unfreezer
    unfreezer = GradualUnfreezer(model)

    # Training data (sample from all operations)
    n_train = min(10000, len(all_ops))
    train_indices = torch.randperm(len(all_ops))[:n_train]
    train_ops = all_ops[train_indices]

    # Initial coverage
    initial_metrics = compute_coverage(model, all_ops, device)
    print(f"\nInitial coverage: {initial_metrics['coverage']:.4f}")
    print(f"Initial collisions: {initial_metrics['collisions']}")

    history = {
        "epochs": [],
        "coverage": [],
        "collisions": [],
        "constraint_loss": [],
        "total_loss": [],
        "unfrozen_layers": [],
    }

    best_coverage = initial_metrics["coverage"]
    best_state = model.state_dict()

    for epoch in range(epochs):
        # Get parameter groups for this epoch
        param_groups = unfreezer.get_param_groups(base_lr, epoch)

        if not param_groups:
            print(f"Epoch {epoch}: No trainable parameters!")
            continue

        optimizer = optim.AdamW(param_groups, weight_decay=0.01)

        # Training step
        model.train()
        train_ops_device = train_ops.to(device)

        # Forward pass
        outputs = model(train_ops_device, compute_control=False)

        # Main loss (reconstruction + KL)
        recon_loss = outputs.get("loss", torch.tensor(0.0, device=device))

        # Constraint loss (keep encoder close to reference)
        constraint_loss = compute_encoder_constraint_loss(model, reference_state, constraint_weight)

        total_loss = recon_loss + constraint_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Evaluate coverage
        metrics = compute_coverage(model, all_ops, device)

        # Log
        history["epochs"].append(epoch)
        history["coverage"].append(metrics["coverage"])
        history["collisions"].append(metrics["collisions"])
        history["constraint_loss"].append(constraint_loss.item())
        history["total_loss"].append(total_loss.item())
        history["unfrozen_layers"].append(unfreezer.report_status(epoch))

        # Save best
        if metrics["coverage"] > best_coverage:
            best_coverage = metrics["coverage"]
            best_state = model.state_dict().copy()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state,
                    "metrics": metrics,
                    "coverage": best_coverage,
                },
                output_dir / "best.pt",
            )

        # Print progress
        if epoch % 10 == 0 or epoch == epochs - 1:
            status = unfreezer.report_status(epoch)
            print(
                f"Epoch {epoch:3d} | Coverage: {metrics['coverage']:.4f} | "
                f"Collisions: {metrics['collisions']:4d} | "
                f"Constraint: {constraint_loss.item():.4f} | "
                f"Unfrozen: {status['n_unfrozen']}/{status['n_total']}"
            )

        # Early stopping if coverage drops significantly
        if metrics["coverage"] < initial_metrics["coverage"] - 0.1:
            print("\nWARNING: Coverage dropped by >10%, reverting to best state")
            model.load_state_dict(best_state)
            break

    # Save final
    torch.save(
        {
            "epoch": epochs - 1,
            "model_state_dict": model.state_dict(),
            "best_state_dict": best_state,
            "history": history,
            "best_coverage": best_coverage,
        },
        output_dir / "final.pt",
    )

    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax1 = axes[0, 0]
    ax1.plot(history["epochs"], history["coverage"], "b-", linewidth=2)
    ax1.axhline(
        y=initial_metrics["coverage"], color="r", linestyle="--", label=f"Initial: {initial_metrics['coverage']:.4f}"
    )
    ax1.axhline(y=best_coverage, color="g", linestyle="--", label=f"Best: {best_coverage:.4f}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Coverage")
    ax1.set_title("Coverage During Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    ax2.plot(history["epochs"], history["collisions"], "r-", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Collisions")
    ax2.set_title("Latent Space Collisions")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(history["epochs"], history["constraint_loss"], "g-", linewidth=2)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Constraint Loss")
    ax3.set_title("Encoder Constraint Loss")
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    n_unfrozen = [h["n_unfrozen"] for h in history["unfrozen_layers"]]
    ax4.plot(history["epochs"], n_unfrozen, "m-", linewidth=2)
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Number of Trainable Parameters")
    ax4.set_title("Gradual Unfreezing Progress")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=150)
    plt.close()

    print("\nTraining complete!")
    print(f"Best coverage: {best_coverage:.4f}")
    print(f"Results saved to {output_dir}")

    return history, best_state


def main():
    parser = argparse.ArgumentParser(description="Optimize p=3 baseline")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "v5_11_progressive" / "best.pt"),
        help="Reference checkpoint path",
    )
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR / "p3_optimized"), help="Output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--constraint_weight", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load reference checkpoint
    print(f"\n{'=' * 70}")
    print("LOADING REFERENCE CHECKPOINT")
    print(f"{'=' * 70}")

    model, ref_metrics, ref_ckpt = load_reference_checkpoint(checkpoint_path, device)
    ref_state = ref_ckpt.get("model_state_dict") or ref_ckpt.get("model_state") or {}

    print(f"Reference checkpoint: {checkpoint_path.name}")
    print(f"Reference coverage: {ref_metrics.get('coverage', 'N/A')}")

    # Generate all operations
    print(f"\n{'=' * 70}")
    print("GENERATING OPERATION SPACE")
    print(f"{'=' * 70}")

    all_ops = generate_all_ternary_operations()
    all_ops_tensor = torch.tensor(all_ops, dtype=torch.float32)
    print(f"Total operations: {len(all_ops):,}")

    # Initial evaluation
    print(f"\n{'=' * 70}")
    print("INITIAL EVALUATION")
    print(f"{'=' * 70}")

    initial_metrics = compute_coverage(model, all_ops_tensor, device)
    print(f"Coverage: {initial_metrics['coverage']:.4f}")
    print(f"Collisions: {initial_metrics['collisions']}")
    print(f"Mean embedding norm: {initial_metrics['mean_norm']:.4f}")
    print(f"Embedding spread: {initial_metrics['spread']:.4f}")

    # Train
    print(f"\n{'=' * 70}")
    print("OPTIMIZING P=3 BASELINE")
    print(f"{'=' * 70}")

    history, best_state = train_optimized_p3(
        model=model,
        reference_state=ref_state,
        all_ops=all_ops_tensor,
        device=device,
        epochs=args.epochs,
        base_lr=args.lr,
        constraint_weight=args.constraint_weight,
        output_dir=output_dir,
    )

    # Final evaluation
    print(f"\n{'=' * 70}")
    print("FINAL EVALUATION")
    print(f"{'=' * 70}")

    model.load_state_dict(best_state)
    final_metrics = compute_coverage(model, all_ops_tensor, device)
    print(f"Final coverage: {final_metrics['coverage']:.4f}")
    print(f"Final collisions: {final_metrics['collisions']}")
    print(f"Improvement: {final_metrics['coverage'] - initial_metrics['coverage']:+.4f}")

    # Ready for fractional p interpolation?
    print(f"\n{'=' * 70}")
    print("P-ADIC READINESS ASSESSMENT")
    print(f"{'=' * 70}")

    if final_metrics["coverage"] >= 0.99 and final_metrics["collisions"] == 0:
        print("READY for fractional p interpolation!")
        print("Next step: Run fractional_padic_training.py with p=3.01")
    elif final_metrics["coverage"] >= 0.95:
        print("NEARLY READY - coverage is high but not perfect")
        print("Consider: Adjusting constraint weight or training longer")
    else:
        print("NOT READY - coverage needs improvement")
        print("Consider: Using different reference checkpoint or hyperparameters")


if __name__ == "__main__":
    main()
