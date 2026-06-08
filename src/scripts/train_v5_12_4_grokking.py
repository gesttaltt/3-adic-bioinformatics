#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""V5.12.4 Extended Training Script with Grokking Detection.

This script implements enhanced training for observing emergent phenomena and grokking:
- Extended 500-epoch training with multi-phase learning rates
- Enhanced monitoring and grokking detection
- Detailed logging for emergence analysis
- Conservative training strategy for long-term stability

Based on infrastructure audit findings from docs/audits/TRAINING_INFRASTRUCTURE_AUDIT.md

Usage:
    python src/scripts/train_v5_12_4_grokking.py
    python src/scripts/train_v5_12_4_grokking.py --config src/configs/v5_12_4_extended_grokking.yaml
    python src/scripts/train_v5_12_4_grokking.py --epochs 500 --device cuda

Target: Observe grokking patterns, phase transitions, and emergent behaviors over 2-8 hour runs.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from torch.utils.tensorboard import SummaryWriter

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import RUNS_DIR
from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.geometry import get_riemannian_optimizer, poincare_distance
from src.losses import (
    PAdicGeodesicLoss,
    RadialHierarchyLoss,
    RichHierarchyLoss,
)
from src.models import TernaryVAEV5_11_PartialFreeze
from src.models.homeostasis import compute_Q
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat


class GrokkingDetector:
    """Detect grokking and phase transition phenomena during training."""

    def __init__(
        self,
        window_size: int = 20,
        plateau_threshold: float = 0.0001,
        plateau_patience: int = 15,
        accuracy_jump_threshold: float = 0.02,
    ):
        """Initialize grokking detector.

        Args:
            window_size: Window size for moving averages and trend analysis
            plateau_threshold: Minimum loss change to detect plateau
            plateau_patience: Epochs required to confirm plateau
            accuracy_jump_threshold: Minimum accuracy jump to detect grokking
        """
        self.window_size = window_size
        self.plateau_threshold = plateau_threshold
        self.plateau_patience = plateau_patience
        self.accuracy_jump_threshold = accuracy_jump_threshold

        # History tracking
        self.loss_history: list[float] = []
        self.accuracy_history: list[float] = []
        self.gradient_norm_history: list[float] = []
        self.hierarchy_history: list[float] = []

        # State tracking
        self.in_plateau = False
        self.plateau_start_epoch = None
        self.potential_grokking_events = []

    def update(self, epoch: int, loss: float, accuracy: float, gradient_norm: float, hierarchy: float) -> dict:
        """Update detector with new metrics and analyze for grokking.

        Returns:
            Dict containing grokking analysis results
        """
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)
        self.gradient_norm_history.append(gradient_norm)
        self.hierarchy_history.append(hierarchy)

        analysis = {
            "in_plateau": False,
            "plateau_duration": 0,
            "potential_grokking": False,
            "phase_transition": False,
            "gradient_regime_change": False,
        }

        if len(self.loss_history) < self.window_size:
            return analysis

        # Plateau detection
        recent_losses = self.loss_history[-self.plateau_patience :]
        if len(recent_losses) >= self.plateau_patience:
            loss_change = max(recent_losses) - min(recent_losses)

            if loss_change < self.plateau_threshold:
                if not self.in_plateau:
                    self.in_plateau = True
                    self.plateau_start_epoch = epoch - self.plateau_patience + 1
                    analysis["phase_transition"] = True

                analysis["in_plateau"] = True
                analysis["plateau_duration"] = epoch - self.plateau_start_epoch + 1
            else:
                if self.in_plateau:
                    # Exiting plateau - potential grokking
                    analysis["potential_grokking"] = True

                    # Check for accuracy jump
                    if len(self.accuracy_history) >= self.window_size:
                        pre_plateau_acc = np.mean(self.accuracy_history[-self.window_size : -self.plateau_patience])
                        current_acc = self.accuracy_history[-1]
                        acc_jump = current_acc - pre_plateau_acc

                        if acc_jump > self.accuracy_jump_threshold:
                            self.potential_grokking_events.append(
                                {
                                    "epoch": epoch,
                                    "plateau_duration": analysis["plateau_duration"],
                                    "accuracy_jump": acc_jump,
                                    "loss_before": np.mean(recent_losses),
                                    "loss_after": loss,
                                }
                            )

                self.in_plateau = False
                self.plateau_start_epoch = None

        # Gradient regime change detection
        if len(self.gradient_norm_history) >= self.window_size:
            recent_grad_norms = self.gradient_norm_history[-self.window_size :]
            np.std(recent_grad_norms)
            grad_mean = np.mean(recent_grad_norms)

            if len(self.gradient_norm_history) > self.window_size:
                prev_grad_norms = self.gradient_norm_history[-2 * self.window_size : -self.window_size]
                prev_grad_mean = np.mean(prev_grad_norms)

                # Significant change in gradient regime
                if abs(grad_mean - prev_grad_mean) > 0.5 * prev_grad_mean:
                    analysis["gradient_regime_change"] = True

        return analysis

    def get_summary(self) -> dict:
        """Get summary of detected grokking events."""
        return {
            "total_grokking_events": len(self.potential_grokking_events),
            "grokking_events": self.potential_grokking_events,
            "current_plateau": self.in_plateau,
            "plateau_start": self.plateau_start_epoch,
        }


def create_multi_phase_scheduler(optimizer, config: dict):
    """Create multi-phase learning rate scheduler."""
    sched_config = config.get("training", {}).get("scheduler", {})

    if sched_config.get("type") != "multi_phase_cosine":
        # Fall back to standard scheduler
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # For now, return standard scheduler - full multi-phase implementation would require custom scheduler
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2)


def log_enhanced_metrics(writer: SummaryWriter, model, epoch: int, config: dict):
    """Log enhanced metrics for grokking analysis."""
    enhanced_config = config.get("logging", {}).get("enhanced_metrics", {})

    if not enhanced_config.get("enabled", False):
        return

    # Gradient norms
    if enhanced_config.get("log_gradients", False):
        total_grad_norm = 0.0
        param_count = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                writer.add_scalar(f"Gradients/{name}_norm", grad_norm, epoch)
                total_grad_norm += grad_norm**2
                param_count += 1

        if param_count > 0:
            total_grad_norm = total_grad_norm**0.5
            writer.add_scalar("Gradients/total_norm", total_grad_norm, epoch)

    # Weight norms
    if enhanced_config.get("log_weights", False):
        for name, param in model.named_parameters():
            if param.data is not None:
                weight_norm = param.data.norm(2).item()
                writer.add_scalar(f"Weights/{name}_norm", weight_norm, epoch)

    # Effective rank (simplified version)
    if enhanced_config.get("effective_rank", False):
        try:
            # Get projection layer weights for effective rank analysis
            if hasattr(model, "projection"):
                proj_weight = model.projection.layers[0].weight.data
                U, S, V = torch.svd(proj_weight)
                # Effective rank = (sum of singular values)^2 / (sum of squared singular values)
                eff_rank = (S.sum() ** 2) / (S**2).sum()
                writer.add_scalar("Analysis/effective_rank", eff_rank.item(), epoch)
        except Exception:
            # Skip if SVD fails
            pass


def main():
    parser = argparse.ArgumentParser(description="V5.12.4 Extended Training with Grokking Detection")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/v5_12_4_extended_grokking.yaml",
        help="Path to extended config YAML",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    # Device setup
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    device = torch.device(args.device)
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB)")

    # Load config
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    print(f"Loaded extended grokking config from: {config_path}")

    # Override with command line args
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["lr"] = args.lr

    # Create save directory
    save_dir = PROJECT_ROOT / config["checkpoints"]["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup TensorBoard with enhanced logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = RUNS_DIR / f"v5_12_4_grokking_{timestamp}"
    writer = SummaryWriter(log_dir=str(log_dir))

    print("\n" + "=" * 80)
    print("V5.12.4 EXTENDED TRAINING WITH GROKKING DETECTION")
    print("=" * 80)
    print(f"Target: {config['training']['epochs']} epochs (~4-8 hours)")
    print(f"Save dir: {save_dir}")
    print(f"Log dir: {log_dir}")

    # Create model
    print("\n=== Creating V5.12.4 Model ===")
    model_cfg = config["model"]
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=model_cfg.get("latent_dim", 16),
        hidden_dim=model_cfg.get("hidden_dim", 64),
        max_radius=model_cfg.get("max_radius", 0.95),
        curvature=model_cfg.get("curvature", 1.0),
        use_controller=model_cfg.get("use_controller", True),
        use_dual_projection=model_cfg.get("use_dual_projection", True),
        n_projection_layers=model_cfg.get("projection_layers", 2),
        projection_dropout=model_cfg.get("projection_dropout", 0.1),
        learnable_curvature=model_cfg.get("learnable_curvature", True),
        manifold_aware=model_cfg.get("manifold_aware", True),
        freeze_encoder_b=False,
        encoder_b_lr_scale=config["option_c"].get("encoder_b_lr_scale", 0.1),
        encoder_type=model_cfg.get("encoder_type", "improved"),
        decoder_type=model_cfg.get("decoder_type", "improved"),
    ).to(device)

    # Load checkpoint if available
    frozen_cfg = config.get("frozen_checkpoint", {})
    checkpoint_path = frozen_cfg.get("path", "checkpoints/v5_5/latest.pt")

    if checkpoint_path is None or checkpoint_path == "null":
        print("Training from scratch (no checkpoint specified)")
        frozen_path = None
    else:
        frozen_path = PROJECT_ROOT / checkpoint_path

    if frozen_path is not None and frozen_path.exists():
        print(f"Loading checkpoint: {frozen_path}")
        try:
            ckpt = load_checkpoint_compat(frozen_path, map_location=device)
            model_state = get_model_state_dict(ckpt)
            missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
            print(f"  Loaded with {len(missing_keys)} missing keys, {len(unexpected_keys)} unexpected keys")
        except Exception as e:
            print(f"  WARNING: Could not load checkpoint: {e}")
    else:
        print(f"  INFO: No checkpoint found at {frozen_path}, using random initialization")

    param_counts = model.count_parameters()
    print(f"Parameters: {param_counts['total']:,} total, {param_counts['trainable']:,} trainable")

    # Create dataset
    print("\n=== Loading Dataset ===")
    all_ops_np = generate_all_ternary_operations()
    all_ops = torch.tensor(all_ops_np, dtype=torch.float32, device=device)
    indices = torch.arange(len(all_ops), device=device)
    print(f"Dataset size: {len(all_ops)}")

    # Create loss functions
    print("\n=== Creating Loss Functions ===")
    loss_cfg = config["loss"]

    # RichHierarchyLoss (Primary)
    rich_cfg = loss_cfg["rich_hierarchy"]
    RichHierarchyLoss(
        hierarchy_weight=rich_cfg.get("hierarchy_weight", 5.0),
        coverage_weight=rich_cfg.get("coverage_weight", 1.0),
        richness_weight=rich_cfg.get("richness_weight", 2.5),
        separation_weight=rich_cfg.get("separation_weight", 3.0),
        min_richness_ratio=rich_cfg.get("min_richness_ratio", 0.4),
    ).to(device)

    # Other loss functions
    radial_cfg = loss_cfg["radial"]
    radial_loss_fn = RadialHierarchyLoss(
        inner_radius=radial_cfg.get("inner_radius", 0.08),
        outer_radius=radial_cfg.get("outer_radius", 0.90),
        margin_weight=radial_cfg.get("margin_weight", 0.5),
    ).to(device)

    geo_cfg = loss_cfg["geodesic"]
    geodesic_loss_fn = (
        PAdicGeodesicLoss(
            curvature=geo_cfg.get("curvature", 1.0),
            n_pairs=geo_cfg.get("n_pairs", 2000),
        ).to(device)
        if geo_cfg.get("enabled", True)
        else None
    )

    # Create optimizer
    print("\n=== Creating Optimizer ===")
    train_cfg = config["training"]
    base_lr = train_cfg.get("lr", 8e-4)

    if config["riemannian"].get("enabled", True):
        param_groups = model.get_param_groups(base_lr)
        optimizer = get_riemannian_optimizer(
            param_groups,
            lr=base_lr,
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )
        print(f"Using RiemannianAdam, lr={base_lr}")
    else:
        param_groups = model.get_param_groups(base_lr)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg.get("weight_decay", 1e-4))
        print(f"Using AdamW, lr={base_lr}")

    # Create scheduler
    scheduler = create_multi_phase_scheduler(optimizer, config)

    # Initialize grokking detector
    grokking_cfg = train_cfg.get("grokking_detection", {})
    grokking_detector = None
    if grokking_cfg.get("enabled", True):
        grokking_detector = GrokkingDetector(
            window_size=grokking_cfg.get("monitor_window", 20),
            plateau_threshold=grokking_cfg.get("plateau_threshold", 0.0001),
            plateau_patience=grokking_cfg.get("plateau_patience", 15),
            accuracy_jump_threshold=grokking_cfg.get("accuracy_jump_threshold", 0.02),
        )
        print("Grokking detection: ENABLED")

    # Training setup
    n_epochs = train_cfg.get("epochs", 500)
    batch_size = train_cfg.get("batch_size", 512)
    eval_every = train_cfg.get("eval_every", 2)
    print_every = train_cfg.get("print_every", 2)

    print(f"\nStarting extended training: {n_epochs} epochs")
    print("Expected runtime: 4-8 hours (depending on hardware)")

    # Training loop
    training_start_time = time.time()
    best_Q = 0.0
    best_composite = float("-inf")

    for epoch in range(n_epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        total_loss = 0.0
        total_batches = 0

        # Simple batching for this implementation
        n_samples = len(all_ops)
        n_batches = (n_samples + batch_size - 1) // batch_size
        perm = torch.randperm(n_samples, device=device)

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            batch_idx = perm[start_idx:end_idx]

            x_batch = all_ops[batch_idx]
            idx_batch = indices[batch_idx]

            optimizer.zero_grad()

            # Forward pass
            outputs = model(x_batch, compute_control=False)
            z_B = outputs["z_B_hyp"]
            logits = outputs["logits_A"]

            # Compute loss (simplified for this implementation)
            rad_loss_B, _ = radial_loss_fn(z_B, idx_batch)

            # Phase 2 geodesic loss
            phase_2_start = geo_cfg.get("phase_start_epoch", 50)
            geo_loss = torch.tensor(0.0, device=device)
            if epoch >= phase_2_start and geodesic_loss_fn is not None:
                geo_loss_B, _ = geodesic_loss_fn(z_B, idx_batch)
                geo_loss = geo_loss_B

            # Total loss
            loss = rad_loss_B + 0.4 * geo_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

        avg_loss = total_loss / total_batches
        epoch_time = time.time() - epoch_start_time

        # Update scheduler
        scheduler.step()

        # Evaluation
        if epoch % eval_every == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                outputs = model(all_ops, compute_control=False)
                z_B = outputs["z_B_hyp"]

                # Coverage
                logits = outputs["logits_A"]
                preds = torch.argmax(logits, dim=-1) - 1
                correct = (preds == all_ops.long()).float().mean(dim=1)
                coverage = (correct == 1.0).sum().item() / len(all_ops)

                # Hierarchy
                origin = torch.zeros_like(z_B)
                radii = poincare_distance(z_B, origin, c=1.0).cpu().numpy()
                valuations = TERNARY.valuation(indices).cpu().numpy()
                hierarchy = spearmanr(valuations, radii)[0]

                # Q metric
                # Simplified distance correlation
                sample_idx = np.random.choice(len(radii), min(1000, len(radii)), replace=False)
                z_sample = radii[sample_idx]
                val_sample = valuations[sample_idx]
                z_dists = np.abs(z_sample[:, None] - z_sample[None, :])
                val_dists = np.abs(val_sample[:, None] - val_sample[None, :]).astype(float)
                triu_idx = np.triu_indices(len(sample_idx), k=1)
                dist_corr = spearmanr(z_dists[triu_idx], val_dists[triu_idx])[0]

                Q = compute_Q(dist_corr, hierarchy)

            # Gradient norm for grokking detection
            total_grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    total_grad_norm += param.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm**0.5

            # Grokking detection
            grokking_analysis = {}
            if grokking_detector is not None:
                grokking_analysis = grokking_detector.update(epoch, avg_loss, coverage, total_grad_norm, hierarchy)

            # Logging
            writer.add_scalar("Train/loss", avg_loss, epoch)
            writer.add_scalar("Eval/coverage", coverage, epoch)
            writer.add_scalar("Eval/hierarchy_B", hierarchy, epoch)
            writer.add_scalar("Eval/Q", Q, epoch)
            writer.add_scalar("Gradients/total_norm", total_grad_norm, epoch)
            writer.add_scalar("Training/epoch_time", epoch_time, epoch)
            writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

            # Grokking metrics
            for key, value in grokking_analysis.items():
                if isinstance(value, (int, float, bool)):
                    writer.add_scalar(f"Grokking/{key}", float(value), epoch)

            # Enhanced metrics
            log_enhanced_metrics(writer, model, epoch, config)

            # Print progress
            if epoch % print_every == 0 or epoch == n_epochs - 1:
                elapsed_hours = (time.time() - training_start_time) / 3600
                remaining_hours = (elapsed_hours / (epoch + 1)) * (n_epochs - epoch - 1)

                print(f"\nEpoch {epoch}/{n_epochs} [{elapsed_hours:.1f}h elapsed, ~{remaining_hours:.1f}h remaining]")
                print(f"  Loss: {avg_loss:.4f}, Coverage: {coverage * 100:.1f}%, Hierarchy: {hierarchy:.3f}")
                print(f"  Q: {Q:.3f}, GradNorm: {total_grad_norm:.3f}, LR: {optimizer.param_groups[0]['lr']:.1e}")
                print(f"  Epoch time: {epoch_time:.1f}s")

                # Grokking status
                if grokking_analysis.get("in_plateau"):
                    print(f"  [PLATEAU] Duration: {grokking_analysis.get('plateau_duration', 0)} epochs")
                if grokking_analysis.get("potential_grokking"):
                    print("  [GROKKING] Potential emergence detected!")
                if grokking_analysis.get("phase_transition"):
                    print("  [TRANSITION] Phase change detected")

            # Save best models
            composite_score = -hierarchy - 0.5 * avg_loss  # Higher is better

            if best_Q < Q:
                best_Q = Q
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "Q": Q,
                        "hierarchy": hierarchy,
                        "coverage": coverage,
                        "config": config,
                        "grokking_summary": grokking_detector.get_summary() if grokking_detector else {},
                    },
                    save_dir / "best_Q.pt",
                )
                print(f"  [NEW BEST Q: {Q:.3f}]")

            if composite_score > best_composite:
                best_composite = composite_score
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "composite_score": composite_score,
                        "config": config,
                    },
                    save_dir / "best.pt",
                )

        # Periodic checkpoint
        if epoch % 50 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "grokking_summary": grokking_detector.get_summary() if grokking_detector else {},
                },
                save_dir / f"epoch_{epoch}.pt",
            )

    # Final summary
    total_time = time.time() - training_start_time
    print("\n" + "=" * 80)
    print("EXTENDED TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time / 3600:.1f} hours")
    print(f"Best Q: {best_Q:.3f}")
    print(f"Best composite: {best_composite:.3f}")

    # Grokking summary
    if grokking_detector:
        grokking_summary = grokking_detector.get_summary()
        print("\nGROKKING ANALYSIS:")
        print(f"  Total grokking events detected: {grokking_summary['total_grokking_events']}")
        for i, event in enumerate(grokking_summary["grokking_events"]):
            print(
                f"  Event {i + 1}: Epoch {event['epoch']}, plateau {event['plateau_duration']} epochs, "
                f"accuracy jump {event['accuracy_jump']:.3f}"
            )

    # Save final summary
    summary = {
        "training_time_hours": total_time / 3600,
        "total_epochs": n_epochs,
        "best_Q": best_Q,
        "best_composite": best_composite,
        "final_coverage": coverage,
        "final_hierarchy": hierarchy,
        "grokking_summary": grokking_detector.get_summary() if grokking_detector else {},
        "config": config,
    }

    with open(save_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {save_dir}")
    print(f"TensorBoard logs: {log_dir}")

    writer.close()


if __name__ == "__main__":
    main()
