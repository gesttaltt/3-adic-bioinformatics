#!/usr/bin/env python3
"""Training with Advanced Module Integration.

This script demonstrates how to integrate the Tier 4 advanced modules
into the VAE training pipeline:

1. K-FAC Optimizer: Second-order optimization for faster convergence
2. Information Geometry: Fisher information tracking for diagnostics
3. Tropical Geometry: Network expressivity analysis
4. Meta-Learning: Few-shot adaptation hooks

Based on train_homeostatic_rich.py with advanced module integration.

Usage:
    python src/scripts/experiments/epsilon_vae/train_with_advanced_modules.py
    python src/scripts/experiments/epsilon_vae/train_with_advanced_modules.py --use-kfac
    python src/scripts/experiments/epsilon_vae/train_with_advanced_modules.py --analyze-expressivity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.models import TernaryVAEV5_11_PartialFreeze

# =============================================================================
# Advanced Module Imports (Optional - gracefully degrade if unavailable)
# =============================================================================

# Information Geometry
try:
    from src.information import (
        FisherInformationEstimator,
        InformationGeometricAnalyzer,
        KFACOptimizer,
        NaturalGradientOptimizer,
    )

    INFORMATION_AVAILABLE = True
except ImportError:
    INFORMATION_AVAILABLE = False
    print("Warning: Information geometry module not available")

# Tropical Geometry
try:
    from src.tropical import TropicalNNAnalyzer, TropicalSemiring

    TROPICAL_AVAILABLE = True
except ImportError:
    TROPICAL_AVAILABLE = False
    print("Warning: Tropical geometry module not available")

# Meta-Learning
try:
    from src.meta import MAML, PAdicTaskSampler

    META_AVAILABLE = True
except ImportError:
    META_AVAILABLE = False
    print("Warning: Meta-learning module not available")


# =============================================================================
# Training Configuration
# =============================================================================


class AdvancedTrainingConfig:
    """Configuration for advanced module integration."""

    # Base training
    epochs: int = 100
    batch_size: int = 512
    learning_rate: float = 1e-3

    # Advanced modules
    use_kfac: bool = False
    kfac_damping: float = 1e-3
    kfac_update_freq: int = 10

    # Fisher information tracking
    track_fisher: bool = False
    fisher_log_freq: int = 10

    # Tropical analysis
    analyze_expressivity: bool = False
    expressivity_log_freq: int = 25

    # Loss weights
    hierarchy_weight: float = 5.0
    coverage_weight: float = 1.0
    richness_weight: float = 2.0
    separation_weight: float = 3.0

    # Checkpoint
    checkpoint_dir: str = "checkpoints/advanced_integration"
    run_name: str = "advanced_v1"


# =============================================================================
# Advanced Training Loop
# =============================================================================


def create_optimizer(
    model: nn.Module,
    config: AdvancedTrainingConfig,
) -> optim.Optimizer:
    """Create optimizer with optional K-FAC integration."""
    if config.use_kfac and INFORMATION_AVAILABLE:
        print("Using K-FAC optimizer (second-order optimization)")
        return KFACOptimizer(
            model,
            lr=config.learning_rate,
            damping=config.kfac_damping,
            cov_ema_decay=0.95,
            update_freq=config.kfac_update_freq,
        )
    else:
        print("Using AdamW optimizer")
        return optim.AdamW(model.parameters(), lr=config.learning_rate)


def analyze_network_expressivity(
    model: nn.Module,
    epoch: int,
    device: str = "cuda",
) -> dict:
    """Analyze network expressivity using tropical geometry."""
    if not TROPICAL_AVAILABLE:
        return {}

    try:
        # Analyze encoder expressivity
        analyzer = TropicalNNAnalyzer(model.encoder_a)
        expressivity = analyzer.analyze_expressivity()

        # Estimate linear regions (sampling-based)
        n_regions = analyzer.compute_linear_regions(
            input_bounds=(-1, 1),
            n_samples=10000,
            sampling_method="random",
        )

        metrics = {
            "encoder_depth": expressivity.get("depth", 0),
            "max_linear_regions": expressivity.get("max_linear_regions_upper_bound", 0),
            "empirical_regions": n_regions,
            "total_parameters": expressivity.get("total_parameters", 0),
        }

        print(f"\n[Epoch {epoch}] Tropical Expressivity Analysis:")
        print(f"  Encoder depth: {metrics['encoder_depth']}")
        print(f"  Empirical linear regions: {metrics['empirical_regions']:,}")
        print(f"  Upper bound: {metrics['max_linear_regions']:,}")

        return metrics

    except Exception as e:
        print(f"Expressivity analysis failed: {e}")
        return {}


def analyze_fisher_information(
    model: nn.Module,
    dataloader: DataLoader,
    epoch: int,
    analyzer: object | None = None,
) -> dict:
    """Analyze Fisher information for training diagnostics."""
    if not INFORMATION_AVAILABLE or analyzer is None:
        return {}

    try:
        metrics = analyzer.analyze_step(dataloader, n_samples=100)

        print(f"\n[Epoch {epoch}] Fisher Information Analysis:")
        print(f"  Condition number: {metrics.get('condition_number', 0):.2f}")
        print(f"  Trace: {metrics.get('trace', 0):.4f}")
        print(f"  Effective dimensionality: {metrics.get('effective_dim', 0):.1f}")

        # Flatness (generalization proxy)
        flatness = analyzer.flatness_measure(dataloader, epsilon=0.01, n_directions=10)
        metrics["sharpness"] = flatness.get("sharpness", 0)
        print(f"  Sharpness: {metrics['sharpness']:.4f}")

        return metrics

    except Exception as e:
        print(f"Fisher analysis failed: {e}")
        return {}


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    epoch: int,
    config: AdvancedTrainingConfig,
    fisher_analyzer: object | None = None,
) -> dict:
    """Train one epoch with advanced module integration."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for _batch_idx, (indices, ops) in enumerate(dataloader):
        indices = indices.to(device)
        ops = ops.to(device)

        optimizer.zero_grad()

        # Forward pass
        out = model(ops, compute_control=False)

        # Compute loss
        loss_dict = loss_fn(
            z_hyp=out["z_B_hyp"],
            indices=indices,
            logits=out["logits"],
            targets=ops,
        )

        loss = loss_dict["total"]
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches

    # Periodic advanced analysis
    metrics = {"loss": avg_loss}

    if config.track_fisher and epoch % config.fisher_log_freq == 0:
        fisher_metrics = analyze_fisher_information(model, dataloader, epoch, fisher_analyzer)
        metrics.update(fisher_metrics)

    if config.analyze_expressivity and epoch % config.expressivity_log_freq == 0:
        expr_metrics = analyze_network_expressivity(model, epoch, device)
        metrics.update(expr_metrics)

    return metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """Evaluate model on validation data."""
    model.eval()
    all_radii = []
    all_valuations = []
    correct = 0
    total = 0

    with torch.no_grad():
        for indices, ops in dataloader:
            indices = indices.to(device)
            ops = ops.to(device)

            out = model(ops, compute_control=False)

            radii = out["z_B_hyp"].norm(dim=-1)
            valuations = TERNARY.valuation(indices)

            all_radii.extend(radii.cpu().numpy())
            all_valuations.extend(valuations.cpu().numpy())

            preds = out["logits"].argmax(dim=-1)
            correct += (preds == ops.argmax(dim=-1)).sum().item()
            total += ops.shape[0]

    # Compute metrics
    all_radii = np.array(all_radii)
    all_valuations = np.array(all_valuations)

    hierarchy, _ = spearmanr(all_valuations, all_radii)
    coverage = correct / total

    # Richness: average within-level variance
    richness = 0.0
    for v in range(10):
        mask = all_valuations == v
        if mask.sum() > 1:
            richness += np.var(all_radii[mask])
    richness /= 10

    return {
        "coverage": coverage,
        "hierarchy": hierarchy,
        "richness": richness,
    }


# =============================================================================
# Loss Function
# =============================================================================


class RichHierarchyLoss(nn.Module):
    """Loss function balancing hierarchy, coverage, and richness."""

    def __init__(
        self,
        inner_radius: float = 0.1,
        outer_radius: float = 0.9,
        hierarchy_weight: float = 5.0,
        coverage_weight: float = 1.0,
        richness_weight: float = 2.0,
        separation_weight: float = 3.0,
    ):
        super().__init__()
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.hierarchy_weight = hierarchy_weight
        self.coverage_weight = coverage_weight
        self.richness_weight = richness_weight
        self.separation_weight = separation_weight
        self.max_valuation = 9

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
    ) -> dict:
        device = z_hyp.device
        radii = z_hyp.norm(dim=-1)
        valuations = TERNARY.valuation(indices).long().to(device)

        # 1. Hierarchy loss
        hierarchy_loss = torch.tensor(0.0, device=device)
        unique_vals = torch.unique(valuations)
        for v in unique_vals:
            mask = valuations == v
            if mask.sum() > 0:
                mean_r = radii[mask].mean()
                target_r = self.target_radii[v]
                hierarchy_loss = hierarchy_loss + (mean_r - target_r) ** 2
        hierarchy_loss = hierarchy_loss / len(unique_vals)

        # 2. Coverage loss (cross-entropy)
        coverage_loss = nn.functional.cross_entropy(logits, targets.argmax(dim=-1))

        # 3. Richness preservation
        richness_loss = torch.tensor(0.0, device=device)
        for v in unique_vals:
            mask = valuations == v
            if mask.sum() > 1:
                level_radii = radii[mask]
                variance = level_radii.var()
                richness_loss = richness_loss - variance  # Maximize variance

        # 4. Separation loss
        separation_loss = torch.tensor(0.0, device=device)
        sorted_vals = torch.sort(unique_vals)[0]
        for i in range(len(sorted_vals) - 1):
            v1, v2 = sorted_vals[i], sorted_vals[i + 1]
            mask1 = valuations == v1
            mask2 = valuations == v2
            if mask1.sum() > 0 and mask2.sum() > 0:
                mean1 = radii[mask1].mean()
                mean2 = radii[mask2].mean()
                separation_loss = separation_loss + torch.relu(mean2 - mean1 + 0.02)

        total = (
            self.hierarchy_weight * hierarchy_loss
            + self.coverage_weight * coverage_loss
            + self.richness_weight * richness_loss
            + self.separation_weight * separation_loss
        )

        return {
            "total": total,
            "hierarchy": hierarchy_loss.item(),
            "coverage": coverage_loss.item(),
            "richness": richness_loss.item(),
            "separation": separation_loss.item(),
        }


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train with advanced modules")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use-kfac", action="store_true", help="Use K-FAC optimizer")
    parser.add_argument("--track-fisher", action="store_true", help="Track Fisher info")
    parser.add_argument("--analyze-expressivity", action="store_true", help="Tropical analysis")
    parser.add_argument("--run-name", type=str, default="advanced_v1")
    args = parser.parse_args()

    # Config
    config = AdvancedTrainingConfig()
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.use_kfac = args.use_kfac
    config.track_fisher = args.track_fisher
    config.analyze_expressivity = args.analyze_expressivity
    config.run_name = args.run_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Advanced modules available:")
    print(f"  Information Geometry: {INFORMATION_AVAILABLE}")
    print(f"  Tropical Geometry: {TROPICAL_AVAILABLE}")
    print(f"  Meta-Learning: {META_AVAILABLE}")

    # Data
    print("\nGenerating ternary operations...")
    indices, ops = generate_all_ternary_operations()
    dataset = TensorDataset(indices, ops)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Model
    print("Creating model...")
    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=16,
        hidden_dim=64,
        max_radius=0.99,
        curvature=1.0,
        use_controller=False,
        use_dual_projection=True,
    ).to(device)

    # Optimizer (with optional K-FAC)
    optimizer = create_optimizer(model, config)

    # Loss
    loss_fn = RichHierarchyLoss(
        hierarchy_weight=config.hierarchy_weight,
        coverage_weight=config.coverage_weight,
        richness_weight=config.richness_weight,
        separation_weight=config.separation_weight,
    ).to(device)

    # Fisher analyzer (if tracking)
    fisher_analyzer = None
    if config.track_fisher and INFORMATION_AVAILABLE:
        fisher_analyzer = InformationGeometricAnalyzer(model, track_eigenvalues=True)

    # Initial expressivity analysis
    if config.analyze_expressivity and TROPICAL_AVAILABLE:
        print("\nInitial expressivity analysis...")
        analyze_network_expressivity(model, epoch=0, device=device)

    # Training loop
    print(f"\nStarting training for {config.epochs} epochs...")
    best_hierarchy = 0.0

    for epoch in range(1, config.epochs + 1):
        metrics = train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, config, fisher_analyzer)

        # Evaluate
        eval_metrics = evaluate(model, dataloader, device)

        print(
            f"Epoch {epoch:3d} | Loss: {metrics['loss']:.4f} | "
            f"Cov: {eval_metrics['coverage']:.4f} | "
            f"Hier: {eval_metrics['hierarchy']:.4f} | "
            f"Rich: {eval_metrics['richness']:.6f}"
        )

        # Save best
        if abs(eval_metrics["hierarchy"]) > abs(best_hierarchy):
            best_hierarchy = eval_metrics["hierarchy"]
            checkpoint_path = Path(config.checkpoint_dir) / config.run_name / "best.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": eval_metrics,
                    "config": vars(config),
                },
                checkpoint_path,
            )
            print(f"  -> Saved best model (hierarchy: {best_hierarchy:.4f})")

    print(f"\nTraining complete! Best hierarchy: {best_hierarchy:.4f}")


if __name__ == "__main__":
    main()
