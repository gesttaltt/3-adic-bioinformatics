# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Ternary VAE Training Script (Canonical V5.11 Architecture).

Architecture:
- Frozen encoder_A (100% coverage preserved)
- Trainable encoder_B (learns 3-adic structure)
- Dual HyperbolicProjection for radial hierarchy
- PAdicGeodesicLoss (hierarchy + correlation)

Usage:
    python src/scripts/train/train.py
    python src/scripts/train/train.py --config src/configs/ternary.yaml
    python src/scripts/train/train.py --epochs 100 --lr 1e-3

    # With adaptive curriculum (default)
    python src/scripts/train/train.py --option_c --dual_projection

    # Disable adaptive curriculum
    python src/scripts/train/train.py --option_c --dual_projection --no_adaptive

Key features:
- Option C architecture: frozen coverage + trainable structure
- Dual projection: separate hyperbolic projection per VAE
- Stratified sampling: ensures high-valuation points in batches
- Adaptive curriculum (v1.1): tau freezes when hierarchy threshold reached
- Early stopping: stops when no improvement for patience epochs
- Composite scoring: best model selected by hierarchy + loss balance
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from scipy.stats import spearmanr
from torch.utils.tensorboard import SummaryWriter

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR
from src.core import TERNARY
from src.data.generation import generate_all_ternary_operations
from src.geometry import get_riemannian_optimizer, poincare_distance
from src.losses import CombinedZeroStructureLoss, GlobalRankLoss, PAdicGeodesicLoss, RadialHierarchyLoss
from src.models import HomeostasisController, TernaryVAEV5_11, TernaryVAEV5_11_PartialFreeze


def parse_args():
    parser = argparse.ArgumentParser(description="Train V5.11 Ternary VAE")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument(
        "--v5_5_checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "v5_5" / "latest.pt"),
        help="Path to v5.5 checkpoint",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(CHECKPOINTS_DIR / "ternary"),
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--use_controller",
        action="store_true",
        default=False,
        help="Use differentiable controller",
    )
    parser.add_argument("--curvature", type=float, default=1.0, help="Hyperbolic curvature")
    parser.add_argument(
        "--max_radius",
        type=float,
        default=0.95,
        help="Maximum Poincare ball radius",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument(
        "--radial_weight",
        type=float,
        default=2.0,
        help="Weight for radial loss (always active)",
    )
    parser.add_argument(
        "--margin_weight",
        type=float,
        default=1.0,
        help="Weight for radial margin loss",
    )
    parser.add_argument(
        "--no_stratified",
        action="store_true",
        default=False,
        help="Disable stratified sampling (use random)",
    )
    # Option C arguments
    parser.add_argument(
        "--option_c",
        action="store_true",
        default=False,
        help="Use Option C (partial freeze, encoder_B trainable)",
    )
    parser.add_argument(
        "--encoder_b_lr_scale",
        type=float,
        default=0.1,
        help="Learning rate scale for encoder_B in Option C",
    )
    parser.add_argument(
        "--dual_projection",
        action="store_true",
        default=False,
        help="Use separate projections for VAE-A and VAE-B",
    )
    # Adaptive curriculum termination (v1.1)
    parser.add_argument(
        "--hierarchy_threshold",
        type=float,
        default=-0.70,
        help="Radial hierarchy threshold to freeze tau (default: -0.70)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--min_epochs",
        type=int,
        default=30,
        help="Minimum epochs before early stopping can trigger",
    )
    parser.add_argument(
        "--no_adaptive",
        action="store_true",
        default=False,
        help="Disable adaptive curriculum (use fixed tau schedule)",
    )
    # Projection layer capacity (v1.2)
    parser.add_argument(
        "--projection_hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension for projection networks (default: 64)",
    )
    parser.add_argument(
        "--projection_layers",
        type=int,
        default=1,
        help="Number of hidden layers in projection (1=shallow, 2+=deep)",
    )
    # Structural constraints (v1.3)
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-3,
        help="Weight decay for optimizer (default: 1e-3)",
    )
    parser.add_argument(
        "--projection_dropout",
        type=float,
        default=0.1,
        help="Dropout rate in projection networks (default: 0.1)",
    )
    parser.add_argument(
        "--rank_loss_weight",
        type=float,
        default=1.0,
        help="Weight for global rank loss (default: 1.0)",
    )
    parser.add_argument(
        "--no_rank_loss",
        action="store_true",
        default=False,
        help="Disable global rank loss",
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=2000,
        help="Number of pairs for geodesic loss (default: 2000)",
    )
    parser.add_argument(
        "--learnable_weights",
        action="store_true",
        default=False,
        help="Use controller-learned loss weights (requires --use_controller)",
    )
    # Progressive unfreezing (v5.11.6)
    parser.add_argument(
        "--progressive_unfreeze",
        action="store_true",
        default=False,
        help="Progressively unfreeze encoder_A during training",
    )
    parser.add_argument(
        "--unfreeze_start_epoch",
        type=int,
        default=5,
        help="Epoch to start unfreezing encoder_A (default: 5)",
    )
    parser.add_argument(
        "--unfreeze_warmup_epochs",
        type=int,
        default=5,
        help="Epochs to ramp up encoder_A LR from 0 to target (default: 5)",
    )
    parser.add_argument(
        "--encoder_a_lr_scale",
        type=float,
        default=0.05,
        help="Final LR scale for encoder_A (default: 0.05)",
    )
    # Homeostatic control (v5.11.7)
    parser.add_argument(
        "--homeostasis",
        action="store_true",
        default=False,
        help="Enable hierarchical homeostatic freeze/unfreeze control",
    )
    parser.add_argument(
        "--coverage_freeze_threshold",
        type=float,
        default=0.995,
        help="Freeze encoder_A when coverage drops below this (default: 0.995)",
    )
    parser.add_argument(
        "--homeostasis_warmup",
        type=int,
        default=5,
        help="Epochs before homeostasis activates (default: 5)",
    )
    parser.add_argument(
        "--hysteresis_epochs",
        type=int,
        default=3,
        help="Minimum epochs between freeze state changes (default: 3)",
    )
    # Q-gated annealing (v5.11.8)
    parser.add_argument(
        "--enable_annealing",
        action="store_true",
        default=True,
        help="Enable Q-gated threshold annealing (default: True)",
    )
    parser.add_argument(
        "--no_annealing",
        action="store_true",
        default=False,
        help="Disable Q-gated threshold annealing",
    )
    parser.add_argument(
        "--annealing_step",
        type=float,
        default=0.005,
        help="How much to relax thresholds per successful cycle (default: 0.005)",
    )
    parser.add_argument(
        "--coverage_floor",
        type=float,
        default=0.95,
        help="Never relax coverage threshold below this (default: 0.95)",
    )
    # Zero-structure loss (v5.11.9)
    parser.add_argument(
        "--zero_structure",
        action="store_true",
        default=False,
        help="Enable zero-structure loss (V5.11.9)",
    )
    parser.add_argument(
        "--zero_valuation_weight",
        type=float,
        default=1.0,
        help="Weight for zero-valuation loss (default: 1.0)",
    )
    parser.add_argument(
        "--zero_sparsity_weight",
        type=float,
        default=0.5,
        help="Weight for zero-sparsity loss (default: 0.5)",
    )
    # Riemannian optimizer (v5.11.10)
    parser.add_argument(
        "--riemannian",
        action="store_true",
        default=False,
        help="Use RiemannianAdam optimizer (geoopt) instead of AdamW",
    )
    # ManifoldParameter integration (v5.11.11)
    parser.add_argument(
        "--learnable_curvature",
        action="store_true",
        default=False,
        help="Make hyperbolic curvature a learnable parameter (geoopt)",
    )
    parser.add_argument(
        "--manifold_aware",
        action="store_true",
        default=False,
        help="Return ManifoldParameter from projection for Riemannian gradients",
    )
    # Geometric Alignment (Phase 1)
    parser.add_argument(
        "--geometric_weight",
        type=float,
        default=0.0,
        help="Weight for geometric alignment loss (default: 0.0 = disabled)",
    )
    parser.add_argument(
        "--geometric_symmetry",
        type=str,
        default="point_24",
        help="Symmetry group for geometric alignment (default: point_24)",
    )
    # Phase 2: Drug Interaction
    parser.add_argument(
        "--drug_sim_weight",
        type=float,
        default=0.0,
        help="Weight for drug interaction/similarity loss",
    )
    return parser.parse_args()


class AdaptiveCurriculum:
    """Adaptive curriculum with threshold-based tau freezing and early stopping.

    Key features:
    1. Tau freezes when hierarchy threshold is reached (stops pushing curriculum)
    2. Early stopping triggers after patience epochs without improvement
    3. Best model selected by composite score (hierarchy + loss balance)
    """

    def __init__(
        self,
        hierarchy_threshold: float = -0.62,
        patience: int = 20,
        min_epochs: int = 30,
        n_epochs: int = 100,
        enabled: bool = True,
    ):
        self.hierarchy_threshold = hierarchy_threshold
        self.patience = patience
        self.min_epochs = min_epochs
        self.n_epochs = n_epochs
        self.enabled = enabled

        # State
        self.tau_frozen = False
        self.frozen_tau = None
        self.frozen_epoch = None
        self.best_score = float("-inf")  # Higher is better (composite)
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.should_stop = False

    def compute_tau(self, epoch: int) -> float:
        """Compute tau for current epoch, respecting frozen state."""
        if self.tau_frozen and self.frozen_tau is not None:
            return self.frozen_tau
        # Default schedule: 0 -> 1 over 70% of training
        return min(1.0, epoch / (self.n_epochs * 0.7))

    def compute_composite_score(self, radial_corr: float, loss: float) -> float:
        """Compute composite score balancing hierarchy and loss.

        Score = -radial_corr - 0.5 * loss

        - radial_corr is negative (more negative = better), so -radial_corr is positive
        - loss is positive (lower = better), so we subtract it
        - 0.5 weight balances the two (tunable)
        """
        return -radial_corr - 0.5 * loss

    def update(self, epoch: int, radial_corr: float, loss: float) -> dict:
        """Update curriculum state based on current metrics.

        Returns dict with status info.
        """
        if not self.enabled:
            return {"tau_frozen": False, "should_stop": False}

        status = {
            "tau_frozen": self.tau_frozen,
            "frozen_tau": self.frozen_tau,
            "should_stop": False,
            "new_best": False,
            "triggered_freeze": False,
            "triggered_stop": False,
        }

        # Check if we should freeze tau
        if not self.tau_frozen and radial_corr <= self.hierarchy_threshold:
            self.tau_frozen = True
            self.frozen_tau = self.compute_tau(epoch)
            self.frozen_epoch = epoch
            status["triggered_freeze"] = True
            status["tau_frozen"] = True
            status["frozen_tau"] = self.frozen_tau

        # Compute composite score
        score = self.compute_composite_score(radial_corr, loss)

        # Check for improvement
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            status["new_best"] = True
        else:
            self.epochs_without_improvement += 1

        # Check for early stopping
        if epoch >= self.min_epochs and self.epochs_without_improvement >= self.patience:
            self.should_stop = True
            status["should_stop"] = True
            status["triggered_stop"] = True

        return status


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_full_dataset(device: str):
    """Create full dataset of all 19,683 ternary operations."""
    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, dtype=torch.float32, device=device)
    indices = torch.arange(len(operations), device=device)
    return x, indices


def create_stratified_indices(indices: torch.Tensor, batch_size: int, device: str):
    """Create stratified batch indices ensuring all valuation levels represented.

    V5.11.2 FIX: High-valuation points are extremely rare (v≥7 is ~9 out of 19683).
    Random sampling means most batches have NO high-valuation points.

    Solution: Stratified sampling - ensure each batch contains points from
    all valuation levels, with oversampling of rare high-valuation points.

    Args:
        indices: All operation indices
        batch_size: Target batch size
        device: Torch device

    Returns:
        List of batch index tensors, each containing stratified samples
    """
    n_samples = len(indices)
    valuations = TERNARY.valuation(indices).cpu().numpy()

    # Group indices by valuation level
    valuation_groups = {}
    for i, v in enumerate(valuations):
        v = int(v)
        if v not in valuation_groups:
            valuation_groups[v] = []
        valuation_groups[v].append(i)

    # Convert to tensors
    for v in valuation_groups:
        valuation_groups[v] = torch.tensor(valuation_groups[v], device=device)

    # Compute samples per valuation level per batch
    # High valuation = more oversampling
    max(valuation_groups.keys())

    # Allocation: reserve 20% of batch for high-v (v≥4), rest proportional
    high_v_budget = int(batch_size * 0.2)  # 20% for high-valuation
    low_v_budget = batch_size - high_v_budget  # 80% for low-valuation

    # High-v levels: v=4,5,6,7,8,9 (or whatever exists)
    high_v_levels = [v for v in valuation_groups if v >= 4]
    low_v_levels = [v for v in valuation_groups if v < 4]

    batches = []
    n_batches = (n_samples + batch_size - 1) // batch_size

    for _ in range(n_batches):
        batch_indices = []

        # Sample from high-valuation levels (with replacement if needed)
        if high_v_levels:
            per_high_v = max(1, high_v_budget // len(high_v_levels))
            for v in high_v_levels:
                group = valuation_groups[v]
                n_to_sample = min(per_high_v, len(group))
                if len(group) <= n_to_sample:
                    # Take all, with replacement if oversampling needed
                    sample_idx = torch.randint(0, len(group), (per_high_v,), device=device)
                else:
                    sample_idx = torch.randperm(len(group), device=device)[:n_to_sample]
                batch_indices.append(group[sample_idx])

        # Sample from low-valuation levels (proportional to size)
        if low_v_levels:
            total_low = sum(len(valuation_groups[v]) for v in low_v_levels)
            for v in low_v_levels:
                group = valuation_groups[v]
                n_to_sample = max(1, int(low_v_budget * len(group) / total_low))
                sample_idx = torch.randint(0, len(group), (n_to_sample,), device=device)
                batch_indices.append(group[sample_idx])

        # Combine and shuffle
        batch = torch.cat(batch_indices)
        # Trim to exact batch size or pad if needed
        if len(batch) > batch_size:
            batch = batch[torch.randperm(len(batch), device=device)[:batch_size]]
        elif len(batch) < batch_size:
            # Pad with random samples
            extra = torch.randint(0, n_samples, (batch_size - len(batch),), device=device)
            batch = torch.cat([batch, extra])

        batches.append(batch)

    return batches


def compute_metrics(model, x, indices, geodesic_loss_fn, radial_loss_fn, device):
    """Compute comprehensive metrics."""
    model.eval()

    with torch.no_grad():
        outputs = model(x, compute_control=False)
        z_A_hyp = outputs["z_A_hyp"]
        z_B_hyp = outputs["z_B_hyp"]

        # Geodesic loss
        geo_loss_A, geo_metrics_A = geodesic_loss_fn(z_A_hyp, indices)
        geo_loss_B, geo_metrics_B = geodesic_loss_fn(z_B_hyp, indices)

        # Radial loss
        rad_loss_A, rad_metrics_A = radial_loss_fn(z_A_hyp, indices)
        rad_loss_B, rad_metrics_B = radial_loss_fn(z_B_hyp, indices)

        # V5.12.2: Radial distribution using hyperbolic distance
        origin = torch.zeros_like(z_A_hyp)
        radii_A = poincare_distance(z_A_hyp, origin, c=1.0).cpu().numpy()
        radii_B = poincare_distance(z_B_hyp, origin, c=1.0).cpu().numpy()
        valuations = TERNARY.valuation(indices).cpu().numpy()

        # Radial hierarchy correlation (should be NEGATIVE)
        radial_corr_A = spearmanr(valuations, radii_A)[0]
        radial_corr_B = spearmanr(valuations, radii_B)[0]

        # Radius range by valuation
        radius_v0 = radii_A[valuations == 0].mean() if (valuations == 0).any() else 0
        radius_v9 = radii_A[valuations == 9].mean() if (valuations == 9).any() else 0
        radius_range = radii_A.max() - radii_A.min()

        # Coverage check (using frozen decoder with MEAN, not sampled)
        mu_A = outputs["mu_A"]
        logits_A = model.decoder_A(mu_A)
        preds = torch.argmax(logits_A, dim=-1) - 1
        targets = x.long()
        correct = (preds == targets).float().mean(dim=1)
        coverage = (correct == 1.0).sum().item() / len(x)

    return {
        "coverage": coverage,
        "geo_loss_A": geo_loss_A.item(),
        "geo_loss_B": geo_loss_B.item(),
        "rad_loss_A": rad_loss_A.item(),
        "rad_loss_B": rad_loss_B.item(),
        "radial_corr_A": radial_corr_A,
        "radial_corr_B": radial_corr_B,
        "mean_radius_A": radii_A.mean(),
        "mean_radius_B": radii_B.mean(),
        "radius_min_A": radii_A.min(),
        "radius_max_A": radii_A.max(),
        "radius_range_A": radius_range,
        "radius_v0": radius_v0,
        "radius_v9": radius_v9,
        "distance_corr_A": geo_metrics_A.get("distance_correlation", 0),
        "distance_corr_B": geo_metrics_B.get("distance_correlation", 0),
    }


def train_epoch(
    model,
    optimizer,
    x,
    indices,
    geodesic_loss_fn,
    radial_loss_fn,
    batch_size,
    epoch,
    tau,
    radial_weight,
    device,
    use_stratified=True,
    rank_loss_fn=None,
    rank_loss_weight=1.0,
    use_learnable_weights=False,
    zero_structure_loss_fn=None,
    zero_structure_weight=1.0,
    geometric_loss_fn=None,
    geometric_weight=0.0,
    drug_interaction_loss_fn=None,
    drug_interaction_weight=0.0,
):
    """Train one epoch.

    V5.11.1 FIX: Changed loss composition to always include radial loss.
    Instead of curriculum blending (which eliminates radial at tau=1),
    we use: total = geo_loss + radial_weight * rad_loss

    V5.11.2 FIX: Added stratified sampling to ensure high-valuation points
    are represented in every batch. Without this, v≥7 points (only ~9 in dataset)
    may never appear in most batches.

    V5.11.3 FIX: Added global rank loss to enforce monotonic radius ordering.
    This structural constraint forces the projection to learn hierarchy rather
    than fit noise when given extra capacity.

    V5.11.4 FIX: Learnable loss weights via controller.
    When use_learnable_weights=True, tau and radial_weight come from the
    controller's outputs, allowing the model to learn optimal capacity
    allocation on the Pareto frontier dynamically.

    tau now controls geodesic weight, but radial is always active.
    """
    model.train()

    n_samples = len(x)

    if use_stratified:
        # V5.11.2: Use stratified sampling for balanced valuation representation
        batches = create_stratified_indices(indices, batch_size, device)
        n_batches = len(batches)
    else:
        # Original random sampling
        n_batches = (n_samples + batch_size - 1) // batch_size
        perm = torch.randperm(n_samples, device=device)
        batches = None

    # Accumulator variables for epoch metrics
    total_loss = 0.0
    total_geo = 0.0
    total_rad = 0.0
    total_rank = 0.0
    total_zero = 0.0
    total_geo_align = 0.0
    total_drug = 0.0

    for i in range(n_batches):
        if use_stratified:
            batch_idx = batches[i]
        else:
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            batch_idx = perm[start:end]

        x_batch = x[batch_idx]
        idx_batch = indices[batch_idx]

        optimizer.zero_grad()

        # Forward pass
        outputs = model(x_batch, compute_control=model.use_controller)

        z_A_hyp = outputs["z_A_hyp"]
        z_B_hyp = outputs["z_B_hyp"]

        # Compute losses
        geo_loss_A, _ = geodesic_loss_fn(z_A_hyp, idx_batch)
        geo_loss_B, _ = geodesic_loss_fn(z_B_hyp, idx_batch)
        rad_loss_A, rad_metrics_A = radial_loss_fn(z_A_hyp, idx_batch)
        rad_loss_B, rad_metrics_B = radial_loss_fn(z_B_hyp, idx_batch)

        geo_loss = geo_loss_A + geo_loss_B
        rad_loss = rad_loss_A + rad_loss_B

        # V5.11.3: Global rank loss for structural constraint
        rank_loss = torch.tensor(0.0, device=device)
        if rank_loss_fn is not None:
            rank_loss_A, rank_metrics_A = rank_loss_fn(z_A_hyp, idx_batch)
            rank_loss_B, rank_metrics_B = rank_loss_fn(z_B_hyp, idx_batch)
            rank_loss = rank_loss_A + rank_loss_B

        # V5.11.9: Zero-structure loss for exploiting ternary zero semantics
        zero_loss = torch.tensor(0.0, device=device)
        if zero_structure_loss_fn is not None:
            zero_loss_A = zero_structure_loss_fn(z_A_hyp, x_batch)
            zero_loss_B = zero_structure_loss_fn(z_B_hyp, x_batch)
            zero_loss = zero_loss_A + zero_loss_B

        # V5.12: Geometric Alignment Loss (Phase 1)
        geo_align_loss = torch.tensor(0.0, device=device)
        if geometric_loss_fn is not None:
            # We align trainable High-Level structure (z_B) or both?
            # Proposal implies scaffold alignment. Let's align z_B_hyp as it is the "structure" VAE
            align_loss_A, _ = geometric_loss_fn(z_A_hyp)
            align_loss_B, _ = geometric_loss_fn(z_B_hyp)
            geo_align_loss = align_loss_A + align_loss_B

        # V5.13: Drug Interaction Loss (Phase 2)
        drug_loss = torch.tensor(0.0, device=device)
        if drug_interaction_loss_fn is not None and drug_interaction_weight > 0:
            # MOCK: Assuming interaction data is in batch metadata or simulating self-interaction for constraint
            # Real implementation would need x_batch to contain interaction labels
            # For now, we leave as 0 or implement a dummy constaint if needed
            pass

        # V5.11.1 FIX: Always include both losses

        # Radial loss is ALWAYS active (weighted by radial_weight)
        # Geodesic loss ramps up with tau
        # Early: low tau = focus on radial, some geodesic
        # Late: high tau = more geodesic, but radial still active

        # V5.11.4: Use learnable weights from controller if enabled
        if use_learnable_weights and "control" in outputs:
            ctrl = outputs["control"]
            # Controller outputs are [batch, 1] tensors, take mean for scalar weight
            learned_tau = ctrl["tau"].mean()
            learned_radial = ctrl["weight_radial"].mean()
            learned_geo = ctrl["weight_geodesic"].mean()

            # Core loss with learned weights
            core_loss = (
                learned_tau * geo_loss
                + learned_radial * rad_loss
                + rank_loss_weight * rank_loss
                + zero_structure_weight * zero_loss
                + geometric_weight * geo_align_loss
            )

            # V5.11.5: Q-preservation regularization
            # Prevent controller from collapsing to "easy path" (radial-only)
            # Constraint 1: tau must stay above minimum (ensure geodesic learning)
            tau_min = 0.3
            tau_penalty = torch.relu(tau_min - learned_tau) ** 2

            # Constraint 2: radial/geodesic ratio must stay bounded
            # This prevents radial from dominating even when tau is reasonable
            max_ratio = 3.0
            ratio = learned_radial / (learned_geo + 1e-6)
            ratio_penalty = torch.relu(ratio - max_ratio) ** 2

            # Regularization weight (tunable)
            q_reg_weight = 1.0
            q_regularization = q_reg_weight * (tau_penalty + ratio_penalty)

            total_batch_loss = core_loss + q_regularization
        else:
            total_batch_loss = (
                tau * geo_loss
                + radial_weight * rad_loss
                + rank_loss_weight * rank_loss
                + zero_structure_weight * zero_loss
                + geometric_weight * geo_align_loss
            )

        # Backward and optimize
        total_batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), 1.0)
        optimizer.step()

        total_loss += total_batch_loss.item()
        total_geo += geo_loss.item()
        total_rad += rad_loss.item()
        total_rank += rank_loss.item()
        total_zero += zero_loss.item()
        total_geo_align += geo_align_loss.item()
        total_drug += drug_loss.item()

    return {
        "loss": total_loss / n_batches,
        "geo_loss": total_geo / n_batches,
        "rad_loss": total_rad / n_batches,
        "rank_loss": total_rank / n_batches,
        "zero_loss": total_zero / n_batches,
        "geo_align_loss": total_geo_align / n_batches,
        "drug_loss": total_drug / n_batches,
    }


def main():
    args = parse_args()

    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Override with command line args
        for key in ["epochs", "lr", "batch_size", "curvature", "max_radius"]:
            if hasattr(args, key) and getattr(args, key) is not None and key in config:
                config[key] = getattr(args, key)
    else:
        config = vars(args)

    # Setup
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create save directory (differentiate Option A vs C, and dual projection)
    use_option_c = config.get("option_c", False)
    use_dual_proj = config.get("dual_projection", False)

    variant_parts = []
    if use_option_c:
        variant_parts.append("option_c")
    else:
        variant_parts.append("option_a")
    if use_dual_proj:
        variant_parts.append("dual")
    variant = "_".join(variant_parts)

    default_save_dir = str(CHECKPOINTS_DIR / f"ternary_{variant}")
    save_dir = Path(config.get("save_dir", default_save_dir))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup TensorBoard
    from src.config.paths import RUNS_DIR

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = RUNS_DIR / f"ternary_{variant}_{timestamp}"
    writer = SummaryWriter(log_dir=str(log_dir))

    # Create model (Option A or Option C, with optional dual projection)
    encoder_b_lr_scale = config.get("encoder_b_lr_scale", 0.1)
    proj_hidden_dim = config.get("projection_hidden_dim", 64)
    proj_layers = config.get("projection_layers", 1)
    proj_dropout = config.get("projection_dropout", 0.0)
    learnable_curvature = config.get("learnable_curvature", False)

    dual_str = " + DUAL PROJECTION" if use_dual_proj else ""
    capacity_str = f" (hidden={proj_hidden_dim}, layers={proj_layers}, dropout={proj_dropout})"
    curvature_str = " + LEARNABLE_C" if learnable_curvature else ""
    if use_option_c:
        print(
            f"\n=== Creating V5.11 Model (PartialFreeze: encoder_B trainable{dual_str}{capacity_str}{curvature_str}) ==="
        )
        model = TernaryVAEV5_11_PartialFreeze(
            latent_dim=16,
            hidden_dim=proj_hidden_dim,
            max_radius=config.get("max_radius", 0.95),
            curvature=config.get("curvature", 1.0),
            use_controller=config.get("use_controller", False),
            use_dual_projection=use_dual_proj,
            n_projection_layers=proj_layers,
            projection_dropout=proj_dropout,
            learnable_curvature=learnable_curvature,
            freeze_encoder_b=False,  # Key difference: encoder_B trains
            encoder_b_lr_scale=encoder_b_lr_scale,
        )
    else:
        print(f"\n=== Creating V5.11 Model (OPTION A: all encoders frozen{dual_str}{capacity_str}{curvature_str}) ===")
        model = TernaryVAEV5_11(
            latent_dim=16,
            hidden_dim=proj_hidden_dim,
            max_radius=config.get("max_radius", 0.95),
            curvature=config.get("curvature", 1.0),
            use_controller=config.get("use_controller", False),
            use_dual_projection=use_dual_proj,
            n_projection_layers=proj_layers,
            projection_dropout=proj_dropout,
            learnable_curvature=learnable_curvature,
        )

    # Load v5.5 checkpoint
    v5_5_path = Path(config.get("v5_5_checkpoint", str(CHECKPOINTS_DIR / "v5_5" / "latest.pt")))
    if not v5_5_path.exists():
        print(f"ERROR: v5.5 checkpoint not found at {v5_5_path}")
        sys.exit(1)

    model.load_v5_5_checkpoint(v5_5_path, device)

    # Count parameters
    param_counts = model.count_parameters()
    print("\nParameter counts:")
    print(f"  Frozen: {param_counts['frozen']:,}")
    print(f"  Projection: {param_counts['projection']:,}")
    print(f"  Controller: {param_counts['controller']:,}")
    print(f"  Trainable: {param_counts['trainable']:,}")
    print(f"  Total: {param_counts['total']:,}")

    # Create dataset
    print("\n=== Loading Dataset ===")
    x, indices = create_full_dataset(device)
    print(f"Dataset size: {len(x)}")

    # Create loss functions
    geodesic_loss_fn = PAdicGeodesicLoss(
        curvature=config.get("curvature", 1.0),
        max_target_distance=config.get("max_target_distance", 3.0),
        n_pairs=config.get("n_pairs", 2000),
    ).to(device)

    radial_loss_fn = RadialHierarchyLoss(
        inner_radius=config.get("inner_radius", 0.1),
        outer_radius=config.get("outer_radius", 0.85),
        margin_weight=config.get("margin_weight", 1.0),
        use_margin_loss=True,
    ).to(device)

    # Global rank loss (v1.3: structural constraint)
    use_rank_loss = not config.get("no_rank_loss", False)
    rank_loss_fn = GlobalRankLoss(temperature=0.1, n_pairs=2000).to(device) if use_rank_loss else None
    rank_loss_weight = config.get("rank_loss_weight", 1.0)

    # V5.11.9: Zero-structure loss for exploiting ternary zero semantics
    use_zero_structure = config.get("zero_structure", False)
    zero_structure_loss_fn = (
        CombinedZeroStructureLoss(
            valuation_weight=config.get("zero_valuation_weight", 1.0),
            sparsity_weight=config.get("zero_sparsity_weight", 0.5),
            inner_radius=config.get("inner_radius", 0.1),
            outer_radius=config.get("outer_radius", 0.85),
        ).to(device)
        if use_zero_structure
        else None
    )
    zero_structure_weight = config.get("zero_valuation_weight", 1.0) + config.get("zero_sparsity_weight", 0.5)

    # Radial weight (always active, not curriculum-blended)
    radial_weight = config.get("radial_weight", 2.0)
    print(f"\nLoss weights: radial={radial_weight}, margin={config.get('margin_weight', 1.0)}")
    if use_rank_loss:
        print(f"  Global rank loss: weight={rank_loss_weight}")
    if use_zero_structure:
        print(
            f"  Zero-structure loss: valuation={config.get('zero_valuation_weight', 1.0)}, sparsity={config.get('zero_sparsity_weight', 0.5)}"
        )

    # Create optimizer (only trainable parameters)
    base_lr = config.get("lr", 1e-3)
    use_riemannian = config.get("riemannian", False)

    if use_option_c and hasattr(model, "get_param_groups"):
        # Option C: use param groups with different LRs
        param_groups = model.get_param_groups(base_lr)
        if use_riemannian:
            optimizer = get_riemannian_optimizer(
                param_groups,
                lr=base_lr,
                optimizer_type="adam",
                weight_decay=config.get("weight_decay", 1e-4),
            )
            print("\nUsing RiemannianAdam (geoopt) with param groups")
        else:
            optimizer = torch.optim.AdamW(param_groups, weight_decay=config.get("weight_decay", 1e-4))
        print(f"Param groups: {len(param_groups)} groups")
        for i, pg in enumerate(param_groups):
            name = pg.get("name", f"group_{i}")
            print(f"  {name}: {len(pg['params'])} params, lr={pg['lr']:.2e}")
    else:
        # Option A: single learning rate
        if use_riemannian:
            optimizer = get_riemannian_optimizer(
                model.get_trainable_parameters(),
                lr=base_lr,
                optimizer_type="adam",
                weight_decay=config.get("weight_decay", 1e-4),
            )
            print("\nUsing RiemannianAdam (geoopt)")
        else:
            optimizer = torch.optim.AdamW(
                model.get_trainable_parameters(),
                lr=base_lr,
                weight_decay=config.get("weight_decay", 1e-4),
            )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    # Training loop
    print("\n=== Starting Training ===")
    n_epochs = config.get("epochs", 100)
    batch_size = config.get("batch_size", 512)
    use_stratified = not config.get("no_stratified", False)
    print(f"Stratified sampling: {use_stratified}")

    # Adaptive curriculum (v1.1)
    use_adaptive = not config.get("no_adaptive", False)
    curriculum = AdaptiveCurriculum(
        hierarchy_threshold=config.get("hierarchy_threshold", -0.62),
        patience=config.get("patience", 20),
        min_epochs=config.get("min_epochs", 30),
        n_epochs=n_epochs,
        enabled=use_adaptive,
    )
    if use_adaptive:
        print(f"Adaptive curriculum: threshold={curriculum.hierarchy_threshold}, patience={curriculum.patience}")
    else:
        print("Adaptive curriculum: DISABLED (fixed tau schedule)")

    # Learnable weights (v1.4)
    use_learnable_weights = config.get("learnable_weights", False)
    if use_learnable_weights:
        if not config.get("use_controller", False):
            print("WARNING: --learnable_weights requires --use_controller, enabling controller")
        print("Learnable weights: ENABLED (controller learns tau and radial_weight)")

    # Progressive unfreezing (v5.11.6)
    progressive_unfreeze = config.get("progressive_unfreeze", False)
    unfreeze_start = config.get("unfreeze_start_epoch", 5)
    unfreeze_warmup = config.get("unfreeze_warmup_epochs", 5)
    encoder_a_target_lr = config.get("encoder_a_lr_scale", 0.05)
    if progressive_unfreeze:
        print(
            f"Progressive unfreezing: start={unfreeze_start}, warmup={unfreeze_warmup}, target_lr_scale={encoder_a_target_lr}"
        )

    # Homeostatic control (v5.11.7 + v5.11.8 Q-gated annealing)
    use_homeostasis = config.get("homeostasis", False)
    homeostasis = None
    if use_homeostasis:
        enable_annealing = not config.get("no_annealing", False)
        homeostasis = HomeostasisController(
            coverage_freeze_threshold=config.get("coverage_freeze_threshold", 0.995),
            warmup_epochs=config.get("homeostasis_warmup", 5),
            hysteresis_epochs=config.get("hysteresis_epochs", 3),
            enable_annealing=enable_annealing,
            annealing_step=config.get("annealing_step", 0.005),
            coverage_floor=config.get("coverage_floor", 0.95),
        )
        print(
            f"Homeostasis: ENABLED (coverage_freeze={homeostasis.coverage_freeze_threshold}, warmup={homeostasis.warmup_epochs})"
        )
        if enable_annealing:
            print(f"  Q-gated annealing: step={homeostasis.annealing_step}, floor={homeostasis.coverage_floor}")
        # Track previous freeze state for optimizer rebuild
        prev_freeze_state = {
            "encoder_a": True,
            "encoder_b": False,
            "controller": False,
        }

    best_radial_corr = float("inf")  # Want negative, so lower is better
    best_composite_score = float("-inf")

    for epoch in range(n_epochs):
        # Curriculum: tau from adaptive curriculum (may freeze)
        tau = curriculum.compute_tau(epoch)

        # Progressive unfreezing of encoder_A (v5.11.6)
        if progressive_unfreeze and hasattr(model, "set_encoder_a_unfreeze"):
            if epoch < unfreeze_start:
                # Keep frozen
                current_lr_scale = 0.0
            elif epoch < unfreeze_start + unfreeze_warmup:
                # Linear warmup
                progress = (epoch - unfreeze_start) / unfreeze_warmup
                current_lr_scale = progress * encoder_a_target_lr
            else:
                # Fully unfrozen
                current_lr_scale = encoder_a_target_lr

            # Check if this is the first unfreeze (need to add params to optimizer)
            encoder_a_in_optimizer = any(pg.get("name") == "encoder_A" for pg in optimizer.param_groups)

            if current_lr_scale > 0 and not encoder_a_in_optimizer:
                # First time unfreezing - add encoder_A to optimizer
                model.set_encoder_a_unfreeze(current_lr_scale)
                optimizer.add_param_group(
                    {
                        "params": list(model.encoder_A.parameters()),
                        "lr": base_lr * current_lr_scale,
                        "name": "encoder_A",
                    }
                )
                print(f"  [UNFREEZE] encoder_A added to optimizer at epoch {epoch}, lr_scale={current_lr_scale:.4f}")
            elif encoder_a_in_optimizer:
                # Update existing param group LR
                model.set_encoder_a_unfreeze(current_lr_scale)
                for param_group in optimizer.param_groups:
                    if param_group.get("name") == "encoder_A":
                        param_group["lr"] = base_lr * current_lr_scale

        # Train
        train_metrics = train_epoch(
            model,
            optimizer,
            x,
            indices,
            geodesic_loss_fn,
            radial_loss_fn,
            batch_size,
            epoch,
            tau,
            radial_weight,
            device,
            use_stratified=use_stratified,
            rank_loss_fn=rank_loss_fn,
            rank_loss_weight=rank_loss_weight,
            use_learnable_weights=use_learnable_weights,
            zero_structure_loss_fn=zero_structure_loss_fn,
            zero_structure_weight=zero_structure_weight,
        )

        # Evaluate
        eval_metrics = compute_metrics(model, x, indices, geodesic_loss_fn, radial_loss_fn, device)

        # Update scheduler
        scheduler.step()

        # Update adaptive curriculum
        curriculum_status = curriculum.update(epoch, eval_metrics["radial_corr_A"], train_metrics["loss"])

        # Homeostatic control (v5.11.7 + v5.11.8 Q-gated annealing)
        if use_homeostasis and homeostasis is not None:
            # Compute controller gradient norm
            controller_grad_norm = None
            if model.controller is not None:
                total_norm = 0.0
                for p in model.controller.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                controller_grad_norm = total_norm**0.5

            # Update homeostasis state (now with Q tracking)
            homeo_state = homeostasis.update(
                epoch=epoch,
                coverage=eval_metrics["coverage"],
                hierarchy_A=eval_metrics["radial_corr_A"],
                hierarchy_B=eval_metrics["radial_corr_B"],
                dist_corr_A=eval_metrics["distance_corr_A"],
                controller_grad_norm=controller_grad_norm,
            )

            # Check if freeze states changed
            state_changed = (
                homeo_state["encoder_a_frozen"] != prev_freeze_state["encoder_a"]
                or homeo_state["encoder_b_frozen"] != prev_freeze_state["encoder_b"]
                or homeo_state["controller_frozen"] != prev_freeze_state["controller"]
            )

            if state_changed:
                # Apply new freeze states to model
                model.apply_homeostasis_state(homeo_state)

                # Rebuild optimizer with new param groups
                optimizer = model.rebuild_optimizer(optimizer, base_lr)

                # Re-create scheduler for new optimizer
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

                # Update previous state
                prev_freeze_state = {
                    "encoder_a": homeo_state["encoder_a_frozen"],
                    "encoder_b": homeo_state["encoder_b_frozen"],
                    "controller": homeo_state["controller_frozen"],
                }

                # Log events
                for event in homeo_state.get("events", []):
                    print(f"  [HOMEOSTASIS] {event}")

        # Log to TensorBoard
        writer.add_scalar("Train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("Train/geo_loss", train_metrics["geo_loss"], epoch)
        writer.add_scalar("Train/rad_loss", train_metrics["rad_loss"], epoch)
        writer.add_scalar("Train/rank_loss", train_metrics["rank_loss"], epoch)
        writer.add_scalar("Train/tau", tau, epoch)
        writer.add_scalar("Train/tau_frozen", 1.0 if curriculum.tau_frozen else 0.0, epoch)
        writer.add_scalar("Eval/coverage", eval_metrics["coverage"], epoch)
        writer.add_scalar("Eval/radial_corr_A", eval_metrics["radial_corr_A"], epoch)
        writer.add_scalar("Eval/radial_corr_B", eval_metrics["radial_corr_B"], epoch)
        writer.add_scalar("Eval/distance_corr_A", eval_metrics["distance_corr_A"], epoch)
        writer.add_scalar("Eval/mean_radius_A", eval_metrics["mean_radius_A"], epoch)
        writer.add_scalar("Eval/mean_radius_B", eval_metrics["mean_radius_B"], epoch)
        writer.add_scalar("Eval/radius_range_A", eval_metrics["radius_range_A"], epoch)
        writer.add_scalar("Eval/radius_v0", eval_metrics["radius_v0"], epoch)
        writer.add_scalar("Eval/radius_v9", eval_metrics["radius_v9"], epoch)
        writer.add_scalar(
            "Eval/composite_score",
            curriculum.compute_composite_score(eval_metrics["radial_corr_A"], train_metrics["loss"]),
            epoch,
        )
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # Log homeostasis states and Q
        if use_homeostasis and homeostasis is not None:
            writer.add_scalar(
                "Homeostasis/encoder_a_frozen",
                1.0 if homeostasis.encoder_a_frozen else 0.0,
                epoch,
            )
            writer.add_scalar(
                "Homeostasis/encoder_b_frozen",
                1.0 if homeostasis.encoder_b_frozen else 0.0,
                epoch,
            )
            writer.add_scalar(
                "Homeostasis/controller_frozen",
                1.0 if homeostasis.controller_frozen else 0.0,
                epoch,
            )
            writer.add_scalar("Homeostasis/current_Q", homeo_state.get("current_Q", 0), epoch)
            writer.add_scalar("Homeostasis/best_Q", homeo_state.get("best_Q", 0), epoch)
            writer.add_scalar(
                "Homeostasis/coverage_freeze_threshold",
                homeostasis.coverage_freeze_threshold,
                epoch,
            )
            writer.add_scalar(
                "Homeostasis/total_cycles",
                sum(homeostasis.cycle_count.values()),
                epoch,
            )

        # Log learnable curvature if enabled
        if learnable_curvature and hasattr(model.projection, "get_curvature"):
            current_c = model.projection.get_curvature()
            writer.add_scalar("Geometry/curvature", current_c, epoch)

        # Print progress
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"\nEpoch {epoch}/{n_epochs}")
            rank_str = f", rank: {train_metrics['rank_loss']:.4f}" if rank_loss_fn else ""
            print(
                f"  Loss: {train_metrics['loss']:.4f} (geo: {train_metrics['geo_loss']:.4f}, rad: {train_metrics['rad_loss']:.4f}{rank_str})"
            )
            print(f"  Coverage: {eval_metrics['coverage'] * 100:.1f}%")
            print(f"  Radial Hierarchy: A={eval_metrics['radial_corr_A']:.3f}, B={eval_metrics['radial_corr_B']:.3f}")
            print(
                f"  Radius Range: [{eval_metrics['radius_min_A']:.3f}, {eval_metrics['radius_max_A']:.3f}] (range={eval_metrics['radius_range_A']:.3f})"
            )
            print(
                f"  Radius v=0: {eval_metrics['radius_v0']:.3f}, v=9: {eval_metrics['radius_v9']:.3f} (target: 0.85, 0.10)"
            )
            print(f"  Distance Corr: A={eval_metrics['distance_corr_A']:.3f}")
            tau_status = f"tau: {tau:.3f}"
            if curriculum.tau_frozen:
                tau_status += " [FROZEN]"
            homeo_status = ""
            if use_homeostasis and homeostasis is not None:
                homeo_status = f" | {homeostasis.get_state_summary()}"
            print(f"  {tau_status}, LR: {optimizer.param_groups[0]['lr']:.2e}{homeo_status}")

        # Handle curriculum events
        if curriculum_status.get("triggered_freeze"):
            print(
                f"  [TAU FROZEN] Hierarchy threshold {curriculum.hierarchy_threshold} reached at tau={curriculum.frozen_tau:.3f}"
            )

        # Save best model (best = highest composite score: hierarchy + low loss)
        composite_score = curriculum.compute_composite_score(eval_metrics["radial_corr_A"], train_metrics["loss"])
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_radial_corr = eval_metrics["radial_corr_A"]
            checkpoint = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": eval_metrics,
                "train_metrics": train_metrics,
                "composite_score": composite_score,
                "tau": tau,
                "tau_frozen": curriculum.tau_frozen,
                "config": config,
            }
            torch.save(checkpoint, save_dir / "best.pt")
            print(
                f"  [NEW BEST] Composite: {composite_score:.4f} (hierarchy: {eval_metrics['radial_corr_A']:.4f}, loss: {train_metrics['loss']:.4f})"
            )

        # Periodic checkpoint
        if epoch % 20 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "metrics": eval_metrics,
                    "config": config,
                },
                save_dir / f"epoch_{epoch}.pt",
            )

        # Early stopping check
        if curriculum_status.get("triggered_stop"):
            print(f"\n[EARLY STOPPING] No improvement for {curriculum.patience} epochs")
            print(f"  Best epoch: {curriculum.best_epoch}, Best score: {curriculum.best_score:.4f}")
            break

    # Final checkpoint
    final_epoch = epoch  # May be less than n_epochs if early stopped
    torch.save(
        {
            "epoch": final_epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": eval_metrics,
            "config": config,
        },
        save_dir / "latest.pt",
    )

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nFinal Metrics (epoch {final_epoch}):")
    print(f"  Coverage: {eval_metrics['coverage'] * 100:.1f}%")
    print(f"  Radial Hierarchy: A={eval_metrics['radial_corr_A']:.3f}, B={eval_metrics['radial_corr_B']:.3f}")
    print(f"  Radius Range: [{eval_metrics['radius_min_A']:.3f}, {eval_metrics['radius_max_A']:.3f}]")
    print(f"  Radius v=0: {eval_metrics['radius_v0']:.3f} (target: 0.85)")
    print(f"  Radius v=9: {eval_metrics['radius_v9']:.3f} (target: 0.10)")
    print(f"  Distance Correlation: A={eval_metrics['distance_corr_A']:.3f}")

    print(f"\nBest Model (epoch {curriculum.best_epoch}):")
    print(f"  Composite Score: {best_composite_score:.4f}")
    print(f"  Radial Hierarchy: {best_radial_corr:.4f}")

    if curriculum.tau_frozen:
        print("\nAdaptive Curriculum:")
        print(f"  Tau frozen at epoch {curriculum.frozen_epoch} (tau={curriculum.frozen_tau:.3f})")
        print(f"  Hierarchy threshold: {curriculum.hierarchy_threshold}")

    if curriculum.should_stop:
        print("\nEarly Stopping:")
        print(f"  Triggered after {curriculum.patience} epochs without improvement")
        print(f"  Saved {n_epochs - final_epoch - 1} epochs of compute")

    print(f"\nCheckpoints saved to: {save_dir}")
    print(f"TensorBoard logs: {log_dir}")

    writer.close()


if __name__ == "__main__":
    main()
