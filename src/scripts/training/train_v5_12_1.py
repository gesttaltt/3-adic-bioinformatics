# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""V5.12.1 Training Script - Full Hyperbolic Integration.

This script implements the V5.12.1 training pipeline with critical hyperbolic fixes:
- Decoder input uses log_map_zero(z_hyp) instead of mu (Euclidean)
- Metrics computation uses hyperbolic distance instead of Euclidean norm
- Gradient flow through hyperbolic projection for richness preservation

Key Changes from V5.12:
1. Decoder receives tangent space representation (log_map_zero(z_A_hyp))
2. Radii computed as hyperbolic distance from origin, not Euclidean norm
3. Coverage check uses the new hyperbolic representation

Target Metrics:
- Coverage: 100% (may need adaptation period)
- Hierarchy_B: -0.80 to -0.8321
- Richness: >0.005 (improved from V5.12's 0.001)
- r_v9: 0.10-0.15
- dist_corr: >0.7

Device: RTX 2060 SUPER (8GB VRAM) compatible

Usage:
    python src/scripts/training/train_v5_12_1.py
    python src/scripts/training/train_v5_12_1.py --config src/configs/v5_12_1.yaml
    python src/scripts/training/train_v5_12_1.py --epochs 200 --resume
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from torch.utils.tensorboard import SummaryWriter

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import RUNS_DIR
from src.core import TERNARY
from src.core.metrics import compute_comprehensive_metrics
from src.data.generation import generate_all_ternary_operations
from src.geometry import get_riemannian_optimizer
from src.geometry.poincare import poincare_distance
from src.losses import (
    CombinedZeroStructureLoss,
    GlobalRankLoss,
    PAdicGeodesicLoss,
    RadialHierarchyLoss,
    RichHierarchyLoss,
)
from src.models import HomeostasisController, TernaryVAEV5_11_PartialFreeze
from src.models.homeostasis import compute_Q
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_config(config: dict) -> None:
    """Validate V5.12.1 configuration structure.

    Raises:
        ValueError: If required keys are missing or have invalid values
    """
    # Required top-level sections
    required_sections = ["model", "training", "loss", "homeostasis", "checkpoints"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    # Required model parameters
    model_cfg = config["model"]
    required_model_keys = ["latent_dim", "hidden_dim", "max_radius", "curvature"]
    for key in required_model_keys:
        if key not in model_cfg:
            raise ValueError(f"Missing required model parameter: 'model.{key}'")

    # Validate numeric ranges
    if not (0 < model_cfg.get("max_radius", 0.95) < 1.0):
        raise ValueError(f"model.max_radius must be in (0, 1), got {model_cfg.get('max_radius')}")
    if model_cfg.get("curvature", 1.0) <= 0:
        raise ValueError(f"model.curvature must be > 0, got {model_cfg.get('curvature')}")
    if model_cfg.get("latent_dim", 16) < 2:
        raise ValueError(f"model.latent_dim must be >= 2, got {model_cfg.get('latent_dim')}")

    # Required training parameters
    train_cfg = config["training"]
    if train_cfg.get("epochs", 200) < 1:
        raise ValueError(f"training.epochs must be >= 1, got {train_cfg.get('epochs')}")
    if train_cfg.get("batch_size", 512) < 1:
        raise ValueError(f"training.batch_size must be >= 1, got {train_cfg.get('batch_size')}")

    # Required loss sections
    loss_cfg = config["loss"]
    if "rich_hierarchy" not in loss_cfg:
        raise ValueError("Missing required loss section: 'loss.rich_hierarchy'")
    if "radial" not in loss_cfg:
        raise ValueError("Missing required loss section: 'loss.radial'")

    # Validate checkpoint directory
    if "save_dir" not in config.get("checkpoints", {}):
        raise ValueError("Missing required config: 'checkpoints.save_dir'")

    print("Config validation: PASSED")


def check_cuda():
    """Verify CUDA is available and print device info."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Please install PyTorch with CUDA support:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu126")
        sys.exit(1)

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    print(f"\n{'=' * 60}")
    print("V5.12.1 DEVICE CONFIGURATION (Full Hyperbolic Integration)")
    print(f"{'=' * 60}")
    print(f"  Device: {props.name}")
    print(f"  VRAM: {props.total_memory / 1024**3:.1f} GB")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"{'=' * 60}\n")
    return device


def create_stratified_indices(indices: torch.Tensor, batch_size: int, device: str, high_v_ratio: float = 0.25):
    """Create stratified batch indices with enhanced high-valuation sampling.

    V5.12.1: Same stratification as V5.12, 25% high-v budget.

    Args:
        indices: All operation indices
        batch_size: Target batch size
        device: Torch device
        high_v_ratio: Fraction of batch reserved for high-v samples (v>=4)

    Returns:
        List of batch index tensors
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

    # 25% for high-v (v>=4)
    high_v_budget = int(batch_size * high_v_ratio)
    low_v_budget = batch_size - high_v_budget

    high_v_levels = [v for v in valuation_groups if v >= 4]
    low_v_levels = [v for v in valuation_groups if v < 4]

    batches = []
    n_batches = (n_samples + batch_size - 1) // batch_size

    for _ in range(n_batches):
        batch_indices = []

        # Sample from high-valuation levels
        if high_v_levels:
            per_high_v = max(1, high_v_budget // len(high_v_levels))
            for v in high_v_levels:
                group = valuation_groups[v]
                sample_idx = torch.randint(0, len(group), (per_high_v,), device=device)
                batch_indices.append(group[sample_idx])

        # Sample from low-valuation levels (proportional)
        if low_v_levels:
            total_low = sum(len(valuation_groups[v]) for v in low_v_levels)
            for v in low_v_levels:
                group = valuation_groups[v]
                n_to_sample = max(1, int(low_v_budget * len(group) / total_low))
                sample_idx = torch.randint(0, len(group), (n_to_sample,), device=device)
                batch_indices.append(group[sample_idx])

        # Combine and trim to exact batch size
        batch = torch.cat(batch_indices)
        if len(batch) > batch_size:
            batch = batch[torch.randperm(len(batch), device=device)[:batch_size]]
        elif len(batch) < batch_size:
            extra = torch.randint(0, n_samples, (batch_size - len(batch),), device=device)
            batch = torch.cat([batch, extra])

        batches.append(batch)

    return batches


def compute_quick_metrics(model, all_ops, indices, device, curvature=1.0):
    """Compute quick metrics for training monitoring.

    V5.12.1: Uses hyperbolic distance instead of Euclidean norm for radii.
    """
    model.eval()
    batch_size = 4096

    all_radii_A = []
    all_radii_B = []
    all_correct = []

    with torch.no_grad():
        for i in range(0, len(all_ops), batch_size):
            batch_ops = all_ops[i : i + batch_size].to(device)

            out = model(batch_ops, compute_control=False)
            z_A = out["z_A_hyp"]
            z_B = out["z_B_hyp"]

            # V5.12.1: Use hyperbolic distance from origin instead of Euclidean norm
            origin_A = torch.zeros_like(z_A)
            origin_B = torch.zeros_like(z_B)
            radii_A = poincare_distance(z_A, origin_A, c=curvature)
            radii_B = poincare_distance(z_B, origin_B, c=curvature)

            all_radii_A.append(radii_A.cpu().numpy())
            all_radii_B.append(radii_B.cpu().numpy())

            # V5.12.1: Coverage check uses logits_A which now comes from log_map(z_A_hyp)
            logits = out["logits_A"]
            preds = torch.argmax(logits, dim=-1) - 1
            correct = (preds == batch_ops.long()).float().mean(dim=1).cpu().numpy()
            all_correct.append(correct)

    all_radii_A = np.concatenate(all_radii_A)
    all_radii_B = np.concatenate(all_radii_B)
    all_correct = np.concatenate(all_correct)
    valuations = TERNARY.valuation(indices).cpu().numpy()

    coverage = (all_correct == 1.0).mean()
    hierarchy_A = spearmanr(valuations, all_radii_A)[0]
    hierarchy_B = spearmanr(valuations, all_radii_B)[0]

    # Richness (within-level variance) - now in hyperbolic distance space
    richness = 0
    for v in range(10):
        mask = valuations == v
        if mask.sum() > 1:
            richness += all_radii_B[mask].var()
    richness /= 10

    # Radius by valuation (hyperbolic distance)
    r_v0 = all_radii_B[valuations == 0].mean()
    r_v9 = all_radii_B[valuations == 9].mean() if (valuations == 9).any() else np.nan

    # Distance correlation (sampled)
    sample_idx = np.random.choice(len(all_radii_B), min(1000, len(all_radii_B)), replace=False)
    z_sample = all_radii_B[sample_idx]
    val_sample = valuations[sample_idx]
    z_dists = np.abs(z_sample[:, None] - z_sample[None, :])
    val_dists = np.abs(val_sample[:, None] - val_sample[None, :]).astype(float)
    triu_idx = np.triu_indices(len(sample_idx), k=1)
    dist_corr = spearmanr(z_dists[triu_idx], val_dists[triu_idx])[0]

    model.train()

    return {
        "coverage": coverage,
        "hierarchy_A": hierarchy_A,
        "hierarchy_B": hierarchy_B,
        "richness": richness,
        "dist_corr": dist_corr,
        "r_v0": r_v0,
        "r_v9": r_v9,
        "Q": compute_Q(dist_corr, hierarchy_B),
    }


def train_epoch_v5121(
    model,
    optimizer,
    x,
    indices,
    original_radii,
    rich_hierarchy_loss,
    radial_loss_fn,
    rank_loss_fn,
    geodesic_loss_fn,
    zero_structure_loss_fn,
    config,
    epoch,
    device,
):
    """Train one epoch with V5.12.1 full hyperbolic integration.

    Same two-phase strategy as V5.12, but now gradients flow through
    hyperbolic projection to decoder via log_map_zero.

    Phase 1 (epochs 0-50): Structure establishment
        - RichHierarchyLoss (primary)
        - RadialHierarchyLoss (auxiliary)
        - GlobalRankLoss (structural constraint)
        - ZeroStructureLoss

    Phase 2 (epochs 50+): Geometry refinement
        - RichHierarchyLoss (primary)
        - PAdicGeodesicLoss (geodesic refinement)
        - Reduced radial weight
    """
    model.train()

    batch_size = config["training"]["batch_size"]
    high_v_ratio = config["training"].get("high_v_budget_ratio", 0.25)
    batches = create_stratified_indices(indices, batch_size, device, high_v_ratio)
    n_batches = len(batches)

    # Phase determination
    phase_2_start = config["loss"]["geodesic"].get("phase_start_epoch", 50)
    is_phase_2 = epoch >= phase_2_start

    # Loss weights from config
    radial_weight = config["loss"]["radial"].get("radial_weight", 1.0)
    rank_weight = config["loss"]["rank"].get("weight", 0.5)
    zero_weight = config["loss"]["zero_structure"].get("valuation_weight", 0.5)
    geodesic_weight = config["loss"]["geodesic"].get("weight", 0.3) if is_phase_2 else 0.0

    # Reduce radial weight in phase 2
    if is_phase_2:
        radial_weight *= 0.5

    # Accumulators
    total_loss = 0.0
    total_rich = 0.0
    total_radial = 0.0
    total_rank = 0.0
    total_zero = 0.0
    total_geodesic = 0.0

    for batch_idx in batches:
        x_batch = x[batch_idx]
        idx_batch = indices[batch_idx]
        orig_radii_batch = original_radii[batch_idx]

        optimizer.zero_grad()

        # Forward pass - V5.12.1: logits_A now comes from log_map(z_A_hyp)
        out = model(x_batch, compute_control=False)
        z_A = out["z_A_hyp"]
        z_B = out["z_B_hyp"]
        logits = out["logits_A"]  # Now computed from hyperbolic representation

        # === PRIMARY: RichHierarchyLoss (preserves richness) ===
        rich_losses = rich_hierarchy_loss(z_B, idx_batch, logits, x_batch, orig_radii_batch)
        rich_loss = rich_losses["total"]

        # === AUXILIARY: RadialHierarchyLoss ===
        rad_loss_A, _ = radial_loss_fn(z_A, idx_batch)
        rad_loss_B, _ = radial_loss_fn(z_B, idx_batch)
        rad_loss = rad_loss_A + rad_loss_B

        # === STRUCTURAL: GlobalRankLoss ===
        rank_loss = torch.tensor(0.0, device=device)
        if rank_loss_fn is not None:
            rank_loss_A, _ = rank_loss_fn(z_A, idx_batch)
            rank_loss_B, _ = rank_loss_fn(z_B, idx_batch)
            rank_loss = rank_loss_A + rank_loss_B

        # === Zero-structure loss ===
        zero_loss = torch.tensor(0.0, device=device)
        if zero_structure_loss_fn is not None:
            zero_loss_A = zero_structure_loss_fn(z_A, x_batch)
            zero_loss_B = zero_structure_loss_fn(z_B, x_batch)
            zero_loss = zero_loss_A + zero_loss_B

        # === PHASE 2: Geodesic refinement ===
        geo_loss = torch.tensor(0.0, device=device)
        if is_phase_2 and geodesic_loss_fn is not None:
            geo_loss_A, _ = geodesic_loss_fn(z_A, idx_batch)
            geo_loss_B, _ = geodesic_loss_fn(z_B, idx_batch)
            geo_loss = geo_loss_A + geo_loss_B

        # === Total Loss ===
        loss = (
            rich_loss
            + radial_weight * rad_loss
            + rank_weight * rank_loss
            + zero_weight * zero_loss
            + geodesic_weight * geo_loss
        )

        # Backward - V5.12.1: Gradients now flow through hyperbolic projection
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"].get("max_grad_norm", 1.0))
        optimizer.step()

        # Accumulate
        total_loss += loss.item()
        total_rich += rich_loss.item()
        total_radial += rad_loss.item()
        total_rank += rank_loss.item()
        total_zero += zero_loss.item()
        total_geodesic += geo_loss.item()

    return {
        "loss": total_loss / n_batches,
        "rich_loss": total_rich / n_batches,
        "radial_loss": total_radial / n_batches,
        "rank_loss": total_rank / n_batches,
        "zero_loss": total_zero / n_batches,
        "geodesic_loss": total_geodesic / n_batches,
        "phase": 2 if is_phase_2 else 1,
    }


def main():
    parser = argparse.ArgumentParser(description="Train V5.12.1 Model (Full Hyperbolic Integration)")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/v5_12_1.yaml",
        help="Path to V5.12.1 config YAML",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    args = parser.parse_args()

    # Check CUDA
    device = check_cuda()

    # Load config
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    print(f"Loaded V5.12.1 config from: {config_path}")

    # Validate config structure
    validate_config(config)

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

    # Setup TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = RUNS_DIR / f"v5_12_1_hyperbolic_{timestamp}"
    writer = SummaryWriter(log_dir=str(log_dir))

    # === Create Model ===
    print("\n=== Creating V5.12.1 Model (Full Hyperbolic Integration) ===")
    model_cfg = config["model"]
    curvature = model_cfg.get("curvature", 1.0)

    model = TernaryVAEV5_11_PartialFreeze(
        latent_dim=model_cfg.get("latent_dim", 16),
        hidden_dim=model_cfg.get("hidden_dim", 64),
        max_radius=model_cfg.get("max_radius", 0.95),
        curvature=curvature,
        use_controller=model_cfg.get("use_controller", True),
        use_dual_projection=model_cfg.get("use_dual_projection", True),
        n_projection_layers=model_cfg.get("projection_layers", 2),
        projection_dropout=model_cfg.get("projection_dropout", 0.1),
        learnable_curvature=model_cfg.get("learnable_curvature", True),
        manifold_aware=model_cfg.get("manifold_aware", True),
        freeze_encoder_b=False,
        encoder_b_lr_scale=config["option_c"].get("encoder_b_lr_scale", 0.1),
        encoder_a_lr_scale=config["option_c"].get("encoder_a_lr_scale", 0.05),
    )

    # Load frozen checkpoint
    frozen_cfg = config.get("frozen_checkpoint", {})
    frozen_path = PROJECT_ROOT / frozen_cfg.get("path", "checkpoints/v5_5/latest.pt")
    if frozen_path.exists():
        print(f"Loading frozen checkpoint: {frozen_path}")
        ckpt = load_checkpoint_compat(frozen_path, map_location=device)
        model_state = get_model_state_dict(ckpt)
        model.load_state_dict(model_state, strict=False)
        print(f"  Loaded checkpoint (keys: {list(ckpt.keys())[:5]}...)")
        print("  NOTE: Decoder was trained on mu (Euclidean), will adapt to log_map(z_hyp)")
    else:
        print(f"WARNING: Frozen checkpoint not found at {frozen_path}")
        print("  Training will start with random initialization.")

    model = model.to(device)

    # Set initial freeze state
    model.set_encoder_a_frozen(True)
    model.set_encoder_b_frozen(False)
    # V5.12.1 CRITICAL: Unfreeze decoder so it can adapt from mu to log_map(z_hyp)
    model.set_decoder_frozen(False)
    print(f"Freeze state: {model.get_freeze_state_summary()}")
    print("  decoder_A: UNFROZEN (V5.12.1 - adapting to hyperbolic input)")

    # Count parameters
    param_counts = model.count_parameters()
    print(f"\nParameters: {param_counts['total']:,} total, {param_counts['trainable']:,} trainable")

    # === Dataset ===
    print("\n=== Loading Dataset ===")
    all_ops_np = generate_all_ternary_operations()
    all_ops = torch.tensor(all_ops_np, dtype=torch.float32, device=device)
    indices = torch.arange(len(all_ops), device=device)
    print(f"Dataset size: {len(all_ops)}")

    # Get original radii for richness preservation (using hyperbolic distance)
    with torch.no_grad():
        model.eval()
        original_radii = []
        for i in range(0, len(all_ops), 4096):
            batch = all_ops[i : i + 4096]
            out = model(batch, compute_control=False)
            z_B = out["z_B_hyp"]
            origin = torch.zeros_like(z_B)
            radii = poincare_distance(z_B, origin, c=curvature)
            original_radii.append(radii)
        original_radii = torch.cat(original_radii)
        model.train()
    print(f"Original radii (hyperbolic): {original_radii.min():.4f} - {original_radii.max():.4f}")

    # Initial metrics
    init_metrics = compute_quick_metrics(model, all_ops, indices, device, curvature)
    print("\nInitial metrics (V5.12.1 with hyperbolic decoder input):")
    print(f"  Coverage: {init_metrics['coverage'] * 100:.1f}%")
    print(f"  Hierarchy_B: {init_metrics['hierarchy_B']:.4f}")
    print(f"  Richness: {init_metrics['richness']:.6f}")
    print(f"  Q: {init_metrics['Q']:.3f}")
    initial_richness = init_metrics["richness"]

    # === Loss Functions ===
    print("\n=== Creating V5.12.1 Loss Functions ===")
    loss_cfg = config["loss"]

    # V5.12.2: Get curvature for hyperbolic losses
    curvature = model_cfg.get("curvature", 1.0)

    # PRIMARY: RichHierarchyLoss (V5.12.2: uses hyperbolic distance)
    rich_cfg = loss_cfg["rich_hierarchy"]
    rich_hierarchy_loss = RichHierarchyLoss(
        inner_radius=loss_cfg["radial"].get("inner_radius", 0.08),
        outer_radius=loss_cfg["radial"].get("outer_radius", 0.90),
        hierarchy_weight=rich_cfg.get("hierarchy_weight", 5.0),
        coverage_weight=rich_cfg.get("coverage_weight", 1.0),
        richness_weight=rich_cfg.get("richness_weight", 2.0),
        separation_weight=rich_cfg.get("separation_weight", 3.0),
        min_richness_ratio=rich_cfg.get("min_richness_ratio", 0.5),
        curvature=curvature,
    ).to(device)
    print(
        f"  RichHierarchyLoss: hierarchy={rich_cfg.get('hierarchy_weight', 5.0)}, richness={rich_cfg.get('richness_weight', 2.0)}, curvature={curvature}"
    )

    # AUXILIARY: RadialHierarchyLoss (V5.12.2: uses hyperbolic distance)
    radial_cfg = loss_cfg["radial"]
    radial_loss_fn = RadialHierarchyLoss(
        inner_radius=radial_cfg.get("inner_radius", 0.08),
        outer_radius=radial_cfg.get("outer_radius", 0.90),
        margin_weight=radial_cfg.get("margin_weight", 0.5),
        use_margin_loss=True,
        curvature=curvature,
    ).to(device)
    print(
        f"  RadialHierarchyLoss: inner={radial_cfg.get('inner_radius', 0.08)}, outer={radial_cfg.get('outer_radius', 0.90)}, curvature={curvature}"
    )

    # STRUCTURAL: GlobalRankLoss (V5.12.2: uses hyperbolic distance)
    rank_cfg = loss_cfg["rank"]
    rank_loss_fn = (
        GlobalRankLoss(
            temperature=rank_cfg.get("temperature", 0.1),
            n_pairs=rank_cfg.get("n_pairs", 2000),
            curvature=curvature,
        ).to(device)
        if rank_cfg.get("enabled", True)
        else None
    )
    if rank_loss_fn:
        print(f"  GlobalRankLoss: weight={rank_cfg.get('weight', 0.5)}, curvature={curvature}")

    # PHASE 2: PAdicGeodesicLoss
    geo_cfg = loss_cfg["geodesic"]
    geodesic_loss_fn = (
        PAdicGeodesicLoss(
            curvature=geo_cfg.get("curvature", 1.0),
            max_target_distance=geo_cfg.get("max_target_distance", 3.0),
            n_pairs=geo_cfg.get("n_pairs", 2000),
        ).to(device)
        if geo_cfg.get("enabled", True)
        else None
    )
    if geodesic_loss_fn:
        print(f"  PAdicGeodesicLoss: activates at epoch {geo_cfg.get('phase_start_epoch', 50)}")

    # Zero-structure loss (V5.12.2: uses hyperbolic distance)
    zero_cfg = loss_cfg["zero_structure"]
    zero_structure_loss_fn = (
        CombinedZeroStructureLoss(
            valuation_weight=zero_cfg.get("valuation_weight", 0.5),
            sparsity_weight=zero_cfg.get("sparsity_weight", 0.3),
            inner_radius=radial_cfg.get("inner_radius", 0.08),
            outer_radius=radial_cfg.get("outer_radius", 0.90),
            curvature=curvature,
        ).to(device)
        if zero_cfg.get("enabled", True)
        else None
    )
    if zero_structure_loss_fn:
        print(f"  ZeroStructureLoss: enabled, curvature={curvature}")

    # === Homeostasis Controller ===
    homeo_cfg = config["homeostasis"]
    homeostasis = HomeostasisController(
        coverage_freeze_threshold=homeo_cfg.get("coverage_freeze_threshold", 0.995),
        coverage_unfreeze_threshold=homeo_cfg.get("coverage_unfreeze_threshold", 1.0),
        warmup_epochs=homeo_cfg.get("warmup_epochs", 5),
        hysteresis_epochs=homeo_cfg.get("hysteresis_epochs", 3),
        enable_annealing=homeo_cfg.get("enable_annealing", True),
        annealing_step=homeo_cfg.get("annealing_step", 0.003),
        coverage_floor=homeo_cfg.get("coverage_floor", 0.95),
    )
    print(f"\nHomeostasis: enabled, coverage_freeze={homeo_cfg.get('coverage_freeze_threshold', 0.995)}")

    # === Optimizer ===
    train_cfg = config["training"]
    base_lr = train_cfg.get("lr", 1e-3)

    if config["riemannian"].get("enabled", True):
        param_groups = model.get_param_groups(base_lr)
        optimizer = get_riemannian_optimizer(
            param_groups,
            lr=base_lr,
            optimizer_type=config["riemannian"].get("optimizer", "adam"),
            weight_decay=train_cfg.get("weight_decay", 1e-4),
        )
        print(f"\nOptimizer: RiemannianAdam (geoopt), lr={base_lr}")
    else:
        param_groups = model.get_param_groups(base_lr)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg.get("weight_decay", 1e-4))
        print(f"\nOptimizer: AdamW, lr={base_lr}")

    # LR Scheduler
    sched_cfg = train_cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=sched_cfg.get("T_0", 25),
        T_mult=sched_cfg.get("T_mult", 2),
    )

    # Resume from checkpoint if requested
    start_epoch = 0
    best_Q = 0.0
    best_hierarchy = 0.0
    epochs_without_improvement = 0
    best_epoch = 0

    if args.resume:
        latest_path = save_dir / "latest.pt"
        if latest_path.exists():
            print(f"\nResuming from: {latest_path}")
            ckpt = load_checkpoint_compat(latest_path, map_location=device)
            model_state = get_model_state_dict(ckpt)
            model.load_state_dict(model_state, strict=False)
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_Q = ckpt.get("best_Q", 0.0)
            best_hierarchy = ckpt.get("best_hierarchy", 0.0)
            print(f"  Resuming from epoch {start_epoch}, best_Q={best_Q:.3f}")

    # === Training Loop ===
    print("\n" + "=" * 60)
    print("V5.12.1 TRAINING (Full Hyperbolic Integration)")
    print("=" * 60)
    print("Key changes from V5.12:")
    print("  - Decoder input: log_map_zero(z_A_hyp) instead of mu_A")
    print("  - Metrics: hyperbolic distance instead of Euclidean norm")
    print("  - Gradient flow through hyperbolic projection")
    print("=" * 60)

    n_epochs = train_cfg.get("epochs", 200)
    eval_every = train_cfg.get("eval_every", 5)
    save_every = train_cfg.get("save_every", 25)
    print_every = train_cfg.get("print_every", 5)

    prev_freeze_state = {"encoder_a": True, "encoder_b": False, "controller": False}

    for epoch in range(start_epoch, n_epochs):
        # === Train ===
        train_metrics = train_epoch_v5121(
            model=model,
            optimizer=optimizer,
            x=all_ops,
            indices=indices,
            original_radii=original_radii,
            rich_hierarchy_loss=rich_hierarchy_loss,
            radial_loss_fn=radial_loss_fn,
            rank_loss_fn=rank_loss_fn,
            geodesic_loss_fn=geodesic_loss_fn,
            zero_structure_loss_fn=zero_structure_loss_fn,
            config=config,
            epoch=epoch,
            device=device,
        )

        # Update scheduler
        scheduler.step()

        # === Evaluate ===
        if epoch % eval_every == 0 or epoch == n_epochs - 1:
            metrics = compute_quick_metrics(model, all_ops, indices, device, curvature)
            richness_ratio = metrics["richness"] / (initial_richness + 1e-10)

            # Update homeostasis
            homeo_state = homeostasis.update(
                epoch=epoch,
                coverage=metrics["coverage"],
                hierarchy_A=metrics["hierarchy_A"],
                hierarchy_B=metrics["hierarchy_B"],
                dist_corr_A=metrics["dist_corr"],
            )

            # Check for freeze state changes
            state_changed = (
                homeo_state["encoder_a_frozen"] != prev_freeze_state["encoder_a"]
                or homeo_state["encoder_b_frozen"] != prev_freeze_state["encoder_b"]
            )

            if state_changed:
                model.apply_homeostasis_state(homeo_state)
                # Rebuild optimizer preserving Riemannian type
                param_groups = model.get_param_groups(base_lr)
                if config["riemannian"].get("enabled", True):
                    optimizer = get_riemannian_optimizer(
                        param_groups,
                        lr=base_lr,
                        optimizer_type=config["riemannian"].get("optimizer", "adam"),
                        weight_decay=train_cfg.get("weight_decay", 1e-4),
                    )
                else:
                    optimizer = torch.optim.AdamW(param_groups, weight_decay=train_cfg.get("weight_decay", 1e-4))
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, T_0=sched_cfg.get("T_0", 25), T_mult=sched_cfg.get("T_mult", 2)
                )
                prev_freeze_state = {
                    "encoder_a": homeo_state["encoder_a_frozen"],
                    "encoder_b": homeo_state["encoder_b_frozen"],
                    "controller": homeo_state["controller_frozen"],
                }
                for event in homeo_state.get("events", []):
                    print(f"  [HOMEOSTASIS] {event}")

            # Log to TensorBoard
            writer.add_scalar("Train/loss", train_metrics["loss"], epoch)
            writer.add_scalar("Train/rich_loss", train_metrics["rich_loss"], epoch)
            writer.add_scalar("Train/radial_loss", train_metrics["radial_loss"], epoch)
            writer.add_scalar("Train/phase", train_metrics["phase"], epoch)
            writer.add_scalar("Eval/coverage", metrics["coverage"], epoch)
            writer.add_scalar("Eval/hierarchy_B", metrics["hierarchy_B"], epoch)
            writer.add_scalar("Eval/richness", metrics["richness"], epoch)
            writer.add_scalar("Eval/richness_ratio", richness_ratio, epoch)
            writer.add_scalar("Eval/Q", metrics["Q"], epoch)
            writer.add_scalar("Eval/r_v0", metrics["r_v0"], epoch)
            writer.add_scalar("Eval/r_v9", metrics["r_v9"], epoch)
            writer.add_scalar("Homeostasis/Q", homeo_state.get("current_Q", 0), epoch)

            # Print progress
            if epoch % print_every == 0 or epoch == n_epochs - 1:
                phase_str = f"Phase {train_metrics['phase']}"
                print(f"\nEpoch {epoch}/{n_epochs} [{phase_str}]")
                print(
                    f"  Loss: {train_metrics['loss']:.4f} (rich: {train_metrics['rich_loss']:.4f}, radial: {train_metrics['radial_loss']:.4f})"
                )
                print(f"  Coverage: {metrics['coverage'] * 100:.2f}%")
                print(f"  Hierarchy_B: {metrics['hierarchy_B']:.4f} (target: -0.80)")
                print(f"  Richness: {metrics['richness']:.6f} (ratio: {richness_ratio:.2f}, target: >0.5)")
                print(f"  r_v0: {metrics['r_v0']:.4f}, r_v9: {metrics['r_v9']:.4f} (hyperbolic distance)")
                print(f"  Q: {metrics['Q']:.3f} (best: {best_Q:.3f})")
                print(f"  Freeze: {model.get_freeze_state_summary()}")

            # Track best models
            is_best_Q = metrics["Q"] > best_Q
            is_best_hier = metrics["hierarchy_B"] < best_hierarchy and metrics["coverage"] > 0.99

            if is_best_Q:
                best_Q = metrics["Q"]
                best_epoch = epoch
                epochs_without_improvement = 0
                print(f"  [NEW BEST Q: {best_Q:.3f}]")

                # Save best Q checkpoint
                full_metrics_q = compute_comprehensive_metrics(model, device)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "metrics": full_metrics_q.to_dict(),
                        "train_metrics": train_metrics,
                        "richness_ratio": richness_ratio,
                        "best_Q": best_Q,
                        "best_hierarchy": best_hierarchy,
                        "homeostasis_state": homeostasis.get_state_summary(),
                        "config": config,
                        "version": config.get("version", {}),
                    },
                    save_dir / "best_Q.pt",
                )

            if is_best_hier:
                best_hierarchy = metrics["hierarchy_B"]
                print(f"  [NEW BEST HIERARCHY: {best_hierarchy:.4f}]")

                # Use ComprehensiveMetrics for checkpoint storage
                full_metrics = compute_comprehensive_metrics(model, device)

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "metrics": full_metrics.to_dict(),
                        "train_metrics": train_metrics,
                        "richness_ratio": richness_ratio,
                        "best_Q": best_Q,
                        "best_hierarchy": best_hierarchy,
                        "homeostasis_state": homeostasis.get_state_summary(),
                        "config": config,
                        "version": config.get("version", {}),
                    },
                    save_dir / "best.pt",
                )

            # Track epochs without improvement
            if not is_best_Q and not is_best_hier:
                epochs_without_improvement += 1
            elif is_best_hier:
                epochs_without_improvement = 0  # Reset on hierarchy improvement too

        # Periodic checkpoint
        if epoch % save_every == 0 or epoch == n_epochs - 1:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_Q": best_Q,
                    "best_hierarchy": best_hierarchy,
                    "epochs_without_improvement": epochs_without_improvement,
                    "config": config,
                },
                save_dir / "latest.pt",
            )

        # Early stopping check
        patience = train_cfg.get("patience", 25)
        min_epochs = train_cfg.get("min_epochs", 40)
        if epoch >= min_epochs and epochs_without_improvement >= patience:
            print(f"\n[EARLY STOPPING] No improvement for {patience} epochs.")
            print(f"  Best Q: {best_Q:.3f} at epoch {best_epoch}")
            print(f"  Best Hierarchy: {best_hierarchy:.4f}")
            break

    # === Final Summary ===
    print("\n" + "=" * 60)
    print("V5.12.1 TRAINING COMPLETE")
    print("=" * 60)

    # Compute final comprehensive metrics
    final_metrics = compute_comprehensive_metrics(model, device)
    print("\nFinal Metrics:")
    print(f"  Coverage: {final_metrics.coverage * 100:.2f}%")
    print(f"  Hierarchy_B: {final_metrics.hierarchy_B:.4f}")
    print(f"  Richness_B: {final_metrics.richness_B:.6f}")
    print(f"  dist_corr_B: {final_metrics.dist_corr_B:.4f}")
    print(f"  r_v0_B: {final_metrics.r_v0_B:.4f}")
    print(f"  r_v9_B: {final_metrics.r_v9_B:.4f}")
    print(f"  Q_B: {final_metrics.Q_B:.3f}")

    # Check against targets
    targets = config.get("targets", {})
    print("\nTarget Comparison:")
    print(f"  Coverage: {final_metrics.coverage * 100:.1f}% (target: {targets.get('coverage', 1.0) * 100:.0f}%)")
    print(f"  Hierarchy_B: {final_metrics.hierarchy_B:.4f} (target: {targets.get('hierarchy_B', -0.80):.2f})")
    print(f"  Richness: {final_metrics.richness_B:.6f} (target: >{targets.get('richness', 0.005):.3f})")
    print(f"  r_v9: {final_metrics.r_v9_B:.4f} (target: <{targets.get('r_v9', 0.15):.2f})")

    print(f"\nCheckpoints saved to: {save_dir}")
    print(f"TensorBoard logs: {log_dir}")

    writer.close()


if __name__ == "__main__":
    main()
