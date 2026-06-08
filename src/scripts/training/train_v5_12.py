# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""V5.12 Production Training Script - Highest Quality Ternary VAE.

This script implements the V5.12 training pipeline combining all best practices:
- RichHierarchyLoss as PRIMARY loss (preserves richness while maximizing hierarchy)
- ComprehensiveMetrics for standardized checkpoint storage
- Two-phase loss strategy (structure → geometry refinement)
- HomeostasisController with Q-tracking
- Riemannian optimizer (geoopt)
- Enhanced stratified sampling (25% high-v budget)

Target Metrics:
- Coverage: 100%
- Hierarchy_B: -0.8321 (ceiling)
- Richness: >0.008
- r_v9: 0.12-0.15 (improved from 0.19)
- dist_corr: >0.7

Device: RTX 2060 SUPER (8GB VRAM) compatible

Usage:
    python src/scripts/training/train_v5_12.py
    python src/scripts/training/train_v5_12.py --config src/configs/v5_12.yaml
    python src/scripts/training/train_v5_12.py --epochs 200 --resume
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as torch_lib
import yaml
from scipy.stats import spearmanr
from torch.utils.tensorboard import SummaryWriter

# Add project root (train_v5_12.py is in src/scripts/training/, so parents[3] is project root)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import RUNS_DIR
from src.core import TERNARY
from src.core.metrics import compute_comprehensive_metrics
from src.data.generation import generate_all_ternary_operations
from src.geometry import get_riemannian_optimizer
from src.losses import (
    CombinedZeroStructureLoss,
    GlobalRankLoss,
    PAdicGeodesicLoss,
    RadialHierarchyLoss,
    RichHierarchyLoss,
)

# from src.losses.adaptive_rich_hierarchy import AdaptiveRichHierarchyLoss, create_adaptive_rich_hierarchy_loss
from src.models import HomeostasisController, TernaryVAEV5_11_PartialFreeze
from src.models.homeostasis import compute_Q
from src.training.adaptive_lr_scheduler import AdaptiveLRScheduler, ValidationMetrics, create_adaptive_lr_scheduler
from src.training.gradient_checkpointing import apply_gradient_checkpointing, create_checkpoint_config
from src.training.grokking_detector import EpochMetrics, GrokDetector, GrokDetectorConfig, TrainingPhase
from src.training.optimizations import MixedPrecisionConfig, MixedPrecisionTrainer
from src.utils.checkpoint import get_model_state_dict, load_checkpoint_compat


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def validate_config(config: dict) -> None:
    """Validate V5.12 configuration structure.

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


def check_cuda(force_cpu: bool = False):
    """Verify CUDA/ROCm is available and print device info.

    Args:
        force_cpu: If True, use CPU even if GPU is available (for testing)
    """
    print(f"\n{'=' * 60}")
    print("V5.12 DEVICE CONFIGURATION")
    print(f"{'=' * 60}")

    if force_cpu:
        print("  Device: CPU (forced for testing)")
        print(f"  PyTorch: {torch_lib.__version__}")
        print("  NOTE: Training will be slow on CPU")
        print(f"{'=' * 60}\n")
        return torch_lib.device("cpu")

    if not torch_lib.cuda.is_available():
        print("WARNING: CUDA/ROCm not available.")
        print("  For NVIDIA: pip install torch --index-url https://download.pytorch.org/whl/cu126")
        print("  For AMD: pip install torch --index-url https://download.pytorch.org/whl/rocm6.0")
        print("  Falling back to CPU (training will be slow)")
        print(f"  PyTorch: {torch_lib.__version__}")
        print(f"{'=' * 60}\n")
        return torch_lib.device("cpu")

    device = torch_lib.device("cuda:0")
    props = torch_lib.cuda.get_device_properties(0)
    print(f"  Device: {props.name}")
    print(f"  VRAM: {props.total_memory / 1024**3:.1f} GB")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  PyTorch: {torch_lib.__version__}")

    # Check for ROCm vs CUDA
    if hasattr(torch_lib.version, "hip") and torch_lib.version.hip:
        print(f"  ROCm/HIP: {torch_lib.version.hip}")
    elif torch_lib.version.cuda:
        print(f"  CUDA: {torch_lib.version.cuda}")

    print(f"{'=' * 60}\n")
    return device


def create_stratified_indices(indices: torch_lib.Tensor, batch_size: int, device: str, high_v_ratio: float = 0.25):
    """Create stratified batch indices with enhanced high-valuation sampling.

    V5.12: Increased high-v budget to 25% (from 20%) for better r_v9 learning.

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
        valuation_groups[v] = torch_lib.tensor(valuation_groups[v], device=device)

    # V5.12: 25% for high-v (v>=4)
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
                sample_idx = torch_lib.randint(0, len(group), (per_high_v,), device=device)
                batch_indices.append(group[sample_idx])

        # Sample from low-valuation levels (proportional)
        if low_v_levels:
            total_low = sum(len(valuation_groups[v]) for v in low_v_levels)
            for v in low_v_levels:
                group = valuation_groups[v]
                n_to_sample = max(1, int(low_v_budget * len(group) / total_low))
                sample_idx = torch_lib.randint(0, len(group), (n_to_sample,), device=device)
                batch_indices.append(group[sample_idx])

        # Combine and trim to exact batch size
        batch = torch_lib.cat(batch_indices)
        if len(batch) > batch_size:
            batch = batch[torch_lib.randperm(len(batch), device=device)[:batch_size]]
        elif len(batch) < batch_size:
            extra = torch_lib.randint(0, n_samples, (batch_size - len(batch),), device=device)
            batch = torch_lib.cat([batch, extra])

        batches.append(batch)

    return batches


def compute_quick_metrics(model, all_ops, indices, device):
    """Compute quick metrics for training monitoring."""
    model.eval()
    batch_size = 4096

    all_radii_A = []
    all_radii_B = []
    all_correct = []

    with torch_lib.no_grad():
        for i in range(0, len(all_ops), batch_size):
            batch_ops = all_ops[i : i + batch_size].to(device)

            out = model(batch_ops, compute_control=False)
            z_A = out["z_A_hyp"]
            z_B = out["z_B_hyp"]

            all_radii_A.append(z_A.norm(dim=-1).cpu().numpy())
            all_radii_B.append(z_B.norm(dim=-1).cpu().numpy())

            # Coverage check
            logits = model.decoder_A(out["mu_A"])
            preds = torch_lib.argmax(logits, dim=-1) - 1
            correct = (preds == batch_ops.long()).float().mean(dim=1).cpu().numpy()
            all_correct.append(correct)

    all_radii_A = np.concatenate(all_radii_A)
    all_radii_B = np.concatenate(all_radii_B)
    all_correct = np.concatenate(all_correct)
    valuations = TERNARY.valuation(indices).cpu().numpy()

    coverage = (all_correct == 1.0).mean()
    hierarchy_A = spearmanr(valuations, all_radii_A)[0]
    hierarchy_B = spearmanr(valuations, all_radii_B)[0]

    # Richness (within-level variance)
    richness = 0
    for v in range(10):
        mask = valuations == v
        if mask.sum() > 1:
            richness += all_radii_B[mask].var()
    richness /= 10

    # Radius by valuation
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


def train_epoch_v512(
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
    mp_trainer=None,
):
    """Train one epoch with V5.12 two-phase strategy.

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

        # Mixed precision forward pass and loss computation (Phase 1.2)
        if mp_trainer is not None:
            with mp_trainer.autocast():
                # Forward pass
                out = model(x_batch, compute_control=False)
                z_A = out["z_A_hyp"]
                z_B = out["z_B_hyp"]
                logits = out["logits_A"]  # Already computed in forward pass

                # === PRIMARY: RichHierarchyLoss (preserves richness) ===
                rich_losses = rich_hierarchy_loss(z_B, idx_batch, logits, x_batch, orig_radii_batch)
                rich_loss = rich_losses["total"]

                # === AUXILIARY: RadialHierarchyLoss ===
                rad_loss_A, _ = radial_loss_fn(z_A, idx_batch)
                rad_loss_B, _ = radial_loss_fn(z_B, idx_batch)
                rad_loss = rad_loss_A + rad_loss_B

                # === STRUCTURAL: GlobalRankLoss ===
                rank_loss = torch_lib.tensor(0.0, device=device)
                if rank_loss_fn is not None:
                    rank_loss_A, _ = rank_loss_fn(z_A, idx_batch)
                    rank_loss_B, _ = rank_loss_fn(z_B, idx_batch)
                    rank_loss = rank_loss_A + rank_loss_B

                # === Zero-structure loss ===
                zero_loss = torch_lib.tensor(0.0, device=device)
                if zero_structure_loss_fn is not None:
                    zero_loss_A = zero_structure_loss_fn(z_A, x_batch)
                    zero_loss_B = zero_structure_loss_fn(z_B, x_batch)
                    zero_loss = zero_loss_A + zero_loss_B

                # === PHASE 2: Geodesic refinement ===
                geo_loss = torch_lib.tensor(0.0, device=device)
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

            # Mixed precision backward and optimizer step
            mp_trainer.backward(loss)
            mp_trainer.step(
                optimizer, clip_grad_norm=config["training"].get("max_grad_norm", 1.0), parameters=model.parameters()
            )
        else:
            # Standard precision fallback
            out = model(x_batch, compute_control=False)
            z_A = out["z_A_hyp"]
            z_B = out["z_B_hyp"]
            logits = out["logits_A"]

            rich_losses = rich_hierarchy_loss(z_B, idx_batch, logits, x_batch, orig_radii_batch)
            rich_loss = rich_losses["total"]

            rad_loss_A, _ = radial_loss_fn(z_A, idx_batch)
            rad_loss_B, _ = radial_loss_fn(z_B, idx_batch)
            rad_loss = rad_loss_A + rad_loss_B

            rank_loss = torch_lib.tensor(0.0, device=device)
            if rank_loss_fn is not None:
                rank_loss_A, _ = rank_loss_fn(z_A, idx_batch)
                rank_loss_B, _ = rank_loss_fn(z_B, idx_batch)
                rank_loss = rank_loss_A + rank_loss_B

            zero_loss = torch_lib.tensor(0.0, device=device)
            if zero_structure_loss_fn is not None:
                zero_loss_A = zero_structure_loss_fn(z_A, x_batch)
                zero_loss_B = zero_structure_loss_fn(z_B, x_batch)
                zero_loss = zero_loss_A + zero_loss_B

            geo_loss = torch_lib.tensor(0.0, device=device)
            if is_phase_2 and geodesic_loss_fn is not None:
                geo_loss_A, _ = geodesic_loss_fn(z_A, idx_batch)
                geo_loss_B, _ = geodesic_loss_fn(z_B, idx_batch)
                geo_loss = geo_loss_A + geo_loss_B

            loss = (
                rich_loss
                + radial_weight * rad_loss
                + rank_weight * rank_loss
                + zero_weight * zero_loss
                + geodesic_weight * geo_loss
            )

            # Standard backward and optimizer step
            loss.backward()
            torch_lib.nn.utils.clip_grad_norm_(model.parameters(), config["training"].get("max_grad_norm", 1.0))
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
    parser = argparse.ArgumentParser(description="Train V5.12 Production Model")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/v5_12.yaml",
        help="Path to V5.12 config YAML",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on (cuda, cpu, or rocm)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode (for testing)")
    args = parser.parse_args()

    # Check CUDA/ROCm
    force_cpu = args.cpu or args.device == "cpu"
    device = check_cuda(force_cpu=force_cpu)

    # Load config
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    print(f"Loaded V5.12 config from: {config_path}")

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
    log_dir = RUNS_DIR / f"v5_12_production_{timestamp}"
    writer = SummaryWriter(log_dir=str(log_dir))

    # === Create Model ===
    print("\n=== Creating V5.12 Model ===")
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
        encoder_a_lr_scale=config["option_c"].get("encoder_a_lr_scale", 0.05),
    )

    # Load frozen checkpoint
    frozen_cfg = config.get("frozen_checkpoint", {})
    checkpoint_path = frozen_cfg.get("path", "checkpoints/v5_5/latest.pt")

    if checkpoint_path is None or checkpoint_path == "null":
        print("Training from scratch (no checkpoint specified)")
        frozen_path = None
    else:
        frozen_path = PROJECT_ROOT / checkpoint_path

    if frozen_path is not None and frozen_path.exists():
        print(f"Loading frozen checkpoint: {frozen_path}")
        ckpt = load_checkpoint_compat(frozen_path, map_location=device)
        model_state = get_model_state_dict(ckpt)
        model.load_state_dict(model_state, strict=False)
        print(f"  Loaded checkpoint (keys: {list(ckpt.keys())[:5]}...)")
    else:
        print(f"WARNING: Frozen checkpoint not found at {frozen_path}")
        print("  Training will start with random initialization.")

    model = model.to(device)

    # Apply torch_lib.compile optimization if enabled (Phase 1.1)
    compile_config = config.get("torch_lib_compile", {})
    if compile_config.get("enabled", False) and hasattr(torch_lib, "compile"):
        try:
            backend = compile_config.get("backend", "eager")
            mode = compile_config.get("mode", "default")
            fullgraph = compile_config.get("fullgraph", False)

            print(f"🚀 Applying torch_lib.compile: backend={backend}, mode={mode}, fullgraph={fullgraph}")

            # Set dynamo config to suppress compilation errors and fallback to eager
            try:
                import torch._dynamo as _dynamo

                _dynamo.config.suppress_errors = True
            except ImportError:
                pass  # torch._dynamo not available

            compiled_model = torch_lib.compile(model, backend=backend, mode=mode, fullgraph=fullgraph)

            # Test compilation with a small forward pass
            test_input = torch_lib.randn(2, 9, device=device)
            with torch_lib.no_grad():
                _ = compiled_model(test_input)

            model = compiled_model
            print("✅ torch_lib.compile optimization enabled and tested!")
        except Exception as e:
            print(f"⚠️  torch_lib.compile failed ({e}), continuing with eager mode")
            print("    Model will train normally without compilation optimization")
    elif compile_config.get("enabled", False):
        print("⚠️  torch_lib.compile requested but not available (PyTorch < 2.0)")

    # Apply Gradient Checkpointing (Phase 2.1)
    checkpoint_config = create_checkpoint_config(config)
    if checkpoint_config.enabled:
        print(f"🚀 Applying gradient checkpointing: segments={checkpoint_config.segments}")
        print(f"  encoder_checkpoint={checkpoint_config.encoder_checkpoint}")
        print(f"  decoder_checkpoint={checkpoint_config.decoder_checkpoint}")

        model = apply_gradient_checkpointing(model, checkpoint_config)
        print("✅ 30-40% VRAM reduction expected (trade: 10-15% slower training)!")
    else:
        print("⚠️  Gradient checkpointing disabled")

    # Setup Mixed Precision Training (Phase 1.2)
    mp_config_dict = config.get("mixed_precision", {})
    if mp_config_dict.get("enabled", False):
        mp_config = MixedPrecisionConfig(
            enabled=mp_config_dict.get("enabled", False),
            dtype=mp_config_dict.get("dtype", "float16"),
            init_scale=mp_config_dict.get("init_scale", 65536.0),
            growth_factor=mp_config_dict.get("growth_factor", 2.0),
            backoff_factor=mp_config_dict.get("backoff_factor", 0.5),
            growth_interval=mp_config_dict.get("growth_interval", 2000),
        )
        mp_trainer = MixedPrecisionTrainer(mp_config)
        print(f"🚀 Mixed precision enabled: dtype={mp_config.dtype}, init_scale={mp_config.init_scale}")
        print("✅ 2.0x speedup + 20-30% VRAM reduction expected!")
    else:
        mp_trainer = None
        print("⚠️  Mixed precision disabled")

    # Set initial freeze state
    model.set_encoder_a_frozen(True)
    model.set_encoder_b_frozen(False)
    print(f"Freeze state: {model.get_freeze_state_summary()}")

    # Count parameters
    param_counts = model.count_parameters()
    print(f"\nParameters: {param_counts['total']:,} total, {param_counts['trainable']:,} trainable")

    # === Dataset ===
    print("\n=== Loading Dataset ===")
    all_ops_np = generate_all_ternary_operations()
    all_ops = torch_lib.tensor(all_ops_np, dtype=torch_lib.float32, device=device)
    indices = torch_lib.arange(len(all_ops), device=device)
    print(f"Dataset size: {len(all_ops)}")

    # Get original radii for richness preservation
    with torch_lib.no_grad():
        model.eval()
        original_radii = []
        for i in range(0, len(all_ops), 4096):
            batch = all_ops[i : i + 4096]
            out = model(batch, compute_control=False)
            original_radii.append(out["z_B_hyp"].norm(dim=-1))
        original_radii = torch_lib.cat(original_radii)
        model.train()
    print(f"Original radii: {original_radii.min():.4f} - {original_radii.max():.4f}")

    # Initial metrics
    init_metrics = compute_quick_metrics(model, all_ops, indices, device)
    print("\nInitial metrics:")
    print(f"  Coverage: {init_metrics['coverage'] * 100:.1f}%")
    print(f"  Hierarchy_B: {init_metrics['hierarchy_B']:.4f}")
    print(f"  Richness: {init_metrics['richness']:.6f}")
    print(f"  Q: {init_metrics['Q']:.3f}")
    initial_richness = init_metrics["richness"]

    # === Loss Functions ===
    print("\n=== Creating V5.12 Loss Functions ===")
    loss_cfg = config["loss"]

    # PRIMARY: AdaptiveRichHierarchyLoss (Phase 2.2)
    rich_cfg = loss_cfg["rich_hierarchy"]
    use_adaptive_loss = False  # loss_cfg.get('adaptive_loss', {}).get('enabled', False)

    if use_adaptive_loss:
        # rich_hierarchy_loss = create_adaptive_rich_hierarchy_loss(loss_cfg).to(device)
        pass
        print(
            f"  🚀 AdaptiveRichHierarchyLoss: curriculum={loss_cfg.get('adaptive_loss', {}).get('enable_curriculum', True)}"
        )
        print(f"    difficulty_adaptive={loss_cfg.get('adaptive_loss', {}).get('enable_difficulty_adaptive', True)}")
        print(
            f"    performance_rebalancing={loss_cfg.get('adaptive_loss', {}).get('enable_performance_rebalancing', True)}"
        )
    else:
        rich_hierarchy_loss = RichHierarchyLoss(
            inner_radius=loss_cfg["radial"].get("inner_radius", 0.08),
            outer_radius=loss_cfg["radial"].get("outer_radius", 0.90),
            hierarchy_weight=rich_cfg.get("hierarchy_weight", 5.0),
            coverage_weight=rich_cfg.get("coverage_weight", 1.0),
            richness_weight=rich_cfg.get("richness_weight", 2.0),
            separation_weight=rich_cfg.get("separation_weight", 3.0),
            min_richness_ratio=rich_cfg.get("min_richness_ratio", 0.5),
        ).to(device)
        print(
            f"  RichHierarchyLoss: hierarchy={rich_cfg.get('hierarchy_weight', 5.0)}, richness={rich_cfg.get('richness_weight', 2.0)}"
        )

    # AUXILIARY: RadialHierarchyLoss
    radial_cfg = loss_cfg["radial"]
    radial_loss_fn = RadialHierarchyLoss(
        inner_radius=radial_cfg.get("inner_radius", 0.08),
        outer_radius=radial_cfg.get("outer_radius", 0.90),
        margin_weight=radial_cfg.get("margin_weight", 0.5),
        use_margin_loss=True,
    ).to(device)
    print(
        f"  RadialHierarchyLoss: inner={radial_cfg.get('inner_radius', 0.08)}, outer={radial_cfg.get('outer_radius', 0.90)}"
    )

    # STRUCTURAL: GlobalRankLoss
    rank_cfg = loss_cfg["rank"]
    rank_loss_fn = (
        GlobalRankLoss(
            temperature=rank_cfg.get("temperature", 0.1),
            n_pairs=rank_cfg.get("n_pairs", 2000),
        ).to(device)
        if rank_cfg.get("enabled", True)
        else None
    )
    if rank_loss_fn:
        print(f"  GlobalRankLoss: weight={rank_cfg.get('weight', 0.5)}")

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

    # Zero-structure loss
    zero_cfg = loss_cfg["zero_structure"]
    zero_structure_loss_fn = (
        CombinedZeroStructureLoss(
            valuation_weight=zero_cfg.get("valuation_weight", 0.5),
            sparsity_weight=zero_cfg.get("sparsity_weight", 0.3),
            inner_radius=radial_cfg.get("inner_radius", 0.08),
            outer_radius=radial_cfg.get("outer_radius", 0.90),
        ).to(device)
        if zero_cfg.get("enabled", True)
        else None
    )
    if zero_structure_loss_fn:
        print("  ZeroStructureLoss: enabled")

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
        optimizer = torch_lib.optim.AdamW(param_groups, weight_decay=train_cfg.get("weight_decay", 1e-4))
        print(f"\nOptimizer: AdamW, lr={base_lr}")

    # LR Scheduler (Phase 2.3: Adaptive validation-based scheduling)
    sched_cfg = train_cfg.get("scheduler", {})
    use_adaptive_lr = sched_cfg.get("adaptive_lr", {}).get("enabled", False)

    if use_adaptive_lr:
        scheduler = create_adaptive_lr_scheduler(optimizer, train_cfg)
        print(
            f"🚀 Adaptive LR Scheduler: monitoring {train_cfg.get('adaptive_lr', {}).get('primary_metric', 'hierarchy_correlation')}"
        )
        print(
            f"  patience={train_cfg.get('adaptive_lr', {}).get('patience', 8)}, factor={train_cfg.get('adaptive_lr', {}).get('factor', 0.5)}"
        )
    else:
        scheduler = torch_lib.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_cfg.get("T_0", 25),
            T_mult=sched_cfg.get("T_mult", 2),
        )
        print(f"LR Scheduler: CosineAnnealingWarmRestarts, T_0={sched_cfg.get('T_0', 25)}")

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

    # === Initialize Grokking Detection ===
    print("🔍 Initializing grokking detection for real-time monitoring...")
    grok_config = GrokDetectorConfig(
        short_window=5,  # Smaller window for faster detection
        long_window=20,  # Smaller window for faster detection
        min_epochs_for_detection=10,  # Reduce from 50 to 10 for shorter runs
        grokking_patience=50,  # Reduce from 100 to 50
        memorization_loss_threshold=0.5,  # Higher threshold (our losses are ~3.0)
        improvement_threshold=0.01,  # Smaller threshold for subtle improvements
    )
    grok_detector = GrokDetector(grok_config)

    # === Training Loop ===
    print("\n" + "=" * 60)
    print("V5.12 PRODUCTION TRAINING")
    print("=" * 60)

    n_epochs = train_cfg.get("epochs", 200)
    eval_every = train_cfg.get("eval_every", 5)
    save_every = train_cfg.get("save_every", 25)
    print_every = train_cfg.get("print_every", 5)

    prev_freeze_state = {"encoder_a": True, "encoder_b": False, "controller": False}

    for epoch in range(start_epoch, n_epochs):
        # === Train ===
        train_metrics = train_epoch_v512(
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
            mp_trainer=mp_trainer,
        )

        # === Evaluate ===
        if epoch % eval_every == 0 or epoch == n_epochs - 1:
            metrics = compute_quick_metrics(model, all_ops, indices, device)
            richness_ratio = metrics["richness"] / (initial_richness + 1e-10)

            # === Adaptive Loss Feedback (Phase 2.2) ===
            if hasattr(rich_hierarchy_loss, "update_training_state"):
                # Update training state for adaptive mechanisms (use simple step count)
                total_steps = epoch * (19683 // config["training"]["batch_size"])  # Approximate steps
                rich_hierarchy_loss.update_training_state(epoch, total_steps)

                # Provide performance feedback for rebalancing
                coverage_accuracy = metrics["coverage"]
                hierarchy_correlation = abs(metrics["hierarchy_B"])  # Use absolute value for correlation metric
                rich_hierarchy_loss.update_performance_metrics(
                    hierarchy_correlation=hierarchy_correlation,
                    richness_ratio=richness_ratio,
                    coverage_accuracy=coverage_accuracy,
                )

            # === Adaptive LR Scheduling (Phase 2.3) ===
            if hasattr(scheduler, "step") and isinstance(scheduler, AdaptiveLRScheduler):
                # Create ValidationMetrics for adaptive scheduler
                val_metrics = ValidationMetrics(
                    epoch=epoch,
                    primary_metric=abs(metrics["hierarchy_B"]),  # Default: hierarchy correlation
                    hierarchy_correlation=abs(metrics["hierarchy_B"]),
                    coverage_accuracy=metrics["coverage"],
                    richness_ratio=richness_ratio,
                    loss_value=train_metrics["loss"],
                )

                scheduler_state = scheduler.step(val_metrics)

                # Log scheduler decisions
                if scheduler_state["phase"] != "warmup" and epoch % print_every == 0:
                    print(f"  📊 LR Scheduler: {scheduler_state['phase']} (LR: {scheduler_state['current_lr']:.6f})")
                    if scheduler_state["num_bad_epochs"] > 0:
                        print(
                            f"    Bad epochs: {scheduler_state['num_bad_epochs']}/{scheduler_state['current_patience']}"
                        )

                # Check for early stopping
                if scheduler_state["should_stop_early"]:
                    print("  ⏹️  Early stopping triggered by LR scheduler")
                    break
            else:
                # Standard scheduler step
                scheduler.step()

        else:
            # Update scheduler even when not evaluating (for cosine annealing)
            if not isinstance(scheduler, AdaptiveLRScheduler):
                scheduler.step()

            # === Grokking Detection ===
            epoch_metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                val_loss=metrics.get("val_loss", train_metrics["loss"]),  # Use train_loss as proxy if no val
                correlation=abs(metrics["hierarchy_B"]),  # Use hierarchy as correlation proxy
                coverage=metrics["coverage"] * 100,  # Convert to percentage
                weight_norm=sum(p.norm().item() for p in model.parameters() if p.requires_grad),
                gradient_norm=train_metrics.get("grad_norm", 0.0),
            )

            grok_analysis = grok_detector.update(epoch_metrics)

            # Report grokking insights
            if grok_analysis.current_phase != TrainingPhase.WARMUP:
                phase_emoji = {
                    TrainingPhase.MEMORIZATION: "🧠",
                    TrainingPhase.PLATEAU: "📊",
                    TrainingPhase.GROKKING: "⚡",
                    TrainingPhase.DEGRADATION: "⚠️",
                    TrainingPhase.CONVERGED: "✅",
                }.get(grok_analysis.current_phase, "🔄")

                print(
                    f"  🔍 Grokking: {phase_emoji} {grok_analysis.current_phase.value.upper()} "
                    f"(p={grok_analysis.grokking_probability:.3f}, trend={grok_analysis.trend_direction})"
                )

                if grok_analysis.warnings:
                    for warning in grok_analysis.warnings:
                        print(f"    ⚠️  {warning}")

                if grok_analysis.recommendations:
                    for rec in grok_analysis.recommendations:
                        print(f"    💡 {rec}")

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
                    optimizer = torch_lib.optim.AdamW(param_groups, weight_decay=train_cfg.get("weight_decay", 1e-4))
                # Recreate scheduler (preserve adaptive settings)
                if use_adaptive_lr:
                    scheduler = create_adaptive_lr_scheduler(optimizer, train_cfg)
                else:
                    scheduler = torch_lib.optim.lr_scheduler.CosineAnnealingWarmRestarts(
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
                print(f"  r_v0: {metrics['r_v0']:.4f}, r_v9: {metrics['r_v9']:.4f} (target: 0.12-0.15)")
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
                torch_lib.save(
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

                torch_lib.save(
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
            torch_lib.save(
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
    print("V5.12 TRAINING COMPLETE")
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
    print(f"  Richness: {final_metrics.richness_B:.6f} (target: >{targets.get('richness', 0.007):.3f})")
    print(f"  r_v9: {final_metrics.r_v9_B:.4f} (target: <{targets.get('r_v9', 0.15):.2f})")

    print(f"\nCheckpoints saved to: {save_dir}")
    print(f"TensorBoard logs: {log_dir}")

    writer.close()


if __name__ == "__main__":
    main()
