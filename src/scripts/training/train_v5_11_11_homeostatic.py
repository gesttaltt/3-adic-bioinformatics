# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""V5.11.11 Homeostatic Training Script - Optimized for RTX 2060 SUPER (8GB VRAM).

This script trains the V5.11.11 model with full homeostatic control including:
- Q-gated threshold annealing
- Learnable hyperbolic curvature
- Riemannian optimization
- Mixed precision training (FP16)

Device: AMD Ryzen + NVIDIA GeForce RTX 2060 SUPER
Checkpoint naming: v5_11_11_homeostatic_ale_device

Usage:
    python src/scripts/training/train_v5_11_11_homeostatic.py
    python src/scripts/training/train_v5_11_11_homeostatic.py --epochs 200
    python src/scripts/training/train_v5_11_11_homeostatic.py --resume
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import CHECKPOINTS_DIR


def check_cuda():
    """Verify CUDA is available and print device info."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Please install PyTorch with CUDA support:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu126")
        sys.exit(1)

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    print(f"\n{'=' * 60}")
    print("DEVICE CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"  Device: {props.name}")
    print(f"  VRAM: {props.total_memory / 1024**3:.1f} GB")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"{'=' * 60}\n")
    return device


def check_v5_5_checkpoint(config: dict) -> bool:
    """Check if v5.5 checkpoint exists."""
    checkpoint_path = Path(config.get("frozen_checkpoint", {}).get("path", ""))
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    return checkpoint_path.exists()


def create_v5_5_checkpoint(device: torch.device, save_dir: Path):
    """Train a v5.5 model from scratch to achieve 100% coverage.

    This is a simplified training loop that focuses on coverage only.
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Training v5.5 Base Model (Coverage Training)")
    print("=" * 60)
    print("No existing v5.5 checkpoint found. Training from scratch...")

    from scipy.stats import spearmanr
    from torch.utils.tensorboard import SummaryWriter

    from src.core import TERNARY
    from src.data.generation import generate_all_ternary_operations

    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)

    # Simple dual-encoder VAE for coverage
    class TernaryVAEv5_5(torch.nn.Module):
        def __init__(self, latent_dim=16):
            super().__init__()
            self.latent_dim = latent_dim

            # Encoder A (coverage encoder)
            self.encoder_A = torch.nn.Sequential(
                torch.nn.Linear(9, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
            )
            self.fc_mu_A = torch.nn.Linear(64, latent_dim)
            self.fc_logvar_A = torch.nn.Linear(64, latent_dim)

            # Encoder B (structure encoder - same architecture)
            self.encoder_B = torch.nn.Sequential(
                torch.nn.Linear(9, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
            )
            self.fc_mu_B = torch.nn.Linear(64, latent_dim)
            self.fc_logvar_B = torch.nn.Linear(64, latent_dim)

            # Decoder A (reconstruction decoder)
            self.decoder_A = torch.nn.Sequential(
                torch.nn.Linear(latent_dim, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 27),  # 9 positions x 3 classes
            )

        def encode(self, x):
            h_A = self.encoder_A(x)
            mu_A = self.fc_mu_A(h_A)
            logvar_A = self.fc_logvar_A(h_A)

            h_B = self.encoder_B(x)
            mu_B = self.fc_mu_B(h_B)
            logvar_B = self.fc_logvar_B(h_B)

            return mu_A, logvar_A, mu_B, logvar_B

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            return self.decoder_A(z)

        def forward(self, x):
            mu_A, logvar_A, mu_B, logvar_B = self.encode(x)
            z_A = self.reparameterize(mu_A, logvar_A)
            logits = self.decode(z_A)
            return {
                "logits": logits,
                "mu_A": mu_A,
                "logvar_A": logvar_A,
                "mu_B": mu_B,
                "logvar_B": logvar_B,
                "z_A": z_A,
            }

    # Create model
    model = TernaryVAEv5_5(latent_dim=16).to(device)

    # Create dataset
    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, dtype=torch.float32, device=device)
    indices = torch.arange(len(operations), device=device)

    # Training config for v5.5
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    n_epochs = 200
    batch_size = 512
    best_coverage = 0.0

    print(f"Training v5.5 for {n_epochs} epochs...")
    print(f"Dataset size: {len(x)}")
    print(f"Batch size: {batch_size}")

    writer = SummaryWriter(log_dir=str(PROJECT_ROOT / "runs" / "v5_5_base"))

    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(x), device=device)
        total_loss = 0.0
        n_batches = (len(x) + batch_size - 1) // batch_size

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, len(x))
            batch_idx = perm[start:end]
            x_batch = x[batch_idx]

            optimizer.zero_grad()
            outputs = model(x_batch)

            # Reconstruction loss
            logits = outputs["logits"].view(-1, 9, 3)
            targets = (x_batch.long() + 1).clamp(0, 2)  # Map -1,0,1 to 0,1,2
            recon_loss = criterion(logits.permute(0, 2, 1), targets)

            # KL divergence (with free bits)
            mu_A, logvar_A = outputs["mu_A"], outputs["logvar_A"]
            kl_loss = -0.5 * torch.mean(1 + logvar_A - mu_A.pow(2) - logvar_A.exp())
            kl_loss = torch.clamp(kl_loss, min=0.3)  # Free bits

            loss = recon_loss + 0.1 * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        # Evaluate coverage
        model.eval()
        with torch.no_grad():
            outputs = model(x)
            logits = outputs["logits"].view(-1, 9, 3)
            preds = torch.argmax(logits, dim=-1) - 1
            targets = x.long()
            correct = (preds == targets).float().mean(dim=1)
            coverage = (correct == 1.0).sum().item() / len(x)

            # Radial correlation for monitoring
            mu_A = outputs["mu_A"]
            radii = torch.norm(mu_A, dim=1).cpu().numpy()
            valuations = TERNARY.valuation(indices).cpu().numpy()
            radial_corr = spearmanr(valuations, radii)[0]

        writer.add_scalar("Train/loss", total_loss / n_batches, epoch)
        writer.add_scalar("Eval/coverage", coverage, epoch)
        writer.add_scalar("Eval/radial_corr", radial_corr, epoch)

        if epoch % 10 == 0 or coverage > best_coverage:
            print(
                f"  Epoch {epoch:3d}: loss={total_loss / n_batches:.4f}, "
                f"coverage={coverage * 100:.1f}%, radial_corr={radial_corr:.3f}"
            )

        # Save best
        if coverage > best_coverage:
            best_coverage = coverage
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "coverage": coverage,
                    "radial_corr": radial_corr,
                },
                save_dir / "latest.pt",
            )

            if coverage >= 1.0:
                print(f"\n  *** 100% COVERAGE ACHIEVED at epoch {epoch}! ***\n")
                break

    writer.close()

    print(f"\nV5.5 training complete. Best coverage: {best_coverage * 100:.1f}%")
    print(f"Checkpoint saved to: {save_dir / 'latest.pt'}")

    return save_dir / "latest.pt"


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train V5.11.11 Homeostatic Model - RTX 2060 SUPER Optimized")
    parser.add_argument(
        "--config",
        type=str,
        default="src/configs/v5_11_11_homeostatic_ale_device.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--skip_v5_5", action="store_true", help="Skip v5.5 training if missing")
    args = parser.parse_args()

    # Check CUDA
    device = check_cuda()

    # Load config
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    print(f"Loaded config from: {config_path}")

    # Override config with command line args
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["lr"] = args.lr

    # Check/create v5.5 checkpoint
    v5_5_dir = CHECKPOINTS_DIR / "v5_5"
    if not check_v5_5_checkpoint(config):
        if args.skip_v5_5:
            print("WARNING: v5.5 checkpoint not found and --skip_v5_5 specified.")
            print("Training will start with random initialization.")
        else:
            print(f"v5.5 checkpoint not found at: {config.get('frozen_checkpoint', {}).get('path')}")
            v5_5_path = create_v5_5_checkpoint(device, v5_5_dir)
            config["frozen_checkpoint"] = {"path": str(v5_5_path)}
    else:
        print(f"Using existing v5.5 checkpoint: {config['frozen_checkpoint']['path']}")

    # Now run the main V5.11.11 training
    print("\n" + "=" * 60)
    print("PHASE 2: Training V5.11.11 Homeostatic Model")
    print("=" * 60)

    # Build command line args for the main train.py
    train_args = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "scripts" / "train.py"),
        "--option_c",
        "--dual_projection",
        "--homeostasis",
        "--riemannian",
        "--learnable_curvature",
        "--manifold_aware",
        "--zero_structure",
        f"--epochs={config['training']['epochs']}",
        f"--batch_size={config['training']['batch_size']}",
        f"--lr={config['training']['lr']}",
        f"--v5_5_checkpoint={config['frozen_checkpoint']['path']}",
        f"--save_dir={CHECKPOINTS_DIR / 'v5_11_11_homeostatic_ale_device'}",
        f"--hierarchy_threshold={config['training'].get('hierarchy_threshold', -0.70)}",
        f"--patience={config['training'].get('patience', 20)}",
        f"--min_epochs={config['training'].get('min_epochs', 30)}",
        f"--coverage_freeze_threshold={config['homeostasis'].get('coverage_freeze_threshold', 0.995)}",
        f"--homeostasis_warmup={config['homeostasis'].get('warmup_epochs', 5)}",
        f"--hysteresis_epochs={config['homeostasis'].get('hysteresis_epochs', 3)}",
        f"--annealing_step={config['homeostasis'].get('annealing_step', 0.005)}",
        f"--coverage_floor={config['homeostasis'].get('coverage_floor', 0.95)}",
        f"--projection_hidden_dim={config['model'].get('projection_hidden_dim', 64)}",
        f"--projection_layers={config['model'].get('projection_layers', 2)}",
        f"--projection_dropout={config['model'].get('projection_dropout', 0.1)}",
        f"--weight_decay={config['training'].get('weight_decay', 1e-3)}",
        f"--radial_weight={config['loss']['radial'].get('radial_weight', 2.0)}",
        f"--margin_weight={config['loss']['radial'].get('margin_weight', 1.0)}",
        f"--rank_loss_weight={config['loss']['rank'].get('weight', 1.0)}",
        f"--zero_valuation_weight={config['loss']['zero_structure'].get('valuation_weight', 1.0)}",
        f"--zero_sparsity_weight={config['loss']['zero_structure'].get('sparsity_weight', 0.5)}",
        f"--n_pairs={config['loss']['geodesic'].get('n_pairs', 2000)}",
    ]

    if config["homeostasis"].get("enable_annealing", True):
        train_args.append("--enable_annealing")
    else:
        train_args.append("--no_annealing")

    # Print training configuration
    print("\nTraining Configuration:")
    print(f"  Epochs: {config['training']['epochs']}")
    print(f"  Batch Size: {config['training']['batch_size']}")
    print(f"  Learning Rate: {config['training']['lr']}")
    print("  Homeostasis: ENABLED")
    print(f"  Q-gated Annealing: {config['homeostasis'].get('enable_annealing', True)}")
    print("  Riemannian Optimizer: ENABLED")
    print("  Learnable Curvature: ENABLED")
    print("  Zero-Structure Loss: ENABLED")
    print("\nCheckpoint: v5_11_11_homeostatic_ale_device")
    print("TensorBoard: runs/ternary_option_c_dual_*")

    # Execute training
    import subprocess

    print("\nStarting training...")
    print("-" * 60)

    result = subprocess.run(train_args, cwd=str(PROJECT_ROOT))

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"\nCheckpoint saved to: {CHECKPOINTS_DIR / 'v5_11_11_homeostatic_ale_device'}/")
        print("  - best.pt: Best model (highest composite score)")
        print("  - latest.pt: Final model")
    else:
        print(f"\nERROR: Training failed with return code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
