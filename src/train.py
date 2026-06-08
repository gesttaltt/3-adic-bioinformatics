# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unified Training Launcher for Ternary VAE.

This module provides a single entry point for all training modes:
- v5.11: Standard Ternary VAE training
- v5.11.11: Homeostatic control training
- epsilon: Epsilon-VAE meta-learning
- hiv: HIV-specific codon VAE
- swarm: Multi-agent swarm training
- predictors: Train downstream predictors

Usage:
    python -m src.train --mode v5.11 --epochs 100
    python -m src.train --mode v5.11.11 --config src/configs/v5_11_11_homeostatic_ale_device.yaml
    python -m src.train --mode epsilon --checkpoints-dir outputs/models
    python -m src.train --mode hiv --dataset stanford
    python -m src.train --mode predictors --predictor resistance

    # List available modes
    python -m src.train --list-modes

    # Quick test (5 epochs)
    python -m src.train --mode v5.11 --quick
"""

import argparse
import sys
from pathlib import Path


def list_modes():
    """Print available training modes."""
    modes = """
Available Training Modes:
========================

  v5.11       Standard Ternary VAE (frozen encoder_A + trainable encoder_B)
              Usage: python -m src.train --mode v5.11 --epochs 100

  v5.11.11    Homeostatic control with Q-gated annealing
              Usage: python -m src.train --mode v5.11.11 --config src/configs/v5_11_11_homeostatic_ale_device.yaml

  epsilon     Epsilon-VAE meta-learning on checkpoints
              Usage: python -m src.train --mode epsilon --checkpoints-dir outputs/

  hiv         HIV-specific codon VAE training
              Usage: python -m src.train --mode hiv --dataset stanford

  swarm       Multi-agent swarm training (experimental)
              Usage: python -m src.train --mode swarm --n-agents 3

  predictors  Train downstream predictors (resistance, escape, neutralization)
              Usage: python -m src.train --mode predictors --predictor resistance

Options:
  --quick     Run quick 5-epoch smoke test
  --config    Path to YAML configuration file
  --device    Device to use (cuda/cpu)
  --epochs    Number of training epochs
  --save-dir  Directory to save outputs
"""
    print(modes)


def train_v5_11(args):
    """Train standard V5.11 Ternary VAE."""
    import subprocess

    cmd = [
        sys.executable,
        "src/scripts/train.py",
        "--epochs",
        str(args.epochs),
        "--device",
        args.device,
        "--batch_size",
        str(args.batch_size),
    ]

    if args.config:
        cmd.extend(["--config", str(args.config)])

    if args.save_dir:
        cmd.extend(["--save_dir", str(args.save_dir)])

    # Add standard options for best performance
    cmd.extend(
        [
            "--option_c",
            "--dual_projection",
            "--riemannian",
        ]
    )

    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def train_v5_11_11(args):
    """Train V5.11.11 with homeostatic control."""
    import subprocess

    script_path = Path("src/scripts/training/train_v5_11_11_homeostatic.py")

    if not script_path.exists():
        # Fallback to main train.py with homeostasis enabled
        cmd = [
            sys.executable,
            "src/scripts/train.py",
            "--epochs",
            str(args.epochs),
            "--device",
            args.device,
            "--batch_size",
            str(args.batch_size),
            "--homeostasis",
            "--option_c",
            "--dual_projection",
            "--riemannian",
            "--learnable_curvature",
            "--manifold_aware",
        ]
    else:
        cmd = [
            sys.executable,
            str(script_path),
        ]

    if args.config:
        cmd.extend(["--config", str(args.config)])

    if args.save_dir:
        cmd.extend(["--save_dir", str(args.save_dir)])

    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def train_epsilon(args):
    """Train Epsilon-VAE for checkpoint exploration."""
    import subprocess

    script_path = Path("src/scripts/epsilon_vae/train_epsilon_vae.py")

    cmd = [
        sys.executable,
        str(script_path),
    ]

    if args.checkpoints_dir:
        cmd.extend(["--checkpoint_dir", str(args.checkpoints_dir)])

    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])

    if args.save_dir:
        cmd.extend(["--save_dir", str(args.save_dir)])

    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def train_hiv(args):
    """Train HIV-specific codon VAE."""
    import subprocess

    script_path = Path("src/scripts/hiv/train_codon_vae_hiv.py")

    cmd = [
        sys.executable,
        str(script_path),
        "--epochs",
        str(args.epochs),
    ]

    if args.dataset:
        cmd.extend(["--dataset", args.dataset])

    if args.save_dir:
        cmd.extend(["--save_dir", str(args.save_dir)])

    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd).returncode


def train_swarm(args):
    """Train with multi-agent swarm."""
    print("Swarm training mode")

    import torch

    from src.models import SwarmVAE
    from src.training import SwarmTrainer

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Create SwarmVAE
    model = SwarmVAE(
        latent_dim=16,
        n_agents=args.n_agents,
        communication_rounds=3,
    ).to(device)

    print(f"Created SwarmVAE with {args.n_agents} agents")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = SwarmTrainer(
        model=model,
        device=device,
    )

    # Generate data
    from src.data.generation import generate_all_ternary_operations

    operations = generate_all_ternary_operations()
    x = torch.tensor(operations, dtype=torch.float32, device=device)

    print(f"Training on {len(x)} operations for {args.epochs} epochs...")

    # Train
    for epoch in range(args.epochs):
        loss = trainer.train_step(x)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{args.epochs}: loss={loss:.4f}")

    # Save
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_dir / "swarm_vae.pt")
        print(f"Saved to {save_dir / 'swarm_vae.pt'}")

    return 0


def train_predictors(args):
    """Train downstream predictors."""
    print(f"Training {args.predictor} predictor...")

    import numpy as np

    from src.models.predictors import (
        EscapePredictor,
        NeutralizationPredictor,
        ResistancePredictor,
        TropismClassifier,
    )

    predictor_map = {
        "resistance": ResistancePredictor,
        "escape": EscapePredictor,
        "neutralization": NeutralizationPredictor,
        "tropism": TropismClassifier,
    }

    if args.predictor not in predictor_map:
        print(f"Unknown predictor: {args.predictor}")
        print(f"Available: {list(predictor_map.keys())}")
        return 1

    PredictorClass = predictor_map[args.predictor]

    # Create predictor
    predictor = PredictorClass()

    # Generate synthetic data for demonstration
    print("Generating synthetic training data...")
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples) if args.predictor == "tropism" else np.random.randn(n_samples)

    # Split
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Train
    print(f"Training on {len(X_train)} samples...")
    predictor.fit(X_train, y_train)

    # Evaluate
    metrics = predictor.evaluate(X_test, y_test)
    print(f"Test metrics: {metrics}")

    # Save
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        predictor.save(save_dir / f"{args.predictor}_predictor.pkl")
        print(f"Saved to {save_dir / f'{args.predictor}_predictor.pkl'}")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Unified Training Launcher for Ternary VAE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["v5.11", "v5.11.11", "epsilon", "hiv", "swarm", "predictors"],
        help="Training mode",
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="List available training modes",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        dest="batch_size",
        type=int,
        default=512,
        help="Batch size (default: 512)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--save-dir",
        "-o",
        dest="save_dir",
        type=Path,
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick 5-epoch smoke test",
    )

    # Mode-specific arguments
    parser.add_argument(
        "--checkpoints-dir",
        dest="checkpoints_dir",
        type=Path,
        help="Checkpoint directory for epsilon mode",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset name for HIV mode",
    )
    parser.add_argument(
        "--n-agents",
        dest="n_agents",
        type=int,
        default=3,
        help="Number of agents for swarm mode",
    )
    parser.add_argument(
        "--predictor",
        type=str,
        default="resistance",
        choices=["resistance", "escape", "neutralization", "tropism"],
        help="Predictor type for predictors mode",
    )

    args = parser.parse_args()

    # Handle --list-modes
    if args.list_modes:
        list_modes()
        return 0

    # Require mode if not listing
    if not args.mode:
        parser.print_help()
        print("\nError: --mode is required")
        return 1

    # Quick mode overrides epochs
    if args.quick:
        args.epochs = 5
        print("Quick mode: running 5 epochs")

    # Dispatch to appropriate trainer
    mode_handlers = {
        "v5.11": train_v5_11,
        "v5.11.11": train_v5_11_11,
        "epsilon": train_epsilon,
        "hiv": train_hiv,
        "swarm": train_swarm,
        "predictors": train_predictors,
    }

    handler = mode_handlers.get(args.mode)
    if handler:
        return handler(args)
    else:
        print(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
