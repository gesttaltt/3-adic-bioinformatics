#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Master Training Script - Train All Models with SOTA Components.

This script runs all training pipelines sequentially to create a complete
set of trained models for research applications.

Models Trained:
1. Contrastive Pretrained Encoder (BYOL with CodonPositiveSampler)
2. Diffusion Model (D3PM with CodonUsageLoss)
3. Multitask Disease Predictor (with VariantEscapeHead)
4. Meta-Learning Model (with MetaLearningEscapeHead)

Estimated Total Time: 4-8 hours (RTX 2060 SUPER)

Usage:
    # Train all models with default settings
    python src/scripts/training/train_all_models.py

    # Quick test mode (reduced epochs)
    python src/scripts/training/train_all_models.py --quick

    # Train specific models
    python src/scripts/training/train_all_models.py --models contrastive diffusion

    # Full training with all enhancements
    python src/scripts/training/train_all_models.py --full
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def run_training(name: str, command: list[str], description: str) -> bool:
    """Run a training script and return success status."""
    print("\n" + "=" * 70)
    print(f"  TRAINING: {name}")
    print(f"  {description}")
    print("=" * 70)
    print(f"Command: {' '.join(command)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 70)

    try:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
        )
        print(f"\n[SUCCESS] {name} completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {name} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n[ERROR] {name} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train All Models")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode (10 epochs per model)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full training with all enhancements (longer)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["contrastive", "diffusion", "multitask", "meta"],
        default=["contrastive", "diffusion", "multitask", "meta"],
        help="Which models to train",
    )
    parser.add_argument(
        "--diseases",
        nargs="+",
        default=["hiv", "cancer", "ra", "neuro"],
        help="Diseases to train on",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  MASTER TRAINING SCRIPT - ALL SOTA MODELS")
    print("=" * 70)
    print(f"  Mode: {'Quick (10 epochs)' if args.quick else 'Full' if args.full else 'Standard'}")
    print(f"  Models: {', '.join(args.models)}")
    print(f"  Diseases: {', '.join(args.diseases)}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}
    python_exe = sys.executable

    # 1. Contrastive Pretraining
    if "contrastive" in args.models:
        cmd = [
            python_exe,
            "src/scripts/training/train_contrastive_pretrain.py",
            "--method",
            "byol",
            "--use-codon-sampler",
            "--evaluate",
            "-y",
        ]
        if args.quick:
            cmd.append("--quick")
        else:
            cmd.extend(["--epochs", "50" if not args.full else "100"])

        results["contrastive"] = run_training(
            "Contrastive Pretraining (BYOL + CodonPositiveSampler)",
            cmd,
            "Self-supervised learning with biology-aware positive sampling",
        )

    # 2. Diffusion Model
    if "diffusion" in args.models:
        cmd = [
            python_exe,
            "src/scripts/training/train_diffusion_codon.py",
            "--disease",
            "hiv",
            "--use-codon-loss",
            "--evaluate",
            "-y",
        ]
        if args.quick:
            cmd.append("--quick")
        else:
            cmd.extend(["--epochs", "50" if not args.full else "100"])

        results["diffusion"] = run_training(
            "Diffusion Model (D3PM + CodonUsageLoss)",
            cmd,
            "Discrete diffusion with biological constraints",
        )

    # 3. Multitask Disease Predictor
    if "multitask" in args.models:
        cmd = [
            python_exe,
            "src/scripts/training/train_multitask_disease.py",
            "--diseases",
            *args.diseases,
            "--use-escape-head",
            "--evaluate",
            "-y",
        ]
        if args.quick:
            cmd.append("--quick")
        else:
            cmd.extend(["--epochs", "50" if not args.full else "100"])

        results["multitask"] = run_training(
            "Multitask Disease Predictor (+ VariantEscapeHead)",
            cmd,
            "GradNorm multi-task learning with EVEscape-inspired escape prediction",
        )

    # 4. Meta-Learning
    if "meta" in args.models:
        cmd = [
            python_exe,
            "src/scripts/training/train_meta_learning.py",
            "--disease",
            "hiv",
            "--algorithm",
            "reptile",
            "--use-escape-head",
            "--evaluate",
            "-y",
        ]
        if args.quick:
            cmd.extend(["--n-epochs", "100"])
        else:
            cmd.extend(["--n-epochs", "500" if not args.full else "1000"])

        results["meta"] = run_training(
            "Meta-Learning (Reptile + MetaLearningEscapeHead)",
            cmd,
            "Few-shot learning for rapid variant adaptation",
        )

    # Summary
    print("\n" + "=" * 70)
    print("  TRAINING SUMMARY")
    print("=" * 70)

    for model, success in results.items():
        status = "[SUCCESS]" if success else "[FAILED]"
        print(f"  {status} {model}")

    # List checkpoints
    print("\n  Checkpoints saved to:")
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    if checkpoint_dir.exists():
        for d in sorted(checkpoint_dir.iterdir()):
            if d.is_dir():
                files = list(d.glob("*.pt"))
                print(f"    {d.name}/: {len(files)} file(s)")

    print("\n" + "=" * 70)
    print(f"  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Return success if all trained
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
