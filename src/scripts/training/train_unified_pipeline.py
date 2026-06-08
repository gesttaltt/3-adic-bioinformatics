#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unified Research Pipeline - Complete Multi-Disease Training.

Orchestrates the complete training pipeline for all disease domains:

1. Pretraining Phase
   - BYOL contrastive learning for sequence representations
   - V5.5 base model for coverage

2. Core Model Phase
   - V5.11.11 Homeostatic VAE with P-adic geometry
   - Codon Encoder for genetic embeddings

3. Downstream Models Phase
   - Diffusion models for sequence generation
   - Multi-task disease predictors
   - Meta-learning for variant adaptation

4. Analysis Phase
   - Spectral analysis of learned representations
   - Cross-disease transfer evaluation
   - Epsilon-VAE for meta-learning

Hardware: RTX 2060 SUPER (8GB VRAM)
Total Duration: 12-24 hours (full pipeline)

Usage:
    # Full pipeline
    python src/scripts/training/train_unified_pipeline.py

    # Specific phases
    python src/scripts/training/train_unified_pipeline.py --phase pretrain
    python src/scripts/training/train_unified_pipeline.py --phase core
    python src/scripts/training/train_unified_pipeline.py --phase downstream

    # Quick test
    python src/scripts/training/train_unified_pipeline.py --quick
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Phase(Enum):
    PRETRAIN = "pretrain"
    CORE = "core"
    DOWNSTREAM = "downstream"
    ANALYSIS = "analysis"
    ALL = "all"


class DeviceType(Enum):
    GPU = "gpu"
    CPU = "cpu"


@dataclass
class TrainingStep:
    """A single training step in the pipeline."""

    name: str
    script: str
    description: str
    phase: Phase
    device: DeviceType
    estimated_hours: float
    dependencies: list[str] = field(default_factory=list)
    args: list[str] = field(default_factory=list)
    quick_args: list[str] = field(default_factory=list)
    optional: bool = False
    diseases: list[str] = field(default_factory=lambda: ["all"])


# Complete pipeline definition
PIPELINE = [
    # Phase 1: Pretraining
    TrainingStep(
        name="byol-pretrain",
        script="src/scripts/training/train_contrastive_pretrain.py",
        description="BYOL contrastive pretraining on multi-disease sequences",
        phase=Phase.PRETRAIN,
        device=DeviceType.GPU,
        estimated_hours=2.0,
        args=["--method", "byol", "-y"],
        quick_args=["--method", "byol", "--quick", "-y"],
        diseases=["hiv", "cancer", "ra", "neuro"],
    ),
    TrainingStep(
        name="v5.5-base",
        script="src/scripts/training/launch_homeostatic_training.py",
        description="V5.5 Base Model (100% coverage foundation)",
        phase=Phase.PRETRAIN,
        device=DeviceType.GPU,
        estimated_hours=2.0,
        args=["--yes"],
        quick_args=["--quick", "--yes"],
    ),
    # Phase 2: Core Models
    TrainingStep(
        name="v5.11.11-homeostatic",
        script="src/scripts/training/launch_homeostatic_training.py",
        description="V5.11.11 Homeostatic VAE with P-adic geometry",
        phase=Phase.CORE,
        device=DeviceType.GPU,
        estimated_hours=6.0,
        dependencies=["v5.5-base"],
        args=["--yes"],
        quick_args=["--quick", "--yes"],
    ),
    TrainingStep(
        name="codon-encoder",
        script="research/bioinformatics/genetic_code/scripts/09_train_codon_encoder_3adic.py",
        description="Codon Encoder with 3-adic hierarchy",
        phase=Phase.CORE,
        device=DeviceType.GPU,
        estimated_hours=1.0,
        dependencies=["v5.5-base"],
        args=[],
        quick_args=["--epochs", "10"],
    ),
    # Phase 3: Downstream Models
    TrainingStep(
        name="diffusion-hiv",
        script="src/scripts/training/train_diffusion_codon.py",
        description="Diffusion model for HIV codon generation",
        phase=Phase.DOWNSTREAM,
        device=DeviceType.GPU,
        estimated_hours=3.0,
        dependencies=["v5.11.11-homeostatic"],
        args=["--disease", "hiv", "-y"],
        quick_args=["--disease", "hiv", "--quick", "-y"],
        diseases=["hiv"],
    ),
    TrainingStep(
        name="diffusion-cancer",
        script="src/scripts/training/train_diffusion_codon.py",
        description="Diffusion model for cancer sequence generation",
        phase=Phase.DOWNSTREAM,
        device=DeviceType.GPU,
        estimated_hours=3.0,
        dependencies=["v5.11.11-homeostatic"],
        args=["--disease", "cancer", "-y"],
        quick_args=["--disease", "cancer", "--quick", "-y"],
        diseases=["cancer"],
        optional=True,
    ),
    TrainingStep(
        name="multitask-predictor",
        script="src/scripts/training/train_multitask_disease.py",
        description="Multi-task disease predictor with GradNorm",
        phase=Phase.DOWNSTREAM,
        device=DeviceType.GPU,
        estimated_hours=4.0,
        dependencies=["v5.11.11-homeostatic"],
        args=["-y"],
        quick_args=["--quick", "-y"],
        diseases=["hiv", "cancer", "ra", "neuro"],
    ),
    TrainingStep(
        name="meta-learning-hiv",
        script="src/scripts/training/train_meta_learning.py",
        description="Meta-learning for HIV variant adaptation",
        phase=Phase.DOWNSTREAM,
        device=DeviceType.GPU,
        estimated_hours=2.0,
        dependencies=["v5.11.11-homeostatic"],
        args=["--disease", "hiv", "-y"],
        quick_args=["--disease", "hiv", "--quick", "-y"],
        diseases=["hiv"],
    ),
    # Phase 4: Analysis
    TrainingStep(
        name="epsilon-vae",
        script="src/scripts/epsilon_vae/train_epsilon_vae_hybrid.py",
        description="Epsilon-VAE for meta-learning on latent space",
        phase=Phase.ANALYSIS,
        device=DeviceType.GPU,
        estimated_hours=3.0,
        dependencies=["v5.11.11-homeostatic"],
        args=[],
        quick_args=["--epochs", "20"],
        optional=True,
    ),
    TrainingStep(
        name="spectral-analysis",
        script="research/bioinformatics/spectral_analysis_over_models/scripts/04_padic_spectral_analysis.py",
        description="P-adic spectral analysis of learned representations",
        phase=Phase.ANALYSIS,
        device=DeviceType.CPU,
        estimated_hours=1.5,
        dependencies=["v5.11.11-homeostatic"],
        optional=True,
    ),
]


def print_banner():
    """Print pipeline banner."""
    print("\n" + "=" * 80)
    print("  UNIFIED RESEARCH PIPELINE - MULTI-DISEASE TRAINING")
    print("  Hardware: RTX 2060 SUPER (8GB VRAM)")
    print("=" * 80)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")


def print_plan(steps: list[TrainingStep], quick: bool = False):
    """Print the execution plan."""
    print("EXECUTION PLAN")
    print("-" * 80)

    total_hours = 0
    phases = {}

    for step in steps:
        if step.phase not in phases:
            phases[step.phase] = []
        phases[step.phase].append(step)
        hours = step.estimated_hours * (0.1 if quick else 1.0)
        total_hours += hours

    for phase in [Phase.PRETRAIN, Phase.CORE, Phase.DOWNSTREAM, Phase.ANALYSIS]:
        if phase in phases:
            print(f"\n{phase.value.upper()} PHASE:")
            for step in phases[phase]:
                hours = step.estimated_hours * (0.1 if quick else 1.0)
                device_str = f"[{step.device.value.upper()}]"
                optional_str = " (optional)" if step.optional else ""
                diseases_str = f" [{', '.join(step.diseases)}]" if step.diseases != ["all"] else ""
                print(f"  {device_str:6} {step.name:25} ~{hours:.1f}h{optional_str}{diseases_str}")
                print(f"         {step.description}")
                if step.dependencies:
                    print(f"         Depends on: {', '.join(step.dependencies)}")

    print("\n" + "-" * 80)
    print(f"Total estimated time: ~{total_hours:.1f} hours")
    if quick:
        print("  (Quick mode - reduced epochs)")
    print("-" * 80 + "\n")


def check_dependencies(step: TrainingStep, completed: set[str]) -> bool:
    """Check if all dependencies are satisfied."""
    return all(dep in completed for dep in step.dependencies)


def run_step(
    step: TrainingStep,
    quick: bool = False,
    dry_run: bool = False,
) -> tuple[bool, float]:
    """Run a single training step."""
    script_path = PROJECT_ROOT / step.script

    if not script_path.exists():
        print(f"  [WARN] Script not found: {script_path}")
        return False, 0.0

    # Build command
    cmd = [sys.executable, str(script_path)]
    if quick and step.quick_args:
        cmd.extend(step.quick_args)
    else:
        cmd.extend(step.args)

    print(f"\n{'#' * 80}")
    print(f"  STEP: {step.name}")
    print(f"  Phase: {step.phase.value}")
    print(f"  Script: {step.script}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Device: {step.device.value.upper()}")
    print(f"  Estimated: ~{step.estimated_hours:.1f}h")
    print(f"{'#' * 80}\n")

    if dry_run:
        print("  [DRY RUN] Would execute above command")
        return True, 0.0

    start_time = time.time()

    try:
        subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            check=True,
        )
        elapsed = time.time() - start_time
        print(f"\n  [OK] {step.name} completed in {elapsed / 3600:.2f}h")
        return True, elapsed

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n  [ERROR] {step.name} failed after {elapsed / 3600:.2f}h")
        print(f"  Exit code: {e.returncode}")
        return False, elapsed

    except KeyboardInterrupt:
        print(f"\n  [INTERRUPTED] {step.name} was interrupted")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Unified Research Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["pretrain", "core", "downstream", "analysis", "all"],
        help="Phase to run (default: all)",
    )
    parser.add_argument(
        "--diseases",
        nargs="+",
        default=["hiv", "cancer", "ra", "neuro"],
        help="Diseases to train on",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="Steps to skip",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Only run specific steps",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Include optional steps",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (reduced epochs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without executing",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts",
    )
    args = parser.parse_args()

    print_banner()

    # Filter steps
    target_phase = Phase(args.phase) if args.phase != "all" else None

    if args.only:
        steps = [s for s in PIPELINE if s.name in args.only]
    else:
        steps = []
        for step in PIPELINE:
            # Phase filter
            if target_phase and step.phase != target_phase:
                continue

            # Skip filter
            if step.name in args.skip:
                continue

            # Optional filter
            if step.optional and not args.include_optional:
                continue

            # Disease filter
            if step.diseases != ["all"] and not any(d in args.diseases for d in step.diseases):
                continue

            steps.append(step)

    if not steps:
        print("No steps to execute!")
        return 1

    # Print plan
    print_plan(steps, args.quick)

    if args.dry_run:
        print("[DRY RUN] No training will be executed.")
        return 0

    # Confirm
    if not args.yes:
        try:
            input("Press Enter to start pipeline (Ctrl+C to cancel)...")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 0

    # Execute
    completed = set()
    failed = set()
    timings = {}

    # Check for existing checkpoints
    checkpoint_checks = {
        "v5.5-base": PROJECT_ROOT / "checkpoints/v5_5/latest.pt",
        "v5.11.11-homeostatic": PROJECT_ROOT / "checkpoints/v5_11_11_homeostatic_rtx2060s/latest.pt",
        "codon-encoder": PROJECT_ROOT / "checkpoints/codon_encoder_3adic/latest.pt",
    }

    for step_name, checkpoint_path in checkpoint_checks.items():
        if checkpoint_path.exists():
            print(f"[OK] {step_name} checkpoint exists, marking as completed")
            completed.add(step_name)

    # Run steps
    for step in steps:
        if step.name in completed:
            print(f"\n[SKIP] {step.name}: already completed")
            continue

        if not check_dependencies(step, completed):
            missing = [d for d in step.dependencies if d not in completed]
            print(f"\n[SKIP] {step.name}: missing dependencies ({', '.join(missing)})")
            if step.optional:
                continue
            else:
                failed.add(step.name)
                print(f"[ERROR] Required step {step.name} cannot run. Stopping.")
                break

        success, elapsed = run_step(step, args.quick, args.dry_run)
        timings[step.name] = elapsed

        if success:
            completed.add(step.name)
        else:
            failed.add(step.name)
            if not step.optional:
                print(f"\n[ERROR] Required step {step.name} failed. Stopping pipeline.")
                break

    # Summary
    print("\n" + "=" * 80)
    print("  PIPELINE COMPLETE")
    print("=" * 80)

    total_time = sum(timings.values())
    print(f"\n  Total time: {total_time / 3600:.2f} hours")
    print(f"  Completed: {len(completed)} steps")
    print(f"  Failed: {len(failed)} steps")

    if completed:
        print("\n  Completed steps:")
        for name in sorted(completed):
            time_str = f" ({timings.get(name, 0) / 3600:.2f}h)" if name in timings else ""
            print(f"    [OK] {name}{time_str}")

    if failed:
        print("\n  Failed steps:")
        for name in sorted(failed):
            print(f"    [FAIL] {name}")

    # List available checkpoints
    print("\n  Checkpoints saved:")
    checkpoint_dirs = [
        "checkpoints/byol_pretrain",
        "checkpoints/v5_5",
        "checkpoints/v5_11_11_homeostatic_rtx2060s",
        "checkpoints/codon_encoder_3adic",
        "checkpoints/diffusion_hiv",
        "checkpoints/diffusion_cancer",
        "checkpoints/multitask_disease",
        "checkpoints/meta_learning_hiv",
        "checkpoints/epsilon_vae_hybrid",
    ]

    for dir_path in checkpoint_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists() and any(full_path.iterdir()):
            print(f"    {dir_path}")

    print("\n" + "=" * 80)

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
