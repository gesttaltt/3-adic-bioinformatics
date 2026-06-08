#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Master Training Orchestrator - Train All Components.

This script orchestrates training of all components in the correct order,
handling dependencies and parallel execution where possible.

Hardware: RTX 2060 SUPER (8GB VRAM)
Total Duration: ~12-16 hours (full pipeline)

Training Phases:
================
Phase 1: Foundation (Sequential - GPU)
  - V5.5 Base Model (coverage training) - Required by all downstream

Phase 2: Main Model + Preprocessing (Parallel)
  - V5.11.11 Homeostatic (GPU) - Main model training
  - Extract Embeddings (CPU) - Preprocessing for Epsilon-VAE

Phase 3: Downstream Models (Can parallelize some)
  - Epsilon-VAE Hybrid (GPU) - Meta-learning
  - Codon Encoder (GPU) - Codon embeddings
  - Spectral Analysis (CPU) - Can run with GPU training

Usage:
    # Full training pipeline
    python src/scripts/training/train_all.py

    # Start from specific phase
    python src/scripts/training/train_all.py --start-phase 2

    # Skip specific components
    python src/scripts/training/train_all.py --skip epsilon-vae codon-encoder

    # Dry run (show plan only)
    python src/scripts/training/train_all.py --dry-run

    # Quick test mode (5 epochs each)
    python src/scripts/training/train_all.py --quick
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path


class DeviceType(Enum):
    GPU = "gpu"
    CPU = "cpu"
    ANY = "any"


@dataclass
class TrainingComponent:
    """A trainable component in the pipeline."""

    name: str
    script: str
    description: str
    device: DeviceType
    estimated_hours: float
    phase: int
    dependencies: list[str]
    args: list[str] = None
    optional: bool = False

    def __post_init__(self):
        if self.args is None:
            self.args = []


# Define all training components
COMPONENTS = {
    # Phase 1: Foundation
    "v5.5-base": TrainingComponent(
        name="v5.5-base",
        script="src/scripts/training/launch_homeostatic_training.py",
        description="V5.5 Base Model (100% coverage)",
        device=DeviceType.GPU,
        estimated_hours=2.0,
        phase=1,
        dependencies=[],
        args=["--yes"],  # Just runs phase 1 internally
    ),
    # Phase 2: Main Model
    "v5.11.11-homeostatic": TrainingComponent(
        name="v5.11.11-homeostatic",
        script="src/scripts/training/launch_homeostatic_training.py",
        description="V5.11.11 Homeostatic (main model)",
        device=DeviceType.GPU,
        estimated_hours=6.0,
        phase=2,
        dependencies=["v5.5-base"],
        args=["--yes"],
    ),
    # Phase 2 (parallel): Preprocessing
    "extract-embeddings": TrainingComponent(
        name="extract-embeddings",
        script="src/scripts/epsilon_vae/extract_embeddings.py",
        description="Extract embeddings for Epsilon-VAE",
        device=DeviceType.CPU,
        estimated_hours=0.5,
        phase=2,
        dependencies=["v5.5-base"],
        optional=True,
    ),
    # Phase 3: Downstream models
    "epsilon-vae-hybrid": TrainingComponent(
        name="epsilon-vae-hybrid",
        script="src/scripts/epsilon_vae/train_epsilon_vae_hybrid.py",
        description="Epsilon-VAE Hybrid (meta-learning)",
        device=DeviceType.GPU,
        estimated_hours=3.0,
        phase=3,
        dependencies=["v5.11.11-homeostatic", "extract-embeddings"],
        optional=True,
    ),
    "codon-encoder": TrainingComponent(
        name="codon-encoder",
        script="research/bioinformatics/genetic_code/scripts/09_train_codon_encoder_3adic.py",
        description="Codon Encoder (3-adic)",
        device=DeviceType.GPU,
        estimated_hours=1.0,
        phase=3,
        dependencies=["v5.11.11-homeostatic"],
        optional=True,
    ),
    "spectral-analysis": TrainingComponent(
        name="spectral-analysis",
        script="research/bioinformatics/spectral_analysis_over_models/scripts/04_padic_spectral_analysis.py",
        description="P-adic Spectral Analysis",
        device=DeviceType.CPU,
        estimated_hours=1.5,
        phase=3,
        dependencies=["v5.11.11-homeostatic"],
        optional=True,
    ),
}

# Quick test configurations
QUICK_ARGS = {
    "v5.5-base": ["--quick", "--yes"],
    "v5.11.11-homeostatic": ["--quick", "--yes"],
    "epsilon-vae-hybrid": ["--epochs", "10"],
    "codon-encoder": ["--epochs", "5"],
}


def print_banner():
    """Print master training banner."""
    print("\n" + "=" * 70)
    print("  MASTER TRAINING ORCHESTRATOR")
    print("  Hardware: RTX 2060 SUPER (8GB VRAM)")
    print("=" * 70)
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")


def print_plan(components: list[TrainingComponent], quick: bool = False):
    """Print the training plan."""
    print("TRAINING PLAN")
    print("-" * 70)

    total_hours = 0
    phases = {}

    for comp in components:
        if comp.phase not in phases:
            phases[comp.phase] = []
        phases[comp.phase].append(comp)
        hours = comp.estimated_hours * (0.1 if quick else 1.0)
        total_hours += hours

    for phase_num in sorted(phases.keys()):
        print(f"\nPhase {phase_num}:")
        for comp in phases[phase_num]:
            hours = comp.estimated_hours * (0.1 if quick else 1.0)
            device_str = f"[{comp.device.value.upper()}]"
            optional_str = " (optional)" if comp.optional else ""
            print(f"  {device_str:6} {comp.name:25} ~{hours:.1f}h{optional_str}")
            print(f"         {comp.description}")

    print("\n" + "-" * 70)
    print(f"Total estimated time: ~{total_hours:.1f} hours")
    if quick:
        print("  (Quick mode - reduced epochs)")
    print("-" * 70 + "\n")


def check_dependencies(component: TrainingComponent, completed: set[str]) -> bool:
    """Check if all dependencies are satisfied."""
    return all(dep in completed for dep in component.dependencies)


def run_component(
    component: TrainingComponent,
    project_root: Path,
    quick: bool = False,
    dry_run: bool = False,
) -> bool:
    """Run a training component."""
    script_path = project_root / component.script

    if not script_path.exists():
        print(f"  [WARN] Script not found: {script_path}")
        return False

    # Build command
    cmd = [sys.executable, str(script_path)]

    if quick and component.name in QUICK_ARGS:
        cmd.extend(QUICK_ARGS[component.name])
    else:
        cmd.extend(component.args)

    print(f"\n{'=' * 70}")
    print(f"  RUNNING: {component.name}")
    print(f"  Script: {component.script}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Device: {component.device.value.upper()}")
    print(f"  Estimated: ~{component.estimated_hours:.1f}h")
    print(f"{'=' * 70}\n")

    if dry_run:
        print("  [DRY RUN] Would execute above command")
        return True

    start_time = time.time()

    try:
        subprocess.run(
            cmd,
            cwd=str(project_root),
            check=True,
        )
        elapsed = time.time() - start_time
        print(f"\n  [OK] {component.name} completed in {elapsed / 3600:.2f}h")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n  [ERROR] {component.name} failed after {elapsed / 3600:.2f}h")
        print(f"  Exit code: {e.returncode}")
        return False

    except KeyboardInterrupt:
        print(f"\n  [INTERRUPTED] {component.name} was interrupted")
        raise


def run_parallel_cpu(
    components: list[TrainingComponent],
    project_root: Path,
    quick: bool = False,
    dry_run: bool = False,
) -> dict[str, bool]:
    """Run CPU components in parallel (as background processes)."""
    import threading

    results = {}
    threads = []

    def run_in_thread(comp):
        results[comp.name] = run_component(comp, project_root, quick, dry_run)

    for comp in components:
        if comp.device == DeviceType.CPU:
            t = threading.Thread(target=run_in_thread, args=(comp,))
            threads.append(t)
            t.start()

    for t in threads:
        t.join()

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Master Training Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/scripts/training/train_all.py              # Full training
    python src/scripts/training/train_all.py --quick      # Quick test (5 epochs each)
    python src/scripts/training/train_all.py --dry-run    # Show plan only
    python src/scripts/training/train_all.py --start-phase 2  # Skip phase 1
    python src/scripts/training/train_all.py --only v5.11.11-homeostatic  # Single component
    python src/scripts/training/train_all.py --skip epsilon-vae-hybrid codon-encoder
        """,
    )
    parser.add_argument("--quick", action="store_true", help="Quick test mode (reduced epochs)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    parser.add_argument("--start-phase", type=int, default=1, help="Start from specific phase")
    parser.add_argument("--only", nargs="+", help="Run only specific components")
    parser.add_argument("--skip", nargs="+", default=[], help="Skip specific components")
    parser.add_argument("--include-optional", action="store_true", help="Include optional components")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")
    args = parser.parse_args()

    print_banner()

    project_root = Path(__file__).resolve().parents[2]

    # Filter components
    if args.only:
        components = [COMPONENTS[name] for name in args.only if name in COMPONENTS]
    else:
        components = [
            comp
            for name, comp in COMPONENTS.items()
            if name not in args.skip
            and comp.phase >= args.start_phase
            and (args.include_optional or not comp.optional or comp.phase <= 2)
        ]

    if not components:
        print("No components to train!")
        return 1

    # Print plan
    print_plan(components, args.quick)

    if args.dry_run:
        print("[DRY RUN] No training will be executed.")
        return 0

    # Confirm
    if not args.yes:
        try:
            input("Press Enter to start training (Ctrl+C to cancel)...")
        except (EOFError, KeyboardInterrupt):
            print("\nCancelled.")
            return 0

    # Execute training phases
    completed = set()
    failed = set()

    # Check which dependencies are already satisfied (checkpoints exist)
    v5_5_path = project_root / "checkpoints" / "v5_5" / "latest.pt"
    if v5_5_path.exists():
        print("[OK] V5.5 checkpoint exists, skipping Phase 1")
        completed.add("v5.5-base")

    v5_11_path = project_root / "checkpoints" / "v5_11_11_homeostatic_rtx2060s" / "latest.pt"
    if v5_11_path.exists():
        print("[OK] V5.11.11 checkpoint exists")
        completed.add("v5.11.11-homeostatic")

    # Group by phase
    phases = {}
    for comp in components:
        if comp.phase not in phases:
            phases[comp.phase] = []
        phases[comp.phase].append(comp)

    # Execute each phase
    for phase_num in sorted(phases.keys()):
        print(f"\n{'#' * 70}")
        print(f"  PHASE {phase_num}")
        print(f"{'#' * 70}")

        phase_components = phases[phase_num]

        # Separate GPU and CPU components
        gpu_comps = [c for c in phase_components if c.device == DeviceType.GPU and c.name not in completed]
        cpu_comps = [c for c in phase_components if c.device == DeviceType.CPU and c.name not in completed]

        # Run GPU components sequentially
        for comp in gpu_comps:
            if not check_dependencies(comp, completed):
                print(f"  [SKIP] {comp.name}: dependencies not met")
                continue

            success = run_component(comp, project_root, args.quick, args.dry_run)
            if success:
                completed.add(comp.name)
            else:
                failed.add(comp.name)
                # For required components, stop on failure
                if not comp.optional:
                    print(f"\n[ERROR] Required component {comp.name} failed. Stopping.")
                    break

        # Run CPU components (can be parallel)
        for comp in cpu_comps:
            if not check_dependencies(comp, completed):
                print(f"  [SKIP] {comp.name}: dependencies not met")
                continue

            success = run_component(comp, project_root, args.quick, args.dry_run)
            if success:
                completed.add(comp.name)
            else:
                failed.add(comp.name)

    # Summary
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Completed: {len(completed)} components")
    print(f"  Failed: {len(failed)} components")

    if completed:
        print("\n  Completed:")
        for name in sorted(completed):
            print(f"    [OK] {name}")

    if failed:
        print("\n  Failed:")
        for name in sorted(failed):
            print(f"    [FAIL] {name}")

    print("=" * 70 + "\n")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
