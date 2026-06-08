#!/usr/bin/env python3
"""Comprehensive Experiment Runner for HIV Drug Resistance Prediction.

This script orchestrates all experiments to address the performance gap
between drug classes:
- PI drugs: +0.922 correlation (excellent)
- NRTI drugs: +0.07 correlation (poor - TAM complexity)
- NNRTI drugs: +0.19 correlation (poor - RT length)
- INI drugs: +0.14 correlation (poor - low data)

Experiment Categories:
1. Enhanced Training: TAM-aware encoding for NRTI
2. MAML Few-Shot: Rapid adaptation for low-data drugs
3. Multi-Task Learning: Joint training within drug classes
4. External Validation: Stanford HIVdb, temporal, cross-cohort
5. Transformer Experiments: Stable transformers for long sequences

Copyright 2024-2025 AI Whisperers
Licensed under PolyForm Noncommercial License 1.0.0
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("experiment_runner.log"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for comprehensive experiments."""

    # Output settings
    results_dir: str = "results/comprehensive_experiments"
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Drug groups
    pi_drugs: list[str] = field(default_factory=lambda: ["ATV", "DRV", "FPV", "IDV", "LPV", "NFV", "SQV", "TPV"])
    nrti_drugs: list[str] = field(default_factory=lambda: ["3TC", "ABC", "AZT", "D4T", "DDI", "FTC", "TDF"])
    nnrti_drugs: list[str] = field(default_factory=lambda: ["EFV", "ETR", "NVP", "RPV"])
    ini_drugs: list[str] = field(default_factory=lambda: ["BIC", "CAB", "DTG", "EVG", "RAL"])

    # Experiment flags
    run_enhanced_training: bool = True
    run_maml_evaluation: bool = True
    run_multitask_training: bool = True
    run_external_validation: bool = True
    run_transformer_experiments: bool = True
    run_ablation_studies: bool = False  # Optional deep dive

    # Training settings
    epochs: int = 100
    batch_size: int = 64
    device: str = "cuda"

    # Parallelization
    max_parallel_jobs: int = 1  # GPU memory constraints


class ExperimentRunner:
    """Orchestrates comprehensive experiments."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: dict[str, Any] = {
            "timestamp": config.timestamp,
            "config": config.__dict__,
            "experiments": {},
        }

        # Create results directory
        self.results_path = Path(config.results_dir) / config.timestamp
        self.results_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Results will be saved to: {self.results_path}")

    def run_script(
        self,
        script_path: str,
        args: list[str] = None,
        timeout: int = 7200,  # 2 hours default
    ) -> dict[str, Any]:
        """Run a Python script and capture output.

        Args:
            script_path: Path to script
            args: Command line arguments
            timeout: Timeout in seconds

        Returns:
            Dict with success status, output, and timing
        """
        args = args or []
        cmd = [sys.executable, script_path] + args

        logger.info(f"Running: {' '.join(cmd)}")
        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(project_root),
            )

            elapsed = time.time() - start_time

            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "elapsed_seconds": elapsed,
            }

        except subprocess.TimeoutExpired:
            elapsed = time.time() - start_time
            logger.warning(f"Script timed out after {timeout}s: {script_path}")
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Timeout after {timeout}s",
                "elapsed_seconds": elapsed,
            }
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Script failed: {e}")
            return {
                "success": False,
                "returncode": -1,
                "stdout": "",
                "stderr": str(e),
                "elapsed_seconds": elapsed,
            }

    def run_enhanced_training(self) -> dict[str, Any]:
        """Run enhanced training experiments with TAM integration."""
        logger.info("\n" + "=" * 60)
        logger.info("ENHANCED TRAINING (TAM Integration)")
        logger.info("=" * 60)

        results = {}

        # NRTI drugs - need TAM-aware training
        for drug in self.config.nrti_drugs[:3]:  # Start with subset
            logger.info(f"Training enhanced model for {drug}")

            result = self.run_script(
                "src/scripts/experiments/run_enhanced_training.py",
                args=[
                    "--drug",
                    drug,
                    "--drug-class",
                    "NRTI",
                    "--epochs",
                    str(self.config.epochs),
                    "--device",
                    self.config.device,
                ],
            )

            results[drug] = result

            # Save intermediate results
            self._save_checkpoint("enhanced_training", results)

        return results

    def run_maml_evaluation(self) -> dict[str, Any]:
        """Run MAML few-shot evaluation."""
        logger.info("\n" + "=" * 60)
        logger.info("MAML FEW-SHOT EVALUATION")
        logger.info("=" * 60)

        results = {}

        # Test on held-out PI drugs
        held_out_pi = ["TPV", "DRV"]
        for drug in held_out_pi:
            logger.info(f"MAML evaluation for {drug}")

            result = self.run_script(
                "src/scripts/experiments/run_maml_evaluation.py",
                args=[
                    "--eval-drug",
                    drug,
                    "--drug-class",
                    "PI",
                    "--device",
                    self.config.device,
                ],
            )
            results[f"PI_{drug}"] = result

        # Test on INI drugs (low data scenario)
        for drug in self.config.ini_drugs[:2]:
            logger.info(f"MAML evaluation for INI {drug}")

            result = self.run_script(
                "src/scripts/experiments/run_maml_evaluation.py",
                args=[
                    "--eval-drug",
                    drug,
                    "--drug-class",
                    "INI",
                    "--device",
                    self.config.device,
                ],
            )
            results[f"INI_{drug}"] = result

        self._save_checkpoint("maml_evaluation", results)
        return results

    def run_multitask_training(self) -> dict[str, Any]:
        """Run multi-task training experiments."""
        logger.info("\n" + "=" * 60)
        logger.info("MULTI-TASK TRAINING (GradNorm)")
        logger.info("=" * 60)

        results = {}

        # Multi-task on PI drugs (proven to work well)
        logger.info("Multi-task training on PI drug class")
        result = self.run_script(
            "src/scripts/experiments/run_multitask_training.py",
            args=[
                "--drug-class",
                "PI",
                "--epochs",
                str(self.config.epochs),
                "--device",
                self.config.device,
            ],
        )
        results["PI_multitask"] = result

        # Multi-task on NRTI drugs
        logger.info("Multi-task training on NRTI drug class")
        result = self.run_script(
            "src/scripts/experiments/run_multitask_training.py",
            args=[
                "--drug-class",
                "NRTI",
                "--epochs",
                str(self.config.epochs),
                "--device",
                self.config.device,
            ],
        )
        results["NRTI_multitask"] = result

        self._save_checkpoint("multitask_training", results)
        return results

    def run_external_validation(self) -> dict[str, Any]:
        """Run external validation experiments."""
        logger.info("\n" + "=" * 60)
        logger.info("EXTERNAL VALIDATION")
        logger.info("=" * 60)

        results = {}

        # Representative drugs from each class
        test_drugs = ["DRV", "AZT", "EFV", "DTG"]  # PI, NRTI, NNRTI, INI

        result = self.run_script(
            "src/scripts/experiments/run_external_validation.py",
            args=[
                "--drugs",
                *test_drugs,
                "--results-dir",
                str(self.results_path / "external_validation"),
                "--device",
                self.config.device,
            ],
        )
        results["external_validation"] = result

        self._save_checkpoint("external_validation", results)
        return results

    def run_transformer_experiments(self) -> dict[str, Any]:
        """Run transformer experiments for long sequences."""
        logger.info("\n" + "=" * 60)
        logger.info("STABLE TRANSFORMER EXPERIMENTS")
        logger.info("=" * 60)

        results = {}

        # Test on RT drugs (long sequences)
        rt_drugs = self.config.nrti_drugs[:2] + self.config.nnrti_drugs[:2]

        for drug in rt_drugs:
            drug_class = "NRTI" if drug in self.config.nrti_drugs else "NNRTI"
            logger.info(f"Transformer experiment for {drug} ({drug_class})")

            # Use the enhanced training script with transformer flag
            result = self.run_script(
                "src/scripts/experiments/run_enhanced_training.py",
                args=[
                    "--drug",
                    drug,
                    "--drug-class",
                    drug_class,
                    "--use-transformer",
                    "--epochs",
                    str(min(self.config.epochs, 50)),  # Shorter for transformer
                    "--device",
                    self.config.device,
                ],
            )
            results[f"{drug_class}_{drug}_transformer"] = result

        self._save_checkpoint("transformer_experiments", results)
        return results

    def run_ablation_studies(self) -> dict[str, Any]:
        """Run ablation studies on key components."""
        logger.info("\n" + "=" * 60)
        logger.info("ABLATION STUDIES")
        logger.info("=" * 60)

        results = {}

        # Test different components
        ablation_configs = [
            {"name": "no_tam", "args": ["--no-tam"]},
            {"name": "no_attention", "args": ["--no-attention"]},
            {"name": "baseline", "args": ["--baseline"]},
        ]

        test_drug = "AZT"  # Representative NRTI

        for ablation in ablation_configs:
            logger.info(f"Ablation: {ablation['name']}")

            result = self.run_script(
                "src/scripts/experiments/run_enhanced_training.py",
                args=[
                    "--drug",
                    test_drug,
                    "--drug-class",
                    "NRTI",
                    "--epochs",
                    str(min(self.config.epochs, 30)),
                    "--device",
                    self.config.device,
                ]
                + ablation["args"],
            )
            results[ablation["name"]] = result

        self._save_checkpoint("ablation_studies", results)
        return results

    def _save_checkpoint(self, experiment_name: str, results: dict[str, Any]) -> None:
        """Save intermediate results."""
        self.results["experiments"][experiment_name] = results

        checkpoint_path = self.results_path / f"{experiment_name}_checkpoint.json"
        with open(checkpoint_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def run_all(self) -> dict[str, Any]:
        """Run all configured experiments."""
        start_time = time.time()

        logger.info("\n" + "=" * 80)
        logger.info("COMPREHENSIVE EXPERIMENT SUITE")
        logger.info(f"Started: {datetime.now().isoformat()}")
        logger.info("=" * 80)

        if self.config.run_enhanced_training:
            self.results["experiments"]["enhanced_training"] = self.run_enhanced_training()

        if self.config.run_maml_evaluation:
            self.results["experiments"]["maml_evaluation"] = self.run_maml_evaluation()

        if self.config.run_multitask_training:
            self.results["experiments"]["multitask_training"] = self.run_multitask_training()

        if self.config.run_external_validation:
            self.results["experiments"]["external_validation"] = self.run_external_validation()

        if self.config.run_transformer_experiments:
            self.results["experiments"]["transformer"] = self.run_transformer_experiments()

        if self.config.run_ablation_studies:
            self.results["experiments"]["ablation"] = self.run_ablation_studies()

        # Calculate total time
        total_time = time.time() - start_time
        self.results["total_time_seconds"] = total_time
        self.results["completed"] = datetime.now().isoformat()

        # Save final results
        final_path = self.results_path / "final_results.json"
        with open(final_path, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENTS COMPLETE")
        logger.info(f"Total time: {total_time / 60:.1f} minutes")
        logger.info(f"Results saved to: {self.results_path}")
        logger.info("=" * 80)

        return self.results

    def generate_summary_report(self) -> str:
        """Generate a human-readable summary report."""
        lines = [
            "# Comprehensive Experiment Summary",
            f"\nGenerated: {datetime.now().isoformat()}",
            f"Results directory: {self.results_path}",
            "",
            "## Experiment Status",
            "",
        ]

        for exp_name, exp_results in self.results.get("experiments", {}).items():
            lines.append(f"\n### {exp_name.replace('_', ' ').title()}")

            if isinstance(exp_results, dict):
                successes = sum(1 for r in exp_results.values() if isinstance(r, dict) and r.get("success", False))
                total = len(exp_results)
                lines.append(f"- Success rate: {successes}/{total}")

                for name, result in exp_results.items():
                    if isinstance(result, dict):
                        status = "✓" if result.get("success") else "✗"
                        elapsed = result.get("elapsed_seconds", 0)
                        lines.append(f"  - {name}: {status} ({elapsed:.1f}s)")

        lines.append("\n## Performance Summary")
        lines.append("\n(Detailed metrics available in individual result files)")

        total_time = self.results.get("total_time_seconds", 0)
        lines.append(f"\n## Total Runtime: {total_time / 60:.1f} minutes")

        report = "\n".join(lines)

        # Save report
        report_path = self.results_path / "summary_report.md"
        with open(report_path, "w") as f:
            f.write(report)

        return report


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive HIV drug resistance experiments")

    # Experiment selection
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--enhanced", action="store_true", help="Run enhanced training")
    parser.add_argument("--maml", action="store_true", help="Run MAML evaluation")
    parser.add_argument("--multitask", action="store_true", help="Run multi-task training")
    parser.add_argument("--validation", action="store_true", help="Run external validation")
    parser.add_argument("--transformer", action="store_true", help="Run transformer experiments")
    parser.add_argument("--ablation", action="store_true", help="Run ablation studies")

    # Settings
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--results-dir", type=str, default="results/comprehensive_experiments")

    args = parser.parse_args()

    # Configure experiments
    config = ExperimentConfig(
        epochs=args.epochs,
        device=args.device,
        results_dir=args.results_dir,
    )

    # Set experiment flags
    if args.all:
        config.run_enhanced_training = True
        config.run_maml_evaluation = True
        config.run_multitask_training = True
        config.run_external_validation = True
        config.run_transformer_experiments = True
        config.run_ablation_studies = True
    else:
        config.run_enhanced_training = args.enhanced
        config.run_maml_evaluation = args.maml
        config.run_multitask_training = args.multitask
        config.run_external_validation = args.validation
        config.run_transformer_experiments = args.transformer
        config.run_ablation_studies = args.ablation

    # If no specific experiments selected, run core set
    if not any([args.all, args.enhanced, args.maml, args.multitask, args.validation, args.transformer, args.ablation]):
        logger.info("No experiments specified, running core experiments")
        config.run_enhanced_training = True
        config.run_maml_evaluation = True
        config.run_multitask_training = True

    # Run experiments
    runner = ExperimentRunner(config)
    runner.run_all()

    # Generate report
    report = runner.generate_summary_report()
    print("\n" + report)


if __name__ == "__main__":
    main()
