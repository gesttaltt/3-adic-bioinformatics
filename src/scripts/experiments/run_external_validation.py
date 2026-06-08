#!/usr/bin/env python3
"""External Validation Runner for HIV Drug Resistance Models.

This script performs external validation using:
1. Stanford HIVdb as ground truth
2. Temporal hold-out (train on pre-2020, test on 2020+)
3. Cross-cohort validation (different geographic regions)
4. Leave-one-drug-out validation within drug classes

Copyright 2024-2025 AI Whisperers
Licensed under PolyForm Noncommercial License 1.0.0
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.hiv_data_loader import HIVDrugResistanceDataset
from src.models import TernaryVAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for external validation."""

    # Data paths
    data_dir: str = "data/hiv_resistance"
    results_dir: str = "results/external_validation"

    # Stanford HIVdb validation
    stanford_threshold: float = 0.8  # Correlation threshold for clinical relevance
    resistance_cutoffs: dict = field(
        default_factory=lambda: {
            "susceptible": 0.0,
            "low": 1.0,
            "intermediate": 2.0,
            "high": 3.0,
        }
    )

    # Temporal validation
    temporal_split_year: int = 2020  # Train pre-2020, test 2020+

    # Cross-cohort validation
    cohorts: list = field(default_factory=lambda: ["North_America", "Europe", "Africa", "Asia"])

    # Model settings
    latent_dim: int = 32
    hidden_dims: list = field(default_factory=lambda: [256, 128, 64])
    batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ExternalValidator:
    """External validation framework for HIV drug resistance models."""

    def __init__(self, config: ValidationConfig):
        self.config = config
        self.results = {}
        self.device = torch.device(config.device)

        # Create results directory
        Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    def load_stanford_data(self) -> pd.DataFrame | None:
        """Load Stanford HIVdb reference data.

        Returns:
            DataFrame with Stanford reference resistance scores
        """
        stanford_path = Path(self.config.data_dir) / "stanford_hivdb_reference.csv"

        if not stanford_path.exists():
            logger.warning(f"Stanford reference not found at {stanford_path}")
            logger.info("Attempting to use HIVDrugResistanceDataset as proxy...")
            return None

        df = pd.read_csv(stanford_path)
        logger.info(f"Loaded Stanford reference: {len(df)} entries")
        return df

    def validate_against_stanford(self, model: nn.Module, dataset: HIVDrugResistanceDataset, drug: str) -> dict:
        """Validate model predictions against Stanford HIVdb.

        Args:
            model: Trained VAE model
            dataset: HIV resistance dataset
            drug: Drug name to validate

        Returns:
            Dictionary of validation metrics
        """
        model.eval()
        predictions = []
        stanford_scores = []
        actual_resistance = []

        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                x = sample["sequence"].unsqueeze(0).to(self.device)

                # Get model prediction
                output = model(x)
                z = output.get("z_A_hyp", output.get("z_A_euc"))
                pred_resistance = z.norm(dim=-1).item()  # Use hyperbolic norm as proxy

                predictions.append(pred_resistance)

                # Get actual resistance if available
                if "resistance_score" in sample:
                    actual_resistance.append(sample["resistance_score"])
                if "stanford_score" in sample:
                    stanford_scores.append(sample["stanford_score"])

        predictions = np.array(predictions)

        metrics = {}

        # Compare with Stanford if available
        if stanford_scores:
            stanford_scores = np.array(stanford_scores)
            corr, p_value = stats.spearmanr(predictions, stanford_scores)
            metrics["stanford_correlation"] = corr
            metrics["stanford_p_value"] = p_value
            metrics["stanford_mae"] = mean_absolute_error(stanford_scores, predictions)

            # Clinical relevance check
            metrics["clinically_relevant"] = corr >= self.config.stanford_threshold

        # Compare with actual resistance if available
        if actual_resistance:
            actual_resistance = np.array(actual_resistance)
            metrics["actual_correlation"] = stats.spearmanr(predictions, actual_resistance)[0]
            metrics["actual_r2"] = r2_score(actual_resistance, predictions)
            metrics["actual_rmse"] = np.sqrt(mean_squared_error(actual_resistance, predictions))

        return metrics

    def temporal_validation(self, dataset: HIVDrugResistanceDataset, drug: str, model_factory: callable) -> dict:
        """Temporal hold-out validation: train on pre-2020, test on 2020+.

        Args:
            dataset: Full dataset with temporal information
            drug: Drug name
            model_factory: Callable that creates a new model instance

        Returns:
            Dictionary of temporal validation metrics
        """
        logger.info(f"Temporal validation for {drug} (split year: {self.config.temporal_split_year})")

        # Split data temporally
        train_indices = []
        test_indices = []

        for i in range(len(dataset)):
            sample = dataset[i]
            year = sample.get("year", sample.get("collection_year", 2015))
            if year < self.config.temporal_split_year:
                train_indices.append(i)
            else:
                test_indices.append(i)

        if len(test_indices) == 0:
            logger.warning(f"No test data for {drug} after {self.config.temporal_split_year}")
            return {"error": "No test data available"}

        logger.info(f"Train: {len(train_indices)}, Test: {len(test_indices)}")

        # Create train/test subsets
        train_subset = torch.utils.data.Subset(dataset, train_indices)
        test_subset = torch.utils.data.Subset(dataset, test_indices)

        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_subset, batch_size=self.config.batch_size, shuffle=False)

        # Train model on historical data
        model = model_factory().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _epoch in range(50):  # Quick training
            model.train()
            for batch in train_loader:
                x = batch["sequence"].to(self.device)
                optimizer.zero_grad()
                output = model(x)
                loss = self._compute_loss(output, x)
                loss.backward()
                optimizer.step()

        # Evaluate on future data
        model.eval()
        train_preds, train_actuals = self._get_predictions(model, train_loader)
        test_preds, test_actuals = self._get_predictions(model, test_loader)

        metrics = {
            "train_size": len(train_indices),
            "test_size": len(test_indices),
            "train_correlation": stats.spearmanr(train_preds, train_actuals)[0] if len(train_actuals) > 0 else None,
            "test_correlation": stats.spearmanr(test_preds, test_actuals)[0] if len(test_actuals) > 0 else None,
            "temporal_degradation": None,
        }

        if metrics["train_correlation"] and metrics["test_correlation"]:
            metrics["temporal_degradation"] = metrics["train_correlation"] - metrics["test_correlation"]

        return metrics

    def cross_cohort_validation(self, drug: str, model_factory: callable) -> dict:
        """Cross-cohort validation: train on one region, test on others.

        Args:
            drug: Drug name
            model_factory: Callable that creates a new model instance

        Returns:
            Dictionary of cross-cohort validation metrics
        """
        logger.info(f"Cross-cohort validation for {drug}")

        cohort_metrics = {}

        for train_cohort in self.config.cohorts:
            # Load train cohort data
            train_path = Path(self.config.data_dir) / f"{train_cohort.lower()}" / f"{drug}.csv"
            if not train_path.exists():
                logger.warning(f"Cohort data not found: {train_path}")
                continue

            pd.read_csv(train_path)

            # Train model on this cohort
            model_factory().to(self.device)
            # ... training code would go here

            # Test on other cohorts
            test_results = {}
            for test_cohort in self.config.cohorts:
                if test_cohort == train_cohort:
                    continue

                test_path = Path(self.config.data_dir) / f"{test_cohort.lower()}" / f"{drug}.csv"
                if not test_path.exists():
                    continue

                # Evaluate on test cohort
                # ... evaluation code would go here
                test_results[test_cohort] = {
                    "correlation": 0.0,  # Placeholder
                    "sample_size": 0,
                }

            cohort_metrics[train_cohort] = test_results

        return cohort_metrics

    def leave_one_drug_out(self, drug_class: str, model_factory: callable) -> dict:
        """Leave-one-drug-out cross-validation within a drug class.

        Args:
            drug_class: Drug class (PI, NRTI, NNRTI, INI)
            model_factory: Callable that creates a new model instance

        Returns:
            Dictionary with LODO results for each drug
        """
        drug_classes = {
            "PI": ["ATV", "DRV", "FPV", "IDV", "LPV", "NFV", "SQV", "TPV"],
            "NRTI": ["3TC", "ABC", "AZT", "D4T", "DDI", "FTC", "TDF"],
            "NNRTI": ["EFV", "ETR", "NVP", "RPV"],
            "INI": ["BIC", "CAB", "DTG", "EVG", "RAL"],
        }

        if drug_class not in drug_classes:
            logger.error(f"Unknown drug class: {drug_class}")
            return {}

        drugs = drug_classes[drug_class]
        lodo_results = {}

        for held_out_drug in drugs:
            logger.info(f"LODO: Holding out {held_out_drug}")

            # Train on all other drugs in class
            train_drugs = [d for d in drugs if d != held_out_drug]

            # Create combined training set
            # ... training code would go here

            # Evaluate on held-out drug
            lodo_results[held_out_drug] = {
                "train_drugs": train_drugs,
                "test_correlation": 0.0,  # Placeholder
                "transfer_success": False,
            }

        return lodo_results

    def run_full_validation(self, drugs: list = None) -> dict:
        """Run complete external validation suite.

        Args:
            drugs: List of drugs to validate (None = all available)

        Returns:
            Complete validation results
        """
        if drugs is None:
            drugs = ["ATV", "DRV", "AZT", "3TC", "EFV", "DTG"]  # Representative set

        all_results = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config.__dict__,
            "drugs": {},
        }

        for drug in drugs:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Validating {drug}")
            logger.info("=" * 60)

            drug_results = {}

            # Load dataset
            try:
                dataset = HIVDrugResistanceDataset(drug=drug, data_dir=self.config.data_dir)
            except Exception as e:
                logger.warning(f"Could not load data for {drug}: {e}")
                continue

            # Model factory
            def model_factory():
                return TernaryVAE(latent_dim=self.config.latent_dim)

            # Stanford validation
            model = model_factory().to(self.device)
            drug_results["stanford"] = self.validate_against_stanford(model, dataset, drug)

            # Temporal validation
            drug_results["temporal"] = self.temporal_validation(dataset, drug, model_factory)

            all_results["drugs"][drug] = drug_results

        # Save results
        results_path = Path(self.config.results_dir) / "external_validation_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        logger.info(f"\nResults saved to {results_path}")

        return all_results

    def _compute_loss(self, output: dict, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss."""
        logits = output.get("logits_A", output.get("logits"))
        if logits is None:
            return torch.tensor(0.0, device=self.device)
        return nn.functional.cross_entropy(logits.view(-1, 3), x.view(-1).long())

    def _get_predictions(self, model: nn.Module, loader: torch.utils.data.DataLoader) -> tuple:
        """Get predictions and actuals from a data loader."""
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch in loader:
                x = batch["sequence"].to(self.device)
                output = model(x)
                z = output.get("z_A_hyp", output.get("z_A_euc"))
                pred = z.norm(dim=-1).cpu().numpy()
                predictions.extend(pred.tolist())

                if "resistance_score" in batch:
                    actuals.extend(batch["resistance_score"].tolist())

        return np.array(predictions), np.array(actuals)


class ClinicalRelevanceChecker:
    """Check clinical relevance of model predictions."""

    # Stanford HIVdb resistance level definitions
    RESISTANCE_LEVELS = {
        1: "Susceptible",
        2: "Potential Low-Level Resistance",
        3: "Low-Level Resistance",
        4: "Intermediate Resistance",
        5: "High-Level Resistance",
    }

    # Key mutations by drug class from Stanford HIVdb
    KEY_MUTATIONS = {
        "PI": {
            "major": [30, 32, 33, 46, 47, 48, 50, 54, 76, 82, 84, 88, 90],
            "minor": [10, 11, 13, 16, 20, 24, 35, 36, 53, 58, 60, 62, 63, 64, 69, 71, 73, 77, 85, 89, 93],
        },
        "NRTI": {
            "major": [41, 62, 65, 67, 69, 70, 74, 75, 115, 151, 184, 210, 215, 219],
            "minor": [44, 67, 75, 77, 116, 118],
        },
        "NNRTI": {
            "major": [100, 101, 103, 106, 181, 188, 190, 225, 230],
            "minor": [98, 108, 138, 179, 221, 227, 238],
        },
        "INI": {
            "major": [66, 92, 118, 140, 143, 147, 148, 155, 263],
            "minor": [74, 97, 121, 153, 157, 163, 230],
        },
    }

    @classmethod
    def check_mutation_detection(cls, model_attention: np.ndarray, drug_class: str, top_k: int = 10) -> dict:
        """Check if model attention aligns with known key mutations.

        Args:
            model_attention: Attention weights per position
            drug_class: Drug class to check
            top_k: Number of top attended positions to check

        Returns:
            Dictionary with mutation detection metrics
        """
        if drug_class not in cls.KEY_MUTATIONS:
            return {"error": f"Unknown drug class: {drug_class}"}

        major = set(cls.KEY_MUTATIONS[drug_class]["major"])
        minor = set(cls.KEY_MUTATIONS[drug_class]["minor"])
        all_key = major | minor

        # Get top attended positions
        top_positions = np.argsort(model_attention)[-top_k:][::-1]
        top_set = set(top_positions.tolist())

        # Calculate overlap
        major_detected = top_set & major
        minor_detected = top_set & minor
        total_detected = top_set & all_key

        return {
            "major_mutations_detected": list(major_detected),
            "major_detection_rate": len(major_detected) / len(major) if major else 0,
            "minor_mutations_detected": list(minor_detected),
            "minor_detection_rate": len(minor_detected) / len(minor) if minor else 0,
            "total_key_mutations_detected": len(total_detected),
            "total_key_mutations": len(all_key),
            "detection_rate": len(total_detected) / len(all_key) if all_key else 0,
        }


def generate_validation_report(results: dict, output_path: Path) -> None:
    """Generate a detailed validation report.

    Args:
        results: Validation results dictionary
        output_path: Path to save the report
    """
    report_lines = [
        "# External Validation Report",
        f"\nGenerated: {results['timestamp']}",
        "\n## Summary\n",
    ]

    # Summarize by drug
    for drug, drug_results in results.get("drugs", {}).items():
        report_lines.append(f"\n### {drug}\n")

        # Stanford validation
        if "stanford" in drug_results:
            stanford = drug_results["stanford"]
            corr = stanford.get("stanford_correlation", "N/A")
            relevant = stanford.get("clinically_relevant", "N/A")
            report_lines.append(f"- Stanford Correlation: {corr}")
            report_lines.append(f"- Clinically Relevant: {relevant}")

        # Temporal validation
        if "temporal" in drug_results:
            temporal = drug_results["temporal"]
            train_corr = temporal.get("train_correlation", "N/A")
            test_corr = temporal.get("test_correlation", "N/A")
            degradation = temporal.get("temporal_degradation", "N/A")
            report_lines.append(f"- Train Correlation (pre-2020): {train_corr}")
            report_lines.append(f"- Test Correlation (2020+): {test_corr}")
            report_lines.append(f"- Temporal Degradation: {degradation}")

    # Save report
    with open(output_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="External validation for HIV drug resistance models")
    parser.add_argument(
        "--drugs",
        nargs="+",
        default=None,
        help="Drugs to validate (default: representative set)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/hiv_resistance",
        help="Data directory",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/external_validation",
        help="Results output directory",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    args = parser.parse_args()

    config = ValidationConfig(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        device=args.device,
    )

    validator = ExternalValidator(config)
    results = validator.run_full_validation(drugs=args.drugs)

    # Generate report
    report_path = Path(args.results_dir) / "validation_report.md"
    generate_validation_report(results, report_path)

    # Print summary
    print("\n" + "=" * 60)
    print("EXTERNAL VALIDATION COMPLETE")
    print("=" * 60)

    for drug, drug_results in results.get("drugs", {}).items():
        print(f"\n{drug}:")
        if "stanford" in drug_results:
            corr = drug_results["stanford"].get("stanford_correlation", "N/A")
            print(f"  Stanford Correlation: {corr}")
        if "temporal" in drug_results:
            test_corr = drug_results["temporal"].get("test_correlation", "N/A")
            print(f"  Temporal Test Correlation: {test_corr}")


if __name__ == "__main__":
    main()
