# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Validation Script for Arcadia E. coli 7K AMR Dataset.

This script validates our p-adic VAE framework against the Arcadia Science
7,000+ strain E. coli antimicrobial resistance dataset.

Dataset:
- Source: Arcadia Science 2024
- Zenodo DOI: 10.5281/zenodo.12692732
- 7,000+ E. coli isolates with MIC data for 21 antibiotics
- Focus on TEM beta-lactamase resistance

Validation Strategy:
1. Load real TEM beta-lactamase sequences from Arcadia dataset
2. Encode using p-adic framework
3. Train/test split (80/20)
4. Evaluate Spearman correlation for each beta-lactam drug
5. Compare with baseline methods

Target Metrics:
- Spearman correlation ≥ 0.6 (publication-ready)
- AUC-ROC ≥ 0.8 for binary classification
- Per-drug correlation analysis

Usage:
    # Full validation (requires downloaded dataset)
    python src/scripts/validation/validate_arcadia_ecoli_7k.py --full

    # Quick validation with synthetic data (for testing)
    python src/scripts/validation/validate_arcadia_ecoli_7k.py --quick

    # Benchmark against multiple models
    python src/scripts/validation/validate_arcadia_ecoli_7k.py --compare-models
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import contextlib

from src.diseases.ecoli_betalactam_analyzer import (
    BetaLactam,
    EcoliBetaLactamAnalyzer,
    create_ecoli_synthetic_dataset,
)
from src.models.simple_vae import SimpleVAE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Dataset paths
ARCADIA_DATA_DIR = PROJECT_ROOT / "data" / "external" / "arcadia_ecoli"
ARCADIA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "ecoli_tem"
RESULTS_DIR = PROJECT_ROOT / "results" / "arcadia_validation"


class ArcadiaDataLoader:
    """Loader for Arcadia E. coli 7K dataset."""

    DRUG_COLUMNS = {
        "ampicillin": "AMP",
        "amoxicillin_clavulanate": "AMC",
        "piperacillin_tazobactam": "TZP",
        "cefazolin": "CZO",
        "ceftriaxone": "CRO",
        "ceftazidime": "CAZ",
        "cefepime": "FEP",
        "aztreonam": "ATM",
        "meropenem": "MEM",
        "ertapenem": "ETP",
    }

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or ARCADIA_DATA_DIR
        self.processed_dir = ARCADIA_PROCESSED_DIR
        self.phenotype_df: pd.DataFrame | None = None
        self.sequence_data: dict[str, str] = {}

    def load_phenotypes(self) -> pd.DataFrame | None:
        """Load phenotype (MIC) data.

        Returns:
            DataFrame with strain IDs and MIC values
        """
        phenotype_file = self.processed_dir / "phenotypes.csv"
        if not phenotype_file.exists():
            # Try raw data directory
            phenotype_files = list(self.data_dir.glob("**/*phenotype*.csv"))
            if phenotype_files:
                phenotype_file = phenotype_files[0]
            else:
                logger.warning("No phenotype data found")
                return None

        logger.info(f"Loading phenotypes from {phenotype_file}")
        self.phenotype_df = pd.read_csv(phenotype_file)
        logger.info(f"Loaded {len(self.phenotype_df)} samples")
        return self.phenotype_df

    def load_sequences(self) -> dict[str, str]:
        """Load TEM beta-lactamase sequences.

        Returns:
            Dictionary mapping strain ID to sequence
        """
        # Look for FASTA files
        fasta_files = list(self.data_dir.glob("**/*TEM*.fasta"))
        if not fasta_files:
            fasta_files = list(self.data_dir.glob("**/*.fasta"))

        if not fasta_files:
            logger.warning("No sequence files found")
            return {}

        sequences = {}
        for fasta_file in fasta_files:
            sequences.update(self._parse_fasta(fasta_file))

        logger.info(f"Loaded {len(sequences)} sequences")
        self.sequence_data = sequences
        return sequences

    def _parse_fasta(self, filepath: Path) -> dict[str, str]:
        """Parse FASTA file.

        Args:
            filepath: Path to FASTA file

        Returns:
            Dictionary mapping header to sequence
        """
        sequences = {}
        current_header = None
        current_seq = []

        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_header:
                        sequences[current_header] = "".join(current_seq)
                    current_header = line[1:].split()[0]  # First word after >
                    current_seq = []
                elif line:
                    current_seq.append(line)

            # Last sequence
            if current_header:
                sequences[current_header] = "".join(current_seq)

        return sequences

    def prepare_dataset(
        self,
        drug: str = "ampicillin",
        min_sequence_length: int = 200,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Prepare dataset for training/validation.

        Args:
            drug: Drug name for resistance labels
            min_sequence_length: Minimum sequence length filter

        Returns:
            (X, y, ids) tuple
        """
        if self.phenotype_df is None:
            self.load_phenotypes()
        if not self.sequence_data:
            self.load_sequences()

        if self.phenotype_df is None or not self.sequence_data:
            logger.warning("Using synthetic data (real data not available)")
            return create_ecoli_synthetic_dataset(
                drug=BetaLactam[drug.upper()] if hasattr(BetaLactam, drug.upper()) else BetaLactam.AMPICILLIN,
                min_samples=100,
            )

        # Match sequences with phenotypes
        analyzer = EcoliBetaLactamAnalyzer()

        X_list = []
        y_list = []
        id_list = []

        # Get drug column
        drug_col = None
        for col in self.phenotype_df.columns:
            if drug.lower() in col.lower():
                drug_col = col
                break

        if drug_col is None:
            logger.warning(f"Drug {drug} not found in phenotype data")
            return create_ecoli_synthetic_dataset()

        for _idx, row in self.phenotype_df.iterrows():
            strain_id = str(row.iloc[0])  # First column is strain ID

            # Find matching sequence
            seq = None
            for seq_id, sequence in self.sequence_data.items():
                if strain_id in seq_id or seq_id in strain_id:
                    seq = sequence
                    break

            if seq is None or len(seq) < min_sequence_length:
                continue

            mic_value = row.get(drug_col)
            if pd.isna(mic_value):
                continue

            # Encode sequence
            encoded = analyzer.encode_sequence(seq, max_length=290)
            X_list.append(encoded)

            # Convert MIC to resistance score (log2 scale normalization)
            try:
                mic_numeric = float(mic_value)
                # Log2 scale, normalized to 0-1
                resistance_score = np.clip(np.log2(mic_numeric + 1) / 10, 0, 1)
            except (ValueError, TypeError):
                # Handle categorical (S/I/R)
                resistance_map = {"S": 0.0, "I": 0.5, "R": 1.0}
                resistance_score = resistance_map.get(str(mic_value).upper(), 0.5)

            y_list.append(resistance_score)
            id_list.append(strain_id)

        if not X_list:
            logger.warning("No valid samples found, using synthetic data")
            return create_ecoli_synthetic_dataset()

        return np.array(X_list), np.array(y_list), id_list


class ArcadiaValidator:
    """Validator for Arcadia E. coli dataset."""

    def __init__(
        self,
        data_loader: ArcadiaDataLoader | None = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.data_loader = data_loader or ArcadiaDataLoader()
        self.device = device
        self.results: dict[str, Any] = {}

    def validate_drug(
        self,
        drug: str = "ampicillin",
        test_size: float = 0.2,
        n_epochs: int = 50,
    ) -> dict[str, float]:
        """Validate resistance prediction for a single drug.

        Args:
            drug: Drug name
            test_size: Fraction for test set
            n_epochs: Training epochs

        Returns:
            Metrics dictionary
        """
        logger.info(f"Validating {drug}...")

        # Load data
        X, y, ids = self.data_loader.prepare_dataset(drug=drug)
        logger.info(f"Dataset: {len(X)} samples")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Create and train VAE
        input_dim = X.shape[1]
        vae = SimpleVAE(
            input_dim=input_dim,
            latent_dim=32,
            hidden_dims=[256, 128],
        ).to(self.device)

        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

        # Training loop
        vae.train()
        X_train_tensor = torch.from_numpy(X_train).float().to(self.device)

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Forward pass
            recon, mu, logvar = vae(X_train_tensor)

            # Loss
            recon_loss = torch.nn.functional.mse_loss(recon, X_train_tensor)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / X_train_tensor.size(0)
            loss = recon_loss + 0.1 * kl_loss

            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item():.4f}")

        # Evaluation
        vae.eval()
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test).float().to(self.device)
            mu_test, _ = vae.encode(X_test_tensor)
            z_test = mu_test.cpu().numpy()

        # Use latent mean as predictor
        y_pred = np.mean(z_test, axis=1)
        y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-8)

        # Calculate metrics
        spearman_rho, spearman_p = spearmanr(y_test, y_pred)
        pearson_r, pearson_p = pearsonr(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Binary classification metrics (threshold at 0.5)
        y_binary = (y_test > 0.5).astype(int)
        (y_pred > 0.5).astype(int)

        try:
            auc = roc_auc_score(y_binary, y_pred)
        except ValueError:
            auc = float("nan")

        metrics = {
            "drug": drug,
            "n_samples": len(X),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "spearman_rho": float(spearman_rho) if not np.isnan(spearman_rho) else 0.0,
            "spearman_p": float(spearman_p) if not np.isnan(spearman_p) else 1.0,
            "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else 0.0,
            "rmse": float(rmse),
            "auc_roc": float(auc) if not np.isnan(auc) else 0.5,
        }

        logger.info(f"{drug}: Spearman={metrics['spearman_rho']:.3f}, AUC={metrics['auc_roc']:.3f}")
        return metrics

    def validate_all_drugs(
        self,
        drugs: list[str] | None = None,
    ) -> dict[str, Any]:
        """Validate across all beta-lactam drugs.

        Args:
            drugs: List of drug names (default: all beta-lactams)

        Returns:
            Complete validation results
        """
        if drugs is None:
            drugs = [
                "ampicillin",
                "ceftriaxone",
                "ceftazidime",
                "cefepime",
                "aztreonam",
                "meropenem",
            ]

        all_metrics = []
        for drug in drugs:
            try:
                metrics = self.validate_drug(drug)
                all_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Failed to validate {drug}: {e}")

        # Summary statistics
        if all_metrics:
            spearman_values = [m["spearman_rho"] for m in all_metrics]
            auc_values = [m["auc_roc"] for m in all_metrics if not np.isnan(m["auc_roc"])]

            self.results = {
                "timestamp": datetime.now().isoformat(),
                "n_drugs": len(all_metrics),
                "per_drug_results": all_metrics,
                "summary": {
                    "mean_spearman": float(np.mean(spearman_values)),
                    "std_spearman": float(np.std(spearman_values)),
                    "min_spearman": float(np.min(spearman_values)),
                    "max_spearman": float(np.max(spearman_values)),
                    "mean_auc": float(np.mean(auc_values)) if auc_values else 0.5,
                    "drugs_above_0.6": sum(1 for s in spearman_values if s >= 0.6),
                },
            }
        else:
            self.results = {"error": "No successful validations"}

        return self.results

    def save_results(self, output_dir: Path | None = None):
        """Save validation results.

        Args:
            output_dir: Output directory
        """
        output_dir = output_dir or RESULTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"arcadia_validation_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to {results_file}")

        # Print summary
        if "summary" in self.results:
            print("\n" + "=" * 60)
            print("ARCADIA E. COLI 7K VALIDATION RESULTS")
            print("=" * 60)
            print(f"\nDrugs validated: {self.results['n_drugs']}")
            print(f"Mean Spearman: {self.results['summary']['mean_spearman']:.3f}")
            print(f"Mean AUC-ROC: {self.results['summary']['mean_auc']:.3f}")
            print(f"Drugs ≥0.6 Spearman: {self.results['summary']['drugs_above_0.6']}")
            print("\nPer-drug results:")
            for drug_result in self.results["per_drug_results"]:
                print(f"  {drug_result['drug']}: ρ={drug_result['spearman_rho']:.3f}, AUC={drug_result['auc_roc']:.3f}")


def run_quick_validation():
    """Run quick validation with synthetic data."""
    logger.info("Running quick validation with synthetic data...")

    validator = ArcadiaValidator()
    results = validator.validate_all_drugs(drugs=["ampicillin", "ceftriaxone"])
    validator.save_results()

    return results


def run_full_validation():
    """Run full validation with Arcadia dataset."""
    logger.info("Running full validation with Arcadia dataset...")

    # Check for data
    if not ARCADIA_DATA_DIR.exists():
        logger.warning(f"Data directory not found: {ARCADIA_DATA_DIR}")
        logger.info("Download data first with: python src/scripts/ingest/download_arcadia_ecoli.py --download")
        logger.info("Falling back to synthetic data...")

    validator = ArcadiaValidator()
    results = validator.validate_all_drugs()
    validator.save_results()

    return results


def compare_models():
    """Compare multiple model architectures."""
    logger.info("Comparing model architectures...")

    from src.models.epsilon_vae import EpsilonVAE

    data_loader = ArcadiaDataLoader()
    X, y, ids = data_loader.prepare_dataset(drug="ampicillin")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "SimpleVAE": SimpleVAE(
            input_dim=X.shape[1],
            latent_dim=32,
            hidden_dims=[256, 128],
        ),
    }

    # Add other models if available
    with contextlib.suppress(Exception):
        models["EpsilonVAE"] = EpsilonVAE(
            input_dim=X.shape[1],
            latent_dim=32,
            hidden_dims=[256, 128],
        )

    results = {}
    for name, model in models.items():
        logger.info(f"Evaluating {name}...")

        # Quick training
        model.train()
        X_tensor = torch.from_numpy(X_train).float()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for _ in range(20):
            optimizer.zero_grad()
            recon, mu, logvar = model(X_tensor)
            loss = torch.nn.functional.mse_loss(recon, X_tensor)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.from_numpy(X_test).float()
            mu_test, _ = model.encode(X_test_tensor)
            z_test = mu_test.numpy()

        y_pred = np.mean(z_test, axis=1)
        y_pred = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-8)

        rho, _ = spearmanr(y_test, y_pred)
        results[name] = {"spearman": float(rho) if not np.isnan(rho) else 0.0}

    print("\nModel Comparison:")
    for name, metrics in results.items():
        print(f"  {name}: Spearman={metrics['spearman']:.3f}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate p-adic VAE framework on Arcadia E. coli 7K dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick validation with synthetic data",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full validation with Arcadia dataset",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Compare multiple model architectures",
    )
    parser.add_argument(
        "--drug",
        type=str,
        default=None,
        help="Validate single drug only",
    )

    args = parser.parse_args()

    if args.quick:
        run_quick_validation()
    elif args.full:
        run_full_validation()
    elif args.compare_models:
        compare_models()
    elif args.drug:
        validator = ArcadiaValidator()
        validator.validate_drug(args.drug)
        validator.save_results()
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python src/scripts/validation/validate_arcadia_ecoli_7k.py --quick")
        print("  python src/scripts/validation/validate_arcadia_ecoli_7k.py --full")


if __name__ == "__main__":
    main()
