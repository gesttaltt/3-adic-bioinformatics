# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Disease-specific experiment implementations.

This module provides experiment classes specialized for:
- HIV drug resistance prediction
- SARS-CoV-2 variant analysis
- Tuberculosis MDR prediction
- Influenza vaccine strain selection

All experiments follow the same reproducible framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base_experiment import (
    BaseExperiment,
    ExperimentConfig,
    ExperimentResult,
    MetricComputer,
)

logger = logging.getLogger(__name__)


@dataclass
class DiseaseExperimentConfig(ExperimentConfig):
    """Configuration for disease-specific experiments.

    Extends base config with disease-specific parameters.
    """

    disease: str = "hiv"
    target: str = ""  # Drug name or specific target
    gene: str = ""  # Gene/protein (e.g., "PI", "RT", "IN")

    # Data paths
    data_dir: str = "data/research"

    # Loss function weights
    ranking_weight: float = 0.3
    contrastive_weight: float = 0.1

    # Disease-specific
    use_transfer_learning: bool = False
    pretrain_diseases: list[str] = field(default_factory=list)


class DiseaseExperiment(BaseExperiment):
    """Experiment class for single disease/drug prediction.

    Implements the full training and evaluation pipeline for
    drug resistance or other disease-specific predictions.
    """

    def __init__(
        self,
        config: DiseaseExperimentConfig,
        model_class: type[nn.Module] | None = None,
    ):
        """Initialize disease experiment.

        Args:
            config: Experiment configuration
            model_class: Optional custom model class
        """
        super().__init__(config)
        self.config: DiseaseExperimentConfig = config
        self.model_class = model_class or self._get_default_model_class()

    def _get_default_model_class(self) -> type[nn.Module]:
        """Get default model class for the disease."""
        # Import here to avoid circular imports
        from src.models import TernaryVAE

        return TernaryVAE

    def create_model(self) -> nn.Module:
        """Create a fresh model instance."""
        # Simple MLP for now - can be replaced with TernaryVAE
        model = SimpleVAE(
            input_dim=self._input_dim,
            latent_dim=self.config.latent_dim,
            hidden_dims=self.config.hidden_dims,
        )
        return model.to(self.device)

    def train_fold(
        self,
        model: nn.Module,
        train_data: tuple[np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray],
    ) -> dict[str, Any]:
        """Train model on one fold."""
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Store input dim for model creation
        self._input_dim = X_train.shape[1]

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)

        # Training loop
        history = {"train_loss": [], "val_loss": [], "val_spearman": []}
        best_val_spearman = -1.0
        patience_counter = 0

        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                output = model(batch_x)

                # Compute loss
                loss = self._compute_loss(output, batch_x, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            # Validation
            val_preds, val_metrics = self.evaluate(model, val_data)
            val_spearman = val_metrics.get("spearman", 0.0)

            history["train_loss"].append(epoch_loss / len(train_loader))
            history["val_loss"].append(val_metrics.get("rmse", 0.0))
            history["val_spearman"].append(val_spearman)

            # Early stopping
            if val_spearman > best_val_spearman:
                best_val_spearman = val_spearman
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        return history

    def _compute_loss(
        self,
        output: dict[str, torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss."""
        losses = {}

        # Reconstruction loss
        if "x_recon" in output:
            losses["recon"] = nn.functional.mse_loss(output["x_recon"], x)

        # KL divergence
        if "mu" in output and "logvar" in output:
            kl = -0.5 * torch.sum(1 + output["logvar"] - output["mu"].pow(2) - output["logvar"].exp())
            losses["kl"] = 0.001 * kl / x.size(0)

        # Ranking loss (correlation with fitness)
        if "z" in output and self.config.ranking_weight > 0:
            z_proj = output["z"][:, 0]  # First latent dimension
            z_centered = z_proj - z_proj.mean()
            y_centered = y - y.mean()

            z_std = torch.sqrt(torch.sum(z_centered**2) + 1e-8)
            y_std = torch.sqrt(torch.sum(y_centered**2) + 1e-8)

            corr = torch.sum(z_centered * y_centered) / (z_std * y_std)
            losses["rank"] = self.config.ranking_weight * (-corr)

        return sum(losses.values())

    def evaluate(
        self,
        model: nn.Module,
        test_data: tuple[np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Evaluate model on test data."""
        X_test, y_test = test_data

        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            output = model(X_tensor)

            # Use first latent dimension as prediction
            if "z" in output:
                predictions = output["z"][:, 0].cpu().numpy()
            else:
                predictions = output["x_recon"].mean(dim=1).cpu().numpy()

        # Compute metrics
        metrics = MetricComputer.compute_all(y_test, predictions, "regression")

        return predictions, metrics


class CrossDiseaseExperiment:
    """Run experiments across multiple diseases for comparison.

    This class orchestrates experiments across HIV, SARS-CoV-2, TB,
    and Influenza for publication-ready cross-disease validation.
    """

    def __init__(
        self,
        diseases: list[str],
        base_config: ExperimentConfig | None = None,
        model_class: type[nn.Module] | None = None,
    ):
        """Initialize cross-disease experiment.

        Args:
            diseases: List of disease identifiers
            base_config: Base configuration (disease-specific overrides applied)
            model_class: Model class to use for all diseases
        """
        self.diseases = diseases
        self.base_config = base_config or ExperimentConfig()
        self.model_class = model_class
        self.results: dict[str, ExperimentResult] = {}

    def run_all(self) -> dict[str, ExperimentResult]:
        """Run experiments on all diseases.

        Returns:
            Dictionary mapping disease name to results
        """
        for disease in self.diseases:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Running experiment for: {disease.upper()}")
            logger.info(f"{'=' * 60}\n")

            try:
                result = self.run_disease(disease)
                self.results[disease] = result
            except Exception as e:
                logger.error(f"Failed to run {disease}: {e}")
                continue

        return self.results

    def run_disease(self, disease: str) -> ExperimentResult:
        """Run experiment for a single disease.

        Args:
            disease: Disease identifier

        Returns:
            Experiment result
        """
        # Create disease-specific config
        config = DiseaseExperimentConfig(
            name=f"{disease}_experiment",
            disease=disease,
            **{k: v for k, v in self.base_config.to_dict().items() if k not in ["name"]},
        )

        # Load data
        X, y = self._load_disease_data(disease, config)

        # Run experiment
        experiment = DiseaseExperiment(config, self.model_class)
        experiment._input_dim = X.shape[1]
        return experiment.run(X, y)

    def _load_disease_data(self, disease: str, config: DiseaseExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
        """Load data for a disease.

        This method should be extended for each new disease.
        """
        data_loaders = {
            "hiv": self._load_hiv_data,
            "sars_cov_2": self._load_sars_cov_2_data,
            "tuberculosis": self._load_tb_data,
            "influenza": self._load_influenza_data,
        }

        loader = data_loaders.get(disease)
        if loader is None:
            raise ValueError(f"Unknown disease: {disease}")

        return loader(config)

    def _load_hiv_data(self, config: DiseaseExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
        """Load HIV drug resistance data."""
        data_dir = Path(config.data_dir)
        target = config.target or "DRV"  # Default to Darunavir

        # Determine file based on drug
        drug_to_gene = {
            "FPV": "pi",
            "ATV": "pi",
            "IDV": "pi",
            "LPV": "pi",
            "NFV": "pi",
            "SQV": "pi",
            "TPV": "pi",
            "DRV": "pi",
            "ABC": "nrti",
            "AZT": "nrti",
            "D4T": "nrti",
            "DDI": "nrti",
            "FTC": "nrti",
            "3TC": "nrti",
            "TDF": "nrti",
            "DOR": "nnrti",
            "EFV": "nnrti",
            "ETR": "nnrti",
            "NVP": "nnrti",
            "RPV": "nnrti",
            "BIC": "ini",
            "CAB": "ini",
            "DTG": "ini",
            "EVG": "ini",
            "RAL": "ini",
        }

        gene = drug_to_gene.get(target, "pi")
        file_path = data_dir / f"stanford_hivdb_{gene}.txt"

        if not file_path.exists():
            raise FileNotFoundError(f"HIV data not found: {file_path}")

        # Load and process data
        df = pd.read_csv(file_path, sep="\t", low_memory=False)

        # Find drug column
        drug_col = None
        for col in df.columns:
            if col.upper() == target.upper():
                drug_col = col
                break

        if drug_col is None:
            raise ValueError(f"Drug {target} not found in {file_path}")

        # One-hot encode sequences
        X, y = self._encode_hiv_sequences(df, drug_col)
        return X, y

    def _encode_hiv_sequences(self, df: pd.DataFrame, drug_col: str) -> tuple[np.ndarray, np.ndarray]:
        """One-hot encode HIV sequences."""
        # Filter valid data
        df = df[[col for col in df.columns if col.startswith("P")] + [drug_col]]
        df[drug_col] = pd.to_numeric(df[drug_col], errors="coerce")
        df = df.dropna(subset=[drug_col])

        # Get position columns
        pos_cols = [col for col in df.columns if col.startswith("P") and col[1:].isdigit()]
        pos_cols = sorted(pos_cols, key=lambda x: int(x[1:]))

        # Encode
        aa_alphabet = "ACDEFGHIKLMNPQRSTVWY*-"
        n_aa = len(aa_alphabet)
        aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}

        X = np.zeros((len(df), len(pos_cols) * n_aa), dtype=np.float32)

        for i, (_, row) in enumerate(df.iterrows()):
            for j, col in enumerate(pos_cols):
                aa = str(row[col]).upper() if pd.notna(row[col]) else "-"
                aa = aa[0] if len(aa) > 0 else "-"
                idx = aa_to_idx.get(aa, aa_to_idx["-"])
                X[i, j * n_aa + idx] = 1.0

        y = df[drug_col].values.astype(np.float32)
        return X, y

    def _load_sars_cov_2_data(self, config: DiseaseExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
        """Load SARS-CoV-2 data.

        Uses synthetic data from disease analyzer for testing.
        For production, load from GISAID or CoV-RDB.
        """
        from src.diseases.sars_cov2_analyzer import (
            SARSCoV2Gene,
            create_sars_cov2_dataset,
        )

        # Determine target gene based on config
        gene = SARSCoV2Gene.NSP5  # Default: Mpro for Paxlovid resistance
        if config.gene:
            gene_mapping = {
                "spike": SARSCoV2Gene.SPIKE,
                "nsp5": SARSCoV2Gene.NSP5,
                "mpro": SARSCoV2Gene.NSP5,
            }
            gene = gene_mapping.get(config.gene.lower(), SARSCoV2Gene.NSP5)

        X, y, _ = create_sars_cov2_dataset(
            gene=gene,
            include_resistance=True,
            min_samples=100,
        )
        return X, y

    def _load_tb_data(self, config: DiseaseExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
        """Load Tuberculosis data.

        Uses synthetic data from disease analyzer for testing.
        For production, load from WHO Catalogue or CRyPTIC.
        """
        from src.diseases.tuberculosis_analyzer import (
            TBDrug,
            create_tb_synthetic_dataset,
        )

        # Map target drug from config
        drug = TBDrug.RIFAMPICIN  # Default
        if config.target:
            drug_mapping = {
                "rif": TBDrug.RIFAMPICIN,
                "rifampicin": TBDrug.RIFAMPICIN,
                "inh": TBDrug.ISONIAZID,
                "isoniazid": TBDrug.ISONIAZID,
                "emb": TBDrug.ETHAMBUTOL,
                "ethambutol": TBDrug.ETHAMBUTOL,
                "pza": TBDrug.PYRAZINAMIDE,
                "pyrazinamide": TBDrug.PYRAZINAMIDE,
                "lfx": TBDrug.LEVOFLOXACIN,
                "mfx": TBDrug.MOXIFLOXACIN,
                "amk": TBDrug.AMIKACIN,
                "bdq": TBDrug.BEDAQUILINE,
                "lzd": TBDrug.LINEZOLID,
            }
            drug = drug_mapping.get(config.target.lower(), TBDrug.RIFAMPICIN)

        X, y, _ = create_tb_synthetic_dataset(
            drug=drug,
            min_samples=100,
        )
        return X, y

    def _load_influenza_data(self, config: DiseaseExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
        """Load Influenza data.

        Uses synthetic data from disease analyzer for testing.
        For production, load from GISAID or FluDB.
        """
        from src.diseases.influenza_analyzer import (
            InfluenzaDrug,
            InfluenzaSubtype,
            create_influenza_synthetic_dataset,
        )

        # Map subtype from config
        subtype = InfluenzaSubtype.H3N2  # Default
        if config.gene:
            subtype_mapping = {
                "h3n2": InfluenzaSubtype.H3N2,
                "h1n1": InfluenzaSubtype.H1N1_SEASONAL,
                "h1n1_pdm": InfluenzaSubtype.H1N1_PANDEMIC,
                "b_victoria": InfluenzaSubtype.B_VICTORIA,
                "b_yamagata": InfluenzaSubtype.B_YAMAGATA,
            }
            subtype = subtype_mapping.get(config.gene.lower(), InfluenzaSubtype.H3N2)

        # Map target drug from config
        drug = InfluenzaDrug.OSELTAMIVIR  # Default
        if config.target:
            drug_mapping = {
                "oseltamivir": InfluenzaDrug.OSELTAMIVIR,
                "tamiflu": InfluenzaDrug.OSELTAMIVIR,
                "zanamivir": InfluenzaDrug.ZANAMIVIR,
                "relenza": InfluenzaDrug.ZANAMIVIR,
                "baloxavir": InfluenzaDrug.BALOXAVIR,
                "xofluza": InfluenzaDrug.BALOXAVIR,
                "peramivir": InfluenzaDrug.PERAMIVIR,
            }
            drug = drug_mapping.get(config.target.lower(), InfluenzaDrug.OSELTAMIVIR)

        X, y, _ = create_influenza_synthetic_dataset(
            subtype=subtype,
            drug=drug,
            min_samples=100,
        )
        return X, y

    def generate_comparison_report(self) -> str:
        """Generate comparison report across diseases.

        Returns:
            Markdown-formatted report
        """
        if not self.results:
            return "No results available. Run experiments first."

        lines = [
            "# Cross-Disease Experiment Results",
            "",
            "## Summary",
            "",
            "| Disease | Spearman | RMSE | Samples |",
            "|---------|----------|------|---------|",
        ]

        for disease, result in self.results.items():
            spearman = result.metrics.get("spearman_mean", "N/A")
            rmse = result.metrics.get("rmse_mean", "N/A")
            n_samples = len(result.predictions["y_true"]) if result.predictions else "N/A"

            if isinstance(spearman, float):
                spearman = f"{spearman:.4f}"
            if isinstance(rmse, float):
                rmse = f"{rmse:.4f}"

            lines.append(f"| {disease} | {spearman} | {rmse} | {n_samples} |")

        lines.extend(
            [
                "",
                "## Per-Disease Details",
                "",
            ]
        )

        for disease, result in self.results.items():
            lines.append(f"### {disease.upper()}")
            lines.append("")
            lines.append(result.summary())
            lines.append("")

        return "\n".join(lines)


class SimpleVAE(nn.Module):
    """Simple VAE for baseline experiments."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dims: list[int] = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]

        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(h_dim),
                    nn.Dropout(0.1),
                ]
            )
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)

        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(h_dim),
                    nn.Dropout(0.1),
                ]
            )
            in_dim = h_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std

        x_recon = self.decoder(z)

        return {
            "x_recon": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }
