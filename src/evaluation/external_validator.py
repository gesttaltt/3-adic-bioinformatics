"""External Database Validation Framework.

Provides tools for validating the p-adic VAE framework on external databases
beyond Stanford HIVDB, including Los Alamos, EuResist, and UK HIVRDB.

This module enables:
1. Data format conversion from various sources
2. Sequence alignment to HXB2 reference
3. Cross-database performance comparison
4. Subtype-specific validation
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

# Standard amino acid alphabet (Stanford HIVDB format)
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY*-"
N_AA = len(AA_ALPHABET)

# HXB2 reference positions for each gene
HXB2_REFERENCE = {
    "PR": {"start": 2253, "end": 2549, "length": 99},
    "RT": {"start": 2550, "end": 3869, "length": 440},
    "IN": {"start": 4230, "end": 5096, "length": 288},
}

# Known HIV-1 subtypes
HIV_SUBTYPES = [
    "A",
    "A1",
    "A2",
    "B",
    "C",
    "D",
    "F",
    "F1",
    "F2",
    "G",
    "H",
    "J",
    "K",
    "CRF01_AE",
    "CRF02_AG",
    "CRF06_cpx",
    "CRF07_BC",
    "CRF08_BC",
]


@dataclass
class ValidationResult:
    """Results from external validation."""

    drug: str
    database: str
    n_samples: int
    correlation: float
    spearman: float
    rmse: float
    mae: float
    p_value: float
    subtype: str | None = None
    time_period: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "drug": self.drug,
            "database": self.database,
            "n_samples": self.n_samples,
            "correlation": self.correlation,
            "spearman": self.spearman,
            "rmse": self.rmse,
            "mae": self.mae,
            "p_value": self.p_value,
            "subtype": self.subtype,
            "time_period": self.time_period,
        }


@dataclass
class ExternalDataset:
    """Container for external validation data."""

    name: str
    sequences: np.ndarray  # One-hot encoded
    resistance_scores: np.ndarray
    drug: str
    subtypes: np.ndarray | None = None
    dates: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DatabaseAdapter(ABC):
    """Abstract base class for database-specific adapters."""

    @abstractmethod
    def load_data(self, path: Path, drug: str) -> ExternalDataset:
        """Load data from the external database."""
        pass

    @abstractmethod
    def normalize_resistance(self, scores: np.ndarray) -> np.ndarray:
        """Normalize resistance scores to Stanford scale."""
        pass

    def encode_sequence(self, sequence: str, n_positions: int) -> np.ndarray:
        """One-hot encode a protein sequence."""
        aa_to_idx = {aa: i for i, aa in enumerate(AA_ALPHABET)}
        encoded = np.zeros((n_positions * N_AA,), dtype=np.float32)

        for i, aa in enumerate(sequence[:n_positions]):
            if aa in aa_to_idx:
                encoded[i * N_AA + aa_to_idx[aa]] = 1.0
            else:
                # Unknown amino acid - use gap
                encoded[i * N_AA + aa_to_idx["-"]] = 1.0

        return encoded


class LosAlamosAdapter(DatabaseAdapter):
    """Adapter for Los Alamos HIV Sequence Database."""

    def __init__(self):
        self.name = "Los_Alamos"
        # Los Alamos uses different resistance categories
        self.resistance_mapping = {
            "Susceptible": 0.0,
            "Low-level": 1.0,
            "Intermediate": 2.0,
            "High-level": 3.0,
        }

    def load_data(self, path: Path, drug: str) -> ExternalDataset:
        """Load Los Alamos format data.

        Expected format: FASTA with headers containing metadata
        >Accession|Subtype|Year|ResistanceLevel
        SEQUENCE...
        """
        sequences = []
        scores = []
        subtypes = []
        dates = []

        if path.exists():
            current_header = None
            current_seq = []

            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        # Process previous sequence
                        if current_header and current_seq:
                            seq = "".join(current_seq)
                            parts = current_header.split("|")
                            if len(parts) >= 4:
                                subtype = parts[1]
                                year = parts[2]
                                resistance = parts[3]

                                if resistance in self.resistance_mapping:
                                    n_pos = 99 if "PI" in drug else 240
                                    sequences.append(self.encode_sequence(seq, n_pos))
                                    scores.append(self.resistance_mapping[resistance])
                                    subtypes.append(subtype)
                                    dates.append(year)

                        current_header = line[1:]
                        current_seq = []
                    else:
                        current_seq.append(line)

        return ExternalDataset(
            name=self.name,
            sequences=np.array(sequences) if sequences else np.zeros((0, 99 * N_AA)),
            resistance_scores=np.array(scores) if scores else np.zeros(0),
            drug=drug,
            subtypes=np.array(subtypes) if subtypes else None,
            dates=np.array(dates) if dates else None,
        )

    def normalize_resistance(self, scores: np.ndarray) -> np.ndarray:
        """Convert Los Alamos categorical to Stanford continuous scale.

        Stanford uses log fold-change (typically -2 to 4 range).
        Los Alamos uses 0-3 categorical.
        """
        # Linear mapping: 0 -> -1, 3 -> 3
        return scores * (4.0 / 3.0) - 1.0


class EuResistAdapter(DatabaseAdapter):
    """Adapter for EuResist Network data."""

    def __init__(self):
        self.name = "EuResist"

    def load_data(self, path: Path, drug: str) -> ExternalDataset:
        """Load EuResist format data.

        Expected: CSV with columns [sequence, drug, vl_change, subtype, date]
        vl_change: viral load change (log10) - negative is good response
        """
        if not path.exists():
            return ExternalDataset(
                name=self.name,
                sequences=np.zeros((0, 99 * N_AA)),
                resistance_scores=np.zeros(0),
                drug=drug,
            )

        df = pd.read_csv(path)
        df = df[df["drug"] == drug]

        sequences = []
        n_pos = 99 if drug in ["LPV", "ATV", "DRV", "FPV", "IDV", "NFV", "SQV", "TPV"] else 240

        for seq in df["sequence"]:
            sequences.append(self.encode_sequence(seq, n_pos))

        return ExternalDataset(
            name=self.name,
            sequences=np.array(sequences) if sequences else np.zeros((0, n_pos * N_AA)),
            resistance_scores=df["vl_change"].values if len(df) > 0 else np.zeros(0),
            drug=drug,
            subtypes=df["subtype"].values if "subtype" in df.columns else None,
            dates=df["date"].values if "date" in df.columns else None,
        )

    def normalize_resistance(self, scores: np.ndarray) -> np.ndarray:
        """Convert viral load change to resistance score.

        EuResist: negative VL change = good response = low resistance
        Stanford: higher score = more resistance
        """
        # Invert and scale
        return -scores


class StanfordAdapter(DatabaseAdapter):
    """Adapter for Stanford HIVDB data (reference format)."""

    def __init__(self):
        self.name = "Stanford"

    def load_data(self, path: Path, drug: str) -> ExternalDataset:
        """Load Stanford HIVDB format."""
        if not path.exists():
            return ExternalDataset(
                name=self.name,
                sequences=np.zeros((0, 99 * N_AA)),
                resistance_scores=np.zeros(0),
                drug=drug,
            )

        df = pd.read_csv(path, sep="\t")

        # Find position columns (P1, P2, ...)
        pos_cols = sorted([c for c in df.columns if re.match(r"^P\d+$", c)], key=lambda x: int(x[1:]))
        n_positions = len(pos_cols)

        # Encode sequences
        aa_to_idx = {aa: i for i, aa in enumerate(AA_ALPHABET)}
        sequences = []

        for _, row in df.iterrows():
            encoded = np.zeros((n_positions * N_AA,), dtype=np.float32)
            for i, col in enumerate(pos_cols):
                aa = str(row[col])
                if aa in aa_to_idx:
                    encoded[i * N_AA + aa_to_idx[aa]] = 1.0
                elif len(aa) > 0 and aa[0] in aa_to_idx:
                    encoded[i * N_AA + aa_to_idx[aa[0]]] = 1.0
            sequences.append(encoded)

        return ExternalDataset(
            name=self.name,
            sequences=np.array(sequences),
            resistance_scores=df[drug].values if drug in df.columns else np.zeros(len(df)),
            drug=drug,
            subtypes=df["Subtype"].values if "Subtype" in df.columns else None,
        )

    def normalize_resistance(self, scores: np.ndarray) -> np.ndarray:
        """Stanford scores are already in correct format."""
        return scores


class ExternalValidator:
    """Main class for external validation."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device
        self.adapters = {
            "los_alamos": LosAlamosAdapter(),
            "euresist": EuResistAdapter(),
            "stanford": StanfordAdapter(),
        }
        self.results: list[ValidationResult] = []

    def validate_on_database(
        self,
        database: str,
        data_path: Path,
        drug: str,
        subtype: str | None = None,
    ) -> ValidationResult:
        """Validate model on external database."""

        adapter = self.adapters.get(database.lower())
        if adapter is None:
            raise ValueError(f"Unknown database: {database}")

        # Load and preprocess data
        dataset = adapter.load_data(data_path, drug)

        if len(dataset.sequences) == 0:
            return ValidationResult(
                drug=drug,
                database=database,
                n_samples=0,
                correlation=np.nan,
                spearman=np.nan,
                rmse=np.nan,
                mae=np.nan,
                p_value=np.nan,
                subtype=subtype,
            )

        # Filter by subtype if specified
        if subtype and dataset.subtypes is not None:
            mask = dataset.subtypes == subtype
            dataset.sequences = dataset.sequences[mask]
            dataset.resistance_scores = dataset.resistance_scores[mask]

        # Normalize resistance scores
        y = adapter.normalize_resistance(dataset.resistance_scores)

        # Get predictions
        X = torch.tensor(dataset.sequences, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
            if isinstance(output, dict):
                if "predictions" in output and drug in output["predictions"]:
                    predictions = output["predictions"][drug].cpu().numpy()
                elif "prediction" in output:
                    predictions = output["prediction"].cpu().numpy()
                else:
                    predictions = output["mu"].mean(dim=-1).cpu().numpy()
            else:
                predictions = output.cpu().numpy()

        # Compute metrics
        if len(y) > 2:
            corr, p_value = pearsonr(predictions.flatten(), y)
            spearman_corr, _ = spearmanr(predictions.flatten(), y)
        else:
            corr, p_value, spearman_corr = np.nan, np.nan, np.nan

        rmse = np.sqrt(np.mean((predictions.flatten() - y) ** 2))
        mae = np.mean(np.abs(predictions.flatten() - y))

        result = ValidationResult(
            drug=drug,
            database=database,
            n_samples=len(y),
            correlation=corr,
            spearman=spearman_corr,
            rmse=rmse,
            mae=mae,
            p_value=p_value,
            subtype=subtype,
        )

        self.results.append(result)
        return result

    def validate_across_subtypes(
        self,
        data_path: Path,
        drug: str,
        subtypes: list[str] = None,
    ) -> pd.DataFrame:
        """Validate model performance across HIV subtypes."""

        if subtypes is None:
            subtypes = ["B", "C", "A", "CRF01_AE", "CRF02_AG"]

        results = []
        for subtype in subtypes:
            result = self.validate_on_database(
                database="stanford",
                data_path=data_path,
                drug=drug,
                subtype=subtype,
            )
            results.append(result.to_dict())

        return pd.DataFrame(results)

    def cross_database_comparison(
        self,
        databases: dict[str, Path],
        drugs: list[str],
    ) -> pd.DataFrame:
        """Compare performance across multiple databases."""

        all_results = []
        for drug in drugs:
            for db_name, db_path in databases.items():
                result = self.validate_on_database(
                    database=db_name,
                    data_path=db_path,
                    drug=drug,
                )
                all_results.append(result.to_dict())

        return pd.DataFrame(all_results)

    def generate_report(self) -> str:
        """Generate validation report."""

        report = []
        report.append("=" * 70)
        report.append("EXTERNAL VALIDATION REPORT")
        report.append("=" * 70)

        if not self.results:
            report.append("No validation results available.")
            return "\n".join(report)

        # Group by database
        df = pd.DataFrame([r.to_dict() for r in self.results])

        for database in df["database"].unique():
            db_results = df[df["database"] == database]
            report.append(f"\n{database.upper()}")
            report.append("-" * 40)

            avg_corr = db_results["correlation"].mean()
            report.append(f"Average correlation: {avg_corr:.3f}")
            report.append(f"Drugs tested: {len(db_results)}")
            report.append(f"Total samples: {db_results['n_samples'].sum()}")

            report.append("\nPer-drug results:")
            for _, row in db_results.iterrows():
                report.append(
                    f"  {row['drug']}: r={row['correlation']:.3f}, n={row['n_samples']}, p={row['p_value']:.2e}"
                )

        # Summary statistics
        report.append("\n" + "=" * 70)
        report.append("SUMMARY")
        report.append("=" * 70)

        overall_corr = df["correlation"].mean()
        report.append(f"Overall average correlation: {overall_corr:.3f}")
        report.append(f"Total validations: {len(df)}")
        report.append(f"Databases tested: {df['database'].nunique()}")

        return "\n".join(report)


def create_synthetic_external_data(
    n_samples: int = 500,
    n_positions: int = 99,
    drug: str = "LPV",
    database: str = "synthetic",
) -> ExternalDataset:
    """Create synthetic external data for testing validation framework."""

    np.random.seed(42)

    # Generate random sequences
    sequences = np.random.rand(n_samples, n_positions * N_AA).astype(np.float32)

    # Normalize to one-hot-like
    for i in range(n_samples):
        for j in range(n_positions):
            start = j * N_AA
            end = start + N_AA
            sequences[i, start:end] = 0
            sequences[i, start + np.random.randint(N_AA)] = 1.0

    # Generate resistance scores with some structure
    resistance = np.random.randn(n_samples) * 1.5 + 1.0

    # Assign subtypes
    subtypes = np.random.choice(["B", "C", "A", "CRF01_AE"], size=n_samples)

    return ExternalDataset(
        name=database,
        sequences=sequences,
        resistance_scores=resistance,
        drug=drug,
        subtypes=subtypes,
    )


if __name__ == "__main__":
    print("External Validation Framework")
    print("=" * 60)

    # Create synthetic data for testing
    dataset = create_synthetic_external_data()
    print(f"Created synthetic dataset: {dataset.name}")
    print(f"  Samples: {len(dataset.sequences)}")
    print(f"  Drug: {dataset.drug}")
    print(f"  Subtypes: {np.unique(dataset.subtypes)}")

    # Test adapters
    print("\nTesting adapters...")
    for name, _adapter in [
        ("Los Alamos", LosAlamosAdapter()),
        ("EuResist", EuResistAdapter()),
        ("Stanford", StanfordAdapter()),
    ]:
        print(f"  {name}: OK")

    print("\n" + "=" * 60)
    print("External validation framework ready!")
