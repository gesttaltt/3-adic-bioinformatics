# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""ProteinGym dataset loader for large-scale diverse mutations.

ProteinGym contains 500K+ mutations across diverse protein families,
enabling training on large-scale data for improved generalization.

Reference: Notin et al. (2022) - ProteinGym: Large-Scale Benchmarks
           for Protein Fitness Prediction and Design
"""

from __future__ import annotations

import csv
import gzip
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from src.bioinformatics.data.preprocessing import (
    compute_features,
)


@dataclass
class ProteinGymRecord:
    """Container for a ProteinGym mutation record."""

    protein_id: str
    mutation: str  # Format: "A123V" or "A123V:B456W" for multi-point
    fitness: float  # DMS fitness score
    ddg: float | None = None  # DDG if available
    sequence: str | None = None

    @property
    def is_single_point(self) -> bool:
        """Check if single-point mutation."""
        return ":" not in self.mutation

    @property
    def wild_type(self) -> str:
        """Extract wild-type AA (first mutation only)."""
        return self.mutation[0]

    @property
    def position(self) -> int:
        """Extract position (first mutation only)."""
        # Handle format like "A123V"
        num_str = ""
        for c in self.mutation[1:]:
            if c.isdigit():
                num_str += c
            else:
                break
        return int(num_str) if num_str else 0

    @property
    def mutant(self) -> str:
        """Extract mutant AA (first mutation only)."""
        return self.mutation[-1]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ProteinGymRecord:
        return cls(**data)


class ProteinGymDataset(Dataset):
    """PyTorch Dataset for ProteinGym mutations (in-memory)."""

    def __init__(
        self,
        records: list[ProteinGymRecord],
        aa_embeddings: dict[str, torch.Tensor] | None = None,
        curvature: float = 1.0,
        use_fitness_as_label: bool = True,
    ):
        """Initialize dataset.

        Args:
            records: List of ProteinGymRecord objects
            aa_embeddings: Optional AA embeddings for hyperbolic features
            curvature: Poincaré ball curvature
            use_fitness_as_label: Use fitness score instead of DDG
        """
        self.records = records
        self.aa_embeddings = aa_embeddings
        self.curvature = curvature
        self.use_fitness_as_label = use_fitness_as_label

        self._features = []
        self._labels = []
        self._compute_all_features()

    def _compute_all_features(self) -> None:
        """Compute features for all records."""
        from src.bioinformatics.data.preprocessing import add_hyperbolic_features

        for record in self.records:
            if not record.is_single_point:
                continue  # Skip multi-point for now

            features = compute_features(
                wild_type=record.wild_type,
                mutant=record.mutant,
            )

            if self.aa_embeddings is not None:
                features = add_hyperbolic_features(
                    features,
                    record.wild_type,
                    record.mutant,
                    self.aa_embeddings,
                    self.curvature,
                )

            self._features.append(features.to_array(include_hyperbolic=self.aa_embeddings is not None))

            # Use fitness or DDG as label
            if self.use_fitness_as_label:
                self._labels.append(record.fitness)
            else:
                self._labels.append(record.ddg if record.ddg is not None else 0.0)

        self._features = np.array(self._features, dtype=np.float32)
        self._labels = np.array(self._labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self._features[idx]),
            torch.tensor(self._labels[idx], dtype=torch.float32),
        )

    @property
    def feature_dim(self) -> int:
        return self._features.shape[1] if len(self._features) > 0 else 0

    def get_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Get raw numpy arrays for sklearn."""
        return self._features, self._labels


class ProteinGymIterableDataset(IterableDataset):
    """Iterable Dataset for streaming large ProteinGym data."""

    def __init__(
        self,
        data_dir: Path,
        aa_embeddings: dict[str, torch.Tensor] | None = None,
        curvature: float = 1.0,
        single_point_only: bool = True,
    ):
        """Initialize streaming dataset.

        Args:
            data_dir: Directory containing ProteinGym CSV files
            aa_embeddings: Optional AA embeddings
            curvature: Poincaré ball curvature
            single_point_only: Only yield single-point mutations
        """
        self.data_dir = Path(data_dir)
        self.aa_embeddings = aa_embeddings
        self.curvature = curvature
        self.single_point_only = single_point_only

        # Find all CSV files
        self.csv_files = list(self.data_dir.glob("*.csv")) + list(self.data_dir.glob("*.csv.gz"))

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        from src.bioinformatics.data.preprocessing import add_hyperbolic_features

        for csv_file in self.csv_files:
            # Handle gzipped files
            f = gzip.open(csv_file, "rt") if csv_file.suffix == ".gz" else open(csv_file, newline="")

            try:
                reader = csv.DictReader(f)
                for row in reader:
                    mutation = row.get("mutant") or row.get("mutation")
                    if not mutation:
                        continue

                    # Skip multi-point if requested
                    if self.single_point_only and ":" in mutation:
                        continue

                    # Parse mutation
                    try:
                        wt = mutation[0]
                        mut = mutation[-1]
                        fitness = float(row.get("DMS_score") or row.get("fitness") or 0)
                    except (ValueError, IndexError):
                        continue

                    # Validate amino acids
                    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
                    if wt not in valid_aa or mut not in valid_aa:
                        continue

                    # Compute features
                    features = compute_features(wild_type=wt, mutant=mut)

                    if self.aa_embeddings is not None:
                        features = add_hyperbolic_features(features, wt, mut, self.aa_embeddings, self.curvature)

                    feature_array = features.to_array(include_hyperbolic=self.aa_embeddings is not None)

                    yield (
                        torch.from_numpy(feature_array),
                        torch.tensor(fitness, dtype=torch.float32),
                    )
            finally:
                f.close()


class ProteinGymLoader:
    """Loader for ProteinGym large-scale mutation data.

    Handles downloading, caching, and loading of ProteinGym
    substitution benchmark data.
    """

    # ProteinGym download URLs (Harvard/Marks Lab) - v1.3
    BASE_URL = "https://marks.hms.harvard.edu/proteingym"
    VERSION = "v1.3"
    SUBSTITUTIONS_URL = f"{BASE_URL}/ProteinGym_{VERSION}/DMS_ProteinGym_substitutions.zip"

    def __init__(self, data_dir: Path | None = None):
        """Initialize loader.

        Args:
            data_dir: Directory for data files
        """
        if data_dir is None:
            data_dir = Path(__file__).parents[4] / "data" / "bioinformatics" / "ddg" / "proteingym"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download(self, force: bool = False) -> bool:
        """Download ProteinGym substitutions data.

        Args:
            force: Force re-download even if exists

        Returns:
            True if successful
        """
        zip_path = self.data_dir / "ProteinGym_substitutions.zip"

        if zip_path.exists() and not force:
            print(f"ProteinGym data already exists at {zip_path}")
            return True

        print("Downloading ProteinGym substitutions (~1GB)...")
        print(f"URL: {self.SUBSTITUTIONS_URL}")

        try:
            urllib.request.urlretrieve(
                self.SUBSTITUTIONS_URL,
                zip_path,
                reporthook=self._download_progress,
            )
            print("\nDownload complete!")

            # Extract
            import zipfile

            print("Extracting...")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.data_dir)
            print("Extraction complete!")

            return True

        except Exception as e:
            print(f"Download failed: {e}")
            return False

    @staticmethod
    def _download_progress(block_num, block_size, total_size):
        """Progress callback for downloads."""
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            print(f"\rProgress: {percent:.1f}% ({downloaded / 1e6:.1f} MB)", end="")

    def list_proteins(self) -> list[str]:
        """List available protein datasets."""
        csv_files = list(self.data_dir.glob("**/*.csv"))
        return sorted(set(f.stem.split("_")[0] for f in csv_files))

    def load_protein(
        self,
        protein_id: str,
        single_point_only: bool = True,
    ) -> list[ProteinGymRecord]:
        """Load records for a specific protein.

        Args:
            protein_id: Protein identifier
            single_point_only: Only load single-point mutations

        Returns:
            List of ProteinGymRecord objects
        """
        # Find matching CSV file
        matches = list(self.data_dir.glob(f"**/*{protein_id}*.csv"))
        if not matches:
            raise FileNotFoundError(f"No data found for protein: {protein_id}")

        records = []
        for csv_file in matches:
            with open(csv_file, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    mutation = row.get("mutant") or row.get("mutation")
                    if not mutation:
                        continue

                    if single_point_only and ":" in mutation:
                        continue

                    try:
                        fitness = float(row.get("DMS_score") or row.get("fitness") or 0)
                    except ValueError:
                        continue

                    records.append(
                        ProteinGymRecord(
                            protein_id=protein_id,
                            mutation=mutation,
                            fitness=fitness,
                            sequence=row.get("sequence"),
                        )
                    )

        return records

    # Proteins with non-standard DMS assay scales (binding affinities, enzyme kinetics, etc.)
    # These have scores in thousands/millions instead of typical log-enrichment ratios
    NON_STANDARD_PROTEINS = frozenset(
        [
            "B2L11",  # Binding affinity (values ~10^7)
            "D7PM05",  # Somermeyer_2022 enzyme kinetics (values ~10^4)
            "Q6WV12",  # Somermeyer_2022 enzyme kinetics (values ~10^4)
            "Q8WTC7",  # Somermeyer_2022 enzyme kinetics (values ~10^4)
            "KCNH2",  # Ion channel function (values 0-130)
            "CCDB",  # Toxicity survival (values -87 to -1)
            "SCN5A",  # Channel function (values -205 to -10)
        ]
    )

    def load_all(
        self,
        max_per_protein: int = 1000,
        single_point_only: bool = True,
        filter_non_standard: bool = True,
    ) -> list[ProteinGymRecord]:
        """Load all available records.

        Args:
            max_per_protein: Maximum mutations per protein
            single_point_only: Only load single-point mutations
            filter_non_standard: Filter out proteins with non-standard DMS scales

        Returns:
            List of ProteinGymRecord objects
        """
        all_records = []

        for protein_id in self.list_proteins():
            # Skip proteins with non-standard assay scales
            if filter_non_standard and protein_id in self.NON_STANDARD_PROTEINS:
                continue

            try:
                records = self.load_protein(protein_id, single_point_only)

                # Additional runtime check for score range
                if filter_non_standard and records:
                    fitness_vals = [r.fitness for r in records]
                    if max(fitness_vals) > 20 or min(fitness_vals) < -20:
                        continue  # Skip proteins with extreme values

                # Sample if too many
                if len(records) > max_per_protein:
                    np.random.shuffle(records)
                    records = records[:max_per_protein]
                all_records.extend(records)
            except Exception:
                continue

        return all_records

    def create_dataset(
        self,
        records: list[ProteinGymRecord] | None = None,
        aa_embeddings: dict[str, torch.Tensor] | None = None,
        curvature: float = 1.0,
        use_fitness_as_label: bool = True,
        max_records: int = 100000,
    ) -> ProteinGymDataset:
        """Create PyTorch dataset.

        Args:
            records: Records to use (loads all if None)
            aa_embeddings: AA embeddings for hyperbolic features
            curvature: Poincaré ball curvature
            use_fitness_as_label: Use fitness instead of DDG
            max_records: Maximum number of records

        Returns:
            ProteinGymDataset
        """
        if records is None:
            records = self.load_all()

        if len(records) > max_records:
            np.random.shuffle(records)
            records = records[:max_records]

        return ProteinGymDataset(
            records=records,
            aa_embeddings=aa_embeddings,
            curvature=curvature,
            use_fitness_as_label=use_fitness_as_label,
        )

    def create_streaming_dataset(
        self,
        aa_embeddings: dict[str, torch.Tensor] | None = None,
        curvature: float = 1.0,
    ) -> ProteinGymIterableDataset:
        """Create streaming dataset for large-scale training.

        Args:
            aa_embeddings: AA embeddings for hyperbolic features
            curvature: Poincaré ball curvature

        Returns:
            ProteinGymIterableDataset
        """
        return ProteinGymIterableDataset(
            data_dir=self.data_dir,
            aa_embeddings=aa_embeddings,
            curvature=curvature,
        )

    def get_statistics(self, records: list[ProteinGymRecord]) -> dict:
        """Get statistics for records."""
        fitness_values = [r.fitness for r in records]
        proteins = set(r.protein_id for r in records)

        return {
            "n_records": len(records),
            "n_proteins": len(proteins),
            "fitness_mean": float(np.mean(fitness_values)) if fitness_values else 0,
            "fitness_std": float(np.std(fitness_values)) if fitness_values else 0,
            "fitness_min": float(min(fitness_values)) if fitness_values else 0,
            "fitness_max": float(max(fitness_values)) if fitness_values else 0,
        }


__all__ = [
    "ProteinGymRecord",
    "ProteinGymDataset",
    "ProteinGymIterableDataset",
    "ProteinGymLoader",
]
