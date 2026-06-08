# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""S669 benchmark dataset loader.

The S669 dataset is a standard benchmark for DDG prediction methods,
containing 669 single-point mutations with experimental stability data.

Reference: Pucci et al. (2018) - Quantification of biases in predictions
           of protein stability changes upon mutations
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.bioinformatics.data.preprocessing import (
    compute_features,
)


@dataclass
class S669Record:
    """Container for an S669 mutation record."""

    pdb_id: str
    chain: str
    position: int
    wild_type: str
    mutant: str
    ddg: float  # kcal/mol
    uniprot_id: str | None = None
    protein_name: str | None = None

    @property
    def mutation_string(self) -> str:
        """Standard mutation notation."""
        return f"{self.wild_type}{self.position}{self.mutant}"

    @property
    def full_id(self) -> str:
        """Full mutation ID."""
        return f"{self.pdb_id}_{self.chain}_{self.mutation_string}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> S669Record:
        return cls(**data)


class S669Dataset(Dataset):
    """PyTorch Dataset for S669 mutations."""

    def __init__(
        self,
        records: list[S669Record],
        aa_embeddings: dict[str, torch.Tensor] | None = None,
        curvature: float = 1.0,
    ):
        """Initialize dataset.

        Args:
            records: List of S669Record objects
            aa_embeddings: Optional AA embeddings for hyperbolic features
            curvature: Poincaré ball curvature
        """
        self.records = records
        self.aa_embeddings = aa_embeddings
        self.curvature = curvature

        self._features = []
        self._labels = []
        self._compute_all_features()

    def _compute_all_features(self) -> None:
        """Compute features for all records."""
        from src.bioinformatics.data.preprocessing import add_hyperbolic_features

        for record in self.records:
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
            self._labels.append(record.ddg)

        self._features = np.array(self._features, dtype=np.float32)
        self._labels = np.array(self._labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.records)

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


class S669Loader:
    """Loader for S669 benchmark dataset.

    Provides the standard S669 benchmark used for fair comparison
    with literature methods like ESM-1v, FoldX, Rosetta.
    """

    def __init__(self, data_dir: Path | None = None):
        """Initialize loader.

        Args:
            data_dir: Directory for data files
        """
        if data_dir is None:
            # Go up from src/bioinformatics/data/s669_loader.py to repo root (3 levels)
            data_dir = Path(__file__).parents[3] / "data" / "bioinformatics" / "ddg" / "s669"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing S669 file in partner package
        # Go up from src/bioinformatics/data/s669_loader.py to repo root
        self._partner_path = (
            Path(__file__).parents[3]
            / "deliverables"
            / "partners"
            / "protein_stability_ddg"
            / "reproducibility"
            / "data"
            / "s669.csv"
        )

    def load_from_csv(self, csv_path: Path | None = None) -> list[S669Record]:
        """Load S669 records from CSV file.

        Args:
            csv_path: Path to CSV file (uses default if None)

        Returns:
            List of S669Record objects
        """
        if csv_path is None:
            # Try full dataset first, then fall back to curated subset
            full_path = self.data_dir / "s669_full.csv"
            if full_path.exists():
                csv_path = full_path
            elif self._partner_path.exists():
                csv_path = self._partner_path
            else:
                csv_path = full_path  # Will raise FileNotFoundError below

        if not csv_path.exists():
            raise FileNotFoundError(
                f"S669 dataset not found at {csv_path}. Download from DDG-EMB or use download_s669()."
            )

        records = []
        with open(csv_path, newline="") as f:
            # Try to detect delimiter
            sample = f.read(1024)
            f.seek(0)
            delimiter = "," if "," in sample else "\t"

            reader = csv.DictReader(f, delimiter=delimiter)

            for row in reader:
                try:
                    # Handle different column naming conventions
                    pdb = row.get("pdb_id") or row.get("PDB") or row.get("pdb")
                    chain = row.get("chain") or row.get("Chain") or "A"
                    position = int(row.get("position") or row.get("Position") or row.get("pos"))
                    wt = (row.get("wild_type") or row.get("WT") or row.get("wt")).upper()
                    mut = (row.get("mutant") or row.get("MUT") or row.get("mut")).upper()
                    ddg = float(row.get("ddg_exp") or row.get("ddg") or row.get("DDG"))

                    # Validate amino acids
                    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
                    if wt not in valid_aa or mut not in valid_aa:
                        continue

                    records.append(
                        S669Record(
                            pdb_id=pdb,
                            chain=chain,
                            position=position,
                            wild_type=wt,
                            mutant=mut,
                            ddg=ddg,
                            uniprot_id=row.get("uniprot_id"),
                            protein_name=row.get("protein_name"),
                        )
                    )
                except (ValueError, KeyError, TypeError):
                    continue

        return records

    def load_curated_subset(
        self,
        csv_path: Path | None = None,
        n_samples: int = 52,
    ) -> list[S669Record]:
        """Load curated N=52 subset for comparison with ProTherm.

        This subset contains well-characterized alanine-scanning mutations
        for fair comparison with the ProTherm dataset.

        Args:
            csv_path: Path to full S669 CSV
            n_samples: Number of samples to include

        Returns:
            List of curated S669Record objects
        """
        all_records = self.load_from_csv(csv_path)

        # Filter for alanine-scanning mutations (most reliable)
        ala_scan = [r for r in all_records if r.mutant == "A"]

        # Sort by DDG variance (prefer extremes for better signal)
        ala_scan.sort(key=lambda r: abs(r.ddg), reverse=True)

        # Take top n_samples with good spread
        selected = []
        seen_proteins = set()
        for r in ala_scan:
            if len(selected) >= n_samples:
                break
            # Ensure protein diversity
            if len([s for s in selected if s.pdb_id == r.pdb_id]) < 5:
                selected.append(r)
                seen_proteins.add(r.pdb_id)

        return selected

    def get_statistics(self, records: list[S669Record]) -> dict:
        """Get statistics for a set of records."""
        ddg_values = [r.ddg for r in records]
        proteins = set(r.pdb_id for r in records)

        return {
            "n_records": len(records),
            "n_proteins": len(proteins),
            "ddg_mean": float(np.mean(ddg_values)) if ddg_values else 0,
            "ddg_std": float(np.std(ddg_values)) if ddg_values else 0,
            "ddg_min": float(min(ddg_values)) if ddg_values else 0,
            "ddg_max": float(max(ddg_values)) if ddg_values else 0,
            "n_destabilizing": sum(1 for d in ddg_values if d > 1.0),
            "n_neutral": sum(1 for d in ddg_values if -1.0 <= d <= 1.0),
            "n_stabilizing": sum(1 for d in ddg_values if d < -1.0),
        }

    def create_dataset(
        self,
        records: list[S669Record] | None = None,
        aa_embeddings: dict[str, torch.Tensor] | None = None,
        curvature: float = 1.0,
        use_curated_subset: bool = False,
    ) -> S669Dataset:
        """Create PyTorch dataset.

        Args:
            records: Records to use (loads full if None)
            aa_embeddings: AA embeddings for hyperbolic features
            curvature: Poincaré ball curvature
            use_curated_subset: Use N=52 curated subset

        Returns:
            S669Dataset
        """
        if records is None:
            records = self.load_curated_subset() if use_curated_subset else self.load_from_csv()

        return S669Dataset(
            records=records,
            aa_embeddings=aa_embeddings,
            curvature=curvature,
        )

    def save_records(self, records: list[S669Record], path: Path) -> None:
        """Save records to JSON file."""
        data = [r.to_dict() for r in records]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_records(self, path: Path) -> list[S669Record]:
        """Load records from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return [S669Record.from_dict(d) for d in data]


__all__ = [
    "S669Record",
    "S669Dataset",
    "S669Loader",
]
