# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Unified dataset registry for DDG prediction.

Provides a single interface to access all DDG-related datasets
with consistent preprocessing and feature extraction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import ConcatDataset, Dataset

from src.bioinformatics.data.proteingym_loader import (
    ProteinGymLoader,
)
from src.bioinformatics.data.protherm_loader import (
    ProThermLoader,
)
from src.bioinformatics.data.s669_loader import (
    S669Loader,
)

DatasetType = Literal["protherm", "s669", "proteingym", "combined"]


@dataclass
class DatasetInfo:
    """Information about a dataset."""

    name: str
    n_samples: int
    n_proteins: int
    feature_dim: int
    label_type: str  # "ddg" or "fitness"
    source: str
    statistics: dict = field(default_factory=dict)


class DatasetRegistry:
    """Unified registry for all DDG-related datasets.

    Provides consistent access to:
    - ProTherm: High-quality curated mutations (N=176+)
    - S669: Benchmark dataset (N=669)
    - ProteinGym: Large-scale diverse mutations (N=500K+)

    Usage:
        registry = DatasetRegistry()
        dataset = registry.get_dataset("protherm", aa_embeddings=embeddings)
        combined = registry.get_combined_dataset(["protherm", "s669"])
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        aa_embeddings: dict[str, torch.Tensor] | None = None,
        curvature: float = 1.0,
    ):
        """Initialize registry.

        Args:
            data_dir: Base directory for all data
            aa_embeddings: Shared AA embeddings for all datasets
            curvature: Poincaré ball curvature
        """
        if data_dir is None:
            data_dir = Path(__file__).parents[4] / "data" / "bioinformatics" / "ddg"
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.aa_embeddings = aa_embeddings
        self.curvature = curvature

        # Initialize loaders
        self._protherm_loader = ProThermLoader(self.data_dir / "protherm")
        self._s669_loader = S669Loader(self.data_dir / "s669")
        self._proteingym_loader = ProteinGymLoader(self.data_dir / "proteingym")

        # Cache loaded datasets
        self._cache: dict[str, Dataset] = {}

    def get_dataset(
        self,
        name: DatasetType,
        use_cache: bool = True,
        **kwargs,
    ) -> Dataset:
        """Get a dataset by name.

        Args:
            name: Dataset name ("protherm", "s669", "proteingym")
            use_cache: Use cached dataset if available
            **kwargs: Additional arguments passed to loader

        Returns:
            PyTorch Dataset
        """
        cache_key = f"{name}_{hash(str(kwargs))}"

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        if name == "protherm":
            dataset = self._protherm_loader.create_dataset(
                aa_embeddings=self.aa_embeddings,
                curvature=self.curvature,
                **kwargs,
            )
        elif name == "s669":
            dataset = self._s669_loader.create_dataset(
                aa_embeddings=self.aa_embeddings,
                curvature=self.curvature,
                **kwargs,
            )
        elif name == "proteingym":
            dataset = self._proteingym_loader.create_dataset(
                aa_embeddings=self.aa_embeddings,
                curvature=self.curvature,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown dataset: {name}")

        if use_cache:
            self._cache[cache_key] = dataset

        return dataset

    def get_combined_dataset(
        self,
        names: list[DatasetType],
        **kwargs,
    ) -> ConcatDataset:
        """Get a combined dataset from multiple sources.

        Args:
            names: List of dataset names to combine
            **kwargs: Arguments passed to each loader

        Returns:
            ConcatDataset containing all specified datasets
        """
        datasets = [self.get_dataset(name, **kwargs) for name in names]
        return ConcatDataset(datasets)

    def get_info(self, name: DatasetType) -> DatasetInfo:
        """Get information about a dataset.

        Args:
            name: Dataset name

        Returns:
            DatasetInfo object
        """
        dataset = self.get_dataset(name)

        if name == "protherm":
            db = self._protherm_loader.load_or_create()
            stats = db.get_statistics()
            n_proteins = stats["n_proteins"]
            label_type = "ddg"
            source = "ProTherm Database (curated)"
        elif name == "s669":
            records = self._s669_loader.load_from_csv()
            stats = self._s669_loader.get_statistics(records)
            n_proteins = stats["n_proteins"]
            label_type = "ddg"
            source = "S669 Benchmark (Pucci et al.)"
        elif name == "proteingym":
            records = self._proteingym_loader.load_all(max_per_protein=100)
            stats = self._proteingym_loader.get_statistics(records)
            n_proteins = stats["n_proteins"]
            label_type = "fitness"
            source = "ProteinGym (Marks Lab)"
        else:
            raise ValueError(f"Unknown dataset: {name}")

        return DatasetInfo(
            name=name,
            n_samples=len(dataset),
            n_proteins=n_proteins,
            feature_dim=dataset.feature_dim if hasattr(dataset, "feature_dim") else 0,
            label_type=label_type,
            source=source,
            statistics=stats,
        )

    def list_datasets(self) -> list[str]:
        """List available datasets."""
        return ["protherm", "s669", "proteingym"]

    def set_embeddings(
        self,
        aa_embeddings: dict[str, torch.Tensor],
        curvature: float = 1.0,
    ) -> None:
        """Update AA embeddings and clear cache.

        Args:
            aa_embeddings: New AA embeddings
            curvature: Poincaré ball curvature
        """
        self.aa_embeddings = aa_embeddings
        self.curvature = curvature
        self._cache.clear()

    def load_embeddings_from_file(
        self,
        path: Path,
        device: str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Load AA embeddings from file.

        Supports both PyTorch (.pt) and JSON formats.

        Args:
            path: Path to embeddings file
            device: Device to load to

        Returns:
            Dictionary of AA embeddings
        """
        path = Path(path)

        if path.suffix == ".pt":
            data = torch.load(path, map_location=device, weights_only=True)
            embeddings = data["embeddings"] if isinstance(data, dict) and "embeddings" in data else data
        elif path.suffix == ".json":
            with open(path) as f:
                data = json.load(f)
            embeddings = {aa: torch.tensor(emb, device=device, dtype=torch.float32) for aa, emb in data.items()}
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        self.set_embeddings(embeddings)
        return embeddings

    def get_train_val_split(
        self,
        name: DatasetType,
        val_ratio: float = 0.2,
        seed: int = 42,
        **kwargs,
    ) -> tuple[Dataset, Dataset]:
        """Get train/validation split for a dataset.

        Args:
            name: Dataset name
            val_ratio: Validation set ratio
            seed: Random seed
            **kwargs: Arguments passed to loader

        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        from torch.utils.data import random_split

        dataset = self.get_dataset(name, **kwargs)
        n_val = int(len(dataset) * val_ratio)
        n_train = len(dataset) - n_val

        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=generator)

        return train_dataset, val_dataset

    def save_manifest(self, path: Path | None = None) -> None:
        """Save dataset manifest with all info.

        Args:
            path: Output path (default: data_dir/DATASET_MANIFEST.json)
        """
        if path is None:
            path = self.data_dir / "DATASET_MANIFEST.json"

        manifest = {
            "version": "1.0",
            "datasets": {},
        }

        for name in self.list_datasets():
            try:
                info = self.get_info(name)
                manifest["datasets"][name] = {
                    "n_samples": info.n_samples,
                    "n_proteins": info.n_proteins,
                    "feature_dim": info.feature_dim,
                    "label_type": info.label_type,
                    "source": info.source,
                    "statistics": info.statistics,
                }
            except Exception as e:
                manifest["datasets"][name] = {"error": str(e)}

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)


__all__ = [
    "DatasetType",
    "DatasetInfo",
    "DatasetRegistry",
]
