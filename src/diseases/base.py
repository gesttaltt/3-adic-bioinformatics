# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Base classes for multi-disease framework."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from torch.utils.data import Dataset


class DiseaseType(Enum):
    """Types of diseases/pathogens supported."""

    VIRAL = "viral"
    AUTOIMMUNE = "autoimmune"
    NEURODEGENERATIVE = "neurodegenerative"
    CANCER = "cancer"
    BACTERIAL = "bacterial"
    PARASITIC = "parasitic"
    FUNGAL = "fungal"
    ONCOLOGY = "oncology"


class TaskType(Enum):
    """Types of prediction tasks."""

    RESISTANCE = "resistance"
    ESCAPE = "escape"
    BINDING = "binding"
    AGGREGATION = "aggregation"
    PTM = "ptm"  # Post-translational modification
    EXPRESSION = "expression"
    STABILITY = "stability"
    IMMUNOGENICITY = "immunogenicity"
    FITNESS = "fitness"
    VACCINE = "vaccine"
    ANTIGENICITY = "antigenicity"


@dataclass
class DiseaseConfig:
    """Configuration for a disease domain.

    Attributes:
        name: Short identifier (e.g., "hiv", "ra", "alzheimers")
        display_name: Human-readable name
        disease_type: Category of disease
        tasks: List of prediction tasks for this disease
        data_sources: Dictionary of data source names to paths/URLs
        loss_weights: Weights for different loss components
        special_losses: List of disease-specific loss function names
        codon_features: Features to extract from codon sequences
        structure_features: Features from protein structures
        api_endpoints: External API endpoints for data/validation
    """

    name: str
    display_name: str
    disease_type: DiseaseType
    tasks: list[TaskType]
    data_sources: dict[str, str] = field(default_factory=dict)
    loss_weights: dict[str, float] = field(default_factory=dict)
    special_losses: list[str] = field(default_factory=list)
    codon_features: list[str] = field(default_factory=list)
    structure_features: list[str] = field(default_factory=list)
    api_endpoints: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_loss_weight(self, loss_name: str, default: float = 1.0) -> float:
        """Get weight for a specific loss component."""
        return self.loss_weights.get(loss_name, default)

    def has_task(self, task: TaskType) -> bool:
        """Check if disease has a specific task."""
        return task in self.tasks


class DiseaseDataset(Dataset, ABC):
    """Abstract base class for disease-specific datasets.

    Subclasses must implement:
    - __len__: Return number of samples
    - __getitem__: Return a sample dict with required keys
    - load_data: Load data from sources
    """

    def __init__(
        self,
        config: DiseaseConfig,
        split: str = "train",
        transform: Callable | None = None,
    ):
        """Initialize dataset.

        Args:
            config: Disease configuration
            split: Data split ("train", "val", "test")
            transform: Optional transform to apply to samples
        """
        self.config = config
        self.split = split
        self.transform = transform
        self.samples = []
        self.load_data()

    @abstractmethod
    def load_data(self) -> None:
        """Load data from configured sources."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return number of samples."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sample.

        Returns:
            Dictionary with at least:
            - "sequence": Tensor of codon indices or embeddings
            - "labels": Task-specific labels
            - "disease": Disease name string
        """
        pass

    def get_sample_weights(self) -> torch.Tensor | None:
        """Get per-sample weights for imbalanced datasets."""
        return None


class DiseasePredictor(torch.nn.Module, ABC):
    """Abstract base class for disease-specific prediction heads."""

    def __init__(self, config: DiseaseConfig, input_dim: int):
        """Initialize predictor.

        Args:
            config: Disease configuration
            input_dim: Dimension of input embeddings
        """
        super().__init__()
        self.config = config
        self.input_dim = input_dim

    @abstractmethod
    def forward(self, embeddings: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            embeddings: Input embeddings (batch, seq_len, dim) or (batch, dim)
            **kwargs: Task-specific inputs

        Returns:
            Dictionary of task predictions
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute task losses.

        Returns:
            Dictionary of loss values by task
        """
        pass


class DiseaseAnalyzer(ABC):
    """Abstract base class for disease-specific analysis."""

    def __init__(self, config: DiseaseConfig):
        """Initialize analyzer.

        Args:
            config: Disease configuration
        """
        self.config = config

    @abstractmethod
    def analyze(
        self,
        sequences: list[str],
        embeddings: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Run disease-specific analysis.

        Args:
            sequences: List of sequences to analyze
            embeddings: Optional precomputed embeddings
            **kwargs: Analysis-specific parameters

        Returns:
            Dictionary of analysis results
        """
        pass

    @abstractmethod
    def validate_predictions(
        self,
        predictions: dict[str, torch.Tensor],
        ground_truth: dict[str, Any],
    ) -> dict[str, float]:
        """Validate predictions against ground truth.

        Returns:
            Dictionary of validation metrics
        """
        pass
