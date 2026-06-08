"""Base classes for multi-organism data loading.

Provides abstract base classes and common utilities for loading
biological sequences from various organisms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
import torch


class OrganismType(Enum):
    """Supported organism types."""

    # Viruses
    HIV = auto()
    HBV = auto()
    HCV = auto()
    INFLUENZA = auto()
    SARS_COV_2 = auto()
    DENGUE = auto()
    ZIKA = auto()

    # Bacteria
    TB = auto()  # Mycobacterium tuberculosis
    MRSA = auto()  # Staphylococcus aureus
    ECOLI = auto()  # Escherichia coli
    STREP = auto()  # Streptococcus

    # Parasites
    MALARIA = auto()  # Plasmodium falciparum

    # Proteins (organism-independent)
    ANTIBODY = auto()
    TCR = auto()
    KINASE = auto()
    GPCR = auto()


class SequenceType(Enum):
    """Types of biological sequences."""

    DNA = auto()
    RNA = auto()
    PROTEIN = auto()
    CODON = auto()


@dataclass
class SequenceRecord:
    """A biological sequence record with metadata."""

    id: str
    sequence: str
    organism: OrganismType
    sequence_type: SequenceType

    # Optional metadata
    gene: str | None = None
    protein: str | None = None
    subtype: str | None = None
    country: str | None = None
    year: int | None = None
    host: str | None = None

    # Phenotype labels (for validation)
    drug_resistance: dict[str, float] | None = None
    immune_escape: dict[str, float] | None = None
    tropism: str | None = None
    fitness: float | None = None

    # Additional annotations
    mutations: list[str] | None = None
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_tensor(self, encoding: str = "onehot") -> torch.Tensor:
        """Convert sequence to tensor representation."""
        if encoding == "onehot":
            return self._encode_onehot()
        elif encoding == "ordinal":
            return self._encode_ordinal()
        else:
            raise ValueError(f"Unknown encoding: {encoding}")

    def _encode_onehot(self) -> torch.Tensor:
        """One-hot encode the sequence."""
        if self.sequence_type == SequenceType.DNA:
            alphabet = "ACGT"
        elif self.sequence_type == SequenceType.RNA:
            alphabet = "ACGU"
        elif self.sequence_type == SequenceType.PROTEIN:
            alphabet = "ACDEFGHIKLMNPQRSTVWY"
        else:
            raise ValueError(f"Cannot one-hot encode {self.sequence_type}")

        char_to_idx = {c: i for i, c in enumerate(alphabet)}
        n_chars = len(alphabet)

        encoded = np.zeros((len(self.sequence), n_chars), dtype=np.float32)
        for i, char in enumerate(self.sequence.upper()):
            if char in char_to_idx:
                encoded[i, char_to_idx[char]] = 1.0
            # Unknown characters left as zeros

        return torch.from_numpy(encoded)

    def _encode_ordinal(self) -> torch.Tensor:
        """Ordinal encode the sequence."""
        if self.sequence_type == SequenceType.DNA:
            alphabet = "ACGT"
        elif self.sequence_type == SequenceType.RNA:
            alphabet = "ACGU"
        elif self.sequence_type == SequenceType.PROTEIN:
            alphabet = "ACDEFGHIKLMNPQRSTVWY"
        else:
            raise ValueError(f"Cannot ordinal encode {self.sequence_type}")

        char_to_idx = {c: i for i, c in enumerate(alphabet)}

        encoded = np.zeros(len(self.sequence), dtype=np.int64)
        for i, char in enumerate(self.sequence.upper()):
            encoded[i] = char_to_idx.get(char, -1)

        return torch.from_numpy(encoded)


class OrganismLoader(ABC):
    """Abstract base class for organism-specific data loaders."""

    def __init__(
        self,
        organism: OrganismType,
        cache_dir: Path | None = None,
        max_sequences: int | None = None,
    ):
        self.organism = organism
        self.cache_dir = cache_dir or Path("data/cache")
        self.max_sequences = max_sequences
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def load_sequences(self) -> list[SequenceRecord]:
        """Load sequences for this organism."""
        pass

    @abstractmethod
    def get_validation_labels(self) -> dict[str, Any]:
        """Get phenotype labels for validation."""
        pass

    def load_encoded(
        self,
        encoding: str = "onehot",
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load sequences as encoded tensors.

        Args:
            encoding: Encoding type ("onehot", "ordinal", "padic")
            max_length: Maximum sequence length (pad/truncate)

        Returns:
            Tuple of (sequences, labels) tensors
        """
        records = self.load_sequences()

        if max_length is None:
            max_length = max(len(r.sequence) for r in records)

        # Encode sequences
        encoded = []
        labels = []

        for record in records:
            tensor = record.to_tensor(encoding)

            # Pad or truncate
            if tensor.shape[0] < max_length:
                padding = torch.zeros(max_length - tensor.shape[0], tensor.shape[1])
                tensor = torch.cat([tensor, padding], dim=0)
            elif tensor.shape[0] > max_length:
                tensor = tensor[:max_length]

            encoded.append(tensor)

            # Get label (if available)
            if record.drug_resistance:
                labels.append(list(record.drug_resistance.values())[0])
            elif record.fitness is not None:
                labels.append(record.fitness)
            else:
                labels.append(0.0)

        sequences = torch.stack(encoded)
        labels = torch.tensor(labels, dtype=torch.float32)

        return sequences, labels

    def load_padic_encoded(
        self,
        prime: int = 3,
        latent_dim: int = 16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Load sequences with p-adic encoding.

        Uses the codon encoder to create p-adic embeddings.

        Args:
            prime: Prime for p-adic valuation (3 for codons)
            latent_dim: Embedding dimension

        Returns:
            Tuple of (embeddings, labels) tensors
        """
        records = self.load_sequences()

        # Use codon encoder if available
        try:
            from src.encoders.codon_encoder import CodonEncoder

            encoder = CodonEncoder(
                embedding_dim=latent_dim,
                use_padic=True,
                padic_prime=prime,
            )

            embeddings = []
            labels = []

            for record in records:
                # Convert to codons if DNA/RNA
                if record.sequence_type in [SequenceType.DNA, SequenceType.RNA]:
                    seq = record.sequence.upper().replace("U", "T")
                    # Ensure divisible by 3
                    seq = seq[: len(seq) // 3 * 3]
                    embedding = encoder.encode_sequence(seq)
                else:
                    # Protein sequence - use ordinal encoding
                    embedding = record.to_tensor("ordinal").float()

                embeddings.append(embedding)

                if record.drug_resistance:
                    labels.append(list(record.drug_resistance.values())[0])
                elif record.fitness is not None:
                    labels.append(record.fitness)
                else:
                    labels.append(0.0)

            return torch.stack(embeddings), torch.tensor(labels)

        except ImportError:
            # Fallback to simple ordinal encoding
            return self.load_encoded("ordinal")

    def compute_padic_distances(
        self,
        records: list[SequenceRecord],
        prime: int = 3,
    ) -> np.ndarray:
        """Compute pairwise p-adic distances between sequences.

        Args:
            records: List of sequence records
            prime: Prime for p-adic valuation

        Returns:
            Distance matrix (n x n)
        """
        n = len(records)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = self._padic_sequence_distance(
                    records[i].sequence,
                    records[j].sequence,
                    prime,
                )
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def _padic_sequence_distance(
        self,
        seq1: str,
        seq2: str,
        prime: int,
    ) -> float:
        """Compute p-adic distance between two sequences."""
        # Align sequences (simple - same length assumed)
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]

        # Count differences by position
        differences = []
        for i, (c1, c2) in enumerate(zip(seq1, seq2, strict=False)):
            if c1 != c2:
                differences.append(i)

        if not differences:
            return 0.0

        # P-adic distance based on first difference position
        # (captures hierarchical position importance)
        first_diff = differences[0]

        # Valuation based on position divisibility
        valuation = 0
        pos = first_diff + 1  # 1-indexed
        while pos % prime == 0:
            pos //= prime
            valuation += 1

        return prime ** (-valuation)

    def get_statistics(self) -> dict[str, Any]:
        """Get dataset statistics."""
        records = self.load_sequences()

        lengths = [len(r.sequence) for r in records]
        subtypes = [r.subtype for r in records if r.subtype]
        countries = [r.country for r in records if r.country]

        return {
            "organism": self.organism.name,
            "n_sequences": len(records),
            "avg_length": np.mean(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "n_subtypes": len(set(subtypes)),
            "n_countries": len(set(countries)),
            "has_drug_resistance": sum(1 for r in records if r.drug_resistance),
            "has_immune_escape": sum(1 for r in records if r.immune_escape),
        }


class MockOrganismLoader(OrganismLoader):
    """Mock loader for testing with synthetic data."""

    def __init__(
        self,
        organism: OrganismType,
        n_sequences: int = 100,
        seq_length: int = 300,
        **kwargs,
    ):
        super().__init__(organism, **kwargs)
        self.n_sequences = n_sequences
        self.seq_length = seq_length

    def load_sequences(self) -> list[SequenceRecord]:
        """Generate mock sequences."""
        np.random.seed(42)

        alphabet = "ACGT"
        records = []

        for i in range(self.n_sequences):
            seq = "".join(np.random.choice(list(alphabet), self.seq_length))
            records.append(
                SequenceRecord(
                    id=f"mock_{i}",
                    sequence=seq,
                    organism=self.organism,
                    sequence_type=SequenceType.DNA,
                    drug_resistance={"generic": np.random.random()},
                )
            )

        return records

    def get_validation_labels(self) -> dict[str, Any]:
        """Return mock labels."""
        return {
            "drug_resistance": {f"mock_{i}": np.random.random() for i in range(self.n_sequences)},
        }
