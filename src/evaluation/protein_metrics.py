# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""ProteinGym-Style Evaluation Metrics for Sequence Generation.

This module implements comprehensive evaluation metrics for generated
biological sequences, inspired by:

- ProteinGym: Large-scale benchmark for protein fitness prediction
- ProteinBench: Evaluation framework for protein design
- RFdiffusion: Structure-conditioned design metrics
- EvoDiff: Sequence-based diffusion evaluation

Metric Categories:
1. Quality: How good are individual sequences?
2. Novelty: How different from training data?
3. Diversity: How varied is the generated set?
4. Biological Validity: Are sequences biologically realistic?

References:
- Notin et al. (2023): ProteinGym benchmark
- Watson et al. (2023): RFdiffusion evaluation
- Alamdari et al. (2023): EvoDiff metrics
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from src.biology.codons import GENETIC_CODE, codon_index_to_triplet
from src.losses.codon_usage import CodonOptimalityScore, Organism


@dataclass
class QualityMetrics:
    """Quality metrics for individual sequences."""

    # Reconstruction
    reconstruction_accuracy: float = 0.0
    per_position_accuracy: float = 0.0

    # Structure (requires AlphaFold or similar)
    plddt_mean: float = 0.0
    plddt_min: float = 0.0
    tm_score: float = 0.0

    # Codon optimality
    mean_tai: float = 0.0
    mean_cai: float = 0.0


@dataclass
class NoveltyMetrics:
    """Novelty metrics measuring distance from training data."""

    # Sequence identity
    min_seq_identity: float = 1.0  # Closest sequence in training
    mean_seq_identity: float = 1.0
    max_seq_identity: float = 1.0

    # Unique sequences
    unique_fraction: float = 0.0  # Fraction of unique sequences in batch
    novel_fraction: float = 0.0  # Fraction not in training set


@dataclass
class DiversityMetrics:
    """Diversity metrics for a set of generated sequences."""

    # Pairwise diversity
    mean_pairwise_distance: float = 0.0
    min_pairwise_distance: float = 0.0

    # Cluster-based
    n_clusters: int = 0
    cluster_entropy: float = 0.0

    # Coverage
    amino_acid_coverage: float = 0.0  # Fraction of AA types used
    codon_coverage: float = 0.0  # Fraction of codon types used


@dataclass
class BiologicalValidityMetrics:
    """Biological validity metrics."""

    # Basic validity
    no_stop_codons: float = 0.0  # Fraction without internal stops
    valid_start_codon: float = 0.0  # Fraction starting with ATG
    valid_length: float = 0.0  # Fraction with valid length (divisible by 3)

    # Advanced validity
    foldable_fraction: float = 0.0  # Fraction that fold correctly
    functional_fraction: float = 0.0  # Fraction with predicted function

    # Conservation
    conserved_positions: float = 0.0  # Conservation at key positions


@dataclass
class GenerationMetrics:
    """Complete metrics for evaluating sequence generation."""

    quality: QualityMetrics = field(default_factory=QualityMetrics)
    novelty: NoveltyMetrics = field(default_factory=NoveltyMetrics)
    diversity: DiversityMetrics = field(default_factory=DiversityMetrics)
    validity: BiologicalValidityMetrics = field(default_factory=BiologicalValidityMetrics)

    # Summary statistics
    n_sequences: int = 0
    mean_length: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert all metrics to flat dictionary for logging."""
        result = {}

        # Quality metrics
        for key, value in vars(self.quality).items():
            result[f"quality/{key}"] = value

        # Novelty metrics
        for key, value in vars(self.novelty).items():
            result[f"novelty/{key}"] = value

        # Diversity metrics
        for key, value in vars(self.diversity).items():
            result[f"diversity/{key}"] = value

        # Validity metrics
        for key, value in vars(self.validity).items():
            result[f"validity/{key}"] = value

        # Summary
        result["n_sequences"] = self.n_sequences
        result["mean_length"] = self.mean_length

        return result


def compute_sequence_identity(
    seq1: str | list[int] | torch.Tensor,
    seq2: str | list[int] | torch.Tensor,
) -> float:
    """Compute sequence identity between two sequences.

    Args:
        seq1, seq2: Sequences as strings, lists, or tensors

    Returns:
        Fraction of identical positions (0-1)
    """
    # Convert to lists if needed
    if isinstance(seq1, torch.Tensor):
        seq1 = seq1.tolist()
    if isinstance(seq2, torch.Tensor):
        seq2 = seq2.tolist()
    if isinstance(seq1, str):
        seq1 = list(seq1)
    if isinstance(seq2, str):
        seq2 = list(seq2)

    # Align by length
    min_len = min(len(seq1), len(seq2))
    if min_len == 0:
        return 0.0

    matches = sum(1 for a, b in zip(seq1[:min_len], seq2[:min_len], strict=False) if a == b)
    return matches / min_len


def compute_pairwise_distances(
    sequences: torch.Tensor,
    metric: str = "hamming",
) -> torch.Tensor:
    """Compute pairwise distances between sequences.

    Args:
        sequences: (n_seq, seq_len) tensor of sequences
        metric: Distance metric ('hamming', 'edit')

    Returns:
        (n_seq, n_seq) distance matrix
    """
    n_seq = sequences.shape[0]
    seq_len = sequences.shape[1]

    if metric == "hamming":
        # Hamming distance: count of different positions
        # Use broadcasting for efficiency
        seq_expanded = sequences.unsqueeze(1)  # (n, 1, len)
        seq_broadcast = sequences.unsqueeze(0)  # (1, n, len)
        distances = (seq_expanded != seq_broadcast).float().sum(dim=-1) / seq_len

    else:
        # Fallback to pairwise computation
        distances = torch.zeros(n_seq, n_seq, device=sequences.device)
        for i in range(n_seq):
            for j in range(i + 1, n_seq):
                dist = 1.0 - compute_sequence_identity(sequences[i], sequences[j])
                distances[i, j] = dist
                distances[j, i] = dist

    return distances


def compute_cluster_entropy(
    sequences: torch.Tensor,
    n_clusters: int = 10,
) -> tuple[float, int]:
    """Compute cluster entropy of sequence set.

    Args:
        sequences: (n_seq, seq_len) tensor
        n_clusters: Number of clusters for k-means

    Returns:
        (entropy, actual_n_clusters)
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        return 0.0, 0

    n_seq = sequences.shape[0]
    if n_seq < n_clusters:
        n_clusters = max(1, n_seq // 2)

    # Convert to numpy for sklearn
    X = sequences.cpu().numpy().astype(np.float32)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    # Compute entropy
    unique, counts = np.unique(labels, return_counts=True)
    probs = counts / counts.sum()
    entropy = -np.sum(probs * np.log(probs + 1e-8))

    # Normalize by max entropy
    max_entropy = np.log(n_clusters)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return float(normalized_entropy), len(unique)


def check_biological_validity(
    codon_sequences: torch.Tensor,
) -> dict[str, float]:
    """Check biological validity of codon sequences.

    Args:
        codon_sequences: (batch, seq_len) codon indices 0-63

    Returns:
        Dictionary of validity metrics
    """
    batch_size = codon_sequences.shape[0]
    device = codon_sequences.device

    results = {
        "no_stop_codons": 0.0,
        "valid_start_codon": 0.0,
        "valid_length": 1.0,  # Assume valid if we're using codon indices
    }

    # Stop codon indices (TAA, TAG, TGA)
    # In our encoding: TAA=48, TAG=50, TGA=56
    stop_indices = []
    for idx in range(64):
        triplet = codon_index_to_triplet(idx)
        if GENETIC_CODE.get(triplet) == "*":
            stop_indices.append(idx)
    stop_indices = torch.tensor(stop_indices, device=device)

    # Check for internal stop codons
    n_valid = 0
    for seq in codon_sequences:
        # Exclude last position (valid stop codon)
        internal = seq[:-1] if len(seq) > 1 else seq
        has_stop = any(codon.item() in stop_indices.tolist() for codon in internal)
        if not has_stop:
            n_valid += 1
    results["no_stop_codons"] = n_valid / batch_size

    # Check start codon (ATG = index for ATG)
    atg_idx = None
    for idx in range(64):
        if codon_index_to_triplet(idx) == "ATG":
            atg_idx = idx
            break

    if atg_idx is not None:
        starts_with_atg = (codon_sequences[:, 0] == atg_idx).float().mean().item()
        results["valid_start_codon"] = starts_with_atg

    return results


class ProteinGymEvaluator:
    """Comprehensive evaluator for generated sequences.

    Computes all ProteinGym-style metrics for a batch of generated sequences.
    """

    def __init__(
        self,
        training_sequences: torch.Tensor | None = None,
        organism: Organism = Organism.HUMAN,
        use_structure_prediction: bool = False,
    ):
        """Initialize evaluator.

        Args:
            training_sequences: (n_train, seq_len) training set for novelty
            organism: Organism for codon optimality
            use_structure_prediction: Whether to use AlphaFold for structure metrics
        """
        self.training_sequences = training_sequences
        self.organism = organism
        self.use_structure = use_structure_prediction

        # Codon optimality scorer
        self.codon_scorer = CodonOptimalityScore(organism=organism)

    def compute_quality_metrics(
        self,
        generated: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> QualityMetrics:
        """Compute quality metrics for generated sequences.

        Args:
            generated: (batch, seq_len) generated codon indices
            targets: (batch, seq_len) target sequences for reconstruction

        Returns:
            QualityMetrics object
        """
        metrics = QualityMetrics()

        # Reconstruction accuracy
        if targets is not None:
            matches = (generated == targets).float()
            metrics.reconstruction_accuracy = matches.mean().item()
            metrics.per_position_accuracy = matches.mean(dim=0).mean().item()

        # Codon optimality
        with torch.no_grad():
            scores = self.codon_scorer(generated)
            metrics.mean_tai = scores["tai"].mean().item()
            metrics.mean_cai = scores["cai"].mean().item()

        # Structure metrics (placeholder - requires AlphaFold)
        if self.use_structure:
            # Would call AlphaFold here
            metrics.plddt_mean = 0.0
            metrics.tm_score = 0.0

        return metrics

    def compute_novelty_metrics(
        self,
        generated: torch.Tensor,
    ) -> NoveltyMetrics:
        """Compute novelty metrics relative to training set.

        Args:
            generated: (batch, seq_len) generated sequences

        Returns:
            NoveltyMetrics object
        """
        metrics = NoveltyMetrics()
        batch_size = generated.shape[0]

        # Unique sequences in batch
        unique_seqs = torch.unique(generated, dim=0)
        metrics.unique_fraction = len(unique_seqs) / batch_size

        # Compare to training set
        if self.training_sequences is not None and len(self.training_sequences) > 0:
            min_identities = []
            mean_identities = []

            for seq in generated:
                identities = []
                for train_seq in self.training_sequences[:100]:  # Sample for efficiency
                    identity = compute_sequence_identity(seq, train_seq)
                    identities.append(identity)

                min_identities.append(min(identities))
                mean_identities.append(np.mean(identities))

            metrics.min_seq_identity = np.mean(min_identities)
            metrics.mean_seq_identity = np.mean(mean_identities)

            # Novel if min identity < 0.9
            metrics.novel_fraction = sum(1 for x in min_identities if x < 0.9) / len(min_identities)

        return metrics

    def compute_diversity_metrics(
        self,
        generated: torch.Tensor,
    ) -> DiversityMetrics:
        """Compute diversity metrics for generated set.

        Args:
            generated: (batch, seq_len) generated sequences

        Returns:
            DiversityMetrics object
        """
        metrics = DiversityMetrics()
        batch_size = generated.shape[0]

        if batch_size < 2:
            return metrics

        # Pairwise distances
        distances = compute_pairwise_distances(generated, metric="hamming")

        # Extract upper triangle (excluding diagonal)
        mask = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
        pairwise = distances[mask]

        metrics.mean_pairwise_distance = pairwise.mean().item()
        metrics.min_pairwise_distance = pairwise.min().item() if len(pairwise) > 0 else 0.0

        # Cluster entropy
        entropy, n_clusters = compute_cluster_entropy(generated)
        metrics.cluster_entropy = entropy
        metrics.n_clusters = n_clusters

        # Coverage
        unique_codons = torch.unique(generated).numel()
        metrics.codon_coverage = unique_codons / 64  # 64 possible codons

        # Amino acid coverage
        aa_set = set()
        for seq in generated:
            for codon_idx in seq:
                triplet = codon_index_to_triplet(codon_idx.item())
                aa = GENETIC_CODE.get(triplet, "?")
                if aa != "*":
                    aa_set.add(aa)
        metrics.amino_acid_coverage = len(aa_set) / 20  # 20 standard AAs

        return metrics

    def compute_validity_metrics(
        self,
        generated: torch.Tensor,
    ) -> BiologicalValidityMetrics:
        """Compute biological validity metrics.

        Args:
            generated: (batch, seq_len) generated sequences

        Returns:
            BiologicalValidityMetrics object
        """
        validity_dict = check_biological_validity(generated)

        metrics = BiologicalValidityMetrics(
            no_stop_codons=validity_dict["no_stop_codons"],
            valid_start_codon=validity_dict["valid_start_codon"],
            valid_length=validity_dict["valid_length"],
        )

        return metrics

    def evaluate(
        self,
        generated: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> GenerationMetrics:
        """Run complete evaluation.

        Args:
            generated: (batch, seq_len) generated sequences
            targets: (batch, seq_len) optional targets for reconstruction

        Returns:
            Complete GenerationMetrics object
        """
        metrics = GenerationMetrics()

        # Basic stats
        metrics.n_sequences = generated.shape[0]
        metrics.mean_length = generated.shape[1]

        # Compute all metric categories
        metrics.quality = self.compute_quality_metrics(generated, targets)
        metrics.novelty = self.compute_novelty_metrics(generated)
        metrics.diversity = self.compute_diversity_metrics(generated)
        metrics.validity = self.compute_validity_metrics(generated)

        return metrics


def evaluate_generated_sequences(
    generated: torch.Tensor,
    training_set: torch.Tensor | None = None,
    targets: torch.Tensor | None = None,
    organism: Organism = Organism.HUMAN,
) -> GenerationMetrics:
    """Convenience function for sequence evaluation.

    Args:
        generated: (batch, seq_len) generated codon indices
        training_set: Optional training sequences for novelty
        targets: Optional targets for reconstruction metrics
        organism: Organism for codon optimality

    Returns:
        GenerationMetrics object with all metrics
    """
    evaluator = ProteinGymEvaluator(
        training_sequences=training_set,
        organism=organism,
    )
    return evaluator.evaluate(generated, targets)


__all__ = [
    "QualityMetrics",
    "NoveltyMetrics",
    "DiversityMetrics",
    "BiologicalValidityMetrics",
    "GenerationMetrics",
    "ProteinGymEvaluator",
    "evaluate_generated_sequences",
    "compute_sequence_identity",
    "compute_pairwise_distances",
]
