# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Codon Encoder with p-adic Embeddings.

This module implements the `CodonEncoder` (Proposal 3), a specialized embedding
layer for biological sequences that respects the ultrametric structure of the
genetic code.

Key Features:
- Maps codon indices (0-63) to latent space (Dim).
- Computes p-adic (3-adic) distances between codons.
- MDS-based initialization from p-adic distance matrix.
- Amino acid property-aware similarity metrics.
- Integrates with `src.core.ternary` for 3-adic logic.

Usage:
    from src.encoders.codon_encoder import CodonEncoder
    encoder = CodonEncoder(embedding_dim=16)
    z = encoder(codon_indices)

    # Get p-adic distance matrix
    padic_dists = encoder.get_padic_distance_matrix()
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# =============================================================================
# Imports from centralized modules
# =============================================================================
from src.biology.codons import (
    GENETIC_CODE,
    codon_index_to_triplet,
)
from src.core.padic_math import padic_valuation
from src.geometry import poincare_distance

# Amino acid properties (normalized to [-1, 1])
AA_PROPERTIES = {
    # (hydrophobicity, charge, size, polarity)
    "A": (0.62, 0.0, -0.77, -0.5),  # Alanine
    "R": (-2.53, 1.0, 0.69, 1.0),  # Arginine
    "N": (-0.78, 0.0, -0.09, 1.0),  # Asparagine
    "D": (-0.90, -1.0, -0.25, 1.0),  # Aspartic acid
    "C": (0.29, 0.0, -0.41, 0.0),  # Cysteine
    "Q": (-0.85, 0.0, 0.24, 1.0),  # Glutamine
    "E": (-0.74, -1.0, 0.07, 1.0),  # Glutamic acid
    "G": (0.48, 0.0, -1.00, -0.5),  # Glycine
    "H": (-0.40, 0.5, 0.36, 0.5),  # Histidine
    "I": (1.38, 0.0, 0.30, -1.0),  # Isoleucine
    "L": (1.06, 0.0, 0.30, -1.0),  # Leucine
    "K": (-1.50, 1.0, 0.53, 1.0),  # Lysine
    "M": (0.64, 0.0, 0.44, -0.5),  # Methionine
    "F": (1.19, 0.0, 0.77, -1.0),  # Phenylalanine
    "P": (0.12, 0.0, -0.45, -0.5),  # Proline
    "S": (-0.18, 0.0, -0.60, 0.5),  # Serine
    "T": (-0.05, 0.0, -0.28, 0.5),  # Threonine
    "W": (0.81, 0.0, 1.00, -0.5),  # Tryptophan
    "Y": (0.26, 0.0, 0.77, 0.0),  # Tyrosine
    "V": (1.08, 0.0, 0.00, -1.0),  # Valine
    "*": (0.0, 0.0, 0.0, 0.0),  # Stop codon
}


# =============================================================================
# P-adic Distance Functions
# =============================================================================


def compute_padic_valuation(n: int, p: int = 3) -> int:
    """Compute p-adic valuation v_p(n).

    Uses centralized padic_valuation from src.core.padic_math.

    The p-adic valuation is the largest power of p that divides n.
    For n=0, returns infinity (represented as a large number).
    """
    return padic_valuation(n, p)


def compute_padic_distance_between_codons(idx1: int, idx2: int, p: int = 3) -> float:
    """Compute p-adic distance between two codons.

    The p-adic distance is based on how similar the codons are in their
    hierarchical structure (first base, second base, third base).

    For the genetic code, we use a 3-adic metric where:
    - Same first base: distance starts small
    - Same first and second base: even smaller
    - Same codon: distance = 0

    Returns:
        p-adic distance (0 to 1, normalized)
    """
    if idx1 == idx2:
        return 0.0

    # Decode codons to base triplets
    b1_1, b2_1, b3_1 = (idx1 // 16) % 4, (idx1 // 4) % 4, idx1 % 4
    b1_2, b2_2, b3_2 = (idx2 // 16) % 4, (idx2 // 4) % 4, idx2 % 4

    # Hierarchical distance based on position matches
    # First base is most significant (like most significant digit)
    if b1_1 != b1_2:
        return 1.0  # Maximum distance
    elif b2_1 != b2_2:
        return 1.0 / p  # Medium distance
    elif b3_1 != b3_2:
        return 1.0 / (p * p)  # Small distance
    else:
        return 0.0


def compute_amino_acid_distance(idx1: int, idx2: int) -> float:
    """Compute distance based on encoded amino acid properties.

    Uses Euclidean distance in the amino acid property space
    (hydrophobicity, charge, size, polarity).
    """
    triplet1 = codon_index_to_triplet(idx1)
    triplet2 = codon_index_to_triplet(idx2)

    aa1 = GENETIC_CODE.get(triplet1, "*")
    aa2 = GENETIC_CODE.get(triplet2, "*")

    props1 = np.array(AA_PROPERTIES.get(aa1, (0, 0, 0, 0)))
    props2 = np.array(AA_PROPERTIES.get(aa2, (0, 0, 0, 0)))

    return float(np.linalg.norm(props1 - props2))


def compute_full_padic_distance_matrix(n_codons: int = 64, p: int = 3) -> np.ndarray:
    """Compute full 64x64 p-adic distance matrix between all codons."""
    matrix = np.zeros((n_codons, n_codons))

    for i in range(n_codons):
        for j in range(n_codons):
            matrix[i, j] = compute_padic_distance_between_codons(i, j, p)

    return matrix


def compute_combined_distance_matrix(
    n_codons: int = 64,
    padic_weight: float = 0.5,
    aa_weight: float = 0.5,
) -> np.ndarray:
    """Compute combined distance matrix (p-adic + amino acid properties).

    Args:
        n_codons: Number of codons (64 for standard code)
        padic_weight: Weight for p-adic distance component
        aa_weight: Weight for amino acid property distance

    Returns:
        Combined distance matrix, shape (n_codons, n_codons)
    """
    padic_matrix = compute_full_padic_distance_matrix(n_codons)
    aa_matrix = np.zeros((n_codons, n_codons))

    for i in range(n_codons):
        for j in range(n_codons):
            aa_matrix[i, j] = compute_amino_acid_distance(i, j)

    # Normalize amino acid distances to [0, 1]
    if aa_matrix.max() > 0:
        aa_matrix = aa_matrix / aa_matrix.max()

    return padic_weight * padic_matrix + aa_weight * aa_matrix


def mds_from_distance_matrix(
    distance_matrix: np.ndarray,
    n_components: int = 16,
) -> np.ndarray:
    """Compute MDS embedding from distance matrix.

    Classical Multidimensional Scaling to find coordinates that
    approximate the given distance matrix.

    Args:
        distance_matrix: Pairwise distances, shape (N, N)
        n_components: Number of output dimensions

    Returns:
        Coordinates, shape (N, n_components)
    """
    n = distance_matrix.shape[0]

    # Double centering
    D_sq = distance_matrix**2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D_sq @ H

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Take top n_components
    k = min(n_components, len(eigenvalues))
    pos_eigenvalues = np.maximum(eigenvalues[:k], 0)

    # Compute coordinates
    coords = eigenvectors[:, :k] * np.sqrt(pos_eigenvalues)[None, :]

    # Pad if needed
    if k < n_components:
        padding = np.zeros((n, n_components - k))
        coords = np.hstack([coords, padding])

    return coords


class CodonEncoder(nn.Module):
    """Embedding layer for codons with p-adic structure awareness."""

    def __init__(
        self,
        embedding_dim: int = 16,
        padding_idx: int | None = None,
        use_padic_init: bool = True,
    ):
        """Initialize CodonEncoder.

        Args:
            embedding_dim: Size of the embedding vector
            padding_idx: Index to use for padding (zeros out gradients)
            use_padic_init: If True, initializes weights using 3-adic valuation patterns
        """
        super().__init__()
        self.num_codons = 64
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=self.num_codons,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        if use_padic_init:
            self._init_padic_weights()

    def _init_padic_weights(self):
        """Initialize embeddings to reflect 3-adic distances.

        We use the TERNARY module to compute valuations between codon indices
        and project them into the embedding space.

        Since we don't have a direct map from codon integer (0-63) to the
        19683-space of the ternary algebra, we define a mapping or use
        a simpler heuristic:

        Map 0-63 to the first 64 indices of the ternary space (or any subset).
        Then use Multi-Dimensional Scaling (MDS) or PCA on the distance matrix
        to get initial coordinates.

        For this MVP, we'll use a simplified stratified initialization:
        - Codons are grouped by first base (4 groups)
        - Then by second base (16 groups)
        - This mirrors the hierarchical structure.
        """
        with torch.no_grad():
            # Simple hierarchical init
            # 64 codons.
            # Dims 0-3: Encode Base 1
            # Dims 4-7: Encode Base 2
            # Dims 8-11: Encode Base 3
            # Dims 12+: Noise/Free

            # Base one-hot (A, C, G, T) -> 4 dims
            base_map = {
                0: [1, 0, 0, 0],  # A
                1: [0, 1, 0, 0],  # C
                2: [0, 0, 1, 0],  # G
                3: [0, 0, 0, 1],  # T
            }

            weights = self.embedding.weight.data
            scaling = 0.5  # Small scale init

            for i in range(self.num_codons):
                # Decode index i (0-63) to bases
                # format: B1 * 16 + B2 * 4 + B3
                b1 = (i // 16) % 4
                b2 = (i // 4) % 4
                b3 = i % 4

                vec_b1 = torch.tensor(base_map[b1], dtype=torch.float32)
                vec_b2 = torch.tensor(base_map[b2], dtype=torch.float32)
                vec_b3 = torch.tensor(base_map[b3], dtype=torch.float32)

                if self.embedding_dim >= 12:
                    weights[i, 0:4] = vec_b1 * scaling
                    weights[i, 4:8] = vec_b2 * scaling
                    weights[i, 8:12] = vec_b3 * scaling
                    # Remainders stay random (default init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed codon indices.

        Args:
            x: Tensor of codon indices (Batch, SeqLen)

        Returns:
            Tensor of embeddings (Batch, SeqLen, Dim)
        """
        return self.embedding(x)

    def get_distance_matrix(self) -> torch.Tensor:
        """Compute pairwise Euclidean distances between all codon embeddings."""
        w = self.embedding.weight
        return torch.cdist(w, w)

    def get_padic_distance_matrix(self, p: int = 3) -> torch.Tensor:
        """Get p-adic distance matrix between all codons.

        Args:
            p: Prime base for p-adic metric (default 3 for ternary)

        Returns:
            Distance matrix, shape (64, 64)
        """
        matrix = compute_full_padic_distance_matrix(self.num_codons, p)
        return torch.from_numpy(matrix).float()

    def get_amino_acid_distance_matrix(self) -> torch.Tensor:
        """Get amino acid property-based distance matrix.

        Returns:
            Normalized distance matrix, shape (64, 64)
        """
        matrix = np.zeros((self.num_codons, self.num_codons))
        for i in range(self.num_codons):
            for j in range(self.num_codons):
                matrix[i, j] = compute_amino_acid_distance(i, j)

        # Normalize to [0, 1]
        if matrix.max() > 0:
            matrix = matrix / matrix.max()

        return torch.from_numpy(matrix).float()

    def compute_padic_loss(
        self,
        padic_weight: float = 0.5,
        aa_weight: float = 0.5,
    ) -> torch.Tensor:
        """Compute loss encouraging embeddings to preserve p-adic structure.

        The loss penalizes deviations between:
        - Euclidean distances in embedding space
        - Combined p-adic + amino acid property distances

        Args:
            padic_weight: Weight for p-adic distance component
            aa_weight: Weight for amino acid property distance

        Returns:
            Scalar loss tensor
        """
        # Target distance matrix (combined p-adic + AA)
        target = compute_combined_distance_matrix(self.num_codons, padic_weight, aa_weight)
        target = torch.from_numpy(target).float().to(self.embedding.weight.device)

        # Current embedding distances
        current = self.get_distance_matrix()

        # Normalize current to same scale as target
        if current.max() > 0:
            current = current / current.max()

        # MSE loss between distance matrices
        return torch.nn.functional.mse_loss(current, target)

    def get_codon_similarity(
        self,
        idx1: int,
        idx2: int,
        use_hyperbolic: bool = True,
        curvature: float = 1.0,
    ) -> dict:
        """Get detailed similarity metrics between two codons.

        Args:
            idx1: First codon index (0-63)
            idx2: Second codon index (0-63)
            use_hyperbolic: V5.12.2 - Use poincare_distance for hyperbolic embeddings (default True)
            curvature: Hyperbolic curvature for poincare_distance

        Returns:
            Dictionary with similarity metrics
        """
        triplet1 = codon_index_to_triplet(idx1)
        triplet2 = codon_index_to_triplet(idx2)

        aa1 = GENETIC_CODE.get(triplet1, "*")
        aa2 = GENETIC_CODE.get(triplet2, "*")

        padic_dist = compute_padic_distance_between_codons(idx1, idx2)
        aa_dist = compute_amino_acid_distance(idx1, idx2)

        # V5.12.2: Embedding distance (Euclidean or hyperbolic)
        with torch.no_grad():
            emb1 = self.embedding.weight[idx1].unsqueeze(0)
            emb2 = self.embedding.weight[idx2].unsqueeze(0)
            if use_hyperbolic:
                emb_dist = poincare_distance(emb1, emb2, c=curvature).item()
            else:
                emb_dist = torch.norm(emb1 - emb2).item()

        return {
            "codon1": triplet1,
            "codon2": triplet2,
            "amino_acid1": aa1,
            "amino_acid2": aa2,
            "synonymous": aa1 == aa2,
            "padic_distance": padic_dist,
            "aa_property_distance": aa_dist,
            "embedding_distance": emb_dist,
        }

    def get_synonymous_codon_groups(self) -> dict:
        """Group codons by their encoded amino acid.

        Returns:
            Dictionary mapping amino acid -> list of codon indices
        """
        groups: dict = {}
        for idx in range(self.num_codons):
            triplet = codon_index_to_triplet(idx)
            aa = GENETIC_CODE.get(triplet, "*")
            if aa not in groups:
                groups[aa] = []
            groups[aa].append(idx)
        return groups
