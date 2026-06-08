# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""
Co-Evolution Loss for Genetic Code Analysis.

Implementation based on:
- Wong (1975): Co-evolution theory of the genetic code
- Shenhav (2020): Resource conservation principle
- Wnetrzak (2018): Multi-objective optimality

Models the evolutionary constraints that shaped the genetic code,
integrating p-adic structure with biosynthetic pathway relationships.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Amino acid biosynthetic pathway families
# Based on Wong's co-evolution theory
BIOSYNTHETIC_FAMILIES: dict[str, list[str]] = {
    "glutamate": ["E", "Q", "P", "R", "K"],  # Glutamate family
    "aspartate": ["D", "N", "T", "I", "M", "K"],  # Aspartate family
    "serine": ["S", "G", "C"],  # Serine family
    "pyruvate": ["A", "V", "L"],  # Pyruvate family
    "aromatic": ["F", "Y", "W"],  # Aromatic family (shikimate pathway)
    "histidine": ["H"],  # Histidine (unique pathway)
}

# Codon table
CODON_TABLE: dict[str, str] = {
    "UUU": "F",
    "UUC": "F",
    "UUA": "L",
    "UUG": "L",
    "CUU": "L",
    "CUC": "L",
    "CUA": "L",
    "CUG": "L",
    "AUU": "I",
    "AUC": "I",
    "AUA": "I",
    "AUG": "M",
    "GUU": "V",
    "GUC": "V",
    "GUA": "V",
    "GUG": "V",
    "UCU": "S",
    "UCC": "S",
    "UCA": "S",
    "UCG": "S",
    "CCU": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "ACU": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "GCU": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "UAU": "Y",
    "UAC": "Y",
    "UAA": "*",
    "UAG": "*",
    "CAU": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "AAU": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "GAU": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "UGU": "C",
    "UGC": "C",
    "UGA": "*",
    "UGG": "W",
    "CGU": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AGU": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GGU": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

# Amino acid metabolic costs (ATP equivalents)
# From Akashi & Gojobori (2002)
METABOLIC_COSTS: dict[str, float] = {
    "A": 11.7,
    "R": 27.3,
    "N": 14.7,
    "D": 12.7,
    "C": 24.7,
    "Q": 16.3,
    "E": 15.3,
    "G": 11.7,
    "H": 38.3,
    "I": 32.3,
    "L": 27.3,
    "K": 30.3,
    "M": 34.3,
    "F": 52.0,
    "P": 20.3,
    "S": 11.7,
    "T": 18.7,
    "W": 74.3,
    "Y": 50.0,
    "V": 23.3,
}

# Amino acid properties for error minimization
AA_PROPERTIES: dict[str, dict[str, float]] = {
    "A": {"hydrophobicity": 1.8, "volume": 88.6, "polarity": 0, "charge": 0},
    "R": {"hydrophobicity": -4.5, "volume": 173.4, "polarity": 52, "charge": 1},
    "N": {"hydrophobicity": -3.5, "volume": 114.1, "polarity": 3.38, "charge": 0},
    "D": {"hydrophobicity": -3.5, "volume": 111.1, "polarity": 49.7, "charge": -1},
    "C": {"hydrophobicity": 2.5, "volume": 108.5, "polarity": 1.48, "charge": 0},
    "Q": {"hydrophobicity": -3.5, "volume": 143.8, "polarity": 3.53, "charge": 0},
    "E": {"hydrophobicity": -3.5, "volume": 138.4, "polarity": 49.9, "charge": -1},
    "G": {"hydrophobicity": -0.4, "volume": 60.1, "polarity": 0, "charge": 0},
    "H": {"hydrophobicity": -3.2, "volume": 153.2, "polarity": 51.6, "charge": 0.5},
    "I": {"hydrophobicity": 4.5, "volume": 166.7, "polarity": 0.13, "charge": 0},
    "L": {"hydrophobicity": 3.8, "volume": 166.7, "polarity": 0.13, "charge": 0},
    "K": {"hydrophobicity": -3.9, "volume": 168.6, "polarity": 49.5, "charge": 1},
    "M": {"hydrophobicity": 1.9, "volume": 162.9, "polarity": 1.43, "charge": 0},
    "F": {"hydrophobicity": 2.8, "volume": 189.9, "polarity": 0.35, "charge": 0},
    "P": {"hydrophobicity": -1.6, "volume": 112.7, "polarity": 1.58, "charge": 0},
    "S": {"hydrophobicity": -0.8, "volume": 89.0, "polarity": 1.67, "charge": 0},
    "T": {"hydrophobicity": -0.7, "volume": 116.1, "polarity": 1.66, "charge": 0},
    "W": {"hydrophobicity": -0.9, "volume": 227.8, "polarity": 2.1, "charge": 0},
    "Y": {"hydrophobicity": -1.3, "volume": 193.6, "polarity": 1.61, "charge": 0},
    "V": {"hydrophobicity": 4.2, "volume": 140.0, "polarity": 0.13, "charge": 0},
}


@dataclass
class CoEvolutionMetrics:
    """Metrics from co-evolution loss computation."""

    total_loss: float
    biosynthetic_coherence: float
    error_minimization: float
    resource_conservation: float
    padic_structure: float


class BiosyntheticCoherenceLoss(nn.Module):
    """
    Loss enforcing biosynthetic pathway coherence.

    Codons mapping to amino acids from the same biosynthetic
    family should be clustered in latent space.
    """

    def __init__(
        self,
        n_codons: int = 64,
        latent_dim: int = 16,
        temperature: float = 0.1,
    ):
        """
        Initialize biosynthetic coherence loss.

        Args:
            n_codons: Number of codons
            latent_dim: Latent space dimension
            temperature: Softmax temperature for clustering
        """
        super().__init__()
        self.n_codons = n_codons
        self.latent_dim = latent_dim
        self.temperature = temperature

        # Build codon to family mapping
        self.codon_to_family = {}
        for family_name, amino_acids in BIOSYNTHETIC_FAMILIES.items():
            for codon, aa in CODON_TABLE.items():
                if aa in amino_acids:
                    self.codon_to_family[codon] = family_name

        # Create family indices
        self.families = list(BIOSYNTHETIC_FAMILIES.keys())
        self.n_families = len(self.families)

    def forward(self, codon_embeddings: torch.Tensor, codon_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute biosynthetic coherence loss.

        Args:
            codon_embeddings: Codon latent representations (batch, seq, dim)
            codon_indices: Codon indices (batch, seq)

        Returns:
            Coherence loss value
        """
        batch_size, seq_len, dim = codon_embeddings.shape

        # Compute pairwise distances
        emb_flat = codon_embeddings.view(-1, dim)
        dist_matrix = torch.cdist(emb_flat, emb_flat)

        # Create family membership matrix
        codons = list(CODON_TABLE.keys())
        family_matrix = torch.zeros(64, 64, device=codon_embeddings.device)

        for i, codon1 in enumerate(codons):
            for j, codon2 in enumerate(codons):
                family1 = self.codon_to_family.get(codon1)
                family2 = self.codon_to_family.get(codon2)
                if family1 and family2 and family1 == family2:
                    family_matrix[i, j] = 1.0

        # Get families for actual codons
        indices_flat = codon_indices.view(-1)
        n = indices_flat.shape[0]

        same_family = torch.zeros(n, n, device=codon_embeddings.device)
        for i in range(n):
            for j in range(n):
                idx_i = int(indices_flat[i].item())
                idx_j = int(indices_flat[j].item())
                if idx_i < 64 and idx_j < 64:
                    same_family[i, j] = family_matrix[idx_i, idx_j]

        # Loss: minimize distance for same family, maximize for different
        same_family_dist = (dist_matrix * same_family).sum() / (same_family.sum() + 1e-10)
        diff_family_dist = (dist_matrix * (1 - same_family)).sum() / ((1 - same_family).sum() + 1e-10)

        # Contrastive loss
        loss = same_family_dist - 0.5 * diff_family_dist + 1.0
        return F.relu(loss)


class ErrorMinimizationLoss(nn.Module):
    """
    Loss enforcing error minimization in the genetic code.

    Based on Haig & Hurst (1991): single-nucleotide mutations
    should map to amino acids with similar properties.
    """

    def __init__(
        self,
        p: int = 3,
        property_weights: dict[str, float] | None = None,
    ):
        """
        Initialize error minimization loss.

        Args:
            p: Prime for p-adic calculations
            property_weights: Weights for different AA properties
        """
        super().__init__()
        self.p = p
        self.property_weights = property_weights or {
            "hydrophobicity": 0.4,
            "volume": 0.3,
            "polarity": 0.2,
            "charge": 0.1,
        }

        # Build mutation neighbor map (single nucleotide changes)
        self.mutation_neighbors = self._build_mutation_map()

        # Build property difference matrix
        self.property_diff = self._build_property_diff_matrix()

    def _build_mutation_map(self) -> dict[int, list[int]]:
        """Build map of codon indices to their single-mutation neighbors."""
        codons = list(CODON_TABLE.keys())
        nucleotides = ["U", "C", "A", "G"]

        neighbors: dict[int, list[int]] = {}
        for idx, codon in enumerate(codons):
            neighbors[idx] = []
            for pos in range(3):
                for nt in nucleotides:
                    if nt != codon[pos]:
                        mutant = codon[:pos] + nt + codon[pos + 1 :]
                        if mutant in codons:
                            neighbors[idx].append(codons.index(mutant))

        return neighbors

    def _build_property_diff_matrix(self) -> torch.Tensor:
        """Build matrix of property differences between amino acid pairs."""
        amino_acids = list(AA_PROPERTIES.keys())
        n = len(amino_acids)
        diff_matrix = torch.zeros(n, n)

        for i, aa1 in enumerate(amino_acids):
            for j, aa2 in enumerate(amino_acids):
                props1 = AA_PROPERTIES[aa1]
                props2 = AA_PROPERTIES[aa2]

                total_diff = 0.0
                for prop, weight in self.property_weights.items():
                    # Normalize property difference
                    max_diff = max(abs(AA_PROPERTIES[aa][prop]) for aa in amino_acids)
                    if max_diff > 0:
                        diff = abs(props1[prop] - props2[prop]) / max_diff
                        total_diff += weight * diff

                diff_matrix[i, j] = total_diff

        return diff_matrix

    def forward(self, codon_embeddings: torch.Tensor, codon_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute error minimization loss.

        Args:
            codon_embeddings: Codon latent representations (batch, seq, dim)
            codon_indices: Codon indices (batch, seq)

        Returns:
            Error minimization loss
        """
        codons = list(CODON_TABLE.keys())
        aa_list = list(AA_PROPERTIES.keys())
        aa_to_idx = {aa: i for i, aa in enumerate(aa_list)}

        batch_size, seq_len, dim = codon_embeddings.shape

        total_loss = torch.tensor(0.0, device=codon_embeddings.device)
        count = 0

        for b in range(batch_size):
            for s in range(seq_len):
                codon_idx = int(codon_indices[b, s].item())
                if codon_idx >= 64:
                    continue

                codon = codons[codon_idx]
                aa = CODON_TABLE.get(codon, "*")
                if aa == "*":
                    continue

                aa_idx = aa_to_idx.get(aa)
                if aa_idx is None:
                    continue

                # Get embedding for this codon
                codon_embeddings[b, s]

                # Check mutation neighbors
                for neighbor_idx in self.mutation_neighbors.get(codon_idx, []):
                    neighbor_codon = codons[neighbor_idx]
                    neighbor_aa = CODON_TABLE.get(neighbor_codon, "*")
                    if neighbor_aa == "*":
                        continue

                    neighbor_aa_idx = aa_to_idx.get(neighbor_aa)
                    if neighbor_aa_idx is None:
                        continue

                    # Property difference between amino acids
                    prop_diff = self.property_diff[aa_idx, neighbor_aa_idx]

                    # Latent space distance (approximate from same batch)
                    # For simplicity, use p-adic inspired distance
                    padic_dist = abs(codon_idx - neighbor_idx)
                    v3 = 0
                    temp = padic_dist
                    while temp > 0 and temp % self.p == 0:
                        v3 += 1
                        temp //= self.p
                    padic_dist = float(self.p) ** (-v3) if padic_dist > 0 else 0

                    # Loss: embedding distance should correlate with property difference
                    # High property difference -> high embedding distance (low penalty)
                    # Low property difference -> low embedding distance (low penalty)
                    expected_dist = prop_diff  # Normalized to [0, 1]
                    actual_dist = padic_dist

                    total_loss += (expected_dist - actual_dist) ** 2
                    count += 1

        return total_loss / max(count, 1)


class ResourceConservationLoss(nn.Module):
    """
    Loss enforcing resource conservation principle.

    Based on Shenhav & Zeevi (2020): the genetic code optimizes
    for metabolic efficiency.
    """

    def __init__(
        self,
        cost_weight: float = 0.1,
    ):
        """
        Initialize resource conservation loss.

        Args:
            cost_weight: Weight for metabolic cost penalty
        """
        super().__init__()
        self.cost_weight = cost_weight

        # Build codon cost tensor
        codons = list(CODON_TABLE.keys())
        self.codon_costs = torch.zeros(64)
        for idx, codon in enumerate(codons):
            aa = CODON_TABLE.get(codon, "*")
            self.codon_costs[idx] = METABOLIC_COSTS.get(aa, 50.0)  # Default high cost for stop

        # Normalize costs
        self.codon_costs = self.codon_costs / self.codon_costs.max()

    def forward(
        self,
        codon_probabilities: torch.Tensor,
        codon_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute resource conservation loss.

        Args:
            codon_probabilities: Predicted codon probabilities (batch, seq, 64)
            codon_indices: Optional target codon indices

        Returns:
            Resource conservation loss
        """
        device = codon_probabilities.device
        costs = self.codon_costs.to(device)

        # Expected cost under predicted distribution
        expected_cost = (codon_probabilities * costs.unsqueeze(0).unsqueeze(0)).sum(dim=-1)

        # Penalize high expected cost
        loss = self.cost_weight * expected_cost.mean()

        return loss


class PAdicStructureLoss(nn.Module):
    """
    Loss enforcing p-adic structure in codon space.

    Ensures that the learned representations respect the
    ultrametric (tree-like) structure of the ternary encoding.
    """

    def __init__(
        self,
        p: int = 3,
        n_digits: int = 4,
    ):
        """
        Initialize p-adic structure loss.

        Args:
            p: Prime for p-adic field
            n_digits: Number of p-adic digits (4 for 64 codons)
        """
        super().__init__()
        self.p = p
        self.n_digits = n_digits

        # Precompute p-adic distances
        self.padic_distances = self._compute_padic_distances()

    def _compute_padic_distances(self) -> torch.Tensor:
        """Compute pairwise p-adic distances between all codons."""
        n = self.p**self.n_digits
        distances = torch.zeros(n, n)

        for i in range(n):
            for j in range(n):
                if i == j:
                    distances[i, j] = 0
                else:
                    diff = abs(i - j)
                    v = 0
                    while diff % self.p == 0:
                        v += 1
                        diff //= self.p
                    distances[i, j] = float(self.p) ** (-v)

        return distances

    def forward(self, codon_embeddings: torch.Tensor, codon_indices: torch.Tensor) -> torch.Tensor:
        """
        Compute p-adic structure loss.

        Args:
            codon_embeddings: Codon latent representations (batch, seq, dim)
            codon_indices: Codon indices (batch, seq)

        Returns:
            P-adic structure loss
        """
        device = codon_embeddings.device
        padic_dist = self.padic_distances.to(device)

        batch_size, seq_len, dim = codon_embeddings.shape

        # Compute embedding distances
        emb_flat = codon_embeddings.view(-1, dim)
        emb_dist = torch.cdist(emb_flat, emb_flat)

        # Get p-adic distances for actual codons
        indices_flat = codon_indices.view(-1)
        n = indices_flat.shape[0]

        target_dist = torch.zeros(n, n, device=device)
        for i in range(n):
            for j in range(n):
                idx_i = int(min(indices_flat[i].item(), 63))
                idx_j = int(min(indices_flat[j].item(), 63))
                target_dist[i, j] = padic_dist[idx_i, idx_j]

        # Normalize distances
        emb_dist_norm = emb_dist / (emb_dist.max() + 1e-10)
        target_dist_norm = target_dist / (target_dist.max() + 1e-10)

        # Loss: embedding distances should preserve p-adic structure
        # Use correlation-based loss
        emb_flat_vec = emb_dist_norm.view(-1)
        target_flat_vec = target_dist_norm.view(-1)

        # Pearson correlation (maximize)
        emb_centered = emb_flat_vec - emb_flat_vec.mean()
        target_centered = target_flat_vec - target_flat_vec.mean()

        correlation = (emb_centered * target_centered).sum() / (
            torch.sqrt((emb_centered**2).sum() * (target_centered**2).sum()) + 1e-10
        )

        # Loss = 1 - correlation (minimize)
        return 1.0 - correlation


class CoEvolutionLoss(nn.Module):
    """
    Combined co-evolution loss for genetic code optimization.

    Integrates multiple evolutionary constraints:
    - Biosynthetic pathway coherence
    - Error minimization
    - Resource conservation
    - P-adic structure preservation
    """

    def __init__(
        self,
        latent_dim: int = 16,
        p: int = 3,
        weights: dict[str, float] | None = None,
    ):
        """
        Initialize co-evolution loss.

        Args:
            latent_dim: Latent space dimension
            p: Prime for p-adic calculations
            weights: Loss component weights
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.p = p

        self.weights = weights or {
            "biosynthetic": 0.3,
            "error_min": 0.3,
            "resource": 0.2,
            "padic": 0.2,
        }

        # Component losses
        self.biosynthetic_loss = BiosyntheticCoherenceLoss(latent_dim=latent_dim)
        self.error_loss = ErrorMinimizationLoss(p=p)
        self.resource_loss = ResourceConservationLoss()
        self.padic_loss = PAdicStructureLoss(p=p)

    def forward(
        self,
        codon_embeddings: torch.Tensor,
        codon_indices: torch.Tensor,
        codon_probabilities: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Compute combined co-evolution loss.

        Args:
            codon_embeddings: Codon latent representations (batch, seq, dim)
            codon_indices: Codon indices (batch, seq)
            codon_probabilities: Optional predicted codon probabilities

        Returns:
            Dictionary with loss components and metrics
        """
        # Biosynthetic coherence
        bio_loss = self.biosynthetic_loss(codon_embeddings, codon_indices)

        # Error minimization
        error_loss = self.error_loss(codon_embeddings, codon_indices)

        # Resource conservation (if probabilities provided)
        if codon_probabilities is not None:
            resource_loss = self.resource_loss(codon_probabilities, codon_indices)
        else:
            resource_loss = torch.tensor(0.0, device=codon_embeddings.device)

        # P-adic structure
        padic_loss = self.padic_loss(codon_embeddings, codon_indices)

        # Combined loss
        total_loss = (
            self.weights["biosynthetic"] * bio_loss
            + self.weights["error_min"] * error_loss
            + self.weights["resource"] * resource_loss
            + self.weights["padic"] * padic_loss
        )

        return {
            "loss": total_loss,
            "biosynthetic_loss": bio_loss,
            "error_minimization_loss": error_loss,
            "resource_loss": resource_loss,
            "padic_structure_loss": padic_loss,
            "metrics": CoEvolutionMetrics(
                total_loss=total_loss.item(),
                biosynthetic_coherence=1.0 - bio_loss.item(),
                error_minimization=1.0 - error_loss.item(),
                resource_conservation=1.0 - resource_loss.item() if codon_probabilities is not None else 0.0,
                padic_structure=1.0 - padic_loss.item(),
            ),
        }
