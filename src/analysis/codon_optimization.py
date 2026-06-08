# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Citrullination Boundary Codon Optimizer.

This module implements an algorithm to optimize codon choices for maximum
distance to p-adic cluster boundaries, enabling design of "immunologically
silent" sequences for regenerative medicine and therapeutic applications.

Key concepts:
- Codon optimization: Choosing synonymous codons for expression
- Citrullination boundaries: p-adic distances that trigger autoimmunity
- Safe zone: Region in p-adic space far from immune recognition thresholds
- Silent design: Sequences that avoid triggering autoimmune responses

The algorithm:
1. For each Arg codon (CGU, CGC, CGA, CGG, AGA, AGG), compute p-adic distance
   to citrullination boundary
2. Select codons that maximize distance from Goldilocks Zone
3. Optimize surrounding codon context to further reduce immunogenicity
4. Validate that protein function is preserved

References:
- DISCOVERY_CITRULLINATION_BOUNDARIES.md (lines 217-251)
- Wong Co-Evolution Theory for codon selection
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.biology.codons import codon_to_index


@dataclass
class OptimizationResult:
    """Result of codon optimization."""

    original_sequence: str
    optimized_sequence: str
    original_codons: list[str]
    optimized_codons: list[str]
    changes_made: list[tuple[int, str, str]]  # (position, old, new)
    original_padic_distance: float
    optimized_padic_distance: float
    improvement_score: float
    immunogenicity_reduction: float


@dataclass
class CodonChoice:
    """Represents a codon choice with its properties."""

    codon: str
    amino_acid: str
    padic_index: int
    boundary_distance: float
    usage_frequency: float  # In target organism
    tRNA_abundance: float


# Arginine codons
ARGININE_CODONS = ["CGU", "CGC", "CGA", "CGG", "AGA", "AGG"]

# Standard genetic code (RNA codons)
CODON_TABLE = {
    "UUU": "F",
    "UUC": "F",
    "UUA": "L",
    "UUG": "L",
    "UCU": "S",
    "UCC": "S",
    "UCA": "S",
    "UCG": "S",
    "UAU": "Y",
    "UAC": "Y",
    "UAA": "*",
    "UAG": "*",
    "UGU": "C",
    "UGC": "C",
    "UGA": "*",
    "UGG": "W",
    "CUU": "L",
    "CUC": "L",
    "CUA": "L",
    "CUG": "L",
    "CCU": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAU": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGU": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "AUU": "I",
    "AUC": "I",
    "AUA": "I",
    "AUG": "M",
    "ACU": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAU": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGU": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GUU": "V",
    "GUC": "V",
    "GUA": "V",
    "GUG": "V",
    "GCU": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAU": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGU": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

# Synonymous codon groups
SYNONYMOUS_CODONS: dict[str, list[str]] = {}
for codon, aa in CODON_TABLE.items():
    if aa not in SYNONYMOUS_CODONS:
        SYNONYMOUS_CODONS[aa] = []
    SYNONYMOUS_CODONS[aa].append(codon)

# Human codon usage frequencies (simplified)
HUMAN_CODON_USAGE = {
    # Arginine codons (key for citrullination)
    "CGU": 0.08,
    "CGC": 0.19,
    "CGA": 0.11,
    "CGG": 0.20,
    "AGA": 0.20,
    "AGG": 0.22,
    # Other common codons
    "GCU": 0.26,
    "GCC": 0.40,
    "GCA": 0.23,
    "GCG": 0.11,
    "UGU": 0.45,
    "UGC": 0.55,
}


def compute_padic_distance(idx1: int, idx2: int, p: int = 3) -> float:
    """Compute p-adic distance between two codon indices.

    Args:
        idx1: First codon index
        idx2: Second codon index
        p: Prime for p-adic calculation

    Returns:
        p-adic distance
    """
    diff = abs(idx1 - idx2)
    if diff == 0:
        return 0.0

    v = 0
    while diff % p == 0:
        v += 1
        diff //= p

    return float(p) ** (-v)


class PAdicBoundaryAnalyzer:
    """Analyzes p-adic boundaries for citrullination risk.

    Identifies the p-adic distance thresholds where immune
    recognition changes behavior (Goldilocks Zone boundaries).
    """

    def __init__(
        self,
        p: int = 3,
        zone_min: float = 0.15,
        zone_max: float = 0.30,
    ):
        """Initialize analyzer.

        Args:
            p: Prime for p-adic calculations
            zone_min: Lower bound of Goldilocks Zone
            zone_max: Upper bound of Goldilocks Zone
        """
        self.p = p
        self.zone_min = zone_min
        self.zone_max = zone_max

        # Precompute codon p-adic indices
        self.codon_indices = {codon: codon_to_index(codon) for codon in CODON_TABLE}

        # Compute boundary distances for arginine codons
        self._compute_arginine_boundaries()

    def _compute_arginine_boundaries(self) -> None:
        """Precompute boundary distances for arginine codons."""
        self.arg_boundary_distances = {}

        for codon in ARGININE_CODONS:
            idx = self.codon_indices[codon]

            # Compute distances to all other codons
            distances = []
            for other_codon, other_idx in self.codon_indices.items():
                if other_codon != codon:
                    d = compute_padic_distance(idx, other_idx, self.p)
                    distances.append(d)

            # Boundary distance = minimum distance to zone boundaries
            zone_distances = []
            for d in distances:
                if d < self.zone_min:
                    zone_distances.append(self.zone_min - d)
                elif d > self.zone_max:
                    zone_distances.append(d - self.zone_max)
                else:
                    zone_distances.append(0.0)  # In zone

            self.arg_boundary_distances[codon] = min(zone_distances) if zone_distances else 0.0

    def get_safest_arginine_codon(self) -> str:
        """Get the arginine codon furthest from Goldilocks boundaries.

        Returns:
            Safest arginine codon
        """
        best_codon = max(self.arg_boundary_distances.items(), key=lambda x: x[1])[0]
        return best_codon

    def rank_arginine_codons(self) -> list[tuple[str, float]]:
        """Rank arginine codons by safety (distance from boundaries).

        Returns:
            List of (codon, boundary_distance) sorted by safety
        """
        ranked = sorted(self.arg_boundary_distances.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def is_in_danger_zone(self, codon_idx: int) -> bool:
        """Check if a codon index falls in the Goldilocks Zone.

        Args:
            codon_idx: Codon index

        Returns:
            True if in danger zone
        """
        # Check distance to reference (first arginine codon)
        ref_idx = self.codon_indices["CGU"]
        d = compute_padic_distance(codon_idx, ref_idx, self.p)
        return self.zone_min <= d <= self.zone_max


class CodonContextOptimizer(nn.Module):
    """Optimizes codon context around arginine residues.

    Uses learned patterns to select surrounding codons that
    minimize citrullination risk and immunogenicity.
    """

    def __init__(
        self,
        context_size: int = 5,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
    ):
        """Initialize optimizer.

        Args:
            context_size: Number of codons on each side to consider
            embedding_dim: Codon embedding dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.context_size = context_size

        # Codon embedding (64 codons)
        self.codon_embedding = nn.Embedding(64, embedding_dim)

        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(embedding_dim * (2 * context_size + 1), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Safety scorer
        self.safety_scorer = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        # Alternative codon suggester (for each position in context)
        self.codon_suggester = nn.Linear(hidden_dim // 2, 64 * (2 * context_size + 1))

    def encode_codon_sequence(self, codons: list[str]) -> torch.Tensor:
        """Encode a sequence of codons.

        Args:
            codons: List of codon strings

        Returns:
            Tensor of codon indices
        """
        indices = [codon_to_index(c) for c in codons]
        return torch.tensor(indices)

    def forward(
        self,
        codon_contexts: list[list[str]],
    ) -> dict[str, torch.Tensor]:
        """Analyze codon contexts for optimization.

        Args:
            codon_contexts: List of codon sequences (centered on Arg codon)

        Returns:
            Dictionary with safety scores and suggestions
        """
        batch_size = len(codon_contexts)
        context_len = 2 * self.context_size + 1

        # Encode contexts
        encoded = []
        for context in codon_contexts:
            # Pad if necessary
            while len(context) < context_len:
                context.append("NNN")  # Padding codon
            indices = self.encode_codon_sequence(context[:context_len])
            encoded.append(indices)

        encoded = torch.stack(encoded)
        embedded = self.codon_embedding(encoded)  # (batch, context_len, embed_dim)
        flat = embedded.flatten(start_dim=1)

        # Encode context
        context_features = self.context_encoder(flat)

        # Compute safety score
        safety = self.safety_scorer(context_features)

        # Generate codon suggestions
        suggestions = self.codon_suggester(context_features)
        suggestions = suggestions.view(batch_size, context_len, 64)
        suggestion_probs = F.softmax(suggestions, dim=-1)

        return {
            "safety_score": safety.squeeze(-1),
            "context_features": context_features,
            "suggestion_probabilities": suggestion_probs,
        }


class CitrullinationBoundaryOptimizer:
    """Main optimizer for citrullination-safe sequence design.

    Combines p-adic boundary analysis with context optimization
    to generate sequences with minimal autoimmune risk.
    """

    def __init__(
        self,
        p: int = 3,
        zone_min: float = 0.15,
        zone_max: float = 0.30,
        preserve_usage: bool = True,
    ):
        """Initialize optimizer.

        Args:
            p: Prime for p-adic calculations
            zone_min: Lower Goldilocks boundary
            zone_max: Upper Goldilocks boundary
            preserve_usage: Whether to consider codon usage frequency
        """
        self.p = p
        self.preserve_usage = preserve_usage

        self.boundary_analyzer = PAdicBoundaryAnalyzer(p, zone_min, zone_max)
        self.context_optimizer = CodonContextOptimizer()

    def dna_to_codons(self, dna_sequence: str) -> list[str]:
        """Convert DNA sequence to list of codons.

        Args:
            dna_sequence: DNA sequence (must be multiple of 3)

        Returns:
            List of RNA codons
        """
        rna = dna_sequence.upper().replace("T", "U")
        codons = [rna[i : i + 3] for i in range(0, len(rna), 3)]
        return codons

    def codons_to_dna(self, codons: list[str]) -> str:
        """Convert codons back to DNA sequence.

        Args:
            codons: List of RNA codons

        Returns:
            DNA sequence
        """
        rna = "".join(codons)
        return rna.replace("U", "T")

    def find_arginine_positions(self, codons: list[str]) -> list[int]:
        """Find positions of arginine codons.

        Args:
            codons: List of codons

        Returns:
            List of positions containing arginine codons
        """
        positions = []
        for i, codon in enumerate(codons):
            if codon in ARGININE_CODONS:
                positions.append(i)
        return positions

    def optimize_arginine_codons(
        self,
        codons: list[str],
        usage_weight: float = 0.3,
    ) -> list[str]:
        """Optimize arginine codon selection for safety.

        Args:
            codons: Original codon list
            usage_weight: Weight for codon usage frequency

        Returns:
            Optimized codon list
        """
        optimized = codons.copy()
        arg_positions = self.find_arginine_positions(codons)

        # Rank arginine codons by safety
        ranked = self.boundary_analyzer.rank_arginine_codons()

        for pos in arg_positions:
            original = codons[pos]

            # Find best replacement
            best_codon = original
            best_score = -float("inf")

            for codon, boundary_dist in ranked:
                # Safety score from boundary distance
                safety_score = boundary_dist

                # Usage score (if preserving usage)
                if self.preserve_usage:
                    usage = HUMAN_CODON_USAGE.get(codon, 0.1)
                    usage_score = usage
                else:
                    usage_score = 1.0

                # Combined score
                score = (1 - usage_weight) * safety_score + usage_weight * usage_score

                if score > best_score:
                    best_score = score
                    best_codon = codon

            optimized[pos] = best_codon

        return optimized

    def optimize_context(
        self,
        codons: list[str],
        arg_position: int,
        context_size: int = 5,
    ) -> list[str]:
        """Optimize codon context around an arginine.

        Args:
            codons: Full codon list
            arg_position: Position of arginine codon
            context_size: Number of flanking codons to consider

        Returns:
            Optimized codon list
        """
        optimized = codons.copy()

        # Extract context
        start = max(0, arg_position - context_size)
        end = min(len(codons), arg_position + context_size + 1)
        context = codons[start:end]

        # Get optimization suggestions
        with torch.no_grad():
            result = self.context_optimizer([context])
            suggestion_probs = result["suggestion_probabilities"][0]

        # Apply suggestions while preserving amino acid sequence
        for i, pos in enumerate(range(start, end)):
            if pos == arg_position:
                continue  # Skip arginine position (handled separately)

            original_codon = codons[pos]
            aa = CODON_TABLE.get(original_codon)

            if aa and aa != "*":
                # Find synonymous codons
                synonymous = SYNONYMOUS_CODONS.get(aa, [original_codon])

                # Score each alternative
                best_codon = original_codon
                best_prob = 0.0

                for syn_codon in synonymous:
                    syn_idx = codon_to_index(syn_codon)
                    prob = suggestion_probs[i, syn_idx].item()
                    if prob > best_prob:
                        best_prob = prob
                        best_codon = syn_codon

                optimized[pos] = best_codon

        return optimized

    def optimize_sequence(
        self,
        dna_sequence: str,
        optimize_context_flag: bool = True,
    ) -> OptimizationResult:
        """Fully optimize a DNA sequence for citrullination safety.

        Args:
            dna_sequence: Original DNA sequence
            optimize_context_flag: Whether to also optimize context

        Returns:
            OptimizationResult with optimized sequence
        """
        original_codons = self.dna_to_codons(dna_sequence)
        optimized_codons = original_codons.copy()

        # Track changes
        changes = []

        # Step 1: Optimize arginine codons
        optimized_codons = self.optimize_arginine_codons(optimized_codons)

        # Record arginine changes
        for i, (orig, opt) in enumerate(zip(original_codons, optimized_codons, strict=False)):
            if orig != opt:
                changes.append((i, orig, opt))

        # Step 2: Optimize context around arginines
        if optimize_context_flag:
            arg_positions = self.find_arginine_positions(optimized_codons)
            for pos in arg_positions:
                optimized_codons = self.optimize_context(optimized_codons, pos)

            # Record context changes
            for i, (orig, opt) in enumerate(zip(original_codons, optimized_codons, strict=False)):
                if orig != opt and (i, orig, opt) not in changes:
                    changes.append((i, orig, opt))

        # Compute p-adic distances
        original_distance = self._compute_sequence_padic_distance(original_codons)
        optimized_distance = self._compute_sequence_padic_distance(optimized_codons)

        # Compute improvement
        improvement = optimized_distance - original_distance
        immunogenicity_reduction = max(0, improvement / (original_distance + 0.001))

        return OptimizationResult(
            original_sequence=dna_sequence,
            optimized_sequence=self.codons_to_dna(optimized_codons),
            original_codons=original_codons,
            optimized_codons=optimized_codons,
            changes_made=changes,
            original_padic_distance=original_distance,
            optimized_padic_distance=optimized_distance,
            improvement_score=improvement,
            immunogenicity_reduction=immunogenicity_reduction,
        )

    def _compute_sequence_padic_distance(self, codons: list[str]) -> float:
        """Compute average p-adic distance from Goldilocks boundaries.

        Args:
            codons: List of codons

        Returns:
            Average boundary distance for arginine codons
        """
        arg_positions = self.find_arginine_positions(codons)
        if not arg_positions:
            return 1.0  # Maximum safety if no arginines

        distances = []
        for pos in arg_positions:
            codon = codons[pos]
            dist = self.boundary_analyzer.arg_boundary_distances.get(codon, 0.0)
            distances.append(dist)

        return sum(distances) / len(distances)

    def batch_optimize(
        self,
        sequences: list[str],
    ) -> list[OptimizationResult]:
        """Optimize multiple sequences.

        Args:
            sequences: List of DNA sequences

        Returns:
            List of OptimizationResult objects
        """
        return [self.optimize_sequence(seq) for seq in sequences]
