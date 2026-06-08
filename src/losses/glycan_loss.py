# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Sentinel Glycan Loss for glycan shield analysis.

This module implements losses for analyzing glycan shields on viral proteins,
particularly relevant for HIV vaccine design where the glycan shield provides
immune evasion. The p-adic framework helps identify vulnerable epitopes
through geometric analysis of glycan positioning.

Key concepts:
- Glycan density: Number of N-linked glycosylation sites (NXS/NXT motifs)
- Shield coverage: Percentage of surface area protected by glycans
- Vulnerable windows: Regions with low glycan coverage (p-adic "visible" zones)
- Sentinel positions: Key glycan sites that if removed expose epitopes

References:
- 2020_Gainza_MaSIF_Surfaces.md: Surface-based protein analysis
- HIV glycan shield literature (Env protein N-glycans)
"""

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# N-linked glycosylation sequon patterns
GLYCAN_SEQUONS = ["NXS", "NXT"]  # X is any amino acid except P

# Common HIV Env glycan positions (HXB2 numbering)
HIV_ENV_GLYCAN_SITES = [
    88,
    130,
    133,
    137,
    156,
    160,
    197,
    234,
    241,
    262,
    276,
    289,
    295,
    301,
    332,
    339,
    355,
    363,
    386,
    392,
    397,
    406,
    411,
    448,
    460,
    463,
    611,
    616,
    625,
    637,
]

# Glycan types and their approximate sizes (Angstroms radius)
GLYCAN_SIZES = {
    "high_mannose": 15.0,  # Man5-9GlcNAc2
    "complex": 12.0,  # Bi/tri-antennary
    "hybrid": 13.0,  # Mixed mannose/complex
}

# Amino acid one-letter codes
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class GlycanShieldMetrics:
    """Metrics for glycan shield analysis."""

    total_glycans: int
    shield_coverage: float  # Fraction of surface covered
    vulnerable_area: float  # Exposed surface area
    avg_glycan_spacing: float  # Average distance between glycans
    min_glycan_spacing: float  # Minimum spacing (clustering indicator)
    padic_visibility_score: float  # p-adic based visibility
    sentinel_score: float  # Overall vulnerability score


@dataclass
class GlycanSite:
    """Represents a glycosylation site."""

    position: int
    sequon: str  # NXS or NXT
    amino_acid_x: str  # The middle amino acid
    is_occupied: bool = True  # Some sequons are not glycosylated
    glycan_type: str = "high_mannose"


class GlycanSequonDetector(nn.Module):
    """Detects N-linked glycosylation sequons in protein sequences.

    Identifies NXS/NXT motifs where X is any amino acid except proline.
    Uses learned embeddings to predict glycan occupancy probability.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
    ):
        """Initialize detector.

        Args:
            embedding_dim: Dimension of amino acid embeddings
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Amino acid embedding
        self.aa_embedding = nn.Embedding(21, embedding_dim)  # 20 AAs + padding

        # Sequon classifier (looks at triplet context)
        self.sequon_classifier = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # [is_sequon, is_occupied]
        )

        # Create amino acid to index mapping
        self.aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
        self.aa_to_idx["X"] = 20  # Padding/unknown

    def sequence_to_indices(self, sequence: str) -> torch.Tensor:
        """Convert amino acid sequence to indices.

        Args:
            sequence: Amino acid sequence string

        Returns:
            Tensor of indices (seq_len,)
        """
        indices = [self.aa_to_idx.get(aa.upper(), 20) for aa in sequence]
        return torch.tensor(indices)

    def detect_sequons(self, sequence: str) -> list[GlycanSite]:
        """Detect glycosylation sequons in a sequence.

        Args:
            sequence: Protein sequence

        Returns:
            List of GlycanSite objects
        """
        sites = []
        seq_upper = sequence.upper()

        for i in range(len(seq_upper) - 2):
            if seq_upper[i] == "N":
                middle = seq_upper[i + 1]
                third = seq_upper[i + 2]

                # Check if valid sequon (NXS/NXT where X != P)
                if middle != "P" and third in ("S", "T"):
                    sequon_type = f"N{middle}{third}"
                    sites.append(
                        GlycanSite(
                            position=i,
                            sequon=sequon_type,
                            amino_acid_x=middle,
                            is_occupied=True,
                        )
                    )

        return sites

    def forward(
        self,
        sequence_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict glycosylation sites and occupancy.

        Args:
            sequence_indices: Amino acid indices (batch, seq_len)

        Returns:
            Dictionary with sequon_logits and occupancy_probs
        """
        batch_size, seq_len = sequence_indices.shape

        # Embed amino acids
        embeddings = self.aa_embedding(sequence_indices)  # (batch, seq_len, embed_dim)

        # Create triplet features by sliding window
        # Pad sequence
        padded = F.pad(embeddings, (0, 0, 1, 1), mode="constant", value=0)

        # Extract triplets
        triplets = []
        for i in range(seq_len):
            triplet = padded[:, i : i + 3, :].reshape(batch_size, -1)
            triplets.append(triplet)
        triplet_features = torch.stack(triplets, dim=1)  # (batch, seq_len, embed*3)

        # Classify each position
        logits = self.sequon_classifier(triplet_features)  # (batch, seq_len, 2)

        return {
            "sequon_logits": logits[:, :, 0],
            "occupancy_logits": logits[:, :, 1],
            "sequon_probs": torch.sigmoid(logits[:, :, 0]),
            "occupancy_probs": torch.sigmoid(logits[:, :, 1]),
        }


class GlycanShieldAnalyzer(nn.Module):
    """Analyzes glycan shield properties for immune evasion prediction.

    Uses p-adic distances to model hierarchical relationships between
    glycan positions and identify vulnerable regions.
    """

    def __init__(
        self,
        n_positions: int = 700,  # Typical HIV Env length
        glycan_radius: float = 15.0,  # Angstroms
        p: int = 3,
    ):
        """Initialize analyzer.

        Args:
            n_positions: Maximum sequence length
            glycan_radius: Effective shielding radius of glycans
            p: Prime for p-adic calculations (default 3)
        """
        super().__init__()
        self.n_positions = n_positions
        self.glycan_radius = glycan_radius
        self.p = p

        # Precompute p-adic distance matrix for positions
        self._build_padic_distance_matrix()

    def _build_padic_distance_matrix(self) -> None:
        """Build p-adic distance matrix for sequence positions."""
        positions = torch.arange(self.n_positions)
        diff = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))

        # Compute p-adic valuation: v_p(n) = max k such that p^k divides n
        # Handle zero difference specially
        valuations = torch.zeros_like(diff, dtype=torch.float)

        for i in range(1, 10):  # Up to p^9
            divisible = (diff % (self.p**i) == 0) & (diff > 0)
            valuations[divisible] = i

        # p-adic distance: d_p(a, b) = p^(-v_p(a-b))
        padic_dist = torch.where(
            diff == 0,
            torch.zeros_like(valuations),
            torch.pow(float(self.p), -valuations),
        )

        self.register_buffer("padic_distances", padic_dist)

    def compute_shield_coverage(
        self,
        glycan_positions: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        """Compute fraction of sequence covered by glycan shield.

        Args:
            glycan_positions: Tensor of glycan site positions
            sequence_length: Length of protein sequence

        Returns:
            Coverage fraction [0, 1]
        """
        if len(glycan_positions) == 0:
            return torch.tensor(0.0)

        # Create coverage mask
        torch.arange(sequence_length, device=glycan_positions.device)

        # Each glycan covers positions within radius (in sequence space)
        # Approximate: 3.5 Angstroms per residue
        residue_radius = int(self.glycan_radius / 3.5)

        covered = torch.zeros(sequence_length, dtype=torch.bool)
        for gpos in glycan_positions:
            start = max(0, gpos - residue_radius)
            end = min(sequence_length, gpos + residue_radius + 1)
            covered[start:end] = True

        coverage = covered.float().mean()
        return coverage

    def compute_padic_visibility(
        self,
        glycan_positions: torch.Tensor,
        query_position: int,
    ) -> torch.Tensor:
        """Compute p-adic visibility score for a query position.

        Lower values indicate the position is "close" to glycans in p-adic space
        and thus better shielded. Higher values indicate vulnerability.

        Args:
            glycan_positions: Tensor of glycan site positions
            query_position: Position to evaluate

        Returns:
            Visibility score (higher = more visible/vulnerable)
        """
        if len(glycan_positions) == 0:
            return torch.tensor(1.0)

        # Get p-adic distances from query to all glycans
        distances = self.padic_distances[query_position, glycan_positions]

        # Minimum distance determines visibility
        # In p-adic space, close means well-shielded
        min_distance = distances.min()

        return min_distance

    def identify_vulnerable_regions(
        self,
        glycan_positions: torch.Tensor,
        sequence_length: int,
        threshold: float = 0.5,
    ) -> list[tuple[int, int]]:
        """Identify regions vulnerable to antibody access.

        Args:
            glycan_positions: Tensor of glycan site positions
            sequence_length: Length of protein sequence
            threshold: Visibility threshold for vulnerability

        Returns:
            List of (start, end) tuples for vulnerable regions
        """
        vulnerabilities = []

        for pos in range(sequence_length):
            visibility = self.compute_padic_visibility(glycan_positions, pos)
            if visibility > threshold:
                vulnerabilities.append(pos)

        # Merge consecutive positions into regions
        regions = []
        if vulnerabilities:
            start = vulnerabilities[0]
            end = start

            for pos in vulnerabilities[1:]:
                if pos == end + 1:
                    end = pos
                else:
                    regions.append((start, end))
                    start = pos
                    end = pos
            regions.append((start, end))

        return regions

    def forward(
        self,
        glycan_positions: torch.Tensor,
        sequence_length: int,
    ) -> GlycanShieldMetrics:
        """Analyze glycan shield properties.

        Args:
            glycan_positions: Tensor of glycan site positions
            sequence_length: Length of protein sequence

        Returns:
            GlycanShieldMetrics dataclass
        """
        n_glycans = len(glycan_positions)

        if n_glycans == 0:
            return GlycanShieldMetrics(
                total_glycans=0,
                shield_coverage=0.0,
                vulnerable_area=1.0,
                avg_glycan_spacing=float("inf"),
                min_glycan_spacing=float("inf"),
                padic_visibility_score=1.0,
                sentinel_score=1.0,
            )

        # Compute coverage
        coverage = self.compute_shield_coverage(glycan_positions, sequence_length)

        # Compute glycan spacing
        sorted_pos = torch.sort(glycan_positions)[0]
        spacings = sorted_pos[1:] - sorted_pos[:-1]
        avg_spacing = spacings.float().mean().item() if len(spacings) > 0 else 0.0
        min_spacing = spacings.min().item() if len(spacings) > 0 else 0.0

        # Compute average p-adic visibility across all positions
        visibilities = []
        for pos in range(0, sequence_length, 10):  # Sample every 10th position
            vis = self.compute_padic_visibility(glycan_positions, pos)
            visibilities.append(vis.item())
        avg_visibility = sum(visibilities) / len(visibilities) if visibilities else 1.0

        # Sentinel score combines coverage and visibility
        sentinel = (1 - coverage.item()) * avg_visibility

        return GlycanShieldMetrics(
            total_glycans=n_glycans,
            shield_coverage=coverage.item(),
            vulnerable_area=1 - coverage.item(),
            avg_glycan_spacing=avg_spacing,
            min_glycan_spacing=min_spacing,
            padic_visibility_score=avg_visibility,
            sentinel_score=sentinel,
        )


class SentinelGlycanLoss(nn.Module):
    """Loss function for optimizing glycan shield vulnerability.

    This loss encourages the model to learn representations that
    distinguish between well-shielded and vulnerable epitopes.

    Use cases:
    - Training epitope visibility predictors
    - Optimizing vaccine immunogen designs
    - Analyzing glycan removal effects
    """

    def __init__(
        self,
        coverage_weight: float = 0.3,
        visibility_weight: float = 0.4,
        spacing_weight: float = 0.3,
        target_coverage: float = 0.7,  # Optimal shield coverage
        p: int = 3,
    ):
        """Initialize sentinel glycan loss.

        Args:
            coverage_weight: Weight for coverage term
            visibility_weight: Weight for visibility term
            spacing_weight: Weight for spacing uniformity term
            target_coverage: Target glycan coverage (for regularization)
            p: Prime for p-adic calculations
        """
        super().__init__()
        self.coverage_weight = coverage_weight
        self.visibility_weight = visibility_weight
        self.spacing_weight = spacing_weight
        self.target_coverage = target_coverage
        self.p = p

        self.analyzer = GlycanShieldAnalyzer(p=p)

    def forward(
        self,
        predicted_glycans: torch.Tensor,
        epitope_visibility: torch.Tensor,
        target_visibility: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute glycan shield optimization loss.

        Args:
            predicted_glycans: Predicted glycan occupancy probabilities (batch, seq_len)
            epitope_visibility: Predicted epitope visibility scores (batch, n_epitopes)
            target_visibility: Target visibility values (optional)

        Returns:
            Dictionary with total loss and component losses
        """
        batch_size = predicted_glycans.shape[0]
        losses = []

        for b in range(batch_size):
            # Get glycan positions (threshold probabilities)
            glycan_mask = predicted_glycans[b] > 0.5
            glycan_positions = torch.where(glycan_mask)[0]

            # Compute shield metrics
            seq_len = predicted_glycans.shape[1]
            metrics = self.analyzer(glycan_positions, seq_len)

            # Coverage loss: penalize deviation from target coverage
            coverage_loss = (metrics.shield_coverage - self.target_coverage) ** 2

            # Visibility loss: encourage low visibility (good shielding)
            visibility_loss = metrics.padic_visibility_score

            # Spacing loss: penalize uneven glycan distribution
            if metrics.avg_glycan_spacing > 0:
                spacing_variance = (
                    metrics.avg_glycan_spacing - metrics.min_glycan_spacing
                ) / metrics.avg_glycan_spacing
            else:
                spacing_variance = 0.0

            total = (
                self.coverage_weight * coverage_loss
                + self.visibility_weight * visibility_loss
                + self.spacing_weight * spacing_variance
            )
            losses.append(total)

        total_loss = torch.tensor(sum(losses) / len(losses)) if losses else torch.tensor(0.0)

        # If target visibility provided, add supervision
        supervision_loss = torch.tensor(0.0)
        if target_visibility is not None:
            supervision_loss = F.mse_loss(epitope_visibility, target_visibility)
            total_loss = total_loss + supervision_loss

        return {
            "loss": total_loss,
            "coverage_loss": torch.tensor(self.coverage_weight * sum(losses) / len(losses) if losses else 0.0),
            "visibility_loss": torch.tensor(self.visibility_weight * sum(losses) / len(losses) if losses else 0.0),
            "spacing_loss": torch.tensor(self.spacing_weight * sum(losses) / len(losses) if losses else 0.0),
            "supervision_loss": supervision_loss,
        }


class GlycanRemovalSimulator(nn.Module):
    """Simulates effects of glycan removal on epitope exposure.

    Used for vaccine design to identify which glycans to remove
    to expose vulnerable epitopes for antibody targeting.
    """

    def __init__(
        self,
        n_positions: int = 700,
        p: int = 3,
    ):
        """Initialize simulator.

        Args:
            n_positions: Maximum sequence length
            p: Prime for p-adic calculations
        """
        super().__init__()
        self.analyzer = GlycanShieldAnalyzer(n_positions=n_positions, p=p)

    def simulate_removal(
        self,
        glycan_positions: torch.Tensor,
        sequence_length: int,
        removal_indices: list[int],
    ) -> tuple[GlycanShieldMetrics, GlycanShieldMetrics]:
        """Simulate effect of removing specific glycans.

        Args:
            glycan_positions: Original glycan positions
            sequence_length: Sequence length
            removal_indices: Indices into glycan_positions to remove

        Returns:
            Tuple of (before_metrics, after_metrics)
        """
        # Compute original metrics
        before = self.analyzer(glycan_positions, sequence_length)

        # Remove specified glycans
        mask = torch.ones(len(glycan_positions), dtype=torch.bool)
        for idx in removal_indices:
            if 0 <= idx < len(glycan_positions):
                mask[idx] = False
        remaining = glycan_positions[mask]

        # Compute new metrics
        after = self.analyzer(remaining, sequence_length)

        return before, after

    def find_optimal_removals(
        self,
        glycan_positions: torch.Tensor,
        sequence_length: int,
        target_epitope: tuple[int, int],
        max_removals: int = 3,
    ) -> list[tuple[list[int], float]]:
        """Find optimal glycans to remove to expose a target epitope.

        Args:
            glycan_positions: Current glycan positions
            sequence_length: Sequence length
            target_epitope: (start, end) of epitope to expose
            max_removals: Maximum number of glycans to remove

        Returns:
            List of (removal_indices, exposure_improvement) sorted by improvement
        """
        epitope_start, epitope_end = target_epitope
        epitope_center = (epitope_start + epitope_end) // 2

        # Score each glycan by its proximity to target epitope
        results = []

        # Try removing each individual glycan
        for i in range(len(glycan_positions)):
            gpos = glycan_positions[i].item()

            # Only consider glycans near the epitope
            if abs(gpos - epitope_center) > 50:
                continue

            before, after = self.simulate_removal(glycan_positions, sequence_length, [i])

            # Compute visibility improvement at epitope center
            before_vis = self.analyzer.compute_padic_visibility(glycan_positions, epitope_center)
            remaining = glycan_positions[torch.arange(len(glycan_positions)) != i]
            after_vis = self.analyzer.compute_padic_visibility(remaining, epitope_center)

            improvement = (after_vis - before_vis).item()
            results.append(([i], improvement))

        # Sort by improvement (higher is better for exposure)
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:max_removals]

    def forward(
        self,
        glycan_positions: torch.Tensor,
        sequence_length: int,
        target_epitopes: list[tuple[int, int]],
    ) -> dict[str, Any]:
        """Analyze glycan removal strategies for multiple epitopes.

        Args:
            glycan_positions: Current glycan positions
            sequence_length: Sequence length
            target_epitopes: List of (start, end) epitopes to expose

        Returns:
            Dictionary with analysis results
        """
        original_metrics = self.analyzer(glycan_positions, sequence_length)

        epitope_analysis = []
        for epitope in target_epitopes:
            optimal = self.find_optimal_removals(glycan_positions, sequence_length, epitope)
            epitope_analysis.append(
                {
                    "epitope": epitope,
                    "recommended_removals": optimal,
                }
            )

        return {
            "original_metrics": original_metrics,
            "epitope_analysis": epitope_analysis,
            "total_glycans": len(glycan_positions),
        }
