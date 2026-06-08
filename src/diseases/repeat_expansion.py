# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Trinucleotide Repeat Expansion Disease Analysis.

This module analyzes diseases caused by trinucleotide repeat expansions
through the lens of p-adic mathematics and the Goldilocks Zone hypothesis.

Key Diseases:
- Huntington's Disease (HD): CAG repeats in HTT gene
- Spinocerebellar Ataxias (SCA): CAG repeats in various ATXN genes
- Fragile X Syndrome: CGG repeats in FMR1 gene
- Myotonic Dystrophy (DM): CTG repeats in DMPK gene

Hypothesis:
The disease threshold (e.g., 36 CAG repeats for HD) corresponds to entry
into the p-adic "Goldilocks Zone" where the repeat count creates specific
ultrametric distances that trigger pathological protein aggregation.

Usage:
    from src.diseases.repeat_expansion import RepeatExpansionAnalyzer

    analyzer = RepeatExpansionAnalyzer()
    result = analyzer.analyze_repeat_padic_distance("huntington", 42)
    print(f"Disease risk: {result.risk_score:.2f}")

References:
    - DOCUMENTATION/.../05_HUNTINGTONS_DISEASE_REPEATS.md
    - Khrennikov (2006) p-adic analysis methods
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.core.padic_math import padic_distance, padic_valuation


class TrinucleotideRepeat(Enum):
    """Trinucleotide repeat types."""

    CAG = "CAG"  # Polyglutamine (Huntington's, SCAs)
    CGG = "CGG"  # Polyarginine (Fragile X)
    CTG = "CTG"  # Polyglutamine/Polyleucine (DM1)
    GAA = "GAA"  # Friedreich's ataxia
    GGC = "GGC"  # Various
    CCG = "CCG"  # Fragile XE


@dataclass
class RepeatDiseaseInfo:
    """Information about a trinucleotide repeat disease."""

    name: str
    repeat: TrinucleotideRepeat
    gene: str
    normal_range: tuple[int, int]  # Normal repeat count range
    intermediate_range: tuple[int, int]  # Pre-mutation/intermediate
    disease_threshold: int  # Minimum for disease
    full_penetrance: int  # Full disease penetrance
    anticipation: bool  # Whether disease shows anticipation
    protein_product: str  # Affected protein

    def is_normal(self, repeat_count: int) -> bool:
        """Check if repeat count is in normal range."""
        return self.normal_range[0] <= repeat_count <= self.normal_range[1]

    def is_intermediate(self, repeat_count: int) -> bool:
        """Check if repeat count is in intermediate/pre-mutation range."""
        return self.intermediate_range[0] <= repeat_count <= self.intermediate_range[1]

    def is_disease(self, repeat_count: int) -> bool:
        """Check if repeat count causes disease."""
        return repeat_count >= self.disease_threshold


# Database of repeat expansion diseases
REPEAT_DISEASES: dict[str, RepeatDiseaseInfo] = {
    "huntington": RepeatDiseaseInfo(
        name="Huntington's Disease",
        repeat=TrinucleotideRepeat.CAG,
        gene="HTT",
        normal_range=(10, 26),
        intermediate_range=(27, 35),
        disease_threshold=36,
        full_penetrance=40,
        anticipation=True,
        protein_product="Huntingtin",
    ),
    "sca1": RepeatDiseaseInfo(
        name="Spinocerebellar Ataxia Type 1",
        repeat=TrinucleotideRepeat.CAG,
        gene="ATXN1",
        normal_range=(6, 35),
        intermediate_range=(36, 38),
        disease_threshold=39,
        full_penetrance=44,
        anticipation=True,
        protein_product="Ataxin-1",
    ),
    "sca3": RepeatDiseaseInfo(
        name="Spinocerebellar Ataxia Type 3 (Machado-Joseph)",
        repeat=TrinucleotideRepeat.CAG,
        gene="ATXN3",
        normal_range=(12, 40),
        intermediate_range=(41, 51),
        disease_threshold=52,
        full_penetrance=60,
        anticipation=True,
        protein_product="Ataxin-3",
    ),
    "fragile_x": RepeatDiseaseInfo(
        name="Fragile X Syndrome",
        repeat=TrinucleotideRepeat.CGG,
        gene="FMR1",
        normal_range=(5, 44),
        intermediate_range=(45, 54),
        disease_threshold=200,
        full_penetrance=200,
        anticipation=True,
        protein_product="FMRP",
    ),
    "myotonic_dystrophy": RepeatDiseaseInfo(
        name="Myotonic Dystrophy Type 1",
        repeat=TrinucleotideRepeat.CTG,
        gene="DMPK",
        normal_range=(5, 34),
        intermediate_range=(35, 49),
        disease_threshold=50,
        full_penetrance=150,
        anticipation=True,
        protein_product="DMPK",
    ),
    "friedreich_ataxia": RepeatDiseaseInfo(
        name="Friedreich's Ataxia",
        repeat=TrinucleotideRepeat.GAA,
        gene="FXN",
        normal_range=(5, 33),
        intermediate_range=(34, 65),
        disease_threshold=66,
        full_penetrance=200,
        anticipation=False,  # Unique: shows expansion in somatic cells
        protein_product="Frataxin",
    ),
}


@dataclass
class RepeatAnalysisResult:
    """Result of repeat expansion analysis."""

    disease: str
    repeat_count: int
    padic_distance: float
    goldilocks_score: float
    risk_score: float
    classification: str  # "normal", "intermediate", "disease"
    aggregation_propensity: float
    predicted_onset_age: int | None


class RepeatExpansionAnalyzer:
    """Analyzer for trinucleotide repeat expansion diseases.

    Uses p-adic mathematics to model the relationship between repeat
    count and disease phenotype, based on the hypothesis that disease
    thresholds correspond to entry into specific ultrametric regions.
    """

    def __init__(self, p: int = 3, goldilocks_center: float = 0.5):
        """Initialize the analyzer.

        Args:
            p: Prime base for p-adic calculations (3 for ternary)
            goldilocks_center: Center of the Goldilocks Zone (0-1)
        """
        self.p = p
        self.goldilocks_center = goldilocks_center
        self.goldilocks_width = 0.2  # Width of Goldilocks Zone

    def _compute_padic_valuation(self, n: int) -> int:
        """Compute p-adic valuation v_p(n).

        Uses centralized padic_valuation from src.core.padic_math.
        """
        return padic_valuation(n, self.p)

    def _compute_repeat_padic_distance(self, repeat_count: int, threshold: int) -> float:
        """Compute p-adic distance of repeat count from disease threshold.

        Uses centralized padic_distance from src.core.padic_math.

        The distance measures how "far" the repeat count is from the
        pathological threshold in p-adic terms.

        Args:
            repeat_count: Number of trinucleotide repeats
            threshold: Disease threshold

        Returns:
            Normalized p-adic distance (0-1)
        """
        return padic_distance(repeat_count, threshold, self.p)

    def _compute_goldilocks_score(self, repeat_count: int, disease_info: RepeatDiseaseInfo) -> float:
        """Compute how close the repeat count is to Goldilocks Zone.

        The Goldilocks Zone represents the "just right" conditions for
        pathological protein behavior - not too stable (normal) and not
        too unstable (toxic aggregates).

        Args:
            repeat_count: Number of repeats
            disease_info: Disease information

        Returns:
            Score from 0 (outside zone) to 1 (center of zone)
        """
        # Map repeat count to normalized position
        range_size = disease_info.full_penetrance - disease_info.normal_range[0]
        if range_size <= 0:
            return 0.0

        normalized_pos = (repeat_count - disease_info.normal_range[0]) / range_size
        normalized_pos = max(0, min(1, normalized_pos))

        # Distance from Goldilocks center
        distance_from_center = abs(normalized_pos - self.goldilocks_center)

        # Gaussian-like score centered at Goldilocks Zone
        score = np.exp(-(distance_from_center**2) / (2 * self.goldilocks_width**2))

        return float(score)

    def _compute_aggregation_propensity(self, repeat_count: int, repeat_type: TrinucleotideRepeat) -> float:
        """Estimate protein aggregation propensity based on repeat count.

        Longer polyglutamine tracks (from CAG repeats) have higher
        aggregation propensity due to hydrogen bonding networks.

        Args:
            repeat_count: Number of repeats
            repeat_type: Type of trinucleotide repeat

        Returns:
            Aggregation propensity score (0-1)
        """
        # Base propensity depends on repeat type
        base_propensity = {
            TrinucleotideRepeat.CAG: 0.8,  # Polyglutamine - high aggregation
            TrinucleotideRepeat.CGG: 0.5,  # Polyarginine - moderate
            TrinucleotideRepeat.CTG: 0.6,  # Mixed
            TrinucleotideRepeat.GAA: 0.3,  # RNA-mediated
            TrinucleotideRepeat.GGC: 0.4,
            TrinucleotideRepeat.CCG: 0.4,
        }

        base = base_propensity.get(repeat_type, 0.5)

        # Exponential increase with repeat count (sigmoid-like)
        # Threshold around 35-40 for polyglutamine
        midpoint = 38
        steepness = 0.15
        length_factor = 1 / (1 + np.exp(-steepness * (repeat_count - midpoint)))

        return float(base * length_factor)

    def _predict_onset_age(self, repeat_count: int, disease_info: RepeatDiseaseInfo) -> int | None:
        """Predict age of disease onset based on repeat count.

        For HD and other CAG diseases, there's an inverse correlation
        between repeat count and age of onset.

        Args:
            repeat_count: Number of repeats
            disease_info: Disease information

        Returns:
            Predicted onset age in years, or None if not applicable
        """
        if not disease_info.is_disease(repeat_count):
            return None

        # Huntington's disease model (validated clinical data)
        if disease_info.gene == "HTT":
            # Langbehn formula approximation
            # Age = 21.54 + exp(9.556 - 0.146 * CAG)
            if repeat_count >= 36:
                age = 21.54 + np.exp(9.556 - 0.146 * repeat_count)
                return max(1, min(100, int(age)))

        # Generic model for other diseases
        excess_repeats = repeat_count - disease_info.disease_threshold
        base_age = 50
        reduction_per_repeat = 1.0

        predicted = base_age - (excess_repeats * reduction_per_repeat)
        return max(1, min(100, int(predicted)))

    def analyze_repeat_padic_distance(
        self,
        disease: str,
        repeat_count: int,
    ) -> RepeatAnalysisResult:
        """Analyze a specific repeat count for a disease.

        Args:
            disease: Disease name (key in REPEAT_DISEASES)
            repeat_count: Number of trinucleotide repeats

        Returns:
            Complete analysis result
        """
        if disease not in REPEAT_DISEASES:
            raise ValueError(f"Unknown disease: {disease}. Available: {list(REPEAT_DISEASES.keys())}")

        info = REPEAT_DISEASES[disease]

        # Compute metrics
        padic_distance = self._compute_repeat_padic_distance(repeat_count, info.disease_threshold)
        goldilocks_score = self._compute_goldilocks_score(repeat_count, info)
        aggregation = self._compute_aggregation_propensity(repeat_count, info.repeat)
        onset_age = self._predict_onset_age(repeat_count, info)

        # Classification
        if info.is_normal(repeat_count):
            classification = "normal"
            risk_score = 0.1 * goldilocks_score
        elif info.is_intermediate(repeat_count):
            classification = "intermediate"
            risk_score = 0.3 + 0.3 * goldilocks_score
        else:
            classification = "disease"
            risk_score = 0.7 + 0.3 * aggregation

        return RepeatAnalysisResult(
            disease=disease,
            repeat_count=repeat_count,
            padic_distance=padic_distance,
            goldilocks_score=goldilocks_score,
            risk_score=risk_score,
            classification=classification,
            aggregation_propensity=aggregation,
            predicted_onset_age=onset_age,
        )

    def find_disease_boundary(
        self,
        disease: str,
        search_range: tuple[int, int] = (1, 100),
    ) -> dict[str, int]:
        """Find repeat counts corresponding to phase transitions.

        Identifies boundaries between normal, intermediate, and disease
        phases using p-adic distance analysis.

        Args:
            disease: Disease name
            search_range: Range of repeat counts to analyze

        Returns:
            Dictionary with boundary repeat counts
        """
        if disease not in REPEAT_DISEASES:
            raise ValueError(f"Unknown disease: {disease}")

        info = REPEAT_DISEASES[disease]

        # Track p-adic distances
        distances = []
        goldilocks_scores = []

        for n in range(search_range[0], search_range[1] + 1):
            dist = self._compute_repeat_padic_distance(n, info.disease_threshold)
            gs = self._compute_goldilocks_score(n, info)
            distances.append((n, dist))
            goldilocks_scores.append((n, gs))

        # Find transition points (local maxima in Goldilocks score derivative)
        gs_array = np.array([gs for _, gs in goldilocks_scores])
        gradient = np.gradient(gs_array)

        # Find where gradient changes sign (inflection points)
        sign_changes = np.where(np.diff(np.sign(gradient)))[0]

        boundaries = {
            "clinical_normal_end": info.normal_range[1],
            "intermediate_start": info.intermediate_range[0],
            "disease_threshold": info.disease_threshold,
            "full_penetrance": info.full_penetrance,
        }

        # Add p-adic derived boundaries
        if len(sign_changes) > 0:
            boundaries["padic_transition_1"] = int(search_range[0] + sign_changes[0])
        if len(sign_changes) > 1:
            boundaries["padic_transition_2"] = int(search_range[0] + sign_changes[1])

        return boundaries

    def compare_diseases(self, repeat_count: int) -> list[RepeatAnalysisResult]:
        """Compare risk across all diseases for a given repeat count.

        Args:
            repeat_count: Number of repeats to analyze

        Returns:
            List of analysis results for each disease
        """
        results = []
        for disease in REPEAT_DISEASES:
            result = self.analyze_repeat_padic_distance(disease, repeat_count)
            results.append(result)

        # Sort by risk score
        results.sort(key=lambda r: r.risk_score, reverse=True)
        return results

    def generate_risk_trajectory(
        self,
        disease: str,
        repeat_range: tuple[int, int] = (10, 80),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate risk trajectory across repeat counts.

        Useful for visualization of how risk changes with repeat count.

        Args:
            disease: Disease name
            repeat_range: Range of repeat counts

        Returns:
            (repeat_counts, risk_scores, goldilocks_scores) arrays
        """
        counts = np.arange(repeat_range[0], repeat_range[1] + 1)
        risks = []
        goldilocks = []

        for n in counts:
            result = self.analyze_repeat_padic_distance(disease, int(n))
            risks.append(result.risk_score)
            goldilocks.append(result.goldilocks_score)

        return counts, np.array(risks), np.array(goldilocks)


__all__ = [
    "RepeatExpansionAnalyzer",
    "RepeatDiseaseInfo",
    "RepeatAnalysisResult",
    "TrinucleotideRepeat",
    "REPEAT_DISEASES",
]
