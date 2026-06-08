# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Long COVID and SARS-CoV-2 Spike Protein Analysis.

This module analyzes SARS-CoV-2 spike protein post-translational modifications
(PTMs) and their relationship to Long COVID through the p-adic framework.

Key Features:
- Spike protein PTM site analysis
- Variant comparison using p-adic distances
- Goldilocks Zone classification for immunogenicity
- Chronic immune activation prediction

Hypothesis:
Spike protein PTMs that fall within specific p-adic "Goldilocks Zones" may
contribute to chronic immune activation observed in Long COVID patients.

Usage:
    from src.diseases.long_covid import LongCOVIDAnalyzer

    analyzer = LongCOVIDAnalyzer()
    result = analyzer.analyze_spike_ptms(spike_seq, ptm_sites=[(614, "D->G")])
    print(f"Immunogenicity: {result.immunogenicity_score:.2f}")

References:
    - DOCUMENTATION/.../04_LONG_COVID_MICROCLOTS.md
    - GISAID SARS-CoV-2 sequence database
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.core.padic_math import compute_goldilocks_score, padic_valuation


class PTMType(Enum):
    """Types of post-translational modifications."""

    GLYCOSYLATION = "glycosylation"  # N-linked or O-linked
    PHOSPHORYLATION = "phosphorylation"
    MUTATION = "mutation"  # Amino acid substitution
    DELETION = "deletion"
    INSERTION = "insertion"


class SpikeRegion(Enum):
    """Spike protein functional regions."""

    NTD = "N-terminal domain"  # aa 14-305
    RBD = "Receptor binding domain"  # aa 319-541
    RBM = "Receptor binding motif"  # aa 437-508 (within RBD)
    SD1 = "Subdomain 1"  # aa 542-591
    SD2 = "Subdomain 2"  # aa 592-685
    FP = "Fusion peptide"  # aa 788-806
    HR1 = "Heptad repeat 1"  # aa 912-984
    HR2 = "Heptad repeat 2"  # aa 1163-1213
    TM = "Transmembrane domain"  # aa 1214-1234
    CT = "Cytoplasmic tail"  # aa 1235-1273


@dataclass
class PTMSite:
    """Information about a post-translational modification site."""

    position: int
    ptm_type: PTMType
    original: str  # Original residue/sequence
    modified: str  # Modified residue/sequence
    region: SpikeRegion | None = None
    conservation_score: float = 0.0  # 0-1, how conserved the site is
    known_variant: str | None = None  # e.g., "Delta", "Omicron"


@dataclass
class SpikeVariant:
    """SARS-CoV-2 spike protein variant information."""

    name: str
    lineage: str  # e.g., "B.1.617.2"
    mutations: list[PTMSite]
    emergence_date: str | None = None
    transmissibility_factor: float = 1.0  # Relative to Wuhan
    immune_escape_score: float = 0.0  # 0-1


@dataclass
class PTMAnalysisResult:
    """Result of PTM analysis."""

    site: PTMSite
    padic_distance: float
    goldilocks_score: float
    immunogenicity_score: float
    chronic_risk: float  # Risk of contributing to chronic activation
    structural_impact: str  # "low", "medium", "high"


@dataclass
class LongCOVIDRiskProfile:
    """Long COVID risk profile based on spike PTM analysis."""

    ptm_results: list[PTMAnalysisResult]
    overall_risk_score: float
    chronic_activation_probability: float
    key_contributing_sites: list[int]  # Position numbers
    microclot_propensity: float
    recommendations: list[str] = field(default_factory=list)


# Known SARS-CoV-2 variants with their defining mutations
KNOWN_VARIANTS: dict[str, SpikeVariant] = {
    "wuhan": SpikeVariant(
        name="Wuhan-Hu-1",
        lineage="A",
        mutations=[],
        emergence_date="2019-12",
        transmissibility_factor=1.0,
        immune_escape_score=0.0,
    ),
    "alpha": SpikeVariant(
        name="Alpha",
        lineage="B.1.1.7",
        mutations=[
            PTMSite(69, PTMType.DELETION, "HV", "-", SpikeRegion.NTD),
            PTMSite(70, PTMType.DELETION, "HV", "-", SpikeRegion.NTD),
            PTMSite(144, PTMType.DELETION, "Y", "-", SpikeRegion.NTD),
            PTMSite(501, PTMType.MUTATION, "N", "Y", SpikeRegion.RBM),
            PTMSite(570, PTMType.MUTATION, "A", "D", SpikeRegion.SD1),
            PTMSite(681, PTMType.MUTATION, "P", "H", SpikeRegion.SD2),
            PTMSite(716, PTMType.MUTATION, "T", "I", None),
            PTMSite(982, PTMType.MUTATION, "S", "A", SpikeRegion.HR1),
            PTMSite(1118, PTMType.MUTATION, "D", "H", None),
        ],
        emergence_date="2020-09",
        transmissibility_factor=1.5,
        immune_escape_score=0.2,
    ),
    "delta": SpikeVariant(
        name="Delta",
        lineage="B.1.617.2",
        mutations=[
            PTMSite(19, PTMType.MUTATION, "T", "R", SpikeRegion.NTD),
            PTMSite(157, PTMType.DELETION, "F", "-", SpikeRegion.NTD),
            PTMSite(158, PTMType.DELETION, "R", "-", SpikeRegion.NTD),
            PTMSite(452, PTMType.MUTATION, "L", "R", SpikeRegion.RBD),
            PTMSite(478, PTMType.MUTATION, "T", "K", SpikeRegion.RBM),
            PTMSite(614, PTMType.MUTATION, "D", "G", SpikeRegion.SD2),
            PTMSite(681, PTMType.MUTATION, "P", "R", SpikeRegion.SD2),
            PTMSite(950, PTMType.MUTATION, "D", "N", SpikeRegion.HR1),
        ],
        emergence_date="2020-10",
        transmissibility_factor=2.0,
        immune_escape_score=0.35,
    ),
    "omicron_ba1": SpikeVariant(
        name="Omicron BA.1",
        lineage="B.1.1.529",
        mutations=[
            PTMSite(67, PTMType.MUTATION, "A", "V", SpikeRegion.NTD),
            PTMSite(69, PTMType.DELETION, "H", "-", SpikeRegion.NTD),
            PTMSite(70, PTMType.DELETION, "V", "-", SpikeRegion.NTD),
            PTMSite(143, PTMType.DELETION, "VYY", "-", SpikeRegion.NTD),
            PTMSite(339, PTMType.MUTATION, "G", "D", SpikeRegion.RBD),
            PTMSite(371, PTMType.MUTATION, "S", "L", SpikeRegion.RBD),
            PTMSite(373, PTMType.MUTATION, "S", "P", SpikeRegion.RBD),
            PTMSite(375, PTMType.MUTATION, "S", "F", SpikeRegion.RBD),
            PTMSite(417, PTMType.MUTATION, "K", "N", SpikeRegion.RBD),
            PTMSite(440, PTMType.MUTATION, "N", "K", SpikeRegion.RBM),
            PTMSite(446, PTMType.MUTATION, "G", "S", SpikeRegion.RBM),
            PTMSite(477, PTMType.MUTATION, "S", "N", SpikeRegion.RBM),
            PTMSite(478, PTMType.MUTATION, "T", "K", SpikeRegion.RBM),
            PTMSite(484, PTMType.MUTATION, "E", "A", SpikeRegion.RBM),
            PTMSite(493, PTMType.MUTATION, "Q", "R", SpikeRegion.RBM),
            PTMSite(496, PTMType.MUTATION, "G", "S", SpikeRegion.RBM),
            PTMSite(498, PTMType.MUTATION, "Q", "R", SpikeRegion.RBM),
            PTMSite(501, PTMType.MUTATION, "N", "Y", SpikeRegion.RBM),
            PTMSite(505, PTMType.MUTATION, "Y", "H", SpikeRegion.RBM),
            PTMSite(547, PTMType.MUTATION, "T", "K", SpikeRegion.SD1),
            PTMSite(614, PTMType.MUTATION, "D", "G", SpikeRegion.SD2),
            PTMSite(655, PTMType.MUTATION, "H", "Y", SpikeRegion.SD2),
            PTMSite(679, PTMType.MUTATION, "N", "K", SpikeRegion.SD2),
            PTMSite(681, PTMType.MUTATION, "P", "H", SpikeRegion.SD2),
            PTMSite(764, PTMType.MUTATION, "N", "K", None),
            PTMSite(796, PTMType.MUTATION, "D", "Y", SpikeRegion.FP),
            PTMSite(856, PTMType.MUTATION, "N", "K", None),
            PTMSite(954, PTMType.MUTATION, "Q", "H", SpikeRegion.HR1),
            PTMSite(969, PTMType.MUTATION, "N", "K", SpikeRegion.HR1),
            PTMSite(981, PTMType.MUTATION, "L", "F", SpikeRegion.HR1),
        ],
        emergence_date="2021-11",
        transmissibility_factor=3.0,
        immune_escape_score=0.7,
    ),
}

# Glycosylation sites on spike protein (conserved N-linked glycans)
SPIKE_GLYCOSYLATION_SITES: list[int] = [
    17,
    61,
    74,
    122,
    149,
    165,
    234,
    282,
    331,
    343,
    603,
    616,
    657,
    709,
    717,
    801,
    1074,
    1098,
    1134,
    1158,
    1173,
    1194,
]


class LongCOVIDAnalyzer:
    """Analyzer for Long COVID and spike protein PTMs.

    Uses p-adic mathematics to model the relationship between spike
    protein modifications and chronic immune activation in Long COVID.
    """

    def __init__(self, p: int = 3, goldilocks_center: float = 0.5):
        """Initialize the analyzer.

        Args:
            p: Prime base for p-adic calculations (3 for ternary)
            goldilocks_center: Center of the Goldilocks Zone (0-1)
        """
        self.p = p
        self.goldilocks_center = goldilocks_center
        self.goldilocks_width = 0.2
        self.spike_length = 1273  # Full spike protein length

    def _get_region(self, position: int) -> SpikeRegion | None:
        """Determine spike protein region for a position."""
        region_ranges = {
            SpikeRegion.NTD: (14, 305),
            SpikeRegion.RBD: (319, 541),
            SpikeRegion.RBM: (437, 508),
            SpikeRegion.SD1: (542, 591),
            SpikeRegion.SD2: (592, 685),
            SpikeRegion.FP: (788, 806),
            SpikeRegion.HR1: (912, 984),
            SpikeRegion.HR2: (1163, 1213),
            SpikeRegion.TM: (1214, 1234),
            SpikeRegion.CT: (1235, 1273),
        }

        for region, (start, end) in region_ranges.items():
            if start <= position <= end:
                return region
        return None

    def _compute_padic_valuation(self, n: int) -> int:
        """Compute p-adic valuation v_p(n).

        Uses centralized padic_valuation from src.core.padic_math.
        """
        return padic_valuation(n, self.p)

    def _compute_position_padic_distance(self, position: int, reference: int = 614) -> float:
        """Compute p-adic distance of position from reference.

        D614G is used as reference as it's a key mutation for viral fitness.

        Args:
            position: Amino acid position
            reference: Reference position (default D614G)

        Returns:
            Normalized p-adic distance (0-1)
        """
        diff = abs(position - reference)
        if diff == 0:
            return 0.0

        valuation = self._compute_padic_valuation(diff)
        distance = 1.0 / (self.p**valuation)
        return distance

    def _compute_immunogenicity(self, site: PTMSite) -> float:
        """Estimate immunogenicity of a PTM site.

        Args:
            site: PTM site information

        Returns:
            Immunogenicity score (0-1)
        """
        base_score = 0.5

        # Region-based immunogenicity modifiers
        region_weights = {
            SpikeRegion.RBD: 1.5,  # High - antibody target
            SpikeRegion.RBM: 1.8,  # Highest - ACE2 binding interface
            SpikeRegion.NTD: 1.3,  # Moderate - supersite
            SpikeRegion.SD1: 0.8,
            SpikeRegion.SD2: 0.9,
            SpikeRegion.FP: 1.2,  # Important for fusion
            SpikeRegion.HR1: 0.7,
            SpikeRegion.HR2: 0.7,
            SpikeRegion.TM: 0.3,  # Low - membrane bound
            SpikeRegion.CT: 0.2,  # Lowest - intracellular
        }

        region = site.region or self._get_region(site.position)
        if region:
            base_score *= region_weights.get(region, 1.0)

        # PTM type modifiers
        ptm_weights = {
            PTMType.MUTATION: 1.0,
            PTMType.DELETION: 1.2,  # Deletions often affect antigenicity
            PTMType.INSERTION: 1.1,
            PTMType.GLYCOSYLATION: 0.8,  # Glycans can shield epitopes
            PTMType.PHOSPHORYLATION: 0.9,
        }
        base_score *= ptm_weights.get(site.ptm_type, 1.0)

        # Proximity to glycosylation sites (shielding effect)
        for glyco_site in SPIKE_GLYCOSYLATION_SITES:
            if abs(site.position - glyco_site) < 10:
                base_score *= 0.85
                break

        return min(1.0, max(0.0, base_score))

    def _compute_goldilocks_score(self, padic_distance: float) -> float:
        """Compute Goldilocks Zone score.

        Uses centralized compute_goldilocks_score from src.core.padic_math.

        Args:
            padic_distance: P-adic distance value

        Returns:
            Score from 0 (outside zone) to 1 (center of zone)
        """
        return compute_goldilocks_score(padic_distance, center=self.goldilocks_center, width=self.goldilocks_width)

    def _compute_chronic_risk(self, site: PTMSite, immunogenicity: float, goldilocks: float) -> float:
        """Compute risk of chronic immune activation.

        Args:
            site: PTM site
            immunogenicity: Immunogenicity score
            goldilocks: Goldilocks Zone score

        Returns:
            Chronic activation risk (0-1)
        """
        # Base risk from immunogenicity and Goldilocks score
        risk = 0.5 * immunogenicity + 0.3 * goldilocks

        # Additional risk factors for RBD/RBM mutations
        region = site.region or self._get_region(site.position)
        if region in [SpikeRegion.RBD, SpikeRegion.RBM]:
            risk *= 1.3

        # Deletions may create neo-epitopes
        if site.ptm_type == PTMType.DELETION:
            risk *= 1.2

        return min(1.0, max(0.0, risk))

    def _assess_structural_impact(self, site: PTMSite) -> str:
        """Assess structural impact of a PTM.

        Args:
            site: PTM site

        Returns:
            Impact level: "low", "medium", or "high"
        """
        # Deletions/insertions have higher structural impact
        if site.ptm_type in [PTMType.DELETION, PTMType.INSERTION]:
            return "high"

        # Mutations in key regions
        region = site.region or self._get_region(site.position)
        high_impact_regions = [SpikeRegion.RBD, SpikeRegion.RBM, SpikeRegion.FP]
        if region in high_impact_regions:
            return "medium" if site.ptm_type == PTMType.MUTATION else "high"

        # Check for radical amino acid changes
        if site.ptm_type == PTMType.MUTATION:
            charged = set("DEKRH")
            hydrophobic = set("AVILMFYW")

            orig_charged = site.original.upper() in charged
            mod_charged = site.modified.upper() in charged
            orig_hydrophobic = site.original.upper() in hydrophobic
            mod_hydrophobic = site.modified.upper() in hydrophobic

            if orig_charged != mod_charged or orig_hydrophobic != mod_hydrophobic:
                return "medium"

        return "low"

    def analyze_ptm(self, site: PTMSite) -> PTMAnalysisResult:
        """Analyze a single PTM site.

        Args:
            site: PTM site to analyze

        Returns:
            Analysis result with p-adic metrics
        """
        padic_distance = self._compute_position_padic_distance(site.position)
        goldilocks_score = self._compute_goldilocks_score(padic_distance)
        immunogenicity = self._compute_immunogenicity(site)
        chronic_risk = self._compute_chronic_risk(site, immunogenicity, goldilocks_score)
        structural_impact = self._assess_structural_impact(site)

        return PTMAnalysisResult(
            site=site,
            padic_distance=padic_distance,
            goldilocks_score=goldilocks_score,
            immunogenicity_score=immunogenicity,
            chronic_risk=chronic_risk,
            structural_impact=structural_impact,
        )

    def analyze_spike_ptms(
        self,
        spike_sequence: str | None = None,
        ptm_sites: list[PTMSite] | None = None,
    ) -> LongCOVIDRiskProfile:
        """Analyze spike protein PTMs for Long COVID risk.

        Args:
            spike_sequence: Optional spike protein sequence
            ptm_sites: List of PTM sites to analyze

        Returns:
            Complete Long COVID risk profile
        """
        if ptm_sites is None:
            ptm_sites = []

        # Analyze each PTM site
        results = [self.analyze_ptm(site) for site in ptm_sites]

        # Calculate aggregate metrics
        if results:
            overall_risk = np.mean([r.chronic_risk for r in results])
            chronic_prob = 1 - np.prod([1 - r.chronic_risk * 0.3 for r in results])

            # Identify key contributing sites
            key_sites = [r.site.position for r in results if r.chronic_risk > 0.5]

            # Microclot propensity based on RBD/RBM mutations
            rbd_mutations = [r for r in results if r.site.region in [SpikeRegion.RBD, SpikeRegion.RBM]]
            microclot_propensity = len(rbd_mutations) / max(len(results), 1)
        else:
            overall_risk = 0.0
            chronic_prob = 0.0
            key_sites = []
            microclot_propensity = 0.0

        # Generate recommendations
        recommendations = self._generate_recommendations(results)

        return LongCOVIDRiskProfile(
            ptm_results=results,
            overall_risk_score=float(overall_risk),
            chronic_activation_probability=float(chronic_prob),
            key_contributing_sites=key_sites,
            microclot_propensity=float(microclot_propensity),
            recommendations=recommendations,
        )

    def _generate_recommendations(self, results: list[PTMAnalysisResult]) -> list[str]:
        """Generate recommendations based on analysis.

        Args:
            results: List of PTM analysis results

        Returns:
            List of recommendation strings
        """
        recommendations = []

        high_risk_sites = [r for r in results if r.chronic_risk > 0.7]
        if high_risk_sites:
            positions = [str(r.site.position) for r in high_risk_sites]
            recommendations.append(f"Monitor positions {', '.join(positions)} for chronic immune activation")

        rbd_mutations = [r for r in results if r.site.region in [SpikeRegion.RBD, SpikeRegion.RBM]]
        if len(rbd_mutations) > 3:
            recommendations.append("High RBD mutation load may indicate immune escape variant")

        goldilocks_sites = [r for r in results if r.goldilocks_score > 0.8]
        if goldilocks_sites:
            recommendations.append("Sites in Goldilocks Zone may contribute to persistent symptoms")

        return recommendations

    def compare_variants(
        self,
        variant_names: list[str] | None = None,
    ) -> dict[str, LongCOVIDRiskProfile]:
        """Compare Long COVID risk profiles across variants.

        Args:
            variant_names: List of variant names to compare (defaults to all known)

        Returns:
            Dictionary mapping variant name to risk profile
        """
        if variant_names is None:
            variant_names = list(KNOWN_VARIANTS.keys())

        results = {}
        for name in variant_names:
            if name in KNOWN_VARIANTS:
                variant = KNOWN_VARIANTS[name]
                profile = self.analyze_spike_ptms(ptm_sites=variant.mutations)
                results[name] = profile

        return results

    def predict_chronic_immune_activation(
        self,
        ptm_profile: LongCOVIDRiskProfile,
        patient_age: int | None = None,
        comorbidities: list[str] | None = None,
    ) -> dict[str, float]:
        """Predict likelihood of chronic immune response.

        Args:
            ptm_profile: PTM analysis profile
            patient_age: Optional patient age for risk adjustment
            comorbidities: Optional list of comorbidities

        Returns:
            Dictionary of risk predictions
        """
        base_risk = ptm_profile.chronic_activation_probability

        # Age adjustment (higher risk for older patients)
        age_factor = 1.0
        if patient_age is not None:
            if patient_age > 60:
                age_factor = 1.3
            elif patient_age > 40:
                age_factor = 1.1
            elif patient_age < 20:
                age_factor = 0.8

        # Comorbidity adjustment
        comorbidity_factor = 1.0
        if comorbidities:
            high_risk_conditions = ["diabetes", "obesity", "autoimmune", "cardiovascular"]
            matching = sum(1 for c in comorbidities if any(h in c.lower() for h in high_risk_conditions))
            comorbidity_factor = 1.0 + (0.15 * matching)

        adjusted_risk = min(1.0, base_risk * age_factor * comorbidity_factor)

        return {
            "base_risk": base_risk,
            "age_adjusted_risk": base_risk * age_factor,
            "final_risk": adjusted_risk,
            "long_covid_probability": adjusted_risk * 0.5,  # Conservative estimate
            "chronic_fatigue_risk": adjusted_risk * 0.6,
            "neurological_risk": adjusted_risk * 0.4,
            "cardiovascular_risk": adjusted_risk * ptm_profile.microclot_propensity,
        }


class SpikeVariantComparator:
    """Compare SARS-CoV-2 spike variants using p-adic metrics."""

    def __init__(self, p: int = 3):
        """Initialize comparator.

        Args:
            p: Prime base for p-adic calculations
        """
        self.p = p
        self.analyzer = LongCOVIDAnalyzer(p=p)

    def _compute_padic_valuation(self, n: int) -> int:
        """Compute p-adic valuation.

        Uses centralized padic_valuation from src.core.padic_math.
        """
        return padic_valuation(n, self.p)

    def compute_variant_distance(self, variant1: str, variant2: str) -> float:
        """Compute p-adic distance between two variants.

        Args:
            variant1: First variant name
            variant2: Second variant name

        Returns:
            P-adic distance between variants
        """
        if variant1 not in KNOWN_VARIANTS or variant2 not in KNOWN_VARIANTS:
            raise ValueError(f"Unknown variant. Available: {list(KNOWN_VARIANTS.keys())}")

        v1 = KNOWN_VARIANTS[variant1]
        v2 = KNOWN_VARIANTS[variant2]

        # Get mutation positions
        pos1 = set(m.position for m in v1.mutations)
        pos2 = set(m.position for m in v2.mutations)

        # Symmetric difference of mutation positions
        unique_to_1 = pos1 - pos2
        unique_to_2 = pos2 - pos1

        if not unique_to_1 and not unique_to_2:
            return 0.0  # Identical mutation profiles

        # Average p-adic distance of differing positions
        all_unique = list(unique_to_1 | unique_to_2)
        distances = []

        for pos in all_unique:
            valuation = self._compute_padic_valuation(pos)
            dist = 1.0 / (self.p**valuation)
            distances.append(dist)

        return float(np.mean(distances)) if distances else 0.0

    def compute_distance_matrix(
        self,
        variant_names: list[str] | None = None,
    ) -> tuple[list[str], np.ndarray]:
        """Compute pairwise distance matrix for variants.

        Args:
            variant_names: List of variants to compare (defaults to all)

        Returns:
            (variant_names, distance_matrix)
        """
        if variant_names is None:
            variant_names = list(KNOWN_VARIANTS.keys())

        n = len(variant_names)
        matrix = np.zeros((n, n))

        for i, v1 in enumerate(variant_names):
            for j, v2 in enumerate(variant_names):
                if i < j:
                    dist = self.compute_variant_distance(v1, v2)
                    matrix[i, j] = dist
                    matrix[j, i] = dist

        return variant_names, matrix

    def find_closest_variant(self, target_mutations: list[PTMSite]) -> tuple[str, float]:
        """Find the known variant closest to a set of mutations.

        Args:
            target_mutations: List of PTM sites

        Returns:
            (variant_name, distance)
        """
        target_positions = set(m.position for m in target_mutations)

        best_variant = None
        best_distance = float("inf")

        for name, variant in KNOWN_VARIANTS.items():
            var_positions = set(m.position for m in variant.mutations)

            # Jaccard-like distance
            intersection = len(target_positions & var_positions)
            union = len(target_positions | var_positions)

            if union > 0:
                similarity = intersection / union
                distance = 1 - similarity
            else:
                distance = 1.0 if target_positions else 0.0

            if distance < best_distance:
                best_distance = distance
                best_variant = name

        return best_variant or "unknown", best_distance

    def rank_variants_by_risk(self) -> list[tuple[str, float]]:
        """Rank known variants by Long COVID risk.

        Returns:
            List of (variant_name, risk_score) tuples, sorted by risk
        """
        profiles = self.analyzer.compare_variants()

        rankings = [(name, profile.overall_risk_score) for name, profile in profiles.items()]

        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings


__all__ = [
    "LongCOVIDAnalyzer",
    "SpikeVariantComparator",
    "PTMType",
    "PTMSite",
    "SpikeRegion",
    "SpikeVariant",
    "PTMAnalysisResult",
    "LongCOVIDRiskProfile",
    "KNOWN_VARIANTS",
    "SPIKE_GLYCOSYLATION_SITES",
]
