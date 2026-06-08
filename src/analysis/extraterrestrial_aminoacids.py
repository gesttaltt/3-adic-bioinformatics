# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Extraterrestrial Amino Acid Analysis.

This module analyzes amino acid distributions from extraterrestrial sources
(asteroids, meteorites) to test the universality of the genetic code.

Key Questions:
- Are amino acids from space compatible with Earth's genetic code?
- Does p-adic geometry reveal universal code optimization?
- What amino acids are preferentially selected in abiotic conditions?

Data Sources:
- NASA OSIRIS-REx (Asteroid Bennu)
- Murchison meteorite (1969, Australia)
- Tagish Lake meteorite (2000, Canada)
- Laboratory abiotic synthesis experiments

Usage:
    from src.analysis.extraterrestrial_aminoacids import AsteroidAminoAcidAnalyzer

    analyzer = AsteroidAminoAcidAnalyzer()
    result = analyzer.analyze_bennu_sample(bennu_data)
    print(f"Earth compatibility: {result.earth_compatibility:.2%}")

References:
    - DOCUMENTATION/.../02_EXTRATERRESTRIAL_GENETIC_CODE.md
    - NASA OSIRIS-REx public data
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class AminoAcidSource(Enum):
    """Source of amino acid sample."""

    ASTEROID_BENNU = "bennu"
    ASTEROID_RYUGU = "ryugu"
    MURCHISON_METEORITE = "murchison"
    TAGISH_LAKE_METEORITE = "tagish_lake"
    ALLENDE_METEORITE = "allende"
    ABIOTIC_SYNTHESIS = "abiotic"
    EARTH_BIOLOGICAL = "earth_bio"


class AminoAcidChirality(Enum):
    """Chirality of amino acid."""

    L_FORM = "L"  # Left-handed (life uses this)
    D_FORM = "D"  # Right-handed
    RACEMIC = "racemic"  # Equal mix


@dataclass
class AminoAcidMeasurement:
    """Measurement of amino acid in a sample."""

    amino_acid: str  # Single letter code
    concentration_ppb: float  # Parts per billion
    chirality: AminoAcidChirality
    uncertainty: float  # Measurement uncertainty


@dataclass
class ExtraterrestrialSample:
    """Sample from an extraterrestrial source."""

    source: AminoAcidSource
    measurements: list[AminoAcidMeasurement]
    total_organic_carbon_ppm: float
    collection_date: str | None = None
    notes: str = ""


@dataclass
class CompatibilityResult:
    """Result of Earth genetic code compatibility analysis."""

    source: AminoAcidSource
    earth_compatibility: float  # 0-1 score
    padic_optimality_score: float
    proteinogenic_fraction: float  # Fraction of 20 standard AAs
    non_proteinogenic_fraction: float  # Non-standard AAs
    chirality_ratio: float  # L/(L+D) ratio
    code_universality_score: float
    key_findings: list[str] = field(default_factory=list)


# Earth's canonical 20 amino acids with their codon assignments
PROTEINOGENIC_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

# Non-proteinogenic amino acids found in meteorites
NON_PROTEINOGENIC_AMINO_ACIDS = {
    "AIB": "α-aminoisobutyric acid",
    "IVA": "isovaline",
    "β-ALA": "β-alanine",
    "GABA": "γ-aminobutyric acid",
    "NVA": "norvaline",
    "NLE": "norleucine",
    "α-ABA": "α-aminobutyric acid",
    "β-AIB": "β-aminoisobutyric acid",
}

# Reference Earth amino acid abundances (normalized, from proteins)
EARTH_AMINO_ACID_ABUNDANCE: dict[str, float] = {
    "A": 0.074,  # Alanine
    "R": 0.042,  # Arginine
    "N": 0.044,  # Asparagine
    "D": 0.059,  # Aspartic acid
    "C": 0.033,  # Cysteine
    "E": 0.058,  # Glutamic acid
    "Q": 0.037,  # Glutamine
    "G": 0.074,  # Glycine
    "H": 0.029,  # Histidine
    "I": 0.038,  # Isoleucine
    "L": 0.076,  # Leucine
    "K": 0.072,  # Lysine
    "M": 0.018,  # Methionine
    "F": 0.040,  # Phenylalanine
    "P": 0.050,  # Proline
    "S": 0.081,  # Serine
    "T": 0.062,  # Threonine
    "W": 0.013,  # Tryptophan
    "Y": 0.033,  # Tyrosine
    "V": 0.068,  # Valine
}

# Murchison meteorite amino acid data (representative values in ppb)
MURCHISON_REFERENCE: dict[str, float] = {
    "G": 2500,  # Glycine - most abundant
    "A": 1800,  # Alanine
    "D": 450,  # Aspartic acid
    "E": 380,  # Glutamic acid
    "S": 200,  # Serine
    "V": 150,  # Valine
    "L": 120,  # Leucine
    "I": 100,  # Isoleucine
    "P": 80,  # Proline
    "T": 60,  # Threonine
    # Non-proteinogenic (higher in meteorites)
    "AIB": 3500,  # α-aminoisobutyric acid - dominant!
    "IVA": 800,  # Isovaline
    "β-ALA": 600,  # β-alanine
}


class AsteroidAminoAcidAnalyzer:
    """Analyzer for extraterrestrial amino acid distributions.

    Compares amino acid patterns from asteroids and meteorites to
    Earth's genetic code to test universality hypotheses.
    """

    def __init__(self, p: int = 3):
        """Initialize the analyzer.

        Args:
            p: Prime base for p-adic calculations
        """
        self.p = p
        self.earth_baseline = EARTH_AMINO_ACID_ABUNDANCE.copy()

    def _compute_padic_valuation(self, n: int) -> int:
        """Compute p-adic valuation v_p(n)."""
        if n == 0:
            return 100
        valuation = 0
        while n % self.p == 0:
            valuation += 1
            n //= self.p
        return valuation

    def _amino_acid_to_index(self, aa: str) -> int:
        """Convert amino acid to numerical index."""
        aa_order = "ACDEFGHIKLMNPQRSTVWY"
        aa = aa.upper()
        if aa in aa_order:
            return aa_order.index(aa) + 1
        return 0  # Non-proteinogenic

    def compute_padic_distance(self, aa1: str, aa2: str) -> float:
        """Compute p-adic distance between two amino acids.

        Args:
            aa1: First amino acid
            aa2: Second amino acid

        Returns:
            P-adic distance (0-1)
        """
        idx1 = self._amino_acid_to_index(aa1)
        idx2 = self._amino_acid_to_index(aa2)

        diff = abs(idx1 - idx2)
        if diff == 0:
            return 0.0

        valuation = self._compute_padic_valuation(diff)
        return 1.0 / (self.p**valuation)

    def normalize_concentrations(
        self,
        measurements: list[AminoAcidMeasurement],
    ) -> dict[str, float]:
        """Normalize amino acid concentrations to frequencies.

        Args:
            measurements: List of AA measurements

        Returns:
            Normalized frequency dictionary
        """
        total = sum(m.concentration_ppb for m in measurements)
        if total == 0:
            return {}

        return {m.amino_acid: m.concentration_ppb / total for m in measurements}

    def compute_earth_compatibility(
        self,
        frequencies: dict[str, float],
    ) -> float:
        """Compute compatibility with Earth's amino acid usage.

        Uses cosine similarity between frequency vectors.

        Args:
            frequencies: Normalized AA frequencies

        Returns:
            Compatibility score (0-1)
        """
        # Build vectors for common amino acids
        common_aas = [aa for aa in PROTEINOGENIC_AMINO_ACIDS if aa in frequencies or aa in self.earth_baseline]

        if not common_aas:
            return 0.0

        sample_vec = np.array([frequencies.get(aa, 0.0) for aa in common_aas])
        earth_vec = np.array([self.earth_baseline.get(aa, 0.0) for aa in common_aas])

        # Normalize vectors
        if np.linalg.norm(sample_vec) == 0 or np.linalg.norm(earth_vec) == 0:
            return 0.0

        sample_vec = sample_vec / np.linalg.norm(sample_vec)
        earth_vec = earth_vec / np.linalg.norm(earth_vec)

        # Cosine similarity
        similarity = float(np.dot(sample_vec, earth_vec))
        return max(0.0, similarity)

    def compute_padic_optimality(
        self,
        frequencies: dict[str, float],
    ) -> float:
        """Compute p-adic optimality score.

        Measures how well the distribution clusters in p-adic space.

        Args:
            frequencies: Normalized AA frequencies

        Returns:
            Optimality score (0-1)
        """
        if not frequencies:
            return 0.0

        # Compute pairwise p-adic distances weighted by abundance
        aas = [aa for aa in frequencies if aa in PROTEINOGENIC_AMINO_ACIDS]

        if len(aas) < 2:
            return 0.0

        weighted_distances = []
        for i, aa1 in enumerate(aas):
            for j, aa2 in enumerate(aas):
                if i < j:
                    dist = self.compute_padic_distance(aa1, aa2)
                    weight = frequencies[aa1] * frequencies[aa2]
                    weighted_distances.append(dist * weight)

        if not weighted_distances:
            return 0.0

        # Lower average distance = more clustered = more optimal
        avg_dist = np.mean(weighted_distances)

        # Convert to optimality score (inverse)
        return 1.0 - min(1.0, avg_dist)

    def compute_chirality_ratio(
        self,
        measurements: list[AminoAcidMeasurement],
    ) -> float:
        """Compute L-form to total ratio.

        Life uses exclusively L-amino acids, so this tests biogenicity.

        Args:
            measurements: List of AA measurements

        Returns:
            L/(L+D) ratio (1.0 = all L, 0.5 = racemic, 0 = all D)
        """
        l_total = 0.0
        d_total = 0.0

        for m in measurements:
            if m.chirality == AminoAcidChirality.L_FORM:
                l_total += m.concentration_ppb
            elif m.chirality == AminoAcidChirality.D_FORM:
                d_total += m.concentration_ppb
            else:  # Racemic - split equally
                l_total += m.concentration_ppb / 2
                d_total += m.concentration_ppb / 2

        total = l_total + d_total
        if total == 0:
            return 0.5  # Undefined, assume racemic

        return l_total / total

    def analyze_sample(
        self,
        sample: ExtraterrestrialSample,
    ) -> CompatibilityResult:
        """Analyze an extraterrestrial sample.

        Args:
            sample: Sample data

        Returns:
            CompatibilityResult with analysis metrics
        """
        frequencies = self.normalize_concentrations(sample.measurements)

        # Compute metrics
        earth_compat = self.compute_earth_compatibility(frequencies)
        padic_opt = self.compute_padic_optimality(frequencies)
        chirality = self.compute_chirality_ratio(sample.measurements)

        # Proteinogenic vs non-proteinogenic
        proteinogenic_conc = sum(
            m.concentration_ppb for m in sample.measurements if m.amino_acid in PROTEINOGENIC_AMINO_ACIDS
        )
        non_protein_conc = sum(
            m.concentration_ppb for m in sample.measurements if m.amino_acid not in PROTEINOGENIC_AMINO_ACIDS
        )
        total_conc = proteinogenic_conc + non_protein_conc

        if total_conc > 0:
            proteinogenic_frac = proteinogenic_conc / total_conc
            non_protein_frac = non_protein_conc / total_conc
        else:
            proteinogenic_frac = 0.0
            non_protein_frac = 0.0

        # Code universality score (combination of factors)
        universality = 0.4 * earth_compat + 0.3 * padic_opt + 0.3 * proteinogenic_frac

        # Generate findings
        findings = []
        if earth_compat > 0.7:
            findings.append("High compatibility with Earth genetic code")
        if padic_opt > 0.6:
            findings.append("P-adic clustering suggests optimal encoding")
        if chirality < 0.6:
            findings.append("Near-racemic mixture indicates abiotic origin")
        if chirality > 0.8:
            findings.append("L-form excess suggests potential biotic origin")
        if non_protein_frac > 0.3:
            findings.append("Significant non-proteinogenic amino acids present")

        return CompatibilityResult(
            source=sample.source,
            earth_compatibility=earth_compat,
            padic_optimality_score=padic_opt,
            proteinogenic_fraction=proteinogenic_frac,
            non_proteinogenic_fraction=non_protein_frac,
            chirality_ratio=chirality,
            code_universality_score=universality,
            key_findings=findings,
        )

    def create_murchison_reference_sample(self) -> ExtraterrestrialSample:
        """Create a reference sample based on Murchison meteorite data.

        Returns:
            ExtraterrestrialSample with Murchison data
        """
        measurements = []

        for aa, conc in MURCHISON_REFERENCE.items():
            # Murchison shows slight L-excess for some AAs
            if aa in PROTEINOGENIC_AMINO_ACIDS:
                chirality = AminoAcidChirality.L_FORM if aa in "GAVS" else AminoAcidChirality.RACEMIC
            else:
                chirality = AminoAcidChirality.RACEMIC

            measurements.append(
                AminoAcidMeasurement(
                    amino_acid=aa,
                    concentration_ppb=conc,
                    chirality=chirality,
                    uncertainty=conc * 0.1,  # 10% uncertainty
                )
            )

        return ExtraterrestrialSample(
            source=AminoAcidSource.MURCHISON_METEORITE,
            measurements=measurements,
            total_organic_carbon_ppm=20.0,
            collection_date="1969-09-28",
            notes="Fell in Murchison, Victoria, Australia",
        )

    def compare_sources(
        self,
        samples: list[ExtraterrestrialSample],
    ) -> dict[str, CompatibilityResult]:
        """Compare multiple extraterrestrial sources.

        Args:
            samples: List of samples to compare

        Returns:
            Dictionary mapping source name to results
        """
        return {sample.source.value: self.analyze_sample(sample) for sample in samples}

    def calculate_prebiotic_padic_score(
        self,
        aa_frequencies: dict[str, float],
    ) -> dict[str, float]:
        """Calculate how well prebiotic AA ratios match p-adic optimal code.

        Args:
            aa_frequencies: Amino acid frequencies

        Returns:
            Dictionary with various p-adic scores
        """
        if not aa_frequencies:
            return {"overall_score": 0.0, "clustering": 0.0, "uniformity": 0.0}

        # P-adic clustering score
        clustering = self.compute_padic_optimality(aa_frequencies)

        # Check if distribution follows p-adic patterns
        # Amino acids should cluster in groups of 3 (ternary)
        proteinogenic = {aa: freq for aa, freq in aa_frequencies.items() if aa in PROTEINOGENIC_AMINO_ACIDS}

        if not proteinogenic:
            return {"overall_score": 0.0, "clustering": clustering, "uniformity": 0.0}

        # Sort by frequency
        sorted_aas = sorted(proteinogenic.items(), key=lambda x: x[1], reverse=True)

        # Check if top amino acids form p-adic groups
        group_scores = []
        for i in range(0, min(9, len(sorted_aas)), 3):
            group = sorted_aas[i : i + 3]
            if len(group) < 2:
                continue

            # Compute within-group p-adic cohesion
            distances = []
            for j, (aa1, _) in enumerate(group):
                for k, (aa2, _) in enumerate(group):
                    if j < k:
                        distances.append(self.compute_padic_distance(aa1, aa2))

            if distances:
                group_scores.append(1.0 - np.mean(distances))

        uniformity = np.mean(group_scores) if group_scores else 0.0

        overall = 0.5 * clustering + 0.5 * uniformity

        return {
            "overall_score": float(overall),
            "clustering": float(clustering),
            "uniformity": float(uniformity),
        }


__all__ = [
    "AsteroidAminoAcidAnalyzer",
    "ExtraterrestrialSample",
    "AminoAcidMeasurement",
    "CompatibilityResult",
    "AminoAcidSource",
    "AminoAcidChirality",
    "PROTEINOGENIC_AMINO_ACIDS",
    "NON_PROTEINOGENIC_AMINO_ACIDS",
    "MURCHISON_REFERENCE",
]
