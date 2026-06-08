# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Extremophile Codon Usage Analysis.

This module analyzes codon usage patterns in extremophile organisms to test
the boundaries of genetic code optimality through p-adic mathematics.

Key Features:
- Codon usage bias analysis for different extremophile categories
- P-adic distance distributions for codon patterns
- Temperature/environment prediction from codon frequencies
- Comparison with mesophile baselines

Target Organisms:
- Pyrococcus furiosus (hyperthermophile, 100°C)
- Deinococcus radiodurans (radiation resistant)
- Halobacterium salinarum (halophile)
- Psychrobacter cryohalolentis (psychrophile)

Usage:
    from src.analysis.extremophile_codons import ExtremophileCodonAnalyzer

    analyzer = ExtremophileCodonAnalyzer()
    result = analyzer.analyze_codon_bias(genome_seq, category="thermophile")
    print(f"GC content: {result.gc_content:.2%}")
    print(f"Predicted temp: {result.predicted_temperature:.1f}°C")

References:
    - DOCUMENTATION/.../03_EXTREMOPHILE_CODON_ADAPTATION.md
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum

import numpy as np


class ExtremophileCategory(Enum):
    """Categories of extremophile organisms."""

    THERMOPHILE = "thermophile"  # High temperature (>45°C)
    HYPERTHERMOPHILE = "hyperthermophile"  # Very high temperature (>80°C)
    PSYCHROPHILE = "psychrophile"  # Low temperature (<15°C)
    RADIORESISTANT = "radioresistant"  # Radiation resistant
    HALOPHILE = "halophile"  # High salt (>2.5M NaCl)
    ACIDOPHILE = "acidophile"  # Low pH (<5)
    ALKALIPHILE = "alkaliphile"  # High pH (>9)
    BAROPHILE = "barophile"  # High pressure (>400 atm)
    MESOPHILE = "mesophile"  # Normal conditions (reference)


# Standard genetic code codon table
CODON_TABLE: dict[str, str] = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}


@dataclass
class CodonUsageResult:
    """Result of codon usage analysis."""

    category: ExtremophileCategory
    codon_frequencies: dict[str, float]
    rscu_values: dict[str, float]  # Relative Synonymous Codon Usage
    gc_content: float
    gc3_content: float  # GC at third codon position
    padic_distances: dict[str, float]
    predicted_temperature: float | None
    enc: float  # Effective Number of Codons
    cai: float | None  # Codon Adaptation Index (if reference available)


@dataclass
class OrganismProfile:
    """Profile of an extremophile organism."""

    name: str
    scientific_name: str
    category: ExtremophileCategory
    optimal_temperature: float | None  # °C
    optimal_ph: float | None
    optimal_salinity: float | None  # M NaCl
    gc_content: float | None
    notes: str = ""


# Reference extremophile profiles
REFERENCE_ORGANISMS: dict[str, OrganismProfile] = {
    "pyrococcus_furiosus": OrganismProfile(
        name="Pyrococcus furiosus",
        scientific_name="Pyrococcus furiosus DSM 3638",
        category=ExtremophileCategory.HYPERTHERMOPHILE,
        optimal_temperature=100.0,
        optimal_ph=7.0,
        optimal_salinity=0.5,
        gc_content=0.408,
        notes="Hyperthermophilic archaeon, found in deep-sea vents",
    ),
    "deinococcus_radiodurans": OrganismProfile(
        name="Deinococcus radiodurans",
        scientific_name="Deinococcus radiodurans R1",
        category=ExtremophileCategory.RADIORESISTANT,
        optimal_temperature=30.0,
        optimal_ph=7.0,
        optimal_salinity=0.0,
        gc_content=0.669,
        notes="Most radiation-resistant organism known",
    ),
    "halobacterium_salinarum": OrganismProfile(
        name="Halobacterium salinarum",
        scientific_name="Halobacterium salinarum NRC-1",
        category=ExtremophileCategory.HALOPHILE,
        optimal_temperature=42.0,
        optimal_ph=7.0,
        optimal_salinity=4.3,
        gc_content=0.657,
        notes="Extreme halophile, requires 2.5-5.2M NaCl",
    ),
    "psychrobacter_cryohalolentis": OrganismProfile(
        name="Psychrobacter cryohalolentis",
        scientific_name="Psychrobacter cryohalolentis K5",
        category=ExtremophileCategory.PSYCHROPHILE,
        optimal_temperature=-10.0,
        optimal_ph=7.0,
        optimal_salinity=0.5,
        gc_content=0.424,
        notes="Antarctic psychrophile, grows at -10°C",
    ),
    "thermus_thermophilus": OrganismProfile(
        name="Thermus thermophilus",
        scientific_name="Thermus thermophilus HB8",
        category=ExtremophileCategory.THERMOPHILE,
        optimal_temperature=65.0,
        optimal_ph=7.5,
        optimal_salinity=0.0,
        gc_content=0.693,
        notes="Source of Taq polymerase, model thermophile",
    ),
    "sulfolobus_solfataricus": OrganismProfile(
        name="Sulfolobus solfataricus",
        scientific_name="Sulfolobus solfataricus P2",
        category=ExtremophileCategory.ACIDOPHILE,
        optimal_temperature=80.0,
        optimal_ph=3.0,
        optimal_salinity=0.0,
        gc_content=0.357,
        notes="Thermoacidophile, grows at pH 2-4 and 75-80°C",
    ),
    "ecoli_k12": OrganismProfile(
        name="E. coli K-12",
        scientific_name="Escherichia coli K-12 MG1655",
        category=ExtremophileCategory.MESOPHILE,
        optimal_temperature=37.0,
        optimal_ph=7.0,
        optimal_salinity=0.0,
        gc_content=0.508,
        notes="Reference mesophile organism",
    ),
}


class ExtremophileCodonAnalyzer:
    """Analyzer for codon usage patterns in extremophile organisms.

    Uses p-adic mathematics to model codon preferences and predict
    environmental adaptation from sequence data.
    """

    def __init__(self, p: int = 3):
        """Initialize the analyzer.

        Args:
            p: Prime base for p-adic calculations (3 for ternary)
        """
        self.p = p
        self._build_codon_mappings()

    def _build_codon_mappings(self) -> None:
        """Build codon to amino acid and synonymous codon mappings."""
        self.codon_to_aa = CODON_TABLE.copy()

        # Group codons by amino acid
        self.aa_to_codons: dict[str, list[str]] = {}
        for codon, aa in self.codon_to_aa.items():
            if aa not in self.aa_to_codons:
                self.aa_to_codons[aa] = []
            self.aa_to_codons[aa].append(codon)

    def _compute_padic_valuation(self, n: int) -> int:
        """Compute p-adic valuation v_p(n)."""
        if n == 0:
            return 100
        valuation = 0
        while n % self.p == 0:
            valuation += 1
            n //= self.p
        return valuation

    def _codon_to_ternary(self, codon: str) -> int:
        """Convert codon to ternary integer representation.

        Maps each nucleotide to a ternary digit and combines into integer.
        T/U=0, C=1, A=2, G=2 (grouping purines)
        """
        base_map = {"T": 0, "U": 0, "C": 1, "A": 2, "G": 2}
        value = 0
        for _i, base in enumerate(codon.upper()):
            if base in base_map:
                value = value * 3 + base_map[base]
        return value

    def count_codons(self, sequence: str) -> dict[str, int]:
        """Count codon occurrences in a sequence.

        Args:
            sequence: DNA/RNA sequence (coding region)

        Returns:
            Dictionary of codon counts
        """
        sequence = sequence.upper().replace("U", "T")

        # Ensure sequence length is multiple of 3
        seq_len = len(sequence) - (len(sequence) % 3)
        sequence = sequence[:seq_len]

        counts: dict[str, int] = Counter()
        for i in range(0, len(sequence), 3):
            codon = sequence[i : i + 3]
            if codon in self.codon_to_aa:
                counts[codon] += 1

        return dict(counts)

    def compute_codon_frequencies(self, counts: dict[str, int]) -> dict[str, float]:
        """Compute codon frequencies from counts.

        Args:
            counts: Dictionary of codon counts

        Returns:
            Dictionary of codon frequencies (0-1)
        """
        total = sum(counts.values())
        if total == 0:
            return {codon: 0.0 for codon in self.codon_to_aa}

        return {codon: count / total for codon, count in counts.items()}

    def compute_rscu(self, counts: dict[str, int]) -> dict[str, float]:
        """Compute Relative Synonymous Codon Usage (RSCU).

        RSCU = (observed frequency) / (expected frequency if no bias)
        RSCU = 1.0 means no bias, >1.0 means preferred, <1.0 means avoided.

        Args:
            counts: Dictionary of codon counts

        Returns:
            Dictionary of RSCU values
        """
        rscu: dict[str, float] = {}

        for aa, codons in self.aa_to_codons.items():
            if aa == "*":  # Skip stop codons
                continue

            # Total count for this amino acid
            aa_total = sum(counts.get(c, 0) for c in codons)

            if aa_total == 0:
                for c in codons:
                    rscu[c] = 0.0
            else:
                n_synonyms = len(codons)
                expected = aa_total / n_synonyms
                for c in codons:
                    observed = counts.get(c, 0)
                    rscu[c] = observed / expected if expected > 0 else 0.0

        return rscu

    def compute_gc_content(self, sequence: str) -> tuple[float, float]:
        """Compute overall GC content and GC3 (third position).

        Args:
            sequence: DNA sequence

        Returns:
            (gc_content, gc3_content)
        """
        sequence = sequence.upper().replace("U", "T")
        seq_len = len(sequence) - (len(sequence) % 3)
        sequence = sequence[:seq_len]

        if len(sequence) == 0:
            return 0.0, 0.0

        # Overall GC
        gc_count = sequence.count("G") + sequence.count("C")
        gc_content = gc_count / len(sequence)

        # GC at third codon position
        gc3_count = 0
        total_codons = 0
        for i in range(2, len(sequence), 3):  # Third positions
            if sequence[i] in "GC":
                gc3_count += 1
            total_codons += 1

        gc3_content = gc3_count / total_codons if total_codons > 0 else 0.0

        return gc_content, gc3_content

    def compute_enc(self, counts: dict[str, int]) -> float:
        """Compute Effective Number of Codons (ENC).

        ENC ranges from 20 (extreme bias) to 61 (no bias).
        Lower values indicate stronger codon usage bias.

        Based on Wright (1990) formula.

        Args:
            counts: Dictionary of codon counts

        Returns:
            ENC value (20-61)
        """
        # Group amino acids by degeneracy
        degeneracy_groups: dict[int, list[str]] = {
            1: ["M", "W"],  # Met, Trp - single codon
            2: ["F", "Y", "H", "Q", "N", "K", "D", "E", "C"],  # 2-fold
            3: ["I"],  # Ile - 3 codons
            4: ["V", "P", "T", "A", "G"],  # 4-fold
            6: ["L", "R", "S"],  # 6-fold
        }

        # Compute F values for each degeneracy class
        f_values: dict[int, list[float]] = {k: [] for k in degeneracy_groups}

        for aa in "ACDEFGHIKLMNPQRSTVWY":
            if aa not in self.aa_to_codons:
                continue

            codons = self.aa_to_codons[aa]
            n = sum(counts.get(c, 0) for c in codons)

            if n <= 1:
                continue

            # Compute F = sum(p_i^2) where p_i is frequency of each codon
            frequencies = [counts.get(c, 0) / n for c in codons]
            f = sum(p**2 for p in frequencies)

            # Find degeneracy class
            deg = len(codons)
            for k, aas in degeneracy_groups.items():
                if aa in aas:
                    deg = k
                    break

            f_values[deg].append(f)

        # Compute ENC components
        enc = 0.0

        # Single codon amino acids
        enc += 2  # Met + Trp

        # For each degeneracy class
        for deg, f_list in f_values.items():
            if deg == 1 or not f_list:
                continue

            avg_f = float(np.mean(f_list))
            if deg == 2:
                enc += 9 / avg_f if avg_f > 0 else 9
            elif deg == 3:
                enc += 1 / avg_f if avg_f > 0 else 1
            elif deg == 4:
                enc += 5 / avg_f if avg_f > 0 else 5
            elif deg == 6:
                enc += 3 / avg_f if avg_f > 0 else 3

        return max(20.0, min(61.0, enc))

    def compute_padic_distances(self, rscu: dict[str, float]) -> dict[str, float]:
        """Compute p-adic distances for codon usage patterns.

        Args:
            rscu: RSCU values for each codon

        Returns:
            P-adic distance for each codon from neutral usage
        """
        distances: dict[str, float] = {}

        for codon, rscu_val in rscu.items():
            # Convert RSCU deviation to p-adic distance
            # RSCU=1 is neutral, deviations from 1 indicate bias
            deviation = abs(rscu_val - 1.0)
            if deviation < 0.01:
                distances[codon] = 0.0
            else:
                # Scale deviation and compute p-adic representation
                scaled = int(deviation * 100)
                valuation = self._compute_padic_valuation(max(1, scaled))
                distances[codon] = 1.0 / (self.p**valuation)

        return distances

    def predict_temperature(self, gc_content: float, gc3_content: float, rscu: dict[str, float]) -> float:
        """Predict optimal growth temperature from codon usage.

        Higher GC content correlates with higher temperature tolerance
        due to increased DNA stability.

        Args:
            gc_content: Overall GC content
            gc3_content: GC at third codon position
            rscu: RSCU values

        Returns:
            Predicted optimal temperature in °C
        """
        # Base temperature from GC content
        # Empirical relationship: higher GC = higher temp tolerance
        # Typical range: 30% GC -> ~20°C, 70% GC -> ~80°C
        base_temp = 20 + (gc_content - 0.30) * 150

        # Adjust based on GC3 vs overall GC
        # Thermophiles often have higher GC3 relative to overall GC
        gc3_factor = gc3_content / gc_content if gc_content > 0 else 1.0
        if gc3_factor > 1.0:
            base_temp += (gc3_factor - 1.0) * 20

        # Adjust based on specific codon preferences
        # GGC (Gly) and CGC (Arg) are preferred in thermophiles
        thermophile_codons = ["GGC", "CGC", "AGC", "ACC"]
        thermo_bias = sum(rscu.get(c, 0) for c in thermophile_codons) / len(thermophile_codons)
        if thermo_bias > 1.0:
            base_temp += (thermo_bias - 1.0) * 10

        return max(-20.0, min(120.0, base_temp))

    def analyze_codon_bias(
        self,
        sequence: str,
        category: ExtremophileCategory | None = None,
    ) -> CodonUsageResult:
        """Analyze codon usage bias in a sequence.

        Args:
            sequence: DNA/RNA coding sequence
            category: Known organism category (optional)

        Returns:
            Complete codon usage analysis result
        """
        # Count codons
        counts = self.count_codons(sequence)

        # Compute statistics
        frequencies = self.compute_codon_frequencies(counts)
        rscu = self.compute_rscu(counts)
        gc_content, gc3_content = self.compute_gc_content(sequence)
        enc = self.compute_enc(counts)
        padic_distances = self.compute_padic_distances(rscu)

        # Predict temperature
        predicted_temp = self.predict_temperature(gc_content, gc3_content, rscu)

        # Infer category if not provided
        if category is None:
            category = self._infer_category(gc_content, predicted_temp)

        return CodonUsageResult(
            category=category,
            codon_frequencies=frequencies,
            rscu_values=rscu,
            gc_content=gc_content,
            gc3_content=gc3_content,
            padic_distances=padic_distances,
            predicted_temperature=predicted_temp,
            enc=enc,
            cai=None,  # Requires reference set
        )

    def _infer_category(self, gc_content: float, predicted_temp: float) -> ExtremophileCategory:
        """Infer organism category from sequence features.

        Args:
            gc_content: GC content
            predicted_temp: Predicted optimal temperature

        Returns:
            Inferred extremophile category
        """
        if predicted_temp > 80:
            return ExtremophileCategory.HYPERTHERMOPHILE
        elif predicted_temp > 45:
            return ExtremophileCategory.THERMOPHILE
        elif predicted_temp < 15:
            return ExtremophileCategory.PSYCHROPHILE
        else:
            return ExtremophileCategory.MESOPHILE

    def compare_to_mesophile(self, result: CodonUsageResult) -> dict[str, float]:
        """Compare codon usage to E. coli K-12 baseline.

        Args:
            result: Codon usage analysis result

        Returns:
            Dictionary of deviation metrics
        """
        # E. coli K-12 reference RSCU values (subset of commonly used codons)
        ecoli_rscu = {
            "TTT": 0.58,
            "TTC": 1.42,  # Phe
            "CTG": 5.02,  # Leu (highly preferred)
            "ATG": 1.00,  # Met
            "GGT": 1.58,
            "GGC": 1.48,  # Gly
            "GCG": 1.34,  # Ala
            "CGT": 2.18,  # Arg
            "AAA": 1.55,
            "AAG": 0.45,  # Lys
            "GAA": 1.52,
            "GAG": 0.48,  # Glu
        }

        deviations = {}
        for codon, ecoli_val in ecoli_rscu.items():
            obs_val = result.rscu_values.get(codon, 0.0)
            deviations[codon] = obs_val - ecoli_val

        # Summary statistics
        deviations["mean_deviation"] = float(np.mean(list(deviations.values())))
        deviations["gc_deviation"] = result.gc_content - 0.508  # E. coli GC

        return deviations

    def analyze_organism(self, sequence: str, organism_name: str) -> dict:
        """Analyze sequence with reference organism profile.

        Args:
            sequence: DNA sequence
            organism_name: Key in REFERENCE_ORGANISMS

        Returns:
            Combined analysis with profile comparison
        """
        if organism_name not in REFERENCE_ORGANISMS:
            raise ValueError(f"Unknown organism: {organism_name}. Available: {list(REFERENCE_ORGANISMS.keys())}")

        profile = REFERENCE_ORGANISMS[organism_name]
        result = self.analyze_codon_bias(sequence, category=profile.category)

        comparison = {
            "profile": profile,
            "analysis": result,
            "gc_match": abs(result.gc_content - (profile.gc_content or 0.5)) < 0.05,
            "temp_prediction_error": None,
        }

        if profile.optimal_temperature is not None and result.predicted_temperature is not None:
            comparison["temp_prediction_error"] = abs(result.predicted_temperature - profile.optimal_temperature)

        return comparison


__all__ = [
    "ExtremophileCodonAnalyzer",
    "ExtremophileCategory",
    "CodonUsageResult",
    "OrganismProfile",
    "REFERENCE_ORGANISMS",
    "CODON_TABLE",
]
