# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Quantum Biology Analysis.

This module analyzes p-adic patterns in quantum-active biological sites,
focusing on enzyme catalytic sites where quantum tunneling plays a role.

Key Features:
- Catalytic site p-adic signature analysis
- Tunneling probability prediction
- Known quantum enzyme database
- Correlation of p-adic patterns with tunneling efficiency

Hypothesis:
Quantum tunneling sites have distinct p-adic signatures that can be
used to predict enzyme catalytic efficiency.

Usage:
    from src._experimental.quantum.biology import QuantumBiologyAnalyzer

    analyzer = QuantumBiologyAnalyzer()
    result = analyzer.analyze_catalytic_site(sequence, [85, 90, 110])
    print(f"Tunneling score: {result['tunneling_score']:.3f}")

References:
    - DOCUMENTATION/.../07_QUANTUM_BIOLOGY_SIGNATURES.md
    - Video analysis on quantum coherence in enzymes
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class QuantumMechanism(Enum):
    """Types of quantum mechanisms in biology."""

    PROTON_TUNNELING = "proton_tunneling"
    HYDROGEN_TUNNELING = "hydrogen_tunneling"
    ELECTRON_TRANSFER = "electron_transfer"
    VIBRATIONAL_COUPLING = "vibrational_coupling"
    COHERENT_ENERGY_TRANSFER = "coherent_energy_transfer"


@dataclass
class QuantumEnzyme:
    """Information about an enzyme with known quantum effects."""

    name: str
    ec_number: str  # Enzyme Commission number
    mechanism: QuantumMechanism
    active_site_residues: list[str]  # Key residues (e.g., ["H95", "D220"])
    tunneling_distance: float  # Å
    activation_barrier: float  # kcal/mol
    isotope_effect: float  # KIE (kinetic isotope effect)
    notes: str = ""


# Database of enzymes with documented quantum effects
QUANTUM_ENZYMES: dict[str, QuantumEnzyme] = {
    "photosystem_II": QuantumEnzyme(
        name="Photosystem II",
        ec_number="1.10.3.9",
        mechanism=QuantumMechanism.ELECTRON_TRANSFER,
        active_site_residues=["Y161", "H190", "D170", "E189"],
        tunneling_distance=4.0,
        activation_barrier=12.0,
        isotope_effect=2.0,
        notes="Water oxidation complex, O2 evolution",
    ),
    "cytochrome_c_oxidase": QuantumEnzyme(
        name="Cytochrome c Oxidase",
        ec_number="7.1.1.9",
        mechanism=QuantumMechanism.ELECTRON_TRANSFER,
        active_site_residues=["H240", "H291", "H290", "Y244"],
        tunneling_distance=3.5,
        activation_barrier=10.0,
        isotope_effect=1.8,
        notes="Terminal oxidase in respiratory chain",
    ),
    "aromatic_amine_dehydrogenase": QuantumEnzyme(
        name="Aromatic Amine Dehydrogenase",
        ec_number="1.4.99.4",
        mechanism=QuantumMechanism.HYDROGEN_TUNNELING,
        active_site_residues=["C108", "W109", "D128"],
        tunneling_distance=2.8,
        activation_barrier=8.0,
        isotope_effect=55.0,  # Very large KIE - quantum tunneling signature
        notes="Classic example of H-tunneling, studied by Scrutton lab",
    ),
    "soybean_lipoxygenase": QuantumEnzyme(
        name="Soybean Lipoxygenase-1",
        ec_number="1.13.11.12",
        mechanism=QuantumMechanism.HYDROGEN_TUNNELING,
        active_site_residues=["H499", "H504", "H690", "N694", "I553"],
        tunneling_distance=2.5,
        activation_barrier=6.0,
        isotope_effect=81.0,  # Largest known enzyme KIE
        notes="Hydrogen abstraction via quantum tunneling",
    ),
    "alcohol_dehydrogenase": QuantumEnzyme(
        name="Alcohol Dehydrogenase",
        ec_number="1.1.1.1",
        mechanism=QuantumMechanism.HYDROGEN_TUNNELING,
        active_site_residues=["C46", "H67", "C174"],
        tunneling_distance=3.0,
        activation_barrier=7.5,
        isotope_effect=3.5,
        notes="NAD+-dependent, temperature-independent KIE",
    ),
    "fmo_complex": QuantumEnzyme(
        name="Fenna-Matthews-Olson Complex",
        ec_number="",
        mechanism=QuantumMechanism.COHERENT_ENERGY_TRANSFER,
        active_site_residues=["H25", "Y16", "F265"],
        tunneling_distance=15.0,  # Long-range energy transfer
        activation_barrier=2.0,
        isotope_effect=1.0,  # Not relevant for energy transfer
        notes="Photosynthetic antenna protein, quantum coherence at room temp",
    ),
    "methylamine_dehydrogenase": QuantumEnzyme(
        name="Methylamine Dehydrogenase",
        ec_number="1.4.99.3",
        mechanism=QuantumMechanism.PROTON_TUNNELING,
        active_site_residues=["D76", "W57", "C107", "W108"],
        tunneling_distance=2.6,
        activation_barrier=7.0,
        isotope_effect=17.0,
        notes="TTQ cofactor, proton-coupled electron transfer",
    ),
}

# Amino acid properties relevant to quantum tunneling
TUNNELING_RESIDUES: dict[str, float] = {
    # Residues that frequently participate in tunneling reactions
    "H": 0.9,  # Histidine - proton relay
    "C": 0.8,  # Cysteine - thiolate chemistry
    "D": 0.7,  # Aspartate - proton acceptor
    "E": 0.7,  # Glutamate - proton acceptor
    "Y": 0.6,  # Tyrosine - radical chemistry
    "W": 0.5,  # Tryptophan - electron relay
    "K": 0.4,  # Lysine - proton donor
    "R": 0.3,  # Arginine - electrostatic
    "N": 0.3,  # Asparagine - H-bonding
    "Q": 0.3,  # Glutamine - H-bonding
    "S": 0.2,  # Serine - mild nucleophile
    "T": 0.2,  # Threonine - mild nucleophile
}


@dataclass
class CatalyticSiteAnalysis:
    """Result of catalytic site analysis."""

    site_sequence: str
    residue_positions: list[int]
    padic_clustering_score: float
    tunneling_propensity: float
    predicted_kie: float  # Predicted kinetic isotope effect
    mechanism_likelihood: dict[QuantumMechanism, float]
    critical_residues: list[str]


class QuantumBiologyAnalyzer:
    """Analyzer for p-adic patterns in quantum-active biological sites.

    Uses p-adic mathematics to identify signatures associated with
    quantum tunneling in enzyme catalytic sites.
    """

    def __init__(self, p: int = 3):
        """Initialize the analyzer.

        Args:
            p: Prime base for p-adic calculations
        """
        self.p = p

    def _compute_padic_valuation(self, n: int) -> int:
        """Compute p-adic valuation v_p(n)."""
        if n == 0:
            return 100
        valuation = 0
        while n % self.p == 0:
            valuation += 1
            n //= self.p
        return valuation

    def _compute_padic_distance(self, a: int, b: int) -> float:
        """Compute p-adic distance between two integers."""
        diff = abs(a - b)
        if diff == 0:
            return 0.0
        valuation = self._compute_padic_valuation(diff)
        return 1.0 / (self.p**valuation)

    def _residue_to_index(self, residue: str) -> int:
        """Convert residue to numerical index for p-adic calculation."""
        aa_order = "ACDEFGHIKLMNPQRSTVWY"
        residue = residue.upper()
        if residue in aa_order:
            return aa_order.index(residue) + 1
        return 10  # Default for unknown

    def analyze_catalytic_site(
        self,
        sequence: str,
        active_site_positions: list[int],
    ) -> CatalyticSiteAnalysis:
        """Analyze a catalytic site for quantum tunneling signatures.

        Args:
            sequence: Full protein sequence
            active_site_positions: Residue positions in active site (0-based)

        Returns:
            CatalyticSiteAnalysis with p-adic metrics
        """
        # Extract active site residues
        site_residues = []
        for pos in active_site_positions:
            if 0 <= pos < len(sequence):
                site_residues.append(sequence[pos])
        site_sequence = "".join(site_residues)

        # Compute p-adic clustering
        clustering_score = self._compute_clustering_score(active_site_positions)

        # Compute tunneling propensity
        tunneling_score = self._compute_tunneling_propensity(site_residues)

        # Predict kinetic isotope effect
        predicted_kie = self._predict_kie(site_residues, clustering_score)

        # Mechanism likelihood
        mechanism_likelihood = self._assess_mechanism_likelihood(site_residues)

        # Identify critical residues
        critical = self._identify_critical_residues(site_residues)

        return CatalyticSiteAnalysis(
            site_sequence=site_sequence,
            residue_positions=active_site_positions,
            padic_clustering_score=clustering_score,
            tunneling_propensity=tunneling_score,
            predicted_kie=predicted_kie,
            mechanism_likelihood=mechanism_likelihood,
            critical_residues=critical,
        )

    def _compute_clustering_score(self, positions: list[int]) -> float:
        """Compute p-adic clustering score for residue positions.

        Lower clustering (higher p-adic distances) may indicate
        optimized geometry for quantum effects.

        Args:
            positions: List of residue positions

        Returns:
            Clustering score (0-1, lower = more spread)
        """
        if len(positions) < 2:
            return 0.0

        distances = []
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i < j:
                    dist = self._compute_padic_distance(pos1 + 1, pos2 + 1)
                    distances.append(dist)

        # Average distance (higher = more spread in p-adic terms)
        avg_distance = np.mean(distances) if distances else 0.0

        # Convert to clustering score (inverted)
        return float(1.0 - min(1.0, avg_distance))

    def _compute_tunneling_propensity(self, residues: list[str]) -> float:
        """Compute tunneling propensity based on residue composition.

        Args:
            residues: List of single-letter amino acid codes

        Returns:
            Tunneling propensity score (0-1)
        """
        if not residues:
            return 0.0

        scores = [TUNNELING_RESIDUES.get(r.upper(), 0.1) for r in residues]
        return float(np.mean(scores))

    def _predict_kie(self, residues: list[str], clustering: float) -> float:
        """Predict kinetic isotope effect.

        Higher KIE (>7) suggests quantum tunneling contribution.

        Args:
            residues: Active site residues
            clustering: P-adic clustering score

        Returns:
            Predicted KIE value
        """
        # Base KIE from tunneling propensity
        tunneling = self._compute_tunneling_propensity(residues)

        # KIE scales with tunneling propensity
        # Range from 1 (no tunneling) to ~80 (extensive tunneling)
        base_kie = 1 + tunneling * 50

        # Clustering effect (less clustering = potentially longer tunneling distance)
        clustering_factor = 1 + (1 - clustering) * 0.5

        predicted = base_kie * clustering_factor

        return min(100, max(1.0, predicted))

    def _assess_mechanism_likelihood(self, residues: list[str]) -> dict[QuantumMechanism, float]:
        """Assess likelihood of different quantum mechanisms.

        Args:
            residues: Active site residues

        Returns:
            Dictionary of mechanism to likelihood score
        """
        residue_set = set(r.upper() for r in residues)

        likelihoods = {}

        # Proton tunneling - requires H, D, E, K
        proton_residues = {"H", "D", "E", "K"}
        proton_count = len(residue_set & proton_residues)
        likelihoods[QuantumMechanism.PROTON_TUNNELING] = proton_count / 4.0

        # Hydrogen tunneling - requires C, D, E, H
        hydrogen_residues = {"C", "D", "E", "H"}
        h_count = len(residue_set & hydrogen_residues)
        likelihoods[QuantumMechanism.HYDROGEN_TUNNELING] = h_count / 4.0

        # Electron transfer - requires Y, W, H, Fe/Cu ligands (C, H, M)
        et_residues = {"Y", "W", "H", "C", "M"}
        et_count = len(residue_set & et_residues)
        likelihoods[QuantumMechanism.ELECTRON_TRANSFER] = et_count / 5.0

        # Vibrational coupling - aromatic residues
        aromatic = {"F", "Y", "W", "H"}
        arom_count = len(residue_set & aromatic)
        likelihoods[QuantumMechanism.VIBRATIONAL_COUPLING] = arom_count / 4.0

        # Coherent energy transfer - requires aromatic network
        likelihoods[QuantumMechanism.COHERENT_ENERGY_TRANSFER] = arom_count / 6.0

        return likelihoods

    def _identify_critical_residues(self, residues: list[str]) -> list[str]:
        """Identify residues critical for quantum effects.

        Args:
            residues: Active site residues

        Returns:
            List of critical residues
        """
        critical = []
        for r in residues:
            r_upper = r.upper()
            if TUNNELING_RESIDUES.get(r_upper, 0) >= 0.5:
                critical.append(r_upper)
        return list(set(critical))

    def predict_tunneling_probability(
        self,
        site_analysis: CatalyticSiteAnalysis,
        temperature_kelvin: float = 298.15,
    ) -> float:
        """Predict quantum tunneling probability.

        Uses the p-adic signature and residue composition to estimate
        tunneling contribution to catalysis.

        Args:
            site_analysis: Result from analyze_catalytic_site
            temperature_kelvin: Temperature in Kelvin

        Returns:
            Tunneling probability (0-1)
        """
        # Temperature factor (tunneling less temperature-dependent)
        temp_factor = np.exp(-1000 / temperature_kelvin)

        # Base probability from tunneling propensity
        base_prob = site_analysis.tunneling_propensity

        # KIE contribution (high KIE = high tunneling)
        kie_factor = min(1.0, (site_analysis.predicted_kie - 1) / 50)

        # P-adic clustering (less clustering = potentially better geometry)
        geometry_factor = 1.0 - 0.5 * site_analysis.padic_clustering_score

        probability = base_prob * (0.5 + 0.5 * kie_factor) * geometry_factor * temp_factor

        return float(min(1.0, max(0.0, probability)))

    def compare_to_known_enzyme(
        self,
        site_analysis: CatalyticSiteAnalysis,
    ) -> tuple[str, float]:
        """Find most similar known quantum enzyme.

        Args:
            site_analysis: Result from analyze_catalytic_site

        Returns:
            (enzyme_name, similarity_score)
        """
        best_match = None
        best_score = 0.0

        for name, enzyme in QUANTUM_ENZYMES.items():
            score = self._compute_enzyme_similarity(site_analysis, enzyme)
            if score > best_score:
                best_score = score
                best_match = name

        return best_match or "unknown", best_score

    def _compute_enzyme_similarity(
        self,
        analysis: CatalyticSiteAnalysis,
        enzyme: QuantumEnzyme,
    ) -> float:
        """Compute similarity between analysis and known enzyme."""
        # Residue overlap
        analysis_residues = set(analysis.site_sequence)
        enzyme_residues = set()
        for res in enzyme.active_site_residues:
            if len(res) >= 1:
                enzyme_residues.add(res[-1] if res[-1].isalpha() else res[0])

        if analysis_residues and enzyme_residues:
            overlap = len(analysis_residues & enzyme_residues) / len(analysis_residues | enzyme_residues)
        else:
            overlap = 0.0

        # Mechanism match
        max_mech_score = max(analysis.mechanism_likelihood.values()) if analysis.mechanism_likelihood else 0
        mech_match = analysis.mechanism_likelihood.get(enzyme.mechanism, 0.0) / max(0.01, max_mech_score)

        # KIE match (closer to enzyme's KIE is better)
        kie_diff = abs(analysis.predicted_kie - enzyme.isotope_effect)
        kie_match = max(0, 1 - kie_diff / 100)

        return (overlap + mech_match + kie_match) / 3


__all__ = [
    "QuantumBiologyAnalyzer",
    "QuantumEnzyme",
    "QuantumMechanism",
    "CatalyticSiteAnalysis",
    "QUANTUM_ENZYMES",
    "TUNNELING_RESIDUES",
]
