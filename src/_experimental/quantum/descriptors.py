# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Quantum-Chemical Descriptor Computation.

This module computes quantum-chemical descriptors for biological molecules
that can be integrated into the VAE training pipeline.

Key Features:
- HOMO-LUMO gap estimation
- Dipole moment calculations
- Electron density descriptors
- P-adic representation of quantum properties

Note: For accurate DFT calculations, external software (ORCA, Psi4) is
recommended. This module provides estimated values based on amino acid
and structural properties suitable for ML integration.

Usage:
    from src._experimental.quantum.descriptors import QuantumBioDescriptor

    descriptor = QuantumBioDescriptor()
    result = descriptor.compute_sequence_descriptors("MVLSPADKTNVKAAW")
    print(f"HOMO-LUMO gap: {result.homo_lumo_gap:.2f} eV")

References:
    - DOCUMENTATION/.../07_QUANTUM_BIOLOGY_SIGNATURES.md
    - DOCUMENTATION/.../Quantum_Biology_Signatures/proposal.md
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


class AminoAcidQuantumProperties:
    """Quantum-chemical properties of amino acids.

    These are semi-empirical estimates based on published DFT calculations.
    Values represent gas-phase calculations at B3LYP/6-31G* level.
    """

    # HOMO-LUMO gaps in eV (estimated from DFT calculations)
    HOMO_LUMO_GAP: dict[str, float] = {
        "A": 7.2,  # Alanine
        "R": 5.8,  # Arginine - large conjugated system
        "N": 6.5,  # Asparagine
        "D": 6.3,  # Aspartic acid
        "C": 5.9,  # Cysteine - thiol group
        "E": 6.4,  # Glutamic acid
        "Q": 6.6,  # Glutamine
        "G": 7.5,  # Glycine - simplest
        "H": 5.5,  # Histidine - aromatic, can coordinate metals
        "I": 7.1,  # Isoleucine
        "L": 7.1,  # Leucine
        "K": 6.0,  # Lysine
        "M": 5.7,  # Methionine - sulfur
        "F": 5.2,  # Phenylalanine - aromatic
        "P": 7.0,  # Proline - cyclic
        "S": 6.8,  # Serine
        "T": 6.7,  # Threonine
        "W": 4.8,  # Tryptophan - extended aromatic
        "Y": 5.0,  # Tyrosine - aromatic with OH
        "V": 7.2,  # Valine
    }

    # Dipole moments in Debye (estimated)
    DIPOLE_MOMENT: dict[str, float] = {
        "A": 1.7,
        "R": 4.5,  # Charged guanidinium
        "N": 3.8,
        "D": 2.8,  # Carboxylate
        "C": 1.5,
        "E": 2.9,  # Carboxylate
        "Q": 3.6,
        "G": 1.2,
        "H": 2.5,
        "I": 1.4,
        "L": 1.5,
        "K": 3.2,  # Charged amine
        "M": 1.9,
        "F": 0.8,  # Symmetric aromatic
        "P": 2.0,
        "S": 2.5,
        "T": 2.3,
        "W": 2.1,
        "Y": 1.6,
        "V": 1.4,
    }

    # Polarizability in Å³ (estimated)
    POLARIZABILITY: dict[str, float] = {
        "A": 6.0,
        "R": 17.0,
        "N": 9.5,
        "D": 8.5,
        "C": 8.5,
        "E": 10.0,
        "Q": 11.0,
        "G": 4.0,
        "H": 12.0,
        "I": 11.5,
        "L": 11.5,
        "K": 14.0,
        "M": 13.5,
        "F": 15.0,
        "P": 8.5,
        "S": 6.5,
        "T": 8.0,
        "W": 19.0,
        "Y": 16.0,
        "V": 9.5,
    }

    # Ionization potential in eV (estimated)
    IONIZATION_POTENTIAL: dict[str, float] = {
        "A": 9.8,
        "R": 8.2,
        "N": 9.0,
        "D": 9.5,
        "C": 8.8,
        "E": 9.4,
        "Q": 9.1,
        "G": 10.0,
        "H": 8.5,
        "I": 9.7,
        "L": 9.7,
        "K": 8.3,
        "M": 8.6,
        "F": 9.0,
        "P": 9.5,
        "S": 9.3,
        "T": 9.2,
        "W": 8.1,
        "Y": 8.5,
        "V": 9.6,
    }

    # Electron affinity in eV (estimated)
    ELECTRON_AFFINITY: dict[str, float] = {
        "A": 0.2,
        "R": 0.8,
        "N": 0.5,
        "D": 1.2,  # Carboxylate
        "C": 0.4,
        "E": 1.1,  # Carboxylate
        "Q": 0.4,
        "G": 0.1,
        "H": 0.9,
        "I": 0.2,
        "L": 0.2,
        "K": 0.3,
        "M": 0.6,
        "F": 0.8,
        "P": 0.3,
        "S": 0.4,
        "T": 0.3,
        "W": 1.2,
        "Y": 1.0,
        "V": 0.2,
    }


@dataclass
class QuantumDescriptorResult:
    """Result of quantum descriptor calculation."""

    sequence: str
    homo_lumo_gap: float  # Average HOMO-LUMO gap in eV
    dipole_moment: float  # Average dipole moment in Debye
    polarizability: float  # Sum of polarizabilities in Å³
    ionization_potential: float  # Average ionization potential in eV
    electron_affinity: float  # Average electron affinity in eV
    hardness: float  # Chemical hardness = (IP - EA) / 2
    softness: float  # Chemical softness = 1 / hardness
    electronegativity: float  # Mulliken electronegativity = (IP + EA) / 2
    electrophilicity_index: float  # ω = χ² / (2η)
    padic_signature: dict[str, float] = field(default_factory=dict)
    per_residue_homo_lumo: list[float] = field(default_factory=list)


class QuantumBioDescriptor:
    """Compute quantum-chemical descriptors for biological sequences.

    Uses semi-empirical estimates based on amino acid properties for
    rapid descriptor calculation suitable for ML integration.
    """

    def __init__(self, p: int = 3):
        """Initialize the descriptor calculator.

        Args:
            p: Prime base for p-adic calculations
        """
        self.p = p
        self.aa_props = AminoAcidQuantumProperties()

    def _compute_padic_valuation(self, n: int) -> int:
        """Compute p-adic valuation v_p(n)."""
        if n == 0:
            return 100
        valuation = 0
        while n % self.p == 0:
            valuation += 1
            n //= self.p
        return valuation

    def get_residue_homo_lumo(self, residue: str) -> float:
        """Get HOMO-LUMO gap for a single residue.

        Args:
            residue: Single-letter amino acid code

        Returns:
            HOMO-LUMO gap in eV
        """
        return self.aa_props.HOMO_LUMO_GAP.get(residue.upper(), 6.5)

    def get_residue_dipole(self, residue: str) -> float:
        """Get dipole moment for a single residue.

        Args:
            residue: Single-letter amino acid code

        Returns:
            Dipole moment in Debye
        """
        return self.aa_props.DIPOLE_MOMENT.get(residue.upper(), 2.0)

    def compute_sequence_descriptors(self, sequence: str) -> QuantumDescriptorResult:
        """Compute quantum descriptors for a protein sequence.

        Args:
            sequence: Amino acid sequence (single-letter codes)

        Returns:
            QuantumDescriptorResult with computed descriptors
        """
        sequence = sequence.upper()

        # Per-residue values
        homo_lumo_values = [self.get_residue_homo_lumo(aa) for aa in sequence]
        dipole_values = [self.get_residue_dipole(aa) for aa in sequence]
        polarizability_values = [self.aa_props.POLARIZABILITY.get(aa, 10.0) for aa in sequence]
        ip_values = [self.aa_props.IONIZATION_POTENTIAL.get(aa, 9.0) for aa in sequence]
        ea_values = [self.aa_props.ELECTRON_AFFINITY.get(aa, 0.5) for aa in sequence]

        # Aggregate values
        avg_homo_lumo = np.mean(homo_lumo_values) if homo_lumo_values else 6.5
        avg_dipole = np.mean(dipole_values) if dipole_values else 2.0
        total_polarizability = sum(polarizability_values)
        avg_ip = np.mean(ip_values) if ip_values else 9.0
        avg_ea = np.mean(ea_values) if ea_values else 0.5

        # Derived descriptors
        hardness = (avg_ip - avg_ea) / 2
        softness = 1 / hardness if hardness > 0 else float("inf")
        electronegativity = (avg_ip + avg_ea) / 2
        electrophilicity = electronegativity**2 / (2 * hardness) if hardness > 0 else 0

        # P-adic signature
        padic_signature = self._compute_padic_signature(homo_lumo_values)

        return QuantumDescriptorResult(
            sequence=sequence,
            homo_lumo_gap=float(avg_homo_lumo),
            dipole_moment=float(avg_dipole),
            polarizability=float(total_polarizability),
            ionization_potential=float(avg_ip),
            electron_affinity=float(avg_ea),
            hardness=float(hardness),
            softness=float(softness),
            electronegativity=float(electronegativity),
            electrophilicity_index=float(electrophilicity),
            padic_signature=padic_signature,
            per_residue_homo_lumo=homo_lumo_values,
        )

    def _compute_padic_signature(self, values: list[float]) -> dict[str, float]:
        """Compute p-adic signature from a list of values.

        Args:
            values: List of numerical values

        Returns:
            Dictionary of p-adic signature metrics
        """
        if not values:
            return {"valuation_sum": 0, "normalized_distance": 0}

        # Convert to integers for p-adic calculation
        scaled = [int(v * 100) for v in values]

        # Sum of valuations
        valuations = [self._compute_padic_valuation(s) for s in scaled]
        valuation_sum = sum(v for v in valuations if v < 100)

        # Pairwise p-adic distances
        distances = []
        for i in range(len(scaled)):
            for j in range(i + 1, len(scaled)):
                diff = abs(scaled[i] - scaled[j])
                if diff > 0:
                    val = self._compute_padic_valuation(diff)
                    dist = 1.0 / (self.p**val)
                    distances.append(dist)

        avg_distance = np.mean(distances) if distances else 0.0

        return {
            "valuation_sum": float(valuation_sum),
            "normalized_distance": float(avg_distance),
            "min_gap": float(min(values)) if values else 0,
            "max_gap": float(max(values)) if values else 0,
            "gap_variance": float(np.var(values)) if values else 0,
        }

    def compute_binding_site_descriptors(
        self,
        sequence: str,
        binding_residues: list[int],
    ) -> QuantumDescriptorResult:
        """Compute descriptors specifically for binding site residues.

        Args:
            sequence: Full protein sequence
            binding_residues: List of residue indices (0-based) in binding site

        Returns:
            QuantumDescriptorResult for binding site only
        """
        # Extract binding site residues
        binding_seq = "".join(sequence[i] for i in binding_residues if 0 <= i < len(sequence))

        if not binding_seq:
            # Return empty result
            return QuantumDescriptorResult(
                sequence="",
                homo_lumo_gap=0.0,
                dipole_moment=0.0,
                polarizability=0.0,
                ionization_potential=0.0,
                electron_affinity=0.0,
                hardness=0.0,
                softness=float("inf"),
                electronegativity=0.0,
                electrophilicity_index=0.0,
            )

        return self.compute_sequence_descriptors(binding_seq)

    def estimate_tunneling_barrier(self, homo_lumo_gap: float, distance_angstrom: float = 3.0) -> float:
        """Estimate quantum tunneling barrier height.

        Uses a simplified model where tunneling probability is related to
        HOMO-LUMO gap and transfer distance.

        Args:
            homo_lumo_gap: HOMO-LUMO gap in eV
            distance_angstrom: Transfer distance in Å

        Returns:
            Estimated tunneling probability (0-1)
        """
        # Simplified tunneling probability model
        # P ∝ exp(-2 * sqrt(2m*V) * d / ħ)
        # Using effective parameters

        # Barrier height proportional to HOMO-LUMO gap
        barrier_ev = homo_lumo_gap * 0.5  # Approximate barrier

        # Decay constant (simplified)
        decay_constant = 1.0  # Å^-1 for typical organic molecules

        # Probability estimate
        probability = np.exp(-2 * decay_constant * distance_angstrom * np.sqrt(barrier_ev))

        return float(min(1.0, max(0.0, probability)))

    def to_feature_vector(self, result: QuantumDescriptorResult) -> torch.Tensor:
        """Convert descriptor result to feature vector for ML.

        Args:
            result: QuantumDescriptorResult

        Returns:
            1D tensor of normalized descriptor values
        """
        features = [
            result.homo_lumo_gap / 10.0,  # Normalize to ~0-1 range
            result.dipole_moment / 5.0,
            result.polarizability / 500.0,
            result.ionization_potential / 12.0,
            result.electron_affinity / 2.0,
            result.hardness / 5.0,
            1.0 / (1.0 + result.softness),  # Bounded softness
            result.electronegativity / 10.0,
            result.electrophilicity_index / 10.0,
            result.padic_signature.get("normalized_distance", 0.0),
            result.padic_signature.get("gap_variance", 0.0) / 5.0,
        ]

        return torch.tensor(features, dtype=torch.float32)

    def batch_compute(self, sequences: list[str]) -> list[QuantumDescriptorResult]:
        """Compute descriptors for multiple sequences.

        Args:
            sequences: List of amino acid sequences

        Returns:
            List of QuantumDescriptorResult objects
        """
        return [self.compute_sequence_descriptors(seq) for seq in sequences]


__all__ = [
    "QuantumBioDescriptor",
    "QuantumDescriptorResult",
    "AminoAcidQuantumProperties",
]
