# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Candida auris Antifungal Resistance Analyzer.

This module provides analysis of antifungal resistance in Candida auris,
an emerging multi-drug resistant fungal pathogen.

Based on:
- CDC AR Lab Network
- CLSI antifungal breakpoints
- FungiDB

Key Features:
- Echinocandin resistance (FKS1 mutations)
- Azole resistance (ERG11 mutations)
- Amphotericin B resistance
- Clade-specific resistance patterns

C. auris Clades:
- Clade I: South Asian
- Clade II: East Asian (rare infections)
- Clade III: African
- Clade IV: South American

Clinical Significance:
- High mortality (30-60%)
- Difficult to identify
- Environmental persistence
- Healthcare-associated outbreaks

Usage:
    from src.diseases.candida_analyzer import CandidaAnalyzer

    analyzer = CandidaAnalyzer()
    results = analyzer.analyze(sequences, clade="I")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

from .base import DiseaseAnalyzer, DiseaseConfig, DiseaseType, TaskType


class CandidaClade(Enum):
    """C. auris geographic clades."""

    CLADE_I = "I"  # South Asian
    CLADE_II = "II"  # East Asian
    CLADE_III = "III"  # African
    CLADE_IV = "IV"  # South American
    CLADE_V = "V"  # Iranian


class CandidaGene(Enum):
    """Genes associated with antifungal resistance."""

    # Echinocandin resistance
    FKS1 = "FKS1"  # Hot spot regions 1 and 2

    # Azole resistance
    ERG11 = "ERG11"  # Lanosterol 14-alpha-demethylase
    TAC1B = "TAC1B"  # Transcription factor
    CDR1 = "CDR1"  # Efflux pump
    MDR1 = "MDR1"  # Efflux pump

    # Amphotericin B resistance
    ERG3 = "ERG3"
    ERG6 = "ERG6"


class Antifungal(Enum):
    """Antifungal drugs for Candida."""

    # Echinocandins
    ANIDULAFUNGIN = "anidulafungin"
    CASPOFUNGIN = "caspofungin"
    MICAFUNGIN = "micafungin"

    # Azoles
    FLUCONAZOLE = "fluconazole"
    VORICONAZOLE = "voriconazole"
    ITRACONAZOLE = "itraconazole"
    POSACONAZOLE = "posaconazole"
    ISAVUCONAZOLE = "isavuconazole"

    # Polyenes
    AMPHOTERICIN_B = "amphotericin_b"

    # 5-FC
    FLUCYTOSINE = "flucytosine"


@dataclass
class CandidaConfig(DiseaseConfig):
    """Configuration for Candida analysis."""

    name: str = "candida_auris"
    display_name: str = "Candida auris"
    disease_type: DiseaseType = DiseaseType.FUNGAL
    tasks: list[TaskType] = field(
        default_factory=lambda: [
            TaskType.RESISTANCE,
        ]
    )

    data_sources: dict[str, str] = field(
        default_factory=lambda: {
            "cdc_ar": "https://www.cdc.gov/candida-auris/",
            "fungidb": "https://fungidb.org/",
            "clsi": "https://clsi.org/",
        }
    )


# FKS1 Hot Spot Mutations (Echinocandin resistance)
FKS1_HOTSPOT1 = {
    # Hot spot 1 (positions 639-663 in C. auris)
    639: {"F": {"mutations": ["I", "L", "V"], "effect": "high"}},
    641: {"S": {"mutations": ["P", "F", "Y"], "effect": "high"}},  # Key position
    643: {"D": {"mutations": ["V", "A", "G", "E"], "effect": "moderate"}},
    645: {"R": {"mutations": ["K", "G", "S"], "effect": "moderate"}},
    649: {"R": {"mutations": ["G"], "effect": "low"}},
}

FKS1_HOTSPOT2 = {
    # Hot spot 2 (positions 1353-1357 in C. auris)
    1354: {"R": {"mutations": ["G", "H", "S"], "effect": "high"}},
    1355: {"D": {"mutations": ["E", "A"], "effect": "moderate"}},
}

# ERG11 Mutations (Azole resistance)
ERG11_MUTATIONS = {
    # Key positions for azole resistance
    61: {"A": {"mutations": ["V"], "effect": "low"}},
    132: {"K": {"mutations": ["R", "T"], "effect": "high"}},
    143: {"F": {"mutations": ["L", "R"], "effect": "high"}},  # Important
    220: {"Y": {"mutations": ["F", "H"], "effect": "high"}},  # Key for fluconazole
    400: {"G": {"mutations": ["D"], "effect": "moderate"}},
    447: {"F": {"mutations": ["L", "I", "S"], "effect": "high"}},  # Key position
    448: {"G": {"mutations": ["E", "S", "A"], "effect": "moderate"}},
    449: {"Y": {"mutations": ["F", "H", "N", "S", "C"], "effect": "high"}},
    450: {"G": {"mutations": ["E", "V"], "effect": "moderate"}},
    466: {"V": {"mutations": ["I"], "effect": "low"}},
}

# ERG3/ERG6 mutations (Amphotericin B resistance - rare)
ERG3_MUTATIONS = {
    # Rarely mutated
    271: {"S": {"mutations": ["P"], "effect": "high"}},
}


class CandidaAnalyzer(DiseaseAnalyzer):
    """Analyzer for C. auris antifungal resistance.

    Provides:
    - Echinocandin resistance (FKS1)
    - Azole resistance (ERG11)
    - Amphotericin B resistance
    - Pan-resistance detection
    """

    def __init__(self, config: CandidaConfig | None = None):
        """Initialize analyzer."""
        self.config = config or CandidaConfig()
        super().__init__(self.config)

        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY-X*"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_alphabet)}

    def analyze(
        self,
        sequences: dict[CandidaGene, list[str]],
        clade: CandidaClade = CandidaClade.CLADE_I,
        embeddings: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Analyze C. auris sequences for antifungal resistance.

        Args:
            sequences: Dictionary mapping gene to sequences
            clade: Geographic clade
            embeddings: Optional embeddings

        Returns:
            Analysis results
        """
        results = {
            "n_sequences": len(next(iter(sequences.values()))) if sequences else 0,
            "clade": clade.value,
            "genes_analyzed": [g.value for g in sequences],
            "drug_resistance": {},
            "pan_resistance_alert": [],
        }

        # Echinocandin resistance (FKS1)
        if CandidaGene.FKS1 in sequences:
            results["drug_resistance"]["echinocandins"] = self._analyze_fks1(sequences[CandidaGene.FKS1])

        # Azole resistance (ERG11)
        if CandidaGene.ERG11 in sequences:
            results["drug_resistance"]["azoles"] = self._analyze_erg11(sequences[CandidaGene.ERG11])

        # Amphotericin B resistance (ERG3)
        if CandidaGene.ERG3 in sequences:
            results["drug_resistance"]["amphotericin_b"] = self._analyze_erg3(sequences[CandidaGene.ERG3])

        # Pan-resistance check
        results["pan_resistance_alert"] = self._check_pan_resistance(results["drug_resistance"])

        return results

    def _analyze_fks1(self, sequences: list[str]) -> dict[str, Any]:
        """Analyze FKS1 hot spots for echinocandin resistance."""
        results = {
            "gene": "FKS1",
            "scores": [],
            "classifications": [],
            "mutations": [],
            "hotspot1": [],
            "hotspot2": [],
        }

        for seq in sequences:
            score = 0.0
            mutations = []
            hs1_muts = []
            hs2_muts = []

            # Check hot spot 1
            for pos, info in FKS1_HOTSPOT1.items():
                if pos <= len(seq):
                    seq_aa = seq[pos - 1]
                    ref_aa = list(info.keys())[0]

                    if seq_aa != ref_aa and seq_aa in info[ref_aa]["mutations"]:
                        effect = info[ref_aa]["effect"]
                        effect_scores = {"high": 1.0, "moderate": 0.5, "low": 0.2}
                        score += effect_scores.get(effect, 0.3)

                        mut = {
                            "position": pos,
                            "ref": ref_aa,
                            "alt": seq_aa,
                            "hotspot": 1,
                            "effect": effect,
                        }
                        mutations.append(mut)
                        hs1_muts.append(f"{ref_aa}{pos}{seq_aa}")

            # Check hot spot 2
            for pos, info in FKS1_HOTSPOT2.items():
                if pos <= len(seq):
                    seq_aa = seq[pos - 1]
                    ref_aa = list(info.keys())[0]

                    if seq_aa != ref_aa and seq_aa in info[ref_aa]["mutations"]:
                        effect = info[ref_aa]["effect"]
                        effect_scores = {"high": 1.0, "moderate": 0.5, "low": 0.2}
                        score += effect_scores.get(effect, 0.3)

                        mut = {
                            "position": pos,
                            "ref": ref_aa,
                            "alt": seq_aa,
                            "hotspot": 2,
                            "effect": effect,
                        }
                        mutations.append(mut)
                        hs2_muts.append(f"{ref_aa}{pos}{seq_aa}")

            normalized = min(score / 2.0, 1.0)
            results["scores"].append(normalized)
            results["mutations"].append(mutations)
            results["hotspot1"].append(hs1_muts)
            results["hotspot2"].append(hs2_muts)

            if normalized < 0.2:
                results["classifications"].append("susceptible")
            elif normalized < 0.5:
                results["classifications"].append("intermediate")
            else:
                results["classifications"].append("resistant")

        return results

    def _analyze_erg11(self, sequences: list[str]) -> dict[str, Any]:
        """Analyze ERG11 for azole resistance."""
        results = {
            "gene": "ERG11",
            "scores": [],
            "classifications": [],
            "mutations": [],
        }

        for seq in sequences:
            score = 0.0
            mutations = []

            for pos, info in ERG11_MUTATIONS.items():
                if pos <= len(seq):
                    seq_aa = seq[pos - 1]
                    ref_aa = list(info.keys())[0]

                    if seq_aa != ref_aa and seq_aa in info[ref_aa]["mutations"]:
                        effect = info[ref_aa]["effect"]
                        effect_scores = {"high": 1.0, "moderate": 0.5, "low": 0.2}
                        score += effect_scores.get(effect, 0.3)

                        mutations.append(
                            {
                                "position": pos,
                                "ref": ref_aa,
                                "alt": seq_aa,
                                "effect": effect,
                            }
                        )

            normalized = min(score / 3.0, 1.0)
            results["scores"].append(normalized)
            results["mutations"].append(mutations)

            if normalized < 0.2:
                results["classifications"].append("susceptible")
            elif normalized < 0.5:
                results["classifications"].append("reduced_susceptibility")
            else:
                results["classifications"].append("resistant")

        return results

    def _analyze_erg3(self, sequences: list[str]) -> dict[str, Any]:
        """Analyze ERG3 for amphotericin B resistance."""
        results = {
            "gene": "ERG3",
            "scores": [],
            "classifications": [],
            "mutations": [],
        }

        for seq in sequences:
            score = 0.0
            mutations = []

            for pos, info in ERG3_MUTATIONS.items():
                if pos <= len(seq):
                    seq_aa = seq[pos - 1]
                    ref_aa = list(info.keys())[0]

                    if seq_aa != ref_aa and seq_aa in info[ref_aa]["mutations"]:
                        effect = info[ref_aa]["effect"]
                        score += 1.0
                        mutations.append(
                            {
                                "position": pos,
                                "ref": ref_aa,
                                "alt": seq_aa,
                                "effect": effect,
                            }
                        )

            normalized = min(score, 1.0)
            results["scores"].append(normalized)
            results["mutations"].append(mutations)

            if normalized < 0.5:
                results["classifications"].append("susceptible")
            else:
                results["classifications"].append("resistant")

        return results

    def _check_pan_resistance(self, drug_resistance: dict) -> list[dict[str, Any]]:
        """Check for pan-resistance (resistance to all 3 classes)."""
        alerts = []

        # Determine number of sequences
        n_seq = 0
        for data in drug_resistance.values():
            if "classifications" in data:
                n_seq = max(n_seq, len(data["classifications"]))

        for i in range(n_seq):
            echo_r = False
            azole_r = False
            ampho_r = False

            if "echinocandins" in drug_resistance:
                if i < len(drug_resistance["echinocandins"].get("classifications", [])):
                    echo_r = drug_resistance["echinocandins"]["classifications"][i] == "resistant"

            if "azoles" in drug_resistance:
                if i < len(drug_resistance["azoles"].get("classifications", [])):
                    azole_r = drug_resistance["azoles"]["classifications"][i] in ["resistant", "reduced_susceptibility"]

            if "amphotericin_b" in drug_resistance:
                if i < len(drug_resistance["amphotericin_b"].get("classifications", [])):
                    ampho_r = drug_resistance["amphotericin_b"]["classifications"][i] == "resistant"

            alert = {
                "isolate": i,
                "echinocandin_resistant": echo_r,
                "azole_resistant": azole_r,
                "amphotericin_b_resistant": ampho_r,
                "pan_resistant": echo_r and azole_r and ampho_r,
                "alert_level": "critical"
                if (echo_r and azole_r and ampho_r)
                else (
                    "high"
                    if (echo_r and azole_r) or (echo_r and ampho_r)
                    else ("moderate" if echo_r or ampho_r else "low")
                ),
            }
            alerts.append(alert)

        return alerts

    def validate_predictions(
        self,
        predictions: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> dict[str, float]:
        """Validate against phenotypic antifungal testing."""
        from scipy.stats import spearmanr

        metrics = {}

        for drug_class in predictions.get("drug_resistance", {}):
            if drug_class in ground_truth:
                pred = np.array(predictions["drug_resistance"][drug_class]["scores"])
                true = np.array(ground_truth[drug_class])

                if len(pred) == len(true):
                    rho, pval = spearmanr(pred, true)
                    metrics[f"{drug_class}_spearman"] = float(rho) if not np.isnan(rho) else 0.0

        return metrics

    def encode_sequence(
        self,
        sequence: str,
        max_length: int = 500,
    ) -> np.ndarray:
        """One-hot encode sequence."""
        n_aa = len(self.aa_alphabet)
        encoding = np.zeros(max_length * n_aa, dtype=np.float32)

        for j, aa in enumerate(sequence[:max_length]):
            idx = self.aa_to_idx.get(aa.upper(), self.aa_to_idx["X"])
            encoding[j * n_aa + idx] = 1.0

        return encoding


def create_candida_synthetic_dataset(
    gene: CandidaGene = CandidaGene.FKS1,
    min_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create synthetic Candida dataset.

    Args:
        gene: Target gene for resistance analysis
        min_samples: Minimum number of samples to generate

    Returns:
        (X, y, ids)
    """
    from src.diseases.utils.synthetic_data import (
        create_mutation_based_dataset,
        ensure_minimum_samples,
    )

    # FKS1 hotspot positions: HS1 at 639-649, HS2 at 1354-1355
    # ERG11 positions: up to 466
    # Need max_length to cover all mutation positions

    if gene == CandidaGene.FKS1:
        # FKS1 needs 1400 AA to cover both hotspot regions
        max_length = 1400
        reference = "M" + "A" * (max_length - 1)
        mutation_db = {**FKS1_HOTSPOT1, **FKS1_HOTSPOT2}
    elif gene == CandidaGene.ERG11:
        # ERG11 mutations go up to position 466
        max_length = 500
        reference = "M" + "A" * (max_length - 1)
        mutation_db = ERG11_MUTATIONS
    else:
        # ERG3 mutations at position 271
        max_length = 300
        reference = "M" + "A" * (max_length - 1)
        mutation_db = ERG3_MUTATIONS

    analyzer = CandidaAnalyzer()

    # Use utility to create dataset with mutation combinations
    # encode_fn signature: (sequence, max_length) -> np.ndarray
    X, y, ids = create_mutation_based_dataset(
        reference_sequence=reference,
        mutation_db=mutation_db,
        encode_fn=lambda s, ml: analyzer.encode_sequence(s, max_length=ml),
        max_length=max_length,
        n_random_mutants=30,
        seed=42,
    )

    # Ensure minimum samples
    X, y, ids = ensure_minimum_samples(X, y, ids, min_samples=min_samples, seed=42)

    return X, y, ids
