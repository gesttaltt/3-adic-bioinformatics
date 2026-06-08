# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Hepatitis B Virus (HBV) Drug Resistance Analyzer.

This module provides analysis of HBV drug resistance mutations for
nucleos(t)ide analogues (NAs) targeting the reverse transcriptase domain.

Based on:
- HBVdb (https://hbvdb.ibcp.fr/)
- EASL HBV Clinical Practice Guidelines
- AASLD HBV Guidance

Key Features:
- RT domain resistance mutations for all NAs
- Genotype-specific considerations
- Cross-resistance pattern detection
- Overlapping reading frame awareness (RT/S overlap)

HBV Genotypes:
- A-H (with different geographic distributions)
- Genotype affects treatment response

Drugs:
- Entecavir (ETV) - first-line
- Tenofovir (TDF/TAF) - first-line
- Lamivudine (LAM) - older, high resistance
- Adefovir (ADV) - limited use
- Telbivudine (LdT) - limited use

Usage:
    from src.diseases.hbv_analyzer import HBVAnalyzer

    analyzer = HBVAnalyzer()
    results = analyzer.analyze(sequences, genotype="D")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

from .base import DiseaseAnalyzer, DiseaseConfig, DiseaseType, TaskType


class HBVGenotype(Enum):
    """HBV genotypes."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"
    H = "H"


class HBVGene(Enum):
    """HBV genes."""

    P = "P"  # Polymerase (contains RT domain)
    S = "S"  # Surface antigen (overlaps with P)
    C = "C"  # Core
    X = "X"  # X protein


class HBVDrug(Enum):
    """HBV antiviral drugs (nucleos(t)ide analogues)."""

    # First-line (high barrier to resistance)
    ENTECAVIR = "entecavir"
    TENOFOVIR_DF = "tenofovir_df"  # TDF
    TENOFOVIR_AF = "tenofovir_af"  # TAF

    # Older agents (lower barrier)
    LAMIVUDINE = "lamivudine"
    ADEFOVIR = "adefovir"
    TELBIVUDINE = "telbivudine"


@dataclass
class HBVConfig(DiseaseConfig):
    """Configuration for HBV analysis."""

    name: str = "hbv"
    display_name: str = "Hepatitis B Virus"
    disease_type: DiseaseType = DiseaseType.VIRAL
    tasks: list[TaskType] = field(
        default_factory=lambda: [
            TaskType.RESISTANCE,
        ]
    )

    data_sources: dict[str, str] = field(
        default_factory=lambda: {
            "hbvdb": "https://hbvdb.ibcp.fr/",
            "easl": "https://easl.eu/",
            "aasld": "https://www.aasld.org/",
            "stanford_hbv": "https://hivdb.stanford.edu/hbv/",
        }
    )


# HBV RT Domain Resistance Mutations
# Numbering based on RT domain (starts at aa 1 of RT)
# Note: RT overlaps with S gene
# Consolidated to avoid duplicate keys (Python dict limitation)
# Expanded database from EASL 2023 and AASLD guidelines

RT_MUTATIONS = {
    # Early domain mutations
    80: {"L": {"mutations": ["V", "I"], "effect": "low", "drugs": ["lamivudine"]}},
    # rtA181 mutations (LAM/ADV cross-resistance)
    181: {"A": {"mutations": ["T", "V", "S"], "effect": "high", "drugs": ["lamivudine", "adefovir", "telbivudine"]}},
    # YMDD motif region (catalytic domain)
    169: {"I": {"mutations": ["T"], "effect": "moderate", "drugs": ["entecavir"]}},
    173: {"V": {"mutations": ["L", "M"], "effect": "moderate", "drugs": ["lamivudine", "entecavir"]}},
    # rtL180M - Primary LAM resistance (precedes M204V/I)
    180: {"L": {"mutations": ["M"], "effect": "high", "drugs": ["lamivudine", "telbivudine", "entecavir"]}},
    # rtT184 - ETV resistance (requires LAM background)
    184: {"T": {"mutations": ["G", "S", "I", "A", "L"], "effect": "high", "drugs": ["entecavir"]}},
    # Tenofovir resistance (rare but emerging)
    194: {"A": {"mutations": ["T"], "effect": "moderate", "drugs": ["tenofovir_df", "tenofovir_af"]}},
    # rtS202 - Secondary ETV resistance
    202: {"S": {"mutations": ["G", "I", "C"], "effect": "high", "drugs": ["entecavir"]}},
    # rtM204V/I - YMDD motif (primary LAM/LdT resistance)
    204: {"M": {"mutations": ["V", "I", "S"], "effect": "high", "drugs": ["lamivudine", "telbivudine", "entecavir"]}},
    # Secondary/compensatory mutations
    207: {"V": {"mutations": ["I"], "effect": "low", "drugs": ["lamivudine"]}},
    213: {"L": {"mutations": ["M"], "effect": "low", "drugs": ["lamivudine"]}},
    # Adefovir resistance mutations
    233: {"I": {"mutations": ["V"], "effect": "moderate", "drugs": ["adefovir"]}},
    236: {"N": {"mutations": ["T"], "effect": "high", "drugs": ["adefovir", "tenofovir_df", "tenofovir_af"]}},
    # rtM250 - Secondary ETV resistance
    250: {"M": {"mutations": ["V", "I", "L"], "effect": "high", "drugs": ["entecavir"]}},
    # Additional clinically relevant mutations (EASL 2023)
    # rtI169T + others for ETV
    163: {"R": {"mutations": ["K", "Q"], "effect": "low", "drugs": ["entecavir"]}},
    # rtT128N - Vaccine escape that also affects some NAs
    128: {"T": {"mutations": ["N", "I"], "effect": "low", "drugs": ["lamivudine"]}},
    # rtV214A - Compensatory mutation
    214: {"V": {"mutations": ["A"], "effect": "low", "drugs": ["lamivudine", "entecavir"]}},
    # rtQ215S - Compensatory mutation
    215: {"Q": {"mutations": ["S", "H"], "effect": "low", "drugs": ["lamivudine", "entecavir"]}},
    # rtN238 mutations - ADV/TDF pathway
    238: {"N": {"mutations": ["T", "H", "D"], "effect": "moderate", "drugs": ["adefovir", "tenofovir_df"]}},
    # rtP237H - Rare TDF mutation
    237: {"P": {"mutations": ["H"], "effect": "low", "drugs": ["tenofovir_df", "tenofovir_af"]}},
    # rtS219A - Minor LAM resistance
    219: {"S": {"mutations": ["A"], "effect": "low", "drugs": ["lamivudine"]}},
    # rtI233V precursor changes
    229: {"F": {"mutations": ["Y", "L"], "effect": "low", "drugs": ["adefovir"]}},
    # rtA222 - Background polymorphism affecting resistance
    222: {"A": {"mutations": ["T", "V"], "effect": "low", "drugs": ["entecavir"]}},
}

# Cross-resistance patterns
CROSS_RESISTANCE = {
    "lamivudine": ["telbivudine"],  # LAM resistance = LdT resistance
    "telbivudine": ["lamivudine"],
    "adefovir": [],  # Mostly distinct
    "entecavir": ["lamivudine"],  # ETV resistance requires LAM background
    "tenofovir_df": ["tenofovir_af"],  # TDF = TAF
    "tenofovir_af": ["tenofovir_df"],
}


class HBVAnalyzer(DiseaseAnalyzer):
    """Analyzer for HBV NA resistance.

    Provides:
    - RT domain mutation detection
    - Cross-resistance assessment
    - S gene impact analysis (vaccine escape)
    """

    def __init__(self, config: HBVConfig | None = None):
        """Initialize analyzer."""
        self.config = config or HBVConfig()
        super().__init__(self.config)

        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY-X*"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_alphabet)}

    def analyze(
        self,
        sequences: dict[HBVGene, list[str]],
        genotype: HBVGenotype = HBVGenotype.D,
        embeddings: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Analyze HBV sequences for drug resistance.

        Args:
            sequences: Dictionary mapping gene to sequences
            genotype: HBV genotype
            embeddings: Optional embeddings

        Returns:
            Analysis results
        """
        results = {
            "n_sequences": len(next(iter(sequences.values()))) if sequences else 0,
            "genotype": genotype.value,
            "genes_analyzed": [g.value for g in sequences],
            "drug_resistance": {},
            "cross_resistance": {},
            "s_gene_impact": [],
        }

        # Analyze RT domain
        if HBVGene.P in sequences:
            for drug in HBVDrug:
                results["drug_resistance"][drug.value] = self._predict_drug_resistance(sequences[HBVGene.P], drug)

            # Cross-resistance analysis
            results["cross_resistance"] = self._analyze_cross_resistance(results["drug_resistance"])

            # S gene impact (RT mutations affecting surface antigen)
            results["s_gene_impact"] = self._analyze_s_gene_impact(sequences[HBVGene.P])

        return results

    def _predict_drug_resistance(
        self,
        sequences: list[str],
        drug: HBVDrug,
    ) -> dict[str, Any]:
        """Predict resistance for a specific drug."""
        results = {
            "drug": drug.value,
            "scores": [],
            "classifications": [],
            "mutations": [],
        }

        for seq in sequences:
            score = 0.0
            mutations = []

            for pos, info in RT_MUTATIONS.items():
                if pos <= 0 or pos > len(seq):
                    continue

                seq_aa = seq[pos - 1]
                ref_aa = list(info.keys())[0]

                # Handle alternative references (e.g., "S/T")
                ref_aas = ref_aa.split("/") if "/" in ref_aa else [ref_aa]

                for ref in ref_aas:
                    if seq_aa != ref and seq_aa in info[ref_aa]["mutations"]:
                        if drug.value in info[ref_aa]["drugs"]:
                            effect = info[ref_aa]["effect"]
                            effect_scores = {"high": 1.0, "moderate": 0.5, "low": 0.2}
                            score += effect_scores.get(effect, 0.3)

                            mutations.append(
                                {
                                    "position": pos,
                                    "ref": ref_aa,
                                    "alt": seq_aa,
                                    "effect": effect,
                                    "notation": f"rt{ref_aa}{pos}{seq_aa}",
                                }
                            )

            # Normalize
            max_score = 4.0
            normalized = min(score / max_score, 1.0)

            results["scores"].append(normalized)
            results["mutations"].append(mutations)

            # Classification
            if normalized < 0.1:
                classification = "susceptible"
            elif normalized < 0.3:
                classification = "reduced_susceptibility"
            else:
                classification = "resistant"

            results["classifications"].append(classification)

        return results

    def _analyze_cross_resistance(self, drug_resistance: dict) -> dict[str, list[str]]:
        """Analyze cross-resistance patterns."""
        cross_res = {}

        for drug, data in drug_resistance.items():
            resistant_indices = [i for i, c in enumerate(data.get("classifications", [])) if c == "resistant"]

            if resistant_indices:
                cross_drugs = CROSS_RESISTANCE.get(drug, [])
                cross_res[drug] = {
                    "resistant_sequences": resistant_indices,
                    "cross_resistant_to": cross_drugs,
                }

        return cross_res

    def _analyze_s_gene_impact(self, rt_sequences: list[str]) -> list[dict[str, Any]]:
        """Analyze impact of RT mutations on overlapping S gene.

        The RT domain overlaps with the surface (S) gene.
        Some RT mutations affect S protein, potentially causing:
        - Vaccine escape
        - Diagnostic escape
        """
        # Key positions where RT mutations affect S antigen
        rt_to_s_impact = {
            173: {"s_position": 163, "impact": "possible vaccine escape"},
            180: {"s_position": 170, "impact": "diagnostic escape"},
            181: {"s_position": 171, "impact": "surface antigen change"},
            204: {"s_position": 194, "impact": "possible diagnostic escape"},
        }

        impacts = []
        for _i, seq in enumerate(rt_sequences):
            seq_impacts = []
            for rt_pos, s_info in rt_to_s_impact.items():
                if rt_pos <= len(seq):
                    ref_info = RT_MUTATIONS.get(rt_pos, {})
                    if ref_info:
                        ref_aa = list(ref_info.keys())[0]
                        if "/" in ref_aa:
                            ref_aa = ref_aa.split("/")[0]
                        seq_aa = seq[rt_pos - 1]

                        if seq_aa != ref_aa:
                            seq_impacts.append(
                                {
                                    "rt_position": rt_pos,
                                    "s_position": s_info["s_position"],
                                    "mutation": f"rt{ref_aa}{rt_pos}{seq_aa}",
                                    "impact": s_info["impact"],
                                }
                            )

            impacts.append(seq_impacts)

        return impacts

    def validate_predictions(
        self,
        predictions: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> dict[str, float]:
        """Validate against phenotypic data."""
        from scipy.stats import spearmanr

        metrics = {}

        for drug in predictions.get("drug_resistance", {}):
            if drug in ground_truth:
                pred = np.array(predictions["drug_resistance"][drug]["scores"])
                true = np.array(ground_truth[drug])

                if len(pred) == len(true):
                    rho, pval = spearmanr(pred, true)
                    metrics[f"{drug}_spearman"] = float(rho) if not np.isnan(rho) else 0.0

        return metrics

    def encode_sequence(
        self,
        sequence: str,
        max_length: int = 350,
    ) -> np.ndarray:
        """One-hot encode RT sequence."""
        n_aa = len(self.aa_alphabet)
        encoding = np.zeros(max_length * n_aa, dtype=np.float32)

        for j, aa in enumerate(sequence[:max_length]):
            idx = self.aa_to_idx.get(aa.upper(), self.aa_to_idx["X"])
            encoding[j * n_aa + idx] = 1.0

        return encoding


def create_hbv_synthetic_dataset(
    drug: HBVDrug = HBVDrug.ENTECAVIR,
    min_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create synthetic HBV dataset for testing.

    Args:
        drug: Target drug for resistance
        min_samples: Minimum number of samples to generate

    Returns:
        (X, y, ids)
    """
    from src.diseases.utils.synthetic_data import (
        create_mutation_based_dataset,
        ensure_minimum_samples,
    )

    # Build reference sequence with correct wild-type amino acids at mutation positions
    # RT domain is ~350 AA, mutations go up to position 250
    ref_length = 350
    reference = list("M" + "A" * (ref_length - 1))

    # Set correct wild-type amino acids at each mutation position
    for pos, info in RT_MUTATIONS.items():
        if pos <= ref_length:
            ref_aa = list(info.keys())[0]
            # Handle alternative references like "S/T" - use first
            if "/" in ref_aa:
                ref_aa = ref_aa.split("/")[0]
            reference[pos - 1] = ref_aa

    reference = "".join(reference)

    # Filter mutation_db for this drug
    drug_mutation_db = {}
    for pos, info in RT_MUTATIONS.items():
        key = list(info.keys())[0]
        if drug.value in info[key].get("drugs", []):
            drug_mutation_db[pos] = info

    analyzer = HBVAnalyzer()

    # Use utility to create dataset
    X, y, ids = create_mutation_based_dataset(
        reference_sequence=reference,
        mutation_db=drug_mutation_db if drug_mutation_db else RT_MUTATIONS,
        encode_fn=lambda s, ml: analyzer.encode_sequence(s, max_length=ml),
        max_length=350,
        n_random_mutants=30,
        seed=42,
    )

    # Ensure minimum samples
    X, y, ids = ensure_minimum_samples(X, y, ids, min_samples=min_samples, seed=42)

    return X, y, ids
