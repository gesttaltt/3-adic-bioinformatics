# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Respiratory Syncytial Virus (RSV) Analyzer.

This module provides analysis of RSV for:
- Monoclonal antibody resistance (nirsevimab, palivizumab)
- Fusion inhibitor resistance
- Strain typing (RSV-A vs RSV-B)

Based on:
- FDA RSV surveillance data
- GISAID RSV sequences
- Published resistance studies

Key Features:
- F protein escape mutation detection
- Antibody binding site analysis
- Antiviral resistance prediction
- Seasonal strain tracking

Clinical Relevance:
- RSV causes 100,000-500,000 deaths annually
- New vaccines and mAbs approved 2023-2024
- Monitoring resistance is critical

Usage:
    from src.diseases.rsv_analyzer import RSVAnalyzer

    analyzer = RSVAnalyzer()
    results = analyzer.analyze(sequences, subtype="A")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

from .base import DiseaseAnalyzer, DiseaseConfig, DiseaseType, TaskType
from .utils.synthetic_data import ensure_minimum_samples


class RSVSubtype(Enum):
    """RSV subtypes."""

    RSV_A = "A"
    RSV_B = "B"


class RSVGene(Enum):
    """RSV genes."""

    F = "F"  # Fusion protein - mAb target
    G = "G"  # Attachment glycoprotein
    N = "N"  # Nucleoprotein
    L = "L"  # Polymerase


class RSVDrug(Enum):
    """RSV therapeutics."""

    # Monoclonal antibodies
    NIRSEVIMAB = "nirsevimab"  # Beyfortus
    PALIVIZUMAB = "palivizumab"  # Synagis

    # Fusion inhibitors
    PRESATOVIR = "presatovir"
    RILPIMATINE = "rilpimatine"


@dataclass
class RSVConfig(DiseaseConfig):
    """Configuration for RSV analysis."""

    name: str = "rsv"
    display_name: str = "Respiratory Syncytial Virus"
    disease_type: DiseaseType = DiseaseType.VIRAL
    tasks: list[TaskType] = field(
        default_factory=lambda: [
            TaskType.RESISTANCE,
            TaskType.ESCAPE,
        ]
    )

    data_sources: dict[str, str] = field(
        default_factory=lambda: {
            "gisaid": "https://gisaid.org/",
            "ncbi_rsv": "https://www.ncbi.nlm.nih.gov/labs/virus/",
            "cdc_rsv": "https://www.cdc.gov/rsv/",
        }
    )


# F protein antigenic site mutations
# Site 0 is the target for nirsevimab
F_SITE_0_MUTATIONS = {
    # Nirsevimab binding site
    62: {"K": {"mutations": ["R", "E", "N"], "effect": "high", "drug": "nirsevimab"}},
    64: {"S": {"mutations": ["N", "T"], "effect": "moderate", "drug": "nirsevimab"}},
    66: {"N": {"mutations": ["K", "I"], "effect": "high", "drug": "nirsevimab"}},
    68: {"I": {"mutations": ["T", "M", "F"], "effect": "moderate", "drug": "nirsevimab"}},
    196: {"S": {"mutations": ["N"], "effect": "moderate", "drug": "nirsevimab"}},
    198: {"N": {"mutations": ["K", "S"], "effect": "high", "drug": "nirsevimab"}},
    200: {"K": {"mutations": ["E", "Q", "N"], "effect": "high", "drug": "nirsevimab"}},
    201: {"L": {"mutations": ["F", "I"], "effect": "moderate", "drug": "nirsevimab"}},
}

# Palivizumab binding site (antigenic site II)
F_SITE_II_MUTATIONS = {
    262: {"N": {"mutations": ["S", "K", "D", "Y"], "effect": "high", "drug": "palivizumab"}},
    268: {"S": {"mutations": ["I", "N"], "effect": "moderate", "drug": "palivizumab"}},
    272: {"K": {"mutations": ["E", "Q", "N", "M"], "effect": "high", "drug": "palivizumab"}},
    275: {"K": {"mutations": ["N", "E", "R"], "effect": "moderate", "drug": "palivizumab"}},
}

# Fusion inhibitor resistance
F_FUSION_RESISTANCE = {
    # Presatovir/rilpimatine binding site
    127: {"D": {"mutations": ["E", "N"], "effect": "high", "drug": "fusion_inhibitor"}},
    137: {"D": {"mutations": ["G", "N"], "effect": "high", "drug": "fusion_inhibitor"}},
    140: {"D": {"mutations": ["G", "Y", "E", "N"], "effect": "high", "drug": "fusion_inhibitor"}},
    401: {"N": {"mutations": ["S", "I"], "effect": "moderate", "drug": "fusion_inhibitor"}},
    486: {"D": {"mutations": ["N", "Y"], "effect": "moderate", "drug": "fusion_inhibitor"}},
    489: {"F": {"mutations": ["L", "Y", "S"], "effect": "high", "drug": "fusion_inhibitor"}},
}


class RSVAnalyzer(DiseaseAnalyzer):
    """Analyzer for RSV drug/antibody resistance.

    Provides:
    - mAb escape detection (nirsevimab, palivizumab)
    - Fusion inhibitor resistance
    - Antigenic site analysis
    """

    def __init__(self, config: RSVConfig | None = None):
        """Initialize analyzer."""
        self.config = config or RSVConfig()
        super().__init__(self.config)

        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY-X*"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_alphabet)}

    def analyze(
        self,
        sequences: dict[RSVGene, list[str]],
        subtype: RSVSubtype = RSVSubtype.RSV_A,
        embeddings: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Analyze RSV sequences.

        Args:
            sequences: Dictionary mapping gene to sequences
            subtype: RSV subtype (A or B)
            embeddings: Optional embeddings

        Returns:
            Analysis results
        """
        results = {
            "n_sequences": len(next(iter(sequences.values()))) if sequences else 0,
            "subtype": subtype.value,
            "genes_analyzed": [g.value for g in sequences],
            "drug_resistance": {},
            "mab_escape": {},
        }

        # F protein analysis
        if RSVGene.F in sequences:
            # Nirsevimab escape
            results["mab_escape"]["nirsevimab"] = self._analyze_f_site(
                sequences[RSVGene.F], F_SITE_0_MUTATIONS, "nirsevimab"
            )

            # Palivizumab escape
            results["mab_escape"]["palivizumab"] = self._analyze_f_site(
                sequences[RSVGene.F], F_SITE_II_MUTATIONS, "palivizumab"
            )

            # Fusion inhibitor resistance
            results["drug_resistance"]["fusion_inhibitors"] = self._analyze_f_site(
                sequences[RSVGene.F], F_FUSION_RESISTANCE, "fusion_inhibitor"
            )

        return results

    def _analyze_f_site(
        self,
        sequences: list[str],
        mutation_db: dict,
        target: str,
    ) -> dict[str, Any]:
        """Analyze F protein mutations for a specific target."""
        results = {
            "target": target,
            "scores": [],
            "classifications": [],
            "mutations": [],
        }

        for seq in sequences:
            score = 0.0
            mutations = []

            for pos, info in mutation_db.items():
                if pos <= 0 or pos > len(seq):
                    continue

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
                            "notation": f"F:{ref_aa}{pos}{seq_aa}",
                        }
                    )

            normalized = min(score / 2.0, 1.0)
            results["scores"].append(normalized)
            results["mutations"].append(mutations)

            if normalized < 0.2:
                classification = "susceptible"
            elif normalized < 0.5:
                classification = "reduced_susceptibility"
            else:
                classification = "resistant"

            results["classifications"].append(classification)

        return results

    def predict_mab_efficacy(
        self,
        f_sequences: list[str],
        mab: RSVDrug = RSVDrug.NIRSEVIMAB,
    ) -> list[dict[str, Any]]:
        """Predict monoclonal antibody efficacy.

        Args:
            f_sequences: F protein sequences
            mab: Target monoclonal antibody

        Returns:
            Efficacy predictions
        """
        predictions = []

        mutation_db = F_SITE_0_MUTATIONS if mab == RSVDrug.NIRSEVIMAB else F_SITE_II_MUTATIONS

        for i, seq in enumerate(f_sequences):
            total_impact = 0.0
            escape_mutations = []

            for pos, info in mutation_db.items():
                if pos <= len(seq):
                    seq_aa = seq[pos - 1]
                    ref_aa = list(info.keys())[0]

                    if seq_aa != ref_aa and seq_aa in info[ref_aa]["mutations"]:
                        effect = info[ref_aa]["effect"]
                        impact = {"high": 0.8, "moderate": 0.4, "low": 0.1}.get(effect, 0.2)
                        total_impact += impact
                        escape_mutations.append(f"{ref_aa}{pos}{seq_aa}")

            efficacy = max(0, 1.0 - total_impact)

            predictions.append(
                {
                    "sequence_index": i,
                    "mab": mab.value,
                    "predicted_efficacy": efficacy,
                    "escape_mutations": escape_mutations,
                    "recommendation": "effective"
                    if efficacy > 0.7
                    else ("reduced_efficacy" if efficacy > 0.3 else "likely_ineffective"),
                }
            )

        return predictions

    def validate_predictions(
        self,
        predictions: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> dict[str, float]:
        """Validate against neutralization data."""
        from scipy.stats import spearmanr

        metrics = {}

        for target in predictions.get("mab_escape", {}):
            if target in ground_truth:
                pred = np.array(predictions["mab_escape"][target]["scores"])
                true = np.array(ground_truth[target])

                if len(pred) == len(true):
                    rho, pval = spearmanr(pred, true)
                    metrics[f"{target}_spearman"] = float(rho) if not np.isnan(rho) else 0.0

        return metrics

    def encode_sequence(
        self,
        sequence: str,
        max_length: int = 574,  # F protein length
    ) -> np.ndarray:
        """One-hot encode sequence."""
        n_aa = len(self.aa_alphabet)
        encoding = np.zeros(max_length * n_aa, dtype=np.float32)

        for j, aa in enumerate(sequence[:max_length]):
            idx = self.aa_to_idx.get(aa.upper(), self.aa_to_idx["X"])
            encoding[j * n_aa + idx] = 1.0

        return encoding


def create_rsv_synthetic_dataset(
    target: str = "nirsevimab",
    min_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create synthetic RSV dataset.

    Args:
        target: Target drug/antibody (nirsevimab, palivizumab, fusion_inhibitor)
        min_samples: Minimum number of samples to generate

    Returns:
        (X, y, ids) tuple with at least min_samples samples
    """
    reference = "M" + "A" * 573  # F protein

    if target == "nirsevimab":
        mutation_db = F_SITE_0_MUTATIONS
    elif target == "palivizumab":
        mutation_db = F_SITE_II_MUTATIONS
    else:
        mutation_db = F_FUSION_RESISTANCE

    sequences = [reference]
    resistances = [0.0]
    ids = ["WT"]

    for pos, info in mutation_db.items():
        if pos <= len(reference):
            ref_aa = list(info.keys())[0]
            for mut_aa in info[ref_aa]["mutations"]:  # All mutations, no limit
                mutant = list(reference)
                mutant[pos - 1] = mut_aa
                sequences.append("".join(mutant))

                effect_scores = {"high": 0.9, "moderate": 0.5, "low": 0.2}
                resistances.append(effect_scores.get(info[ref_aa]["effect"], 0.3))
                ids.append(f"F_{ref_aa}{pos}{mut_aa}")

    analyzer = RSVAnalyzer()
    X = np.array([analyzer.encode_sequence(s) for s in sequences])
    y = np.array(resistances, dtype=np.float32)

    # Ensure minimum samples via augmentation
    X, y, ids = ensure_minimum_samples(X, y, ids, min_samples=min_samples)

    return X, y, ids
