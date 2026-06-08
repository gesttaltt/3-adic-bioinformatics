# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Plasmodium Malaria Drug Resistance Analyzer.

This module provides analysis of antimalarial drug resistance mutations
in Plasmodium falciparum and P. vivax.

Based on:
- WHO Artemisinin Resistance Markers (2024)
- PlasmoDB (https://plasmodb.org/)
- MalariaGEN (https://www.malariagen.net/)

Key Features:
- Artemisinin resistance (PfKelch13)
- Chloroquine resistance (PfCRT, PfMDR1)
- Sulfadoxine-pyrimethamine resistance (PfDHFR, PfDHPS)
- Piperaquine resistance (PfPlasmepsin 2/3)
- Multi-drug resistance detection

Artemisinin resistance is a WHO priority due to:
- Delayed parasite clearance
- Spread in Southeast Asia
- Threat to global malaria control

Usage:
    from src.diseases.malaria_analyzer import MalariaAnalyzer

    analyzer = MalariaAnalyzer()
    results = analyzer.analyze(sequences, species="Pf")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

from .base import DiseaseAnalyzer, DiseaseConfig, DiseaseType, TaskType
from .utils.synthetic_data import ensure_minimum_samples


class PlasmodiumSpecies(Enum):
    """Plasmodium species."""

    P_FALCIPARUM = "Pf"  # Most deadly
    P_VIVAX = "Pv"  # Most widespread
    P_MALARIAE = "Pm"
    P_OVALE = "Po"
    P_KNOWLESI = "Pk"  # Zoonotic


class MalariaGene(Enum):
    """Key genes for drug resistance."""

    # Artemisinin resistance
    KELCH13 = "pfkelch13"

    # Chloroquine/Amodiaquine resistance
    CRT = "pfcrt"
    MDR1 = "pfmdr1"

    # Antifolate resistance
    DHFR = "pfdhfr"
    DHPS = "pfdhps"

    # Piperaquine resistance
    PLASMEPSIN2 = "pfpm2"
    PLASMEPSIN3 = "pfpm3"

    # Atovaquone resistance
    CYTB = "pfcytb"


class MalariaDrug(Enum):
    """Antimalarial drugs."""

    # Artemisinin combination therapies (ACTs)
    ARTEMETHER = "artemether"
    ARTESUNATE = "artesunate"
    DIHYDROARTEMISININ = "dha"

    # Partner drugs
    LUMEFANTRINE = "lumefantrine"
    PIPERAQUINE = "piperaquine"
    AMODIAQUINE = "amodiaquine"
    MEFLOQUINE = "mefloquine"

    # Older drugs
    CHLOROQUINE = "chloroquine"
    SULFADOXINE = "sulfadoxine"
    PYRIMETHAMINE = "pyrimethamine"

    # Others
    ATOVAQUONE = "atovaquone"
    PROGUANIL = "proguanil"


@dataclass
class MalariaConfig(DiseaseConfig):
    """Configuration for malaria analysis."""

    name: str = "malaria"
    display_name: str = "Malaria (Plasmodium)"
    disease_type: DiseaseType = DiseaseType.PARASITIC
    tasks: list[TaskType] = field(
        default_factory=lambda: [
            TaskType.RESISTANCE,
        ]
    )

    data_sources: dict[str, str] = field(
        default_factory=lambda: {
            "plasmodb": "https://plasmodb.org/",
            "malariagen": "https://www.malariagen.net/",
            "who_markers": "https://www.who.int/publications/",
            "wwarn": "https://www.wwarn.org/",
        }
    )


# PfKelch13 Artemisinin Resistance Mutations
# WHO validated and candidate markers

KELCH13_MUTATIONS = {
    # WHO Validated Markers (associated with delayed clearance)
    449: {"G": {"mutations": ["A"], "effect": "high", "category": "validated"}},
    458: {"N": {"mutations": ["Y"], "effect": "high", "category": "validated"}},
    469: {"C": {"mutations": ["Y", "F"], "effect": "high", "category": "validated"}},
    476: {"M": {"mutations": ["I"], "effect": "high", "category": "validated"}},
    493: {"Y": {"mutations": ["H"], "effect": "high", "category": "validated"}},
    539: {"R": {"mutations": ["T"], "effect": "high", "category": "validated"}},
    543: {"I": {"mutations": ["T"], "effect": "high", "category": "validated"}},
    553: {"P": {"mutations": ["L"], "effect": "high", "category": "validated"}},
    561: {"R": {"mutations": ["H"], "effect": "high", "category": "validated"}},
    574: {"P": {"mutations": ["L"], "effect": "high", "category": "validated"}},
    575: {"C": {"mutations": ["Y"], "effect": "high", "category": "validated"}},
    580: {"C": {"mutations": ["Y"], "effect": "high", "category": "validated"}},  # Most common
    # WHO Candidate/Associated Markers
    441: {"P": {"mutations": ["L"], "effect": "moderate", "category": "candidate"}},
    481: {"A": {"mutations": ["V"], "effect": "moderate", "category": "candidate"}},
    527: {"N": {"mutations": ["I"], "effect": "moderate", "category": "candidate"}},
    533: {"G": {"mutations": ["S"], "effect": "moderate", "category": "candidate"}},
    568: {"V": {"mutations": ["G"], "effect": "moderate", "category": "candidate"}},
    622: {"A": {"mutations": ["I"], "effect": "moderate", "category": "candidate"}},
    675: {"A": {"mutations": ["V"], "effect": "moderate", "category": "candidate"}},
}

# PfCRT Chloroquine Resistance
CRT_MUTATIONS = {
    72: {"C": {"mutations": ["S"], "effect": "high", "haplotype": "CVIET/SVMNT"}},
    74: {"M": {"mutations": ["I"], "effect": "high", "haplotype": "CVIET"}},
    75: {"N": {"mutations": ["E"], "effect": "high", "haplotype": "CVIET"}},
    76: {"K": {"mutations": ["T"], "effect": "high", "haplotype": "all"}},  # Key marker
    220: {"A": {"mutations": ["S"], "effect": "moderate", "haplotype": "associated"}},
    271: {"Q": {"mutations": ["E"], "effect": "moderate", "haplotype": "associated"}},
    326: {"N": {"mutations": ["S", "D"], "effect": "moderate", "haplotype": "associated"}},
    356: {"I": {"mutations": ["T", "L"], "effect": "moderate", "haplotype": "associated"}},
    371: {"R": {"mutations": ["I"], "effect": "moderate", "haplotype": "associated"}},
}

# PfMDR1 Multi-drug resistance
MDR1_MUTATIONS = {
    86: {"N": {"mutations": ["Y"], "effect": "high", "drugs": ["chloroquine", "amodiaquine", "lumefantrine"]}},
    184: {"Y": {"mutations": ["F"], "effect": "moderate", "drugs": ["lumefantrine", "mefloquine"]}},
    1034: {"S": {"mutations": ["C"], "effect": "moderate", "drugs": ["mefloquine"]}},
    1042: {"N": {"mutations": ["D"], "effect": "moderate", "drugs": ["mefloquine", "lumefantrine"]}},
    1246: {"D": {"mutations": ["Y"], "effect": "moderate", "drugs": ["amodiaquine", "chloroquine"]}},
}

# PfDHFR Antifolate (pyrimethamine) resistance
DHFR_MUTATIONS = {
    51: {"N": {"mutations": ["I"], "effect": "high"}},
    59: {"C": {"mutations": ["R"], "effect": "high"}},
    108: {"S": {"mutations": ["N", "T"], "effect": "high"}},  # Key marker
    164: {"I": {"mutations": ["L"], "effect": "high"}},  # High-level resistance
}

# PfDHPS Antifolate (sulfadoxine) resistance
DHPS_MUTATIONS = {
    436: {"S": {"mutations": ["A", "F"], "effect": "moderate"}},
    437: {"A": {"mutations": ["G"], "effect": "high"}},  # Key marker
    540: {"K": {"mutations": ["E"], "effect": "high"}},
    581: {"A": {"mutations": ["G"], "effect": "high"}},
    613: {"A": {"mutations": ["T", "S"], "effect": "moderate"}},
}

# PfCytb Atovaquone resistance
CYTB_MUTATIONS = {
    268: {"Y": {"mutations": ["S", "C", "N"], "effect": "high"}},  # Most common
}


class MalariaAnalyzer(DiseaseAnalyzer):
    """Analyzer for Plasmodium drug resistance.

    Provides:
    - Artemisinin resistance detection (K13 mutations)
    - Partner drug resistance profiling
    - Multi-drug resistance assessment
    - Geographic spread tracking support
    """

    def __init__(self, config: MalariaConfig | None = None):
        """Initialize analyzer."""
        self.config = config or MalariaConfig()
        super().__init__(self.config)

        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY-X*"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_alphabet)}

    def analyze(
        self,
        sequences: dict[MalariaGene, list[str]],
        species: PlasmodiumSpecies = PlasmodiumSpecies.P_FALCIPARUM,
        embeddings: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Analyze Plasmodium sequences for drug resistance.

        Args:
            sequences: Dictionary mapping gene to sequences
            species: Plasmodium species
            embeddings: Optional embeddings

        Returns:
            Analysis results
        """
        results = {
            "n_sequences": len(next(iter(sequences.values()))) if sequences else 0,
            "species": species.value,
            "genes_analyzed": [g.value for g in sequences],
            "drug_resistance": {},
            "artemisinin_status": [],
            "act_efficacy": [],
        }

        # Artemisinin resistance (K13)
        if MalariaGene.KELCH13 in sequences:
            k13_results = self._analyze_kelch13(sequences[MalariaGene.KELCH13])
            results["drug_resistance"]["artemisinin"] = k13_results
            results["artemisinin_status"] = k13_results.get("classifications", [])

        # Chloroquine resistance (CRT)
        if MalariaGene.CRT in sequences:
            results["drug_resistance"]["chloroquine"] = self._analyze_crt(sequences[MalariaGene.CRT])

        # MDR1
        if MalariaGene.MDR1 in sequences:
            results["drug_resistance"]["mdr1_profile"] = self._analyze_mdr1(sequences[MalariaGene.MDR1])

        # Antifolates (DHFR + DHPS)
        if MalariaGene.DHFR in sequences:
            results["drug_resistance"]["pyrimethamine"] = self._analyze_dhfr(sequences[MalariaGene.DHFR])
        if MalariaGene.DHPS in sequences:
            results["drug_resistance"]["sulfadoxine"] = self._analyze_dhps(sequences[MalariaGene.DHPS])

        # Atovaquone
        if MalariaGene.CYTB in sequences:
            results["drug_resistance"]["atovaquone"] = self._analyze_cytb(sequences[MalariaGene.CYTB])

        # ACT efficacy prediction
        results["act_efficacy"] = self._predict_act_efficacy(results["drug_resistance"])

        return results

    def _analyze_kelch13(self, sequences: list[str]) -> dict[str, Any]:
        """Analyze K13 propeller domain for artemisinin resistance."""
        results = {
            "gene": "pfkelch13",
            "scores": [],
            "classifications": [],
            "mutations": [],
            "who_category": [],
        }

        for seq in sequences:
            score = 0.0
            mutations = []
            categories = []

            for pos, info in KELCH13_MUTATIONS.items():
                if pos <= 0 or pos > len(seq):
                    continue

                seq_aa = seq[pos - 1]
                ref_aa = list(info.keys())[0]

                if seq_aa != ref_aa and seq_aa in info[ref_aa]["mutations"]:
                    effect = info[ref_aa]["effect"]
                    category = info[ref_aa]["category"]
                    effect_scores = {"high": 1.0, "moderate": 0.5}
                    score += effect_scores.get(effect, 0.3)
                    categories.append(category)

                    mutations.append(
                        {
                            "position": pos,
                            "ref": ref_aa,
                            "alt": seq_aa,
                            "effect": effect,
                            "who_category": category,
                            "notation": f"K13 {ref_aa}{pos}{seq_aa}",
                        }
                    )

            # Normalize
            max_score = 3.0
            normalized = min(score / max_score, 1.0)

            results["scores"].append(normalized)
            results["mutations"].append(mutations)

            # WHO classification
            if any(c == "validated" for c in categories):
                results["who_category"].append("WHO_validated")
                classification = "artemisinin_resistant"
            elif any(c == "candidate" for c in categories):
                results["who_category"].append("WHO_candidate")
                classification = "suspected_resistant"
            else:
                results["who_category"].append("none")
                classification = "susceptible"

            results["classifications"].append(classification)

        return results

    def _analyze_crt(self, sequences: list[str]) -> dict[str, Any]:
        """Analyze PfCRT for chloroquine resistance."""
        return self._scan_mutations(sequences, CRT_MUTATIONS, "pfcrt", "chloroquine")

    def _analyze_mdr1(self, sequences: list[str]) -> dict[str, Any]:
        """Analyze PfMDR1 for multi-drug resistance."""
        return self._scan_mutations(sequences, MDR1_MUTATIONS, "pfmdr1", "multiple")

    def _analyze_dhfr(self, sequences: list[str]) -> dict[str, Any]:
        """Analyze PfDHFR for pyrimethamine resistance."""
        return self._scan_mutations(sequences, DHFR_MUTATIONS, "pfdhfr", "pyrimethamine")

    def _analyze_dhps(self, sequences: list[str]) -> dict[str, Any]:
        """Analyze PfDHPS for sulfadoxine resistance."""
        return self._scan_mutations(sequences, DHPS_MUTATIONS, "pfdhps", "sulfadoxine")

    def _analyze_cytb(self, sequences: list[str]) -> dict[str, Any]:
        """Analyze PfCytb for atovaquone resistance."""
        return self._scan_mutations(sequences, CYTB_MUTATIONS, "pfcytb", "atovaquone")

    def _scan_mutations(
        self,
        sequences: list[str],
        mutation_db: dict,
        gene: str,
        drug: str,
    ) -> dict[str, Any]:
        """Generic mutation scanning."""
        results = {
            "gene": gene,
            "drug": drug,
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
                    effect = info[ref_aa].get("effect", "moderate")
                    effect_scores = {"high": 1.0, "moderate": 0.5, "low": 0.2}
                    score += effect_scores.get(effect, 0.3)

                    mutations.append(
                        {
                            "position": pos,
                            "ref": ref_aa,
                            "alt": seq_aa,
                            "effect": effect,
                            "notation": f"{gene}_{ref_aa}{pos}{seq_aa}",
                        }
                    )

            max_score = 4.0
            normalized = min(score / max_score, 1.0)

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

    def _predict_act_efficacy(self, drug_resistance: dict) -> list[dict[str, Any]]:
        """Predict ACT efficacy based on resistance profiles."""
        efficacy = []

        # Number of sequences (from artemisinin data if available)
        n_seq = len(drug_resistance.get("artemisinin", {}).get("scores", []))

        for i in range(n_seq):
            # Get scores for each component
            art_score = (
                drug_resistance.get("artemisinin", {}).get("scores", [0])[i]
                if i < len(drug_resistance.get("artemisinin", {}).get("scores", []))
                else 0
            )

            # Check partner drugs
            mdr1_score = (
                drug_resistance.get("mdr1_profile", {}).get("scores", [0])[i]
                if i < len(drug_resistance.get("mdr1_profile", {}).get("scores", []))
                else 0
            )

            # Predict efficacy for common ACTs
            al_efficacy = 1.0 - (art_score * 0.5 + mdr1_score * 0.3)  # Artemether-lumefantrine
            dha_ppq = 1.0 - (art_score * 0.6)  # DHA-piperaquine

            efficacy.append(
                {
                    "isolate": i,
                    "AL_efficacy": max(0, min(1, al_efficacy)),
                    "DHAPPQ_efficacy": max(0, min(1, dha_ppq)),
                    "treatment_recommendation": self._get_treatment_rec(art_score, mdr1_score),
                }
            )

        return efficacy

    def _get_treatment_rec(self, art_score: float, mdr1_score: float) -> str:
        """Get treatment recommendation."""
        if art_score > 0.5:
            return "Consider extended ACT duration or alternative"
        elif mdr1_score > 0.5:
            return "Monitor treatment response"
        else:
            return "Standard ACT regimen"

    def validate_predictions(
        self,
        predictions: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> dict[str, float]:
        """Validate against phenotypic data."""
        from scipy.stats import spearmanr

        metrics = {}

        for gene in predictions.get("drug_resistance", {}):
            if gene in ground_truth:
                pred = np.array(predictions["drug_resistance"][gene]["scores"])
                true = np.array(ground_truth[gene])

                if len(pred) == len(true):
                    rho, pval = spearmanr(pred, true)
                    metrics[f"{gene}_spearman"] = float(rho) if not np.isnan(rho) else 0.0

        return metrics

    def encode_sequence(
        self,
        sequence: str,
        max_length: int = 726,  # K13 propeller domain
    ) -> np.ndarray:
        """One-hot encode sequence."""
        n_aa = len(self.aa_alphabet)
        encoding = np.zeros(max_length * n_aa, dtype=np.float32)

        for j, aa in enumerate(sequence[:max_length]):
            idx = self.aa_to_idx.get(aa.upper(), self.aa_to_idx["X"])
            encoding[j * n_aa + idx] = 1.0

        return encoding


def create_malaria_synthetic_dataset(
    gene: MalariaGene = MalariaGene.KELCH13,
    min_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create synthetic malaria dataset for testing.

    Args:
        gene: Target gene (KELCH13, CRT, MDR1, etc.)
        min_samples: Minimum number of samples to generate

    Returns:
        (X, y, ids) tuple with at least min_samples samples
    """
    if gene == MalariaGene.KELCH13:
        mutation_db = KELCH13_MUTATIONS
    elif gene == MalariaGene.CRT:
        mutation_db = CRT_MUTATIONS
    else:
        mutation_db = KELCH13_MUTATIONS

    # Build reference with correct WT amino acids at mutation positions
    max_pos = max(mutation_db.keys()) if mutation_db else 700
    ref_length = max(726, max_pos + 10)
    reference = list("M" + "A" * (ref_length - 1))

    # Set correct wild-type amino acids at each mutation position
    for pos, info in mutation_db.items():
        if pos <= ref_length:
            ref_aa = list(info.keys())[0]
            reference[pos - 1] = ref_aa

    reference = "".join(reference)

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
                ids.append(f"{gene.value}_{ref_aa}{pos}{mut_aa}")

    analyzer = MalariaAnalyzer()
    X = np.array([analyzer.encode_sequence(s) for s in sequences])
    y = np.array(resistances, dtype=np.float32)

    # Ensure minimum samples via augmentation
    X, y, ids = ensure_minimum_samples(X, y, ids, min_samples=min_samples)

    return X, y, ids
