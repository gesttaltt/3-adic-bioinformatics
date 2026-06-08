# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Influenza Virus Analyzer for Vaccine Selection and Drug Resistance.

This module provides analysis of influenza A and B viruses for:
- Vaccine strain selection (antigenic drift prediction)
- Drug resistance (oseltamivir, zanamivir, baloxavir)
- Antigenic cartography
- Seasonal evolution tracking

Based on WHO Global Influenza Surveillance and Response System (GISRS)
and GISAID EpiFlu database.

Key Features:
- Hemagglutinin (HA) antigenic distance computation
- Neuraminidase inhibitor resistance (NAIs)
- PA inhibitor resistance (baloxavir)
- Seasonal pattern recognition

Subtypes Supported:
- H1N1 (seasonal, pandemic 2009)
- H3N2 (seasonal, most variable)
- Influenza B (Victoria, Yamagata lineages)
- H5N1 (avian, pandemic potential)

Usage:
    from src.diseases.influenza_analyzer import InfluenzaAnalyzer

    analyzer = InfluenzaAnalyzer()
    results = analyzer.analyze(sequences, subtype="H3N2")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

from .base import DiseaseAnalyzer, DiseaseConfig, DiseaseType, TaskType


class InfluenzaSubtype(Enum):
    """Influenza virus subtypes."""

    H1N1_SEASONAL = "H1N1_seasonal"
    H1N1_PANDEMIC = "H1N1_pdm09"  # 2009 pandemic strain
    H3N2 = "H3N2"
    B_VICTORIA = "B_Victoria"
    B_YAMAGATA = "B_Yamagata"
    H5N1 = "H5N1"  # Avian
    H7N9 = "H7N9"  # Avian


class InfluenzaGene(Enum):
    """Influenza gene segments."""

    # Surface glycoproteins (most important for vaccines/immunity)
    HA = "HA"  # Hemagglutinin - vaccine target
    NA = "NA"  # Neuraminidase - drug target for NAIs

    # Internal proteins
    PB2 = "PB2"  # Polymerase basic 2
    PB1 = "PB1"  # Polymerase basic 1
    PA = "PA"  # Polymerase acidic - baloxavir target
    NP = "NP"  # Nucleoprotein
    M1 = "M1"  # Matrix protein 1
    M2 = "M2"  # Matrix protein 2 (amantadine target)
    NS1 = "NS1"  # Non-structural protein 1
    NS2 = "NS2"  # Nuclear export protein


class InfluenzaDrug(Enum):
    """Influenza antiviral drugs."""

    # Neuraminidase inhibitors (NAIs)
    OSELTAMIVIR = "oseltamivir"  # Tamiflu
    ZANAMIVIR = "zanamivir"  # Relenza
    PERAMIVIR = "peramivir"  # Rapivab
    LANINAMIVIR = "laninamivir"  # Inavir

    # Cap-dependent endonuclease inhibitor
    BALOXAVIR = "baloxavir"  # Xofluza

    # M2 inhibitors (largely resistant now)
    AMANTADINE = "amantadine"
    RIMANTADINE = "rimantadine"


@dataclass
class InfluenzaConfig(DiseaseConfig):
    """Configuration for influenza analysis."""

    name: str = "influenza"
    display_name: str = "Influenza Virus"
    disease_type: DiseaseType = DiseaseType.VIRAL
    tasks: list[TaskType] = field(
        default_factory=lambda: [
            TaskType.RESISTANCE,
            TaskType.ESCAPE,
            TaskType.FITNESS,
        ]
    )

    # Data sources
    data_sources: dict[str, str] = field(
        default_factory=lambda: {
            "gisaid": "https://gisaid.org/",
            "fludb": "https://www.fludb.org/",
            "ncbi_influenza": "https://www.ncbi.nlm.nih.gov/genomes/FLU/",
            "who_cc": "https://www.cdc.gov/flu/about/professionals/who-ccs.htm",
        }
    )


# Neuraminidase (NA) resistance mutations
# Based on WHO antiviral susceptibility monitoring

NA_H1N1_MUTATIONS = {
    # Oseltamivir resistance
    275: {"H": {"mutations": ["Y"], "drug": "oseltamivir", "effect": "high"}},
    295: {"N": {"mutations": ["S"], "drug": "oseltamivir", "effect": "moderate"}},
    136: {"I": {"mutations": ["V", "K", "T"], "drug": "oseltamivir", "effect": "moderate"}},
    # Zanamivir resistance
    119: {"E": {"mutations": ["V", "G", "D"], "drug": "zanamivir", "effect": "moderate"}},
    152: {"R": {"mutations": ["K"], "drug": "zanamivir", "effect": "moderate"}},
    # Cross-resistance
    274: {"H": {"mutations": ["Y"], "drug": "cross", "effect": "high"}},
    293: {"R": {"mutations": ["K"], "drug": "cross", "effect": "high"}},
}

NA_H3N2_MUTATIONS = {
    # H3N2 NA uses different numbering
    119: {"E": {"mutations": ["V", "G", "D", "I", "A"], "drug": "oseltamivir", "effect": "moderate"}},
    151: {"D": {"mutations": ["E", "G", "N", "A", "V"], "drug": "oseltamivir", "effect": "moderate"}},
    198: {"D": {"mutations": ["N"], "drug": "oseltamivir", "effect": "moderate"}},
    222: {"S": {"mutations": ["G", "R", "I", "T"], "drug": "oseltamivir", "effect": "high"}},
    292: {"R": {"mutations": ["K"], "drug": "cross", "effect": "high"}},
    294: {"N": {"mutations": ["S"], "drug": "oseltamivir", "effect": "high"}},
}

NA_B_MUTATIONS = {
    # Influenza B NA
    152: {"R": {"mutations": ["K"], "drug": "cross", "effect": "high"}},
    198: {"D": {"mutations": ["N", "E"], "drug": "oseltamivir", "effect": "moderate"}},
    222: {"I": {"mutations": ["T", "V"], "drug": "cross", "effect": "moderate"}},
    294: {"H": {"mutations": ["Y"], "drug": "oseltamivir", "effect": "high"}},
    371: {"G": {"mutations": ["R"], "drug": "peramivir", "effect": "moderate"}},
}

# Combined NA mutations dictionary by subtype
NA_MUTATIONS_BY_SUBTYPE = {
    "H1N1": NA_H1N1_MUTATIONS,
    "H3N2": NA_H3N2_MUTATIONS,
    "B": NA_B_MUTATIONS,
}


def _merge_na_mutations():
    """Merge all NA mutation dictionaries into a flat structure with 'drugs' key."""
    merged = {}
    for pos, info in NA_H1N1_MUTATIONS.items():
        for ref_aa, data in info.items():
            merged[pos] = {ref_aa: {"mutations": data["mutations"], "effect": data["effect"], "drugs": [data["drug"]]}}
    for pos, info in NA_H3N2_MUTATIONS.items():
        if pos in merged:
            for ref_aa, data in info.items():
                if ref_aa in merged[pos]:
                    if data["drug"] not in merged[pos][ref_aa]["drugs"]:
                        merged[pos][ref_aa]["drugs"].append(data["drug"])
                else:
                    merged[pos][ref_aa] = {
                        "mutations": data["mutations"],
                        "effect": data["effect"],
                        "drugs": [data["drug"]],
                    }
        else:
            for ref_aa, data in info.items():
                merged[pos] = {
                    ref_aa: {"mutations": data["mutations"], "effect": data["effect"], "drugs": [data["drug"]]}
                }
    for pos, info in NA_B_MUTATIONS.items():
        if pos in merged:
            for ref_aa, data in info.items():
                if ref_aa in merged[pos]:
                    if data["drug"] not in merged[pos][ref_aa]["drugs"]:
                        merged[pos][ref_aa]["drugs"].append(data["drug"])
                else:
                    merged[pos][ref_aa] = {
                        "mutations": data["mutations"],
                        "effect": data["effect"],
                        "drugs": [data["drug"]],
                    }
        else:
            for ref_aa, data in info.items():
                merged[pos] = {
                    ref_aa: {"mutations": data["mutations"], "effect": data["effect"], "drugs": [data["drug"]]}
                }
    return merged


# Flat combined NA mutations dictionary with integer position keys
NA_MUTATIONS = _merge_na_mutations()

# PA mutations for baloxavir resistance
PA_MUTATIONS = {
    # PA cap-dependent endonuclease active site
    38: {"I": {"mutations": ["T", "M", "F"], "drug": "baloxavir", "effect": "high"}},
    199: {"E": {"mutations": ["G"], "drug": "baloxavir", "effect": "moderate"}},
    # Substitutions conferring reduced susceptibility
    37: {"A": {"mutations": ["T"], "drug": "baloxavir", "effect": "low"}},
    41: {"E": {"mutations": ["G"], "drug": "baloxavir", "effect": "moderate"}},
}

# M2 mutations (amantadine/rimantadine resistance)
# Most circulating strains are now resistant
M2_MUTATIONS = {
    26: {"L": {"mutations": ["F"], "drug": "m2_inhibitor", "effect": "high"}},
    27: {"V": {"mutations": ["A"], "drug": "m2_inhibitor", "effect": "high"}},
    30: {"A": {"mutations": ["T", "S"], "drug": "m2_inhibitor", "effect": "high"}},
    31: {"S": {"mutations": ["N"], "drug": "m2_inhibitor", "effect": "high"}},  # Most common
    34: {"G": {"mutations": ["E"], "drug": "m2_inhibitor", "effect": "high"}},
}

# HA antigenic sites (for drift prediction)
# H3N2 HA1 antigenic sites - key for vaccine selection
HA_H3N2_ANTIGENIC_SITES = {
    "A": [122, 124, 126, 130, 131, 132, 133, 135, 137, 138, 140, 142, 143, 144, 145, 146],
    "B": [155, 156, 157, 158, 159, 160, 163, 164, 186, 188, 189, 190, 192, 193, 194, 196, 197, 198],
    "C": [44, 45, 46, 47, 48, 50, 51, 53, 54, 275, 276, 278, 279],
    "D": [
        96,
        102,
        103,
        117,
        121,
        167,
        170,
        171,
        172,
        173,
        174,
        175,
        176,
        177,
        179,
        182,
        201,
        203,
        207,
        208,
        209,
        212,
        213,
        214,
        215,
        216,
        217,
    ],
    "E": [57, 59, 62, 63, 67, 75, 78, 81, 82, 83, 86, 87, 88, 91, 92, 94, 109, 260, 261, 262, 265],
}

# Drug to gene mapping
DRUG_GENE_MAP = {
    InfluenzaDrug.OSELTAMIVIR: InfluenzaGene.NA,
    InfluenzaDrug.ZANAMIVIR: InfluenzaGene.NA,
    InfluenzaDrug.PERAMIVIR: InfluenzaGene.NA,
    InfluenzaDrug.LANINAMIVIR: InfluenzaGene.NA,
    InfluenzaDrug.BALOXAVIR: InfluenzaGene.PA,
    InfluenzaDrug.AMANTADINE: InfluenzaGene.M2,
    InfluenzaDrug.RIMANTADINE: InfluenzaGene.M2,
}


class InfluenzaAnalyzer(DiseaseAnalyzer):
    """Analyzer for influenza drug resistance and antigenic drift.

    Provides:
    - NAI resistance prediction (oseltamivir, zanamivir, peramivir)
    - Baloxavir resistance prediction
    - Antigenic distance computation
    - Vaccine strain selection support
    """

    def __init__(self, config: InfluenzaConfig | None = None):
        """Initialize analyzer.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or InfluenzaConfig()
        super().__init__(self.config)

        # Amino acid encoding
        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY-X*"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_alphabet)}

    def analyze(
        self,
        sequences: dict[InfluenzaGene, list[str]],
        subtype: InfluenzaSubtype = InfluenzaSubtype.H3N2,
        embeddings: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Analyze influenza sequences.

        Args:
            sequences: Dictionary mapping gene to list of sequences
            subtype: Influenza subtype
            embeddings: Optional precomputed embeddings

        Returns:
            Analysis results
        """
        results = {
            "n_sequences": len(next(iter(sequences.values()))) if sequences else 0,
            "subtype": subtype.value,
            "genes_analyzed": [g.value for g in sequences],
            "drug_resistance": {},
            "antigenic_analysis": {},
        }

        # Drug resistance
        for drug in InfluenzaDrug:
            gene = DRUG_GENE_MAP.get(drug)
            if gene and gene in sequences:
                drug_results = self.predict_drug_resistance(sequences[gene], drug, subtype)
                results["drug_resistance"][drug.value] = drug_results

        # Antigenic analysis (HA)
        if InfluenzaGene.HA in sequences:
            results["antigenic_analysis"] = self._analyze_antigenic_drift(sequences[InfluenzaGene.HA], subtype)

        return results

    def predict_drug_resistance(
        self,
        sequences: list[str],
        drug: InfluenzaDrug,
        subtype: InfluenzaSubtype,
    ) -> dict[str, Any]:
        """Predict resistance for a specific drug.

        Args:
            sequences: Gene sequences (NA, PA, or M2)
            drug: Target drug
            subtype: Virus subtype

        Returns:
            Resistance predictions
        """
        results = {
            "drug": drug.value,
            "scores": [],
            "classifications": [],
            "mutations": [],
        }

        # Get appropriate mutation database
        if drug in [
            InfluenzaDrug.OSELTAMIVIR,
            InfluenzaDrug.ZANAMIVIR,
            InfluenzaDrug.PERAMIVIR,
            InfluenzaDrug.LANINAMIVIR,
        ]:
            if "H1N1" in subtype.value:
                mutation_db = NA_H1N1_MUTATIONS
            elif subtype == InfluenzaSubtype.H3N2:
                mutation_db = NA_H3N2_MUTATIONS
            elif "B_" in subtype.value:
                mutation_db = NA_B_MUTATIONS
            else:
                mutation_db = NA_H3N2_MUTATIONS  # Default
        elif drug == InfluenzaDrug.BALOXAVIR:
            mutation_db = PA_MUTATIONS
        elif drug in [InfluenzaDrug.AMANTADINE, InfluenzaDrug.RIMANTADINE]:
            mutation_db = M2_MUTATIONS
        else:
            mutation_db = {}

        for seq in sequences:
            score = 0.0
            mutations = []

            for pos, info in mutation_db.items():
                if pos <= 0 or pos > len(seq):
                    continue

                seq_aa = seq[pos - 1]
                ref_aa = list(info.keys())[0]

                if seq_aa != ref_aa and seq_aa in info[ref_aa]["mutations"]:
                    # Check if this mutation affects our target drug
                    mut_drug = info[ref_aa]["drug"]
                    if (
                        mut_drug == drug.value
                        or mut_drug == "cross"
                        or (
                            mut_drug == "m2_inhibitor" and drug in [InfluenzaDrug.AMANTADINE, InfluenzaDrug.RIMANTADINE]
                        )
                    ):
                        effect = info[ref_aa]["effect"]
                        effect_scores = {"high": 1.0, "moderate": 0.5, "low": 0.2}
                        score += effect_scores.get(effect, 0.3)
                        mutations.append(
                            {
                                "position": pos,
                                "ref": ref_aa,
                                "alt": seq_aa,
                                "effect": effect,
                                "notation": f"{ref_aa}{pos}{seq_aa}",
                            }
                        )

            # Normalize
            max_score = 3.0
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

    def _analyze_antigenic_drift(
        self,
        ha_sequences: list[str],
        subtype: InfluenzaSubtype,
    ) -> dict[str, Any]:
        """Analyze antigenic drift from HA sequences.

        Args:
            ha_sequences: HA gene sequences
            subtype: Virus subtype

        Returns:
            Antigenic analysis results
        """
        results = {
            "n_sequences": len(ha_sequences),
            "antigenic_site_mutations": [],
            "drift_scores": [],
            "cluster_potential": [],
        }

        if not ha_sequences:
            return results

        # Use H3N2 antigenic sites for drift analysis
        # (Most studied, most variable)
        antigenic_sites = HA_H3N2_ANTIGENIC_SITES

        # Reference (first sequence or consensus)
        reference = ha_sequences[0]

        for seq in ha_sequences:
            site_mutations = {site: [] for site in antigenic_sites}
            total_antigenic_changes = 0

            for site_name, positions in antigenic_sites.items():
                for pos in positions:
                    if pos <= len(seq) and pos <= len(reference):
                        if seq[pos - 1] != reference[pos - 1]:
                            site_mutations[site_name].append(
                                {
                                    "position": pos,
                                    "ref": reference[pos - 1],
                                    "alt": seq[pos - 1],
                                }
                            )
                            total_antigenic_changes += 1

            results["antigenic_site_mutations"].append(site_mutations)

            # Drift score (normalized)
            total_sites = sum(len(p) for p in antigenic_sites.values())
            drift_score = total_antigenic_changes / max(total_sites, 1)
            results["drift_scores"].append(drift_score)

            # Cluster potential (significant drift if >3% of antigenic sites changed)
            cluster_potential = "low"
            if drift_score > 0.03:
                cluster_potential = "moderate"
            if drift_score > 0.06:
                cluster_potential = "high"
            results["cluster_potential"].append(cluster_potential)

        return results

    def compute_antigenic_distance(
        self,
        seq1: str,
        seq2: str,
        subtype: InfluenzaSubtype = InfluenzaSubtype.H3N2,
    ) -> float:
        """Compute antigenic distance between two HA sequences.

        Uses weighted Hamming distance focusing on antigenic sites.

        Args:
            seq1: First HA sequence
            seq2: Second HA sequence
            subtype: Virus subtype

        Returns:
            Antigenic distance score
        """
        antigenic_sites = HA_H3N2_ANTIGENIC_SITES

        # Weights for different antigenic sites
        # Site A is receptor binding, most important
        site_weights = {
            "A": 2.0,
            "B": 1.5,
            "C": 1.0,
            "D": 1.2,
            "E": 1.0,
        }

        weighted_distance = 0.0
        total_weight = 0.0

        for site_name, positions in antigenic_sites.items():
            weight = site_weights.get(site_name, 1.0)

            for pos in positions:
                if pos <= len(seq1) and pos <= len(seq2):
                    total_weight += weight
                    if seq1[pos - 1] != seq2[pos - 1]:
                        weighted_distance += weight

        if total_weight > 0:
            return weighted_distance / total_weight
        return 0.0

    def recommend_vaccine_strain(
        self,
        candidate_sequences: list[str],
        circulating_sequences: list[str],
        subtype: InfluenzaSubtype = InfluenzaSubtype.H3N2,
    ) -> dict[str, Any]:
        """Recommend vaccine strain from candidates.

        Selects candidate with minimum average antigenic distance
        to circulating strains.

        Args:
            candidate_sequences: Potential vaccine strains
            circulating_sequences: Currently circulating strains
            subtype: Virus subtype

        Returns:
            Recommendation with distances
        """
        results = {
            "n_candidates": len(candidate_sequences),
            "n_circulating": len(circulating_sequences),
            "candidate_scores": [],
            "recommended_index": 0,
            "recommended_score": float("inf"),
        }

        for i, candidate in enumerate(candidate_sequences):
            distances = []
            for circulating in circulating_sequences:
                dist = self.compute_antigenic_distance(candidate, circulating, subtype)
                distances.append(dist)

            avg_distance = np.mean(distances) if distances else float("inf")
            max_distance = np.max(distances) if distances else float("inf")

            results["candidate_scores"].append(
                {
                    "index": i,
                    "avg_distance": float(avg_distance),
                    "max_distance": float(max_distance),
                    "coverage": float(np.mean([d < 0.1 for d in distances])) if distances else 0.0,
                }
            )

            if avg_distance < results["recommended_score"]:
                results["recommended_score"] = float(avg_distance)
                results["recommended_index"] = i

        return results

    def validate_predictions(
        self,
        predictions: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> dict[str, float]:
        """Validate predictions against phenotypic data."""
        from scipy.stats import spearmanr

        metrics = {}

        for drug in predictions.get("drug_resistance", {}):
            if drug in ground_truth:
                pred = np.array(predictions["drug_resistance"][drug]["scores"])
                true = np.array(ground_truth[drug])

                if len(pred) == len(true):
                    rho, pval = spearmanr(pred, true)
                    metrics[f"{drug}_spearman"] = float(rho) if not np.isnan(rho) else 0.0
                    metrics[f"{drug}_pvalue"] = float(pval)

        return metrics

    def encode_sequence(
        self,
        sequence: str,
        max_length: int | None = None,
    ) -> np.ndarray:
        """One-hot encode a sequence."""
        if max_length is None:
            max_length = len(sequence)

        n_aa = len(self.aa_alphabet)
        encoding = np.zeros(max_length * n_aa, dtype=np.float32)

        for j, aa in enumerate(sequence[:max_length]):
            idx = self.aa_to_idx.get(aa.upper(), self.aa_to_idx["X"])
            encoding[j * n_aa + idx] = 1.0

        return encoding


def create_influenza_synthetic_dataset(
    subtype: InfluenzaSubtype = InfluenzaSubtype.H3N2,
    drug: InfluenzaDrug = InfluenzaDrug.OSELTAMIVIR,
    min_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create synthetic influenza dataset for testing.

    In production, use GISAID or FluDB data.

    Args:
        subtype: Influenza subtype
        drug: Target drug for resistance
        min_samples: Minimum number of samples to generate

    Returns:
        (X, y, sequence_ids)
    """
    from src.diseases.utils.synthetic_data import (
        create_mutation_based_dataset,
        ensure_minimum_samples,
    )

    # Select mutation database based on subtype
    if "H3N2" in subtype.value:
        mutation_db = NA_H3N2_MUTATIONS
    elif "H1N1" in subtype.value:
        mutation_db = NA_H1N1_MUTATIONS
    elif "B_" in subtype.value:
        mutation_db = NA_B_MUTATIONS
    else:
        mutation_db = NA_H3N2_MUTATIONS  # Default

    # Build reference sequence with correct wild-type amino acids at mutation positions
    # NA protein is ~470 AA, use 500 to cover all positions
    max_pos = max(mutation_db.keys()) if mutation_db else 300
    ref_length = max(500, max_pos + 10)

    # Start with placeholder sequence
    reference = list("M" + "A" * (ref_length - 1))

    # Set correct wild-type amino acids at each mutation position
    # This ensures WT encodes differently from mutants
    for pos, info in mutation_db.items():
        if pos <= len(reference):
            ref_aa = list(info.keys())[0]  # Get expected WT amino acid (e.g., 'E' for position 119)
            reference[pos - 1] = ref_aa

    reference = "".join(reference)

    analyzer = InfluenzaAnalyzer()
    max_len = ref_length

    # Use utility to create dataset with proper mutation combinations
    X, y, ids = create_mutation_based_dataset(
        reference_sequence=reference,
        mutation_db=mutation_db,
        encode_fn=analyzer.encode_sequence,
        max_length=max_len,
        n_random_mutants=30,
        seed=42,
    )

    # Ensure minimum samples
    X, y, ids = ensure_minimum_samples(X, y, ids, min_samples=min_samples, seed=42)

    return X, y, ids
