# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Hepatitis C Virus (HCV) Direct-Acting Antiviral Resistance Analyzer.

This module provides analysis of HCV drug resistance mutations for
direct-acting antivirals (DAAs) targeting NS3/4A, NS5A, and NS5B.

Based on:
- EASL HCV Treatment Guidelines
- HCV-Glue database (http://hcv-glue.cvr.gla.ac.uk/)
- AASLD-IDSA HCV Guidance

Key Features:
- NS3/4A protease inhibitor resistance (grazoprevir, glecaprevir, voxilaprevir)
- NS5A inhibitor resistance (ledipasvir, velpatasvir, pibrentasvir)
- NS5B polymerase inhibitor resistance (sofosbuvir)
- Genotype-specific resistance profiles
- Resistance-associated substitutions (RAS) detection

HCV Genotypes:
- GT1a/1b (most common in US/Europe)
- GT2 (good DAA response)
- GT3 (harder to treat)
- GT4, GT5, GT6 (regional)

Usage:
    from src.diseases.hcv_analyzer import HCVAnalyzer

    analyzer = HCVAnalyzer()
    results = analyzer.analyze(sequences, genotype="1a", gene=HCVGene.NS5A)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

from .base import DiseaseAnalyzer, DiseaseConfig, DiseaseType, TaskType
from .utils.synthetic_data import ensure_minimum_samples


class HCVGenotype(Enum):
    """HCV genotypes."""

    GT1A = "1a"
    GT1B = "1b"
    GT2 = "2"
    GT3 = "3"
    GT4 = "4"
    GT5 = "5"
    GT6 = "6"


class HCVGene(Enum):
    """HCV genes relevant for DAA resistance."""

    NS3 = "NS3"  # Protease - target for PIs
    NS4A = "NS4A"  # Protease cofactor
    NS5A = "NS5A"  # Replication complex - target for NS5A inhibitors
    NS5B = "NS5B"  # Polymerase - target for nucleos(t)ide/non-nucleoside inhibitors


class HCVDrug(Enum):
    """HCV direct-acting antivirals."""

    # NS3/4A Protease Inhibitors (PIs)
    SIMEPREVIR = "simeprevir"
    PARITAPREVIR = "paritaprevir"
    GRAZOPREVIR = "grazoprevir"
    GLECAPREVIR = "glecaprevir"
    VOXILAPREVIR = "voxilaprevir"

    # NS5A Inhibitors
    LEDIPASVIR = "ledipasvir"
    DACLATASVIR = "daclatasvir"
    OMBITASVIR = "ombitasvir"
    ELBASVIR = "elbasvir"
    VELPATASVIR = "velpatasvir"
    PIBRENTASVIR = "pibrentasvir"

    # NS5B Nucleos(t)ide Inhibitors
    SOFOSBUVIR = "sofosbuvir"

    # NS5B Non-nucleoside Inhibitors
    DASABUVIR = "dasabuvir"


@dataclass
class HCVConfig(DiseaseConfig):
    """Configuration for HCV analysis."""

    name: str = "hcv"
    display_name: str = "Hepatitis C Virus"
    disease_type: DiseaseType = DiseaseType.VIRAL
    tasks: list[TaskType] = field(
        default_factory=lambda: [
            TaskType.RESISTANCE,
        ]
    )

    data_sources: dict[str, str] = field(
        default_factory=lambda: {
            "hcv_glue": "http://hcv-glue.cvr.gla.ac.uk/",
            "easl": "https://easl.eu/",
            "aasld_idsa": "https://www.hcvguidelines.org/",
        }
    )


# NS3 Protease Inhibitor Resistance-Associated Substitutions (RAS)
# Positions and mutations that confer resistance

NS3_RAS_GT1A = {
    # Major resistance positions
    36: {"V": {"mutations": ["A", "M", "L", "G"], "effect": "moderate", "drugs": ["simeprevir", "paritaprevir"]}},
    43: {"Q": {"mutations": ["K", "R"], "effect": "low", "drugs": ["paritaprevir"]}},
    54: {"T": {"mutations": ["A", "S"], "effect": "moderate", "drugs": ["paritaprevir", "grazoprevir"]}},
    55: {"V": {"mutations": ["A", "I"], "effect": "low", "drugs": ["paritaprevir"]}},
    56: {
        "Y": {
            "mutations": ["H", "F"],
            "effect": "high",
            "drugs": ["simeprevir", "grazoprevir", "glecaprevir", "voxilaprevir"],
        }
    },
    80: {"Q": {"mutations": ["K", "R", "L"], "effect": "moderate", "drugs": ["simeprevir"]}},
    122: {"S": {"mutations": ["G", "R"], "effect": "low", "drugs": ["simeprevir"]}},
    155: {
        "R": {"mutations": ["K", "T", "S"], "effect": "high", "drugs": ["simeprevir", "paritaprevir", "grazoprevir"]}
    },
    156: {"A": {"mutations": ["G", "T", "V", "S"], "effect": "high", "drugs": ["all_pi"]}},
    168: {
        "D": {
            "mutations": ["A", "E", "G", "H", "K", "N", "T", "V", "Y"],
            "effect": "high",
            "drugs": ["simeprevir", "paritaprevir", "grazoprevir", "glecaprevir", "voxilaprevir"],
        }
    },
    170: {"I": {"mutations": ["V", "T"], "effect": "low", "drugs": ["simeprevir"]}},
    175: {"M": {"mutations": ["L"], "effect": "low", "drugs": ["paritaprevir"]}},
}

NS3_RAS_GT1B = {
    36: {"V": {"mutations": ["A", "M", "L"], "effect": "low", "drugs": ["simeprevir"]}},
    54: {"T": {"mutations": ["A", "S"], "effect": "moderate", "drugs": ["paritaprevir"]}},
    55: {"V": {"mutations": ["A"], "effect": "low", "drugs": ["paritaprevir"]}},
    56: {"Y": {"mutations": ["H", "F"], "effect": "high", "drugs": ["simeprevir", "grazoprevir"]}},
    80: {"Q": {"mutations": ["K", "R"], "effect": "low", "drugs": ["simeprevir"]}},
    155: {"R": {"mutations": ["K", "Q"], "effect": "moderate", "drugs": ["simeprevir", "paritaprevir"]}},
    156: {"A": {"mutations": ["G", "T", "V", "S"], "effect": "high", "drugs": ["all_pi"]}},
    168: {"D": {"mutations": ["A", "E", "G", "V", "Y"], "effect": "high", "drugs": ["simeprevir", "grazoprevir"]}},
    170: {"V": {"mutations": ["I", "T"], "effect": "low", "drugs": ["simeprevir"]}},
}

# NS5A Resistance-Associated Substitutions
NS5A_RAS_GT1A = {
    24: {"K": {"mutations": ["R"], "effect": "low", "drugs": ["ledipasvir"]}},
    28: {
        "M": {
            "mutations": ["T", "V", "A"],
            "effect": "high",
            "drugs": ["ledipasvir", "daclatasvir", "elbasvir", "velpatasvir"],
        }
    },
    29: {"P": {"mutations": ["R"], "effect": "low", "drugs": ["daclatasvir"]}},
    30: {
        "L": {
            "mutations": ["R", "H", "K", "S", "Q"],
            "effect": "high",
            "drugs": ["ledipasvir", "daclatasvir", "elbasvir", "velpatasvir"],
        }
    },
    31: {
        "L": {
            "mutations": ["M", "V", "F"],
            "effect": "high",
            "drugs": ["ledipasvir", "daclatasvir", "elbasvir", "velpatasvir"],
        }
    },
    32: {"P": {"mutations": ["L"], "effect": "low", "drugs": ["daclatasvir"]}},
    58: {"H": {"mutations": ["D"], "effect": "moderate", "drugs": ["ledipasvir", "daclatasvir", "elbasvir"]}},
    62: {"S": {"mutations": ["L"], "effect": "low", "drugs": ["daclatasvir"]}},
    92: {"A": {"mutations": ["K", "T"], "effect": "moderate", "drugs": ["elbasvir", "pibrentasvir"]}},
    93: {
        "Y": {
            "mutations": ["C", "F", "H", "N", "S"],
            "effect": "high",
            "drugs": ["ledipasvir", "daclatasvir", "elbasvir", "velpatasvir"],
        }
    },
}

NS5A_RAS_GT1B = {
    28: {"L": {"mutations": ["M", "T"], "effect": "moderate", "drugs": ["daclatasvir"]}},
    30: {"L": {"mutations": ["R"], "effect": "low", "drugs": ["daclatasvir"]}},
    31: {
        "L": {
            "mutations": ["M", "V", "F", "I"],
            "effect": "high",
            "drugs": ["ledipasvir", "daclatasvir", "ombitasvir", "elbasvir"],
        }
    },
    54: {"P": {"mutations": ["S"], "effect": "low", "drugs": ["daclatasvir"]}},
    58: {"H": {"mutations": ["D"], "effect": "low", "drugs": ["daclatasvir"]}},
    92: {"A": {"mutations": ["K"], "effect": "moderate", "drugs": ["elbasvir"]}},
    93: {"Y": {"mutations": ["H", "N", "S"], "effect": "high", "drugs": ["ledipasvir", "daclatasvir", "elbasvir"]}},
}

NS5A_RAS_GT3 = {
    # GT3 is harder to treat
    30: {"A": {"mutations": ["K", "S"], "effect": "high", "drugs": ["daclatasvir", "velpatasvir"]}},
    31: {"L": {"mutations": ["M"], "effect": "moderate", "drugs": ["daclatasvir"]}},
    62: {"S": {"mutations": ["L"], "effect": "low", "drugs": ["daclatasvir"]}},
    93: {"Y": {"mutations": ["H"], "effect": "high", "drugs": ["daclatasvir", "velpatasvir"]}},
}

# NS5B Polymerase RAS (rare for sofosbuvir)
NS5B_RAS = {
    # Sofosbuvir resistance is rare
    159: {"L": {"mutations": ["F"], "effect": "high", "drugs": ["sofosbuvir"]}},
    282: {"S": {"mutations": ["T", "R", "G", "C"], "effect": "high", "drugs": ["sofosbuvir"]}},
    320: {"C": {"mutations": ["S"], "effect": "moderate", "drugs": ["sofosbuvir"]}},
    321: {"V": {"mutations": ["A", "I"], "effect": "moderate", "drugs": ["sofosbuvir"]}},
    # Dasabuvir resistance
    314: {"L": {"mutations": ["H"], "effect": "low", "drugs": ["dasabuvir"]}},
    316: {"C": {"mutations": ["N", "Y", "H"], "effect": "high", "drugs": ["dasabuvir"]}},
    368: {"S": {"mutations": ["T"], "effect": "moderate", "drugs": ["dasabuvir"]}},
    411: {"M": {"mutations": ["T", "I"], "effect": "high", "drugs": ["dasabuvir"]}},
    414: {"A": {"mutations": ["T", "V"], "effect": "moderate", "drugs": ["dasabuvir"]}},
    448: {"Y": {"mutations": ["C", "H"], "effect": "high", "drugs": ["dasabuvir"]}},
    553: {"A": {"mutations": ["T", "V"], "effect": "moderate", "drugs": ["dasabuvir"]}},
    556: {"G": {"mutations": ["R"], "effect": "moderate", "drugs": ["dasabuvir"]}},
    559: {"D": {"mutations": ["G", "N"], "effect": "moderate", "drugs": ["dasabuvir"]}},
}


class HCVAnalyzer(DiseaseAnalyzer):
    """Analyzer for HCV DAA resistance.

    Provides:
    - RAS detection for NS3/4A, NS5A, NS5B
    - Genotype-specific resistance profiles
    - Treatment regimen recommendations
    """

    def __init__(self, config: HCVConfig | None = None):
        """Initialize analyzer."""
        self.config = config or HCVConfig()
        super().__init__(self.config)

        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY-X*"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_alphabet)}

    def analyze(
        self,
        sequences: dict[HCVGene, list[str]],
        genotype: HCVGenotype = HCVGenotype.GT1A,
        embeddings: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Analyze HCV sequences for drug resistance.

        Args:
            sequences: Dictionary mapping gene to sequences
            genotype: HCV genotype
            embeddings: Optional precomputed embeddings

        Returns:
            Analysis results
        """
        results = {
            "n_sequences": len(next(iter(sequences.values()))) if sequences else 0,
            "genotype": genotype.value,
            "genes_analyzed": [g.value for g in sequences],
            "drug_resistance": {},
            "ras_summary": {},
        }

        # Analyze each gene
        if HCVGene.NS3 in sequences:
            results["drug_resistance"]["NS3_PI"] = self._analyze_ns3(sequences[HCVGene.NS3], genotype)

        if HCVGene.NS5A in sequences:
            results["drug_resistance"]["NS5A_inhibitors"] = self._analyze_ns5a(sequences[HCVGene.NS5A], genotype)

        if HCVGene.NS5B in sequences:
            results["drug_resistance"]["NS5B_inhibitors"] = self._analyze_ns5b(sequences[HCVGene.NS5B])

        # RAS summary
        results["ras_summary"] = self._summarize_ras(results["drug_resistance"])

        return results

    def _analyze_ns3(
        self,
        sequences: list[str],
        genotype: HCVGenotype,
    ) -> dict[str, Any]:
        """Analyze NS3 protease for PI resistance."""
        if genotype in [HCVGenotype.GT1A]:
            mutation_db = NS3_RAS_GT1A
        elif genotype in [HCVGenotype.GT1B]:
            mutation_db = NS3_RAS_GT1B
        else:
            mutation_db = NS3_RAS_GT1A  # Default

        return self._scan_mutations(sequences, mutation_db, "NS3")

    def _analyze_ns5a(
        self,
        sequences: list[str],
        genotype: HCVGenotype,
    ) -> dict[str, Any]:
        """Analyze NS5A for inhibitor resistance."""
        if genotype == HCVGenotype.GT1A:
            mutation_db = NS5A_RAS_GT1A
        elif genotype == HCVGenotype.GT1B:
            mutation_db = NS5A_RAS_GT1B
        elif genotype == HCVGenotype.GT3:
            mutation_db = NS5A_RAS_GT3
        else:
            mutation_db = NS5A_RAS_GT1A

        return self._scan_mutations(sequences, mutation_db, "NS5A")

    def _analyze_ns5b(
        self,
        sequences: list[str],
    ) -> dict[str, Any]:
        """Analyze NS5B polymerase for resistance."""
        return self._scan_mutations(sequences, NS5B_RAS, "NS5B")

    def _scan_mutations(
        self,
        sequences: list[str],
        mutation_db: dict,
        gene: str,
    ) -> dict[str, Any]:
        """Scan sequences for RAS mutations."""
        results = {
            "gene": gene,
            "scores": [],
            "classifications": [],
            "mutations": [],
            "affected_drugs": [],
        }

        for seq in sequences:
            score = 0.0
            mutations = []
            drugs_affected = set()

            for pos, info in mutation_db.items():
                if pos <= 0 or pos > len(seq):
                    continue

                seq_aa = seq[pos - 1]
                ref_aa = list(info.keys())[0]

                if seq_aa != ref_aa and seq_aa in info[ref_aa]["mutations"]:
                    effect = info[ref_aa]["effect"]
                    effect_scores = {"high": 1.0, "moderate": 0.5, "low": 0.2}
                    score += effect_scores.get(effect, 0.3)

                    drugs = info[ref_aa].get("drugs", [])
                    drugs_affected.update(drugs)

                    mutations.append(
                        {
                            "position": pos,
                            "ref": ref_aa,
                            "alt": seq_aa,
                            "effect": effect,
                            "drugs": drugs,
                            "notation": f"{ref_aa}{pos}{seq_aa}",
                        }
                    )

            # Normalize
            max_score = 5.0
            normalized = min(score / max_score, 1.0)

            results["scores"].append(normalized)
            results["mutations"].append(mutations)
            results["affected_drugs"].append(list(drugs_affected))

            # Classification
            if normalized < 0.1:
                classification = "susceptible"
            elif normalized < 0.3:
                classification = "possible_resistance"
            else:
                classification = "resistant"

            results["classifications"].append(classification)

        return results

    def _summarize_ras(self, drug_resistance: dict) -> dict[str, Any]:
        """Summarize RAS findings across all genes."""
        summary = {
            "total_ras": 0,
            "high_impact_ras": 0,
            "drugs_with_resistance": set(),
            "treatment_considerations": [],
        }

        for gene_data in drug_resistance.values():
            for mutations in gene_data.get("mutations", []):
                for mut in mutations:
                    summary["total_ras"] += 1
                    if mut.get("effect") == "high":
                        summary["high_impact_ras"] += 1
                    for drug in mut.get("drugs", []):
                        summary["drugs_with_resistance"].add(drug)

        summary["drugs_with_resistance"] = list(summary["drugs_with_resistance"])

        # Treatment recommendations
        if "sofosbuvir" in summary["drugs_with_resistance"]:
            summary["treatment_considerations"].append("S282T detected - consider extended treatment duration")
        if summary["high_impact_ras"] > 2:
            summary["treatment_considerations"].append("Multiple high-impact RAS - consider pan-genotypic regimen")

        return summary

    def validate_predictions(
        self,
        predictions: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> dict[str, float]:
        """Validate predictions against phenotypic data."""
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


def create_hcv_synthetic_dataset(
    genotype: HCVGenotype = HCVGenotype.GT1A,
    gene: HCVGene = HCVGene.NS5A,
    min_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create synthetic HCV dataset for testing.

    Args:
        genotype: HCV genotype (GT1A, GT1B, GT3, etc.)
        gene: Target gene (NS5A, NS3, NS5B)
        min_samples: Minimum number of samples to generate

    Returns:
        (X, y, ids) tuple with at least min_samples samples
    """
    reference = "M" + "A" * 99  # Simplified reference

    if gene == HCVGene.NS5A:
        mutation_db = NS5A_RAS_GT1A if genotype == HCVGenotype.GT1A else NS5A_RAS_GT1B
    elif gene == HCVGene.NS3:
        mutation_db = NS3_RAS_GT1A if genotype == HCVGenotype.GT1A else NS3_RAS_GT1B
    else:
        mutation_db = NS5B_RAS

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

    analyzer = HCVAnalyzer()
    X = np.array([analyzer.encode_sequence(s) for s in sequences])
    y = np.array(resistances, dtype=np.float32)

    # Ensure minimum samples via augmentation
    X, y, ids = ensure_minimum_samples(X, y, ids, min_samples=min_samples)

    return X, y, ids


# Helper method for encoding
HCVAnalyzer.encode_sequence = lambda self, seq, max_len=100: np.array(
    [
        1.0 if i == self.aa_to_idx.get(aa.upper(), self.aa_to_idx["X"]) else 0.0
        for j, aa in enumerate(seq[:max_len])
        for i in range(len(self.aa_alphabet))
    ]
    + [0.0] * (max_len - len(seq)) * len(self.aa_alphabet),
    dtype=np.float32,
)[: max_len * len(self.aa_alphabet)]
