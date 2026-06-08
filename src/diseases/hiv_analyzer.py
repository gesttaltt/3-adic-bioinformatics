# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""HIV Drug Resistance Analyzer.

This module provides analysis of HIV drug resistance mutations for
antiretroviral therapy (ART) based on the Stanford HIVDB database.

Based on:
- Stanford HIVDB (https://hivdb.stanford.edu/)
- IAS-USA Resistance Mutation List
- WHO HIV Drug Resistance Guidelines

Key Features:
- Multi-gene analysis (RT, PR, IN)
- 23+ drug predictions across 4 drug classes
- Cross-resistance pattern detection
- Mutation penalty scoring (HIVDB algorithm)

Drug Classes:
- NRTI: Nucleoside RT Inhibitors (7 drugs)
- NNRTI: Non-nucleoside RT Inhibitors (5 drugs)
- PI: Protease Inhibitors (8 drugs)
- INSTI: Integrase Strand Transfer Inhibitors (5 drugs)

Usage:
    from src.diseases.hiv_analyzer import HIVAnalyzer, create_hiv_synthetic_dataset

    analyzer = HIVAnalyzer()
    results = analyzer.analyze(sequences, gene="RT")

    # Or create synthetic dataset for validation
    X, y, ids = create_hiv_synthetic_dataset()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

from .base import DiseaseAnalyzer, DiseaseConfig, DiseaseType, TaskType


class HIVGene(Enum):
    """HIV genes relevant for drug resistance."""

    RT = "RT"  # Reverse Transcriptase (NRTI, NNRTI targets)
    PR = "PR"  # Protease (PI targets)
    IN = "IN"  # Integrase (INSTI targets)


class HIVDrugClass(Enum):
    """HIV drug classes."""

    NRTI = "NRTI"  # Nucleoside RT Inhibitors
    NNRTI = "NNRTI"  # Non-nucleoside RT Inhibitors
    PI = "PI"  # Protease Inhibitors
    INSTI = "INSTI"  # Integrase Strand Transfer Inhibitors


class HIVDrug(Enum):
    """HIV antiretroviral drugs (23 drugs across 4 classes)."""

    # NRTI (7 drugs)
    ABC = "ABC"  # Abacavir
    AZT = "AZT"  # Zidovudine
    D4T = "D4T"  # Stavudine
    DDI = "DDI"  # Didanosine
    FTC = "FTC"  # Emtricitabine
    LAM = "3TC"  # Lamivudine (3TC)
    TDF = "TDF"  # Tenofovir DF

    # NNRTI (5 drugs)
    DOR = "DOR"  # Doravirine
    EFV = "EFV"  # Efavirenz
    ETR = "ETR"  # Etravirine
    NVP = "NVP"  # Nevirapine
    RPV = "RPV"  # Rilpivirine

    # PI (8 drugs)
    ATV = "ATV"  # Atazanavir
    DRV = "DRV"  # Darunavir
    FPV = "FPV"  # Fosamprenavir
    IDV = "IDV"  # Indinavir
    LPV = "LPV"  # Lopinavir
    NFV = "NFV"  # Nelfinavir
    SQV = "SQV"  # Saquinavir
    TPV = "TPV"  # Tipranavir

    # INSTI (5 drugs)
    BIC = "BIC"  # Bictegravir
    CAB = "CAB"  # Cabotegravir
    DTG = "DTG"  # Dolutegravir
    EVG = "EVG"  # Elvitegravir
    RAL = "RAL"  # Raltegravir


# Drug to class mapping
DRUG_TO_CLASS = {
    HIVDrug.ABC: HIVDrugClass.NRTI,
    HIVDrug.AZT: HIVDrugClass.NRTI,
    HIVDrug.D4T: HIVDrugClass.NRTI,
    HIVDrug.DDI: HIVDrugClass.NRTI,
    HIVDrug.FTC: HIVDrugClass.NRTI,
    HIVDrug.LAM: HIVDrugClass.NRTI,
    HIVDrug.TDF: HIVDrugClass.NRTI,
    HIVDrug.DOR: HIVDrugClass.NNRTI,
    HIVDrug.EFV: HIVDrugClass.NNRTI,
    HIVDrug.ETR: HIVDrugClass.NNRTI,
    HIVDrug.NVP: HIVDrugClass.NNRTI,
    HIVDrug.RPV: HIVDrugClass.NNRTI,
    HIVDrug.ATV: HIVDrugClass.PI,
    HIVDrug.DRV: HIVDrugClass.PI,
    HIVDrug.FPV: HIVDrugClass.PI,
    HIVDrug.IDV: HIVDrugClass.PI,
    HIVDrug.LPV: HIVDrugClass.PI,
    HIVDrug.NFV: HIVDrugClass.PI,
    HIVDrug.SQV: HIVDrugClass.PI,
    HIVDrug.TPV: HIVDrugClass.PI,
    HIVDrug.BIC: HIVDrugClass.INSTI,
    HIVDrug.CAB: HIVDrugClass.INSTI,
    HIVDrug.DTG: HIVDrugClass.INSTI,
    HIVDrug.EVG: HIVDrugClass.INSTI,
    HIVDrug.RAL: HIVDrugClass.INSTI,
}

# Gene to drug classes mapping
GENE_TO_DRUG_CLASSES = {
    HIVGene.RT: [HIVDrugClass.NRTI, HIVDrugClass.NNRTI],
    HIVGene.PR: [HIVDrugClass.PI],
    HIVGene.IN: [HIVDrugClass.INSTI],
}


@dataclass
class HIVConfig(DiseaseConfig):
    """Configuration for HIV analysis."""

    name: str = "hiv"
    display_name: str = "Human Immunodeficiency Virus"
    disease_type: DiseaseType = DiseaseType.VIRAL
    tasks: list[TaskType] = field(
        default_factory=lambda: [
            TaskType.RESISTANCE,
            TaskType.ESCAPE,
        ]
    )

    data_sources: dict[str, str] = field(
        default_factory=lambda: {
            "hivdb": "https://hivdb.stanford.edu/",
            "lanl": "https://www.hiv.lanl.gov/",
            "who_hiv": "https://www.who.int/teams/global-hiv-hepatitis-and-stis-programmes",
        }
    )


# Reference sequences (HXB2 strain)
REFERENCE_SEQUENCES = {
    HIVGene.RT: "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDKEFRKYTAFTIPSTNNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGLTTPDKKHQKEPPFLWMGYELHPDKWTVQPIMLPEKDSWTVNDIQKLVGKLNWASQIYPGIKVRQLCKLLRGAKALTEVIPLTEEAELELAENREILKEPVHGVYYDPSKDLIAEIQKQGQGQWTYQIYQEPFKNLKTGKYARTRGAHTNDVKQLTEAVQKIATESIVIWGKTPKFRLPIQKETWETWWTEYWQATWIPEWEFVNTPPLVKLWYQLEKEPIIGAETFYVDGAANRETKLGKAGYVTDRGRQKVVPLTDTTNQKTELQAIHLALQDSGLEVNIVTDSQYALGIIQAQPDRSESEVVNQIIEE",
    HIVGene.PR: "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF",
    HIVGene.IN: "FLDGIDKAQDEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGKVILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTVHTDNGSNFTSTTVKAACWWAGIKQEFGIPYNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGIGGYSAGERIVDIIATDIQTKELQKQITKIQNFRVYYRDSRDPLWKGPAKLLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASRQDED",
}


# RT Mutations (NRTI and NNRTI resistance)
# Format: position -> {ref_aa: {mutations: [...], effect: level, drugs: [...]}}
RT_MUTATIONS = {
    # NRTI Resistance Mutations
    # M184V/I - Major 3TC/FTC resistance
    184: {"M": {"mutations": ["V", "I"], "effect": "high", "drugs": ["3TC", "FTC", "ABC"]}},
    # K65R - TDF resistance
    65: {"K": {"mutations": ["R", "E", "N"], "effect": "high", "drugs": ["TDF", "ABC", "DDI"]}},
    # K70R/E - TAM pathway
    70: {"K": {"mutations": ["R", "E"], "effect": "moderate", "drugs": ["AZT", "D4T", "TDF"]}},
    # L74V - ABC/DDI resistance
    74: {"L": {"mutations": ["V", "I"], "effect": "moderate", "drugs": ["ABC", "DDI"]}},
    # Y115F - ABC resistance
    115: {"Y": {"mutations": ["F"], "effect": "moderate", "drugs": ["ABC"]}},
    # Q151M complex
    151: {"Q": {"mutations": ["M"], "effect": "high", "drugs": ["AZT", "D4T", "ABC", "DDI", "TDF"]}},
    # TAMs (Thymidine Analog Mutations)
    41: {"M": {"mutations": ["L"], "effect": "moderate", "drugs": ["AZT", "D4T"]}},
    67: {"D": {"mutations": ["N", "G", "E"], "effect": "moderate", "drugs": ["AZT", "D4T"]}},
    210: {"L": {"mutations": ["W"], "effect": "moderate", "drugs": ["AZT", "D4T"]}},
    215: {"T": {"mutations": ["Y", "F"], "effect": "high", "drugs": ["AZT", "D4T"]}},
    219: {"K": {"mutations": ["Q", "E", "N", "R"], "effect": "moderate", "drugs": ["AZT", "D4T"]}},
    # NNRTI Resistance Mutations
    # K103N/S - Major EFV/NVP resistance
    103: {"K": {"mutations": ["N", "S"], "effect": "high", "drugs": ["EFV", "NVP", "DOR"]}},
    # Y181C/I/V - NVP resistance
    181: {"Y": {"mutations": ["C", "I", "V"], "effect": "high", "drugs": ["NVP", "ETR", "RPV"]}},
    # Y188L/C - NNRTI resistance
    188: {"Y": {"mutations": ["L", "C", "H"], "effect": "high", "drugs": ["EFV", "NVP"]}},
    # G190A/S/E - NNRTI resistance
    190: {"G": {"mutations": ["A", "S", "E"], "effect": "high", "drugs": ["NVP", "EFV"]}},
    # E138K - RPV resistance
    138: {"E": {"mutations": ["K", "A", "G", "Q"], "effect": "high", "drugs": ["RPV", "ETR"]}},
    # V106A/M - NNRTI resistance
    106: {"V": {"mutations": ["A", "M"], "effect": "moderate", "drugs": ["NVP", "EFV", "DOR"]}},
    # K101E/P - NNRTI accessory
    101: {"K": {"mutations": ["E", "P", "H"], "effect": "moderate", "drugs": ["NVP", "EFV", "RPV"]}},
    # P225H - ETR resistance
    225: {"P": {"mutations": ["H"], "effect": "moderate", "drugs": ["ETR", "RPV"]}},
    # Additional NRTI mutations (IAS-USA 2023)
    # K65E - Alternative to K65R
    # Already covered above with K65R/E/N
    # T69 insertion complex (multi-NRTI resistance)
    69: {"T": {"mutations": ["D", "N", "S"], "effect": "high", "drugs": ["AZT", "D4T", "ABC", "DDI", "TDF"]}},
    # A62V - Accessory NRTI mutation
    62: {"A": {"mutations": ["V"], "effect": "low", "drugs": ["TDF", "DDI"]}},
    # V75T/M/A - D4T/DDI resistance
    75: {"V": {"mutations": ["T", "M", "A"], "effect": "moderate", "drugs": ["D4T", "DDI"]}},
    # F77L - TAM accessory
    77: {"F": {"mutations": ["L"], "effect": "low", "drugs": ["AZT", "D4T"]}},
    # F116Y - TAM accessory
    116: {"F": {"mutations": ["Y"], "effect": "low", "drugs": ["AZT", "D4T", "ABC"]}},
    # Additional NNRTI mutations
    # A98G - NNRTI accessory
    98: {"A": {"mutations": ["G", "S"], "effect": "low", "drugs": ["NVP", "EFV", "DOR"]}},
    # K100I - DOR resistance
    100: {"K": {"mutations": ["I"], "effect": "moderate", "drugs": ["DOR", "EFV"]}},
    # V108I - NNRTI accessory
    108: {"V": {"mutations": ["I"], "effect": "low", "drugs": ["NVP", "EFV"]}},
    # E138A/G/K/Q/R - Second-gen NNRTI (multiple already covered)
    # V179D/E/F/T - ETR/RPV accessory
    179: {"V": {"mutations": ["D", "E", "F", "T", "L"], "effect": "moderate", "drugs": ["ETR", "RPV", "DOR"]}},
    # Y318F - NNRTI resistance
    318: {"Y": {"mutations": ["F"], "effect": "moderate", "drugs": ["NVP", "DOR"]}},
    # H221Y - RPV/ETR resistance
    221: {"H": {"mutations": ["Y"], "effect": "moderate", "drugs": ["RPV", "ETR"]}},
    # F227L/C - ETR/RPV resistance
    227: {"F": {"mutations": ["L", "C"], "effect": "moderate", "drugs": ["ETR", "RPV"]}},
    # M230L - NNRTI resistance
    230: {"M": {"mutations": ["L", "I"], "effect": "high", "drugs": ["ETR", "RPV", "DOR"]}},
    # L234I - DOR-specific
    234: {"L": {"mutations": ["I"], "effect": "moderate", "drugs": ["DOR"]}},
}


# PR Mutations (PI resistance)
PR_MUTATIONS = {
    # Major PI resistance mutations
    # D30N - NFV
    30: {"D": {"mutations": ["N"], "effect": "high", "drugs": ["NFV"]}},
    # V32I - DRV
    32: {"V": {"mutations": ["I"], "effect": "moderate", "drugs": ["DRV", "FPV", "LPV"]}},
    # M46I/L - Multiple PIs
    46: {"M": {"mutations": ["I", "L"], "effect": "moderate", "drugs": ["ATV", "LPV", "NFV", "FPV"]}},
    # I47V/A - LPV/DRV
    47: {"I": {"mutations": ["V", "A"], "effect": "high", "drugs": ["LPV", "DRV", "FPV"]}},
    # G48V - SQV
    48: {"G": {"mutations": ["V", "M"], "effect": "high", "drugs": ["SQV", "ATV"]}},
    # I50V/L - DRV/ATV
    50: {"I": {"mutations": ["V", "L"], "effect": "high", "drugs": ["DRV", "ATV", "LPV"]}},
    # I54V/L/M - Multiple PIs
    54: {
        "I": {"mutations": ["V", "L", "M", "A", "T", "S"], "effect": "moderate", "drugs": ["DRV", "ATV", "LPV", "FPV"]}
    },
    # L76V - DRV/LPV
    76: {"L": {"mutations": ["V"], "effect": "moderate", "drugs": ["DRV", "LPV", "FPV"]}},
    # V82A/T/F/S - Multiple PIs
    82: {"V": {"mutations": ["A", "T", "F", "S", "L", "M"], "effect": "high", "drugs": ["IDV", "LPV", "ATV"]}},
    # I84V - Multiple PIs
    84: {
        "I": {
            "mutations": ["V", "A", "C"],
            "effect": "high",
            "drugs": ["DRV", "ATV", "LPV", "SQV", "IDV", "NFV", "FPV", "TPV"],
        }
    },
    # N88S - ATV/NFV
    88: {"N": {"mutations": ["S", "D"], "effect": "high", "drugs": ["ATV", "NFV"]}},
    # L90M - Multiple PIs
    90: {"L": {"mutations": ["M"], "effect": "high", "drugs": ["SQV", "NFV", "IDV", "ATV"]}},
    # Minor/accessory mutations
    10: {"L": {"mutations": ["I", "V", "F", "R"], "effect": "low", "drugs": ["DRV", "LPV", "ATV"]}},
    20: {"K": {"mutations": ["R", "I", "M", "T", "V"], "effect": "low", "drugs": ["LPV", "ATV"]}},
    24: {"L": {"mutations": ["I"], "effect": "low", "drugs": ["ATV", "IDV"]}},
    33: {"L": {"mutations": ["F", "I", "V"], "effect": "low", "drugs": ["DRV", "ATV", "LPV"]}},
    53: {"F": {"mutations": ["L", "Y"], "effect": "low", "drugs": ["ATV", "SQV"]}},
    71: {"A": {"mutations": ["V", "T", "I", "L"], "effect": "low", "drugs": ["LPV", "ATV"]}},
    73: {"G": {"mutations": ["S", "T", "C", "A"], "effect": "low", "drugs": ["ATV", "SQV", "IDV"]}},
}


# IN Mutations (INSTI resistance)
IN_MUTATIONS = {
    # Major INSTI resistance
    # T66I/K - EVG
    66: {"T": {"mutations": ["I", "K", "A"], "effect": "high", "drugs": ["EVG"]}},
    # E92Q/G - EVG/RAL
    92: {"E": {"mutations": ["Q", "G"], "effect": "high", "drugs": ["EVG", "RAL"]}},
    # G118R - DTG/BIC
    118: {"G": {"mutations": ["R"], "effect": "moderate", "drugs": ["RAL", "DTG", "BIC", "CAB"]}},
    # E138K/A/T - INSTI accessory
    138: {"E": {"mutations": ["K", "A", "T"], "effect": "low", "drugs": ["EVG", "RAL"]}},
    # G140S/A/C - RAL/EVG (with Q148)
    140: {"G": {"mutations": ["S", "A", "C"], "effect": "high", "drugs": ["RAL", "EVG", "DTG"]}},
    # Y143R/C/H - RAL
    143: {"Y": {"mutations": ["R", "C", "H"], "effect": "high", "drugs": ["RAL"]}},
    # S147G - DTG pathway
    147: {"S": {"mutations": ["G"], "effect": "moderate", "drugs": ["DTG", "BIC"]}},
    # Q148H/K/R - Major resistance
    148: {"Q": {"mutations": ["H", "K", "R"], "effect": "high", "drugs": ["RAL", "EVG", "DTG", "BIC", "CAB"]}},
    # N155H - RAL/EVG
    155: {"N": {"mutations": ["H", "S"], "effect": "high", "drugs": ["RAL", "EVG"]}},
    # R263K - DTG
    263: {"R": {"mutations": ["K"], "effect": "moderate", "drugs": ["DTG", "BIC"]}},
    # Accessory mutations
    74: {"L": {"mutations": ["M", "I"], "effect": "low", "drugs": ["EVG"]}},
    97: {"T": {"mutations": ["A"], "effect": "low", "drugs": ["RAL", "EVG"]}},
    121: {"F": {"mutations": ["Y"], "effect": "low", "drugs": ["EVG"]}},
    153: {"S": {"mutations": ["F", "Y"], "effect": "low", "drugs": ["DTG"]}},
    230: {"S": {"mutations": ["R", "N"], "effect": "low", "drugs": ["DTG", "RAL"]}},
}


# Combine all mutations by gene
GENE_MUTATIONS = {
    HIVGene.RT: RT_MUTATIONS,
    HIVGene.PR: PR_MUTATIONS,
    HIVGene.IN: IN_MUTATIONS,
}


class HIVAnalyzer(DiseaseAnalyzer):
    """Analyzer for HIV drug resistance.

    Provides:
    - Multi-gene resistance analysis (RT, PR, IN)
    - 23 drug predictions across 4 classes
    - HIVDB-compatible scoring
    - Cross-resistance patterns
    """

    def __init__(self, config: HIVConfig | None = None):
        """Initialize analyzer."""
        self.config = config or HIVConfig()
        super().__init__(self.config)

        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY-X*"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_alphabet)}

    def analyze(
        self,
        sequences: dict[HIVGene, list[str]],
        embeddings: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Analyze HIV sequences for drug resistance.

        Args:
            sequences: Dictionary mapping gene to sequences
            embeddings: Optional embeddings

        Returns:
            Analysis results including per-drug resistance scores
        """
        results = {
            "n_sequences": sum(len(s) for s in sequences.values()),
            "genes_analyzed": [g.value for g in sequences],
            "drug_resistance": {},
            "mutations_detected": {},
            "drug_class_summary": {},
        }

        # Analyze each gene
        for gene, seqs in sequences.items():
            gene_mutations = GENE_MUTATIONS.get(gene, {})
            if not gene_mutations:
                continue

            # Get drugs for this gene
            drug_classes = GENE_TO_DRUG_CLASSES.get(gene, [])
            drugs = [d for d, c in DRUG_TO_CLASS.items() if c in drug_classes]

            # Predict resistance for each drug
            for drug in drugs:
                drug_results = self._predict_drug_resistance(seqs, gene_mutations, drug.value)
                results["drug_resistance"][drug.value] = drug_results

            # Detect all mutations
            results["mutations_detected"][gene.value] = self._detect_mutations(seqs, gene_mutations)

        # Summarize by drug class
        for drug_class in HIVDrugClass:
            class_drugs = [d for d, c in DRUG_TO_CLASS.items() if c == drug_class]
            class_scores = []
            for drug in class_drugs:
                if drug.value in results["drug_resistance"]:
                    class_scores.extend(results["drug_resistance"][drug.value]["scores"])
            if class_scores:
                results["drug_class_summary"][drug_class.value] = {
                    "mean_score": float(np.mean(class_scores)),
                    "max_score": float(np.max(class_scores)),
                    "n_drugs": len(class_drugs),
                }

        return results

    def _predict_drug_resistance(
        self,
        sequences: list[str],
        mutation_db: dict,
        drug: str,
    ) -> dict[str, Any]:
        """Predict resistance for a specific drug."""
        results = {
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
                    drugs_affected = info[ref_aa]["drugs"]
                    # Check if this drug or its abbreviation is affected
                    if drug in drugs_affected or any(drug.startswith(d) for d in drugs_affected):
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

            # Normalize (max theoretical score ~5-6 for heavily resistant)
            max_score = 4.0
            normalized = min(score / max_score, 1.0)

            results["scores"].append(normalized)
            results["mutations"].append(mutations)

            # Classification based on HIVDB levels
            if normalized < 0.1:
                classification = "susceptible"
            elif normalized < 0.25:
                classification = "potential_low_level_resistance"
            elif normalized < 0.5:
                classification = "low_level_resistance"
            elif normalized < 0.75:
                classification = "intermediate_resistance"
            else:
                classification = "high_level_resistance"

            results["classifications"].append(classification)

        return results

    def _detect_mutations(
        self,
        sequences: list[str],
        mutation_db: dict,
    ) -> list[list[dict]]:
        """Detect all mutations in sequences."""
        all_mutations = []

        for seq in sequences:
            seq_mutations = []
            for pos, info in mutation_db.items():
                if pos <= 0 or pos > len(seq):
                    continue

                seq_aa = seq[pos - 1]
                ref_aa = list(info.keys())[0]

                if seq_aa != ref_aa:
                    seq_mutations.append(
                        {
                            "position": pos,
                            "ref": ref_aa,
                            "alt": seq_aa,
                            "is_known_drm": seq_aa in info[ref_aa]["mutations"],
                            "notation": f"{ref_aa}{pos}{seq_aa}",
                        }
                    )

            all_mutations.append(seq_mutations)

        return all_mutations

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
        max_length: int = 560,  # RT is longest at 560 AA
    ) -> np.ndarray:
        """One-hot encode sequence."""
        n_aa = len(self.aa_alphabet)
        encoding = np.zeros(max_length * n_aa, dtype=np.float32)

        for j, aa in enumerate(sequence[:max_length]):
            idx = self.aa_to_idx.get(aa.upper(), self.aa_to_idx["X"])
            encoding[j * n_aa + idx] = 1.0

        return encoding


def create_hiv_synthetic_dataset(
    gene: HIVGene = HIVGene.RT,
    drug_class: HIVDrugClass | None = None,
    min_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create synthetic HIV dataset for testing.

    Args:
        gene: Target gene (RT, PR, or IN)
        drug_class: Optional drug class filter
        min_samples: Minimum number of samples to generate

    Returns:
        (X, y, ids) tuple
    """
    from src.diseases.utils.synthetic_data import (
        create_mutation_based_dataset,
        ensure_minimum_samples,
    )

    reference = REFERENCE_SEQUENCES.get(gene, "")
    mutation_db = GENE_MUTATIONS.get(gene, {})

    if not reference or not mutation_db:
        return np.array([]), np.array([]), []

    # Build reference with correct wild-type amino acids at mutation positions
    ref_list = list(reference)
    for pos, info in mutation_db.items():
        if pos <= len(ref_list):
            ref_aa = list(info.keys())[0]
            ref_list[pos - 1] = ref_aa
    reference_corrected = "".join(ref_list)

    # Filter mutations by drug class if specified
    if drug_class:
        class_drugs = [d.value for d, c in DRUG_TO_CLASS.items() if c == drug_class]
        filtered_db = {}
        for pos, info in mutation_db.items():
            ref_aa = list(info.keys())[0]
            drugs = info[ref_aa]["drugs"]
            if any(d in class_drugs for d in drugs):
                filtered_db[pos] = info
        mutation_db = filtered_db if filtered_db else mutation_db

    analyzer = HIVAnalyzer()

    # Use utility to create dataset
    max_length = len(reference)
    X, y, ids = create_mutation_based_dataset(
        reference_sequence=reference_corrected,
        mutation_db=mutation_db,
        encode_fn=lambda s, ml: analyzer.encode_sequence(s, max_length=ml),
        max_length=max_length,
        n_random_mutants=40,
        seed=42,
    )

    # Ensure minimum samples
    X, y, ids = ensure_minimum_samples(X, y, ids, min_samples=min_samples, seed=42)

    return X, y, ids


def get_hiv_drugs_for_gene(gene: HIVGene) -> list[HIVDrug]:
    """Get list of drugs targeting a specific gene."""
    drug_classes = GENE_TO_DRUG_CLASSES.get(gene, [])
    return [d for d, c in DRUG_TO_CLASS.items() if c in drug_classes]


def get_all_hiv_drugs() -> list[HIVDrug]:
    """Get all HIV drugs."""
    return list(HIVDrug)


def get_hiv_drug_classes() -> list[HIVDrugClass]:
    """Get all drug classes."""
    return list(HIVDrugClass)
