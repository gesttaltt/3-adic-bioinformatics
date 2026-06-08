# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Neisseria gonorrhoeae Analyzer for drug resistance prediction.

N. gonorrhoeae (gonococcus) causes gonorrhea and is a major public health
concern due to emerging multi-drug resistance. Key resistance mechanisms:
- Cephalosporin resistance (penA, porB, mtrR)
- Azithromycin resistance (23S rRNA, mtrR)
- Fluoroquinolone resistance (gyrA, parC)
- Historical penicillin/tetracycline resistance

References:
- WHO (2021) - Global action plan on antimicrobial resistance
- Unemo & Shafer (2014) - Antibiotic resistance in N. gonorrhoeae
- Grad et al. (2016) - Genomic epidemiology of gonococcal resistance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from src.diseases.base import DiseaseAnalyzer, DiseaseConfig, DiseaseType, TaskType


class GCSequenceType(Enum):
    """Major N. gonorrhoeae sequence types (MLST)."""

    ST1901 = "ST1901"  # Common MDR lineage
    ST7363 = "ST7363"  # Ceftriaxone-resistant
    ST1407 = "ST1407"  # Common European clone
    ST9363 = "ST9363"  # XDR lineage
    OTHER = "other"


class GCGene(Enum):
    """N. gonorrhoeae resistance genes."""

    # PBP2 (penicillin-binding protein 2)
    PENA = "penA"  # Cephalosporin resistance

    # Outer membrane proteins
    PORB = "porB"  # Porin - affects drug uptake

    # Efflux pump regulation
    MTRR = "mtrR"  # Repressor of mtrCDE efflux pump
    MTRC = "mtrC"  # Efflux pump component
    MTRD = "mtrD"  # Efflux pump component

    # Ribosomal targets
    RRL_23S = "23S_rRNA"  # Azithromycin resistance (rrl gene)
    RPOB = "rpoB"  # RNA polymerase (rifampin)
    RPSJ = "rpsJ"  # Ribosomal protein S10 (tetracycline)

    # DNA gyrase/topoisomerase
    GYRA = "gyrA"  # Fluoroquinolone resistance
    PARC = "parC"  # Fluoroquinolone resistance

    # Plasmid-mediated
    BLATE = "blaTEM"  # Beta-lactamase
    TETO = "tetO"  # Tetracycline resistance


class GCDrug(Enum):
    """Antibiotics for gonorrhea treatment."""

    # First-line (dual therapy)
    CEFTRIAXONE = "ceftriaxone"
    CEFIXIME = "cefixime"
    AZITHROMYCIN = "azithromycin"

    # Alternatives
    GENTAMICIN = "gentamicin"
    SPECTINOMYCIN = "spectinomycin"
    ERTAPENEM = "ertapenem"

    # Historical/limited use
    CIPROFLOXACIN = "ciprofloxacin"
    PENICILLIN = "penicillin"
    TETRACYCLINE = "tetracycline"


# PenA (PBP2) mutations for cephalosporin resistance
PENA_MUTATIONS = {
    GCDrug.CEFTRIAXONE: {
        310: {"A": 1.0, "V": 2.0, "T": 4.0},  # A310V - early warning
        311: {"I": 1.0, "V": 2.0},
        316: {"I": 1.0, "P": 4.0},
        501: {"A": 1.0, "V": 8.0, "T": 4.0},  # A501V - key mutation
        517: {"G": 1.0, "S": 2.0},
        543: {"A": 1.0, "G": 4.0},  # A543G - mosaic penA
        545: {"I": 1.0, "M": 4.0},  # I545M - mosaic penA
        551: {"H": 1.0, "N": 4.0},  # H551N
    },
    GCDrug.CEFIXIME: {
        310: {"A": 1.0, "V": 4.0, "T": 8.0},
        501: {"A": 1.0, "V": 16.0, "T": 8.0},
        545: {"I": 1.0, "M": 8.0},
    },
}

# PorB mutations affecting drug uptake
PORB_MUTATIONS = {
    120: {"G": 1.0, "K": 2.0, "D": 2.0},  # Loop 3 mutations
    121: {"A": 1.0, "D": 2.0, "S": 1.5},
}

# MtrR mutations for efflux pump overexpression
MTRR_MUTATIONS = {
    # Promoter mutations (-35, -10)
    -35: {"del": 4.0},  # Single nucleotide deletion
    # Coding mutations
    39: {"G": 1.0, "D": 2.0},  # G39D
    40: {"A": 1.0, "D": 3.0, "T": 2.0},  # A40D/T
    45: {"H": 1.0, "Y": 2.0},
}

# 23S rRNA mutations for azithromycin resistance
RRL_23S_MUTATIONS = {
    GCDrug.AZITHROMYCIN: {
        2059: {"A": 1.0, "G": 256.0},  # A2059G - high-level resistance
        2611: {"C": 1.0, "T": 64.0},  # C2611T - high-level resistance
    },
}

# GyrA mutations for fluoroquinolone resistance
GYRA_MUTATIONS = {
    GCDrug.CIPROFLOXACIN: {
        91: {"S": 1.0, "F": 32.0, "Y": 16.0},  # S91F - key mutation
        95: {"D": 1.0, "N": 8.0, "G": 16.0, "A": 8.0},  # D95N/G/A
    },
}

# ParC mutations for fluoroquinolone resistance
PARC_MUTATIONS = {
    GCDrug.CIPROFLOXACIN: {
        87: {"D": 1.0, "N": 4.0, "G": 4.0},  # D87N
        91: {"S": 1.0, "N": 2.0},  # S91N
    },
}

# Drug to gene mapping
DRUG_GENE_MAP = {
    GCDrug.CEFTRIAXONE: GCGene.PENA,
    GCDrug.CEFIXIME: GCGene.PENA,
    GCDrug.AZITHROMYCIN: GCGene.RRL_23S,
    GCDrug.CIPROFLOXACIN: GCGene.GYRA,
    GCDrug.PENICILLIN: GCGene.PENA,
    GCDrug.TETRACYCLINE: GCGene.RPSJ,
}


@dataclass
class GonorrhoeaeConfig(DiseaseConfig):
    """Configuration for N. gonorrhoeae analysis."""

    name: str = "gonorrhoeae"
    display_name: str = "Neisseria gonorrhoeae (Gonorrhea)"
    disease_type: DiseaseType = DiseaseType.BACTERIAL
    tasks: list[TaskType] = field(
        default_factory=lambda: [
            TaskType.RESISTANCE,
            TaskType.FITNESS,
        ]
    )

    # Data sources
    data_sources: dict[str, str] = field(
        default_factory=lambda: {
            "pubmlst": "https://pubmlst.org/organisms/neisseria-spp",
            "pathogenwatch": "https://pathogen.watch/genomes/all?genusId=482",
            "ncbi_pathogen": "https://www.ncbi.nlm.nih.gov/pathogens/",
        }
    )

    # N. gonorrhoeae-specific settings
    predict_mdr: bool = True
    predict_xdr: bool = True
    assess_treatment_options: bool = True

    # Sequence settings
    min_sequence_length: int = 50

    genes: list[str] = field(default_factory=lambda: [g.value for g in GCGene])


class GonorrhoeaeAnalyzer(DiseaseAnalyzer):
    """Analyzer for N. gonorrhoeae drug resistance.

    Features:
    - Cephalosporin resistance prediction (penA)
    - Azithromycin resistance prediction (23S rRNA)
    - Fluoroquinolone resistance prediction (gyrA, parC)
    - Multi-drug resistance (MDR) classification
    - Extensively drug-resistant (XDR) classification
    - Treatment option assessment
    """

    def __init__(self, config: GonorrhoeaeConfig | None = None):
        """Initialize N. gonorrhoeae analyzer.

        Args:
            config: Gonorrhoeae-specific configuration
        """
        self.config = config or GonorrhoeaeConfig()
        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"

    def analyze(
        self,
        sequences: dict[GCGene, list[str]],
        sequence_type: GCSequenceType | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Analyze N. gonorrhoeae sequences for resistance.

        Args:
            sequences: Dictionary mapping gene to protein sequences
            sequence_type: Known MLST sequence type
            **kwargs: Additional parameters

        Returns:
            Comprehensive analysis results
        """
        results = {
            "n_sequences": len(next(iter(sequences.values()), [])),
            "genes_analyzed": [g.value for g in sequences],
            "sequence_type": sequence_type.value if sequence_type else None,
        }

        # Drug resistance prediction
        resistance_results = {}
        for drug in GCDrug:
            target_gene = DRUG_GENE_MAP.get(drug)
            if target_gene and target_gene in sequences:
                drug_results = self.predict_drug_resistance(
                    sequences[target_gene],
                    drug,
                    target_gene,
                )
                resistance_results[drug.value] = drug_results

        if resistance_results:
            results["drug_resistance"] = resistance_results

        # Efflux pump analysis
        if GCGene.MTRR in sequences:
            results["efflux_status"] = self._analyze_efflux(sequences[GCGene.MTRR])

        # MDR/XDR classification
        if self.config.predict_mdr:
            results["mdr_classification"] = self._classify_mdr(resistance_results)

        # Treatment options
        if self.config.assess_treatment_options:
            results["treatment_options"] = self._assess_treatment_options(resistance_results)

        return results

    def predict_drug_resistance(
        self,
        sequences: list[str],
        drug: GCDrug,
        gene: GCGene,
    ) -> dict[str, Any]:
        """Predict drug resistance for a specific drug.

        Args:
            sequences: Protein sequences for target gene
            drug: Drug to predict resistance for
            gene: Target gene

        Returns:
            Drug resistance predictions
        """
        results = {
            "scores": [],
            "classifications": [],
            "mutations": [],
            "mic_predictions": [],
        }

        # Get mutations database
        if gene == GCGene.PENA and drug in PENA_MUTATIONS:
            mutations_db = PENA_MUTATIONS[drug]
        elif gene == GCGene.RRL_23S and drug in RRL_23S_MUTATIONS:
            mutations_db = RRL_23S_MUTATIONS[drug]
        elif gene == GCGene.GYRA and drug in GYRA_MUTATIONS:
            mutations_db = GYRA_MUTATIONS[drug]
        elif gene == GCGene.PARC and drug in PARC_MUTATIONS:
            mutations_db = PARC_MUTATIONS[drug]
        else:
            for seq in sequences:
                results["scores"].append(0.1)
                results["classifications"].append("susceptible")
                results["mutations"].append([])
                results["mic_predictions"].append("<=0.016")
            return results

        for seq in sequences:
            resistance_score = 0.0
            detected_mutations = []
            fold_changes = []

            for pos, aa_effects in mutations_db.items():
                if 0 < pos <= len(seq):
                    aa = seq[pos - 1]
                    if aa in aa_effects:
                        fold_change = aa_effects[aa]
                        if fold_change > 1.0:
                            resistance_score += np.log2(fold_change) / 10
                            fold_changes.append(fold_change)
                            detected_mutations.append(
                                {
                                    "position": pos,
                                    "amino_acid": aa,
                                    "fold_change": fold_change,
                                }
                            )

            results["scores"].append(min(resistance_score, 1.0))
            results["mutations"].append(detected_mutations)

            # MIC prediction based on fold changes
            max_fold = max(fold_changes) if fold_changes else 1.0
            results["mic_predictions"].append(self._predict_mic(drug, max_fold))

            # Classification based on EUCAST breakpoints
            if drug == GCDrug.CEFTRIAXONE:
                if resistance_score >= 0.5:
                    classification = "resistant"
                elif resistance_score >= 0.2:
                    classification = "reduced_susceptibility"
                else:
                    classification = "susceptible"
            else:
                if resistance_score < 0.2:
                    classification = "susceptible"
                elif resistance_score < 0.5:
                    classification = "reduced_susceptibility"
                else:
                    classification = "resistant"

            results["classifications"].append(classification)

        return results

    def _predict_mic(
        self,
        drug: GCDrug,
        fold_change: float,
    ) -> str:
        """Predict MIC category based on fold change.

        Args:
            drug: Target drug
            fold_change: Resistance fold change

        Returns:
            MIC category string
        """
        if drug == GCDrug.CEFTRIAXONE:
            if fold_change >= 8:
                return ">=0.25"
            elif fold_change >= 4:
                return "0.125-0.25"
            elif fold_change >= 2:
                return "0.06-0.125"
            else:
                return "<=0.016"
        elif drug == GCDrug.AZITHROMYCIN:
            if fold_change >= 64:
                return ">=4"
            elif fold_change >= 8:
                return "1-2"
            else:
                return "<=0.5"
        else:
            if fold_change >= 16:
                return "high"
            elif fold_change >= 4:
                return "intermediate"
            else:
                return "low"

    def _analyze_efflux(
        self,
        mtrr_sequences: list[str],
    ) -> dict[str, Any]:
        """Analyze MtrR mutations for efflux pump status.

        Args:
            mtrr_sequences: MtrR repressor sequences

        Returns:
            Efflux pump analysis
        """
        results = {
            "overexpression_risk": [],
            "mutations_detected": [],
        }

        for seq in mtrr_sequences:
            mutations = []
            risk_score = 0.0

            for pos, aa_effects in MTRR_MUTATIONS.items():
                if pos > 0 and 0 < pos <= len(seq):
                    aa = seq[pos - 1]
                    if aa in aa_effects:
                        fold_effect = aa_effects[aa]
                        if fold_effect > 1.0:
                            risk_score += np.log2(fold_effect) / 5
                            mutations.append(
                                {
                                    "position": pos,
                                    "amino_acid": aa,
                                    "effect": fold_effect,
                                }
                            )

            results["mutations_detected"].append(mutations)

            if risk_score >= 0.5:
                results["overexpression_risk"].append("high")
            elif risk_score >= 0.2:
                results["overexpression_risk"].append("moderate")
            else:
                results["overexpression_risk"].append("low")

        return results

    def _classify_mdr(
        self,
        resistance_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Classify multi-drug resistance status.

        MDR: Resistant to >=3 classes
        XDR: Resistant to all first-line options

        Args:
            resistance_data: Drug resistance results

        Returns:
            MDR/XDR classification
        """
        results = {
            "classification": "susceptible",
            "resistance_classes": [],
            "resistant_drugs": [],
        }

        # Drug classes
        cephalosporin_resistant = False
        macrolide_resistant = False
        fluoroquinolone_resistant = False
        penicillin_resistant = False
        tetracycline_resistant = False

        for drug, data in resistance_data.items():
            classifications = data.get("classifications", [])
            if any(c == "resistant" for c in classifications):
                results["resistant_drugs"].append(drug)

                if drug in ["ceftriaxone", "cefixime"]:
                    cephalosporin_resistant = True
                elif drug == "azithromycin":
                    macrolide_resistant = True
                elif drug == "ciprofloxacin":
                    fluoroquinolone_resistant = True
                elif drug == "penicillin":
                    penicillin_resistant = True
                elif drug == "tetracycline":
                    tetracycline_resistant = True

        # Count resistance classes
        if cephalosporin_resistant:
            results["resistance_classes"].append("cephalosporins")
        if macrolide_resistant:
            results["resistance_classes"].append("macrolides")
        if fluoroquinolone_resistant:
            results["resistance_classes"].append("fluoroquinolones")
        if penicillin_resistant:
            results["resistance_classes"].append("penicillins")
        if tetracycline_resistant:
            results["resistance_classes"].append("tetracyclines")

        n_classes = len(results["resistance_classes"])

        # Classification
        if cephalosporin_resistant and macrolide_resistant:
            results["classification"] = "XDR"  # No first-line options
        elif n_classes >= 3:
            results["classification"] = "MDR"
        elif n_classes >= 1:
            results["classification"] = "resistant"
        else:
            results["classification"] = "susceptible"

        return results

    def _assess_treatment_options(
        self,
        resistance_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Assess available treatment options.

        Args:
            resistance_data: Drug resistance results

        Returns:
            Treatment options assessment
        """
        options = {
            "recommended": [],
            "alternative": [],
            "contraindicated": [],
            "notes": [],
        }

        # Check cephalosporin susceptibility
        cro_data = resistance_data.get("ceftriaxone", {})
        cro_susceptible = all(c != "resistant" for c in cro_data.get("classifications", ["susceptible"]))

        # Check azithromycin susceptibility
        azm_data = resistance_data.get("azithromycin", {})
        azm_susceptible = all(c != "resistant" for c in azm_data.get("classifications", ["susceptible"]))

        # Check fluoroquinolone susceptibility
        cip_data = resistance_data.get("ciprofloxacin", {})
        cip_susceptible = all(c != "resistant" for c in cip_data.get("classifications", ["susceptible"]))

        # Standard dual therapy
        if cro_susceptible and azm_susceptible:
            options["recommended"].append("Ceftriaxone 500mg IM + Azithromycin 1g PO")
        elif cro_susceptible and not azm_susceptible:
            options["recommended"].append("Ceftriaxone 500mg IM (monotherapy)")
            options["notes"].append("Azithromycin resistance - monotherapy ceftriaxone")
            if cip_susceptible:
                options["alternative"].append("Add ciprofloxacin if needed")
        elif not cro_susceptible:
            options["notes"].append("Ceftriaxone resistance - refer to specialist")
            options["alternative"].append("Gentamicin 240mg IM + Azithromycin 2g PO")
            options["alternative"].append("Ertapenem 1g IM")

        # Contraindicated drugs
        if not cro_susceptible:
            options["contraindicated"].append("Ceftriaxone")
        if not azm_susceptible:
            options["contraindicated"].append("Azithromycin monotherapy")
        if not cip_susceptible:
            options["contraindicated"].append("Ciprofloxacin")

        return options

    def validate_predictions(
        self,
        predictions: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> dict[str, float]:
        """Validate predictions against phenotypic data.

        Args:
            predictions: Model predictions
            ground_truth: Known values from laboratory data

        Returns:
            Dictionary of validation metrics
        """
        from scipy.stats import spearmanr

        metrics = {}

        # Validate drug resistance predictions
        for drug in predictions.get("drug_resistance", {}):
            drug_name = drug if isinstance(drug, str) else drug.value

            if drug_name in ground_truth.get("drug_resistance", {}):
                pred_scores = predictions["drug_resistance"][drug]["scores"]
                true_scores = ground_truth["drug_resistance"][drug_name]

                if len(pred_scores) == len(true_scores) and len(pred_scores) > 1:
                    corr, p_value = spearmanr(pred_scores, true_scores)
                    metrics[f"{drug_name}_spearman"] = corr
                    metrics[f"{drug_name}_p_value"] = p_value

        # Validate MDR classification
        if "mdr_classification" in predictions and "mdr_status" in ground_truth:
            pred_mdr = predictions["mdr_classification"]["classification"]
            true_mdr = ground_truth["mdr_status"]

            metrics["mdr_correct"] = 1.0 if pred_mdr == true_mdr else 0.0

        return metrics


# Convenience export
__all__ = [
    "GonorrhoeaeAnalyzer",
    "GonorrhoeaeConfig",
    "GCSequenceType",
    "GCGene",
    "GCDrug",
]
