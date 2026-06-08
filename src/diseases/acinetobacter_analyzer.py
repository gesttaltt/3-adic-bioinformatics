# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Acinetobacter baumannii Analyzer for drug resistance prediction.

A. baumannii is a major cause of hospital-acquired infections, often
displaying extensive drug resistance. Key resistance mechanisms:
- Carbapenemase production (OXA-23, OXA-24, OXA-58, OXA-51-like)
- Metallo-beta-lactamases (NDM, IMP, VIM)
- Aminoglycoside-modifying enzymes
- Efflux pumps (AdeABC, AdeIJK)
- Outer membrane protein loss

References:
- Peleg et al. (2008) - A. baumannii: emergence of a successful pathogen
- Dijkshoorn et al. (2007) - International clones of A. baumannii
- CDC (2023) - Antibiotic resistance threats report
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from src.diseases.base import DiseaseAnalyzer, DiseaseConfig, DiseaseType, TaskType


class ABClonalComplex(Enum):
    """Major A. baumannii clonal complexes."""

    IC1 = "IC1"  # International Clone 1
    IC2 = "IC2"  # International Clone 2 (most common)
    IC3 = "IC3"  # International Clone 3
    IC4 = "IC4"
    IC5 = "IC5"
    IC6 = "IC6"
    IC7 = "IC7"
    IC8 = "IC8"
    OTHER = "other"


class ABGene(Enum):
    """A. baumannii resistance and virulence genes."""

    # Intrinsic OXA-51-like (species marker)
    OXA51 = "blaOXA-51-like"

    # Acquired carbapenemases
    OXA23 = "blaOXA-23"  # Most common acquired carbapenemase
    OXA24 = "blaOXA-24"  # OXA-24/40 group
    OXA58 = "blaOXA-58"
    OXA143 = "blaOXA-143"

    # Metallo-beta-lactamases
    NDM = "blaNDM"
    IMP = "blaIMP"
    VIM = "blaVIM"

    # ESBLs
    TEM = "blaTEM"
    SHV = "blaSHV"
    CTX_M = "blaCTX-M"
    PER = "blaPER"
    VEB = "blaVEB"
    GES = "blaGES"
    ADC = "blaADC"  # Acinetobacter-derived cephalosporinase

    # Aminoglycoside resistance
    AAC3 = "aac(3)"  # Gentamicin
    AAC6 = "aac(6')"  # Amikacin
    APH3 = "aph(3')"
    ANT2 = "ant(2'')"
    ARMA = "armA"  # 16S rRNA methylase (pan-aminoglycoside)

    # Efflux pumps
    ADEA = "adeA"  # AdeABC efflux pump
    ADEB = "adeB"
    ADEC = "adeC"
    ADERS = "adeRS"  # Regulator
    ADEI = "adeI"  # AdeIJK efflux pump
    ADEJ = "adeJ"
    ADEK = "adeK"

    # Outer membrane
    CARA = "carA"  # CarO loss - carbapenem resistance
    OMPA = "ompA"

    # Colistin resistance
    PMRAB = "pmrAB"  # Two-component system
    LPXA = "lpxA"  # Lipid A biosynthesis
    LPXC = "lpxC"
    LPXD = "lpxD"

    # Virulence
    BFMA = "bfmA"  # Biofilm
    CSUAB = "csuA/B"  # Pili assembly


class ABDrug(Enum):
    """Antibiotics for A. baumannii treatment."""

    # Carbapenems
    MEROPENEM = "meropenem"
    IMIPENEM = "imipenem"
    DORIPENEM = "doripenem"

    # Aminoglycosides
    AMIKACIN = "amikacin"
    GENTAMICIN = "gentamicin"
    TOBRAMYCIN = "tobramycin"

    # Polymyxins
    COLISTIN = "colistin"
    POLYMYXIN_B = "polymyxin_b"

    # Tetracyclines
    MINOCYCLINE = "minocycline"
    TIGECYCLINE = "tigecycline"

    # Sulbactam combinations
    AMPICILLIN_SULBACTAM = "ampicillin_sulbactam"
    SULBACTAM = "sulbactam"

    # Novel agents
    CEFIDEROCOL = "cefiderocol"
    ERAVACYCLINE = "eravacycline"


# OXA-type carbapenemase markers
OXA_CARBAPENEMASES = {
    ABGene.OXA23: {"meropenem_fold": 8, "imipenem_fold": 16},
    ABGene.OXA24: {"meropenem_fold": 8, "imipenem_fold": 8},
    ABGene.OXA58: {"meropenem_fold": 4, "imipenem_fold": 8},
}

# MBL markers (higher resistance)
MBL_CARBAPENEMASES = {
    ABGene.NDM: {"meropenem_fold": 64, "imipenem_fold": 64},
    ABGene.IMP: {"meropenem_fold": 32, "imipenem_fold": 32},
    ABGene.VIM: {"meropenem_fold": 32, "imipenem_fold": 32},
}

# 16S rRNA methylase - pan-aminoglycoside resistance
ARMA_EFFECT = {
    "amikacin_fold": 256,
    "gentamicin_fold": 128,
    "tobramycin_fold": 128,
}

# Colistin resistance mutations
COLISTIN_MUTATIONS = {
    ABGene.PMRAB: {
        # PmrB mutations
        227: {"S": 1.0, "N": 4.0, "T": 2.0},  # Ser227
        232: {"T": 1.0, "I": 4.0, "A": 2.0},  # Thr232
    },
    ABGene.LPXA: {
        # Truncation/inactivation leads to colistin resistance
        # But also fitness cost
        1: {"M": 1.0, "*": 256.0},  # Truncation
    },
}

# Drug to gene mapping
DRUG_GENE_MAP = {
    ABDrug.MEROPENEM: ABGene.OXA23,
    ABDrug.IMIPENEM: ABGene.OXA23,
    ABDrug.AMIKACIN: ABGene.ARMA,
    ABDrug.GENTAMICIN: ABGene.AAC3,
    ABDrug.COLISTIN: ABGene.PMRAB,
}


@dataclass
class AcinetobacterConfig(DiseaseConfig):
    """Configuration for A. baumannii analysis."""

    name: str = "acinetobacter"
    display_name: str = "Acinetobacter baumannii"
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
            "ncbi_pathogen": "https://www.ncbi.nlm.nih.gov/pathogens/",
            "pubmlst": "https://pubmlst.org/organisms/acinetobacter-baumannii",
            "cdc_ar": "https://www.cdc.gov/drugresistance/",
        }
    )

    # A. baumannii-specific settings
    predict_carbapenem_resistance: bool = True
    detect_mbl: bool = True
    detect_colistin_resistance: bool = True
    classify_clonal_complex: bool = True

    # Sequence settings
    min_sequence_length: int = 50

    genes: list[str] = field(default_factory=lambda: [g.value for g in ABGene])


class AcinetobacterAnalyzer(DiseaseAnalyzer):
    """Analyzer for Acinetobacter baumannii drug resistance.

    Features:
    - Carbapenemase detection (OXA-type and MBLs)
    - Aminoglycoside resistance prediction
    - Colistin resistance detection
    - Efflux pump analysis
    - Clonal complex classification
    - Treatment option assessment
    """

    def __init__(self, config: AcinetobacterConfig | None = None):
        """Initialize A. baumannii analyzer.

        Args:
            config: A. baumannii-specific configuration
        """
        self.config = config or AcinetobacterConfig()
        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"

    def analyze(
        self,
        sequences: dict[ABGene, list[str]],
        clonal_complex: ABClonalComplex | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Analyze A. baumannii sequences for resistance.

        Args:
            sequences: Dictionary mapping gene to protein sequences
            clonal_complex: Known clonal complex
            **kwargs: Additional parameters

        Returns:
            Comprehensive analysis results
        """
        results = {
            "n_sequences": len(next(iter(sequences.values()), [])),
            "genes_analyzed": [g.value for g in sequences],
            "clonal_complex": clonal_complex.value if clonal_complex else None,
        }

        # Species confirmation (OXA-51-like)
        results["species_confirmed"] = ABGene.OXA51 in sequences

        # Carbapenemase detection
        if self.config.predict_carbapenem_resistance:
            results["carbapenemases"] = self._detect_carbapenemases(sequences)

        # Drug resistance prediction
        resistance_results = {}
        for drug in ABDrug:
            target_gene = DRUG_GENE_MAP.get(drug)
            if target_gene and target_gene in sequences:
                drug_results = self.predict_drug_resistance(
                    sequences[target_gene],
                    drug,
                    target_gene,
                )
                resistance_results[drug.value] = drug_results

        # Add carbapenem resistance based on detected genes
        if results.get("carbapenemases", {}).get("detected"):
            if "meropenem" not in resistance_results:
                resistance_results["meropenem"] = {
                    "scores": [0.9],
                    "classifications": ["resistant"],
                    "mutations": [],
                }
            if "imipenem" not in resistance_results:
                resistance_results["imipenem"] = {
                    "scores": [0.9],
                    "classifications": ["resistant"],
                    "mutations": [],
                }

        if resistance_results:
            results["drug_resistance"] = resistance_results

        # Efflux pump analysis
        results["efflux_status"] = self._analyze_efflux(sequences)

        # Colistin resistance
        if self.config.detect_colistin_resistance:
            results["colistin_resistance"] = self._detect_colistin_resistance(sequences)

        # Resistance classification
        results["resistance_profile"] = self._classify_resistance_profile(
            results.get("carbapenemases", {}),
            resistance_results,
            results.get("colistin_resistance", {}),
        )

        # Treatment options
        results["treatment_options"] = self._assess_treatment_options(results)

        return results

    def _detect_carbapenemases(
        self,
        sequences: dict[ABGene, list[str]],
    ) -> dict[str, Any]:
        """Detect carbapenemase genes.

        Args:
            sequences: Gene sequences

        Returns:
            Carbapenemase detection results
        """
        results = {
            "detected": [],
            "oxa_type": [],
            "mbl_type": [],
            "highest_resistance_level": "susceptible",
        }

        max_fold = 1

        # Check OXA-type carbapenemases
        for gene, effects in OXA_CARBAPENEMASES.items():
            if gene in sequences:
                results["detected"].append(gene.value)
                results["oxa_type"].append(gene.value)
                max_fold = max(max_fold, max(effects.values()))

        # Check MBLs (higher resistance)
        for gene, effects in MBL_CARBAPENEMASES.items():
            if gene in sequences:
                results["detected"].append(gene.value)
                results["mbl_type"].append(gene.value)
                max_fold = max(max_fold, max(effects.values()))

        # Classify resistance level
        if max_fold >= 32:
            results["highest_resistance_level"] = "high_level"
        elif max_fold >= 8:
            results["highest_resistance_level"] = "moderate"
        elif max_fold > 1:
            results["highest_resistance_level"] = "low_level"

        return results

    def predict_drug_resistance(
        self,
        sequences: list[str],
        drug: ABDrug,
        gene: ABGene,
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
        }

        # Get mutations database
        if drug == ABDrug.COLISTIN and gene in COLISTIN_MUTATIONS:
            mutations_db = COLISTIN_MUTATIONS[gene]
        elif gene == ABGene.ARMA:
            # armA presence = pan-aminoglycoside resistance
            for seq in sequences:
                results["scores"].append(0.95)
                results["classifications"].append("resistant")
                results["mutations"].append([{"gene": "armA", "effect": "pan-resistance"}])
            return results
        else:
            # Gene presence generally indicates resistance
            for seq in sequences:
                results["scores"].append(0.7)
                results["classifications"].append("resistant")
                results["mutations"].append([])
            return results

        for seq in sequences:
            resistance_score = 0.0
            detected_mutations = []

            for pos, aa_effects in mutations_db.items():
                if 0 < pos <= len(seq):
                    aa = seq[pos - 1]
                    if aa in aa_effects:
                        fold_change = aa_effects[aa]
                        if fold_change > 1.0:
                            resistance_score += np.log2(fold_change) / 10
                            detected_mutations.append(
                                {
                                    "position": pos,
                                    "amino_acid": aa,
                                    "fold_change": fold_change,
                                }
                            )

            results["scores"].append(min(resistance_score, 1.0))
            results["mutations"].append(detected_mutations)

            # Classification
            if resistance_score < 0.2:
                classification = "susceptible"
            elif resistance_score < 0.5:
                classification = "reduced_susceptibility"
            else:
                classification = "resistant"
            results["classifications"].append(classification)

        return results

    def _analyze_efflux(
        self,
        sequences: dict[ABGene, list[str]],
    ) -> dict[str, Any]:
        """Analyze efflux pump status.

        Args:
            sequences: Gene sequences

        Returns:
            Efflux pump analysis
        """
        results = {
            "ade_abc_present": False,
            "ade_ijk_present": False,
            "overexpression_markers": [],
            "impact": "none",
        }

        # AdeABC system
        ade_abc_genes = [ABGene.ADEA, ABGene.ADEB, ABGene.ADEC]
        ade_abc_count = sum(1 for g in ade_abc_genes if g in sequences)
        if ade_abc_count >= 2:
            results["ade_abc_present"] = True

        # AdeIJK system
        ade_ijk_genes = [ABGene.ADEI, ABGene.ADEJ, ABGene.ADEK]
        ade_ijk_count = sum(1 for g in ade_ijk_genes if g in sequences)
        if ade_ijk_count >= 2:
            results["ade_ijk_present"] = True

        # AdeRS regulator
        if ABGene.ADERS in sequences:
            results["overexpression_markers"].append("adeRS present")

        # Impact assessment
        if results["ade_abc_present"] and results["ade_ijk_present"]:
            results["impact"] = "high"
        elif results["ade_abc_present"] or results["ade_ijk_present"]:
            results["impact"] = "moderate"

        return results

    def _detect_colistin_resistance(
        self,
        sequences: dict[ABGene, list[str]],
    ) -> dict[str, Any]:
        """Detect colistin resistance markers.

        Args:
            sequences: Gene sequences

        Returns:
            Colistin resistance analysis
        """
        results = {
            "resistant": False,
            "mechanism": [],
            "mutations": [],
        }

        # Check PmrAB mutations
        if ABGene.PMRAB in sequences:
            mutations_db = COLISTIN_MUTATIONS.get(ABGene.PMRAB, {})
            for seq in sequences[ABGene.PMRAB]:
                for pos, aa_effects in mutations_db.items():
                    if 0 < pos <= len(seq):
                        aa = seq[pos - 1]
                        if aa in aa_effects and aa_effects[aa] > 1.0:
                            results["resistant"] = True
                            results["mechanism"].append("pmrAB mutation")
                            results["mutations"].append(f"position {pos}: {aa}")

        # Check LpxACD loss (colistin resistance via lipid A modification)
        lpx_genes = [ABGene.LPXA, ABGene.LPXC, ABGene.LPXD]
        lpx_absent = [g for g in lpx_genes if g not in sequences]
        if lpx_absent:
            results["mechanism"].append("lpx gene(s) absent/inactivated")
            # Note: lpx loss causes colistin resistance but major fitness cost

        return results

    def _classify_resistance_profile(
        self,
        carbapenemases: dict[str, Any],
        resistance_data: dict[str, Any],
        colistin_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Classify overall resistance profile.

        Args:
            carbapenemases: Carbapenemase detection results
            resistance_data: Drug resistance results
            colistin_data: Colistin resistance results

        Returns:
            Resistance profile classification
        """
        results = {
            "profile": "susceptible",
            "carbapenem_status": "susceptible",
            "colistin_status": "susceptible",
            "mdr": False,
            "xdr": False,
            "pdr": False,
        }

        # Carbapenem resistance
        if carbapenemases.get("detected"):
            results["carbapenem_status"] = "resistant"

        # Colistin resistance
        if colistin_data.get("resistant"):
            results["colistin_status"] = "resistant"

        # Count resistance classes
        resistant_classes = 0

        # Carbapenems
        if results["carbapenem_status"] == "resistant":
            resistant_classes += 1

        # Aminoglycosides
        if any(
            "amikacin" in d or "gentamicin" in d
            for d in resistance_data
            if resistance_data.get(d, {}).get("classifications", [""])[0] == "resistant"
        ):
            resistant_classes += 1

        # Colistin/Polymyxins
        if results["colistin_status"] == "resistant":
            resistant_classes += 1

        # Classification
        if results["carbapenem_status"] == "resistant" and results["colistin_status"] == "resistant":
            results["profile"] = "PDR"
            results["pdr"] = True
        elif results["carbapenem_status"] == "resistant" and resistant_classes >= 2:
            results["profile"] = "XDR"
            results["xdr"] = True
        elif resistant_classes >= 3:
            results["profile"] = "MDR"
            results["mdr"] = True
        elif results["carbapenem_status"] == "resistant":
            results["profile"] = "CRAB"  # Carbapenem-Resistant A. baumannii
        else:
            results["profile"] = "susceptible"

        return results

    def _assess_treatment_options(
        self,
        analysis_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Assess available treatment options.

        Args:
            analysis_results: Full analysis results

        Returns:
            Treatment options assessment
        """
        options = {
            "recommended": [],
            "alternative": [],
            "contraindicated": [],
            "notes": [],
            "combination_therapy": [],
        }

        resistance_profile = analysis_results.get("resistance_profile", {})
        carbapenem_resistant = resistance_profile.get("carbapenem_status") == "resistant"
        colistin_resistant = resistance_profile.get("colistin_status") == "resistant"

        # Carbapenem-susceptible
        if not carbapenem_resistant:
            options["recommended"].append("Meropenem 1g IV q8h")
            options["recommended"].append("Imipenem-cilastatin 500mg IV q6h")
        else:
            options["contraindicated"].append("Carbapenems (monotherapy)")
            options["notes"].append("Carbapenem-resistant - combination therapy recommended")

            # Carbapenem-resistant options
            if not colistin_resistant:
                options["recommended"].append("Colistin (polymyxin-based regimen)")
                options["combination_therapy"].append("Colistin + Meropenem (high-dose)")
                options["combination_therapy"].append("Colistin + Tigecycline")
                options["combination_therapy"].append("Colistin + Sulbactam")
            else:
                options["contraindicated"].append("Colistin (monotherapy)")
                options["notes"].append("PDR strain - limited options")
                options["alternative"].append("Cefiderocol (if available)")
                options["combination_therapy"].append("Triple therapy: Meropenem + Tigecycline + Aminoglycoside")

            # Alternative agents
            options["alternative"].append("Tigecycline (for non-bloodstream infections)")
            options["alternative"].append("High-dose Ampicillin-sulbactam")
            options["alternative"].append("Cefiderocol")

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

        # Validate carbapenemase detection
        if "carbapenemases" in predictions and "carbapenemases" in ground_truth:
            pred_genes = set(predictions["carbapenemases"].get("detected", []))
            true_genes = set(ground_truth["carbapenemases"])

            if pred_genes or true_genes:
                tp = len(pred_genes & true_genes)
                fp = len(pred_genes - true_genes)
                fn = len(true_genes - pred_genes)
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                metrics["carbapenemase_precision"] = precision
                metrics["carbapenemase_recall"] = recall

        return metrics


# Convenience export
__all__ = [
    "AcinetobacterAnalyzer",
    "AcinetobacterConfig",
    "ABClonalComplex",
    "ABGene",
    "ABDrug",
]
