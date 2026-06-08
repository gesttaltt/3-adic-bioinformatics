# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Vancomycin-Resistant Enterococcus (VRE) Analyzer.

VRE is a major healthcare-associated infection concern. Key features:
- van gene clusters (vanA, vanB, vanC, vanD, vanE, vanG)
- Species identification (E. faecium, E. faecalis)
- Daptomycin and linezolid resistance markers
- High-level aminoglycoside resistance (HLAR)

References:
- Werner et al. (2008) - Antibiotic resistance in Enterococci
- Miller et al. (2014) - Development and impact of VRE
- CDC (2023) - Antibiotic resistance threats report
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from src.diseases.base import DiseaseAnalyzer, DiseaseConfig, DiseaseType, TaskType


class EnterococcusSpecies(Enum):
    """Enterococcus species."""

    E_FAECIUM = "E. faecium"  # Most commonly VRE
    E_FAECALIS = "E. faecalis"
    E_GALLINARUM = "E. gallinarum"  # Intrinsic vanC
    E_CASSELIFLAVUS = "E. casseliflavus"  # Intrinsic vanC
    OTHER = "other"


class VREGene(Enum):
    """VRE resistance and virulence genes."""

    # Vancomycin resistance gene clusters
    VANA = "vanA"  # High-level, inducible, transferable
    VANB = "vanB"  # Variable-level, inducible, transferable
    VANC = "vanC"  # Low-level, intrinsic (E. gallinarum/casseliflavus)
    VAND = "vanD"  # Moderate-level, constitutive
    VANE = "vanE"  # Low-level
    VANG = "vanG"  # Low-level

    # Van ligases (key determinants)
    VANH = "vanH"  # D-lac dehydrogenase
    VANX = "vanX"  # D-Ala-D-Ala dipeptidase
    VANY = "vanY"  # D,D-carboxypeptidase

    # Daptomycin resistance
    LIAFSR = "liaFSR"  # Cell envelope stress response
    CLS = "cls"  # Cardiolipin synthase
    GDPD = "gdpD"  # Glycerophosphodiester phosphodiesterase

    # Linezolid resistance
    OPTRA = "optrA"  # ABC transporter
    CFR = "cfr"  # rRNA methyltransferase
    RRL_23S = "23S_rRNA"  # Ribosomal mutations

    # High-level aminoglycoside resistance
    AAC6_APH2 = "aac(6')-Ie-aph(2'')-Ia"  # Gentamicin/streptomycin
    APH3 = "aph(3')"  # Kanamycin

    # Virulence factors
    ESP = "esp"  # Enterococcal surface protein
    GEL = "gelE"  # Gelatinase
    AS = "asa1"  # Aggregation substance


class VREDrug(Enum):
    """Antibiotics for VRE treatment."""

    # Glycopeptides
    VANCOMYCIN = "vancomycin"
    TEICOPLANIN = "teicoplanin"

    # Lipopeptides
    DAPTOMYCIN = "daptomycin"

    # Oxazolidinones
    LINEZOLID = "linezolid"
    TEDIZOLID = "tedizolid"

    # Aminoglycosides
    GENTAMICIN = "gentamicin"
    STREPTOMYCIN = "streptomycin"

    # Other
    TIGECYCLINE = "tigecycline"
    QUINUPRISTIN_DALFOPRISTIN = "quinupristin-dalfopristin"  # E. faecium only
    AMPICILLIN = "ampicillin"
    NITROFURANTOIN = "nitrofurantoin"  # UTI only


# VanA phenotype mutations
VANA_MARKERS = {
    # VanH - D-Lac dehydrogenase
    VREGene.VANH: {
        50: {"G": 1.0, "S": 0.8, "D": 0.5},  # Active site
    },
    # VanX - D-Ala-D-Ala dipeptidase
    VREGene.VANX: {
        71: {"D": 1.0, "N": 0.7, "E": 0.9},  # Metal binding
    },
}

# VanB phenotype markers
VANB_MARKERS = {
    VREGene.VANB: {
        # Variable MIC based on expression
        123: {"S": 1.0, "T": 0.8, "A": 0.6},
    },
}

# Daptomycin resistance mutations
DAPTOMYCIN_MUTATIONS = {
    VREGene.LIAFSR: {
        # LiaR response regulator
        20: {"W": 1.0, "G": 4.0, "C": 3.0},  # W20G - key mutation
        73: {"T": 1.0, "I": 2.0, "A": 1.5},
    },
    VREGene.CLS: {
        # Cardiolipin synthase
        16: {"L": 1.0, "V": 2.0, "I": 1.5},
        115: {"I": 1.0, "T": 3.0},
    },
    VREGene.GDPD: {
        120: {"D": 1.0, "N": 2.0, "E": 1.5},
    },
}

# Linezolid resistance mutations
LINEZOLID_MUTATIONS = {
    VREGene.RRL_23S: {
        # Domain V mutations
        2576: {"G": 1.0, "T": 64.0, "U": 32.0},  # G2576T - high-level
    },
}

# Drug to gene mapping
DRUG_GENE_MAP = {
    VREDrug.VANCOMYCIN: VREGene.VANA,  # Primary check
    VREDrug.TEICOPLANIN: VREGene.VANA,
    VREDrug.DAPTOMYCIN: VREGene.LIAFSR,
    VREDrug.LINEZOLID: VREGene.RRL_23S,
    VREDrug.GENTAMICIN: VREGene.AAC6_APH2,
}


@dataclass
class VREConfig(DiseaseConfig):
    """Configuration for VRE analysis."""

    name: str = "vre"
    display_name: str = "Vancomycin-Resistant Enterococcus"
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
            "pubmlst": "https://pubmlst.org/organisms/enterococcus-faecium",
            "cdc_ar": "https://www.cdc.gov/drugresistance/",
        }
    )

    # VRE-specific settings
    classify_species: bool = True
    detect_virulence: bool = True
    check_hlar: bool = True  # High-level aminoglycoside resistance

    # Sequence settings
    min_sequence_length: int = 50

    genes: list[str] = field(default_factory=lambda: [g.value for g in VREGene])


class VREAnalyzer(DiseaseAnalyzer):
    """Analyzer for Vancomycin-Resistant Enterococcus.

    Features:
    - Van genotype classification (vanA, vanB, vanC)
    - Daptomycin resistance prediction
    - Linezolid resistance prediction
    - High-level aminoglycoside resistance (HLAR)
    - Species identification
    - Virulence factor detection
    - Treatment option assessment
    """

    def __init__(self, config: VREConfig | None = None):
        """Initialize VRE analyzer.

        Args:
            config: VRE-specific configuration
        """
        self.config = config or VREConfig()
        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"

    def analyze(
        self,
        sequences: dict[VREGene, list[str]],
        species: EnterococcusSpecies | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Analyze VRE sequences for resistance and virulence.

        Args:
            sequences: Dictionary mapping gene to protein sequences
            species: Known Enterococcus species
            **kwargs: Additional parameters

        Returns:
            Comprehensive analysis results
        """
        results = {
            "n_sequences": len(next(iter(sequences.values()), [])),
            "genes_analyzed": [g.value for g in sequences],
            "species": species.value if species else None,
        }

        # Van genotype determination
        results["van_genotype"] = self._determine_van_genotype(sequences)

        # Drug resistance prediction
        resistance_results = {}
        for drug in VREDrug:
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

        # HLAR detection
        if self.config.check_hlar:
            results["hlar_status"] = self._detect_hlar(sequences)

        # Virulence factors
        if self.config.detect_virulence:
            results["virulence_factors"] = self._detect_virulence(sequences)

        # Treatment options
        results["treatment_options"] = self._assess_treatment_options(
            results.get("van_genotype", {}),
            resistance_results,
            species,
        )

        return results

    def _determine_van_genotype(
        self,
        sequences: dict[VREGene, list[str]],
    ) -> dict[str, Any]:
        """Determine van gene cluster genotype.

        Args:
            sequences: Gene sequences

        Returns:
            Van genotype information
        """
        results = {
            "genotype": "susceptible",
            "genes_detected": [],
            "phenotype": "vancomycin_susceptible",
            "teicoplanin_status": "susceptible",
        }

        # Check for van genes
        if VREGene.VANA in sequences:
            results["genes_detected"].append("vanA")
            results["genotype"] = "vanA"
            results["phenotype"] = "high_level_resistance"
            results["teicoplanin_status"] = "resistant"
        elif VREGene.VANB in sequences:
            results["genes_detected"].append("vanB")
            results["genotype"] = "vanB"
            results["phenotype"] = "variable_resistance"
            results["teicoplanin_status"] = "susceptible"
        elif VREGene.VANC in sequences:
            results["genes_detected"].append("vanC")
            results["genotype"] = "vanC"
            results["phenotype"] = "low_level_resistance"
            results["teicoplanin_status"] = "susceptible"
        elif VREGene.VAND in sequences:
            results["genes_detected"].append("vanD")
            results["genotype"] = "vanD"
            results["phenotype"] = "moderate_resistance"
            results["teicoplanin_status"] = "susceptible"

        # Check supporting genes
        if VREGene.VANH in sequences:
            results["genes_detected"].append("vanH")
        if VREGene.VANX in sequences:
            results["genes_detected"].append("vanX")

        return results

    def predict_drug_resistance(
        self,
        sequences: list[str],
        drug: VREDrug,
        gene: VREGene,
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
        if drug == VREDrug.DAPTOMYCIN and gene in DAPTOMYCIN_MUTATIONS:
            mutations_db = DAPTOMYCIN_MUTATIONS[gene]
        elif drug == VREDrug.LINEZOLID and gene in LINEZOLID_MUTATIONS:
            mutations_db = LINEZOLID_MUTATIONS[gene]
        elif drug in [VREDrug.VANCOMYCIN, VREDrug.TEICOPLANIN]:
            # Van gene presence is the main determinant
            for seq in sequences:
                results["scores"].append(0.9 if len(seq) > 100 else 0.1)
                results["classifications"].append("resistant" if len(seq) > 100 else "susceptible")
                results["mutations"].append([])
            return results
        else:
            # Gene presence indicates resistance
            for seq in sequences:
                results["scores"].append(0.8)
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

    def _detect_hlar(
        self,
        sequences: dict[VREGene, list[str]],
    ) -> dict[str, Any]:
        """Detect high-level aminoglycoside resistance.

        Args:
            sequences: Gene sequences

        Returns:
            HLAR status
        """
        results = {
            "gentamicin_hlar": False,
            "streptomycin_hlar": False,
            "genes_detected": [],
            "synergy_expected": True,
        }

        # aac(6')-Ie-aph(2'')-Ia confers gentamicin HLAR
        if VREGene.AAC6_APH2 in sequences:
            results["gentamicin_hlar"] = True
            results["genes_detected"].append("aac(6')-Ie-aph(2'')-Ia")
            results["synergy_expected"] = False

        # aph(3') can affect aminoglycoside activity
        if VREGene.APH3 in sequences:
            results["genes_detected"].append("aph(3')")

        return results

    def _detect_virulence(
        self,
        sequences: dict[VREGene, list[str]],
    ) -> dict[str, Any]:
        """Detect virulence factors.

        Args:
            sequences: Gene sequences

        Returns:
            Virulence factor analysis
        """
        results = {
            "factors_detected": [],
            "virulence_score": 0.0,
            "biofilm_potential": "low",
        }

        score = 0.0

        if VREGene.ESP in sequences:
            results["factors_detected"].append("esp (enterococcal surface protein)")
            score += 0.3

        if VREGene.GEL in sequences:
            results["factors_detected"].append("gelE (gelatinase)")
            score += 0.2

        if VREGene.AS in sequences:
            results["factors_detected"].append("asa1 (aggregation substance)")
            score += 0.3

        results["virulence_score"] = min(score, 1.0)

        # Biofilm potential
        if score >= 0.5:
            results["biofilm_potential"] = "high"
        elif score >= 0.2:
            results["biofilm_potential"] = "moderate"

        return results

    def _assess_treatment_options(
        self,
        van_genotype: dict[str, Any],
        resistance_data: dict[str, Any],
        species: EnterococcusSpecies | None = None,
    ) -> dict[str, Any]:
        """Assess available treatment options.

        Args:
            van_genotype: Van genotype results
            resistance_data: Drug resistance results
            species: Enterococcus species

        Returns:
            Treatment options assessment
        """
        options = {
            "recommended": [],
            "alternative": [],
            "contraindicated": [],
            "notes": [],
        }

        genotype = van_genotype.get("genotype", "susceptible")

        # Check daptomycin
        dap_data = resistance_data.get("daptomycin", {})
        dap_susceptible = all(c != "resistant" for c in dap_data.get("classifications", ["susceptible"]))

        # Check linezolid
        lzd_data = resistance_data.get("linezolid", {})
        lzd_susceptible = all(c != "resistant" for c in lzd_data.get("classifications", ["susceptible"]))

        if genotype == "susceptible":
            options["recommended"].append("Ampicillin (if susceptible)")
            options["recommended"].append("Vancomycin")
        else:
            # VRE - vancomycin resistant
            options["contraindicated"].append("Vancomycin")

            if genotype == "vanA":
                options["contraindicated"].append("Teicoplanin")
                options["notes"].append("VanA phenotype - high-level glycopeptide resistance")

            if dap_susceptible:
                options["recommended"].append("Daptomycin 8-10 mg/kg IV daily")
            else:
                options["notes"].append("Daptomycin non-susceptible - consider alternatives")

            if lzd_susceptible:
                options["recommended"].append("Linezolid 600mg PO/IV BID")
            else:
                options["contraindicated"].append("Linezolid")

            # Alternatives
            options["alternative"].append("Tigecycline (if not bacteremia)")

            # Q/D only for E. faecium
            if species == EnterococcusSpecies.E_FAECIUM:
                options["alternative"].append("Quinupristin-dalfopristin")
            elif species == EnterococcusSpecies.E_FAECALIS:
                options["notes"].append("E. faecalis - intrinsically resistant to Q/D")

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

        # Validate van genotype
        if "van_genotype" in predictions and "van_genotype" in ground_truth:
            pred_geno = predictions["van_genotype"].get("genotype")
            true_geno = ground_truth["van_genotype"]

            metrics["van_genotype_correct"] = 1.0 if pred_geno == true_geno else 0.0

        return metrics


# Convenience export
__all__ = [
    "VREAnalyzer",
    "VREConfig",
    "EnterococcusSpecies",
    "VREGene",
    "VREDrug",
]
