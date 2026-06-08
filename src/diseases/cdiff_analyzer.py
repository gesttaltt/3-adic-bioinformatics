# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Clostridioides difficile Analyzer for drug resistance and virulence prediction.

Clostridioides difficile (C. diff) is a major cause of healthcare-associated
diarrhea and colitis. Key features:
- Toxin production (TcdA, TcdB) - primary virulence factors
- Binary toxin (CDTa, CDTb) - associated with hypervirulent strains
- Antibiotic resistance (metronidazole, vancomycin, fidaxomicin)
- Ribotype classification (027, 078, 001, 017, etc.)
- Spore formation and persistence

References:
- Leffler & Lamont (2015) - C. difficile infection
- Spigaglia (2016) - Antibiotic resistance in C. difficile
- Martin et al. (2016) - Molecular epidemiology and typing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from src.diseases.base import DiseaseAnalyzer, DiseaseConfig, DiseaseType, TaskType


class CDiffRibotype(Enum):
    """Major C. difficile ribotypes."""

    RT027 = "027"  # Hypervirulent, fluoroquinolone-resistant
    RT078 = "078"  # Hypervirulent, community-associated
    RT001 = "001"  # Common hospital strain
    RT017 = "017"  # TcdA-negative, TcdB-positive
    RT014 = "014"  # Common hospital strain
    RT002 = "002"  # Common strain
    RT106 = "106"  # UK endemic
    RT020 = "020"  # Fluoroquinolone-resistant
    OTHER = "other"


class CDiffGene(Enum):
    """C. difficile virulence and resistance genes."""

    # Toxin genes
    TCDA = "tcdA"  # Toxin A (enterotoxin)
    TCDB = "tcdB"  # Toxin B (cytotoxin)
    CDTA = "cdtA"  # Binary toxin A subunit
    CDTB = "cdtB"  # Binary toxin B subunit

    # Toxin regulators
    TCDC = "tcdC"  # Negative regulator of toxin expression
    TCDR = "tcdR"  # Positive regulator of toxin expression
    TCDE = "tcdE"  # Holin-like protein

    # Resistance genes
    VAN = "vanG"  # Vancomycin resistance (rare)
    NIM = "nim"  # Nitroimidazole (metronidazole) resistance
    GYRA = "gyrA"  # Fluoroquinolone resistance
    GYRB = "gyrB"  # Fluoroquinolone resistance
    RPOB = "rpoB"  # Rifamycin resistance

    # Spore formation
    SPOA = "spoA"  # Sporulation sigma factor
    SPOC = "spoC"  # Sporulation gene


class CDiffDrug(Enum):
    """Antibiotics for C. difficile treatment."""

    # First-line treatments
    VANCOMYCIN = "vancomycin"
    FIDAXOMICIN = "fidaxomicin"

    # Second-line/alternative
    METRONIDAZOLE = "metronidazole"
    RIFAXIMIN = "rifaximin"

    # Other
    TEICOPLANIN = "teicoplanin"
    TIGECYCLINE = "tigecycline"
    BEZLOTOXUMAB = "bezlotoxumab"  # Anti-toxin B antibody


# Metronidazole (nitroimidazole) resistance mutations
METRONIDAZOLE_MUTATIONS = {
    CDiffGene.NIM: {
        # nim gene variants associated with resistance
        1: {"M": 1.0, "I": 2.0, "L": 3.0},  # Start codon variants
    },
}

# Vancomycin reduced susceptibility markers
VANCOMYCIN_MARKERS = {
    CDiffGene.VAN: {
        # vanG cluster markers
        50: {"A": 1.0, "V": 1.5, "T": 2.0},
    },
}

# Fidaxomicin resistance mutations
FIDAXOMICIN_MUTATIONS = {
    CDiffGene.RPOB: {
        # RNA polymerase mutations
        516: {"D": 1.0, "N": 2.5, "G": 3.0},  # Asp516
        547: {"V": 1.0, "M": 2.0, "I": 2.5},  # Val547
    },
}

# Fluoroquinolone resistance mutations (affects colonization, not treatment)
FLUOROQUINOLONE_MUTATIONS = {
    CDiffGene.GYRA: {
        83: {"T": 1.0, "I": 50.0, "A": 20.0},  # Thr83Ile - high level resistance
        87: {"D": 1.0, "N": 10.0, "G": 8.0},  # Asp87
    },
    CDiffGene.GYRB: {
        426: {"D": 1.0, "N": 5.0, "G": 3.0},  # Asp426
    },
}

# TcdC mutations associated with hypervirulence
TCDC_MUTATIONS = {
    # 18bp deletion at position 117 (truncation)
    117: {"deletion": "hypervirulent"},
    # Single nucleotide deletion at position 39
    39: {"deletion": "moderate_hypervirulence"},
}

# Toxin A (TcdA) domains
TCDA_DOMAINS = {
    "glucosyltransferase": list(range(1, 546)),
    "cysteine_protease": list(range(546, 767)),
    "translocation": list(range(767, 1851)),
    "receptor_binding": list(range(1851, 2710)),
}

# Toxin B (TcdB) domains
TCDB_DOMAINS = {
    "glucosyltransferase": list(range(1, 546)),
    "cysteine_protease": list(range(546, 767)),
    "translocation": list(range(767, 1851)),
    "receptor_binding": list(range(1851, 2366)),
}

# Ribotype markers based on 16S-23S rRNA intergenic spacer
RIBOTYPE_MARKERS = {
    CDiffRibotype.RT027: {
        "tcdc_deletion": True,
        "binary_toxin": True,
        "fq_resistance": True,
    },
    CDiffRibotype.RT078: {
        "tcdc_deletion": True,
        "binary_toxin": True,
        "fq_resistance": False,
    },
    CDiffRibotype.RT017: {
        "tcda_negative": True,
        "binary_toxin": False,
        "fq_resistance": False,
    },
}

# Drug to gene mapping
DRUG_GENE_MAP = {
    CDiffDrug.VANCOMYCIN: CDiffGene.VAN,
    CDiffDrug.METRONIDAZOLE: CDiffGene.NIM,
    CDiffDrug.FIDAXOMICIN: CDiffGene.RPOB,
    CDiffDrug.RIFAXIMIN: CDiffGene.RPOB,
}


@dataclass
class CDiffConfig(DiseaseConfig):
    """Configuration for C. difficile analysis."""

    name: str = "cdiff"
    display_name: str = "Clostridioides difficile Infection"
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
            "pubmlst": "https://pubmlst.org/organisms/clostridioides-difficile",
            "ncbi_pathogen": "https://www.ncbi.nlm.nih.gov/pathogens/",
            "enterobase": "https://enterobase.warwick.ac.uk/",
        }
    )

    # C. diff-specific settings
    predict_toxin_expression: bool = True
    predict_hypervirulence: bool = True
    classify_ribotype: bool = True

    # Sequence settings
    min_sequence_length: int = 100

    genes: list[str] = field(default_factory=lambda: [g.value for g in CDiffGene])


class CDiffAnalyzer(DiseaseAnalyzer):
    """Analyzer for C. difficile drug resistance and virulence.

    Features:
    - Drug resistance prediction (metronidazole, vancomycin, fidaxomicin)
    - Ribotype classification
    - Toxin gene analysis (TcdA, TcdB, CDT)
    - Hypervirulence marker detection
    - Recurrence risk prediction
    """

    def __init__(self, config: CDiffConfig | None = None):
        """Initialize C. difficile analyzer.

        Args:
            config: C. diff-specific configuration
        """
        self.config = config or CDiffConfig()
        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"

    def analyze(
        self,
        sequences: dict[CDiffGene, list[str]],
        ribotype: CDiffRibotype | None = None,
        prior_cdi: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Analyze C. difficile sequences for resistance and virulence.

        Args:
            sequences: Dictionary mapping gene to protein sequences
            ribotype: Known ribotype
            prior_cdi: Whether patient had prior C. diff infection
            **kwargs: Additional parameters

        Returns:
            Comprehensive analysis results
        """
        results = {
            "n_sequences": len(next(iter(sequences.values()), [])),
            "genes_analyzed": [g.value for g in sequences],
            "ribotype": ribotype.value if ribotype else None,
            "prior_cdi": prior_cdi,
        }

        # Drug resistance prediction
        resistance_results = {}
        for drug in CDiffDrug:
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

        # Toxin analysis
        if self.config.predict_toxin_expression:
            toxin_results = {}
            if CDiffGene.TCDA in sequences:
                toxin_results["tcdA"] = self._analyze_toxin(sequences[CDiffGene.TCDA], "tcdA")
            if CDiffGene.TCDB in sequences:
                toxin_results["tcdB"] = self._analyze_toxin(sequences[CDiffGene.TCDB], "tcdB")
            if CDiffGene.CDTA in sequences or CDiffGene.CDTB in sequences:
                toxin_results["binary_toxin"] = {
                    "present": True,
                    "cdtA": CDiffGene.CDTA in sequences,
                    "cdtB": CDiffGene.CDTB in sequences,
                }
            if toxin_results:
                results["toxin_analysis"] = toxin_results

        # Hypervirulence prediction
        if self.config.predict_hypervirulence:
            results["hypervirulence"] = self._predict_hypervirulence(sequences, ribotype)

        # Ribotype classification
        if self.config.classify_ribotype and ribotype is None:
            results["ribotype_classification"] = self._classify_ribotype(sequences)

        # Recurrence risk
        results["recurrence_risk"] = self._predict_recurrence_risk(
            sequences, prior_cdi, results.get("drug_resistance", {})
        )

        return results

    def predict_drug_resistance(
        self,
        sequences: list[str],
        drug: CDiffDrug,
        gene: CDiffGene,
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

        # Get mutations database based on drug
        if drug == CDiffDrug.METRONIDAZOLE:
            mutations_db = METRONIDAZOLE_MUTATIONS.get(gene, {})
        elif drug == CDiffDrug.VANCOMYCIN:
            mutations_db = VANCOMYCIN_MARKERS.get(gene, {})
        elif drug in [CDiffDrug.FIDAXOMICIN, CDiffDrug.RIFAXIMIN]:
            mutations_db = FIDAXOMICIN_MUTATIONS.get(gene, {})
        else:
            # No specific mutation data
            for seq in sequences:
                results["scores"].append(0.1)
                results["classifications"].append("susceptible")
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
                            if fold_change > 1.5:
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

    def _analyze_toxin(
        self,
        sequences: list[str],
        toxin_type: str,
    ) -> dict[str, Any]:
        """Analyze toxin gene sequences.

        Args:
            sequences: Toxin gene sequences
            toxin_type: 'tcdA' or 'tcdB'

        Returns:
            Toxin analysis results
        """
        domains = TCDA_DOMAINS if toxin_type == "tcdA" else TCDB_DOMAINS

        results = {
            "present": True,
            "expression_levels": [],
            "domain_integrity": [],
            "variant_type": [],
        }

        for seq in sequences:
            # Check domain integrity
            domain_analysis = {}
            total_integrity = 0

            for domain_name, positions in domains.items():
                if len(seq) >= max(positions):
                    domain_seq = seq[min(positions) : max(positions)]
                    # Check for truncations or major deletions
                    integrity = 1.0 if len(domain_seq) > 0.9 * len(positions) else len(domain_seq) / len(positions)
                    domain_analysis[domain_name] = integrity
                    total_integrity += integrity
                else:
                    domain_analysis[domain_name] = 0.0

            avg_integrity = total_integrity / max(len(domains), 1)
            results["domain_integrity"].append(domain_analysis)

            # Expression level prediction based on sequence features
            # Higher integrity = higher potential expression
            if avg_integrity > 0.9:
                expression = "high"
            elif avg_integrity > 0.7:
                expression = "moderate"
            else:
                expression = "low"
            results["expression_levels"].append(expression)

            # Variant classification
            if avg_integrity > 0.95:
                variant = "wild_type"
            elif avg_integrity > 0.8:
                variant = "minor_variant"
            else:
                variant = "major_variant"
            results["variant_type"].append(variant)

        return results

    def _predict_hypervirulence(
        self,
        sequences: dict[CDiffGene, list[str]],
        ribotype: CDiffRibotype | None = None,
    ) -> dict[str, Any]:
        """Predict hypervirulence markers.

        Args:
            sequences: Gene sequences
            ribotype: Known ribotype

        Returns:
            Hypervirulence assessment
        """
        results = {
            "markers": [],
            "score": 0.0,
            "classification": "non-hypervirulent",
        }

        score = 0.0

        # Known hypervirulent ribotypes
        if ribotype in [CDiffRibotype.RT027, CDiffRibotype.RT078]:
            results["markers"].append(f"Hypervirulent ribotype ({ribotype.value})")
            score += 0.5

        # Binary toxin presence
        if CDiffGene.CDTA in sequences or CDiffGene.CDTB in sequences:
            results["markers"].append("Binary toxin (CDT) present")
            score += 0.3

        # TcdC mutations/deletions
        if CDiffGene.TCDC in sequences:
            tcdc_seqs = sequences[CDiffGene.TCDC]
            for seq in tcdc_seqs:
                # Check for truncation (shorter than expected)
                if len(seq) < 200:  # Normal TcdC ~232 aa
                    results["markers"].append("TcdC truncation detected")
                    score += 0.3
                    break

        # Fluoroquinolone resistance (marker for epidemic strains)
        if CDiffGene.GYRA in sequences:
            for seq in sequences[CDiffGene.GYRA]:
                if len(seq) > 83 and seq[82] in ["I", "A"]:  # Thr83Ile/Ala
                    results["markers"].append("Fluoroquinolone resistance (GyrA Thr83)")
                    score += 0.2
                    break

        results["score"] = min(score, 1.0)

        # Classification
        if score >= 0.6:
            results["classification"] = "hypervirulent"
        elif score >= 0.3:
            results["classification"] = "potentially_hypervirulent"
        else:
            results["classification"] = "non-hypervirulent"

        return results

    def _classify_ribotype(
        self,
        sequences: dict[CDiffGene, list[str]],
    ) -> dict[str, Any]:
        """Classify ribotype based on sequence markers.

        Args:
            sequences: Gene sequences

        Returns:
            Ribotype classification
        """
        results = {
            "predicted_ribotype": "unknown",
            "confidence": 0.0,
            "markers_detected": [],
        }

        scores = {rt: 0.0 for rt in CDiffRibotype}

        # Check for binary toxin
        has_binary_toxin = CDiffGene.CDTA in sequences or CDiffGene.CDTB in sequences

        # Check for TcdA
        has_tcda = CDiffGene.TCDA in sequences

        # Check for TcdC truncation
        has_tcdc_truncation = False
        if CDiffGene.TCDC in sequences:
            for seq in sequences[CDiffGene.TCDC]:
                if len(seq) < 200:
                    has_tcdc_truncation = True
                    break

        # Check for FQ resistance
        has_fq_resistance = False
        if CDiffGene.GYRA in sequences:
            for seq in sequences[CDiffGene.GYRA]:
                if len(seq) > 83 and seq[82] in ["I", "A"]:
                    has_fq_resistance = True
                    break

        # Score ribotypes based on markers
        if has_binary_toxin and has_tcdc_truncation and has_fq_resistance:
            scores[CDiffRibotype.RT027] = 0.8
            results["markers_detected"].extend(["binary_toxin", "tcdC_truncation", "fq_resistance"])
        elif has_binary_toxin and has_tcdc_truncation:
            scores[CDiffRibotype.RT078] = 0.7
            results["markers_detected"].extend(["binary_toxin", "tcdC_truncation"])
        elif not has_tcda:
            scores[CDiffRibotype.RT017] = 0.7
            results["markers_detected"].append("tcdA_negative")
        else:
            scores[CDiffRibotype.RT001] = 0.4  # Default common strain

        # Find best match
        best_ribotype = max(scores, key=scores.get)
        best_score = scores[best_ribotype]

        if best_score > 0:
            results["predicted_ribotype"] = best_ribotype.value
            results["confidence"] = best_score

        return results

    def _predict_recurrence_risk(
        self,
        sequences: dict[CDiffGene, list[str]],
        prior_cdi: bool,
        resistance_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Predict risk of CDI recurrence.

        Args:
            sequences: Gene sequences
            prior_cdi: Prior CDI history
            resistance_data: Drug resistance analysis

        Returns:
            Recurrence risk assessment
        """
        results = {
            "risk_score": 0.0,
            "risk_factors": [],
            "risk_level": "low",
        }

        score = 0.0

        # Prior CDI is major risk factor
        if prior_cdi:
            score += 0.4
            results["risk_factors"].append("Prior CDI episode")

        # Drug resistance increases recurrence risk
        for drug, data in resistance_data.items():
            if "resistant" in data.get("classifications", []):
                score += 0.2
                results["risk_factors"].append(f"{drug} resistance")

        # Binary toxin associated with recurrence
        if CDiffGene.CDTA in sequences or CDiffGene.CDTB in sequences:
            score += 0.15
            results["risk_factors"].append("Binary toxin present")

        # Spore genes (persistence)
        if CDiffGene.SPOA in sequences or CDiffGene.SPOC in sequences:
            score += 0.1
            results["risk_factors"].append("Sporulation genes present")

        results["risk_score"] = min(score, 1.0)

        # Risk classification
        if score >= 0.5:
            results["risk_level"] = "high"
        elif score >= 0.3:
            results["risk_level"] = "moderate"
        else:
            results["risk_level"] = "low"

        return results

    def get_treatment_recommendations(
        self,
        analysis_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate treatment recommendations based on analysis.

        Args:
            analysis_results: Results from analyze()

        Returns:
            Treatment recommendations
        """
        recommendations = {
            "first_line": [],
            "alternatives": [],
            "cautions": [],
            "monitoring": [],
        }

        resistance_data = analysis_results.get("drug_resistance", {})
        hypervirulence = analysis_results.get("hypervirulence", {})
        recurrence = analysis_results.get("recurrence_risk", {})

        # Check vancomycin susceptibility
        vanco_data = resistance_data.get("vancomycin", {})
        vanco_resistant = "resistant" in vanco_data.get("classifications", [])

        # Check fidaxomicin susceptibility
        fidaxo_data = resistance_data.get("fidaxomicin", {})
        fidaxo_resistant = "resistant" in fidaxo_data.get("classifications", [])

        # First episode, non-severe
        if not vanco_resistant and not fidaxo_resistant:
            if hypervirulence.get("classification") == "hypervirulent":
                recommendations["first_line"].append("Fidaxomicin 200mg PO BID x 10 days")
                recommendations["first_line"].append("Consider bezlotoxumab for recurrence prevention")
            else:
                recommendations["first_line"].append("Vancomycin 125mg PO QID x 10 days")
                recommendations["alternatives"].append("Fidaxomicin 200mg PO BID x 10 days")

        elif vanco_resistant:
            recommendations["first_line"].append("Fidaxomicin 200mg PO BID x 10 days")
            recommendations["cautions"].append("Vancomycin resistance detected - avoid vancomycin")

        elif fidaxo_resistant:
            recommendations["first_line"].append("Vancomycin 125mg PO QID x 10 days")
            recommendations["cautions"].append("Fidaxomicin resistance detected")

        # High recurrence risk
        if recurrence.get("risk_level") == "high":
            recommendations["monitoring"].append("Consider bezlotoxumab for recurrence prevention")
            recommendations["monitoring"].append("Close follow-up for recurrence symptoms")

        # Hypervirulent strain
        if hypervirulence.get("classification") == "hypervirulent":
            recommendations["cautions"].append("Hypervirulent strain - monitor for severe disease")
            recommendations["monitoring"].append("Consider ICU-level monitoring if clinically indicated")

        return recommendations

    def validate_predictions(
        self,
        predictions: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> dict[str, float]:
        """Validate predictions against phenotypic data.

        Args:
            predictions: Model predictions
            ground_truth: Known values from clinical/laboratory data

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

        # Validate ribotype predictions
        if "ribotype_classification" in predictions and "ribotype" in ground_truth:
            pred_ribotype = predictions["ribotype_classification"].get("predicted_ribotype")
            true_ribotype = ground_truth["ribotype"]

            metrics["ribotype_correct"] = 1.0 if pred_ribotype == true_ribotype else 0.0

        # Validate hypervirulence predictions
        if "hypervirulence" in predictions and "hypervirulent" in ground_truth:
            pred_hyper = predictions["hypervirulence"].get("classification") == "hypervirulent"
            true_hyper = ground_truth["hypervirulent"]

            metrics["hypervirulence_correct"] = 1.0 if pred_hyper == true_hyper else 0.0

        # Validate recurrence predictions
        if "recurrence_risk" in predictions and "recurrence" in ground_truth:
            pred_score = predictions["recurrence_risk"].get("risk_score", 0)
            true_recurrence = ground_truth["recurrence"]

            if isinstance(true_recurrence, list) and isinstance(pred_score, list):
                if len(pred_score) == len(true_recurrence) and len(pred_score) > 1:
                    corr, _ = spearmanr(pred_score, true_recurrence)
                    metrics["recurrence_spearman"] = corr

        return metrics


# Convenience export
__all__ = [
    "CDiffAnalyzer",
    "CDiffConfig",
    "CDiffRibotype",
    "CDiffGene",
    "CDiffDrug",
]
