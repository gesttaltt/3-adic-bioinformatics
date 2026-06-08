# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Zika Virus Analyzer for drug resistance and neurovirulence prediction.

Zika virus (ZIKV) is a mosquito-borne Flavivirus closely related to Dengue.
Key concerns include:
- Congenital Zika Syndrome (CZS) causing microcephaly
- Guillain-Barré Syndrome (GBS) association
- Sexual transmission capability
- NS3 protease and NS5 polymerase as drug targets

References:
- Lindenbach & Rice (2003) - Flavivirus molecular biology
- Musso & Gubler (2016) - Zika virus
- Petersen et al. (2016) - Zika virus mechanisms of pathogenesis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from src.diseases.base import DiseaseAnalyzer, DiseaseConfig, DiseaseType, TaskType


class ZikaLineage(Enum):
    """Zika virus lineages."""

    AFRICAN = "African"
    ASIAN = "Asian"
    AMERICAN = "American"  # Derived from Asian lineage


class ZikaGene(Enum):
    """Zika virus genes."""

    # Structural proteins
    C = "C"  # Capsid
    prM = "prM"  # Pre-membrane
    E = "E"  # Envelope

    # Non-structural proteins
    NS1 = "NS1"  # Immune evasion
    NS2A = "NS2A"  # Membrane-associated
    NS2B = "NS2B"  # NS3 protease cofactor
    NS3 = "NS3"  # Protease/helicase
    NS4A = "NS4A"  # Membrane rearrangement
    NS4B = "NS4B"  # Replication complex
    NS5 = "NS5"  # RNA-dependent RNA polymerase/methyltransferase


class ZikaDrug(Enum):
    """Zika antiviral drugs (experimental/repurposed)."""

    # NS3 protease inhibitors
    TEMOPORFIN = "temoporfin"
    NICLOSAMIDE = "niclosamide"
    NITAZOXANIDE = "nitazoxanide"

    # NS5 polymerase inhibitors
    SOFOSBUVIR = "sofosbuvir"
    RIBAVIRIN = "ribavirin"
    FAVIPIRAVIR = "favipiravir"
    GALIDESIVIR = "galidesivir"

    # Other mechanisms
    CHLOROQUINE = "chloroquine"  # Entry inhibitor
    MEMANTINE = "memantine"  # Neuroprotective


# NS3 protease resistance mutations
NS3_MUTATIONS = {
    ZikaDrug.TEMOPORFIN: {
        80: {"A": 1.0, "V": 2.5, "T": 2.0},
        130: {"Y": 1.0, "H": 1.8, "F": 1.5},
        135: {"T": 1.0, "A": 3.0, "S": 2.0},
    },
    ZikaDrug.NICLOSAMIDE: {
        51: {"H": 1.0, "Y": 2.0, "Q": 1.5},
        82: {"V": 1.0, "A": 2.5, "L": 1.8},
    },
}

# NS5 polymerase resistance mutations
NS5_RDPP_MUTATIONS = {
    ZikaDrug.SOFOSBUVIR: {
        282: {"S": 1.0, "T": 8.0, "A": 4.0},  # Key RdRp mutation
        320: {"R": 1.0, "K": 2.0, "Q": 3.0},
    },
    ZikaDrug.RIBAVIRIN: {
        415: {"F": 1.0, "Y": 2.5, "L": 2.0},
        530: {"D": 1.0, "N": 3.0, "E": 1.5},
    },
    ZikaDrug.FAVIPIRAVIR: {
        323: {"R": 1.0, "K": 2.5, "H": 2.0},
        540: {"T": 1.0, "A": 3.0, "S": 2.5},
    },
}

# NS5 methyltransferase resistance mutations
NS5_MTASE_MUTATIONS = {
    63: {"K": 1.0, "R": 1.5, "E": 2.0},  # SAM binding
    146: {"E": 1.0, "D": 1.8, "Q": 2.0},  # Cap binding
    227: {"D": 1.0, "E": 1.5, "N": 2.5},  # Catalytic
}

# Neurovirulence-associated markers
NEUROVIRULENCE_MARKERS = {
    ZikaGene.E: {
        # Receptor binding domain mutations affecting neural tropism
        154: {"N": "high", "S": "moderate", "D": "low"},  # Glycosylation site
        315: {"T": "high", "A": "moderate", "S": "low"},
        330: {"S": "high", "G": "moderate", "A": "low"},
    },
    ZikaGene.NS1: {
        # Immune evasion affecting CNS invasion
        188: {"V": "high", "A": "moderate", "T": "low"},
    },
    ZikaGene.prM: {
        # Secretion efficiency
        139: {"S": "high", "N": "moderate", "T": "low"},
    },
}

# Congenital Zika Syndrome risk markers
CZS_RISK_POSITIONS = {
    ZikaGene.E: {
        # Placental barrier crossing
        90: {"D": "high", "E": "moderate", "N": "low"},
        215: {"K": "high", "R": "moderate", "Q": "low"},
    },
    ZikaGene.NS4B: {
        # Persistence in fetal tissue
        106: {"L": "high", "I": "moderate", "V": "low"},
    },
}

# GBS-associated markers (Guillain-Barré Syndrome)
GBS_MARKERS = {
    ZikaGene.E: {
        # Autoimmune cross-reactivity
        175: {"V": "high", "I": "moderate", "A": "low"},
    },
    ZikaGene.NS1: {
        # Molecular mimicry with gangliosides
        205: {"S": "high", "T": "moderate", "A": "low"},
    },
}

# Lineage-specific marker positions
LINEAGE_MARKERS = {
    ZikaLineage.AFRICAN: {
        ZikaGene.E: {67: "V", 103: "V", 473: "M"},
    },
    ZikaLineage.ASIAN: {
        ZikaGene.E: {67: "A", 103: "A", 473: "V"},
    },
    ZikaLineage.AMERICAN: {
        ZikaGene.E: {67: "A", 103: "A", 473: "M"},  # Asian-derived with changes
    },
}


# Drug to gene mapping
DRUG_GENE_MAP = {
    ZikaDrug.TEMOPORFIN: ZikaGene.NS3,
    ZikaDrug.NICLOSAMIDE: ZikaGene.NS3,
    ZikaDrug.NITAZOXANIDE: ZikaGene.NS3,
    ZikaDrug.SOFOSBUVIR: ZikaGene.NS5,
    ZikaDrug.RIBAVIRIN: ZikaGene.NS5,
    ZikaDrug.FAVIPIRAVIR: ZikaGene.NS5,
    ZikaDrug.GALIDESIVIR: ZikaGene.NS5,
    ZikaDrug.CHLOROQUINE: ZikaGene.E,
    ZikaDrug.MEMANTINE: ZikaGene.NS5,  # Acts on host targets
}


@dataclass
class ZikaConfig(DiseaseConfig):
    """Configuration for Zika virus analysis."""

    name: str = "zika"
    display_name: str = "Zika Fever"
    disease_type: DiseaseType = DiseaseType.VIRAL
    tasks: list[TaskType] = field(
        default_factory=lambda: [
            TaskType.RESISTANCE,
            TaskType.ESCAPE,
            TaskType.FITNESS,
            TaskType.ANTIGENICITY,
        ]
    )

    # Data sources
    data_sources: dict[str, str] = field(
        default_factory=lambda: {
            "genbank": "https://www.ncbi.nlm.nih.gov/genbank/",
            "vipr": "https://www.viprbrc.org/",
            "nextstrain": "https://nextstrain.org/zika",
        }
    )

    # Zika-specific settings
    predict_czs_risk: bool = True
    predict_gbs_risk: bool = True
    predict_neurovirulence: bool = True
    classify_lineage: bool = True

    # Sequence settings
    min_sequence_length: int = 100

    genes: list[str] = field(default_factory=lambda: [g.value for g in ZikaGene])


class ZikaAnalyzer(DiseaseAnalyzer):
    """Analyzer for Zika virus drug resistance and neurovirulence.

    Features:
    - Drug resistance prediction for NS3/NS5 inhibitors
    - Lineage classification (African/Asian/American)
    - Congenital Zika Syndrome risk assessment
    - Guillain-Barré Syndrome association markers
    - Neurovirulence prediction
    """

    def __init__(self, config: ZikaConfig | None = None):
        """Initialize Zika analyzer.

        Args:
            config: Zika-specific configuration
        """
        self.config = config or ZikaConfig()
        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"

    def analyze(
        self,
        sequences: dict[ZikaGene, list[str]],
        lineage: ZikaLineage | None = None,
        pregnancy_context: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Analyze Zika sequences for resistance and virulence.

        Args:
            sequences: Dictionary mapping gene to protein sequences
            lineage: Known viral lineage
            pregnancy_context: Whether analysis is for pregnant patient
            **kwargs: Additional parameters

        Returns:
            Comprehensive analysis results
        """
        results = {
            "n_sequences": len(next(iter(sequences.values()), [])),
            "genes_analyzed": [g.value for g in sequences],
            "lineage": lineage.value if lineage else None,
            "pregnancy_context": pregnancy_context,
        }

        # Drug resistance prediction
        if ZikaGene.NS3 in sequences or ZikaGene.NS5 in sequences:
            results["drug_resistance"] = {}

            for drug in ZikaDrug:
                target_gene = DRUG_GENE_MAP.get(drug)
                if target_gene and target_gene in sequences:
                    drug_results = self.predict_drug_resistance(
                        sequences[target_gene],
                        drug,
                        target_gene,
                    )
                    results["drug_resistance"][drug.value] = drug_results

        # Lineage classification
        if self.config.classify_lineage and ZikaGene.E in sequences:
            results["lineage_classification"] = self._classify_lineage(sequences[ZikaGene.E])

        # Neurovirulence prediction
        if self.config.predict_neurovirulence:
            neurovirulence_results = {}
            for gene, seqs in sequences.items():
                if gene in NEUROVIRULENCE_MARKERS:
                    neurovirulence_results[gene.value] = self._analyze_neurovirulence(seqs, gene)
            if neurovirulence_results:
                results["neurovirulence"] = neurovirulence_results

        # CZS risk (especially important in pregnancy context)
        if self.config.predict_czs_risk:
            czs_results = {}
            for gene, seqs in sequences.items():
                if gene in CZS_RISK_POSITIONS:
                    czs_results[gene.value] = self._analyze_czs_risk(seqs, gene)
            if czs_results:
                results["czs_risk"] = czs_results
                results["czs_risk"]["overall_risk"] = self._calculate_overall_czs_risk(czs_results)

        # GBS risk
        if self.config.predict_gbs_risk:
            gbs_results = {}
            for gene, seqs in sequences.items():
                if gene in GBS_MARKERS:
                    gbs_results[gene.value] = self._analyze_gbs_risk(seqs, gene)
            if gbs_results:
                results["gbs_risk"] = gbs_results

        return results

    def predict_drug_resistance(
        self,
        sequences: list[str],
        drug: ZikaDrug,
        gene: ZikaGene,
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
        if gene == ZikaGene.NS3 and drug in NS3_MUTATIONS:
            mutations_db = NS3_MUTATIONS[drug]
        elif gene == ZikaGene.NS5 and drug in NS5_RDPP_MUTATIONS:
            mutations_db = NS5_RDPP_MUTATIONS[drug]
        else:
            # No specific mutation data - use generic scoring
            for seq in sequences:
                score = 0.1  # Low baseline
                results["scores"].append(score)
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
                                    {"position": pos, "amino_acid": aa, "fold_change": fold_change}
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

    def _classify_lineage(
        self,
        e_sequences: list[str],
    ) -> dict[str, Any]:
        """Classify viral lineage from E protein sequences.

        Args:
            e_sequences: Envelope protein sequences

        Returns:
            Lineage classification results
        """
        results = {
            "predictions": [],
            "confidences": [],
        }

        lineage_scores = {st: 0 for st in ZikaLineage}

        for seq in e_sequences:
            seq_lineage_scores = {st: 0 for st in ZikaLineage}

            for lineage, markers in LINEAGE_MARKERS.items():
                if ZikaGene.E in markers:
                    for pos, expected_aa in markers[ZikaGene.E].items():
                        if 0 < pos <= len(seq) and seq[pos - 1] == expected_aa:
                            seq_lineage_scores[lineage] += 1

            # Find best match
            best_lineage = max(seq_lineage_scores, key=seq_lineage_scores.get)
            best_score = seq_lineage_scores[best_lineage]
            total_markers = len(next(iter(LINEAGE_MARKERS.values())).get(ZikaGene.E, {}))

            confidence = best_score / max(total_markers, 1)
            results["predictions"].append(best_lineage.value)
            results["confidences"].append(confidence)
            lineage_scores[best_lineage] += 1

        # Overall prediction
        if results["predictions"]:
            most_common = max(lineage_scores, key=lineage_scores.get)
            results["predicted_lineage"] = most_common.value
            results["overall_confidence"] = np.mean(results["confidences"])

        return results

    def _analyze_neurovirulence(
        self,
        sequences: list[str],
        gene: ZikaGene,
    ) -> dict[str, Any]:
        """Analyze neurovirulence markers.

        Args:
            sequences: Protein sequences
            gene: Target gene

        Returns:
            Neurovirulence analysis
        """
        results = {
            "risk_scores": [],
            "risk_levels": [],
            "markers_detected": [],
        }

        markers = NEUROVIRULENCE_MARKERS.get(gene, {})

        for seq in sequences:
            score = 0.0
            detected = []

            for pos, risk_map in markers.items():
                if 0 < pos <= len(seq):
                    aa = seq[pos - 1]
                    if aa in risk_map:
                        risk_level = risk_map[aa]
                        if risk_level == "high":
                            score += 0.4
                        elif risk_level == "moderate":
                            score += 0.2
                        else:
                            score += 0.1

                        detected.append(
                            {
                                "position": pos,
                                "amino_acid": aa,
                                "risk_level": risk_level,
                            }
                        )

            results["risk_scores"].append(min(score, 1.0))
            results["markers_detected"].append(detected)

            # Classification
            if score < 0.3:
                level = "low"
            elif score < 0.6:
                level = "moderate"
            else:
                level = "high"
            results["risk_levels"].append(level)

        return results

    def _analyze_czs_risk(
        self,
        sequences: list[str],
        gene: ZikaGene,
    ) -> dict[str, Any]:
        """Analyze Congenital Zika Syndrome risk markers.

        Args:
            sequences: Protein sequences
            gene: Target gene

        Returns:
            CZS risk analysis
        """
        results = {
            "risk_scores": [],
            "risk_levels": [],
            "markers_detected": [],
        }

        markers = CZS_RISK_POSITIONS.get(gene, {})

        for seq in sequences:
            score = 0.0
            detected = []

            for pos, risk_map in markers.items():
                if 0 < pos <= len(seq):
                    aa = seq[pos - 1]
                    if aa in risk_map:
                        risk_level = risk_map[aa]
                        if risk_level == "high":
                            score += 0.5
                        elif risk_level == "moderate":
                            score += 0.25
                        else:
                            score += 0.1

                        detected.append(
                            {
                                "position": pos,
                                "amino_acid": aa,
                                "risk_level": risk_level,
                            }
                        )

            results["risk_scores"].append(min(score, 1.0))
            results["markers_detected"].append(detected)

            # Classification
            if score < 0.3:
                level = "low"
            elif score < 0.5:
                level = "moderate"
            else:
                level = "high"
            results["risk_levels"].append(level)

        return results

    def _calculate_overall_czs_risk(
        self,
        czs_results: dict[str, Any],
    ) -> str:
        """Calculate overall CZS risk from multiple genes.

        Args:
            czs_results: CZS analysis by gene

        Returns:
            Overall risk level
        """
        all_scores = []
        for gene_data in czs_results.values():
            if isinstance(gene_data, dict) and "risk_scores" in gene_data:
                all_scores.extend(gene_data["risk_scores"])

        if not all_scores:
            return "unknown"

        avg_score = np.mean(all_scores)
        if avg_score < 0.3:
            return "low"
        elif avg_score < 0.5:
            return "moderate"
        elif avg_score < 0.7:
            return "high"
        else:
            return "very_high"

    def _analyze_gbs_risk(
        self,
        sequences: list[str],
        gene: ZikaGene,
    ) -> dict[str, Any]:
        """Analyze Guillain-Barré Syndrome risk markers.

        Args:
            sequences: Protein sequences
            gene: Target gene

        Returns:
            GBS risk analysis
        """
        results = {
            "risk_scores": [],
            "risk_levels": [],
            "markers_detected": [],
        }

        markers = GBS_MARKERS.get(gene, {})

        for seq in sequences:
            score = 0.0
            detected = []

            for pos, risk_map in markers.items():
                if 0 < pos <= len(seq):
                    aa = seq[pos - 1]
                    if aa in risk_map:
                        risk_level = risk_map[aa]
                        if risk_level == "high":
                            score += 0.5
                        elif risk_level == "moderate":
                            score += 0.25
                        else:
                            score += 0.1

                        detected.append(
                            {
                                "position": pos,
                                "amino_acid": aa,
                                "risk_level": risk_level,
                            }
                        )

            results["risk_scores"].append(min(score, 1.0))
            results["markers_detected"].append(detected)

            # Classification
            if score < 0.3:
                level = "low"
            elif score < 0.5:
                level = "moderate"
            else:
                level = "high"
            results["risk_levels"].append(level)

        return results

    def get_pregnancy_recommendations(
        self,
        analysis_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate pregnancy-specific recommendations based on analysis.

        Args:
            analysis_results: Results from analyze()

        Returns:
            Pregnancy-specific recommendations
        """
        recommendations = {
            "monitoring_level": "standard",
            "actions": [],
            "warnings": [],
        }

        # Check CZS risk
        czs_risk = analysis_results.get("czs_risk", {}).get("overall_risk", "unknown")
        if czs_risk in ["high", "very_high"]:
            recommendations["monitoring_level"] = "intensive"
            recommendations["warnings"].append(
                "High risk for Congenital Zika Syndrome - recommend detailed fetal monitoring"
            )
            recommendations["actions"].append("Serial fetal ultrasounds recommended")
            recommendations["actions"].append("Consider amniocentesis for PCR testing")

        # Check neurovirulence
        neuro_data = analysis_results.get("neurovirulence", {})
        high_neuro = any(
            "high" in gene_data.get("risk_levels", [])
            for gene_data in neuro_data.values()
            if isinstance(gene_data, dict)
        )
        if high_neuro:
            recommendations["warnings"].append("High neurovirulence markers detected")

        # Drug recommendations
        drug_resistance = analysis_results.get("drug_resistance", {})
        susceptible_drugs = [
            drug for drug, data in drug_resistance.items() if data.get("classifications", [""])[0] == "susceptible"
        ]
        if susceptible_drugs:
            recommendations["potential_treatments"] = susceptible_drugs

        return recommendations

    def validate_predictions(
        self,
        predictions: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> dict[str, float]:
        """Validate predictions against phenotypic data.

        Args:
            predictions: Model predictions (drug resistance, CZS risk, etc.)
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

        # Validate lineage predictions
        if "lineage_classification" in predictions and "lineage" in ground_truth:
            pred_lineages = predictions["lineage_classification"].get("predictions", [])
            true_lineages = ground_truth["lineage"]

            if isinstance(true_lineages, list) and len(pred_lineages) == len(true_lineages):
                correct = sum(1 for p, t in zip(pred_lineages, true_lineages, strict=False) if p == t)
                metrics["lineage_accuracy"] = correct / max(len(true_lineages), 1)

        # Validate CZS risk predictions
        if "czs_risk" in predictions and "czs_outcomes" in ground_truth:
            all_scores = []
            for gene_data in predictions["czs_risk"].values():
                if isinstance(gene_data, dict) and "risk_scores" in gene_data:
                    all_scores.extend(gene_data["risk_scores"])

            true_outcomes = ground_truth["czs_outcomes"]
            if len(all_scores) == len(true_outcomes) and len(all_scores) > 1:
                corr, p_value = spearmanr(all_scores, true_outcomes)
                metrics["czs_risk_spearman"] = corr
                metrics["czs_risk_p_value"] = p_value

        # Validate neurovirulence predictions
        if "neurovirulence" in predictions and "neuro_outcomes" in ground_truth:
            all_scores = []
            for gene_data in predictions["neurovirulence"].values():
                if isinstance(gene_data, dict) and "risk_scores" in gene_data:
                    all_scores.extend(gene_data["risk_scores"])

            true_outcomes = ground_truth["neuro_outcomes"]
            if len(all_scores) == len(true_outcomes) and len(all_scores) > 1:
                corr, p_value = spearmanr(all_scores, true_outcomes)
                metrics["neuro_spearman"] = corr

        return metrics


# Convenience export
__all__ = [
    "ZikaAnalyzer",
    "ZikaConfig",
    "ZikaLineage",
    "ZikaGene",
    "ZikaDrug",
]
