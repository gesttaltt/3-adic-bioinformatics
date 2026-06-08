# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Dengue Virus Analyzer for Drug Resistance and ADE Risk Prediction.

This module provides analysis of Dengue virus serotypes for:
- Drug resistance prediction (NS3 protease, NS5 polymerase inhibitors)
- Antibody-dependent enhancement (ADE) risk scoring
- NS1 antigenicity for diagnostic sensitivity
- Serotype classification and cross-reactivity

Based on WHO dengue guidelines and ViPR database.

Key Features:
- Support for all four serotypes (DENV-1 through DENV-4)
- NS3 protease inhibitor resistance prediction
- NS5 RdRp inhibitor resistance (similar to HCV NS5B)
- Cross-serotype immune escape prediction

Data Sources:
- ViPR: https://www.viprbrc.org/
- NCBI Dengue: https://www.ncbi.nlm.nih.gov/genomes/VirusVariation/
- DengueNet: WHO surveillance

Usage:
    from src.diseases.dengue_analyzer import DengueAnalyzer

    analyzer = DengueAnalyzer()
    results = analyzer.analyze(sequences, serotype=DengueSerotype.DENV2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

from .base import DiseaseAnalyzer, DiseaseConfig, DiseaseType, TaskType


class DengueSerotype(Enum):
    """Dengue virus serotypes."""

    DENV1 = "DENV-1"
    DENV2 = "DENV-2"
    DENV3 = "DENV-3"
    DENV4 = "DENV-4"


class DengueGene(Enum):
    """Dengue virus gene segments."""

    # Structural proteins
    C = "C"  # Capsid
    prM = "prM"  # Precursor membrane
    E = "E"  # Envelope - immune target

    # Non-structural proteins
    NS1 = "NS1"  # Secreted, diagnostic target
    NS2A = "NS2A"
    NS2B = "NS2B"  # NS3 cofactor
    NS3 = "NS3"  # Protease/helicase - drug target
    NS4A = "NS4A"
    NS4B = "NS4B"
    NS5 = "NS5"  # RdRp/MTase - drug target


class DengueDrug(Enum):
    """Dengue antiviral drug candidates."""

    # NS3 protease inhibitors
    ASUNAPREVIR = "asunaprevir"  # HCV drug, cross-reactive
    BORTEZOMIB = "bortezomib"  # Protease inhibitor

    # NS5 polymerase inhibitors
    SOFOSBUVIR = "sofosbuvir"  # HCV NS5B inhibitor
    BALAPIRAVIR = "balapiravir"  # Nucleoside analog
    RIBAVIRIN = "ribavirin"  # Broad-spectrum

    # NS5 MTase inhibitors
    SINEFUNGIN = "sinefungin"  # SAH analog

    # NS4B inhibitors
    NITAZOXANIDE = "nitazoxanide"  # Broad-spectrum

    # Entry inhibitors
    CELGOSIVIR = "celgosivir"  # Alpha-glucosidase inhibitor


@dataclass
class DengueConfig(DiseaseConfig):
    """Configuration for dengue analysis."""

    name: str = "dengue"
    display_name: str = "Dengue Fever"
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
            "vipr": "https://www.viprbrc.org/",
            "ncbi_virus": "https://www.ncbi.nlm.nih.gov/genomes/VirusVariation/",
            "who_dengue": "https://www.who.int/denguecontrol/",
        }
    )


# NS3 protease resistance mutations
# Based on flavivirus NS3 structure and HCV cross-resistance patterns
NS3_MUTATIONS = {
    # Catalytic site mutations
    51: {"H": {"mutations": ["Y", "F"], "effect": "high", "drugs": ["asunaprevir"]}},
    75: {"D": {"mutations": ["E", "N"], "effect": "moderate", "drugs": ["asunaprevir"]}},
    80: {"Q": {"mutations": ["K", "R"], "effect": "moderate", "drugs": ["asunaprevir"]}},
    # Substrate binding mutations
    132: {"A": {"mutations": ["V", "T"], "effect": "moderate", "drugs": ["asunaprevir"]}},
    135: {"T": {"mutations": ["A", "S"], "effect": "low", "drugs": ["bortezomib"]}},
    # Helicase domain mutations (affect ATP binding)
    290: {"K": {"mutations": ["R", "Q"], "effect": "moderate", "drugs": ["asunaprevir", "bortezomib"]}},
    298: {"D": {"mutations": ["E", "N"], "effect": "moderate", "drugs": ["asunaprevir"]}},
}

# NS5 RdRp resistance mutations
# Based on HCV NS5B cross-resistance and flavivirus-specific positions
NS5_RDPP_MUTATIONS = {
    # Active site mutations
    310: {"G": {"mutations": ["S", "A"], "effect": "high", "drugs": ["sofosbuvir", "balapiravir"]}},
    314: {"S": {"mutations": ["G", "A"], "effect": "high", "drugs": ["sofosbuvir"]}},
    # Palm domain
    244: {"C": {"mutations": ["S", "F"], "effect": "moderate", "drugs": ["balapiravir"]}},
    282: {"S": {"mutations": ["T"], "effect": "high", "drugs": ["sofosbuvir"]}},
    # Thumb domain
    316: {"L": {"mutations": ["F", "M"], "effect": "moderate", "drugs": ["balapiravir", "ribavirin"]}},
    368: {"M": {"mutations": ["V", "I"], "effect": "moderate", "drugs": ["ribavirin"]}},
}

# NS5 MTase resistance mutations
NS5_MTASE_MUTATIONS = {
    # SAM binding site
    55: {"K": {"mutations": ["R", "Q"], "effect": "moderate", "drugs": ["sinefungin"]}},
    80: {"D": {"mutations": ["E", "N"], "effect": "moderate", "drugs": ["sinefungin"]}},
    105: {"E": {"mutations": ["D", "Q"], "effect": "low", "drugs": ["sinefungin"]}},
}

# NS1 antigenic regions for diagnostic sensitivity
NS1_ANTIGENIC_SITES = {
    "site_A": [20, 21, 22, 23, 24, 37, 38, 39, 40, 41],
    "site_B": [111, 112, 113, 114, 115, 127, 128, 129, 130],
    "site_C": [181, 182, 183, 184, 185, 186, 193, 194, 195],
    "site_D": [227, 228, 229, 230, 231, 246, 247, 248, 249],
}

# Envelope (E) protein domains for ADE and neutralization
E_PROTEIN_DOMAINS = {
    "domain_I": list(range(1, 53)) + list(range(134, 193)) + list(range(280, 296)),
    "domain_II": list(range(53, 134)) + list(range(193, 280)),
    "domain_III": list(range(296, 395)),  # Major neutralizing domain
    "fusion_loop": list(range(98, 112)),  # Highly conserved
}

# Cross-reactive epitopes for ADE risk
ADE_RISK_EPITOPES = {
    # Epitopes that generate cross-reactive but non-neutralizing Abs
    "ade_epitope_1": list(range(100, 110)),  # Fusion loop adjacent
    "ade_epitope_2": list(range(225, 235)),  # Domain II
    "ade_epitope_3": list(range(270, 280)),  # Domain II/III junction
}

# Serotype-specific conserved residues
SEROTYPE_MARKERS = {
    DengueSerotype.DENV1: {
        DengueGene.E: {17: "V", 155: "A", 390: "H"},
    },
    DengueSerotype.DENV2: {
        DengueGene.E: {17: "I", 155: "T", 390: "Y"},
    },
    DengueSerotype.DENV3: {
        DengueGene.E: {17: "V", 155: "A", 390: "H"},
    },
    DengueSerotype.DENV4: {
        DengueGene.E: {17: "V", 155: "V", 390: "N"},
    },
}


# Drug to gene mapping
DRUG_GENE_MAP = {
    DengueDrug.ASUNAPREVIR: DengueGene.NS3,
    DengueDrug.BORTEZOMIB: DengueGene.NS3,
    DengueDrug.SOFOSBUVIR: DengueGene.NS5,
    DengueDrug.BALAPIRAVIR: DengueGene.NS5,
    DengueDrug.RIBAVIRIN: DengueGene.NS5,
    DengueDrug.SINEFUNGIN: DengueGene.NS5,
    DengueDrug.NITAZOXANIDE: DengueGene.NS4B,
    DengueDrug.CELGOSIVIR: DengueGene.E,
}


class DengueAnalyzer(DiseaseAnalyzer):
    """Analyzer for dengue virus drug resistance and ADE risk.

    Provides:
    - NS3 protease inhibitor resistance prediction
    - NS5 polymerase/MTase inhibitor resistance prediction
    - ADE (antibody-dependent enhancement) risk scoring
    - NS1 antigenicity for diagnostic sensitivity
    - Serotype classification
    """

    def __init__(self, config: DengueConfig | None = None):
        """Initialize analyzer.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or DengueConfig()
        super().__init__(self.config)

        # Amino acid encoding
        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY-X*"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_alphabet)}

    def analyze(
        self,
        sequences: dict[DengueGene, list[str]],
        serotype: DengueSerotype | None = None,
        embeddings: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Analyze dengue virus sequences.

        Args:
            sequences: Dictionary mapping gene to list of sequences
            serotype: Known serotype (auto-detected if None)
            embeddings: Optional precomputed embeddings

        Returns:
            Analysis results
        """
        results = {
            "n_sequences": len(next(iter(sequences.values()))) if sequences else 0,
            "serotype": serotype.value if serotype else "unknown",
            "genes_analyzed": [g.value for g in sequences],
            "drug_resistance": {},
            "ade_risk": {},
            "ns1_antigenicity": {},
            "serotype_classification": {},
        }

        # Auto-detect serotype if not provided
        if serotype is None and DengueGene.E in sequences:
            serotype_results = self._classify_serotype(sequences[DengueGene.E])
            results["serotype_classification"] = serotype_results
            if serotype_results.get("predicted_serotype"):
                serotype = DengueSerotype(serotype_results["predicted_serotype"])
                results["serotype"] = serotype.value

        # Drug resistance for each drug
        for drug in DengueDrug:
            gene = DRUG_GENE_MAP.get(drug)
            if gene and gene in sequences:
                drug_results = self.predict_drug_resistance(sequences[gene], drug, gene)
                results["drug_resistance"][drug.value] = drug_results

        # ADE risk analysis (E protein)
        if DengueGene.E in sequences:
            results["ade_risk"] = self._analyze_ade_risk(sequences[DengueGene.E], serotype)

        # NS1 antigenicity
        if DengueGene.NS1 in sequences:
            results["ns1_antigenicity"] = self._analyze_ns1_antigenicity(sequences[DengueGene.NS1])

        return results

    def predict_drug_resistance(
        self,
        sequences: list[str],
        drug: DengueDrug,
        gene: DengueGene,
    ) -> dict[str, Any]:
        """Predict resistance for a specific drug.

        Args:
            sequences: Gene sequences (NS3, NS5, etc.)
            drug: Target drug
            gene: Target gene

        Returns:
            Resistance predictions
        """
        results = {
            "drug": drug.value,
            "gene": gene.value,
            "scores": [],
            "classifications": [],
            "mutations": [],
        }

        # Select appropriate mutation database
        if gene == DengueGene.NS3:
            mutation_db = NS3_MUTATIONS
        elif gene == DengueGene.NS5:
            # Combine RdRp and MTase mutations
            mutation_db = {**NS5_RDPP_MUTATIONS, **NS5_MTASE_MUTATIONS}
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
                data = info[ref_aa]

                if seq_aa != ref_aa and seq_aa in data["mutations"]:
                    # Check if this mutation affects our target drug
                    if drug.value in data["drugs"]:
                        effect = data["effect"]
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

    def _classify_serotype(
        self,
        e_sequences: list[str],
    ) -> dict[str, Any]:
        """Classify dengue serotype from E protein sequences.

        Args:
            e_sequences: Envelope protein sequences

        Returns:
            Serotype classification results
        """
        results = {
            "predictions": [],
            "confidences": [],
            "predicted_serotype": None,
        }

        serotype_scores = {st: 0 for st in DengueSerotype}

        for seq in e_sequences:
            seq_serotype_scores = {st: 0 for st in DengueSerotype}

            for serotype, markers in SEROTYPE_MARKERS.items():
                if DengueGene.E in markers:
                    for pos, expected_aa in markers[DengueGene.E].items():
                        if 0 < pos <= len(seq) and seq[pos - 1] == expected_aa:
                            seq_serotype_scores[serotype] += 1

            # Find best match
            best_serotype = max(seq_serotype_scores, key=seq_serotype_scores.get)
            best_score = seq_serotype_scores[best_serotype]
            total_markers = len(next(iter(SEROTYPE_MARKERS.values())).get(DengueGene.E, {}))

            confidence = best_score / max(total_markers, 1)
            results["predictions"].append(best_serotype.value)
            results["confidences"].append(confidence)
            serotype_scores[best_serotype] += 1

        # Overall prediction
        if results["predictions"]:
            most_common = max(serotype_scores, key=serotype_scores.get)
            results["predicted_serotype"] = most_common.value
            results["overall_confidence"] = np.mean(results["confidences"])

        return results

    def _analyze_ade_risk(
        self,
        e_sequences: list[str],
        serotype: DengueSerotype | None = None,
    ) -> dict[str, Any]:
        """Analyze antibody-dependent enhancement risk.

        ADE occurs when cross-reactive but non-neutralizing antibodies
        from prior infection enhance viral entry in secondary infection.

        Args:
            e_sequences: Envelope protein sequences
            serotype: Current serotype

        Returns:
            ADE risk analysis
        """
        results = {
            "risk_scores": [],
            "risk_levels": [],
            "cross_reactive_epitopes": [],
            "fusion_loop_conservation": [],
        }

        for seq in e_sequences:
            # Analyze ADE risk epitopes
            ade_score = 0.0
            epitope_variations = []

            for epitope_name, positions in ADE_RISK_EPITOPES.items():
                variations = 0
                for pos in positions:
                    if 0 < pos <= len(seq):
                        # Check for non-conservative substitutions
                        aa = seq[pos - 1]
                        # Simple heuristic: charged <-> hydrophobic = high ADE risk
                        if aa in "DEKR":  # Charged
                            variations += 0.3
                        elif aa in "AVILMFYW":  # Hydrophobic
                            variations += 0.1

                epitope_variations.append(
                    {
                        "epitope": epitope_name,
                        "variation_score": variations / len(positions),
                    }
                )
                ade_score += variations

            # Normalize ADE score
            max_ade = 3.0
            normalized_ade = min(ade_score / max_ade, 1.0)

            # Fusion loop conservation (high conservation = lower ADE risk)
            fusion_conserved = 0
            for pos in E_PROTEIN_DOMAINS["fusion_loop"]:
                if 0 < pos <= len(seq):
                    # Fusion loop should be highly conserved
                    fusion_conserved += 1

            fusion_conservation = fusion_conserved / len(E_PROTEIN_DOMAINS["fusion_loop"])

            # Final risk score (higher epitope variation + lower fusion conservation = higher risk)
            final_risk = normalized_ade * (1 - 0.5 * fusion_conservation)

            results["risk_scores"].append(final_risk)
            results["cross_reactive_epitopes"].append(epitope_variations)
            results["fusion_loop_conservation"].append(fusion_conservation)

            # Risk level
            if final_risk < 0.3:
                risk_level = "low"
            elif final_risk < 0.6:
                risk_level = "moderate"
            else:
                risk_level = "high"

            results["risk_levels"].append(risk_level)

        return results

    def _analyze_ns1_antigenicity(
        self,
        ns1_sequences: list[str],
    ) -> dict[str, Any]:
        """Analyze NS1 antigenicity for diagnostic sensitivity.

        Mutations in NS1 antigenic sites may affect diagnostic test performance.

        Args:
            ns1_sequences: NS1 protein sequences

        Returns:
            NS1 antigenicity analysis
        """
        results = {
            "antigenicity_scores": [],
            "site_variations": [],
            "diagnostic_impact": [],
        }

        for seq in ns1_sequences:
            site_data = {}
            total_variations = 0

            for site_name, positions in NS1_ANTIGENIC_SITES.items():
                variations = 0
                site_seq = ""

                for pos in positions:
                    if 0 < pos <= len(seq):
                        site_seq += seq[pos - 1]
                        # Count non-standard amino acids
                        if seq[pos - 1] in "X*-":
                            variations += 1

                site_data[site_name] = {
                    "sequence": site_seq,
                    "variations": variations,
                    "length": len(positions),
                }
                total_variations += variations

            # Antigenicity score (lower variation = better for diagnostics)
            total_positions = sum(len(p) for p in NS1_ANTIGENIC_SITES.values())
            antigenicity = 1.0 - (total_variations / max(total_positions, 1))

            results["antigenicity_scores"].append(antigenicity)
            results["site_variations"].append(site_data)

            # Diagnostic impact
            if antigenicity > 0.9:
                impact = "minimal"
            elif antigenicity > 0.7:
                impact = "low"
            elif antigenicity > 0.5:
                impact = "moderate"
            else:
                impact = "high"

            results["diagnostic_impact"].append(impact)

        return results

    def get_cross_protection_matrix(
        self,
        serotype: DengueSerotype,
    ) -> dict[str, float]:
        """Get cross-protection estimates between serotypes.

        Args:
            serotype: Primary serotype

        Returns:
            Cross-protection estimates for each serotype
        """
        # Cross-protection matrix based on epidemiological data
        # Values represent estimated protection from prior infection
        cross_protection = {
            DengueSerotype.DENV1: {
                DengueSerotype.DENV1: 1.0,
                DengueSerotype.DENV2: 0.3,
                DengueSerotype.DENV3: 0.3,
                DengueSerotype.DENV4: 0.2,
            },
            DengueSerotype.DENV2: {
                DengueSerotype.DENV1: 0.3,
                DengueSerotype.DENV2: 1.0,
                DengueSerotype.DENV3: 0.25,
                DengueSerotype.DENV4: 0.2,
            },
            DengueSerotype.DENV3: {
                DengueSerotype.DENV1: 0.3,
                DengueSerotype.DENV2: 0.25,
                DengueSerotype.DENV3: 1.0,
                DengueSerotype.DENV4: 0.2,
            },
            DengueSerotype.DENV4: {
                DengueSerotype.DENV1: 0.2,
                DengueSerotype.DENV2: 0.2,
                DengueSerotype.DENV3: 0.2,
                DengueSerotype.DENV4: 1.0,
            },
        }

        return {st.value: v for st, v in cross_protection.get(serotype, {}).items()}

    def predict_severe_dengue_risk(
        self,
        e_sequence: str,
        prior_serotypes: list[DengueSerotype],
        current_serotype: DengueSerotype | None = None,
    ) -> dict[str, Any]:
        """Predict risk of severe dengue based on secondary infection.

        Args:
            e_sequence: Current infection E protein sequence
            prior_serotypes: Previous dengue infections
            current_serotype: Current infection serotype

        Returns:
            Severe dengue risk assessment
        """
        results = {
            "secondary_infection": len(prior_serotypes) > 0,
            "prior_serotypes": [s.value for s in prior_serotypes],
            "ade_risk_factors": [],
            "overall_risk": "unknown",
        }

        if not prior_serotypes:
            results["overall_risk"] = "low"
            results["risk_score"] = 0.1
            return results

        # Secondary infection increases risk
        base_risk = 0.3

        # Analyze E protein for ADE epitopes
        ade_analysis = self._analyze_ade_risk([e_sequence], current_serotype)
        ade_score = ade_analysis["risk_scores"][0] if ade_analysis["risk_scores"] else 0

        # Cross-reactivity increases risk
        if current_serotype:
            cross_protection = self.get_cross_protection_matrix(current_serotype)
            for prior in prior_serotypes:
                protection = cross_protection.get(prior.value, 0.2)
                # Low cross-protection = higher ADE risk
                if protection < 0.3:
                    results["ade_risk_factors"].append(f"Low cross-protection from {prior.value}")
                    base_risk += 0.15

        # Combine risks
        final_risk = min(base_risk + ade_score * 0.5, 1.0)
        results["risk_score"] = final_risk

        # Risk classification
        if final_risk < 0.3:
            results["overall_risk"] = "low"
        elif final_risk < 0.5:
            results["overall_risk"] = "moderate"
        elif final_risk < 0.7:
            results["overall_risk"] = "high"
        else:
            results["overall_risk"] = "very_high"

        return results

    def validate_predictions(
        self,
        predictions: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> dict[str, float]:
        """Validate predictions against phenotypic data.

        Args:
            predictions: Model predictions (drug resistance, serotype, etc.)
            ground_truth: Known values from clinical/laboratory data

        Returns:
            Dictionary of validation metrics (Spearman correlation, accuracy, etc.)
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

        # Validate serotype predictions
        if "serotype" in predictions and "serotype" in ground_truth:
            pred_serotypes = predictions.get("serotype_predictions", [])
            true_serotypes = ground_truth["serotype"]

            if isinstance(true_serotypes, list) and len(pred_serotypes) == len(true_serotypes):
                correct = sum(1 for p, t in zip(pred_serotypes, true_serotypes, strict=False) if p == t)
                metrics["serotype_accuracy"] = correct / max(len(true_serotypes), 1)

        # Validate ADE risk predictions
        if "ade_risk" in predictions and "ade_outcomes" in ground_truth:
            ade_scores = predictions["ade_risk"].get("risk_scores", [])
            true_outcomes = ground_truth["ade_outcomes"]

            if len(ade_scores) == len(true_outcomes) and len(ade_scores) > 1:
                corr, p_value = spearmanr(ade_scores, true_outcomes)
                metrics["ade_risk_spearman"] = corr
                metrics["ade_risk_p_value"] = p_value

        # Validate severe dengue predictions
        if "severe_dengue_risk" in predictions and "severe_outcomes" in ground_truth:
            pred_risk = predictions["severe_dengue_risk"].get("risk_score", 0)
            true_outcome = ground_truth["severe_outcomes"]

            # Binary classification if multiple samples
            if isinstance(pred_risk, list) and isinstance(true_outcome, list):
                if len(pred_risk) == len(true_outcome) and len(pred_risk) > 1:
                    corr, p_value = spearmanr(pred_risk, true_outcome)
                    metrics["severe_risk_spearman"] = corr

        return metrics


# Convenience export
__all__ = [
    "DengueAnalyzer",
    "DengueConfig",
    "DengueSerotype",
    "DengueGene",
    "DengueDrug",
]
