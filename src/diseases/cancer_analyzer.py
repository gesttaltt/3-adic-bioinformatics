# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Cancer Targeted Therapy Resistance Analyzer.

This module provides analysis of acquired resistance mutations to
targeted therapies in cancer, focusing on EGFR and BRAF.

Based on:
- cBioPortal (https://www.cbioportal.org/)
- COSMIC (https://cancer.sanger.ac.uk/cosmic)
- OncoKB (https://www.oncokb.org/)

Key Features:
- EGFR TKI resistance (lung cancer)
- BRAF inhibitor resistance (melanoma)
- Resistance mechanism classification
- Next-generation drug recommendations

Cancer Types:
- NSCLC (Non-small cell lung cancer) - EGFR
- Melanoma - BRAF
- CRC (Colorectal cancer) - KRAS, BRAF

Clinical Relevance:
- Guides therapy selection
- Predicts treatment failure
- Informs clinical trial eligibility

Usage:
    from src.diseases.cancer_analyzer import CancerAnalyzer

    analyzer = CancerAnalyzer()
    results = analyzer.analyze(sequences, gene=CancerGene.EGFR)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

from .base import DiseaseAnalyzer, DiseaseConfig, DiseaseType, TaskType


class CancerType(Enum):
    """Cancer types with targetable mutations."""

    NSCLC = "nsclc"  # Non-small cell lung cancer
    MELANOMA = "melanoma"
    CRC = "crc"  # Colorectal cancer
    GIST = "gist"  # Gastrointestinal stromal tumor
    BREAST = "breast"
    THYROID = "thyroid"


class CancerGene(Enum):
    """Oncogenes/tumor suppressors with targetable mutations."""

    EGFR = "EGFR"
    BRAF = "BRAF"
    KRAS = "KRAS"
    NRAS = "NRAS"
    ALK = "ALK"
    ROS1 = "ROS1"
    MET = "MET"
    HER2 = "HER2"
    KIT = "KIT"
    PDGFRA = "PDGFRA"
    PIK3CA = "PIK3CA"


class TargetedTherapy(Enum):
    """Targeted therapy drugs."""

    # EGFR TKIs (generations)
    GEFITINIB = "gefitinib"  # 1st gen
    ERLOTINIB = "erlotinib"  # 1st gen
    AFATINIB = "afatinib"  # 2nd gen
    OSIMERTINIB = "osimertinib"  # 3rd gen
    LAZERTINIB = "lazertinib"  # 3rd gen

    # BRAF inhibitors
    VEMURAFENIB = "vemurafenib"
    DABRAFENIB = "dabrafenib"
    ENCORAFENIB = "encorafenib"

    # MEK inhibitors (often combined with BRAF-i)
    TRAMETINIB = "trametinib"
    COBIMETINIB = "cobimetinib"
    BINIMETINIB = "binimetinib"

    # ALK/ROS1 inhibitors
    CRIZOTINIB = "crizotinib"
    ALECTINIB = "alectinib"
    LORLATINIB = "lorlatinib"


@dataclass
class CancerConfig(DiseaseConfig):
    """Configuration for cancer analysis."""

    name: str = "cancer"
    display_name: str = "Cancer Targeted Therapy"
    disease_type: DiseaseType = DiseaseType.CANCER
    tasks: list[TaskType] = field(
        default_factory=lambda: [
            TaskType.RESISTANCE,
        ]
    )

    data_sources: dict[str, str] = field(
        default_factory=lambda: {
            "cbioportal": "https://www.cbioportal.org/",
            "cosmic": "https://cancer.sanger.ac.uk/cosmic",
            "oncokb": "https://www.oncokb.org/",
            "clinvar": "https://www.ncbi.nlm.nih.gov/clinvar/",
        }
    )


# EGFR Mutations (NSCLC)
# Sensitizing mutations that predict TKI response
EGFR_SENSITIZING = {
    # Exon 19 deletions (common)
    746: {"E": {"mutations": ["_del"], "effect": "sensitizing", "frequency": "common"}},
    # L858R (exon 21) - most common
    858: {"L": {"mutations": ["R"], "effect": "sensitizing", "frequency": "common"}},
    # Less common
    719: {"G": {"mutations": ["S", "A", "C"], "effect": "sensitizing", "frequency": "uncommon"}},
    790: {"T": {"mutations": ["M"], "effect": "resistance", "frequency": "common"}},  # Initially resistant context
}

# EGFR Resistance mutations
EGFR_RESISTANCE = {
    # T790M - "gatekeeper" mutation (1st/2nd gen TKI resistance)
    790: {"T": {"mutations": ["M"], "effect": "high", "generation": "1st_2nd", "bypass": ["osimertinib"]}},
    # C797S - osimertinib resistance
    797: {"C": {"mutations": ["S", "G"], "effect": "high", "generation": "3rd", "bypass": []}},
    # Other resistance
    792: {"G": {"mutations": ["R"], "effect": "moderate", "generation": "3rd", "bypass": []}},
    796: {"L": {"mutations": ["S"], "effect": "moderate", "generation": "3rd", "bypass": []}},
    724: {"S": {"mutations": ["P"], "effect": "moderate", "generation": "1st_2nd", "bypass": ["osimertinib"]}},
    # MET amplification bypass - detected separately
    # EGFR amplification
}

# BRAF Mutations (Melanoma, CRC)
BRAF_MUTATIONS = {
    # V600E - most common (90% of BRAF mutations)
    600: {
        "V": {"mutations": ["E"], "effect": "sensitizing", "response": "high", "drugs": ["vemurafenib", "dabrafenib"]}
    },
    # V600K - second most common
    600: {
        "V": {"mutations": ["K"], "effect": "sensitizing", "response": "high", "drugs": ["vemurafenib", "dabrafenib"]}
    },
    # Other V600 variants
    600: {
        "V": {"mutations": ["D", "R", "M"], "effect": "sensitizing", "response": "moderate", "drugs": ["dabrafenib"]}
    },
}

# BRAF Inhibitor Resistance
BRAF_RESISTANCE = {
    # Bypass through RAS/RAF/MEK pathway
    # NRAS mutations (Q61)
    # MAP2K1 (MEK1) mutations
    # BRAF amplification
    # In-target resistance (rare)
    # BRAF splice variants
}

# KRAS mutations (CRC, NSCLC, Pancreatic)
KRAS_MUTATIONS = {
    12: {"G": {"mutations": ["D", "V", "C", "S", "A", "R"], "effect": "activating", "targetable": "G12C_only"}},
    13: {"G": {"mutations": ["D", "C", "S", "R", "V"], "effect": "activating", "targetable": "limited"}},
    61: {"Q": {"mutations": ["H", "L", "R", "K"], "effect": "activating", "targetable": "no"}},
    146: {"A": {"mutations": ["T", "V"], "effect": "activating", "targetable": "no"}},
}

# ALK Resistance (NSCLC)
ALK_RESISTANCE = {
    # Crizotinib resistance
    1156: {"L": {"mutations": ["M"], "effect": "high", "drug": "crizotinib"}},
    1174: {"C": {"mutations": ["Y"], "effect": "moderate", "drug": "crizotinib"}},
    1196: {"G": {"mutations": ["M", "R"], "effect": "high", "drug": "crizotinib"}},
    1202: {"S": {"mutations": ["R"], "effect": "moderate", "drug": "alectinib"}},
    1269: {"G": {"mutations": ["A", "S"], "effect": "high", "drug": "crizotinib"}},
    # Lorlatinib resistance
    1198: {"L": {"mutations": ["F"], "effect": "high", "drug": "lorlatinib"}},
    # G1202R del - compound mutations
}


class CancerAnalyzer(DiseaseAnalyzer):
    """Analyzer for cancer targeted therapy resistance.

    Provides:
    - Sensitizing mutation detection
    - Resistance mutation detection
    - Treatment recommendations
    - Resistance mechanism classification
    """

    def __init__(self, config: CancerConfig | None = None):
        """Initialize analyzer."""
        self.config = config or CancerConfig()
        super().__init__(self.config)

        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY-X*"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_alphabet)}

    def analyze(
        self,
        sequences: dict[CancerGene, list[str]],
        cancer_type: CancerType = CancerType.NSCLC,
        embeddings: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Analyze tumor sequences for therapy response/resistance.

        Args:
            sequences: Dictionary mapping gene to sequences
            cancer_type: Cancer type
            embeddings: Optional embeddings

        Returns:
            Analysis results
        """
        results = {
            "n_sequences": len(next(iter(sequences.values()))) if sequences else 0,
            "cancer_type": cancer_type.value,
            "genes_analyzed": [g.value for g in sequences],
            "sensitizing_mutations": {},
            "resistance_mutations": {},
            "treatment_recommendations": [],
        }

        # EGFR analysis
        if CancerGene.EGFR in sequences:
            results["sensitizing_mutations"]["EGFR"] = self._analyze_egfr_sensitizing(sequences[CancerGene.EGFR])
            results["resistance_mutations"]["EGFR"] = self._analyze_egfr_resistance(sequences[CancerGene.EGFR])

        # BRAF analysis
        if CancerGene.BRAF in sequences:
            results["sensitizing_mutations"]["BRAF"] = self._analyze_braf(sequences[CancerGene.BRAF])

        # KRAS analysis
        if CancerGene.KRAS in sequences:
            results["sensitizing_mutations"]["KRAS"] = self._analyze_kras(sequences[CancerGene.KRAS])

        # ALK analysis
        if CancerGene.ALK in sequences:
            results["resistance_mutations"]["ALK"] = self._analyze_alk_resistance(sequences[CancerGene.ALK])

        # Treatment recommendations
        results["treatment_recommendations"] = self._generate_recommendations(
            results["sensitizing_mutations"], results["resistance_mutations"], cancer_type
        )

        return results

    def _analyze_egfr_sensitizing(self, sequences: list[str]) -> dict[str, Any]:
        """Analyze EGFR for sensitizing mutations."""
        results = {
            "scores": [],
            "mutations": [],
            "tkI_sensitive": [],
        }

        for seq in sequences:
            mutations = []
            sensitive = False

            for pos, info in EGFR_SENSITIZING.items():
                if pos <= len(seq):
                    seq_aa = seq[pos - 1]
                    ref_aa = list(info.keys())[0]

                    if seq_aa != ref_aa and seq_aa in info[ref_aa]["mutations"]:
                        effect = info[ref_aa]["effect"]
                        if effect == "sensitizing":
                            sensitive = True

                        mutations.append(
                            {
                                "position": pos,
                                "ref": ref_aa,
                                "alt": seq_aa,
                                "effect": effect,
                                "frequency": info[ref_aa].get("frequency", "unknown"),
                                "notation": f"EGFR {ref_aa}{pos}{seq_aa}",
                            }
                        )

            results["mutations"].append(mutations)
            results["tkI_sensitive"].append(sensitive)
            results["scores"].append(1.0 if sensitive else 0.0)

        return results

    def _analyze_egfr_resistance(self, sequences: list[str]) -> dict[str, Any]:
        """Analyze EGFR for resistance mutations."""
        results = {
            "scores": [],
            "mutations": [],
            "classifications": [],
            "bypass_options": [],
        }

        for seq in sequences:
            score = 0.0
            mutations = []
            bypasses = set()

            for pos, info in EGFR_RESISTANCE.items():
                if pos <= len(seq):
                    seq_aa = seq[pos - 1]
                    ref_aa = list(info.keys())[0]

                    if seq_aa != ref_aa and seq_aa in info[ref_aa]["mutations"]:
                        effect = info[ref_aa]["effect"]
                        effect_scores = {"high": 1.0, "moderate": 0.5, "low": 0.2}
                        score += effect_scores.get(effect, 0.3)

                        bypass = info[ref_aa].get("bypass", [])
                        bypasses.update(bypass)

                        mutations.append(
                            {
                                "position": pos,
                                "ref": ref_aa,
                                "alt": seq_aa,
                                "effect": effect,
                                "generation": info[ref_aa].get("generation", "unknown"),
                                "bypass_options": bypass,
                                "notation": f"EGFR {ref_aa}{pos}{seq_aa}",
                            }
                        )

            normalized = min(score / 2.0, 1.0)
            results["scores"].append(normalized)
            results["mutations"].append(mutations)
            results["bypass_options"].append(list(bypasses))

            if normalized < 0.2:
                results["classifications"].append("sensitive")
            elif normalized < 0.5:
                results["classifications"].append("partially_resistant")
            else:
                results["classifications"].append("resistant")

        return results

    def _analyze_braf(self, sequences: list[str]) -> dict[str, Any]:
        """Analyze BRAF mutations."""
        results = {
            "scores": [],
            "mutations": [],
            "v600_status": [],
        }

        for seq in sequences:
            mutations = []
            v600 = False

            # Check position 600
            if len(seq) >= 600:
                seq_aa = seq[599]  # 0-indexed

                if seq_aa == "E":
                    v600 = True
                    mutations.append(
                        {
                            "position": 600,
                            "ref": "V",
                            "alt": "E",
                            "effect": "sensitizing",
                            "notation": "BRAF V600E",
                            "response": "high",
                        }
                    )
                elif seq_aa == "K":
                    v600 = True
                    mutations.append(
                        {
                            "position": 600,
                            "ref": "V",
                            "alt": "K",
                            "effect": "sensitizing",
                            "notation": "BRAF V600K",
                            "response": "high",
                        }
                    )
                elif seq_aa in ["D", "R", "M"]:
                    v600 = True
                    mutations.append(
                        {
                            "position": 600,
                            "ref": "V",
                            "alt": seq_aa,
                            "effect": "sensitizing",
                            "notation": f"BRAF V600{seq_aa}",
                            "response": "moderate",
                        }
                    )

            results["mutations"].append(mutations)
            results["v600_status"].append(v600)
            results["scores"].append(1.0 if v600 else 0.0)

        return results

    def _analyze_kras(self, sequences: list[str]) -> dict[str, Any]:
        """Analyze KRAS mutations."""
        results = {
            "scores": [],
            "mutations": [],
            "g12c_status": [],
            "targetable": [],
        }

        for seq in sequences:
            mutations = []
            g12c = False
            targetable = False

            for pos, info in KRAS_MUTATIONS.items():
                if pos <= len(seq):
                    seq_aa = seq[pos - 1]
                    ref_aa = list(info.keys())[0]

                    if seq_aa != ref_aa and seq_aa in info[ref_aa]["mutations"]:
                        is_g12c = pos == 12 and seq_aa == "C"
                        if is_g12c:
                            g12c = True
                            targetable = True

                        mutations.append(
                            {
                                "position": pos,
                                "ref": ref_aa,
                                "alt": seq_aa,
                                "effect": info[ref_aa]["effect"],
                                "notation": f"KRAS {ref_aa}{pos}{seq_aa}",
                                "targetable": "sotorasib/adagrasib" if is_g12c else info[ref_aa]["targetable"],
                            }
                        )

            results["mutations"].append(mutations)
            results["g12c_status"].append(g12c)
            results["targetable"].append(targetable)
            results["scores"].append(1.0 if mutations else 0.0)

        return results

    def _analyze_alk_resistance(self, sequences: list[str]) -> dict[str, Any]:
        """Analyze ALK resistance mutations."""
        results = {
            "scores": [],
            "mutations": [],
            "classifications": [],
        }

        for seq in sequences:
            score = 0.0
            mutations = []

            for pos, info in ALK_RESISTANCE.items():
                if pos <= len(seq):
                    seq_aa = seq[pos - 1]
                    ref_aa = list(info.keys())[0]

                    if seq_aa != ref_aa and seq_aa in info[ref_aa]["mutations"]:
                        effect = info[ref_aa]["effect"]
                        effect_scores = {"high": 1.0, "moderate": 0.5, "low": 0.2}
                        score += effect_scores.get(effect, 0.3)

                        mutations.append(
                            {
                                "position": pos,
                                "ref": ref_aa,
                                "alt": seq_aa,
                                "effect": effect,
                                "resistant_to": info[ref_aa].get("drug", "unknown"),
                                "notation": f"ALK {ref_aa}{pos}{seq_aa}",
                            }
                        )

            normalized = min(score / 2.0, 1.0)
            results["scores"].append(normalized)
            results["mutations"].append(mutations)

            if normalized < 0.2:
                results["classifications"].append("sensitive")
            else:
                results["classifications"].append("resistant")

        return results

    def _generate_recommendations(
        self,
        sensitizing: dict,
        resistance: dict,
        cancer_type: CancerType,
    ) -> list[dict[str, Any]]:
        """Generate treatment recommendations."""
        recommendations = []

        # Determine number of samples
        n_seq = 0
        for gene_data in sensitizing.values():
            if "scores" in gene_data:
                n_seq = max(n_seq, len(gene_data["scores"]))

        for i in range(n_seq):
            rec = {
                "sample": i,
                "recommended_therapies": [],
                "avoid_therapies": [],
                "clinical_trial_eligible": [],
                "rationale": [],
            }

            # EGFR-mutant NSCLC
            if "EGFR" in sensitizing:
                egfr_sens = sensitizing["EGFR"]
                egfr_res = resistance.get("EGFR", {})

                if i < len(egfr_sens.get("tkI_sensitive", [])) and egfr_sens["tkI_sensitive"][i]:
                    # Check for resistance
                    if i < len(egfr_res.get("classifications", [])):
                        if egfr_res["classifications"][i] == "sensitive":
                            rec["recommended_therapies"].append("osimertinib (1st line)")
                            rec["rationale"].append("EGFR sensitizing mutation, no resistance detected")
                        elif "T790M" in str(egfr_res.get("mutations", [[]])[i]):
                            rec["recommended_therapies"].append("osimertinib")
                            rec["avoid_therapies"].append("1st/2nd gen EGFR TKIs")
                            rec["rationale"].append("T790M resistance - use 3rd gen TKI")
                        elif "C797S" in str(egfr_res.get("mutations", [[]])[i]):
                            rec["avoid_therapies"].append("osimertinib")
                            rec["clinical_trial_eligible"].append("4th gen EGFR TKI trials")
                            rec["rationale"].append("C797S resistance - osimertinib ineffective")

            # BRAF-mutant
            if "BRAF" in sensitizing:
                braf_data = sensitizing["BRAF"]
                if i < len(braf_data.get("v600_status", [])) and braf_data["v600_status"][i]:
                    if cancer_type == CancerType.MELANOMA:
                        rec["recommended_therapies"].append("dabrafenib + trametinib")
                        rec["rationale"].append("BRAF V600 mutation - BRAF/MEK inhibition")
                    elif cancer_type == CancerType.CRC:
                        rec["recommended_therapies"].append("encorafenib + cetuximab")
                        rec["rationale"].append("BRAF V600E CRC - different regimen than melanoma")

            # KRAS G12C
            if "KRAS" in sensitizing:
                kras_data = sensitizing["KRAS"]
                if i < len(kras_data.get("g12c_status", [])) and kras_data["g12c_status"][i]:
                    rec["recommended_therapies"].append("sotorasib or adagrasib")
                    rec["rationale"].append("KRAS G12C - targetable with covalent inhibitors")

            recommendations.append(rec)

        return recommendations

    def validate_predictions(
        self,
        predictions: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> dict[str, float]:
        """Validate against clinical response data."""
        from scipy.stats import spearmanr

        metrics = {}

        for gene in predictions.get("resistance_mutations", {}):
            if gene in ground_truth:
                pred = np.array(predictions["resistance_mutations"][gene]["scores"])
                true = np.array(ground_truth[gene])

                if len(pred) == len(true):
                    rho, pval = spearmanr(pred, true)
                    metrics[f"{gene}_spearman"] = float(rho) if not np.isnan(rho) else 0.0

        return metrics

    def encode_sequence(
        self,
        sequence: str,
        max_length: int = 1000,
    ) -> np.ndarray:
        """One-hot encode sequence."""
        n_aa = len(self.aa_alphabet)
        encoding = np.zeros(max_length * n_aa, dtype=np.float32)

        for j, aa in enumerate(sequence[:max_length]):
            idx = self.aa_to_idx.get(aa.upper(), self.aa_to_idx["X"])
            encoding[j * n_aa + idx] = 1.0

        return encoding


def create_cancer_synthetic_dataset(
    gene: CancerGene = CancerGene.EGFR,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create synthetic cancer dataset."""
    reference = "M" + "A" * 999

    if gene == CancerGene.EGFR:
        mutation_db = EGFR_RESISTANCE
    elif gene == CancerGene.ALK:
        mutation_db = ALK_RESISTANCE
    else:
        mutation_db = KRAS_MUTATIONS

    sequences = [reference]
    resistances = [0.0]
    ids = ["WT"]

    for pos, info in mutation_db.items():
        if pos <= len(reference):
            ref_aa = list(info.keys())[0]
            for mut_aa in info[ref_aa]["mutations"][:2]:
                mutant = list(reference)
                mutant[pos - 1] = mut_aa
                sequences.append("".join(mutant))

                effect = info[ref_aa].get("effect", "moderate")
                effect_scores = {
                    "high": 0.9,
                    "moderate": 0.5,
                    "low": 0.2,
                    "activating": 0.8,
                    "sensitizing": 0.7,
                    "resistance": 0.9,
                }
                resistances.append(effect_scores.get(effect, 0.5))
                ids.append(f"{gene.value}_{ref_aa}{pos}{mut_aa}")

    analyzer = CancerAnalyzer()
    X = np.array([analyzer.encode_sequence(s) for s in sequences])
    y = np.array(resistances, dtype=np.float32)

    return X, y, ids
