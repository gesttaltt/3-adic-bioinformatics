# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tuberculosis Multi-Drug Resistance (MDR-TB) Analyzer.

This module provides comprehensive analysis of Mycobacterium tuberculosis
drug resistance mutations for 13 first and second-line TB drugs.

Based on WHO TB Mutation Catalogue (2021, 2023):
https://www.who.int/publications/i/item/9789240028173

Key Features:
- Drug resistance prediction for 13 drugs
- MDR-TB and XDR-TB classification
- Transfer learning from HIV resistance patterns
- P-adic encoding for mutation distance

Drug Classes:
1. First-line: Rifampicin, Isoniazid, Ethambutol, Pyrazinamide
2. Fluoroquinolones: Levofloxacin, Moxifloxacin
3. Second-line injectables: Amikacin, Capreomycin, Kanamycin
4. Newer drugs: Bedaquiline, Linezolid, Clofazimine, Delamanid

Usage:
    from src.diseases.tuberculosis_analyzer import TuberculosisAnalyzer

    analyzer = TuberculosisAnalyzer()
    results = analyzer.analyze_resistance(sequences, drug="rifampicin")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

from .base import DiseaseAnalyzer, DiseaseConfig, DiseaseType, TaskType


class TBDrug(Enum):
    """TB drugs with resistance data available."""

    # First-line drugs
    RIFAMPICIN = "RIF"
    ISONIAZID = "INH"
    ETHAMBUTOL = "EMB"
    PYRAZINAMIDE = "PZA"

    # Fluoroquinolones
    LEVOFLOXACIN = "LFX"
    MOXIFLOXACIN = "MFX"

    # Second-line injectables
    AMIKACIN = "AMK"
    CAPREOMYCIN = "CAP"
    KANAMYCIN = "KAN"

    # Newer/critical drugs
    BEDAQUILINE = "BDQ"
    LINEZOLID = "LZD"
    CLOFAZIMINE = "CFZ"
    DELAMANID = "DLM"


class TBDrugCategory(Enum):
    """WHO TB drug categories."""

    FIRST_LINE = "first_line"
    FLUOROQUINOLONES = "fluoroquinolones"
    SECOND_LINE_INJECTABLE = "second_line_injectable"
    NEWER_AGENTS = "newer_agents"

    @classmethod
    def get_category(cls, drug: TBDrug) -> TBDrugCategory:
        """Get category for a drug."""
        first_line = {TBDrug.RIFAMPICIN, TBDrug.ISONIAZID, TBDrug.ETHAMBUTOL, TBDrug.PYRAZINAMIDE}
        fqs = {TBDrug.LEVOFLOXACIN, TBDrug.MOXIFLOXACIN}
        injectables = {TBDrug.AMIKACIN, TBDrug.CAPREOMYCIN, TBDrug.KANAMYCIN}

        if drug in first_line:
            return cls.FIRST_LINE
        elif drug in fqs:
            return cls.FLUOROQUINOLONES
        elif drug in injectables:
            return cls.SECOND_LINE_INJECTABLE
        else:
            return cls.NEWER_AGENTS


class TBGene(Enum):
    """TB genes associated with drug resistance."""

    # Rifampicin resistance
    RPOB = "rpoB"

    # Isoniazid resistance
    KATG = "katG"
    INHA = "inhA"
    FABG1 = "fabG1"
    AHPC = "ahpC"

    # Ethambutol resistance
    EMBB = "embB"
    EMBA = "embA"
    EMBC = "embC"

    # Pyrazinamide resistance
    PNCA = "pncA"
    RPSA = "rpsA"
    PAND = "panD"

    # Fluoroquinolone resistance
    GYRA = "gyrA"
    GYRB = "gyrB"

    # Injectable resistance
    RRS = "rrs"
    EIS = "eis"

    # Bedaquiline/Clofazimine resistance
    ATPE = "atpE"
    RV0678 = "Rv0678"
    PEPQ = "pepQ"

    # Linezolid resistance
    RPL3 = "rpl3"
    RRL = "rrl"

    # Delamanid resistance
    DDN = "ddn"
    FGD1 = "fgd1"
    FBIA = "fbiA"
    FBIB = "fbiB"
    FBIC = "fbiC"


class ResistanceLevel(Enum):
    """WHO resistance classification levels."""

    SUSCEPTIBLE = "S"
    RESISTANT = "R"
    UNCERTAIN = "U"
    NOT_EVALUABLE = "N"


@dataclass
class TuberculosisConfig(DiseaseConfig):
    """Configuration for TB analysis."""

    name: str = "tuberculosis"
    display_name: str = "Tuberculosis (M. tuberculosis)"
    disease_type: DiseaseType = DiseaseType.BACTERIAL
    tasks: list[TaskType] = field(
        default_factory=lambda: [
            TaskType.RESISTANCE,
        ]
    )

    # Data sources
    data_sources: dict[str, str] = field(
        default_factory=lambda: {
            "who_catalogue": "https://www.who.int/publications/i/item/9789240028173",
            "cryptic": "https://www.crypticproject.org/",
            "tbportals": "https://tbportals.niaid.nih.gov/",
            "patric": "https://www.bv-brc.org/",
        }
    )


# WHO TB Mutation Catalogue - Key mutations by gene
# Format: {position: {ref_aa: mutation_info}}

RPOB_MUTATIONS = {
    # Rifampicin resistance-determining region (RRDR): codons 426-452
    426: {"D": {"mutations": ["Y", "N"], "effect": "high"}},
    430: {"L": {"mutations": ["P"], "effect": "high"}},
    431: {"S": {"mutations": ["T", "G"], "effect": "low"}},
    432: {"Q": {"mutations": ["K", "L", "P"], "effect": "high"}},
    434: {"M": {"mutations": ["I", "V"], "effect": "low"}},
    435: {"H": {"mutations": ["Y", "R", "D", "C", "L", "N", "P", "Q"], "effect": "high"}},
    437: {"Q": {"mutations": ["L", "R"], "effect": "low"}},
    441: {"L": {"mutations": ["V"], "effect": "low"}},
    445: {"H": {"mutations": ["D", "Y", "L", "R", "N", "C", "Q", "S"], "effect": "high"}},
    450: {"S": {"mutations": ["L", "W", "Q"], "effect": "high"}},
    451: {"L": {"mutations": ["P"], "effect": "high"}},
    452: {"I": {"mutations": ["L"], "effect": "low"}},
}

KATG_MUTATIONS = {
    # Isoniazid resistance via catalase-peroxidase
    315: {"S": {"mutations": ["T", "N", "I", "G", "R"], "effect": "high"}},
    463: {"R": {"mutations": ["L"], "effect": "low"}},
    270: {"W": {"mutations": ["G"], "effect": "high"}},
    275: {"W": {"mutations": ["S"], "effect": "moderate"}},
    291: {"G": {"mutations": ["S"], "effect": "moderate"}},
    300: {"W": {"mutations": ["C", "G"], "effect": "high"}},
    311: {"Y": {"mutations": ["C"], "effect": "moderate"}},
    381: {"A": {"mutations": ["V"], "effect": "low"}},
}

INHA_PROMOTER_MUTATIONS = {
    # inhA promoter mutations (positions relative to start codon)
    -15: {"C": {"mutations": ["T"], "effect": "moderate"}},
    -8: {"T": {"mutations": ["C", "A"], "effect": "moderate"}},
    -17: {"G": {"mutations": ["T"], "effect": "low"}},
}

EMBB_MUTATIONS = {
    # Ethambutol resistance
    306: {"M": {"mutations": ["V", "I", "L"], "effect": "high"}},
    354: {"D": {"mutations": ["A"], "effect": "moderate"}},
    406: {"G": {"mutations": ["A", "S", "D", "C"], "effect": "moderate"}},
    497: {"Q": {"mutations": ["R"], "effect": "low"}},
}

GYRA_MUTATIONS = {
    # Fluoroquinolone resistance (QRDR)
    88: {"A": {"mutations": ["V", "G"], "effect": "high"}},
    90: {"A": {"mutations": ["V"], "effect": "high"}},
    91: {"S": {"mutations": ["P"], "effect": "high"}},
    94: {"D": {"mutations": ["G", "A", "N", "Y", "H"], "effect": "high"}},
}

GYRB_MUTATIONS = {
    # Additional FQ resistance
    461: {"A": {"mutations": ["V"], "effect": "moderate"}},
    499: {"N": {"mutations": ["D", "T"], "effect": "high"}},
    500: {"E": {"mutations": ["V", "D"], "effect": "moderate"}},
}

RRS_MUTATIONS = {
    # Aminoglycoside/injectable resistance (16S rRNA)
    1401: {"A": {"mutations": ["G"], "effect": "high"}},
    1402: {"C": {"mutations": ["T"], "effect": "high"}},
    1484: {"G": {"mutations": ["T"], "effect": "high"}},
}

ATPE_MUTATIONS = {
    # Bedaquiline resistance (ATP synthase)
    28: {"A": {"mutations": ["V", "P"], "effect": "high"}},
    63: {"E": {"mutations": ["D"], "effect": "moderate"}},
    66: {"I": {"mutations": ["M"], "effect": "high"}},
}

RV0678_MUTATIONS = {
    # Bedaquiline/Clofazimine cross-resistance
    # (Many frameshift/nonsense mutations - simplified)
    1: {"M": {"mutations": ["*"], "effect": "high"}},  # Start codon loss
}

PNCA_MUTATIONS = {
    # Pyrazinamide resistance
    # High diversity - most non-synonymous mutations cause resistance
    3: {"D": {"mutations": ["A", "H", "N", "G"], "effect": "high"}},
    8: {"L": {"mutations": ["P", "R"], "effect": "high"}},
    46: {"H": {"mutations": ["D", "R", "Y"], "effect": "high"}},
    47: {"D": {"mutations": ["N", "V", "G"], "effect": "high"}},
    57: {"G": {"mutations": ["D"], "effect": "high"}},
    71: {"H": {"mutations": ["Y", "D", "R"], "effect": "high"}},
    139: {"K": {"mutations": ["E"], "effect": "high"}},
}


# Drug to gene mapping
DRUG_GENE_MAP = {
    TBDrug.RIFAMPICIN: [TBGene.RPOB],
    TBDrug.ISONIAZID: [TBGene.KATG, TBGene.INHA, TBGene.FABG1, TBGene.AHPC],
    TBDrug.ETHAMBUTOL: [TBGene.EMBB, TBGene.EMBA, TBGene.EMBC],
    TBDrug.PYRAZINAMIDE: [TBGene.PNCA, TBGene.RPSA, TBGene.PAND],
    TBDrug.LEVOFLOXACIN: [TBGene.GYRA, TBGene.GYRB],
    TBDrug.MOXIFLOXACIN: [TBGene.GYRA, TBGene.GYRB],
    TBDrug.AMIKACIN: [TBGene.RRS, TBGene.EIS],
    TBDrug.CAPREOMYCIN: [TBGene.RRS],
    TBDrug.KANAMYCIN: [TBGene.RRS, TBGene.EIS],
    TBDrug.BEDAQUILINE: [TBGene.ATPE, TBGene.RV0678, TBGene.PEPQ],
    TBDrug.LINEZOLID: [TBGene.RPL3, TBGene.RRL],
    TBDrug.CLOFAZIMINE: [TBGene.RV0678],
    TBDrug.DELAMANID: [TBGene.DDN, TBGene.FGD1, TBGene.FBIA, TBGene.FBIB, TBGene.FBIC],
}

# Gene to mutation database mapping
GENE_MUTATION_DB = {
    TBGene.RPOB: RPOB_MUTATIONS,
    TBGene.KATG: KATG_MUTATIONS,
    TBGene.EMBB: EMBB_MUTATIONS,
    TBGene.GYRA: GYRA_MUTATIONS,
    TBGene.GYRB: GYRB_MUTATIONS,
    TBGene.RRS: RRS_MUTATIONS,
    TBGene.ATPE: ATPE_MUTATIONS,
    TBGene.RV0678: RV0678_MUTATIONS,
    TBGene.PNCA: PNCA_MUTATIONS,
}

# Aliases for backward compatibility
GENE_MUTATIONS = GENE_MUTATION_DB
DRUG_TO_GENE = DRUG_GENE_MAP


class TuberculosisAnalyzer(DiseaseAnalyzer):
    """Analyzer for M. tuberculosis drug resistance.

    Provides:
    - Individual drug resistance prediction
    - MDR-TB detection (RIF + INH resistance)
    - XDR-TB detection (MDR + FQ + injectable)
    - Pre-XDR-TB detection
    - Mutation cataloguing
    """

    def __init__(self, config: TuberculosisConfig | None = None):
        """Initialize analyzer.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or TuberculosisConfig()
        super().__init__(self.config)

        # Amino acid encoding
        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY-X*"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_alphabet)}

    def analyze(
        self,
        sequences: dict[TBGene, list[str]],
        embeddings: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Analyze TB gene sequences for drug resistance.

        Args:
            sequences: Dictionary mapping gene to list of sequences
            embeddings: Optional precomputed embeddings
            **kwargs: Additional parameters

        Returns:
            Analysis results dictionary
        """
        results = {
            "n_isolates": len(next(iter(sequences.values()))) if sequences else 0,
            "genes_analyzed": list(sequences.keys()),
            "drug_resistance": {},
            "mdr_classification": [],
            "mutations_detected": {},
        }

        # Analyze each drug
        for drug in TBDrug:
            drug_results = self.predict_drug_resistance(sequences, drug)
            results["drug_resistance"][drug.value] = drug_results

        # Classify MDR/XDR
        results["mdr_classification"] = self._classify_mdr_xdr(results["drug_resistance"])

        return results

    def predict_drug_resistance(
        self,
        sequences: dict[TBGene, list[str]],
        drug: TBDrug,
    ) -> dict[str, Any]:
        """Predict resistance for a specific drug.

        Args:
            sequences: Gene sequences
            drug: Target drug

        Returns:
            Resistance predictions
        """
        results = {
            "drug": drug.value,
            "relevant_genes": [g.value for g in DRUG_GENE_MAP.get(drug, [])],
            "scores": [],
            "classifications": [],
            "mutations": [],
        }

        # Get relevant genes
        relevant_genes = DRUG_GENE_MAP.get(drug, [])

        # Determine number of isolates
        n_isolates = 0
        for gene in relevant_genes:
            if gene in sequences:
                n_isolates = max(n_isolates, len(sequences[gene]))

        if n_isolates == 0:
            return results

        # Analyze each isolate
        for i in range(n_isolates):
            isolate_score = 0.0
            isolate_mutations = []

            for gene in relevant_genes:
                if gene not in sequences or i >= len(sequences[gene]):
                    continue

                seq = sequences[gene][i]
                mutation_db = GENE_MUTATION_DB.get(gene, {})

                # Check each known resistance position
                for pos, info in mutation_db.items():
                    if pos <= 0 or pos > len(seq):
                        continue

                    seq_aa = seq[pos - 1] if pos <= len(seq) else "-"
                    ref_aa = list(info.keys())[0]

                    if seq_aa != ref_aa and seq_aa in info[ref_aa]["mutations"]:
                        effect = info[ref_aa]["effect"]
                        effect_scores = {"high": 1.0, "moderate": 0.6, "low": 0.3}
                        isolate_score += effect_scores.get(effect, 0.5)
                        isolate_mutations.append(
                            {
                                "gene": gene.value,
                                "position": pos,
                                "ref": ref_aa,
                                "alt": seq_aa,
                                "effect": effect,
                                "notation": f"{gene.value}_{ref_aa}{pos}{seq_aa}",
                            }
                        )

            # Normalize score
            max_score = 5.0
            normalized = min(isolate_score / max_score, 1.0)

            results["scores"].append(normalized)
            results["mutations"].append(isolate_mutations)

            # Classification
            if normalized < 0.1:
                classification = ResistanceLevel.SUSCEPTIBLE
            elif normalized < 0.3:
                classification = ResistanceLevel.UNCERTAIN
            else:
                classification = ResistanceLevel.RESISTANT

            results["classifications"].append(classification.value)

        return results

    def _classify_mdr_xdr(self, drug_resistance: dict[str, dict]) -> list[dict[str, Any]]:
        """Classify isolates as MDR, pre-XDR, or XDR.

        Definitions (WHO 2021):
        - MDR-TB: Resistant to RIF + INH
        - Pre-XDR-TB: MDR + FQ resistant
        - XDR-TB: Pre-XDR + BDQ or LZD resistant
        """
        classifications = []

        n_isolates = len(drug_resistance.get("RIF", {}).get("scores", []))

        for i in range(n_isolates):
            rif_score = (
                drug_resistance.get("RIF", {}).get("scores", [0])[i]
                if i < len(drug_resistance.get("RIF", {}).get("scores", []))
                else 0
            )
            inh_score = (
                drug_resistance.get("INH", {}).get("scores", [0])[i]
                if i < len(drug_resistance.get("INH", {}).get("scores", []))
                else 0
            )
            lfx_score = (
                drug_resistance.get("LFX", {}).get("scores", [0])[i]
                if i < len(drug_resistance.get("LFX", {}).get("scores", []))
                else 0
            )
            mfx_score = (
                drug_resistance.get("MFX", {}).get("scores", [0])[i]
                if i < len(drug_resistance.get("MFX", {}).get("scores", []))
                else 0
            )
            bdq_score = (
                drug_resistance.get("BDQ", {}).get("scores", [0])[i]
                if i < len(drug_resistance.get("BDQ", {}).get("scores", []))
                else 0
            )
            lzd_score = (
                drug_resistance.get("LZD", {}).get("scores", [0])[i]
                if i < len(drug_resistance.get("LZD", {}).get("scores", []))
                else 0
            )

            # Thresholds
            R_THRESHOLD = 0.3

            rif_r = rif_score >= R_THRESHOLD
            inh_r = inh_score >= R_THRESHOLD
            fq_r = max(lfx_score, mfx_score) >= R_THRESHOLD
            group_a_r = max(bdq_score, lzd_score) >= R_THRESHOLD

            if rif_r and inh_r and fq_r and group_a_r:
                classification = "XDR-TB"
            elif rif_r and inh_r and fq_r:
                classification = "pre-XDR-TB"
            elif rif_r and inh_r:
                classification = "MDR-TB"
            elif rif_r:
                classification = "RR-TB"  # Rifampicin-resistant
            else:
                classification = "DS-TB"  # Drug-susceptible

            classifications.append(
                {
                    "isolate": i,
                    "classification": classification,
                    "rif_resistant": rif_r,
                    "inh_resistant": inh_r,
                    "fq_resistant": fq_r,
                    "group_a_resistant": group_a_r,
                }
            )

        return classifications

    def validate_predictions(
        self,
        predictions: dict[str, torch.Tensor],
        ground_truth: dict[str, Any],
    ) -> dict[str, float]:
        """Validate predictions against phenotypic DST."""
        from scipy.stats import spearmanr

        metrics = {}

        for drug in predictions:
            if drug in ground_truth:
                pred = predictions[drug].numpy() if isinstance(predictions[drug], torch.Tensor) else predictions[drug]
                true = np.array(ground_truth[drug])

                rho, pval = spearmanr(pred, true)
                metrics[f"{drug}_spearman"] = float(rho) if not np.isnan(rho) else 0.0
                metrics[f"{drug}_pvalue"] = float(pval)

        return metrics

    def encode_gene_sequence(
        self,
        sequence: str,
        max_length: int | None = None,
    ) -> np.ndarray:
        """One-hot encode a gene sequence."""
        if max_length is None:
            max_length = len(sequence)

        n_aa = len(self.aa_alphabet)
        encoding = np.zeros(max_length * n_aa, dtype=np.float32)

        for j, aa in enumerate(sequence[:max_length]):
            idx = self.aa_to_idx.get(aa.upper(), self.aa_to_idx["X"])
            encoding[j * n_aa + idx] = 1.0

        return encoding


def create_tb_synthetic_dataset(
    drug: TBDrug = TBDrug.RIFAMPICIN,
    min_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create synthetic TB dataset for testing.

    In production, use WHO catalogue or CRyPTIC data.

    Args:
        drug: Target TB drug for resistance prediction
        min_samples: Minimum number of samples to generate

    Returns:
        (X, y, mutation_ids)
    """
    from src.diseases.utils.synthetic_data import (
        create_mutation_based_dataset,
        ensure_minimum_samples,
    )

    # TB genes are large - need sequences long enough to cover mutation positions
    # rpoB RRDR region: positions 426-452 (need ~500 AA)
    # katG: positions up to 463 (need ~500 AA)
    max_pos = 500

    if drug == TBDrug.RIFAMPICIN:
        # Create reference sequence long enough for rpoB RRDR (positions 426-452)
        reference = "M" + "A" * (max_pos - 1)  # 500 AA sequence
        mutation_db = RPOB_MUTATIONS
    elif drug == TBDrug.ISONIAZID:
        # katG mutations go up to position 463
        reference = "M" + "A" * (max_pos - 1)
        mutation_db = KATG_MUTATIONS
    elif drug == TBDrug.ETHAMBUTOL:
        reference = "M" + "A" * (max_pos - 1)
        mutation_db = EMBB_MUTATIONS
    elif drug in [TBDrug.LEVOFLOXACIN, TBDrug.MOXIFLOXACIN]:
        reference = "M" + "A" * (max_pos - 1)
        mutation_db = {**GYRA_MUTATIONS, **GYRB_MUTATIONS}
    elif drug == TBDrug.PYRAZINAMIDE:
        reference = "M" + "A" * (max_pos - 1)
        mutation_db = PNCA_MUTATIONS
    elif drug in [TBDrug.AMIKACIN, TBDrug.KANAMYCIN, TBDrug.CAPREOMYCIN]:
        # rrs mutations at positions 1401-1484 (16S rRNA)
        # For protein-level, use reasonable length
        reference = "M" + "A" * (max_pos - 1)
        mutation_db = RRS_MUTATIONS
    elif drug == TBDrug.BEDAQUILINE:
        reference = "M" + "A" * (max_pos - 1)
        mutation_db = {**ATPE_MUTATIONS, **RV0678_MUTATIONS}
    else:
        reference = "M" + "A" * (max_pos - 1)
        mutation_db = RPOB_MUTATIONS

    analyzer = TuberculosisAnalyzer()

    # Use new synthetic data utilities
    # encode_fn signature: (sequence, max_length) -> np.ndarray
    X, y, ids = create_mutation_based_dataset(
        reference_sequence=reference,
        mutation_db=mutation_db,
        encode_fn=lambda s, ml: analyzer.encode_gene_sequence(s, max_length=ml),
        max_length=max_pos,
        n_random_mutants=30,
        seed=42,
    )

    # Ensure minimum samples
    X, y, ids = ensure_minimum_samples(X, y, ids, min_samples=min_samples, seed=42)

    return X, y, ids
