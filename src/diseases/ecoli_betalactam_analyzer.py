# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""E. coli TEM β-lactamase Resistance Analyzer.

This module provides analysis of TEM β-lactamase variants in E. coli,
the simplest and most well-characterized antibiotic resistance mechanism.

Why TEM β-lactamase?
- Single gene (blaTEM) encodes the enzyme
- Only 4-5 key positions determine ESBL phenotype
- 90% of ampicillin resistance in E. coli comes from TEM-1
- Decades of research with clear structure-function relationships
- Excellent data availability (Arcadia 7K dataset, CARD)

Variant Classification:
- TEM-1: Original β-lactamase (penicillin resistance only)
- ESBL: Extended-spectrum (cephalosporin resistance)
- IRT: Inhibitor-resistant (clavulanate resistance)
- CMT: Complex mutant (ESBL + IRT)

Data Sources:
- Arcadia Science: 7,000+ E. coli strains with MIC data
- CARD Database: TEM variant sequences and annotations
- Literature: Well-documented mutation effects

Usage:
    from src.diseases.ecoli_betalactam_analyzer import EcoliBetaLactamAnalyzer

    analyzer = EcoliBetaLactamAnalyzer()
    results = analyzer.analyze(sequences, drug=BetaLactam.AMPICILLIN)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

from .base import DiseaseAnalyzer, DiseaseConfig, DiseaseType, TaskType


class EcoliGene(Enum):
    """E. coli genes associated with β-lactam resistance."""

    BLA_TEM = "blaTEM"  # TEM β-lactamase (primary)
    BLA_SHV = "blaSHV"  # SHV β-lactamase (similar mechanism)
    BLA_CTX_M = "blaCTX-M"  # CTX-M β-lactamase (cefotaximase)


class BetaLactam(Enum):
    """β-lactam antibiotics for resistance prediction."""

    # Penicillins (TEM-1 confers resistance)
    AMPICILLIN = "ampicillin"
    AMOXICILLIN = "amoxicillin"

    # Cephalosporins (ESBL mutations required)
    CEFTAZIDIME = "ceftazidime"  # 3rd gen - indicator for ESBL
    CEFOTAXIME = "cefotaxime"  # 3rd gen
    CEFTRIAXONE = "ceftriaxone"  # 3rd gen
    CEFEPIME = "cefepime"  # 4th gen

    # β-lactamase inhibitor combinations
    AMOX_CLAVULANATE = "amoxicillin_clavulanate"
    PIPERACILLIN_TAZO = "piperacillin_tazobactam"


class TEMVariant(Enum):
    """TEM β-lactamase variant classification."""

    TEM_1 = "TEM-1"  # Original, penicillin only
    ESBL = "ESBL"  # Extended-spectrum
    IRT = "IRT"  # Inhibitor-resistant
    CMT = "CMT"  # Complex mutant (ESBL + IRT)


@dataclass
class EcoliBetaLactamConfig(DiseaseConfig):
    """Configuration for E. coli β-lactam resistance analysis."""

    name: str = "ecoli_betalactam"
    display_name: str = "E. coli TEM β-lactamase"
    disease_type: DiseaseType = DiseaseType.BACTERIAL
    tasks: list[TaskType] = field(default_factory=lambda: [TaskType.RESISTANCE])

    data_sources: dict[str, str] = field(
        default_factory=lambda: {
            "arcadia": "https://github.com/Arcadia-Science/2024-Ecoli-amr-genotype-phenotype_7000strains",
            "card": "https://card.mcmaster.ca/",
            "ncbi_amr": "https://www.ncbi.nlm.nih.gov/pathogens/antimicrobial-resistance/AMRFinder/",
            "zenodo": "https://zenodo.org/records/12692732",
        }
    )


# TEM β-lactamase mutation database
# Based on literature and CARD annotations
# Position numbering follows Ambler standard (mature protein)

TEM_MUTATIONS = {
    # ESBL-conferring mutations (extend spectrum to cephalosporins)
    # These mutations widen the active site to accommodate bulky cephalosporins
    104: {"E": {"mutations": ["K"], "effect": "high", "phenotype": "ESBL", "note": "Widens active site entrance"}},
    164: {"R": {"mutations": ["S", "H"], "effect": "high", "phenotype": "ESBL", "note": "Omega loop flexibility"}},
    238: {
        "G": {"mutations": ["S"], "effect": "high", "phenotype": "ESBL", "note": "Key ESBL mutation, found in TEM-15"}
    },
    240: {"E": {"mutations": ["K"], "effect": "high", "phenotype": "ESBL", "note": "Often co-occurs with G238S"}},
    # Inhibitor resistance mutations (IRT - clavulanate resistance)
    # These mutations reduce binding of β-lactamase inhibitors
    69: {
        "M": {
            "mutations": ["I", "L", "V"],
            "effect": "moderate",
            "phenotype": "IRT",
            "note": "Reduces inhibitor binding",
        }
    },
    130: {
        "S": {"mutations": ["G"], "effect": "moderate", "phenotype": "IRT", "note": "Serine to glycine, IRT mechanism"}
    },
    244: {"R": {"mutations": ["S", "C", "H"], "effect": "moderate", "phenotype": "IRT", "note": "Common IRT mutation"}},
    275: {
        "R": {"mutations": ["L", "Q"], "effect": "moderate", "phenotype": "IRT", "note": "Reduces clavulanate efficacy"}
    },
    276: {"N": {"mutations": ["D"], "effect": "moderate", "phenotype": "IRT", "note": "Adjacent to active site"}},
    # Stabilizing mutations (enable other mutations by restoring folding)
    182: {
        "M": {
            "mutations": ["T"],
            "effect": "low",
            "phenotype": "stabilizer",
            "note": "Global suppressor, restores stability",
        }
    },
    39: {
        "Q": {
            "mutations": ["K"],
            "effect": "low",
            "phenotype": "stabilizer",
            "note": "Enables acquisition of other mutations",
        }
    },
}

# Drug-phenotype mapping
# Which mutations confer resistance to which drugs
DRUG_PHENOTYPE_MAP = {
    BetaLactam.AMPICILLIN: ["TEM-1", "ESBL", "IRT", "CMT"],  # TEM-1 is enough
    BetaLactam.AMOXICILLIN: ["TEM-1", "ESBL", "IRT", "CMT"],
    BetaLactam.CEFTAZIDIME: ["ESBL", "CMT"],  # Need ESBL mutations
    BetaLactam.CEFOTAXIME: ["ESBL", "CMT"],
    BetaLactam.CEFTRIAXONE: ["ESBL", "CMT"],
    BetaLactam.CEFEPIME: ["ESBL", "CMT"],
    BetaLactam.AMOX_CLAVULANATE: ["IRT", "CMT"],  # Need IRT mutations
    BetaLactam.PIPERACILLIN_TAZO: ["IRT", "CMT"],
}

# TEM-1 reference sequence (mature protein, 263 amino acids)
# UniProt: P62593, Ambler numbering
TEM1_REFERENCE = (
    "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRP"
    "EERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVREL"
    "CSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTM"
    "PAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGS"
    "RGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
)


class EcoliBetaLactamAnalyzer(DiseaseAnalyzer):
    """Analyzer for E. coli TEM β-lactamase resistance.

    This is the simplest disease analyzer, designed for:
    - Framework validation
    - Clear genotype-phenotype relationships
    - Baseline performance establishment

    Provides:
    - Per-drug resistance prediction
    - TEM variant classification (TEM-1/ESBL/IRT/CMT)
    - Mutation detection and scoring
    """

    def __init__(self, config: EcoliBetaLactamConfig | None = None):
        """Initialize analyzer.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or EcoliBetaLactamConfig()
        super().__init__(self.config)

        # Amino acid encoding alphabet
        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY-X*"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_alphabet)}

    def analyze(
        self,
        sequences: dict[EcoliGene, list[str]],
        drug: BetaLactam = BetaLactam.AMPICILLIN,
        embeddings: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Analyze TEM sequences for β-lactam resistance.

        Args:
            sequences: Dictionary mapping gene to list of sequences
            drug: Target drug for resistance prediction
            embeddings: Optional precomputed embeddings

        Returns:
            Analysis results dictionary
        """
        results = {
            "n_sequences": len(next(iter(sequences.values()))) if sequences else 0,
            "gene": EcoliGene.BLA_TEM.value,
            "target_drug": drug.value,
            "resistance": {},
            "variant_classification": [],
            "mutations_detected": [],
        }

        # Analyze TEM sequences
        if EcoliGene.BLA_TEM in sequences:
            tem_seqs = sequences[EcoliGene.BLA_TEM]

            # Predict resistance for target drug
            results["resistance"] = self._predict_drug_resistance(tem_seqs, drug)

            # Classify variants
            results["variant_classification"] = [self._classify_tem_variant(seq) for seq in tem_seqs]

            # Detect all mutations
            results["mutations_detected"] = [self._detect_mutations(seq) for seq in tem_seqs]

        return results

    def _predict_drug_resistance(
        self,
        sequences: list[str],
        drug: BetaLactam,
    ) -> dict[str, Any]:
        """Predict resistance for a specific drug.

        Args:
            sequences: TEM sequences
            drug: Target drug

        Returns:
            Resistance predictions
        """
        results = {
            "drug": drug.value,
            "scores": [],
            "classifications": [],
            "mutations": [],
        }

        # Get phenotypes that confer resistance to this drug
        resistance_phenotypes = DRUG_PHENOTYPE_MAP.get(drug, [])

        for seq in sequences:
            variant = self._classify_tem_variant(seq)
            mutations = self._detect_mutations(seq)

            # Base score depends on TEM presence and variant type
            if variant["type"] in resistance_phenotypes:
                base_score = 0.7 if variant["type"] == "TEM-1" else 0.9
            else:
                base_score = 0.1

            # Adjust based on mutation count and effects
            mutation_score = 0.0
            for mut in mutations:
                effect = mut.get("effect", "moderate")
                phenotype = mut.get("phenotype", "unknown")

                # Score mutations based on relevance to drug
                if drug in [BetaLactam.CEFTAZIDIME, BetaLactam.CEFOTAXIME, BetaLactam.CEFTRIAXONE, BetaLactam.CEFEPIME]:
                    if phenotype == "ESBL":
                        effect_scores = {"high": 0.3, "moderate": 0.15, "low": 0.05}
                        mutation_score += effect_scores.get(effect, 0.1)
                elif drug in [BetaLactam.AMOX_CLAVULANATE, BetaLactam.PIPERACILLIN_TAZO]:
                    if phenotype == "IRT":
                        effect_scores = {"high": 0.3, "moderate": 0.15, "low": 0.05}
                        mutation_score += effect_scores.get(effect, 0.1)
                else:
                    # Penicillins - any TEM mutation contributes
                    effect_scores = {"high": 0.1, "moderate": 0.05, "low": 0.02}
                    mutation_score += effect_scores.get(effect, 0.05)

            # Final score (capped at 1.0)
            final_score = min(base_score + mutation_score, 1.0)
            results["scores"].append(final_score)

            # Classification
            if final_score < 0.3:
                classification = "susceptible"
            elif final_score < 0.6:
                classification = "intermediate"
            else:
                classification = "resistant"

            results["classifications"].append(classification)
            results["mutations"].append(mutations)

        return results

    def _classify_tem_variant(self, sequence: str) -> dict[str, Any]:
        """Classify TEM sequence as TEM-1, ESBL, IRT, or CMT.

        Args:
            sequence: TEM amino acid sequence

        Returns:
            Variant classification with details
        """
        mutations = self._detect_mutations(sequence)

        has_esbl = any(m.get("phenotype") == "ESBL" for m in mutations)
        has_irt = any(m.get("phenotype") == "IRT" for m in mutations)

        if has_esbl and has_irt:
            variant_type = "CMT"
            description = "Complex mutant (ESBL + IRT)"
        elif has_esbl:
            variant_type = "ESBL"
            description = "Extended-spectrum β-lactamase"
        elif has_irt:
            variant_type = "IRT"
            description = "Inhibitor-resistant TEM"
        else:
            variant_type = "TEM-1"
            description = "Original TEM (penicillin resistance only)"

        return {
            "type": variant_type,
            "description": description,
            "esbl_mutations": sum(1 for m in mutations if m.get("phenotype") == "ESBL"),
            "irt_mutations": sum(1 for m in mutations if m.get("phenotype") == "IRT"),
            "stabilizer_mutations": sum(1 for m in mutations if m.get("phenotype") == "stabilizer"),
        }

    def _detect_mutations(self, sequence: str) -> list[dict[str, Any]]:
        """Detect known TEM mutations in sequence.

        Args:
            sequence: TEM amino acid sequence

        Returns:
            List of detected mutations with details
        """
        mutations = []

        for pos, info in TEM_MUTATIONS.items():
            if pos > len(sequence):
                continue

            ref_aa = list(info.keys())[0]
            seq_aa = sequence[pos - 1] if pos <= len(sequence) else "-"

            if seq_aa != ref_aa and seq_aa in info[ref_aa]["mutations"]:
                mutations.append(
                    {
                        "position": pos,
                        "ref": ref_aa,
                        "alt": seq_aa,
                        "notation": f"{ref_aa}{pos}{seq_aa}",
                        "effect": info[ref_aa]["effect"],
                        "phenotype": info[ref_aa]["phenotype"],
                        "note": info[ref_aa].get("note", ""),
                    }
                )

        return mutations

    def validate_predictions(
        self,
        predictions: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> dict[str, float]:
        """Validate predictions against phenotypic MIC data.

        Args:
            predictions: Model predictions
            ground_truth: Known resistance values

        Returns:
            Validation metrics
        """
        from scipy.stats import spearmanr

        metrics = {}

        pred_scores = predictions.get("resistance", {}).get("scores", [])
        true_scores = ground_truth.get("scores", [])

        if len(pred_scores) == len(true_scores) and len(pred_scores) > 2:
            rho, pval = spearmanr(pred_scores, true_scores)
            metrics["spearman"] = float(rho) if not np.isnan(rho) else 0.0
            metrics["spearman_pvalue"] = float(pval)

            # RMSE
            pred_arr = np.array(pred_scores)
            true_arr = np.array(true_scores)
            metrics["rmse"] = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))

        return metrics

    def encode_sequence(
        self,
        sequence: str,
        max_length: int = 300,
    ) -> np.ndarray:
        """One-hot encode TEM sequence.

        Args:
            sequence: Amino acid sequence
            max_length: Maximum sequence length

        Returns:
            Flattened one-hot encoding
        """
        n_aa = len(self.aa_alphabet)
        encoding = np.zeros(max_length * n_aa, dtype=np.float32)

        for j, aa in enumerate(sequence[:max_length]):
            idx = self.aa_to_idx.get(aa.upper(), self.aa_to_idx["X"])
            encoding[j * n_aa + idx] = 1.0

        return encoding


def create_ecoli_synthetic_dataset(
    drug: BetaLactam = BetaLactam.AMPICILLIN,
    min_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create synthetic E. coli TEM dataset for benchmarking.

    This generates a dataset with:
    - Wild-type TEM-1 (susceptible to cephalosporins)
    - Single mutants (various resistance levels)
    - Combination mutants (ESBL, IRT, CMT phenotypes)

    Args:
        drug: Target drug for resistance scores
        min_samples: Minimum number of samples to generate

    Returns:
        (X, y, ids) tuple where:
        - X: Encoded sequences (n_samples, n_features)
        - y: Resistance scores (0.0-1.0)
        - ids: Sample identifiers
    """
    from src.diseases.utils.synthetic_data import (
        create_mutation_based_dataset,
        ensure_minimum_samples,
    )

    # Use TEM-1 reference (263 AA)
    reference = TEM1_REFERENCE
    max_length = 300  # Slightly larger than TEM-1

    # Filter mutations based on drug
    if drug in [BetaLactam.CEFTAZIDIME, BetaLactam.CEFOTAXIME, BetaLactam.CEFTRIAXONE, BetaLactam.CEFEPIME]:
        # Focus on ESBL mutations for cephalosporins
        mutation_db = {
            pos: info for pos, info in TEM_MUTATIONS.items() if list(info.values())[0].get("phenotype") == "ESBL"
        }
    elif drug in [BetaLactam.AMOX_CLAVULANATE, BetaLactam.PIPERACILLIN_TAZO]:
        # Focus on IRT mutations for inhibitor combinations
        mutation_db = {
            pos: info for pos, info in TEM_MUTATIONS.items() if list(info.values())[0].get("phenotype") == "IRT"
        }
    else:
        # Penicillins - all mutations contribute
        mutation_db = TEM_MUTATIONS

    analyzer = EcoliBetaLactamAnalyzer()

    # Create dataset using shared utilities
    X, y, ids = create_mutation_based_dataset(
        reference_sequence=reference,
        mutation_db=mutation_db,
        encode_fn=lambda s, ml: analyzer.encode_sequence(s, max_length=ml),
        max_length=max_length,
        n_random_mutants=30,
        effect_scores={"high": 0.9, "moderate": 0.5, "low": 0.2},
        seed=42,
    )

    # Ensure minimum samples
    X, y, ids = ensure_minimum_samples(X, y, ids, min_samples=min_samples, seed=42)

    return X, y, ids
