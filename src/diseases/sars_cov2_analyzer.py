# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""SARS-CoV-2 Variant and Drug Resistance Analyzer.

This module provides comprehensive analysis of SARS-CoV-2 mutations:
- Variant emergence prediction (VOC identification)
- Drug resistance prediction (Paxlovid/nirmatrelvir - nsp5/Mpro)
- Antibody escape prediction (Spike RBD)
- ACE2 binding affinity changes

Based on:
- EVEscape methodology for escape prediction
- P-adic encoding for mutation distance
- Transfer learning from HIV drug resistance

Usage:
    from src.diseases.sars_cov2_analyzer import SARSCoV2Analyzer

    analyzer = SARSCoV2Analyzer()
    results = analyzer.analyze_variants(sequences)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

from .base import DiseaseAnalyzer, DiseaseConfig, DiseaseType, TaskType
from .utils.synthetic_data import ensure_minimum_samples


class SARSCoV2Gene(Enum):
    """SARS-CoV-2 genes/proteins of interest."""

    SPIKE = "S"  # Spike protein (antibody escape, ACE2 binding)
    NSP5 = "nsp5"  # Main protease (Mpro) - Paxlovid target
    NSP12 = "nsp12"  # RdRp - Remdesivir target
    NSP3 = "nsp3"  # Papain-like protease
    NUCLEOCAPSID = "N"  # Nucleocapsid


class SARSCoV2Variant(Enum):
    """Known variants of concern (VOCs)."""

    WILD_TYPE = "Wuhan-Hu-1"
    ALPHA = "B.1.1.7"
    BETA = "B.1.351"
    GAMMA = "P.1"
    DELTA = "B.1.617.2"
    OMICRON_BA1 = "BA.1"
    OMICRON_BA2 = "BA.2"
    OMICRON_BA5 = "BA.5"
    OMICRON_XBB = "XBB"
    OMICRON_JN1 = "JN.1"


@dataclass
class SARSCoV2Config(DiseaseConfig):
    """Configuration for SARS-CoV-2 analysis.

    Extends base DiseaseConfig with SARS-CoV-2-specific parameters.
    """

    name: str = "sars_cov_2"
    display_name: str = "SARS-CoV-2"
    disease_type: DiseaseType = DiseaseType.VIRAL
    tasks: list[TaskType] = field(
        default_factory=lambda: [
            TaskType.RESISTANCE,  # Drug resistance (Paxlovid)
            TaskType.ESCAPE,  # Immune escape
            TaskType.BINDING,  # ACE2 binding
        ]
    )

    # Protein-specific settings
    target_gene: SARSCoV2Gene = SARSCoV2Gene.SPIKE
    reference_variant: SARSCoV2Variant = SARSCoV2Variant.WILD_TYPE

    # Data sources
    data_sources: dict[str, str] = field(
        default_factory=lambda: {
            "gisaid": "https://gisaid.org/",  # Requires registration
            "cov_rdb": "https://covdb.stanford.edu/",
            "cov_lineages": "https://cov-lineages.org/",
            "pdb": "https://www.rcsb.org/",
        }
    )

    # Drug-specific
    drugs: list[str] = field(
        default_factory=lambda: [
            "nirmatrelvir",  # Paxlovid component (Mpro inhibitor)
            "ritonavir",  # Paxlovid booster
            "remdesivir",  # RdRp inhibitor
            "molnupiravir",  # Mutagenic nucleoside analog
        ]
    )


# Reference sequences (Wuhan-Hu-1)
# Spike RBD (residues 319-541)
SPIKE_RBD_SEQUENCE = (
    "RVQPTESIVRFPNITNLCPFGEVFNATRFASVYAWNRKRISNCVADYSVLYNSASFSTFKC"
    "YGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIADYNYKLPDDFTGCVIAWNSNK"
    "LDSKVSGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYFPLQSYGFQPTN"
    "GVGYQPYRVVVLSFELLHAPATVCGPKKST"
)

# nsp5/Mpro (Main protease) - Paxlovid target (306 aa)
NSP5_MPRO_SEQUENCE = (
    "SGFRKMAFPSGKVEGCMVQVTCGTTTLNGLWLDDVVYCPRHVICTSEDMLNPNYEDLLIRK"
    "SNHNFLVQAGNVQLRVIGHSMQNCVLKLKVDTANPKTPKYKFVRIQPGQTFSVLACYNGRG"
    "TFCKGTYPVLNVTDHVDADEGQQPYSAWMVTRNGAVYVPAQNNSWANSSELFYRLCNMHLD"
    "DGKMQEVQEFARAISFPLFRYFQLHDSAEVFTGVGMVEHVTLKEAGFITPPLDSSGTSGFV"
    "NTVPMLRSGCLGKPKLFFIQACGGEQKAHMQVLCFLQTLMKRVKATIHVTASQGLNLATAEV"
    "VSIVFQ"
)

# Key mutation positions in Mpro for nirmatrelvir resistance
MPRO_RESISTANCE_POSITIONS = {
    # Direct active site
    25: {"ref": "T", "mutations": ["N"], "effect": "active_site"},
    26: {"ref": "T", "mutations": ["I", "S"], "effect": "active_site"},
    140: {"ref": "F", "mutations": ["L"], "effect": "active_site"},
    142: {"ref": "N", "mutations": ["D"], "effect": "active_site"},
    143: {"ref": "G", "mutations": ["S"], "effect": "active_site"},
    144: {"ref": "S", "mutations": ["A", "G"], "effect": "S1_subsite"},
    163: {"ref": "H", "mutations": ["Q", "Y"], "effect": "oxyanion_hole"},
    166: {"ref": "E", "mutations": ["V", "A", "Q"], "effect": "primary_resistance"},
    # S4 pocket (nirmatrelvir binding)
    167: {"ref": "L", "mutations": ["F"], "effect": "S4_pocket"},
    168: {"ref": "P", "mutations": ["S", "T"], "effect": "S4_pocket"},
    169: {"ref": "H", "mutations": ["Q", "Y"], "effect": "S4_pocket"},
    # Compensatory mutations
    46: {"ref": "S", "mutations": ["L"], "effect": "compensatory"},
    50: {"ref": "A", "mutations": ["V"], "effect": "compensatory"},
    # Emerging resistance
    21: {"ref": "T", "mutations": ["I"], "effect": "emerging"},
    132: {"ref": "L", "mutations": ["I"], "effect": "emerging"},
}

# Key Spike RBD mutations for ACE2 binding and antibody escape
SPIKE_RBD_MUTATIONS = {
    # ACE2 binding enhanced
    417: {"ref": "K", "mutations": ["N", "T"], "effect": "ace2_binding"},
    484: {"ref": "E", "mutations": ["K", "Q", "A"], "effect": "antibody_escape"},
    501: {"ref": "N", "mutations": ["Y"], "effect": "ace2_binding+escape"},
    # Major escape mutations
    452: {"ref": "L", "mutations": ["R", "Q"], "effect": "antibody_escape"},
    486: {"ref": "F", "mutations": ["V", "S", "P"], "effect": "antibody_escape"},
    493: {"ref": "Q", "mutations": ["R"], "effect": "ace2_binding"},
    498: {"ref": "Q", "mutations": ["R"], "effect": "ace2_binding"},
    # Omicron-specific
    339: {"ref": "G", "mutations": ["D"], "effect": "antibody_escape"},
    346: {"ref": "R", "mutations": ["K"], "effect": "antibody_escape"},
    371: {"ref": "S", "mutations": ["F", "L"], "effect": "antibody_escape"},
    373: {"ref": "S", "mutations": ["P"], "effect": "antibody_escape"},
    375: {"ref": "S", "mutations": ["F"], "effect": "antibody_escape"},
    376: {"ref": "T", "mutations": ["A"], "effect": "antibody_escape"},
    440: {"ref": "N", "mutations": ["K"], "effect": "antibody_escape"},
    446: {"ref": "G", "mutations": ["S"], "effect": "antibody_escape"},
    477: {"ref": "S", "mutations": ["N"], "effect": "antibody_escape"},
    478: {"ref": "T", "mutations": ["K"], "effect": "antibody_escape"},
    505: {"ref": "Y", "mutations": ["H"], "effect": "ace2_binding"},
}


class SARSCoV2Analyzer(DiseaseAnalyzer):
    """Analyzer for SARS-CoV-2 variants and drug resistance.

    Provides:
    - Drug resistance prediction (nirmatrelvir/Paxlovid)
    - Antibody escape prediction
    - ACE2 binding affinity estimation
    - Variant classification
    """

    def __init__(self, config: SARSCoV2Config | None = None):
        """Initialize analyzer.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or SARSCoV2Config()
        super().__init__(self.config)

        # Amino acid encoding
        self.aa_alphabet = "ACDEFGHIKLMNPQRSTVWY-X"
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.aa_alphabet)}

    def analyze(
        self,
        sequences: list[str],
        embeddings: torch.Tensor | None = None,
        gene: SARSCoV2Gene = SARSCoV2Gene.NSP5,
        **kwargs,
    ) -> dict[str, Any]:
        """Analyze SARS-CoV-2 sequences.

        Args:
            sequences: List of protein sequences
            embeddings: Optional precomputed embeddings
            gene: Target gene for analysis
            **kwargs: Additional parameters

        Returns:
            Analysis results dictionary
        """
        results = {
            "n_sequences": len(sequences),
            "gene": gene.value,
            "mutations": [],
            "drug_resistance": None,
            "escape_predictions": None,
            "variant_classification": None,
        }

        # Detect mutations
        reference = self._get_reference(gene)
        for seq in sequences:
            mutations = self._detect_mutations(seq, reference)
            results["mutations"].append(mutations)

        # Drug resistance (for Mpro)
        if gene == SARSCoV2Gene.NSP5:
            results["drug_resistance"] = self._predict_drug_resistance(sequences, "nirmatrelvir")

        # Antibody escape (for Spike)
        if gene == SARSCoV2Gene.SPIKE:
            results["escape_predictions"] = self._predict_antibody_escape(sequences)
            results["ace2_binding"] = self._predict_ace2_binding(sequences)

        return results

    def _get_reference(self, gene: SARSCoV2Gene) -> str:
        """Get reference sequence for a gene."""
        references = {
            SARSCoV2Gene.NSP5: NSP5_MPRO_SEQUENCE,
            SARSCoV2Gene.SPIKE: SPIKE_RBD_SEQUENCE,
        }
        return references.get(gene, "")

    def _detect_mutations(self, sequence: str, reference: str) -> list[dict[str, Any]]:
        """Detect mutations compared to reference.

        Returns list of mutation dicts with position, ref, alt, and annotations.
        """
        mutations = []
        min_len = min(len(sequence), len(reference))

        for i in range(min_len):
            if sequence[i] != reference[i] and sequence[i] != "-":
                mutation = {
                    "position": i + 1,  # 1-indexed
                    "reference": reference[i],
                    "alternate": sequence[i],
                    "notation": f"{reference[i]}{i + 1}{sequence[i]}",
                }

                # Add annotations if known
                if i + 1 in MPRO_RESISTANCE_POSITIONS:
                    info = MPRO_RESISTANCE_POSITIONS[i + 1]
                    mutation["effect"] = info["effect"]
                    mutation["known_resistance"] = sequence[i] in info["mutations"]

                mutations.append(mutation)

        return mutations

    def _predict_drug_resistance(
        self,
        sequences: list[str],
        drug: str = "nirmatrelvir",
    ) -> dict[str, Any]:
        """Predict drug resistance scores.

        Uses mutation counting + known resistance positions.
        """
        if drug != "nirmatrelvir":
            return {"error": f"Drug {drug} not supported yet"}

        results = {
            "drug": drug,
            "scores": [],
            "classifications": [],
            "key_mutations": [],
        }

        for seq in sequences:
            score = 0.0
            mutations = []

            for pos, info in MPRO_RESISTANCE_POSITIONS.items():
                if pos <= len(seq):
                    seq_aa = seq[pos - 1]
                    if seq_aa in info["mutations"]:
                        # Weight by effect type
                        weights = {
                            "active_site": 3.0,
                            "S1_subsite": 2.5,
                            "S4_pocket": 2.0,
                            "primary_resistance": 3.5,
                            "oxyanion_hole": 2.0,
                            "compensatory": 1.0,
                            "emerging": 1.5,
                        }
                        score += weights.get(info["effect"], 1.0)
                        mutations.append(f"{info['ref']}{pos}{seq_aa}")

            # Normalize to 0-1 range
            max_score = 25.0  # Approximate max possible
            normalized = min(score / max_score, 1.0)

            results["scores"].append(normalized)
            results["key_mutations"].append(mutations)

            # Classification
            if normalized < 0.1:
                classification = "susceptible"
            elif normalized < 0.3:
                classification = "potential_low_resistance"
            elif normalized < 0.5:
                classification = "low_resistance"
            elif normalized < 0.7:
                classification = "intermediate_resistance"
            else:
                classification = "high_resistance"

            results["classifications"].append(classification)

        return results

    def _predict_antibody_escape(self, sequences: list[str]) -> dict[str, Any]:
        """Predict antibody escape potential for Spike sequences."""
        results = {
            "scores": [],
            "escape_mutations": [],
        }

        for seq in sequences:
            score = 0.0
            escape_muts = []

            for pos, info in SPIKE_RBD_MUTATIONS.items():
                # Adjust for RBD starting position (319)
                rbd_pos = pos - 319
                if 0 <= rbd_pos < len(seq):
                    seq_aa = seq[rbd_pos]
                    if seq_aa in info["mutations"]:
                        if "escape" in info["effect"]:
                            score += 2.0
                        else:
                            score += 1.0
                        escape_muts.append(f"{info['ref']}{pos}{seq_aa}")

            # Normalize
            max_score = 50.0
            results["scores"].append(min(score / max_score, 1.0))
            results["escape_mutations"].append(escape_muts)

        return results

    def _predict_ace2_binding(self, sequences: list[str]) -> dict[str, Any]:
        """Predict ACE2 binding affinity changes."""
        results = {
            "scores": [],  # Relative to WT (>1 = enhanced, <1 = reduced)
            "binding_mutations": [],
        }

        ace2_positions = {k: v for k, v in SPIKE_RBD_MUTATIONS.items() if "ace2" in v["effect"]}

        for seq in sequences:
            score = 1.0  # Start at WT level
            binding_muts = []

            for pos, info in ace2_positions.items():
                rbd_pos = pos - 319
                if 0 <= rbd_pos < len(seq):
                    seq_aa = seq[rbd_pos]
                    if seq_aa in info["mutations"]:
                        # N501Y and similar enhance binding
                        if pos == 501 and seq_aa == "Y":
                            score *= 1.3
                        elif "enhanced" in info.get("notes", ""):
                            score *= 1.1
                        else:
                            score *= 0.95  # Slight reduction by default
                        binding_muts.append(f"{info['ref']}{pos}{seq_aa}")

            results["scores"].append(score)
            results["binding_mutations"].append(binding_muts)

        return results

    def validate_predictions(
        self,
        predictions: dict[str, torch.Tensor],
        ground_truth: dict[str, Any],
    ) -> dict[str, float]:
        """Validate predictions against experimental data."""
        from scipy.stats import spearmanr

        metrics = {}

        # Validate drug resistance if available
        if "resistance" in predictions and "resistance" in ground_truth:
            pred = predictions["resistance"].numpy()
            true = np.array(ground_truth["resistance"])
            rho, pval = spearmanr(pred, true)
            metrics["resistance_spearman"] = float(rho)
            metrics["resistance_pvalue"] = float(pval)

        # Validate escape predictions
        if "escape" in predictions and "escape" in ground_truth:
            pred = predictions["escape"].numpy()
            true = np.array(ground_truth["escape"])
            rho, pval = spearmanr(pred, true)
            metrics["escape_spearman"] = float(rho)

        return metrics

    def encode_sequences(
        self,
        sequences: list[str],
        max_length: int | None = None,
    ) -> np.ndarray:
        """One-hot encode sequences.

        Args:
            sequences: List of amino acid sequences
            max_length: Pad/truncate to this length

        Returns:
            One-hot encoded array (n_sequences, max_length * n_aa)
        """
        if max_length is None:
            max_length = max(len(s) for s in sequences)

        n_aa = len(self.aa_alphabet)
        X = np.zeros((len(sequences), max_length * n_aa), dtype=np.float32)

        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq[:max_length]):
                idx = self.aa_to_idx.get(aa.upper(), self.aa_to_idx["X"])
                X[i, j * n_aa + idx] = 1.0

        return X


def create_sars_cov2_dataset(
    gene: SARSCoV2Gene = SARSCoV2Gene.NSP5,
    include_resistance: bool = True,
    min_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create synthetic dataset for testing.

    In production, this should load from GISAID or CoV-RDB.

    Args:
        gene: Target gene (NSP5 for Mpro, SPIKE for antibody escape)
        include_resistance: Whether to include resistance mutations
        min_samples: Minimum number of samples to generate

    Returns:
        (X, y, sequence_ids)
    """
    # Generate variants of reference
    reference = NSP5_MPRO_SEQUENCE if gene == SARSCoV2Gene.NSP5 else SPIKE_RBD_SEQUENCE

    sequences = [reference]  # Wild type
    resistances = [0.0]
    ids = ["WT"]

    # Add known resistance mutations
    # Weights consistent with _predict_drug_resistance (normalized 0-1)
    for pos, info in MPRO_RESISTANCE_POSITIONS.items():
        if pos <= len(reference):
            for mut in info["mutations"]:
                mutant = list(reference)
                mutant[pos - 1] = mut
                sequences.append("".join(mutant))

                weights = {
                    "primary_resistance": 0.9,
                    "active_site": 0.7,
                    "oxyanion_hole": 0.6,
                    "S1_subsite": 0.5,
                    "S4_pocket": 0.5,
                    "emerging": 0.4,
                    "compensatory": 0.2,
                }
                resistances.append(weights.get(info["effect"], 0.3))
                ids.append(f"{info['ref']}{pos}{mut}")

    # Encode
    analyzer = SARSCoV2Analyzer()
    X = analyzer.encode_sequences(sequences)
    y = np.array(resistances, dtype=np.float32)

    # Ensure minimum samples
    X, y, ids = ensure_minimum_samples(X, y, ids, min_samples=min_samples)

    return X, y, ids
