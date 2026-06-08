# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""
Multiple Sclerosis Analyzer with P-adic Molecular Mimicry Detection.

Implements analysis tools for MS pathogenesis based on:
- Lanz et al. (2022): EBNA-1/GlialCAM molecular mimicry
- Wucherpfennig (1995): EBV/MBP mimicry hypothesis
- P-adic distance framework for epitope similarity

Key features:
- Molecular mimicry detection between viral and self peptides
- HLA binding prediction with p-adic distance metrics
- Cross-reactive T-cell epitope identification
- Demyelination risk scoring
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
import torch.nn as nn

# Import from shared modules (single source of truth)
from src.analysis.immunology.genetic_risk import MS_HLA_RISK_ALLELES
from src.biology.amino_acids import AMINO_ACID_PROPERTIES
from src.core.padic_math import PADIC_INFINITY, padic_valuation


class MSSubtype(Enum):
    """Multiple Sclerosis clinical subtypes."""

    RRMS = "relapsing_remitting"  # Relapsing-Remitting MS
    SPMS = "secondary_progressive"  # Secondary Progressive MS
    PPMS = "primary_progressive"  # Primary Progressive MS
    CIS = "clinically_isolated_syndrome"  # Clinically Isolated Syndrome


class MyelinTarget(Enum):
    """Myelin protein targets in MS."""

    MBP = "myelin_basic_protein"
    MOG = "myelin_oligodendrocyte_glycoprotein"
    PLP = "proteolipid_protein"
    MAG = "myelin_associated_glycoprotein"
    GLIALCAM = "glial_cell_adhesion_molecule"


# Known molecular mimicry pairs (viral epitope -> self epitope)
MOLECULAR_MIMICRY_PAIRS: dict[str, dict[str, Any]] = {
    "EBNA1_386_405": {
        "viral_sequence": "PRHRDTLMLFSS",
        "self_target": MyelinTarget.GLIALCAM,
        "self_sequence": "NRHSRNMHQALS",
        "hla_restriction": "DRB1*15:01",
        "evidence_level": "high",
        "reference": "Lanz2022",
    },
    "EBNA1_400_413": {
        "viral_sequence": "RRPFFHPVGEADYF",
        "self_target": MyelinTarget.MBP,
        "self_sequence": "VHFFKNIVTPRTP",
        "hla_restriction": "DRB1*15:01",
        "evidence_level": "medium",
        "reference": "Wucherpfennig1995",
    },
    "EBV_gp350": {
        "viral_sequence": "TGSGSGGQPH",
        "self_target": MyelinTarget.MOG,
        "self_sequence": "MEVGWYRPPFSRVV",
        "hla_restriction": "DRB1*04:01",
        "evidence_level": "medium",
        "reference": "Lang2002",
    },
}

# Note: HLA risk alleles and amino acid properties are now imported from shared modules:
# - MS_HLA_RISK_ALLELES from src.analysis.immunology.genetic_risk
# - AMINO_ACID_PROPERTIES from src.biology.amino_acids


@dataclass
class EpitopePair:
    """Represents a potential molecular mimicry pair."""

    viral_epitope: str
    self_epitope: str
    viral_source: str
    self_target: MyelinTarget
    similarity_score: float
    padic_distance: float
    hla_binding_score: float
    cross_reactivity_risk: float


@dataclass
class MSRiskProfile:
    """Risk profile for MS development."""

    genetic_risk: float  # HLA-based risk
    mimicry_risk: float  # Molecular mimicry risk
    environmental_risk: float  # EBV exposure, vitamin D, etc.
    overall_risk: float
    risk_factors: list[str] = field(default_factory=list)
    protective_factors: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class DemyelinationPrediction:
    """Prediction of demyelination pattern."""

    affected_regions: list[str]
    severity_scores: dict[str, float]
    progression_rate: float
    lesion_pattern: str
    predicted_disability: float  # EDSS scale


class MolecularMimicryDetector(nn.Module):
    """
    Detects molecular mimicry between viral and self peptides.

    Uses p-adic distance and structural similarity to identify
    potentially cross-reactive epitopes.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        p: int = 3,
        similarity_threshold: float = 0.7,
    ):
        """
        Initialize mimicry detector.

        Args:
            embedding_dim: Peptide embedding dimension
            hidden_dim: Hidden layer dimension
            p: Prime for p-adic calculations
            similarity_threshold: Threshold for mimicry detection
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.p = p
        self.similarity_threshold = similarity_threshold

        # Amino acid embedding
        self.aa_embedding = nn.Embedding(21, embedding_dim)  # 20 AAs + padding

        # Sequence encoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dim_feedforward=hidden_dim, batch_first=True),
            num_layers=2,
        )

        # Similarity network
        self.similarity_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # P-adic distance MLP
        self.padic_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def sequence_to_indices(self, sequence: str) -> torch.Tensor:
        """Convert amino acid sequence to indices."""
        aa_to_idx = {aa: i for i, aa in enumerate("ARNDCQEGHILKMFPSTWYV")}
        indices = [aa_to_idx.get(aa.upper(), 20) for aa in sequence]
        return torch.tensor(indices)

    def compute_padic_distance(self, seq1: str, seq2: str) -> float:
        """Compute p-adic inspired distance between sequences."""
        # Align sequences (simple approach: same length comparison)
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]

        total_dist = 0.0
        for i, (aa1, aa2) in enumerate(zip(seq1, seq2, strict=False)):
            if aa1 != aa2:
                # Position-weighted p-adic distance
                position_weight = self.p ** (-i % 4)
                total_dist += position_weight

        # Normalize
        return total_dist / min_len if min_len > 0 else 1.0

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode peptide sequence to embedding."""
        indices = self.sequence_to_indices(sequence).unsqueeze(0)
        embedded = self.aa_embedding(indices)
        encoded = self.encoder(embedded)
        # Pool to single vector
        return encoded.mean(dim=1)

    def compute_structural_similarity(self, seq1: str, seq2: str) -> float:
        """Compute structural similarity based on AA properties."""
        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0

        similarity = 0.0
        for aa1, aa2 in zip(seq1[:min_len], seq2[:min_len], strict=False):
            props1 = AMINO_ACID_PROPERTIES.get(aa1.upper(), {})
            props2 = AMINO_ACID_PROPERTIES.get(aa2.upper(), {})

            if props1 and props2:
                # Compare properties
                hydro_sim = 1.0 - abs(props1["hydrophobicity"] - props2["hydrophobicity"]) / 10.0
                charge_sim = 1.0 - abs(props1["charge"] - props2["charge"]) / 2.0
                polar_sim = 1.0 if props1["polarity"] == props2["polarity"] else 0.5
                similarity += (hydro_sim + charge_sim + polar_sim) / 3.0

        return similarity / min_len

    def forward(self, viral_sequence: str, self_sequence: str) -> dict[str, Any]:
        """
        Detect molecular mimicry between sequences.

        Args:
            viral_sequence: Viral peptide sequence
            self_sequence: Self peptide sequence

        Returns:
            Dictionary with mimicry analysis results
        """
        # Encode sequences
        viral_emb = self.encode_sequence(viral_sequence)
        self_emb = self.encode_sequence(self_sequence)

        # Compute neural similarity
        combined = torch.cat([viral_emb, self_emb], dim=-1)
        neural_similarity = self.similarity_net(combined).item()

        # Compute p-adic distance
        padic_dist = self.compute_padic_distance(viral_sequence, self_sequence)

        # Compute structural similarity
        structural_sim = self.compute_structural_similarity(viral_sequence, self_sequence)

        # Combined mimicry score
        # P-adic distance < 0.5 indicates high similarity in ultrametric space
        padic_score = max(0, 1 - padic_dist * 2)
        mimicry_score = 0.4 * neural_similarity + 0.3 * padic_score + 0.3 * structural_sim

        is_mimicry = mimicry_score >= self.similarity_threshold

        return {
            "mimicry_score": mimicry_score,
            "is_mimicry": is_mimicry,
            "neural_similarity": neural_similarity,
            "padic_distance": padic_dist,
            "structural_similarity": structural_sim,
            "viral_embedding": viral_emb,
            "self_embedding": self_emb,
        }


class HLABindingPredictor(nn.Module):
    """
    Predicts HLA binding affinity for peptides.

    Uses p-adic position encoding for anchor residue analysis.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        n_hla_alleles: int = 50,
    ):
        """
        Initialize HLA binding predictor.

        Args:
            embedding_dim: Embedding dimension
            n_hla_alleles: Number of HLA alleles to model
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_hla_alleles = n_hla_alleles

        # Peptide encoder
        self.aa_embedding = nn.Embedding(21, embedding_dim)

        # Position encoding (p-adic inspired)
        self.position_weights = nn.Parameter(torch.zeros(15, embedding_dim))
        self._init_padic_positions()

        # HLA allele embedding
        self.hla_embedding = nn.Embedding(n_hla_alleles, embedding_dim)

        # Binding prediction network
        self.binding_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def _init_padic_positions(self):
        """Initialize position weights with p-adic structure."""
        with torch.no_grad():
            for pos in range(15):
                # Anchor positions (1, 4, 6, 9 for class II) get higher weight
                if pos in [1, 4, 6, 9]:
                    weight = 1.0
                else:
                    # P-adic decay from anchors
                    min_dist = min(abs(pos - anchor) for anchor in [1, 4, 6, 9])
                    weight = 3.0 ** (-min_dist)
                self.position_weights[pos] = weight

    def encode_peptide(self, sequence: str) -> torch.Tensor:
        """Encode peptide with p-adic position weighting."""
        aa_to_idx = {aa: i for i, aa in enumerate("ARNDCQEGHILKMFPSTWYV")}

        # Pad/truncate to 15 residues (typical for class II)
        sequence = sequence[:15].ljust(15, "X")
        indices = [aa_to_idx.get(aa.upper(), 20) for aa in sequence]
        indices = torch.tensor(indices)

        embedded = self.aa_embedding(indices)

        # Apply position weighting
        weighted = embedded * self.position_weights

        return weighted.mean(dim=0)

    def forward(self, peptide: str, hla_idx: int) -> dict[str, float]:
        """
        Predict HLA binding.

        Args:
            peptide: Peptide sequence
            hla_idx: HLA allele index

        Returns:
            Dictionary with binding predictions
        """
        peptide_emb = self.encode_peptide(peptide).unsqueeze(0)
        hla_emb = self.hla_embedding(torch.tensor([hla_idx]))

        combined = torch.cat([peptide_emb, hla_emb], dim=-1)
        binding_score = self.binding_net(combined).item()

        # IC50 estimation (lower = stronger binding)
        ic50_nm = 50000 * (1 - binding_score) ** 3 + 1

        return {
            "binding_score": binding_score,
            "ic50_nm": ic50_nm,
            "is_strong_binder": binding_score > 0.8,
            "is_weak_binder": 0.5 < binding_score <= 0.8,
        }


class MultipleSclerosisAnalyzer:
    """
    Comprehensive MS analysis combining molecular mimicry,
    HLA risk, and disease progression modeling.
    """

    def __init__(
        self,
        p: int = 3,
        device: str = "cpu",
    ):
        """
        Initialize MS analyzer.

        Args:
            p: Prime for p-adic calculations
            device: Computation device
        """
        self.p = p
        self.device = device

        self.mimicry_detector = MolecularMimicryDetector(p=p).to(device)
        self.hla_predictor = HLABindingPredictor().to(device)

        # HLA allele name to index mapping
        self.hla_to_idx: dict[str, int] = {}

    def compute_padic_valuation(self, n: int) -> int:
        """Compute p-adic valuation of n.

        Uses centralized padic_valuation from src.core.padic_math.
        """
        val = padic_valuation(n, self.p)
        # Return infinity for n=0 (core returns PADIC_INFINITY_INT=100)
        return PADIC_INFINITY if n == 0 else val

    def analyze_known_mimicry(self) -> list[EpitopePair]:
        """Analyze known molecular mimicry pairs."""
        results = []

        for pair_name, pair_data in MOLECULAR_MIMICRY_PAIRS.items():
            analysis = self.mimicry_detector(pair_data["viral_sequence"], pair_data["self_sequence"])

            pair = EpitopePair(
                viral_epitope=pair_data["viral_sequence"],
                self_epitope=pair_data["self_sequence"],
                viral_source=pair_name,
                self_target=pair_data["self_target"],
                similarity_score=analysis["mimicry_score"],
                padic_distance=analysis["padic_distance"],
                hla_binding_score=0.0,  # Would need HLA context
                cross_reactivity_risk=analysis["mimicry_score"],
            )
            results.append(pair)

        return results

    def compute_genetic_risk(self, hla_alleles: list[str]) -> tuple[float, list[str], list[str]]:
        """
        Compute MS genetic risk from HLA typing.

        Args:
            hla_alleles: List of HLA alleles

        Returns:
            (risk_score, risk_factors, protective_factors)
        """
        risk_score = 1.0
        risk_factors = []
        protective_factors = []

        for allele in hla_alleles:
            if allele in MS_HLA_RISK_ALLELES:
                odds_ratio = MS_HLA_RISK_ALLELES[allele]
                risk_score *= odds_ratio

                if odds_ratio > 1.0:
                    risk_factors.append(f"{allele} (OR={odds_ratio:.1f})")
                else:
                    protective_factors.append(f"{allele} (OR={odds_ratio:.1f})")

        return risk_score, risk_factors, protective_factors

    def assess_ebv_mimicry_risk(self, ebv_strain_features: dict[str, Any] | None = None) -> float:
        """
        Assess risk from EBV molecular mimicry.

        Args:
            ebv_strain_features: Optional EBV strain-specific features

        Returns:
            Mimicry risk score
        """
        # Analyze known EBNA-1 epitopes
        mimicry_pairs = self.analyze_known_mimicry()

        if not mimicry_pairs:
            return 0.5  # Baseline risk

        # Weight by evidence level
        weighted_risk = 0.0
        total_weight = 0.0

        for pair in mimicry_pairs:
            evidence = MOLECULAR_MIMICRY_PAIRS.get(pair.viral_source, {}).get("evidence_level", "low")
            weight = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(evidence, 0.3)
            weighted_risk += pair.cross_reactivity_risk * weight
            total_weight += weight

        return weighted_risk / total_weight if total_weight > 0 else 0.5

    def compute_risk_profile(
        self,
        hla_alleles: list[str],
        ebv_positive: bool = True,
        vitamin_d_level: float | None = None,
        smoking: bool = False,
        family_history: bool = False,
    ) -> MSRiskProfile:
        """
        Compute comprehensive MS risk profile.

        Args:
            hla_alleles: HLA typing results
            ebv_positive: EBV serostatus
            vitamin_d_level: Vitamin D level (ng/mL)
            smoking: Smoking status
            family_history: Family history of MS

        Returns:
            Complete risk profile
        """
        # Genetic risk
        genetic_risk, risk_factors, protective_factors = self.compute_genetic_risk(hla_alleles)

        # Mimicry risk
        mimicry_risk = self.assess_ebv_mimicry_risk() if ebv_positive else 0.2

        # Environmental factors
        env_risk = 1.0

        if ebv_positive:
            env_risk *= 2.5
            risk_factors.append("EBV seropositive")

        if vitamin_d_level is not None:
            if vitamin_d_level < 20:
                env_risk *= 1.5
                risk_factors.append(f"Low vitamin D ({vitamin_d_level} ng/mL)")
            elif vitamin_d_level > 40:
                env_risk *= 0.7
                protective_factors.append(f"Adequate vitamin D ({vitamin_d_level} ng/mL)")

        if smoking:
            env_risk *= 1.6
            risk_factors.append("Smoking")

        if family_history:
            env_risk *= 2.0
            risk_factors.append("Family history of MS")

        # Combined risk
        overall_risk = genetic_risk * 0.4 + mimicry_risk * 0.3 + env_risk * 0.3

        # Normalize to 0-1 scale
        overall_risk = min(1.0, overall_risk / 10.0)

        # Generate recommendations
        recommendations = []
        if vitamin_d_level is not None and vitamin_d_level < 30:
            recommendations.append("Consider vitamin D supplementation")
        if smoking:
            recommendations.append("Smoking cessation strongly recommended")
        if "DRB1*15:01" in hla_alleles:
            recommendations.append("High genetic risk - regular neurological monitoring advised")

        return MSRiskProfile(
            genetic_risk=min(1.0, genetic_risk / 5.0),
            mimicry_risk=mimicry_risk,
            environmental_risk=min(1.0, env_risk / 5.0),
            overall_risk=overall_risk,
            risk_factors=risk_factors,
            protective_factors=protective_factors,
            recommendations=recommendations,
        )

    def predict_demyelination_pattern(
        self,
        lesion_locations: list[str],
        symptom_onset_age: int,
        subtype: MSSubtype,
    ) -> DemyelinationPrediction:
        """
        Predict demyelination progression pattern.

        Args:
            lesion_locations: Current lesion locations
            symptom_onset_age: Age at symptom onset
            subtype: MS clinical subtype

        Returns:
            Demyelination prediction
        """
        # Common MS lesion regions
        all_regions = [
            "periventricular",
            "juxtacortical",
            "infratentorial",
            "spinal_cord",
            "optic_nerve",
            "corpus_callosum",
        ]

        # Initialize severity scores
        severity_scores = {region: 0.0 for region in all_regions}

        # Score existing lesions
        for location in lesion_locations:
            if location in severity_scores:
                severity_scores[location] = 0.7

        # Predict likely affected regions based on subtype
        if subtype == MSSubtype.RRMS:
            # RRMS: periventricular and optic nerve common
            severity_scores["periventricular"] = max(severity_scores["periventricular"], 0.5)
            severity_scores["optic_nerve"] = max(severity_scores["optic_nerve"], 0.4)
            progression_rate = 0.3
        elif subtype == MSSubtype.PPMS:
            # PPMS: more spinal cord involvement
            severity_scores["spinal_cord"] = max(severity_scores["spinal_cord"], 0.6)
            progression_rate = 0.6
        elif subtype == MSSubtype.SPMS:
            # SPMS: widespread involvement
            for region in severity_scores:
                severity_scores[region] = max(severity_scores[region], 0.4)
            progression_rate = 0.5
        else:  # CIS
            progression_rate = 0.2

        # Age adjustment
        if symptom_onset_age < 30:
            progression_rate *= 0.8  # Younger onset often better prognosis
        elif symptom_onset_age > 50:
            progression_rate *= 1.2

        # Determine lesion pattern
        if severity_scores["periventricular"] > 0.5:
            lesion_pattern = "typical_ms"
        elif severity_scores["spinal_cord"] > 0.5:
            lesion_pattern = "spinal_predominant"
        else:
            lesion_pattern = "atypical"

        # Estimate disability (simplified EDSS)
        active_regions = sum(1 for s in severity_scores.values() if s > 0.3)
        predicted_disability = min(6.0, active_regions * 1.0)

        return DemyelinationPrediction(
            affected_regions=[r for r, s in severity_scores.items() if s > 0.3],
            severity_scores=severity_scores,
            progression_rate=progression_rate,
            lesion_pattern=lesion_pattern,
            predicted_disability=predicted_disability,
        )

    def scan_for_novel_mimicry(
        self,
        viral_proteome: dict[str, str],
        self_proteome: dict[str, str] | None = None,
        window_size: int = 12,
        threshold: float = 0.7,
    ) -> list[EpitopePair]:
        """
        Scan for novel molecular mimicry candidates.

        Args:
            viral_proteome: Dictionary of viral protein sequences
            self_proteome: Dictionary of self protein sequences
            window_size: Sliding window size for epitope detection
            threshold: Mimicry score threshold

        Returns:
            List of potential mimicry pairs
        """
        # Default self-proteome: myelin proteins
        if self_proteome is None:
            self_proteome = {
                "MBP": "ASQKRPSQRHGSKYLATASTMDHARHGFLPRHRDTGILDSIGRFFGGDRGAPKRGSGKDSHHPAR",
                "MOG": "GQFRVIGPRHPIRALVGDEAELPCRISPGKNATGMEVGWYRPPFSRVVHLYRNGKD",
                "PLP": "GLSATVTGGQKGRGSRGQHQAHSLERVCHCLGKWLGHPDKFVGI",
            }

        candidates = []

        for viral_name, viral_seq in viral_proteome.items():
            for _self_name, self_seq in self_proteome.items():
                # Sliding window over viral sequence
                for i in range(len(viral_seq) - window_size + 1):
                    viral_window = viral_seq[i : i + window_size]

                    # Compare to self windows
                    for j in range(len(self_seq) - window_size + 1):
                        self_window = self_seq[j : j + window_size]

                        analysis = self.mimicry_detector(viral_window, self_window)

                        if analysis["mimicry_score"] >= threshold:
                            pair = EpitopePair(
                                viral_epitope=viral_window,
                                self_epitope=self_window,
                                viral_source=f"{viral_name}_{i}-{i + window_size}",
                                self_target=MyelinTarget.MBP,  # Would need proper mapping
                                similarity_score=analysis["mimicry_score"],
                                padic_distance=analysis["padic_distance"],
                                hla_binding_score=0.0,
                                cross_reactivity_risk=analysis["mimicry_score"],
                            )
                            candidates.append(pair)

        # Sort by mimicry score
        candidates.sort(key=lambda x: x.similarity_score, reverse=True)

        return candidates[:20]  # Return top 20
