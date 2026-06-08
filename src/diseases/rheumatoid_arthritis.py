# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Rheumatoid Arthritis Analysis with Citrullination Focus.

This module implements analysis of rheumatoid arthritis (RA) through the
p-adic/geometric lens, with particular focus on citrullination-induced
autoimmunity and the "Goldilocks Zone" hypothesis.

Key concepts:
- Citrullination: Post-translational modification (Arg -> Cit) by PAD enzymes
- ACPA: Anti-citrullinated protein antibodies (diagnostic biomarkers)
- Goldilocks Zone: p-adic distance range where immune recognition breaks down
- HLA-DRB1: Shared epitope alleles increase RA risk

The p-adic framework captures:
- Hierarchical nature of protein epitope recognition
- Distance from self to citrullinated-self in p-adic space
- Threshold effects in autoimmune triggering

References:
- DISCOVERY_CITRULLINATION_BOUNDARIES.md
- CROSS_ANALYSIS_AND_CONCLUSIONS.md
- 2003_Suzuki_PADI4_Haplotypes.md
"""

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

from src.geometry import poincare_distance


class PADEnzyme(Enum):
    """Peptidylarginine deiminase enzymes."""

    PAD1 = "PAD1"  # Epidermis, uterus
    PAD2 = "PAD2"  # Brain, muscle, macrophages
    PAD3 = "PAD3"  # Hair follicles
    PAD4 = "PAD4"  # Neutrophils, granulocytes (key in RA)
    PAD6 = "PAD6"  # Oocytes


class RASubtype(Enum):
    """Rheumatoid arthritis subtypes."""

    ACPA_POSITIVE = "acpa_positive"
    ACPA_NEGATIVE = "acpa_negative"
    RF_POSITIVE = "rf_positive"
    SERONEGATIVE = "seronegative"


@dataclass
class CitrullinationSite:
    """Represents a citrullination site in a protein."""

    protein_name: str
    position: int
    sequence_context: str  # 15-mer centered on Arg
    padic_distance_to_self: float  # Distance from Arg to Cit version
    immunogenicity_score: float  # Predicted immunogenicity (0-1)
    in_goldilocks_zone: bool
    known_acpa_target: bool


@dataclass
class RARiskProfile:
    """Risk profile for rheumatoid arthritis."""

    hla_alleles: list[str]
    shared_epitope_positive: bool
    padi4_haplotype: str | None
    smoking_history: bool
    ebv_positive: bool
    genetic_risk_score: float
    environmental_risk_score: float
    overall_risk: float
    risk_category: str  # low, moderate, high, very_high


@dataclass
class EpitopeAnalysis:
    """Analysis of a citrullinated epitope."""

    sequence: str
    citrullination_positions: list[int]
    native_padic_embedding: torch.Tensor
    citrullinated_padic_embedding: torch.Tensor
    padic_shift: float
    hla_binding_affinity: float
    tcr_cross_reactivity: float
    predicted_pathogenicity: float


# RA-associated HLA alleles (shared epitope)
RA_RISK_HLA_ALLELES = {
    "DRB1*04:01": 5.0,  # Odds ratio
    "DRB1*04:04": 4.5,
    "DRB1*04:05": 4.0,
    "DRB1*01:01": 3.5,
    "DRB1*10:01": 3.0,
    "DRB1*04:08": 2.5,
    "DRB1*14:02": 2.0,
}

# Known citrullinated protein targets in RA
KNOWN_ACPA_TARGETS = {
    "fibrinogen": ["Fibα573-591", "Fibα79-91", "Fibβ60-74"],
    "vimentin": ["Vim60-75", "Vim2-17"],
    "enolase": ["CEP-1", "α-enolase"],
    "collagen_ii": ["CII359-369", "CII1237-1249"],
    "fibronectin": ["FN1035-1050"],
    "histones": ["H2B", "H3", "H4"],
}

# PADI4 haplotypes and their risk associations
PADI4_HAPLOTYPES = {
    "susceptibility": ["GTAC", "GCAC"],  # Associated with RA risk
    "protective": ["ACGG"],
}

# Amino acid properties for citrullination analysis
AMINO_ACID_CHARGE = {
    "R": 1.0,  # Arginine (positive)
    "Cit": 0.0,  # Citrulline (neutral)
    "K": 1.0,
    "H": 0.5,
    "D": -1.0,
    "E": -1.0,
}


class CitrullinationPredictor(nn.Module):
    """Predicts citrullination propensity at arginine residues.

    Uses sequence context to predict likelihood of PAD-mediated
    citrullination at each arginine.
    """

    def __init__(
        self,
        context_size: int = 15,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
    ):
        """Initialize predictor.

        Args:
            context_size: Size of sequence context window
            embedding_dim: Amino acid embedding dimension
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.context_size = context_size

        # Amino acid embedding (20 AAs + X + gap)
        self.aa_embedding = nn.Embedding(22, embedding_dim)

        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(embedding_dim * context_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        # Citrullination predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),
        )

        # PAD enzyme specificity heads
        self.pad_heads = nn.ModuleDict({pad.value: nn.Linear(hidden_dim // 2, 1) for pad in PADEnzyme})

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode amino acid sequence.

        Args:
            sequence: Amino acid sequence

        Returns:
            Tensor of indices
        """
        aa_map = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        aa_map["X"] = 20
        aa_map["-"] = 21

        indices = [aa_map.get(aa.upper(), 20) for aa in sequence]
        return torch.tensor(indices)

    def forward(
        self,
        context_sequences: list[str],
        pad_enzyme: PADEnzyme | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict citrullination propensity.

        Args:
            context_sequences: List of 15-mer sequences centered on Arg
            pad_enzyme: Optional specific PAD enzyme

        Returns:
            Dictionary with predictions
        """
        # Encode sequences
        encoded = torch.stack([self.encode_sequence(seq) for seq in context_sequences])

        # Embed
        embedded = self.aa_embedding(encoded)  # (batch, context, embed_dim)
        flat = embedded.flatten(start_dim=1)

        # Encode context
        context_features = self.context_encoder(flat)

        # Predict overall citrullination propensity
        propensity = self.predictor(context_features)

        result = {"propensity": propensity.squeeze(-1)}

        # Predict PAD-specific propensity if requested
        if pad_enzyme is not None:
            pad_specific = torch.sigmoid(self.pad_heads[pad_enzyme.value](context_features))
            result["pad_specific"] = pad_specific.squeeze(-1)
        else:
            # Predict for all PAD enzymes
            for pad in PADEnzyme:
                pad_score = torch.sigmoid(self.pad_heads[pad.value](context_features))
                result[f"pad_{pad.value.lower()}"] = pad_score.squeeze(-1)

        return result


class PAdicCitrullinationShift(nn.Module):
    """Computes p-adic shift induced by citrullination.

    Models the change in p-adic distance between native and
    citrullinated versions of epitopes.
    """

    def __init__(
        self,
        p: int = 3,
        embedding_dim: int = 64,
        use_hyperbolic: bool = True,
        curvature: float = 1.0,
    ):
        """Initialize shift calculator.

        Args:
            p: Prime for p-adic calculations
            embedding_dim: Embedding dimension for sequences
            use_hyperbolic: V5.12.2 - Use poincare_distance for hyperbolic embeddings (default True)
            curvature: Hyperbolic curvature for poincare_distance
        """
        super().__init__()
        self.p = p
        self.embedding_dim = embedding_dim
        self.use_hyperbolic = use_hyperbolic
        self.curvature = curvature

        # Amino acid embedding
        self.aa_embedding = nn.Embedding(22, embedding_dim)

        # Sequence encoder
        self.encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )

        # P-adic projection
        self.padic_proj = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Tanh(),
        )

    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """Encode amino acid sequence."""
        aa_map = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        aa_map["X"] = 20
        aa_map["Cit"] = 20  # Map citrulline to special token
        aa_map["-"] = 21

        # Handle citrulline (marked as lowercase 'r' or 'cit')
        indices = []
        seq = sequence.upper()
        i = 0
        while i < len(seq):
            if seq[i : i + 3] == "CIT":
                indices.append(20)  # Citrulline
                i += 3
            else:
                indices.append(aa_map.get(seq[i], 20))
                i += 1

        return torch.tensor(indices)

    def forward(
        self,
        native_sequences: list[str],
        citrullinated_sequences: list[str],
    ) -> dict[str, torch.Tensor]:
        """Compute p-adic shift from citrullination.

        Args:
            native_sequences: Original sequences with Arg
            citrullinated_sequences: Sequences with Arg -> Cit

        Returns:
            Dictionary with embeddings and shift values
        """
        batch_size = len(native_sequences)

        # Encode and embed native sequences
        native_encoded = [self.encode_sequence(s) for s in native_sequences]
        max_len = max(len(s) for s in native_encoded)

        # Pad sequences
        native_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
        for i, seq in enumerate(native_encoded):
            native_padded[i, : len(seq)] = seq

        native_emb = self.aa_embedding(native_padded)
        native_out, _ = self.encoder(native_emb)
        native_proj = self.padic_proj(native_out.mean(dim=1))

        # Encode and embed citrullinated sequences
        cit_encoded = [self.encode_sequence(s) for s in citrullinated_sequences]
        cit_padded = torch.zeros(batch_size, max_len, dtype=torch.long)
        for i, seq in enumerate(cit_encoded):
            cit_padded[i, : len(seq)] = seq

        cit_emb = self.aa_embedding(cit_padded)
        cit_out, _ = self.encoder(cit_emb)
        cit_proj = self.padic_proj(cit_out.mean(dim=1))

        # V5.12.2: Compute p-adic shift using hyperbolic or Euclidean distance
        if self.use_hyperbolic:
            shift_magnitude = poincare_distance(native_proj, cit_proj, c=self.curvature)
        else:
            diff = native_proj - cit_proj
            shift_magnitude = torch.norm(diff, dim=-1)

        # Compute p-adic valuation-based distance
        # (simplified - in full implementation would use proper p-adic norm)
        diff = native_proj - cit_proj
        padic_dist = torch.abs(diff).sum(dim=-1) / self.embedding_dim

        return {
            "native_embedding": native_proj,
            "citrullinated_embedding": cit_proj,
            "shift_magnitude": shift_magnitude,
            "padic_distance": padic_dist,
        }


class GoldilocksZoneDetector(nn.Module):
    """Detects whether epitopes fall in the autoimmune Goldilocks Zone.

    The Goldilocks Zone is the p-adic distance range where:
    - Too close: Recognized as self -> tolerance
    - Too far: Recognized as foreign -> cleared efficiently
    - Goldilocks: Ambiguous recognition -> chronic autoimmunity
    """

    def __init__(
        self,
        zone_min: float = 0.15,
        zone_max: float = 0.30,
        p: int = 3,
    ):
        """Initialize detector.

        Args:
            zone_min: Minimum p-adic distance for Goldilocks Zone
            zone_max: Maximum p-adic distance for Goldilocks Zone
            p: Prime for p-adic calculations
        """
        super().__init__()
        self.zone_min = zone_min
        self.zone_max = zone_max
        self.p = p

        # Learnable zone boundaries
        self.boundary_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # [zone_min, zone_max]
            nn.Sigmoid(),
        )

    def is_in_zone(self, padic_distance: torch.Tensor) -> torch.Tensor:
        """Check if distance is in Goldilocks Zone.

        Args:
            padic_distance: p-adic distance values

        Returns:
            Boolean tensor indicating zone membership
        """
        return (padic_distance >= self.zone_min) & (padic_distance <= self.zone_max)

    def zone_risk_score(self, padic_distance: torch.Tensor) -> torch.Tensor:
        """Compute risk score based on proximity to Goldilocks Zone.

        Args:
            padic_distance: p-adic distance values

        Returns:
            Risk scores (0-1, higher = more risky)
        """
        zone_center = (self.zone_min + self.zone_max) / 2
        zone_width = self.zone_max - self.zone_min

        # Gaussian-like scoring centered on zone
        distance_to_center = torch.abs(padic_distance - zone_center)
        risk = torch.exp(-(distance_to_center**2) / (2 * (zone_width / 2) ** 2))

        return risk

    def forward(
        self,
        epitope_features: torch.Tensor,
        padic_distances: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Analyze epitopes for Goldilocks Zone membership.

        Args:
            epitope_features: Feature vectors for epitopes
            padic_distances: p-adic distances from self

        Returns:
            Dictionary with zone analysis
        """
        # Predict adaptive zone boundaries
        boundaries = self.boundary_net(epitope_features)
        adaptive_min = boundaries[:, 0] * 0.5  # Scale to [0, 0.5]
        adaptive_max = boundaries[:, 1] * 0.5 + 0.1  # Scale to [0.1, 0.6]

        # Check zone membership with adaptive boundaries
        in_zone = (padic_distances >= adaptive_min) & (padic_distances <= adaptive_max)

        # Compute risk scores
        risk = self.zone_risk_score(padic_distances)

        return {
            "in_goldilocks_zone": in_zone,
            "risk_score": risk,
            "adaptive_zone_min": adaptive_min,
            "adaptive_zone_max": adaptive_max,
        }


class RheumatoidArthritisAnalyzer:
    """Complete analyzer for rheumatoid arthritis risk and mechanisms.

    Combines citrullination prediction, p-adic shift analysis,
    and Goldilocks Zone detection.
    """

    def __init__(self, p: int = 3):
        """Initialize analyzer.

        Args:
            p: Prime for p-adic calculations
        """
        self.p = p

        self.cit_predictor = CitrullinationPredictor()
        self.padic_shift = PAdicCitrullinationShift(p=p)
        self.goldilocks = GoldilocksZoneDetector(p=p)

    def find_arginine_positions(self, sequence: str) -> list[int]:
        """Find all arginine positions in a sequence.

        Args:
            sequence: Protein sequence

        Returns:
            List of 0-indexed arginine positions
        """
        return [i for i, aa in enumerate(sequence.upper()) if aa == "R"]

    def extract_context(
        self,
        sequence: str,
        position: int,
        context_size: int = 15,
    ) -> str:
        """Extract sequence context around a position.

        Args:
            sequence: Full sequence
            position: Center position
            context_size: Size of context window

        Returns:
            Context sequence
        """
        half_context = context_size // 2
        start = max(0, position - half_context)
        end = min(len(sequence), position + half_context + 1)

        context = sequence[start:end]

        # Pad if necessary
        if len(context) < context_size:
            pad_left = (context_size - len(context)) // 2
            pad_right = context_size - len(context) - pad_left
            context = "-" * pad_left + context + "-" * pad_right

        return context

    def citrullinate_sequence(self, sequence: str, position: int) -> str:
        """Create citrullinated version of sequence.

        Args:
            sequence: Original sequence
            position: Position to citrullinate

        Returns:
            Sequence with Arg -> Cit
        """
        seq_list = list(sequence)
        if seq_list[position].upper() == "R":
            seq_list[position] = "X"  # Use X to represent Cit
        return "".join(seq_list)

    def analyze_protein(
        self,
        protein_name: str,
        sequence: str,
    ) -> list[CitrullinationSite]:
        """Analyze all potential citrullination sites in a protein.

        Args:
            protein_name: Name of protein
            sequence: Protein sequence

        Returns:
            List of CitrullinationSite objects
        """
        sites = []
        arg_positions = self.find_arginine_positions(sequence)

        for pos in arg_positions:
            context = self.extract_context(sequence, pos)
            cit_seq = self.citrullinate_sequence(sequence, pos)

            # Predict citrullination propensity
            with torch.no_grad():
                propensity_result = self.cit_predictor([context])
                propensity = propensity_result["propensity"][0].item()

            # Compute p-adic shift
            native_context = context
            cit_context = self.extract_context(cit_seq, pos)
            with torch.no_grad():
                shift_result = self.padic_shift([native_context], [cit_context])
                padic_dist = shift_result["padic_distance"][0].item()

            # Check Goldilocks Zone
            in_zone = self.goldilocks.zone_min <= padic_dist <= self.goldilocks.zone_max

            # Check if known ACPA target
            known_target = any(protein_name.lower() in target_protein.lower() for target_protein in KNOWN_ACPA_TARGETS)

            sites.append(
                CitrullinationSite(
                    protein_name=protein_name,
                    position=pos,
                    sequence_context=context,
                    padic_distance_to_self=padic_dist,
                    immunogenicity_score=propensity * (1 if in_zone else 0.5),
                    in_goldilocks_zone=in_zone,
                    known_acpa_target=known_target,
                )
            )

        return sites

    def compute_genetic_risk(
        self,
        hla_alleles: list[str],
        padi4_haplotype: str | None = None,
    ) -> tuple[float, list[str]]:
        """Compute genetic risk score for RA.

        Args:
            hla_alleles: List of HLA-DRB1 alleles
            padi4_haplotype: Optional PADI4 haplotype

        Returns:
            Tuple of (risk_score, contributing_alleles)
        """
        risk_score = 1.0
        contributing = []

        # HLA risk
        for allele in hla_alleles:
            if allele in RA_RISK_HLA_ALLELES:
                risk_score *= RA_RISK_HLA_ALLELES[allele]
                contributing.append(allele)

        # PADI4 risk
        if padi4_haplotype:
            if padi4_haplotype in PADI4_HAPLOTYPES["susceptibility"]:
                risk_score *= 1.5
                contributing.append(f"PADI4:{padi4_haplotype}")
            elif padi4_haplotype in PADI4_HAPLOTYPES["protective"]:
                risk_score *= 0.7

        return risk_score, contributing

    def compute_risk_profile(
        self,
        hla_alleles: list[str],
        smoking: bool = False,
        ebv_positive: bool = False,
        padi4_haplotype: str | None = None,
    ) -> RARiskProfile:
        """Compute comprehensive RA risk profile.

        Args:
            hla_alleles: HLA alleles
            smoking: Smoking history
            ebv_positive: EBV infection status
            padi4_haplotype: PADI4 haplotype

        Returns:
            RARiskProfile object
        """
        # Genetic risk
        genetic_risk, contributing = self.compute_genetic_risk(hla_alleles, padi4_haplotype)

        # Check for shared epitope
        shared_epitope = any(allele in RA_RISK_HLA_ALLELES for allele in hla_alleles)

        # Environmental risk
        env_risk = 1.0
        if smoking:
            env_risk *= 2.0  # Smoking doubles risk
        if ebv_positive:
            env_risk *= 1.3

        # Combined risk
        overall_risk = genetic_risk * env_risk

        # Categorize risk
        if overall_risk > 20:
            category = "very_high"
        elif overall_risk > 10:
            category = "high"
        elif overall_risk > 3:
            category = "moderate"
        else:
            category = "low"

        return RARiskProfile(
            hla_alleles=hla_alleles,
            shared_epitope_positive=shared_epitope,
            padi4_haplotype=padi4_haplotype,
            smoking_history=smoking,
            ebv_positive=ebv_positive,
            genetic_risk_score=genetic_risk,
            environmental_risk_score=env_risk,
            overall_risk=overall_risk,
            risk_category=category,
        )

    def analyze_epitope(
        self,
        sequence: str,
        citrullination_positions: list[int],
    ) -> EpitopeAnalysis:
        """Analyze a specific epitope for autoimmune potential.

        Args:
            sequence: Epitope sequence
            citrullination_positions: Positions of citrullination

        Returns:
            EpitopeAnalysis object
        """
        # Create citrullinated version
        cit_seq = list(sequence)
        for pos in citrullination_positions:
            if pos < len(cit_seq) and cit_seq[pos].upper() == "R":
                cit_seq[pos] = "X"
        cit_seq = "".join(cit_seq)

        # Compute p-adic shift
        with torch.no_grad():
            shift_result = self.padic_shift([sequence], [cit_seq])

        # Estimate HLA binding (simplified)
        # In practice, use NetMHCIIpan or similar
        hla_binding = 0.5  # Placeholder

        # Estimate TCR cross-reactivity
        tcr_xreact = shift_result["padic_distance"][0].item()
        if self.goldilocks.zone_min <= tcr_xreact <= self.goldilocks.zone_max:
            tcr_xreact = 0.8  # High cross-reactivity in Goldilocks Zone
        else:
            tcr_xreact = 0.3

        # Predict pathogenicity
        pathogenicity = hla_binding * tcr_xreact

        return EpitopeAnalysis(
            sequence=sequence,
            citrullination_positions=citrullination_positions,
            native_padic_embedding=shift_result["native_embedding"][0],
            citrullinated_padic_embedding=shift_result["citrullinated_embedding"][0],
            padic_shift=shift_result["shift_magnitude"][0].item(),
            hla_binding_affinity=hla_binding,
            tcr_cross_reactivity=tcr_xreact,
            predicted_pathogenicity=pathogenicity,
        )
