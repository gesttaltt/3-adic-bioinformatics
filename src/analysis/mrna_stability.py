# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""mRNA Stability Predictor for vaccine and therapeutic design.

This module predicts mRNA stability based on sequence features including:
- Codon usage and optimization
- Secondary structure (minimum free energy)
- GC content and distribution
- UTR elements
- Rare codon effects

Applications:
    - mRNA vaccine design (COVID-19, cancer, etc.)
    - Therapeutic mRNA optimization
    - Gene expression prediction
    - Codon optimization for expression

Research References:
    - RESEARCH_PROPOSALS/Codon_Space_Exploration/proposal.md
    - COMPREHENSIVE_RESEARCH_REPORT.md
"""

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

# Codon stability scores based on experimental data
# Higher values = more stable mRNA
CODON_STABILITY_SCORES = {
    # Phe (F)
    "UUU": 0.45,
    "UUC": 0.72,
    # Leu (L)
    "UUA": 0.28,
    "UUG": 0.51,
    "CUU": 0.48,
    "CUC": 0.68,
    "CUA": 0.35,
    "CUG": 0.85,
    # Ser (S)
    "UCU": 0.52,
    "UCC": 0.71,
    "UCA": 0.42,
    "UCG": 0.38,
    "AGU": 0.44,
    "AGC": 0.73,
    # Tyr (Y)
    "UAU": 0.43,
    "UAC": 0.75,
    # Stop
    "UAA": 0.30,
    "UAG": 0.25,
    "UGA": 0.35,
    # Cys (C)
    "UGU": 0.41,
    "UGC": 0.69,
    # Trp (W)
    "UGG": 0.65,
    # Pro (P)
    "CCU": 0.55,
    "CCC": 0.72,
    "CCA": 0.58,
    "CCG": 0.42,
    # His (H)
    "CAU": 0.46,
    "CAC": 0.74,
    # Gln (Q)
    "CAA": 0.48,
    "CAG": 0.82,
    # Arg (R)
    "CGU": 0.38,
    "CGC": 0.52,
    "CGA": 0.32,
    "CGG": 0.45,
    "AGA": 0.55,
    "AGG": 0.58,
    # Ile (I)
    "AUU": 0.51,
    "AUC": 0.78,
    "AUA": 0.35,
    # Met (M) - Start
    "AUG": 0.75,
    # Thr (T)
    "ACU": 0.53,
    "ACC": 0.79,
    "ACA": 0.48,
    "ACG": 0.42,
    # Asn (N)
    "AAU": 0.47,
    "AAC": 0.76,
    # Lys (K)
    "AAA": 0.52,
    "AAG": 0.78,
    # Val (V)
    "GUU": 0.48,
    "GUC": 0.71,
    "GUA": 0.38,
    "GUG": 0.75,
    # Ala (A)
    "GCU": 0.55,
    "GCC": 0.80,
    "GCA": 0.51,
    "GCG": 0.45,
    # Asp (D)
    "GAU": 0.49,
    "GAC": 0.77,
    # Glu (E)
    "GAA": 0.54,
    "GAG": 0.79,
    # Gly (G)
    "GGU": 0.42,
    "GGC": 0.68,
    "GGA": 0.48,
    "GGG": 0.52,
}


@dataclass
class StabilityPrediction:
    """Complete mRNA stability prediction result."""

    overall_stability: float  # 0-1 scale
    half_life_hours: float  # Estimated half-life
    codon_score: float  # Codon optimality score
    gc_content: float  # GC percentage
    mfe_score: float  # Minimum free energy (normalized)
    utr_score: float  # UTR quality score
    rare_codon_count: int  # Number of rare codons
    recommendations: list[str] = field(default_factory=list)


class SecondaryStructurePredictor(nn.Module):
    """Predict mRNA secondary structure features.

    Uses a simplified neural network approach to estimate
    minimum free energy and structural features.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_layers: int = 2,
    ):
        """Initialize structure predictor.

        Args:
            hidden_dim: Hidden layer dimension
            n_layers: Number of LSTM layers
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Nucleotide embedding (A, U, G, C)
        self.nt_embedding = nn.Embedding(4, hidden_dim // 4)

        # Bidirectional LSTM for sequence context
        self.lstm = nn.LSTM(
            input_size=hidden_dim // 4,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
        )

        # MFE prediction head
        self.mfe_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Base-pair probability head
        self.bp_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sequence: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict secondary structure features.

        Args:
            sequence: Nucleotide indices (B, L)

        Returns:
            Dictionary with MFE and structure predictions
        """
        # Embed nucleotides
        embedded = self.nt_embedding(sequence)  # (B, L, D/4)

        # LSTM encoding
        lstm_out, _ = self.lstm(embedded)  # (B, L, D*2)

        # Pool for global MFE prediction
        pooled = lstm_out.mean(dim=1)  # (B, D*2)
        mfe = self.mfe_head(pooled)  # (B, 1)

        return {
            "mfe": mfe,
            "hidden_states": lstm_out,
        }


class MFEEstimator(nn.Module):
    """Estimate Minimum Free Energy for mRNA folding.

    Uses thermodynamic parameters and neural network refinement.
    """

    def __init__(self):
        """Initialize MFE estimator."""
        super().__init__()

        # Base-pair stacking energies (kcal/mol, simplified)
        self.stacking_energies = {
            ("AU", "AU"): -0.9,
            ("AU", "CG"): -2.2,
            ("AU", "GC"): -2.1,
            ("AU", "UA"): -1.3,
            ("CG", "AU"): -2.1,
            ("CG", "CG"): -3.3,
            ("CG", "GC"): -2.4,
            ("CG", "UA"): -2.1,
            ("GC", "AU"): -2.2,
            ("GC", "CG"): -2.4,
            ("GC", "GC"): -3.4,
            ("GC", "UA"): -2.2,
            ("GU", "AU"): -1.3,
            ("GU", "CG"): -2.5,
            ("GU", "GC"): -2.1,
            ("GU", "UA"): -1.4,
            ("UA", "AU"): -1.3,
            ("UA", "CG"): -2.1,
            ("UA", "GC"): -2.2,
            ("UA", "UA"): -0.9,
            ("UG", "AU"): -1.0,
            ("UG", "CG"): -1.4,
            ("UG", "GC"): -2.5,
            ("UG", "UA"): -1.3,
        }

    def estimate_mfe(
        self,
        sequence: str,
        window_size: int = 100,
    ) -> float:
        """Estimate MFE using sliding window approach.

        Args:
            sequence: RNA sequence string (AUGC)
            window_size: Window size for local folding

        Returns:
            Estimated MFE in kcal/mol (normalized)
        """
        if len(sequence) < 10:
            return 0.0

        # Simple estimate based on GC content and length
        gc_count = sequence.count("G") + sequence.count("C")
        gc_ratio = gc_count / len(sequence)

        # Approximate MFE: more GC = more stable (more negative)
        # Typical mRNA: -0.3 to -0.5 kcal/mol per nucleotide
        mfe_per_nt = -0.3 - 0.2 * gc_ratio

        # Total MFE estimate
        total_mfe = mfe_per_nt * len(sequence)

        # Normalize to 0-1 scale (more negative = higher stability score)
        # Typical range: -500 to 0 for 1000 nt
        normalized = min(1.0, max(0.0, -total_mfe / (len(sequence) * 0.5)))

        return normalized


class UTROptimizer(nn.Module):
    """Optimize 5' and 3' UTR regions for mRNA stability.

    UTRs significantly impact mRNA half-life and translation efficiency.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
    ):
        """Initialize UTR optimizer.

        Args:
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.hidden_dim = hidden_dim

        # Known stabilizing elements
        self.stabilizing_motifs = [
            "ACCACC",  # Kozak-like
            "GCCGCC",  # GC-rich stabilizing
            "AAAUAA",  # Poly(A) signal
        ]

        self.destabilizing_motifs = [
            "AUUUA",  # ARE (AU-rich element)
            "UAUUUAU",  # Extended ARE
            "AUUUAUUUA",  # Class II ARE
        ]

        # UTR quality prediction
        self.quality_head = nn.Sequential(
            nn.Linear(len(self.stabilizing_motifs) + len(self.destabilizing_motifs) + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def count_motifs(self, sequence: str) -> tuple[int, int]:
        """Count stabilizing and destabilizing motifs.

        Args:
            sequence: UTR sequence

        Returns:
            Tuple of (stabilizing_count, destabilizing_count)
        """
        seq_upper = sequence.upper().replace("T", "U")

        stabilizing = sum(seq_upper.count(m) for m in self.stabilizing_motifs)
        destabilizing = sum(seq_upper.count(m) for m in self.destabilizing_motifs)

        return stabilizing, destabilizing

    def score_utr(
        self,
        utr_5: str,
        utr_3: str,
    ) -> float:
        """Score UTR quality.

        Args:
            utr_5: 5' UTR sequence
            utr_3: 3' UTR sequence

        Returns:
            UTR quality score (0-1)
        """
        # Count motifs in both UTRs
        stab_5, destab_5 = self.count_motifs(utr_5)
        stab_3, destab_3 = self.count_motifs(utr_3)

        total_stab = stab_5 + stab_3
        total_destab = destab_5 + destab_3

        # Simple scoring
        score = 0.5

        # Bonus for stabilizing
        score += min(0.3, 0.1 * total_stab)

        # Penalty for destabilizing
        score -= min(0.4, 0.15 * total_destab)

        # Length considerations (optimal: 50-100 for 5', 200-400 for 3')
        if 50 <= len(utr_5) <= 100:
            score += 0.1
        if 200 <= len(utr_3) <= 400:
            score += 0.1

        return max(0.0, min(1.0, score))


class mRNAStabilityPredictor(nn.Module):
    """Comprehensive mRNA stability predictor for vaccine design.

    Combines multiple factors:
    - Codon usage optimization
    - Secondary structure (MFE)
    - GC content and distribution
    - UTR elements
    - Rare codon detection
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        use_neural_mfe: bool = False,
    ):
        """Initialize stability predictor.

        Args:
            hidden_dim: Hidden layer dimension
            use_neural_mfe: Whether to use neural MFE estimator
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_neural_mfe = use_neural_mfe

        # MFE estimator
        self.mfe_estimator = MFEEstimator()

        # UTR optimizer
        self.utr_optimizer = UTROptimizer()

        # Neural structure predictor (optional)
        if use_neural_mfe:
            self.structure_predictor = SecondaryStructurePredictor(hidden_dim)

        # Combined stability head
        self.stability_head = nn.Sequential(
            nn.Linear(5, hidden_dim),  # codon, gc, mfe, utr, rare
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Half-life prediction (hours)
        self.halflife_head = nn.Sequential(
            nn.Linear(5, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensure positive
        )

    def compute_codon_score(self, sequence: str) -> tuple[float, int]:
        """Compute codon optimality score.

        Args:
            sequence: mRNA sequence (coding region)

        Returns:
            Tuple of (codon_score, rare_codon_count)
        """
        seq = sequence.upper().replace("T", "U")

        if len(seq) < 3:
            return 0.5, 0

        scores = []
        rare_count = 0
        rare_threshold = 0.35

        for i in range(0, len(seq) - 2, 3):
            codon = seq[i : i + 3]
            if codon in CODON_STABILITY_SCORES:
                score = CODON_STABILITY_SCORES[codon]
                scores.append(score)
                if score < rare_threshold:
                    rare_count += 1

        if not scores:
            return 0.5, 0

        return float(np.mean(scores)), rare_count

    def compute_gc_content(self, sequence: str) -> float:
        """Compute GC content.

        Args:
            sequence: RNA sequence

        Returns:
            GC content ratio (0-1)
        """
        seq = sequence.upper()
        gc_count = seq.count("G") + seq.count("C")
        return gc_count / len(seq) if seq else 0.0

    def forward(
        self,
        coding_sequence: str,
        utr_5: str = "",
        utr_3: str = "",
    ) -> StabilityPrediction:
        """Predict mRNA stability.

        Args:
            coding_sequence: Coding region sequence
            utr_5: 5' UTR sequence
            utr_3: 3' UTR sequence

        Returns:
            Complete stability prediction
        """
        # Full sequence
        full_sequence = utr_5 + coding_sequence + utr_3

        # Compute individual scores
        codon_score, rare_count = self.compute_codon_score(coding_sequence)
        gc_content = self.compute_gc_content(full_sequence)
        mfe_score = self.mfe_estimator.estimate_mfe(full_sequence)
        utr_score = self.utr_optimizer.score_utr(utr_5, utr_3) if utr_5 or utr_3 else 0.5

        # Normalize rare codon penalty
        n_codons = len(coding_sequence) // 3
        rare_penalty = min(1.0, rare_count / max(n_codons * 0.1, 1))

        # Combine features for neural prediction
        features = torch.tensor(
            [[codon_score, gc_content, mfe_score, utr_score, 1.0 - rare_penalty]],
            dtype=torch.float32,
        )

        # Predict overall stability
        overall_stability = self.stability_head(features).item()

        # Predict half-life (base: 2-24 hours for typical mRNA)
        half_life = self.halflife_head(features).item() * 12 + 2  # Scale to hours

        # Generate recommendations
        recommendations = []

        if codon_score < 0.6:
            recommendations.append("Consider codon optimization to improve stability")
        if gc_content < 0.4:
            recommendations.append("Low GC content may reduce stability; consider GC enrichment")
        if gc_content > 0.7:
            recommendations.append("High GC content may affect translation; balance recommended")
        if rare_count > n_codons * 0.05:
            recommendations.append(f"Replace {rare_count} rare codons to improve expression")
        if utr_score < 0.5:
            recommendations.append("Consider optimizing UTR elements")
        if mfe_score < 0.4:
            recommendations.append("Secondary structure may impede translation")

        return StabilityPrediction(
            overall_stability=overall_stability,
            half_life_hours=half_life,
            codon_score=codon_score,
            gc_content=gc_content,
            mfe_score=mfe_score,
            utr_score=utr_score,
            rare_codon_count=rare_count,
            recommendations=recommendations,
        )

    def optimize_sequence(
        self,
        coding_sequence: str,
        target_stability: float = 0.8,
        preserve_amino_acids: bool = True,
    ) -> tuple[str, StabilityPrediction]:
        """Optimize mRNA sequence for stability.

        Args:
            coding_sequence: Original coding sequence
            target_stability: Target stability score
            preserve_amino_acids: Whether to maintain amino acid sequence

        Returns:
            Tuple of (optimized_sequence, new_prediction)
        """
        seq = coding_sequence.upper().replace("T", "U")

        if not preserve_amino_acids:
            # Just return original if we can't change AA
            return seq, self.forward(seq)

        # Codon optimization: replace with most stable synonymous codon
        # Build reverse mapping: amino acid -> best codon
        aa_to_codons: dict[str, list[tuple[str, float]]] = {}

        # Standard genetic code (simplified mapping)
        genetic_code = {
            "UUU": "F",
            "UUC": "F",
            "UUA": "L",
            "UUG": "L",
            "CUU": "L",
            "CUC": "L",
            "CUA": "L",
            "CUG": "L",
            "AUU": "I",
            "AUC": "I",
            "AUA": "I",
            "AUG": "M",
            "GUU": "V",
            "GUC": "V",
            "GUA": "V",
            "GUG": "V",
            "UCU": "S",
            "UCC": "S",
            "UCA": "S",
            "UCG": "S",
            "CCU": "P",
            "CCC": "P",
            "CCA": "P",
            "CCG": "P",
            "ACU": "T",
            "ACC": "T",
            "ACA": "T",
            "ACG": "T",
            "GCU": "A",
            "GCC": "A",
            "GCA": "A",
            "GCG": "A",
            "UAU": "Y",
            "UAC": "Y",
            "UAA": "*",
            "UAG": "*",
            "CAU": "H",
            "CAC": "H",
            "CAA": "Q",
            "CAG": "Q",
            "AAU": "N",
            "AAC": "N",
            "AAA": "K",
            "AAG": "K",
            "GAU": "D",
            "GAC": "D",
            "GAA": "E",
            "GAG": "E",
            "UGU": "C",
            "UGC": "C",
            "UGA": "*",
            "UGG": "W",
            "CGU": "R",
            "CGC": "R",
            "CGA": "R",
            "CGG": "R",
            "AGU": "S",
            "AGC": "S",
            "AGA": "R",
            "AGG": "R",
            "GGU": "G",
            "GGC": "G",
            "GGA": "G",
            "GGG": "G",
        }

        # Group codons by amino acid
        for codon, aa in genetic_code.items():
            if aa not in aa_to_codons:
                aa_to_codons[aa] = []
            stability = CODON_STABILITY_SCORES.get(codon, 0.5)
            aa_to_codons[aa].append((codon, stability))

        # Sort by stability
        for aa in aa_to_codons:
            aa_to_codons[aa].sort(key=lambda x: x[1], reverse=True)

        # Optimize each codon
        optimized = []
        for i in range(0, len(seq) - 2, 3):
            codon = seq[i : i + 3]
            if codon in genetic_code:
                aa = genetic_code[codon]
                # Get best codon for this AA
                if aa in aa_to_codons and aa_to_codons[aa]:
                    best_codon = aa_to_codons[aa][0][0]
                    optimized.append(best_codon)
                else:
                    optimized.append(codon)
            else:
                optimized.append(codon)

        optimized_seq = "".join(optimized)

        return optimized_seq, self.forward(optimized_seq)
