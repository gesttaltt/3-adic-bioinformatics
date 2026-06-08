"""TAM-aware encoding for NRTI drug resistance prediction.

Thymidine Analogue Mutations (TAMs) create complex cross-resistance patterns
that standard one-hot encoding fails to capture. This module provides:

1. Explicit TAM pattern detection
2. Cross-resistance pathway features
3. Mutation covariance features
4. Combined encoding for improved NRTI prediction

Reference: Stanford HIVDB TAM interpretation guidelines.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MutationPattern:
    """A pattern of mutations with associated resistance."""

    name: str
    mutations: list[str]  # e.g., ["M41L", "L210W", "T215Y"]
    affected_drugs: list[str]
    resistance_level: str  # "low", "intermediate", "high"
    description: str


# TAM pathway definitions based on Stanford HIVDB
TAM_PATHWAYS = {
    # TAM-1 pathway (more common, higher fitness)
    "TAM1": MutationPattern(
        name="TAM-1 Pathway",
        mutations=["M41L", "L210W", "T215Y"],
        affected_drugs=["AZT", "D4T", "ABC", "TDF", "DDI"],
        resistance_level="high",
        description="Classical TAM-1 pathway with T215Y",
    ),
    # TAM-2 pathway (associated with K70R)
    "TAM2": MutationPattern(
        name="TAM-2 Pathway",
        mutations=["D67N", "K70R", "T215F", "K219Q", "K219E"],
        affected_drugs=["AZT", "D4T", "ABC"],
        resistance_level="intermediate",
        description="TAM-2 pathway with K70R and T215F",
    ),
    # Q151M multi-drug resistance complex
    "Q151M_COMPLEX": MutationPattern(
        name="Q151M Complex",
        mutations=["A62V", "V75I", "F77L", "F116Y", "Q151M"],
        affected_drugs=["AZT", "D4T", "DDI", "ABC", "3TC", "FTC"],
        resistance_level="high",
        description="Multi-NRTI resistance complex",
    ),
    # T69 insertion complex
    "T69_INSERTION": MutationPattern(
        name="T69 Insertion",
        mutations=["T69ins"],  # Any insertion at position 69
        affected_drugs=["AZT", "D4T", "DDI", "ABC", "3TC", "FTC", "TDF"],
        resistance_level="high",
        description="T69 insertion causes multi-NRTI resistance",
    ),
    # K65R pathway (TDF resistance)
    "K65R": MutationPattern(
        name="K65R Pathway",
        mutations=["K65R"],
        affected_drugs=["TDF", "ABC", "DDI", "3TC", "FTC"],
        resistance_level="high",
        description="Primary TDF resistance mutation",
    ),
    # M184V/I (3TC/FTC resistance)
    "M184": MutationPattern(
        name="M184V/I",
        mutations=["M184V", "M184I"],
        affected_drugs=["3TC", "FTC", "ABC"],
        resistance_level="high",
        description="Primary 3TC/FTC resistance, hypersusceptibility to AZT/TDF",
    ),
    # L74V (DDI/ABC resistance)
    "L74V": MutationPattern(
        name="L74V",
        mutations=["L74V", "L74I"],
        affected_drugs=["DDI", "ABC"],
        resistance_level="intermediate",
        description="DDI and ABC resistance",
    ),
    # Y115F (ABC resistance)
    "Y115F": MutationPattern(
        name="Y115F",
        mutations=["Y115F"],
        affected_drugs=["ABC"],
        resistance_level="intermediate",
        description="ABC-specific resistance",
    ),
}

# Cross-resistance relationships
CROSS_RESISTANCE_MATRIX = {
    # (drug1, drug2): correlation coefficient from literature
    ("AZT", "D4T"): 0.85,  # Strong cross-resistance
    ("3TC", "FTC"): 0.95,  # Almost complete cross-resistance
    ("TDF", "ABC"): 0.60,  # Moderate cross-resistance
    ("DDI", "ABC"): 0.55,
    ("AZT", "TDF"): -0.20,  # Antagonistic (M184V increases AZT susceptibility)
}

# RT positions relevant for NRTI resistance
NRTI_KEY_POSITIONS = [
    41,
    44,
    62,
    65,
    67,
    69,
    70,
    74,
    75,
    77,
    115,
    116,
    118,
    151,
    184,
    210,
    215,
    219,
]


def parse_mutation(mutation: str) -> tuple[str, int, str]:
    """Parse mutation string like 'M41L' into (ref_aa, position, mut_aa)."""
    if len(mutation) < 3:
        return ("", 0, "")

    ref_aa = mutation[0]
    mut_aa = mutation[-1]
    position = int(mutation[1:-1])
    return (ref_aa, position, mut_aa)


def detect_tam_patterns(sequence_mutations: set[str]) -> dict[str, float]:
    """Detect TAM patterns in a mutation set.

    Args:
        sequence_mutations: Set of mutations like {"M41L", "L210W", "T215Y"}

    Returns:
        Dictionary mapping pattern name to presence score (0-1)
    """
    pattern_scores = {}

    for pattern_name, pattern in TAM_PATHWAYS.items():
        # Count how many mutations from the pattern are present
        present = 0
        for mut in pattern.mutations:
            if mut in sequence_mutations:
                present += 1
            elif mut.endswith("ins"):
                # Check for insertions at position
                pos = int(mut[1:-3])
                if any(f"{pos}ins" in m or f"T{pos}" in m for m in sequence_mutations):
                    present += 1

        # Score is fraction of pattern present
        score = present / len(pattern.mutations) if pattern.mutations else 0
        pattern_scores[pattern_name] = score

    return pattern_scores


def extract_tam_features(row: pd.Series, position_cols: list[str]) -> np.ndarray:
    """Extract TAM-aware features from a data row.

    Args:
        row: DataFrame row with position columns
        position_cols: List of position column names (e.g., ["RT41", "RT65", ...])

    Returns:
        Feature vector with TAM pattern indicators
    """
    # Reference amino acids at key positions (wild-type)
    REFERENCE = {
        41: "M",
        44: "E",
        62: "A",
        65: "K",
        67: "D",
        69: "T",
        70: "K",
        74: "L",
        75: "V",
        77: "F",
        115: "Y",
        116: "F",
        118: "V",
        151: "Q",
        184: "M",
        210: "L",
        215: "T",
        219: "K",
    }

    # Detect mutations
    mutations = set()
    for col in position_cols:
        if col.startswith("RT"):
            try:
                pos = int(col[2:])
                aa = str(row[col]).upper() if pd.notna(row[col]) else ""
                if aa and pos in REFERENCE and aa != REFERENCE[pos] and aa != "-":
                    mutations.add(f"{REFERENCE[pos]}{pos}{aa}")
            except (ValueError, KeyError):
                continue

    # Get pattern scores
    pattern_scores = detect_tam_patterns(mutations)

    # Create feature vector
    features = []

    # 1. TAM pattern presence scores (8 patterns)
    for pattern_name in TAM_PATHWAYS:
        features.append(pattern_scores.get(pattern_name, 0.0))

    # 2. Total TAM count
    tam_mutations = {"M41L", "D67N", "K70R", "L210W", "T215Y", "T215F", "K219Q", "K219E"}
    tam_count = len(mutations & tam_mutations)
    features.append(tam_count / len(tam_mutations))  # Normalized

    # 3. Key position indicators (binary)
    for pos in NRTI_KEY_POSITIONS:
        col = f"RT{pos}"
        if col in position_cols:
            aa = str(row.get(col, "")).upper()
            ref = REFERENCE.get(pos, "")
            is_mutated = 1.0 if aa and aa != ref and aa != "-" else 0.0
            features.append(is_mutated)
        else:
            features.append(0.0)

    # 4. Pathway interaction features
    # TAM1 + TAM2 co-occurrence
    tam1_score = pattern_scores.get("TAM1", 0)
    tam2_score = pattern_scores.get("TAM2", 0)
    features.append(tam1_score * tam2_score)  # Interaction term

    # Q151M + TAMs (particularly bad)
    q151m_score = pattern_scores.get("Q151M_COMPLEX", 0)
    features.append(q151m_score * (tam1_score + tam2_score) / 2)

    # M184V + TAMs (complex interaction)
    m184_score = pattern_scores.get("M184", 0)
    features.append(m184_score * tam1_score)  # M184V reduces TAM1 impact on some drugs

    # K65R + M184V (antagonistic)
    k65r_score = pattern_scores.get("K65R", 0)
    features.append(k65r_score * m184_score)

    return np.array(features, dtype=np.float32)


class TAMAwareEncoder:
    """Encoder that combines one-hot encoding with TAM features."""

    def __init__(self, position_cols: list[str], aa_alphabet: str = "ACDEFGHIKLMNPQRSTVWY*-"):
        """Initialize encoder.

        Args:
            position_cols: List of position column names
            aa_alphabet: Amino acid alphabet for one-hot encoding
        """
        self.position_cols = position_cols
        self.aa_alphabet = aa_alphabet
        self.aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}

        # Calculate feature dimensions
        self.n_positions = len(position_cols)
        self.n_aa = len(aa_alphabet)
        self.onehot_dim = self.n_positions * self.n_aa

        # TAM features: 8 patterns + 1 count + 18 positions + 4 interactions = 31
        self.tam_dim = len(TAM_PATHWAYS) + 1 + len(NRTI_KEY_POSITIONS) + 4

        self.total_dim = self.onehot_dim + self.tam_dim

    def encode_onehot(self, row: pd.Series) -> np.ndarray:
        """One-hot encode a sequence row."""
        encoded = np.zeros(self.onehot_dim, dtype=np.float32)

        for j, col in enumerate(self.position_cols):
            aa = str(row[col]).upper() if pd.notna(row[col]) else "-"
            if aa in self.aa_to_idx:
                encoded[j * self.n_aa + self.aa_to_idx[aa]] = 1.0
            else:
                encoded[j * self.n_aa + self.aa_to_idx["-"]] = 1.0

        return encoded

    def encode(self, row: pd.Series) -> np.ndarray:
        """Encode a sequence row with both one-hot and TAM features."""
        onehot = self.encode_onehot(row)
        tam_features = extract_tam_features(row, self.position_cols)
        return np.concatenate([onehot, tam_features])

    def encode_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Encode entire dataframe."""
        n_samples = len(df)
        encoded = np.zeros((n_samples, self.total_dim), dtype=np.float32)

        for idx, (_, row) in enumerate(df.iterrows()):
            encoded[idx] = self.encode(row)

        return encoded


def create_drug_specific_features(drug: str) -> list[str]:
    """Get TAM patterns most relevant for a specific drug.

    Args:
        drug: Drug code (e.g., "AZT", "TDF", "3TC")

    Returns:
        List of pattern names relevant for this drug
    """
    relevant_patterns = []

    for pattern_name, pattern in TAM_PATHWAYS.items():
        if drug in pattern.affected_drugs:
            relevant_patterns.append(pattern_name)

    return relevant_patterns


def get_expected_resistance_impact(drug: str) -> dict[str, str]:
    """Get expected impact of each TAM pattern on a drug.

    Returns dict mapping pattern name to impact level.
    """
    impacts = {}

    for pattern_name, pattern in TAM_PATHWAYS.items():
        if drug in pattern.affected_drugs:
            impacts[pattern_name] = pattern.resistance_level
        else:
            impacts[pattern_name] = "none"

    return impacts


class NRTIFeatureExtractor:
    """Complete feature extractor for NRTI resistance prediction."""

    def __init__(self, position_cols: list[str], target_drug: str):
        """Initialize extractor.

        Args:
            position_cols: List of RT position columns
            target_drug: Target drug for drug-specific features
        """
        self.position_cols = position_cols
        self.target_drug = target_drug
        self.tam_encoder = TAMAwareEncoder(position_cols)

        # Drug-specific pattern weights
        self.relevant_patterns = create_drug_specific_features(target_drug)

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract all features for NRTI prediction."""
        # Base TAM-aware encoding
        base_features = self.tam_encoder.encode_dataframe(df)

        # Could add drug-specific feature weighting here
        return base_features

    @property
    def feature_dim(self) -> int:
        return self.tam_encoder.total_dim


if __name__ == "__main__":
    print("TAM-Aware Encoder for NRTI Resistance")
    print("=" * 60)

    # Test pattern detection
    test_mutations = {"M41L", "L210W", "T215Y", "M184V"}
    print(f"\nTest mutations: {test_mutations}")

    patterns = detect_tam_patterns(test_mutations)
    print("\nDetected patterns:")
    for name, score in sorted(patterns.items(), key=lambda x: -x[1]):
        if score > 0:
            print(f"  {name}: {score:.2f}")

    # Test drug relevance
    print("\nDrug-specific relevant patterns:")
    for drug in ["AZT", "TDF", "3TC"]:
        relevant = create_drug_specific_features(drug)
        print(f"  {drug}: {relevant}")

    # Show feature dimensions
    test_cols = [f"RT{p}" for p in range(1, 561)]
    encoder = TAMAwareEncoder(test_cols)
    print("\nFeature dimensions:")
    print(f"  One-hot: {encoder.onehot_dim}")
    print(f"  TAM features: {encoder.tam_dim}")
    print(f"  Total: {encoder.total_dim}")

    print("\n" + "=" * 60)
    print("TAM-aware encoding ready!")
