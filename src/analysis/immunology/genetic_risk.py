# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""HLA genetic risk computation utilities.

Consolidates duplicated compute_genetic_risk() functions from:
- src/diseases/multiple_sclerosis.py (lines 460-484)
- src/diseases/rheumatoid_arthritis.py (lines 586-617)

Provides a unified interface for computing HLA-based genetic risk across diseases.
"""

from dataclasses import dataclass, field

from .types import HLAAlleleRisk

# Disease-specific HLA risk allele dictionaries
# Each entry is allele -> odds ratio (>1 = risk, <1 = protective)

MS_HLA_RISK_ALLELES: dict[str, float] = {
    "DRB1*15:01": 3.1,  # Major MS risk allele
    "DRB1*15:03": 2.8,
    "DRB1*03:01": 1.5,
    "DRB1*04:05": 0.7,  # Protective
    "DRB1*14:01": 0.6,  # Protective
    "A*02:01": 0.8,  # Protective (Class I)
}

RA_HLA_RISK_ALLELES: dict[str, float] = {
    "DRB1*04:01": 5.0,  # Shared epitope
    "DRB1*04:04": 4.5,
    "DRB1*04:05": 4.0,
    "DRB1*01:01": 3.5,
    "DRB1*10:01": 3.0,
    "DRB1*04:08": 2.5,
    "DRB1*14:02": 2.0,
}

T1D_HLA_RISK_ALLELES: dict[str, float] = {
    "DRB1*03:01-DQB1*02:01": 3.6,  # DR3-DQ2
    "DRB1*04:01-DQB1*03:02": 11.4,  # DR4-DQ8
    "DRB1*04:05-DQB1*03:02": 8.5,
    "DRB1*15:01-DQB1*06:02": 0.15,  # Protective
}

CELIAC_HLA_RISK_ALLELES: dict[str, float] = {
    "DQ2.5": 7.0,  # DQA1*05:01/DQB1*02:01
    "DQ8": 2.5,  # DQA1*03:01/DQB1*03:02
    "DQ2.2": 2.0,
}

# Registry of all disease HLA profiles
HLA_RISK_REGISTRY: dict[str, dict[str, float]] = {
    "multiple_sclerosis": MS_HLA_RISK_ALLELES,
    "rheumatoid_arthritis": RA_HLA_RISK_ALLELES,
    "type1_diabetes": T1D_HLA_RISK_ALLELES,
    "celiac": CELIAC_HLA_RISK_ALLELES,
}


@dataclass
class HLARiskProfile:
    """Result of HLA-based genetic risk assessment.

    Attributes:
        risk_score: Combined multiplicative risk score
        risk_alleles: List of risk alleles present
        protective_alleles: List of protective alleles present
        disease: Disease assessed for
        additional_factors: Extra genetic factors (e.g., PADI4 for RA)
    """

    risk_score: float
    risk_alleles: list[HLAAlleleRisk] = field(default_factory=list)
    protective_alleles: list[HLAAlleleRisk] = field(default_factory=list)
    disease: str = ""
    additional_factors: dict[str, float] = field(default_factory=dict)

    @property
    def is_high_risk(self) -> bool:
        """Check if profile indicates high genetic risk."""
        return self.risk_score > 3.0

    @property
    def is_protective(self) -> bool:
        """Check if profile indicates protective genetics."""
        return self.risk_score < 0.7 and len(self.protective_alleles) > 0


def compute_hla_genetic_risk(
    hla_alleles: list[str],
    disease: str,
    additional_factors: dict[str, float] | None = None,
) -> HLARiskProfile:
    """Compute HLA-based genetic risk for a disease.

    This is the unified replacement for:
    - MultipleSclerosisAnalyzer.compute_genetic_risk()
    - RheumatoidArthritisAnalyzer.compute_genetic_risk()

    Args:
        hla_alleles: List of HLA alleles present in individual
        disease: Disease to assess risk for (must be in HLA_RISK_REGISTRY)
        additional_factors: Optional additional genetic factors and their risk

    Returns:
        HLARiskProfile with computed risk score and allele classifications
    """
    if disease not in HLA_RISK_REGISTRY:
        available = ", ".join(HLA_RISK_REGISTRY.keys())
        raise ValueError(f"Unknown disease: {disease}. Available: {available}")

    risk_alleles_dict = HLA_RISK_REGISTRY[disease]

    # Compute multiplicative risk score
    risk_score = 1.0
    risk_alleles = []
    protective_alleles = []

    for allele in hla_alleles:
        # Normalize allele format
        normalized = _normalize_hla_allele(allele)

        if normalized in risk_alleles_dict:
            odds_ratio = risk_alleles_dict[normalized]
            risk_score *= odds_ratio

            allele_risk = HLAAlleleRisk(
                allele=normalized,
                odds_ratio=odds_ratio,
                disease=disease,
            )

            if odds_ratio >= 1.0:
                risk_alleles.append(allele_risk)
            else:
                protective_alleles.append(allele_risk)

    # Apply additional factors
    additional = additional_factors or {}
    for _factor, risk in additional.items():
        risk_score *= risk

    return HLARiskProfile(
        risk_score=risk_score,
        risk_alleles=risk_alleles,
        protective_alleles=protective_alleles,
        disease=disease,
        additional_factors=additional,
    )


def _normalize_hla_allele(allele: str) -> str:
    """Normalize HLA allele format.

    Handles common format variations:
    - HLA-DRB1*15:01 -> DRB1*15:01
    - DRB1*1501 -> DRB1*15:01
    - drb1*15:01 -> DRB1*15:01
    """
    # Remove HLA- prefix
    if allele.upper().startswith("HLA-"):
        allele = allele[4:]

    # Uppercase
    allele = allele.upper()

    # Add colon if missing in 4-digit format
    if "*" in allele:
        parts = allele.split("*")
        if len(parts) == 2 and ":" not in parts[1] and len(parts[1]) == 4:
            parts[1] = parts[1][:2] + ":" + parts[1][2:]
            allele = "*".join(parts)

    return allele


def get_available_diseases() -> list[str]:
    """Get list of diseases with HLA risk profiles available."""
    return list(HLA_RISK_REGISTRY.keys())


def register_disease_hla_profile(disease: str, alleles: dict[str, float]) -> None:
    """Register a new disease HLA risk profile.

    Args:
        disease: Disease name
        alleles: Dictionary of allele -> odds ratio
    """
    HLA_RISK_REGISTRY[disease] = alleles


__all__ = [
    "MS_HLA_RISK_ALLELES",
    "RA_HLA_RISK_ALLELES",
    "T1D_HLA_RISK_ALLELES",
    "CELIAC_HLA_RISK_ALLELES",
    "HLA_RISK_REGISTRY",
    "HLARiskProfile",
    "compute_hla_genetic_risk",
    "get_available_diseases",
    "register_disease_hla_profile",
]
