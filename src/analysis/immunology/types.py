# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Shared dataclasses for immunology analysis.

Consolidates duplicated epitope and HLA-related dataclasses from:
- src/diseases/multiple_sclerosis.py (EpitopePair)
- src/diseases/rheumatoid_arthritis.py (CitrullinationSite, EpitopeAnalysis)
"""

from dataclasses import dataclass, field

import torch


@dataclass
class HLAAlleleRisk:
    """HLA allele with associated disease risk.

    Attributes:
        allele: HLA allele name (e.g., "DRB1*15:01")
        odds_ratio: Disease odds ratio (>1 = risk, <1 = protective)
        disease: Disease this risk applies to
        is_protective: Whether this allele is protective
    """

    allele: str
    odds_ratio: float
    disease: str
    is_protective: bool = field(init=False)

    def __post_init__(self):
        self.is_protective = self.odds_ratio < 1.0


@dataclass
class EpitopeAnalysisResult:
    """Unified epitope analysis result.

    Consolidates fields from:
    - EpitopePair (multiple_sclerosis.py)
    - EpitopeAnalysis (rheumatoid_arthritis.py)
    - CitrullinationSite (rheumatoid_arthritis.py)

    Attributes:
        sequence: Epitope sequence
        source: Source protein or pathogen
        target: Self target (if applicable for mimicry)
        padic_distance: P-adic distance from self or reference
        padic_embedding: Optional p-adic embedding tensor
        hla_binding_score: HLA binding affinity (0-1)
        tcr_cross_reactivity: TCR cross-reactivity risk (0-1)
        immunogenicity_score: Overall immunogenicity (0-1)
        in_goldilocks_zone: Whether in autoimmune risk zone
        modification_positions: Positions of PTMs (e.g., citrullination)
        modification_type: Type of modification (e.g., "citrullination")
        additional_data: Extra disease-specific fields
    """

    sequence: str
    source: str = ""
    target: str = ""
    padic_distance: float = 0.0
    padic_embedding: torch.Tensor | None = None
    hla_binding_score: float = 0.0
    tcr_cross_reactivity: float = 0.0
    immunogenicity_score: float = 0.0
    in_goldilocks_zone: bool = False
    modification_positions: list[int] = field(default_factory=list)
    modification_type: str | None = None
    additional_data: dict = field(default_factory=dict)

    @property
    def is_high_risk(self) -> bool:
        """Check if epitope is high risk for autoimmunity."""
        return self.in_goldilocks_zone and self.tcr_cross_reactivity > 0.5

    @property
    def pathogenicity_score(self) -> float:
        """Compute overall pathogenicity score."""
        # Weighted combination of risk factors
        weights = {
            "goldilocks": 0.3,
            "hla_binding": 0.25,
            "tcr_xreact": 0.25,
            "immunogenicity": 0.2,
        }
        score = (
            weights["goldilocks"] * float(self.in_goldilocks_zone)
            + weights["hla_binding"] * self.hla_binding_score
            + weights["tcr_xreact"] * self.tcr_cross_reactivity
            + weights["immunogenicity"] * self.immunogenicity_score
        )
        return min(1.0, max(0.0, score))


__all__ = [
    "HLAAlleleRisk",
    "EpitopeAnalysisResult",
]
