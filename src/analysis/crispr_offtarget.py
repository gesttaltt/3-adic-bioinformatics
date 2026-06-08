# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""CRISPR Off-Target Landscape Analysis - Backward Compatibility.

This module re-exports from the new modular crispr package.
For new code, import directly from src.analysis.crispr instead.

Example:
    # Preferred (new style):
    from src.analysis.crispr import CRISPROfftargetAnalyzer

    # Legacy (still works):
    from src.analysis.crispr_offtarget import CRISPROfftargetAnalyzer
"""

from src.analysis.crispr import (
    IDX_TO_NUCLEOTIDE,
    NUCLEOTIDE_TO_IDX,
    PAM_SEQUENCES,
    POSITION_WEIGHTS,
    CRISPROfftargetAnalyzer,
    GuideDesignOptimizer,
    GuideSafetyProfile,
    HyperbolicOfftargetEmbedder,
    MismatchType,
    OfftargetActivityPredictor,
    OffTargetSite,
    PAdicSequenceDistance,
    encode_sequence,
    sequence_to_onehot,
)

__all__ = [
    "MismatchType",
    "OffTargetSite",
    "GuideSafetyProfile",
    "NUCLEOTIDE_TO_IDX",
    "IDX_TO_NUCLEOTIDE",
    "PAM_SEQUENCES",
    "POSITION_WEIGHTS",
    "encode_sequence",
    "sequence_to_onehot",
    "PAdicSequenceDistance",
    "HyperbolicOfftargetEmbedder",
    "OfftargetActivityPredictor",
    "CRISPROfftargetAnalyzer",
    "GuideDesignOptimizer",
]
