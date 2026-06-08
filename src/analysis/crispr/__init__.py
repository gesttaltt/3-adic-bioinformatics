# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""CRISPR Off-Target Landscape Analysis using Hyperbolic Geometry.

This package implements analysis of CRISPR-Cas9 guide RNA (gRNA) off-target
effects using p-adic distances and hyperbolic embeddings.

Modules:
    types: Data structures (OffTargetSite, GuideSafetyProfile, etc.)
    padic_distance: p-Adic sequence distance computation
    embedder: Hyperbolic sequence embedding
    predictor: Off-target activity prediction
    analyzer: Complete analysis pipeline
    optimizer: Guide RNA design optimization

Example:
    >>> from src.analysis.crispr import CRISPROfftargetAnalyzer
    >>> analyzer = CRISPROfftargetAnalyzer()
    >>> result = analyzer.analyze_offtarget(guide, offtarget_seq)
"""

# Types and constants
# High-level analysis
from .analyzer import CRISPROfftargetAnalyzer

# Neural network modules
from .embedder import (
    HyperbolicOfftargetEmbedder,
    encode_sequence,
    sequence_to_onehot,
)
from .optimizer import GuideDesignOptimizer

# Core components
from .padic_distance import POSITION_WEIGHTS, PAdicSequenceDistance
from .predictor import OfftargetActivityPredictor
from .types import (
    IDX_TO_NUCLEOTIDE,
    NUCLEOTIDE_TO_IDX,
    PAM_SEQUENCES,
    GuideSafetyProfile,
    MismatchType,
    OffTargetSite,
)

__all__ = [
    # Types
    "MismatchType",
    "OffTargetSite",
    "GuideSafetyProfile",
    # Constants
    "NUCLEOTIDE_TO_IDX",
    "IDX_TO_NUCLEOTIDE",
    "PAM_SEQUENCES",
    "POSITION_WEIGHTS",
    # Utilities
    "encode_sequence",
    "sequence_to_onehot",
    # Classes
    "PAdicSequenceDistance",
    "HyperbolicOfftargetEmbedder",
    "OfftargetActivityPredictor",
    "CRISPROfftargetAnalyzer",
    "GuideDesignOptimizer",
]
