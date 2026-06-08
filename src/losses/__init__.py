# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Loss function components.

This module contains loss computation separated from model architecture.

Architecture:
    - base.py: LossComponent protocol and LossResult dataclass
    - registry.py: LossRegistry for dynamic composition
    - components.py: LossComponent wrappers for all loss types

Legacy classes (for backwards compatibility):
    - DualVAELoss: Aggregated loss (deprecated, use LossRegistry)
    - ReconstructionLoss, KLDivergenceLoss, etc.

New pattern (recommended):
    from src.losses import LossRegistry, ReconstructionLossComponent

    registry = LossRegistry()
    registry.register('recon', ReconstructionLossComponent())
    result = registry.compose(outputs, targets)

Loss Types:
    - Reconstruction: Cross-entropy for ternary operations
    - KL Divergence: With free bits support
    - Entropy: Regularization for diversity
    - Repulsion: Latent space diversity
    - p-Adic: Metric alignment, ranking, hierarchical norms
    - Hyperbolic: Poincare ball geometry (v5.10)
    - Radial: Stratification for 3-adic hierarchy
"""

# New structural components (LossRegistry pattern)
# Autoimmune-aware regularization
from .autoimmunity import HUMAN_CODON_RSCU, AutoimmuneCodonRegularizer, CD4CD8AwareRegularizer
from .base import DualVAELossComponent, LossComponent, LossResult

# Codon usage constraints
from .codon_usage import (
    CodonOptimalityScore,
    CodonUsageConfig,
    CodonUsageLoss,
    Organism,
)

# Co-evolution losses
from .coevolution_loss import (
    BiosyntheticCoherenceLoss,
    CoEvolutionLoss,
    CoEvolutionMetrics,
    ErrorMinimizationLoss,
    PAdicStructureLoss,
    ResourceConservationLoss,
)
from .components import (
    EntropyAlignmentComponent,
    EntropyLossComponent,
    KLDivergenceLossComponent,
    PAdicHyperbolicLossComponent,
    PAdicRankingLossComponent,
    RadialStratificationLossComponent,
    ReconstructionLossComponent,
    RepulsionLossComponent,
)
from .consequence_predictor import ConsequencePredictor, PurposefulRankingLoss, evaluate_addition_accuracy

# NOTE: DualVAELoss is deprecated - use LossRegistry pattern for new code
# Kept for backward compatibility with existing trainers
from .dual_vae_loss import DualVAELoss, EntropyRegularization, KLDivergenceLoss, ReconstructionLoss, RepulsionLoss

# Epistasis losses
from .epistasis_loss import (
    DrugInteractionLoss,
    EpistasisLoss,
    EpistasisLossResult,
    LearnedEpistasisLoss,
    MarginRankingLoss,
)

# Fisher-Rao information geometry
from .fisher_rao import (
    FisherRaoConfig,
    FisherRaoDistance,
    FisherRaoKL,
    FisherRaoLoss,
    NaturalGradientRegularizer,
)

# Glycan shield analysis
from .glycan_loss import (
    GlycanRemovalSimulator,
    GlycanSequonDetector,
    GlycanShieldAnalyzer,
    GlycanShieldMetrics,
    GlycanSite,
    SentinelGlycanLoss,
)
from .hyperbolic_prior import HomeostaticHyperbolicPrior, HyperbolicPrior
from .hyperbolic_recon import HomeostaticReconLoss, HyperbolicCentroidLoss, HyperbolicReconLoss

# p-Adic losses - import from new modular subpackage (preferred)
# Also maintains backward compatibility with existing imports
from .padic import (
    # New exports from modular structure
    EuclideanTripletMiner,
    HyperbolicTripletMiner,
    PAdicMetricLoss,
    PAdicNormLoss,
    PAdicRankingLoss,
    PAdicRankingLossHyperbolic,
    PAdicRankingLossV2,
    TripletBatch,
    TripletMiner,
    compute_3adic_valuation_batch,
)
from .padic_geodesic import (
    CombinedGeodesicLoss,
    GlobalRankLoss,
    MonotonicRadialLoss,
    PAdicGeodesicLoss,
    RadialHierarchyLoss,
    poincare_distance,
)
from .radial_stratification import RadialStratificationLoss, compute_single_index_valuation
from .registry import (
    LossComponentRegistry,
    LossGroup,
    LossRegistry,
    create_registry_from_config,
    create_registry_from_training_config,
    create_registry_with_plugins,
)
from .rich_hierarchy import RichHierarchyLoss
from .zero_structure import (
    CombinedZeroStructureLoss,
    ZeroSparsityLoss,
    ZeroValuationLoss,
    compute_operation_zero_count,
    compute_operation_zero_valuation,
)

# Appetitive losses archived (unused in active training)
# See src/losses/archive/appetitive_losses.py for legacy code


__all__ = [
    # Legacy classes (deprecated but kept for backward compatibility)
    # New code should use LossRegistry pattern instead
    "ReconstructionLoss",
    "KLDivergenceLoss",
    "EntropyRegularization",
    "RepulsionLoss",
    "DualVAELoss",
    "PAdicMetricLoss",
    "PAdicRankingLoss",
    "PAdicRankingLossV2",
    "PAdicRankingLossHyperbolic",
    "PAdicNormLoss",
    # Triplet mining utilities (new modular structure)
    "TripletBatch",
    "TripletMiner",
    "EuclideanTripletMiner",
    "HyperbolicTripletMiner",
    "compute_3adic_valuation_batch",
    # Appetitive exports removed (archived)
    "HyperbolicPrior",
    "HomeostaticHyperbolicPrior",
    "HyperbolicReconLoss",
    "HomeostaticReconLoss",
    "HyperbolicCentroidLoss",
    "RadialStratificationLoss",
    "compute_single_index_valuation",
    # V5.11.9 Zero Structure Loss
    "ZeroValuationLoss",
    "ZeroSparsityLoss",
    "CombinedZeroStructureLoss",
    "compute_operation_zero_valuation",
    "compute_operation_zero_count",
    # V5.11 Unified Geodesic Loss
    "poincare_distance",
    "PAdicGeodesicLoss",
    "RadialHierarchyLoss",
    "CombinedGeodesicLoss",
    "GlobalRankLoss",
    "MonotonicRadialLoss",
    # Rich Hierarchy Loss (richness preservation)
    "RichHierarchyLoss",
    "ConsequencePredictor",
    "evaluate_addition_accuracy",
    "PurposefulRankingLoss",
    # New structural components (LossRegistry pattern)
    "LossResult",
    "LossComponent",
    "DualVAELossComponent",
    "LossRegistry",
    "LossGroup",
    "LossComponentRegistry",
    "create_registry_from_config",
    "create_registry_from_training_config",
    "create_registry_with_plugins",
    "ReconstructionLossComponent",
    "KLDivergenceLossComponent",
    "EntropyLossComponent",
    "RepulsionLossComponent",
    "EntropyAlignmentComponent",
    "PAdicRankingLossComponent",
    "PAdicHyperbolicLossComponent",
    "RadialStratificationLossComponent",
    # Autoimmune-aware regularization
    "AutoimmuneCodonRegularizer",
    "CD4CD8AwareRegularizer",
    "HUMAN_CODON_RSCU",
    # Co-evolution losses
    "CoEvolutionLoss",
    "CoEvolutionMetrics",
    "BiosyntheticCoherenceLoss",
    "ErrorMinimizationLoss",
    "ResourceConservationLoss",
    "PAdicStructureLoss",
    # Glycan shield analysis
    "SentinelGlycanLoss",
    "GlycanShieldAnalyzer",
    "GlycanShieldMetrics",
    "GlycanSite",
    "GlycanSequonDetector",
    "GlycanRemovalSimulator",
    # Fisher-Rao information geometry
    "FisherRaoConfig",
    "FisherRaoDistance",
    "FisherRaoLoss",
    "FisherRaoKL",
    "NaturalGradientRegularizer",
    # Codon usage constraints
    "CodonUsageLoss",
    "CodonUsageConfig",
    "CodonOptimalityScore",
    "Organism",
    # Epistasis losses
    "EpistasisLoss",
    "EpistasisLossResult",
    "LearnedEpistasisLoss",
    "DrugInteractionLoss",
    "MarginRankingLoss",
]
