# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Model definitions for Ternary VAE (canonical V5.11 architecture).

V6.0 additions:
- PLM encoders (ESM-2 integration)
- SE(3)-equivariant structure encoders
- Uncertainty quantification (Bayesian, Evidential, Ensemble)
- Multi-task learning for drug resistance
- Discrete diffusion for sequence generation
- Contrastive learning (BYOL, SimCLR)
- Cross-modal fusion layers
"""

from .curriculum import ContinuousCurriculumModule, CurriculumScheduler
from .differentiable_controller import DifferentiableController
from .homeostasis import HomeostasisController, compute_Q
from .hyperbolic_projection import DualHyperbolicProjection, HyperbolicProjection

# NOTE: SwarmVAE archived to src/ARCHIVE/v5_10_legacy/ - experimental, not in production
# Set to None for backward compatibility with code referencing these
SwarmVAE = None
SwarmAgent = None
AgentConfig = None
AgentRole = None
PheromoneField = None

from .base_vae import BaseVAE, ConditionalBaseVAE, HyperbolicBaseVAE, VAEConfig, VAEOutput
from .epistasis_module import (
    EpistasisModule,
    EpistasisPredictor,
    EpistasisResult,
    HigherOrderInteractionModule,
    PairwiseInteractionModule,
)
from .hierarchical_vae import HierarchicalVAE, HierarchicalVAEConfig
from .improved_components import (
    ImprovedDecoder,
    ImprovedEncoder,
    create_decoder,
    create_encoder,
)
from .structure_aware_vae import (
    InvariantPointAttention,
    SE3Encoder,
    StructureAwareVAE,
    StructureConfig,
    StructureSequenceFusion,
)
from .ternary_vae import (
    FrozenDecoder,
    FrozenEncoder,
    TernaryVAEV5_11,
    TernaryVAEV5_11_OptionC,
    TernaryVAEV5_11_PartialFreeze,
)

# Canonical exports (V5.11 architecture)
TernaryVAE = TernaryVAEV5_11
TernaryVAE_PartialFreeze = TernaryVAEV5_11_PartialFreeze
TernaryVAE_OptionC = TernaryVAEV5_11_OptionC  # Deprecated alias


# V6.0 module imports (lazy loading for optional dependencies)
def _import_plm():
    from .plm import ESM2Config, ESM2Encoder, HyperbolicPLMEncoder, PLMEncoderBase

    return ESM2Encoder, ESM2Config, HyperbolicPLMEncoder, PLMEncoderBase


def _import_equivariant():
    from .equivariant import SE3Config, SE3EquivariantEncoder, SE3WithHyperbolic

    return SE3EquivariantEncoder, SE3Config, SE3WithHyperbolic


def _import_uncertainty():
    from .uncertainty import (
        BayesianPredictor,
        DeepEnsemble,
        EnsemblePredictor,
        EvidentialLoss,
        EvidentialPredictor,
        MCDropoutWrapper,
        UncertaintyWrapper,
    )

    return (
        UncertaintyWrapper,
        BayesianPredictor,
        MCDropoutWrapper,
        EvidentialPredictor,
        EvidentialLoss,
        EnsemblePredictor,
        DeepEnsemble,
    )


def _import_mtl():
    from .mtl import GradNormOptimizer, MTLConfig, MultiTaskResistancePredictor

    return MultiTaskResistancePredictor, MTLConfig, GradNormOptimizer


def _import_diffusion():
    from .diffusion import D3PM, ConditionalGenerator, D3PMConfig, SequenceGenerator

    return D3PM, D3PMConfig, SequenceGenerator, ConditionalGenerator


def _import_contrastive():
    from .contrastive import BYOL, BYOLConfig, SequenceAugmentations, SimCLR

    return BYOL, BYOLConfig, SimCLR, SequenceAugmentations


def _import_fusion():
    from .fusion import CrossModalFusion, FusionConfig, MultimodalConfig, MultimodalEncoder

    return CrossModalFusion, FusionConfig, MultimodalEncoder, MultimodalConfig


__all__ = [
    # Canonical (V5.11)
    "TernaryVAE",
    "TernaryVAE_PartialFreeze",
    "TernaryVAE_OptionC",  # Deprecated alias
    "TernaryVAEV5_11",
    "TernaryVAEV5_11_PartialFreeze",
    "TernaryVAEV5_11_OptionC",  # Deprecated alias
    "FrozenEncoder",
    "FrozenDecoder",
    # Improved components (V5.12.3+)
    "ImprovedEncoder",
    "ImprovedDecoder",
    "create_encoder",
    "create_decoder",
    # Base VAE abstraction
    "BaseVAE",
    "HyperbolicBaseVAE",
    "ConditionalBaseVAE",
    "VAEConfig",
    "VAEOutput",
    # Hierarchical VAE
    "HierarchicalVAE",
    "HierarchicalVAEConfig",
    # Projections
    "HyperbolicProjection",
    "DualHyperbolicProjection",
    # Controllers
    "DifferentiableController",
    # Curriculum
    "ContinuousCurriculumModule",
    "CurriculumScheduler",
    # Homeostasis (V5.11.7 + V5.11.8)
    "HomeostasisController",
    "compute_Q",
    # Swarm VAE (multi-agent architecture)
    "SwarmVAE",
    "SwarmAgent",
    "AgentConfig",
    "AgentRole",
    "PheromoneField",
    # Epistasis module
    "EpistasisModule",
    "EpistasisResult",
    "EpistasisPredictor",
    "PairwiseInteractionModule",
    "HigherOrderInteractionModule",
    # Structure-aware VAE
    "StructureAwareVAE",
    "StructureConfig",
    "SE3Encoder",
    "StructureSequenceFusion",
    "InvariantPointAttention",
    # V6.0 modules (accessible via submodule imports)
    # from src.models.plm import ESM2Encoder
    # from src.models.uncertainty import BayesianPredictor
    # from src.models.mtl import MultiTaskResistancePredictor
    # from src.models.diffusion import D3PM
    # from src.models.contrastive import BYOL
    # from src.models.fusion import CrossModalFusion
]
