# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DDG prediction models.

This module provides:
- DDGVAE: Base VAE for DDG prediction
- MultimodalDDGVAE: Multimodal fusion VAE
- DDGMLPRefiner: MLP refinement layers
- DDGTransformer: Transformer heads for precise predictions
- DDGEnsemble: Ensemble combining all models
"""

from src.bioinformatics.models.ddg_ensemble import DDGEnsemble, FuzzyDDGHead
from src.bioinformatics.models.ddg_mlp_refiner import DDGMLPRefiner, RefinerConfig
from src.bioinformatics.models.ddg_transformer import (
    DDGTransformer,
    HierarchicalTransformer,
    TransformerConfig,
)
from src.bioinformatics.models.ddg_vae import DDGVAE, DDGVAEConfig
from src.bioinformatics.models.multimodal_ddg_vae import (
    CrossModalFusion,
    MultimodalConfig,
    MultimodalDDGVAE,
)

__all__ = [
    "DDGVAE",
    "DDGVAEConfig",
    "MultimodalDDGVAE",
    "CrossModalFusion",
    "MultimodalConfig",
    "DDGMLPRefiner",
    "RefinerConfig",
    "DDGTransformer",
    "HierarchicalTransformer",
    "TransformerConfig",
    "DDGEnsemble",
    "FuzzyDDGHead",
]
