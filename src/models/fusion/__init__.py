# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Cross-modal fusion module.

Provides mechanisms for combining representations from multiple modalities:
- Sequence embeddings (from VAE, PLM)
- Structure embeddings (from SE(3) encoder)
- Property embeddings

Key components:
- CrossModalFusion: Attention-based fusion
- GatedFusion: Gated combination
- MultimodalEncoder: End-to-end multimodal encoding
"""

from src.models.fusion.cross_modal import (
    ConcatFusion,
    CrossModalAttention,
    CrossModalFusion,
    FusionConfig,
    GatedFusion,
)
from src.models.fusion.multimodal import (
    MultimodalConfig,
    MultimodalEncoder,
)

__all__ = [
    "CrossModalFusion",
    "CrossModalAttention",
    "GatedFusion",
    "ConcatFusion",
    "FusionConfig",
    "MultimodalEncoder",
    "MultimodalConfig",
]
