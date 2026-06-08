# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Holographic VAE Components.

Implements AdS/CFT-inspired bulk-to-boundary decoding where:
- Bulk (latent space) represents ancestral/hidden states
- Boundary (sequence space) represents observable sequences
- Propagation follows geodesics with power-law decay

Key insight: Standard MLP decoders ignore the geometric structure of
hyperbolic space. Holographic decoders propagate signals along geodesics,
making them more parameter-efficient and interpretable.
"""

from src.models.holographic.bulk_boundary import (
    BulkBoundaryPropagator,
    GeodesicPropagator,
    RadialDecayFunction,
)
from src.models.holographic.decoder import (
    HolographicDecoder,
    HolographicDecoderConfig,
)

__all__ = [
    "HolographicDecoder",
    "HolographicDecoderConfig",
    "BulkBoundaryPropagator",
    "GeodesicPropagator",
    "RadialDecayFunction",
]
