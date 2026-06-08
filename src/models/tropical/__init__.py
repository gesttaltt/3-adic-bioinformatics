# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tropical VAE Module.

Implements a VAE where the latent space uses tropical (max-plus) algebra
instead of standard linear algebra. This forces the model to learn
discrete, tree-like features natively.

Key insight: In tropical algebra, operations are:
- Addition: max(a, b)
- Multiplication: a + b

This means:
- Linear combinations become piecewise linear (max of linear functions)
- The latent space naturally encodes tree structures
- Geodesics are piecewise linear paths
"""

from src.models.tropical.tropical_layers import (
    TropicalActivation,
    TropicalConv1d,
    TropicalLayerNorm,
    TropicalLinear,
)
from src.models.tropical.tropical_vae import (
    TropicalDecoder,
    TropicalEncoder,
    TropicalVAE,
    TropicalVAEConfig,
)

__all__ = [
    "TropicalLinear",
    "TropicalConv1d",
    "TropicalLayerNorm",
    "TropicalActivation",
    "TropicalVAE",
    "TropicalVAEConfig",
    "TropicalEncoder",
    "TropicalDecoder",
]
