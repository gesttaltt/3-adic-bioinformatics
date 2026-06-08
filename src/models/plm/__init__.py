# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Protein Language Model integration module.

This module provides integration with pretrained protein language models
(ESM-2, ProtTrans) for enhanced sequence representations.

Key components:
- PLMEncoder: Base class for PLM integration
- ESM2Encoder: ESM-2 specific implementation
- HyperbolicPLMEncoder: PLM with hyperbolic projection
"""

from src.models.plm.base import PLMEncoderBase
from src.models.plm.esm_encoder import (
    ESM2Config,
    ESM2Encoder,
)
from src.models.plm.hyperbolic_plm import HyperbolicPLMEncoder

__all__ = [
    "PLMEncoderBase",
    "ESM2Encoder",
    "ESM2Config",
    "HyperbolicPLMEncoder",
]
