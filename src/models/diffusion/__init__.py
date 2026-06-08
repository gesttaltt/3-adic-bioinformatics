# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Discrete diffusion models for sequence generation.

Implements D3PM (Discrete Denoising Diffusion Probabilistic Models)
with extensions for p-adic aware noise schedules.

Key components:
- DiscreteDiffusion: Core D3PM implementation
- PAdicNoiseSchedule: P-adic aware noise schedule
- SequenceGenerator: High-level interface for generation
"""

from src.models.diffusion.d3pm import (
    D3PM,
    D3PMConfig,
)
from src.models.diffusion.noise_schedule import (
    CosineSchedule,
    LinearSchedule,
    NoiseSchedule,
    PAdicNoiseSchedule,
)
from src.models.diffusion.sequence_generator import (
    ConditionalGenerator,
    SequenceGenerator,
)

__all__ = [
    "D3PM",
    "D3PMConfig",
    "NoiseSchedule",
    "LinearSchedule",
    "CosineSchedule",
    "PAdicNoiseSchedule",
    "SequenceGenerator",
    "ConditionalGenerator",
]
