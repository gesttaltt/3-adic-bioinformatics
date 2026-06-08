# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Contrastive learning module for self-supervised pre-training.

Provides BYOL and related methods for learning sequence representations
without labeled data.

Key components:
- BYOL: Bootstrap Your Own Latent (no negative samples)
- SimCLR: Simple Contrastive Learning (with negatives)
- SequenceAugmentations: Augmentation strategies for sequences
"""

from src.models.contrastive.augmentations import (
    CropAugmentation,
    MaskingAugmentation,
    MutationAugmentation,
    SequenceAugmentations,
)
from src.models.contrastive.byol import (
    BYOL,
    BYOLConfig,
    MomentumEncoder,
)
from src.models.contrastive.simclr import (
    NTXentLoss,
    SimCLR,
)

__all__ = [
    # BYOL
    "BYOL",
    "BYOLConfig",
    "MomentumEncoder",
    # SimCLR
    "SimCLR",
    "NTXentLoss",
    # Augmentations
    "SequenceAugmentations",
    "MutationAugmentation",
    "MaskingAugmentation",
    "CropAugmentation",
]
