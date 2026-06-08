# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""SE(3)-Equivariant neural network modules.

This module provides SE(3)-equivariant encoders for protein 3D structure
that preserve rotational and translational symmetry.

Key components:
- SE3EquivariantEncoder: Main structure encoder
- EquivariantBlock: Equivariant message passing layer
- InvariantReadout: Convert to invariant features
"""

from src.models.equivariant.layers import (
    EquivariantBlock,
    InvariantReadout,
    RadialBasisFunctions,
)
from src.models.equivariant.se3_encoder import (
    SE3Config,
    SE3EquivariantEncoder,
)

__all__ = [
    "SE3EquivariantEncoder",
    "SE3Config",
    "EquivariantBlock",
    "InvariantReadout",
    "RadialBasisFunctions",
]
