# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class EncoderProtocol(Protocol):
    """Protocol for VAE Encoders."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns: (mu, logvar)"""
        ...


@runtime_checkable
class DecoderProtocol(Protocol):
    """Protocol for VAE Decoders."""

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns: logits"""
        ...


@runtime_checkable
class ProjectionProtocol(Protocol):
    """Protocol for Hyperbolic Projections."""

    def forward(
        self, z_A: torch.Tensor, z_B: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Projects Euclidean vectors to Hyperbolic space."""
        ...


@runtime_checkable
class ControllerProtocol(Protocol):
    """Protocol for Differentiable Controllers."""

    def forward(self, batch_stats: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns control signals."""
        ...
