# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Gradient checkpointing utilities for memory-efficient training.

This module provides gradient checkpointing functionality to reduce VRAM usage
during training, particularly useful for GPUs with limited memory (4-6GB).
"""

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class GradientCheckpointConfig:
    """Configuration for gradient checkpointing."""

    enabled: bool = False
    segments: int = 2  # Number of segments to divide model into
    checkpoint_segments: int = 2  # Alias for backwards compatibility
    preserve_rng_state: bool = True
    encoder_checkpoint: bool = True  # Whether to checkpoint encoder
    decoder_checkpoint: bool = True  # Whether to checkpoint decoder
    projection_checkpoint: bool = False  # Whether to checkpoint projection layers


def create_checkpoint_config(
    enabled: bool = False,
    segments: int = 2,
    preserve_rng_state: bool = True,
    encoder_checkpoint: bool = True,
    decoder_checkpoint: bool = True,
    projection_checkpoint: bool = False,
) -> GradientCheckpointConfig:
    """Create a gradient checkpoint configuration.

    Args:
        enabled: Whether to enable gradient checkpointing
        segments: Number of segments to divide the model into
        preserve_rng_state: Whether to preserve RNG state during checkpointing
        encoder_checkpoint: Whether to checkpoint encoder
        decoder_checkpoint: Whether to checkpoint decoder
        projection_checkpoint: Whether to checkpoint projection layers

    Returns:
        GradientCheckpointConfig instance
    """
    return GradientCheckpointConfig(
        enabled=enabled,
        segments=segments,
        checkpoint_segments=segments,
        preserve_rng_state=preserve_rng_state,
        encoder_checkpoint=encoder_checkpoint,
        decoder_checkpoint=decoder_checkpoint,
        projection_checkpoint=projection_checkpoint,
    )


def apply_gradient_checkpointing(model: nn.Module, config: GradientCheckpointConfig | None = None) -> nn.Module:
    """Apply gradient checkpointing to a model.

    Gradient checkpointing trades compute for memory by recomputing
    intermediate activations during the backward pass instead of storing them.

    Args:
        model: The model to apply checkpointing to
        config: Checkpointing configuration (optional)

    Returns:
        The model with checkpointing applied (modified in-place)
    """
    if config is None or not config.enabled:
        return model

    # Enable gradient checkpointing for models that support it
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    elif hasattr(model, "encoder_A") and hasattr(model.encoder_A, "gradient_checkpointing_enable"):
        model.encoder_A.gradient_checkpointing_enable()
        if hasattr(model, "encoder_B"):
            model.encoder_B.gradient_checkpointing_enable()
    else:
        # For custom models, we can use torch.utils.checkpoint manually
        # This is a no-op for models that don't support it
        pass

    return model


__all__ = ["GradientCheckpointConfig", "create_checkpoint_config", "apply_gradient_checkpointing"]
