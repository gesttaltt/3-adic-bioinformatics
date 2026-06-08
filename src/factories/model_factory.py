# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

from typing import Any

import torch.nn as nn

from src.models.ternary_vae import (
    DifferentiableController,
    DualHyperbolicProjection,
    FrozenDecoder,
    FrozenEncoder,
    HyperbolicProjection,
    TernaryVAEV5_11,
)


class TernaryModelFactory:
    """Factory for creating Ternary VAE models and components.

    Centralizes complex object construction logic, making it easier to
    inject dependencies or simple configurations for testing.
    """

    @staticmethod
    def create_components(config: dict[str, Any]) -> dict[str, nn.Module]:
        """Create individual components based on config."""
        latent_dim = config.get("latent_dim", 16)
        hidden_dim = config.get("hidden_dim", 64)
        max_radius = config.get("max_radius", 0.95)
        curvature = config.get("curvature", 1.0)

        # Encoders/Decoders (Frozen by default in V5.11)
        encoder_A = FrozenEncoder(latent_dim=latent_dim)
        encoder_B = FrozenEncoder(latent_dim=latent_dim)
        decoder_A = FrozenDecoder(latent_dim=latent_dim)

        # Projection
        if config.get("use_dual_projection", False):
            projection = DualHyperbolicProjection(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                max_radius=max_radius,
                curvature=curvature,
                n_layers=config.get("projection_layers", 1),
                dropout=config.get("projection_dropout", 0.0),
                learnable_curvature=config.get("learnable_curvature", False),
            )
        else:
            projection = HyperbolicProjection(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                max_radius=max_radius,
                curvature=curvature,
                n_layers=config.get("projection_layers", 1),
                dropout=config.get("projection_dropout", 0.0),
                learnable_curvature=config.get("learnable_curvature", False),
            )

        # Controller
        controller = None
        if config.get("use_controller", False):
            controller = DifferentiableController(input_dim=8, hidden_dim=32)

        return {
            "encoder_A": encoder_A,
            "encoder_B": encoder_B,
            "decoder_A": decoder_A,
            "projection": projection,
            "controller": controller,
        }

    @staticmethod
    def create_model(config: dict[str, Any]) -> TernaryVAEV5_11:
        """Create a fully assembled TernaryVAEV5_11 model."""
        # This uses the standard constructor (which we will refactor to accept injections)
        # For now, relying on kwargs pass-through if the model supports it,
        # or just standard init args.

        # Extract explicit args for TernaryVAEV5_11 __init__
        init_args = {
            "latent_dim": config.get("latent_dim", 16),
            "hidden_dim": config.get("hidden_dim", 64),
            "max_radius": config.get("max_radius", 0.95),
            "curvature": config.get("curvature", 1.0),
            "use_controller": config.get("use_controller", False),
            "use_dual_projection": config.get("use_dual_projection", False),
            "n_projection_layers": config.get("projection_layers", 1),
            "projection_dropout": config.get("projection_dropout", 0.0),
            "learnable_curvature": config.get("learnable_curvature", False),
        }

        model = TernaryVAEV5_11(**init_args)
        return model
