# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Ternary VAE v5.11 - Unified Hyperbolic Geometry with Frozen Coverage.

V5.11 ARCHITECTURE
==================

Design Philosophy:
    - Freeze what works (100% coverage from v5.5)
    - Train only what's needed (geometric projection)
    - Full gradient flow (differentiable controller)

Key Components:
    1. FROZEN v5.5 Encoder: 100% coverage preserved, no gradients
    2. Trainable HyperbolicProjection: Learns radial hierarchy
    3. DifferentiableController: Full gradient flow (no .item() calls)
    4. Unified PAdicGeodesicLoss: Hierarchy + correlation via geometry

Key insight: v5.5 achieved 100% coverage but inverted radial hierarchy.
V5.11 freezes the coverage and learns only the geometric projection.

DETAILED ARCHITECTURE
=====================

    Input: x (batch, 9) - Ternary operation {-1, 0, 1}
           19,683 total operations (3^9 space)

    ┌──────────────────────────────────────────────────────────────┐
    │            FROZEN ENCODERS (from v5.5 checkpoint)            │
    │  FrozenEncoder_A (exploration) → mu_A, logvar_A (16D)       │
    │  FrozenEncoder_B (refinement)  → mu_B, logvar_B (16D)       │
    │  NO GRADIENTS - Preserves 100% reconstruction coverage       │
    └──────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌──────────────────────────────────────────────────────────────┐
    │         TRAINABLE HYPERBOLIC PROJECTION                      │
    │  z_euclidean (16D) → MLP(64) → exp_map → z_poincare         │
    │  • Max radius: 0.95 (boundary constraint)                    │
    │  • Curvature: 1.0 (optionally learnable)                     │
    │  TRAINABLE - Learns Euclidean → Poincare mapping            │
    └──────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌──────────────────────────────────────────────────────────────┐
    │         TRAINABLE DIFFERENTIABLE CONTROLLER                  │
    │  z_hyp, model_state → Control signals (rho, lambda)         │
    │  TRAINABLE - Learns adaptive loss weighting                  │
    └──────────────────────────────────────────────────────────────┘
                                │
                                ▼
    ┌──────────────────────────────────────────────────────────────┐
    │                    FROZEN DECODER                            │
    │  z_A (16D) → Linear(16→64→27) → logits (batch, 9, 3)        │
    │  NO GRADIENTS - Used for verification/reconstruction only    │
    └──────────────────────────────────────────────────────────────┘

Output Dict Keys:
    • logits: (batch, 9, 3) reconstruction logits
    • mu_A, logvar_A, mu_B, logvar_B: encoder outputs
    • z_A, z_B: sampled latent codes
    • z_hyp_A, z_hyp_B: hyperbolic projections
    • curvature: current curvature value
    • controller_output: control signals

Target Metrics:
    • Coverage: 100% (maintained by frozen encoder)
    • Radial Hierarchy: r < -0.70 (Spearman correlation)
    • Q (Structure): > 1.5 (learned capacity)

Single responsibility: V5.11 model architecture.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from src.core import TERNARY
from src.geometry.poincare import poincare_distance
from src.utils.checkpoint import load_checkpoint_compat

from .differentiable_controller import DifferentiableController
from .frozen_components import FrozenDecoder, FrozenEncoder
from .hyperbolic_projection import DualHyperbolicProjection, HyperbolicProjection
from .improved_components import ImprovedDecoder, ImprovedEncoder

logger = logging.getLogger(__name__)


class TernaryVAEV5_11(nn.Module):
    """Ternary VAE v5.11 with frozen coverage and trainable hyperbolic projection.

    This model:
    1. Uses frozen v5.5 encoder for 100% coverage (no training)
    2. Projects Euclidean latents to Poincaré ball (trainable)
    3. Uses differentiable controller for loss weighting (trainable)
    4. Trains only for hyperbolic structure (geodesic loss)

    Attributes:
        latent_dim: Latent space dimension
        hidden_dim: Hidden dimension for projection networks
        max_radius: Maximum radius in Poincaré ball
        curvature: Hyperbolic curvature parameter
        use_controller: Whether to use differentiable controller
        use_dual_projection: Whether to use separate A/B projections
    """

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        max_radius: float = 0.95,
        curvature: float = 1.0,
        use_controller: bool = True,
        use_dual_projection: bool = False,
        n_projection_layers: int = 1,
        projection_dropout: float = 0.0,
        learnable_curvature: bool = False,
        encoder_type: str = "frozen",
        decoder_type: str = "frozen",
        encoder_dropout: float = 0.1,
        decoder_dropout: float = 0.1,
        logvar_min: float = -10.0,
        logvar_max: float = 2.0,
        **kwargs,
    ):
        """Initialize TernaryVAEV5_11.

        Args:
            latent_dim: Latent space dimension (must match v5.5)
            hidden_dim: Hidden dimension for projection networks
            max_radius: Maximum radius in Poincaré ball
            curvature: Hyperbolic curvature parameter
            use_controller: Whether to use differentiable controller
            use_dual_projection: Whether to use separate A/B projections
            n_projection_layers: Number of hidden layers in projection
            projection_dropout: Dropout rate for projection networks
            learnable_curvature: If True, curvature becomes learnable
            encoder_type: "frozen" or "improved" (V5.12.4+)
            decoder_type: "frozen" or "improved" (V5.12.4+)
            encoder_dropout: Dropout for improved encoder (default 0.1)
            decoder_dropout: Dropout for improved decoder (default 0.1)
            logvar_min: Min logvar clamp for improved encoder (default -10)
            logvar_max: Max logvar clamp for improved encoder (default 2)

            # Injected components (optional)
            encoder_A: Injected encoder A (optional)
            encoder_B: Injected encoder B (optional)
            decoder_A: Injected decoder A (optional)
            projection: Injected projection module (optional)
            controller: Injected controller module (optional)
            **kwargs: Additional arguments
        """
        super().__init__()

        # Store kwargs for potential future use
        self.kwargs = kwargs

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_radius = max_radius
        self.curvature = curvature
        self.use_controller = use_controller
        self.use_dual_projection = use_dual_projection
        self.n_projection_layers = n_projection_layers
        self.projection_dropout = projection_dropout
        self.learnable_curvature = learnable_curvature
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max

        # Encoders (frozen or improved based on encoder_type)
        # Injection allows mocks for testing
        self.encoder_A = kwargs.pop("encoder_A", None)
        self.encoder_B = kwargs.pop("encoder_B", None)
        if self.encoder_A is None:
            if encoder_type == "improved":
                self.encoder_A = ImprovedEncoder(
                    latent_dim=latent_dim,
                    dropout=encoder_dropout,
                    logvar_min=logvar_min,
                    logvar_max=logvar_max,
                )
            else:
                self.encoder_A = FrozenEncoder(latent_dim=latent_dim)
        if self.encoder_B is None:
            if encoder_type == "improved":
                self.encoder_B = ImprovedEncoder(
                    latent_dim=latent_dim,
                    dropout=encoder_dropout,
                    logvar_min=logvar_min,
                    logvar_max=logvar_max,
                )
            else:
                self.encoder_B = FrozenEncoder(latent_dim=latent_dim)

        # Decoder (frozen or improved based on decoder_type)
        self.decoder_A = kwargs.pop("decoder_A", None)
        if self.decoder_A is None:
            if decoder_type == "improved":
                self.decoder_A = ImprovedDecoder(
                    latent_dim=latent_dim,
                    dropout=decoder_dropout,
                )
            else:
                self.decoder_A = FrozenDecoder(latent_dim=latent_dim)

        # Trainable hyperbolic projection
        self.projection = kwargs.pop("projection", None)
        if self.projection is None:
            if use_dual_projection:
                self.projection = DualHyperbolicProjection(
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    max_radius=max_radius,
                    curvature=curvature,
                    n_layers=n_projection_layers,
                    dropout=projection_dropout,
                    learnable_curvature=learnable_curvature,
                )
            else:
                self.projection = HyperbolicProjection(
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    max_radius=max_radius,
                    curvature=curvature,
                    n_layers=n_projection_layers,
                    dropout=projection_dropout,
                    learnable_curvature=learnable_curvature,
                )

        # Trainable controller
        self.controller = kwargs.pop("controller", None)
        if self.controller is None:
            if use_controller:
                self.controller = DifferentiableController(input_dim=8, hidden_dim=32)
            else:
                self.controller = None

        # Default control values (used when controller is disabled)
        self.default_control = {
            "rho": 0.0,  # No cross-injection for frozen model
            "weight_geodesic": 1.0,
            "weight_radial": 0.5,
            "beta_A": 1.0,
            "beta_B": 1.0,
            "tau": 0.5,
        }

    def load_v5_5_checkpoint(self, checkpoint_path: Path, device: str = "cpu"):
        """Load frozen components from v5.5 checkpoint.

        Supports both FrozenEncoder/Decoder and ImprovedEncoder/Decoder.
        ImprovedEncoder/Decoder use load_v5_5_weights() which handles
        the key mapping for LayerNorm insertion.

        Args:
            checkpoint_path: Path to v5.5 checkpoint file
            device: Device to load to
        """
        checkpoint = load_checkpoint_compat(checkpoint_path, map_location=device)
        model_state = checkpoint["model"]

        # Load encoder_A (improved or frozen)
        if hasattr(self.encoder_A, "load_v5_5_weights"):
            # ImprovedEncoder: use special weight loading
            self.encoder_A.load_v5_5_weights(checkpoint_path, "encoder_A", device)
            logger.info("  encoder_A: Loaded v5.5 weights into ImprovedEncoder")
        else:
            # FrozenEncoder: direct state_dict load
            enc_A_state = {k.replace("encoder_A.", ""): v for k, v in model_state.items() if k.startswith("encoder_A.")}
            self.encoder_A.load_state_dict(enc_A_state)

        # Load encoder_B (improved or frozen)
        if hasattr(self.encoder_B, "load_v5_5_weights"):
            self.encoder_B.load_v5_5_weights(checkpoint_path, "encoder_B", device)
            logger.info("  encoder_B: Loaded v5.5 weights into ImprovedEncoder")
        else:
            enc_B_state = {k.replace("encoder_B.", ""): v for k, v in model_state.items() if k.startswith("encoder_B.")}
            self.encoder_B.load_state_dict(enc_B_state)

        # Load decoder_A (improved or frozen)
        if hasattr(self.decoder_A, "load_v5_5_weights"):
            self.decoder_A.load_v5_5_weights(checkpoint_path, "decoder_A", device)
            logger.info("  decoder_A: Loaded v5.5 weights into ImprovedDecoder")
        else:
            dec_A_state = {k.replace("decoder_A.", ""): v for k, v in model_state.items() if k.startswith("decoder_A.")}
            self.decoder_A.load_state_dict(dec_A_state)

        # Move to device
        self.to(device)

        # Ensure frozen components stay frozen (unless using improved which may be trainable)
        if self.encoder_type == "frozen":
            for param in self.encoder_A.parameters():
                param.requires_grad = False
            for param in self.encoder_B.parameters():
                param.requires_grad = False
        if self.decoder_type == "frozen":
            for param in self.decoder_A.parameters():
                param.requires_grad = False

        logger.info(f"Loaded v5.5 checkpoint from {checkpoint_path}")
        logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"  Encoder type: {self.encoder_type}")
        logger.info(f"  Decoder type: {self.decoder_type}")

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick.

        Args:
            mu: Mean tensor
            logvar: Log variance tensor

        Returns:
            Sampled latent tensor
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, compute_control: bool = True) -> dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            x: Input ternary operations (batch, 9)
            compute_control: Whether to compute controller outputs

        Returns:
            Dict with all outputs
        """
        # Frozen encoding (no gradients)
        with torch.no_grad():
            mu_A, logvar_A = self.encoder_A(x)
            mu_B, logvar_B = self.encoder_B(x)
            z_A_euc = self.reparameterize(mu_A, logvar_A)
            z_B_euc = self.reparameterize(mu_B, logvar_B)

        # Trainable projection to Poincaré ball
        if self.use_dual_projection:
            z_A_hyp, z_B_hyp = self.projection(z_A_euc, z_B_euc)
        else:
            z_A_hyp = self.projection(z_A_euc)
            z_B_hyp = self.projection(z_B_euc)

        # Compute control signals (if enabled)
        if compute_control and self.controller is not None:
            # V5.12.2: Use hyperbolic distance for consistent geometry
            curvature = self.projection.get_curvature() if hasattr(self.projection, "get_curvature") else 1.0
            origin = torch.zeros_like(z_A_hyp)
            radius_A = poincare_distance(z_A_hyp, origin, c=curvature).mean()
            radius_B = poincare_distance(z_B_hyp, origin, c=curvature).mean()

            # Use mean embeddings for other stats
            kl_A = -0.5 * (1 + logvar_A - mu_A.pow(2) - logvar_A.exp()).sum(dim=-1).mean()
            kl_B = -0.5 * (1 + logvar_B - mu_B.pow(2) - logvar_B.exp()).sum(dim=-1).mean()

            # Placeholder for loss-based stats
            geo_loss_placeholder = torch.tensor(0.0, device=x.device)
            rad_loss_placeholder = torch.tensor(0.0, device=x.device)

            # Calculate actual hierarchy values (not placeholders)
            # Convert ternary operations to indices for valuation calculation
            batch_indices = TERNARY.from_ternary(x)  # (batch,) indices
            batch_valuations = TERNARY.valuation(batch_indices).cpu().numpy()  # (batch,) valuations

            # Get radii for hierarchy calculation
            radii_A_batch = poincare_distance(z_A_hyp, origin, c=curvature).detach().cpu().numpy()
            radii_B_batch = poincare_distance(z_B_hyp, origin, c=curvature).detach().cpu().numpy()

            # Calculate hierarchy as Spearman correlation for current batch
            if len(batch_valuations) > 1 and len(np.unique(batch_valuations)) > 1:
                hierarchy_A = spearmanr(batch_valuations, radii_A_batch)[0]
                hierarchy_B = spearmanr(batch_valuations, radii_B_batch)[0]

                # Handle NaN values from spearmanr
                if np.isnan(hierarchy_A):
                    hierarchy_A = 0.0
                if np.isnan(hierarchy_B):
                    hierarchy_B = 0.0
            else:
                # Fallback for small batches or uniform valuations
                hierarchy_A = 0.0
                hierarchy_B = 0.0

            batch_stats = torch.stack(
                [
                    radius_A,
                    radius_B,
                    torch.tensor(hierarchy_A, device=x.device, dtype=torch.float32),  # Real H_A
                    torch.tensor(hierarchy_B, device=x.device, dtype=torch.float32),  # Real H_B
                    kl_A,
                    kl_B,
                    geo_loss_placeholder,
                    rad_loss_placeholder,
                ]
            )

            control = self.controller(batch_stats)
            control = {k: v.squeeze(0) for k, v in control.items()}
        else:
            control = {k: torch.tensor(v, device=x.device) for k, v in self.default_control.items()}

        # Verification reconstruction (no gradients, for monitoring only)
        with torch.no_grad():
            logits_A = self.decoder_A(z_A_euc)

        return {
            # Euclidean latents (from frozen encoder)
            "z_A_euc": z_A_euc,
            "z_B_euc": z_B_euc,
            "mu_A": mu_A,
            "mu_B": mu_B,
            "logvar_A": logvar_A,
            "logvar_B": logvar_B,
            # Hyperbolic latents (from trainable projection)
            "z_A_hyp": z_A_hyp,
            "z_B_hyp": z_B_hyp,
            # Control signals (from trainable controller)
            "control": control,
            # Reconstruction logits (for verification)
            "logits_A": logits_A,
        }

    def get_trainable_parameters(self):
        """Get only trainable parameters (projection + controller).

        Returns:
            List of trainable parameters
        """
        params = list(self.projection.parameters())
        if self.controller is not None:
            params.extend(self.controller.parameters())
        return params

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component.

        Returns:
            Dict with parameter counts by component
        """
        frozen_params = (
            sum(p.numel() for p in self.encoder_A.parameters())
            + sum(p.numel() for p in self.encoder_B.parameters())
            + sum(p.numel() for p in self.decoder_A.parameters())
        )

        projection_params = sum(p.numel() for p in self.projection.parameters())

        controller_params = 0
        if self.controller is not None:
            controller_params = sum(p.numel() for p in self.controller.parameters())

        return {
            "frozen": frozen_params,
            "projection": projection_params,
            "controller": controller_params,
            "trainable": projection_params + controller_params,
            "total": frozen_params + projection_params + controller_params,
        }


# Re-export PartialFreeze variant (and deprecated OptionC alias)
from .ternary_vae_optionc import TernaryVAEV5_11_OptionC, TernaryVAEV5_11_PartialFreeze

__all__ = [
    "FrozenEncoder",
    "FrozenDecoder",
    "TernaryVAEV5_11",
    "TernaryVAEV5_11_PartialFreeze",
    "TernaryVAEV5_11_OptionC",  # Deprecated alias
]
