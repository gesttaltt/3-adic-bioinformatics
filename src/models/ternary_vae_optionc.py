# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Ternary VAE v5.11 Partial Freeze - Dynamic component freeze with homeostatic control.

This module implements the PartialFreeze variant that allows dynamic freeze/unfreeze
of components based on training metrics (coverage, hierarchy, gradient norms).

V5.11.7: Hierarchical homeostasis where:
- encoder_A: coverage-gated (freeze on drop, unfreeze on recovery)
- encoder_B: hierarchy-gated (freeze when VAE-B hierarchy plateaus)
- controller: gradient-gated (freeze when weights stabilize)
- projections: always trainable (fast adaptation layer)

This implements complementary learning systems theory:
- Slow components (encoders) consolidate when objectives met
- Fast components (projections) continuously adapt

Single responsibility: Partial freeze variant with homeostatic control.

Note: This class was previously named "OptionC". The old names are preserved
as aliases for backward compatibility.
"""

import logging
from pathlib import Path

import torch
import torch.optim as optim

from src.geometry.poincare import log_map_zero, poincare_distance

from .ternary_vae import TernaryVAEV5_11

logger = logging.getLogger(__name__)


class TernaryVAEV5_11_PartialFreeze(TernaryVAEV5_11):
    """V5.11 PartialFreeze: Dynamic component freeze with homeostatic control.

    Inherits from V5.11 but allows dynamic freeze/unfreeze of components
    based on training metrics (coverage, hierarchy, gradient norms).

    Attributes:
        freeze_encoder_b: Whether encoder_B is frozen
        encoder_b_lr_scale: Learning rate multiplier for encoder_B
        encoder_a_lr_scale: Learning rate multiplier for encoder_A
        controller_lr_scale: Learning rate multiplier for controller
        freeze_encoder_a: Current freeze state for encoder_A
        freeze_controller: Current freeze state for controller
    """

    def __init__(
        self,
        freeze_encoder_b: bool = False,
        encoder_b_lr_scale: float = 0.1,
        encoder_a_lr_scale: float = 0.05,
        controller_lr_scale: float = 1.0,
        **kwargs,
    ):
        """Initialize PartialFreeze variant.

        Args:
            freeze_encoder_b: Whether to freeze encoder_B
            encoder_b_lr_scale: Learning rate scale for encoder_B
            encoder_a_lr_scale: Learning rate scale for encoder_A
            controller_lr_scale: Learning rate scale for controller
            **kwargs: Arguments passed to TernaryVAEV5_11
        """
        super().__init__(**kwargs)
        self.freeze_encoder_b = freeze_encoder_b
        self.encoder_b_lr_scale = encoder_b_lr_scale
        self.controller_lr_scale = controller_lr_scale

        # Homeostatic freeze states
        self.freeze_encoder_a = True  # Starts frozen
        self.freeze_controller = False  # Starts trainable
        self.encoder_a_lr_scale = encoder_a_lr_scale

    def load_v5_5_checkpoint(self, checkpoint_path: Path, device: str = "cpu"):
        """Load checkpoint with optional encoder_B unfreezing.

        Args:
            checkpoint_path: Path to v5.5 checkpoint
            device: Device to load to
        """
        super().load_v5_5_checkpoint(checkpoint_path, device)

        if not self.freeze_encoder_b:
            # Unfreeze encoder_B for exploration
            for param in self.encoder_B.parameters():
                param.requires_grad = True
            logger.info("  encoder_B UNFROZEN for Option C training")
            logger.info(f"  encoder_B LR scale: {self.encoder_b_lr_scale}")

    def set_encoder_a_unfreeze(self, lr_scale: float):
        """Progressively unfreeze encoder_A with given learning rate scale.

        V5.11.6: Allows gradual unfreezing of encoder_A during training.

        Args:
            lr_scale: Learning rate multiplier (0.0 = frozen, >0 = trainable)
        """
        self.encoder_a_lr_scale = lr_scale
        self.freeze_encoder_a = lr_scale == 0.0

        if not self.freeze_encoder_a:
            for param in self.encoder_A.parameters():
                param.requires_grad = True
        else:
            for param in self.encoder_A.parameters():
                param.requires_grad = False

    def set_encoder_a_frozen(self, frozen: bool):
        """Set encoder_A freeze state (homeostatic control).

        Args:
            frozen: True to freeze, False to unfreeze
        """
        self.freeze_encoder_a = frozen
        for param in self.encoder_A.parameters():
            param.requires_grad = not frozen

    def set_encoder_b_frozen(self, frozen: bool):
        """Set encoder_B freeze state (homeostatic control).

        Args:
            frozen: True to freeze, False to unfreeze
        """
        self.freeze_encoder_b = frozen
        for param in self.encoder_B.parameters():
            param.requires_grad = not frozen

    def set_controller_frozen(self, frozen: bool):
        """Set controller freeze state (homeostatic control).

        Args:
            frozen: True to freeze, False to unfreeze
        """
        if self.controller is None:
            return
        self.freeze_controller = frozen
        for param in self.controller.parameters():
            param.requires_grad = not frozen

    def set_decoder_frozen(self, frozen: bool):
        """Set decoder_A freeze state.

        V5.12.1: Decoder must be trainable when using log_map_zero(z_hyp) input,
        since it was originally trained on mu (Euclidean) and needs to adapt.

        Args:
            frozen: True to freeze, False to unfreeze
        """
        self.freeze_decoder = frozen
        for param in self.decoder_A.parameters():
            param.requires_grad = not frozen

    def apply_homeostasis_state(self, state: dict):
        """Apply freeze states from HomeostasisController.

        Args:
            state: Dict with encoder_a_frozen, encoder_b_frozen, controller_frozen
        """
        if "encoder_a_frozen" in state:
            self.set_encoder_a_frozen(state["encoder_a_frozen"])
        if "encoder_b_frozen" in state:
            self.set_encoder_b_frozen(state["encoder_b_frozen"])
        if "controller_frozen" in state:
            self.set_controller_frozen(state["controller_frozen"])

    def get_freeze_state_summary(self) -> str:
        """Get human-readable freeze state summary.

        Returns:
            Summary string like "enc_A:F enc_B:T ctrl:T"
        """
        states = []
        states.append(f"enc_A:{'F' if self.freeze_encoder_a else 'T'}")
        states.append(f"enc_B:{'F' if self.freeze_encoder_b else 'T'}")
        ctrl_state = "F" if getattr(self, "freeze_controller", False) else "T"
        states.append(f"ctrl:{ctrl_state}")
        return " ".join(states)

    def forward(self, x: torch.Tensor, compute_control: bool = True) -> dict[str, torch.Tensor]:
        """Forward pass with conditional gradient flow for both encoders.

        Override parent to allow gradients through encoder_A and encoder_B
        based on freeze state. V5.11.6 enables progressive unfreezing.

        Args:
            x: Input tensor (batch, input_dim)
            compute_control: Whether to compute controller outputs

        Returns:
            Dict with all model outputs
        """
        # Encoder A: frozen by default, can be progressively unfrozen
        if self.freeze_encoder_a:
            with torch.no_grad():
                mu_A, logvar_A = self.encoder_A(x)
                z_A_euc = self.reparameterize(mu_A, logvar_A)
        else:
            # Allow gradients through encoder_A for progressive unfreezing
            mu_A, logvar_A = self.encoder_A(x)
            z_A_euc = self.reparameterize(mu_A, logvar_A)

        # Encoder B: frozen or trainable depending on config
        if self.freeze_encoder_b:
            with torch.no_grad():
                mu_B, logvar_B = self.encoder_B(x)
                z_B_euc = self.reparameterize(mu_B, logvar_B)
        else:
            # Allow gradients through encoder_B for Option C
            mu_B, logvar_B = self.encoder_B(x)
            z_B_euc = self.reparameterize(mu_B, logvar_B)

        # Trainable projection to Poincaré ball
        if self.use_dual_projection:
            z_A_hyp, z_B_hyp = self.projection(z_A_euc, z_B_euc)
        else:
            z_A_hyp = self.projection(z_A_euc)
            z_B_hyp = self.projection(z_B_euc)

        # Compute control signals (if enabled)
        if compute_control and self.controller is not None:
            # V5.12.2: Use hyperbolic distance (poincare_distance) instead of Euclidean norm
            # This ensures Controller operates in consistent geometry with losses
            curvature_ctrl = self.projection.get_curvature() if hasattr(self.projection, "get_curvature") else 1.0
            origin = torch.zeros_like(z_A_hyp)
            radius_A = poincare_distance(z_A_hyp, origin, c=curvature_ctrl).mean()
            radius_B = poincare_distance(z_B_hyp, origin, c=curvature_ctrl).mean()
            kl_A = -0.5 * (1 + logvar_A - mu_A.pow(2) - logvar_A.exp()).sum(dim=-1).mean()
            kl_B = -0.5 * (1 + logvar_B - mu_B.pow(2) - logvar_B.exp()).sum(dim=-1).mean()

            batch_stats = torch.stack(
                [
                    radius_A,
                    radius_B,
                    torch.tensor(1.0, device=x.device),
                    torch.tensor(1.0, device=x.device),
                    kl_A,
                    kl_B,
                    torch.tensor(0.0, device=x.device),
                    torch.tensor(0.0, device=x.device),
                ]
            )

            control = self.controller(batch_stats)
            control = {k: v.squeeze(0) for k, v in control.items()}
        else:
            control = {k: torch.tensor(v, device=x.device) for k, v in self.default_control.items()}

        # V5.12.1: Decoder uses hyperbolic representation via log_map_zero
        # This creates gradient flow through the hyperbolic projection,
        # providing geometric pressure for richness preservation
        curvature = self.projection.get_curvature() if hasattr(self.projection, "get_curvature") else 1.0
        z_tangent_A = log_map_zero(z_A_hyp, c=curvature)
        logits_A = self.decoder_A(z_tangent_A)

        return {
            "z_A_euc": z_A_euc,
            "z_B_euc": z_B_euc,
            "mu_A": mu_A,
            "mu_B": mu_B,
            "logvar_A": logvar_A,
            "logvar_B": logvar_B,
            "z_A_hyp": z_A_hyp,
            "z_B_hyp": z_B_hyp,
            "z_tangent_A": z_tangent_A,  # V5.12.1: tangent space representation
            "control": control,
            "logits_A": logits_A,
        }

    def get_trainable_parameters(self):
        """Get trainable parameters including optionally unfrozen encoders.

        Returns:
            List of trainable parameters
        """
        params = super().get_trainable_parameters()

        if not self.freeze_encoder_a:
            params.extend(self.encoder_A.parameters())

        if not self.freeze_encoder_b:
            params.extend(self.encoder_B.parameters())

        return params

    def get_param_groups(self, base_lr: float):
        """Get parameter groups with different learning rates.

        Returns param groups for optimizer with encoders at lower LR.
        For dual projection, proj_A and proj_B get separate groups.
        Frozen components are excluded from param groups.

        Args:
            base_lr: Base learning rate

        Returns:
            List of param group dicts for optimizer
        """
        param_groups = []

        # Projections always trainable (fast adaptation layer)
        if self.use_dual_projection:
            param_groups.append(
                {
                    "params": list(self.projection.proj_A.parameters()),
                    "lr": base_lr,
                    "name": "proj_A",
                }
            )
            if hasattr(self.projection, "proj_B"):
                proj_b_params = list(self.projection.proj_B.parameters())
            else:
                proj_b_params = list(self.projection.proj_B_radius.parameters())
            param_groups.append(
                {
                    "params": proj_b_params,
                    "lr": base_lr,
                    "name": "proj_B",
                }
            )
        else:
            param_groups.append(
                {
                    "params": list(self.projection.parameters()),
                    "lr": base_lr,
                    "name": "projection",
                }
            )

        # Controller (can be frozen by homeostasis)
        if self.controller is not None and not getattr(self, "freeze_controller", False):
            param_groups.append(
                {
                    "params": list(self.controller.parameters()),
                    "lr": base_lr * self.controller_lr_scale,
                    "name": "controller",
                }
            )

        # Encoder A (can be frozen by homeostasis)
        if not self.freeze_encoder_a:
            param_groups.append(
                {
                    "params": list(self.encoder_A.parameters()),
                    "lr": base_lr * self.encoder_a_lr_scale,
                    "name": "encoder_A",
                }
            )

        # Encoder B (can be frozen by homeostasis)
        if not self.freeze_encoder_b:
            param_groups.append(
                {
                    "params": list(self.encoder_B.parameters()),
                    "lr": base_lr * self.encoder_b_lr_scale,
                    "name": "encoder_B",
                }
            )

        # Decoder A (V5.12.1: must be trainable when using hyperbolic input)
        if not getattr(self, "freeze_decoder", True):
            param_groups.append(
                {
                    "params": list(self.decoder_A.parameters()),
                    "lr": base_lr,  # Full LR for decoder adaptation
                    "name": "decoder_A",
                }
            )

        return param_groups

    def rebuild_optimizer(self, optimizer, base_lr: float):
        """Rebuild optimizer with current freeze states.

        Called when homeostasis changes freeze states to update optimizer.

        Args:
            optimizer: Current optimizer (for weight_decay)
            base_lr: Base learning rate

        Returns:
            New optimizer with correct param groups
        """
        param_groups = self.get_param_groups(base_lr)
        new_optimizer = optim.AdamW(
            param_groups,
            weight_decay=optimizer.defaults.get("weight_decay", 1e-4),
        )
        return new_optimizer

    def count_parameters(self) -> dict[str, int]:
        """Count parameters by component.

        Returns:
            Dict with parameter counts by component
        """
        counts = super().count_parameters()

        if not self.freeze_encoder_b:
            encoder_b_params = sum(p.numel() for p in self.encoder_B.parameters())
            counts["encoder_b_trainable"] = encoder_b_params
            counts["trainable"] += encoder_b_params
            counts["frozen"] -= encoder_b_params

        return counts


# Backward-compatible alias (deprecated name)
TernaryVAEV5_11_OptionC = TernaryVAEV5_11_PartialFreeze

__all__ = [
    "TernaryVAEV5_11_PartialFreeze",
    "TernaryVAEV5_11_OptionC",  # Deprecated alias for backward compatibility
]
