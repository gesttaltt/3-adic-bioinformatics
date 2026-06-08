# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Epsilon-StateNet: Meta-Learning Controller Coupled with Epsilon-VAE.

This module couples the Epsilon-VAE (checkpoint weight encoder) with a
StateNet controller that learns to make training decisions based on:
1. The latent representation of current model weights (from Epsilon-VAE)
2. Current training metrics (coverage, hierarchy, Q)
3. Training trajectory history

The StateNet learns dynamics that:
- Maintain 100% coverage (p-adic generalization capability)
- Push hierarchy as deep as possible
- Balance exploration vs exploitation in training

Architecture:

    Model Weights ──► Epsilon Encoder ──► z_epsilon (latent position)
                                              │
    Current Metrics ──────────────────────────┤
                                              ▼
                                    ┌─────────────────┐
                                    │    StateNet     │
                                    │   Controller    │
                                    └─────────────────┘
                                              │
                              ┌───────────────┼───────────────┐
                              ▼               ▼               ▼
                         LR Scales      Freeze Mask    Loss Weights
"""

import torch
import torch.nn as nn
from torch import Tensor

from .epsilon_vae import CheckpointEncoder


class StateNet(nn.Module):
    """Neural network controller that outputs training decisions.

    Takes the Epsilon-VAE latent + current metrics and outputs:
    - Learning rate scales for each component
    - Freeze probabilities for each component
    - Loss weight adjustments
    """

    def __init__(
        self,
        epsilon_dim: int = 32,
        metric_dim: int = 8,
        hidden_dim: int = 64,
        n_components: int = 4,  # encoder_A, encoder_B, projection, controller
    ):
        super().__init__()
        self.epsilon_dim = epsilon_dim
        self.metric_dim = metric_dim
        self.n_components = n_components

        input_dim = epsilon_dim + metric_dim

        # Shared encoder
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # LR scale head: outputs log-scale for each component
        # Output is log(lr_scale) to ensure positive values
        self.lr_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_components),
        )

        # Freeze probability head: outputs freeze probability for each component
        self.freeze_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_components),
            nn.Sigmoid(),
        )

        # Loss weight head: outputs adjustments for radial, margin, rank losses
        self.loss_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # radial, margin, rank
            nn.Softplus(),  # Ensure positive
        )

    def forward(
        self,
        z_epsilon: Tensor,
        metrics: Tensor,
    ) -> dict[str, Tensor]:
        """
        Forward pass.

        Args:
            z_epsilon: Latent from Epsilon-VAE [batch, epsilon_dim] or [epsilon_dim]
            metrics: Current metrics [batch, metric_dim] or [metric_dim]

        Returns:
            Dict with:
                lr_scales: [n_components] - learning rate multipliers
                freeze_probs: [n_components] - probability of freezing each component
                loss_weights: [3] - weights for radial, margin, rank losses
        """
        # Handle single sample case
        if z_epsilon.dim() == 1:
            z_epsilon = z_epsilon.unsqueeze(0)
        if metrics.dim() == 1:
            metrics = metrics.unsqueeze(0)

        # Concatenate inputs
        x = torch.cat([z_epsilon, metrics], dim=-1)

        # Shared encoding
        h = self.shared(x)

        # Compute outputs
        lr_log_scales = self.lr_head(h)
        lr_scales = torch.exp(lr_log_scales.clamp(-3, 1))  # Range: ~0.05 to ~2.7

        freeze_probs = self.freeze_head(h)
        loss_weights = self.loss_head(h) + 0.1  # Minimum weight of 0.1

        return {
            "lr_scales": lr_scales.squeeze(0),
            "freeze_probs": freeze_probs.squeeze(0),
            "loss_weights": loss_weights.squeeze(0),
        }


class EpsilonStateNet(nn.Module):
    """Coupled Epsilon-VAE + StateNet controller.

    Provides end-to-end training control based on:
    1. Current model weight state (via Epsilon encoder)
    2. Current training metrics
    3. Learned dynamics for maintaining coverage + hierarchy
    """

    def __init__(
        self,
        embed_dim: int = 64,
        epsilon_dim: int = 32,
        metric_dim: int = 8,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_components: int = 4,
    ):
        super().__init__()
        self.epsilon_dim = epsilon_dim
        self.metric_dim = metric_dim

        # Epsilon encoder (from Epsilon-VAE)
        self.epsilon_encoder = CheckpointEncoder(
            embed_dim=embed_dim,
            latent_dim=epsilon_dim,
            n_heads=n_heads,
        )

        # StateNet controller
        self.statenet = StateNet(
            epsilon_dim=epsilon_dim,
            metric_dim=metric_dim,
            hidden_dim=hidden_dim,
            n_components=n_components,
        )

        # Metric history buffer for trajectory awareness
        self.register_buffer("metric_history", torch.zeros(10, metric_dim))
        self.register_buffer("history_idx", torch.tensor(0))

    def encode_weights(self, weight_blocks: list[Tensor]) -> tuple[Tensor, Tensor]:
        """Encode model weights to latent space."""
        return self.epsilon_encoder(weight_blocks)

    def update_history(self, metrics: Tensor):
        """Update metric history buffer."""
        idx = self.history_idx.item() % 10
        self.metric_history[idx] = metrics
        self.history_idx += 1

    def get_trajectory_features(self) -> Tensor:
        """Get trajectory-aware features from history."""
        if self.history_idx < 2:
            return torch.zeros(4, device=self.metric_history.device)

        # Compute velocity and acceleration from history
        n = min(self.history_idx.item(), 10)
        history = self.metric_history[:n]

        velocity = history[-1] - history[0] if n > 1 else torch.zeros_like(history[0])
        mean_metrics = history.mean(dim=0)

        # Return: [mean_coverage, mean_hierarchy, velocity_coverage, velocity_hierarchy]
        return torch.cat(
            [
                mean_metrics[:2],  # mean coverage, mean hierarchy
                velocity[:2],  # velocity coverage, velocity hierarchy
            ]
        )

    def forward(
        self,
        weight_blocks: list[Tensor],
        current_metrics: Tensor,
    ) -> dict[str, Tensor]:
        """
        Get training control signals.

        Args:
            weight_blocks: List of weight tensors from current model
            current_metrics: Current training metrics [coverage, hier_A, hier_B,
                           dist_corr_A, dist_corr_B, r_v0, r_v9, Q]

        Returns:
            Dict with training control signals
        """
        # Encode current weight state
        mu, logvar = self.encode_weights(weight_blocks)
        z_epsilon = self.epsilon_encoder.reparameterize(mu, logvar)

        # Update history and get trajectory features
        self.update_history(current_metrics)

        # Get control signals from StateNet
        controls = self.statenet(z_epsilon, current_metrics)

        # Add epsilon latent to outputs for monitoring
        controls["z_epsilon"] = z_epsilon
        controls["mu"] = mu
        controls["logvar"] = logvar

        return controls

    def get_freeze_decisions(
        self,
        freeze_probs: Tensor,
        threshold: float = 0.5,
        temperature: float = 1.0,
    ) -> dict[str, bool]:
        """Convert freeze probabilities to discrete decisions.

        Args:
            freeze_probs: [n_components] probabilities
            threshold: Decision threshold
            temperature: Softmax temperature for stochastic decisions

        Returns:
            Dict mapping component names to freeze decisions
        """
        component_names = ["encoder_A", "encoder_B", "projection", "controller"]

        # Deterministic decisions based on threshold
        decisions = {}
        for i, name in enumerate(component_names):
            decisions[name] = freeze_probs[i].item() > threshold

        return decisions


class EpsilonStateNetLoss(nn.Module):
    """Loss function for training the EpsilonStateNet controller.

    The controller learns to:
    1. Maintain high coverage (target: 100%)
    2. Maximize hierarchy depth (more negative is better)
    3. Smooth control signals (avoid oscillation)
    """

    def __init__(
        self,
        coverage_target: float = 1.0,
        hierarchy_target: float = -0.85,
        coverage_weight: float = 10.0,
        hierarchy_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        kl_weight: float = 0.01,
    ):
        super().__init__()
        self.coverage_target = coverage_target
        self.hierarchy_target = hierarchy_target
        self.coverage_weight = coverage_weight
        self.hierarchy_weight = hierarchy_weight
        self.smoothness_weight = smoothness_weight
        self.kl_weight = kl_weight

        # Previous control signals for smoothness
        self.prev_controls = None

    def forward(
        self,
        controls: dict[str, Tensor],
        actual_coverage: float,
        actual_hierarchy: float,
    ) -> dict[str, Tensor]:
        """
        Compute loss for controller training.

        The key insight: we reward the controller when:
        - Coverage stays at target (100%)
        - Hierarchy improves (more negative)

        Args:
            controls: Output from EpsilonStateNet
            actual_coverage: Actual coverage achieved
            actual_hierarchy: Actual hierarchy achieved

        Returns:
            Dict with loss components
        """
        # Coverage loss: heavy penalty for dropping below target
        coverage_error = max(0, self.coverage_target - actual_coverage)
        coverage_loss = self.coverage_weight * (coverage_error**2)

        # Hierarchy loss: reward for getting more negative
        # Use asymmetric loss: no penalty for exceeding target
        hierarchy_error = max(0, actual_hierarchy - self.hierarchy_target)
        hierarchy_loss = self.hierarchy_weight * (hierarchy_error**2)

        # Smoothness loss: penalize large changes in control signals
        smoothness_loss = torch.tensor(0.0, device=controls["lr_scales"].device)
        if self.prev_controls is not None:
            lr_diff = (controls["lr_scales"] - self.prev_controls["lr_scales"]).pow(2).mean()
            freeze_diff = (controls["freeze_probs"] - self.prev_controls["freeze_probs"]).pow(2).mean()
            smoothness_loss = self.smoothness_weight * (lr_diff + freeze_diff)

        # KL loss on epsilon latent
        mu = controls["mu"]
        logvar = controls["logvar"]
        kl_loss = self.kl_weight * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))

        # Total loss
        total = coverage_loss + hierarchy_loss + smoothness_loss + kl_loss

        # Update previous controls
        self.prev_controls = {k: v.detach().clone() for k, v in controls.items() if isinstance(v, Tensor)}

        return {
            "total": total,
            "coverage_loss": torch.tensor(coverage_loss),
            "hierarchy_loss": torch.tensor(hierarchy_loss),
            "smoothness_loss": smoothness_loss,
            "kl_loss": kl_loss,
        }


def create_epsilon_statenet(
    pretrained_epsilon_path: str | None = None,
    **kwargs,
) -> EpsilonStateNet:
    """Create an EpsilonStateNet, optionally loading pretrained Epsilon encoder.

    Args:
        pretrained_epsilon_path: Path to pretrained Epsilon-VAE checkpoint
        **kwargs: Additional arguments for EpsilonStateNet

    Returns:
        Initialized EpsilonStateNet
    """
    model = EpsilonStateNet(**kwargs)

    if pretrained_epsilon_path is not None:
        # Load pretrained Epsilon encoder weights
        ckpt = torch.load(pretrained_epsilon_path, map_location="cpu", weights_only=False)

        # Extract encoder weights
        encoder_state = {}
        for k, v in ckpt.get("model_state_dict", ckpt).items():
            if k.startswith("encoder."):
                encoder_state[k.replace("encoder.", "")] = v

        if encoder_state:
            model.epsilon_encoder.load_state_dict(encoder_state, strict=False)
            print(f"Loaded pretrained Epsilon encoder from {pretrained_epsilon_path}")
        else:
            print("Warning: No encoder weights found in checkpoint")

    return model


__all__ = [
    "StateNet",
    "EpsilonStateNet",
    "EpsilonStateNetLoss",
    "create_epsilon_statenet",
]
