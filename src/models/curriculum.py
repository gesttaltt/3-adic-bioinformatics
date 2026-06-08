# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Continuous Curriculum Module for StateNet-driven training.

This module implements a differentiable curriculum that smoothly transitions
the training focus from radial hierarchy learning to angular discrimination.

Key design: StateNet FULLY CONTROLS the curriculum via delta_curriculum output.
No hard-coded thresholds - the optimal curriculum trajectory emerges from learning.

tau = 0: Pure radial learning (coarse tree structure)
tau = 1: Pure ranking learning (fine angular discrimination)

Single responsibility: Manage curriculum state and loss modulation.
"""

from typing import Any

import torch
import torch.nn as nn


class ContinuousCurriculumModule(nn.Module):
    """Differentiable curriculum FULLY CONTROLLED by StateNet.

    The curriculum position tau ∈ [0, 1] determines the blend between:
    - Radial stratification loss (tree structure via radial position)
    - Ranking loss (angular discrimination between similar points)

    StateNet observes the training state (including radial_loss and tau itself)
    and outputs delta_curriculum to adjust tau. This creates a feedback loop:

        StateNet observes → StateNet decides → tau changes → loss changes →
        model learns → StateNet observes new state → ...

    This allows fully emergent curriculum pacing without predefined schedules.

    Args:
        initial_tau: Starting curriculum position (default: 0.0 = pure radial)
        tau_min: Minimum tau bound (default: 0.0)
        tau_max: Maximum tau bound (default: 1.0)
        tau_scale: Scaling factor for delta_curriculum (default: 0.1)
        momentum: EMA momentum for tau history tracking (default: 0.95)
    """

    def __init__(
        self,
        initial_tau: float = 0.0,
        tau_min: float = 0.0,
        tau_max: float = 1.0,
        tau_scale: float = 0.1,
        momentum: float = 0.95,
    ):
        super().__init__()

        # Core curriculum state (buffer, not parameter - updated by StateNet)
        self.register_buffer("tau", torch.tensor(initial_tau, dtype=torch.float32))
        self.tau: torch.Tensor  # Type hint for mypy

        # Bounds
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_scale = tau_scale
        self.momentum = momentum

        # History tracking for analysis
        self.register_buffer("tau_ema", torch.tensor(initial_tau, dtype=torch.float32))
        self.tau_ema: torch.Tensor  # Type hint for mypy
        self.tau_history: list[float] = []  # For logging/visualization

    def update_tau(self, delta_curriculum: float) -> torch.Tensor:
        """Update tau based on StateNet's delta_curriculum output.

        Positive delta_curriculum: Advance toward ranking (increase tau)
        Negative delta_curriculum: Retreat toward radial (decrease tau)

        StateNet learns when to advance/retreat based on:
        - radial_stratification_loss (low = radial structure learned)
        - ranking_correlations (stagnant = need more ranking focus)
        - coverage trends (improving/degrading)

        Args:
            delta_curriculum: StateNet's curriculum correction in [-1, 1]

        Returns:
            Updated tau value
        """
        # Apply scaled correction
        delta = self.tau_scale * delta_curriculum
        new_tau = self.tau + delta

        # Clamp to bounds
        new_tau = torch.clamp(new_tau, self.tau_min, self.tau_max)

        # Update tau
        self.tau = new_tau

        # Update EMA for smoothed tracking
        self.tau_ema = self.momentum * self.tau_ema + (1 - self.momentum) * new_tau

        # Record history
        self.tau_history.append(new_tau.item())

        return self.tau

    def get_tau(self) -> torch.Tensor:
        """Return current curriculum position."""
        return self.tau

    def get_tau_ema(self) -> torch.Tensor:
        """Return smoothed curriculum position."""
        return self.tau_ema

    def modulate_losses(self, radial_loss: torch.Tensor, ranking_loss: torch.Tensor) -> torch.Tensor:
        """Blend radial and ranking losses based on current tau.

        Loss = (1 - tau) * radial_loss + tau * ranking_loss

        At tau=0: Pure radial (learning tree structure)
        At tau=0.5: Equal blend (transitioning)
        At tau=1: Pure ranking (refining angular discrimination)

        Args:
            radial_loss: Radial stratification loss (scalar tensor)
            ranking_loss: Ranking/triplet loss (scalar tensor)

        Returns:
            Blended structure loss
        """
        tau = self.tau
        return (1 - tau) * radial_loss + tau * ranking_loss

    def get_loss_weights(self) -> tuple[float, float]:
        """Return current (radial_weight, ranking_weight) for logging."""
        tau = self.tau.item()
        return (1 - tau, tau)

    def get_state_dict_extra(self) -> dict[str, Any]:
        """Return extra state for checkpointing."""
        return {
            "tau_history": self.tau_history[-1000:],  # Last 1000 values
        }

    def load_state_dict_extra(self, state_dict: dict[str, Any]):
        """Load extra state from checkpoint."""
        if "tau_history" in state_dict:
            self.tau_history = state_dict["tau_history"]

    def reset(self, initial_tau: float | None = None):
        """Reset curriculum to initial state."""
        if initial_tau is not None:
            self.tau = torch.tensor(initial_tau, dtype=torch.float32, device=self.tau.device)
            self.tau_ema = self.tau.clone()
        else:
            self.tau = torch.tensor(0.0, dtype=torch.float32, device=self.tau.device)
            self.tau_ema = self.tau.clone()
        self.tau_history = []

    def extra_repr(self) -> str:
        return f"tau={self.tau.item():.3f}, tau_min={self.tau_min}, tau_max={self.tau_max}, tau_scale={self.tau_scale}"


class CurriculumScheduler:
    """Optional: Provides curriculum statistics and diagnostics.

    This is a helper class for monitoring curriculum behavior, not for
    controlling it (StateNet controls curriculum).
    """

    def __init__(self, curriculum: ContinuousCurriculumModule):
        self.curriculum = curriculum
        self.delta_history: list[float] = []
        self.radial_loss_history: list[float] = []
        self.ranking_loss_history: list[float] = []

    def record_step(self, delta_curriculum: float, radial_loss: float, ranking_loss: float):
        """Record a curriculum step for analysis."""
        self.delta_history.append(delta_curriculum)
        self.radial_loss_history.append(radial_loss)
        self.ranking_loss_history.append(ranking_loss)

    def get_stats(self) -> dict[str, float | str | int]:
        """Get curriculum statistics."""
        tau_history = self.curriculum.tau_history

        if len(tau_history) < 2:
            return {
                "tau_current": self.curriculum.tau.item(),
                "tau_velocity": 0.0,
                "tau_trend": "stable",
            }

        tau_current = tau_history[-1]
        tau_prev = tau_history[-min(10, len(tau_history)) : -1]
        tau_velocity = tau_current - (sum(tau_prev) / len(tau_prev)) if tau_prev else 0.0

        # Determine trend
        if tau_velocity > 0.01:
            trend = "advancing"
        elif tau_velocity < -0.01:
            trend = "retreating"
        else:
            trend = "stable"

        return {
            "tau_current": tau_current,
            "tau_ema": self.curriculum.tau_ema.item(),
            "tau_velocity": tau_velocity,
            "tau_trend": trend,
            "history_length": len(tau_history),
        }
