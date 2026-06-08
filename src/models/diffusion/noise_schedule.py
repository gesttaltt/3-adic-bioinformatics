# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Noise schedules for discrete diffusion.

Provides various noise schedules including p-adic aware variants
that respect the hierarchical structure of ternary representations.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class NoiseSchedule(ABC):
    """Abstract base class for noise schedules."""

    def __init__(self, n_timesteps: int):
        """Initialize noise schedule.

        Args:
            n_timesteps: Number of diffusion timesteps
        """
        self.n_timesteps = n_timesteps

    @abstractmethod
    def get_rates(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get forward and reverse rates at timestep t.

        Args:
            t: Timestep tensor

        Returns:
            Tuple of (alpha_t, beta_t) rates
        """
        pass

    @abstractmethod
    def get_cumulative(self, t: torch.Tensor) -> torch.Tensor:
        """Get cumulative product up to timestep t.

        Args:
            t: Timestep tensor

        Returns:
            Cumulative alpha product
        """
        pass


class LinearSchedule(NoiseSchedule):
    """Linear noise schedule.

    Simple linear interpolation from beta_start to beta_end.
    """

    def __init__(
        self,
        n_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        """Initialize linear schedule.

        Args:
            n_timesteps: Number of timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
        """
        super().__init__(n_timesteps)

        # Compute beta schedule
        self.betas = torch.linspace(beta_start, beta_end, n_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def get_rates(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get rates at timestep t."""
        t = t.long().clamp(0, self.n_timesteps - 1)
        return self.alphas[t], self.betas[t]

    def get_cumulative(self, t: torch.Tensor) -> torch.Tensor:
        """Get cumulative alpha product."""
        t = t.long().clamp(0, self.n_timesteps - 1)
        return self.alpha_cumprod[t]


class CosineSchedule(NoiseSchedule):
    """Cosine noise schedule.

    Smoother schedule that improves sample quality.

    References:
        - Nichol & Dhariwal (2021): Improved Denoising Diffusion
    """

    def __init__(
        self,
        n_timesteps: int = 1000,
        s: float = 0.008,
    ):
        """Initialize cosine schedule.

        Args:
            n_timesteps: Number of timesteps
            s: Small offset to prevent singularity
        """
        super().__init__(n_timesteps)
        self.s = s

        # Compute schedule
        steps = torch.arange(n_timesteps + 1, dtype=torch.float64)
        f_t = torch.cos(((steps / n_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        self.alpha_cumprod = f_t / f_t[0]
        self.alpha_cumprod = self.alpha_cumprod[:-1].float()

        # Derive betas
        alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alpha_cumprod[:-1]])
        self.betas = 1 - (self.alpha_cumprod / alpha_cumprod_prev)
        self.betas = self.betas.clamp(max=0.999)
        self.alphas = 1 - self.betas

    def get_rates(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get rates at timestep t."""
        t = t.long().clamp(0, self.n_timesteps - 1)
        return self.alphas[t], self.betas[t]

    def get_cumulative(self, t: torch.Tensor) -> torch.Tensor:
        """Get cumulative alpha product."""
        t = t.long().clamp(0, self.n_timesteps - 1)
        return self.alpha_cumprod[t]


class PAdicNoiseSchedule(NoiseSchedule):
    """P-adic aware noise schedule.

    Corrupts positions based on their p-adic valuation,
    preserving hierarchical structure during diffusion.

    Higher valuation positions (more divisible by p) are
    corrupted more slowly, maintaining global structure.
    """

    def __init__(
        self,
        n_timesteps: int = 1000,
        p: int = 3,
        max_length: int = 512,
        base_schedule: str = "cosine",
    ):
        """Initialize p-adic schedule.

        Args:
            n_timesteps: Number of timesteps
            p: Prime for p-adic valuation
            max_length: Maximum sequence length
            base_schedule: Base schedule type ('linear', 'cosine')
        """
        super().__init__(n_timesteps)
        self.p = p
        self.max_length = max_length

        # Base schedule
        if base_schedule == "cosine":
            self.base = CosineSchedule(n_timesteps)
        else:
            self.base = LinearSchedule(n_timesteps)

        # Compute p-adic valuations for positions
        self.valuations = self._compute_valuations(max_length, p)

        # Position-specific rate modifiers
        self.rate_modifiers = self._compute_rate_modifiers()

    def _compute_valuations(self, length: int, p: int) -> torch.Tensor:
        """Compute p-adic valuation for each position.

        v_p(n) = largest k such that p^k divides n.

        Args:
            length: Sequence length
            p: Prime number

        Returns:
            (length,) tensor of valuations
        """
        valuations = torch.zeros(length)

        for i in range(1, length + 1):
            n = i
            v = 0
            while n % p == 0:
                v += 1
                n //= p
            valuations[i - 1] = v

        return valuations

    def _compute_rate_modifiers(self) -> torch.Tensor:
        """Compute position-dependent rate modifiers.

        Higher valuation = slower corruption rate.

        Returns:
            (max_length,) rate modifiers
        """
        # Normalize valuations
        max_val = self.valuations.max()
        normalized = self.valuations / (max_val + 1)

        # Higher valuation -> lower rate (slower corruption)
        modifiers = 1 - 0.5 * normalized

        return modifiers

    def get_rates(
        self,
        t: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get position-aware rates.

        Args:
            t: Timestep tensor
            positions: Position indices (optional)

        Returns:
            Tuple of (alpha_t, beta_t) possibly position-dependent
        """
        alpha, beta = self.base.get_rates(t)

        if positions is not None:
            # Apply position-specific modifiers
            modifiers = self.rate_modifiers[positions]
            beta = beta.unsqueeze(-1) * modifiers
            alpha = 1 - beta

        return alpha, beta

    def get_cumulative(
        self,
        t: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get position-aware cumulative product.

        Args:
            t: Timestep tensor
            positions: Position indices (optional)

        Returns:
            Cumulative alpha product
        """
        base_cumprod = self.base.get_cumulative(t)

        if positions is not None:
            # Modify based on position valuations
            modifiers = self.rate_modifiers[positions]
            # Higher modifier = less corruption at this position
            adjusted = base_cumprod.unsqueeze(-1) ** modifiers

            return adjusted

        return base_cumprod

    def get_corruption_order(self, length: int) -> torch.Tensor:
        """Get order in which positions should be corrupted.

        Low valuation positions are corrupted first.

        Args:
            length: Sequence length

        Returns:
            (length,) indices in corruption order
        """
        valuations = self.valuations[:length]
        # Sort by valuation (ascending = corrupt low valuation first)
        return valuations.argsort()

    def get_denoising_order(self, length: int) -> torch.Tensor:
        """Get order for denoising (reverse of corruption).

        High valuation positions are denoised first.

        Args:
            length: Sequence length

        Returns:
            (length,) indices in denoising order
        """
        valuations = self.valuations[:length]
        # Sort by valuation descending
        return valuations.argsort(descending=True)


class AdaptiveSchedule(NoiseSchedule):
    """Learned noise schedule.

    Parameterizes the schedule with a neural network
    for task-specific optimization.
    """

    def __init__(
        self,
        n_timesteps: int = 1000,
        hidden_dim: int = 64,
    ):
        """Initialize adaptive schedule.

        Args:
            n_timesteps: Number of timesteps
            hidden_dim: Hidden dimension for schedule network
        """
        super().__init__(n_timesteps)

        # Schedule network
        self.schedule_net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Initialize to approximate cosine schedule
        self._initialize_to_cosine()

    def _initialize_to_cosine(self):
        """Initialize network to approximate cosine schedule."""
        cosine = CosineSchedule(self.n_timesteps)

        # Train briefly to match cosine
        optimizer = torch.optim.Adam(self.schedule_net.parameters(), lr=0.01)

        for _ in range(100):
            t = torch.randint(0, self.n_timesteps, (64,))
            t_normalized = t.float().unsqueeze(-1) / self.n_timesteps

            target = cosine.get_cumulative(t)
            pred = self.schedule_net(t_normalized).squeeze(-1)

            loss = nn.functional.mse_loss(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def get_rates(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get learned rates at timestep t."""
        t_normalized = t.float().unsqueeze(-1) / self.n_timesteps
        alpha_cumprod = self.schedule_net(t_normalized).squeeze(-1)

        # Compute alpha and beta from cumulative product
        # alpha_cumprod[t] = prod_{s=0}^{t} alpha[s]
        # Approximate: alpha[t] ≈ alpha_cumprod[t] / alpha_cumprod[t-1]
        t_prev = (t - 1).clamp(min=0)
        t_prev_normalized = t_prev.float().unsqueeze(-1) / self.n_timesteps
        alpha_cumprod_prev = self.schedule_net(t_prev_normalized).squeeze(-1)

        alpha = alpha_cumprod / (alpha_cumprod_prev + 1e-8)
        alpha = alpha.clamp(0.001, 0.999)
        beta = 1 - alpha

        return alpha, beta

    def get_cumulative(self, t: torch.Tensor) -> torch.Tensor:
        """Get learned cumulative product."""
        t_normalized = t.float().unsqueeze(-1) / self.n_timesteps
        return self.schedule_net(t_normalized).squeeze(-1)
