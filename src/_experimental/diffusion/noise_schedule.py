# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Noise schedules for diffusion models.

This module provides various noise schedules for diffusion models,
including linear, cosine, and exponential schedules.

References:
    - Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
    - Nichol & Dhariwal, "Improved Denoising Diffusion" (2021)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class NoiseScheduler(nn.Module):
    """Noise schedule for diffusion models.

    Supports multiple schedule types:
    - linear: Linear interpolation of beta
    - cosine: Cosine schedule (better for small timesteps)
    - sigmoid: Sigmoid schedule for smoother transitions
    - exponential: Exponential schedule

    Args:
        n_steps: Number of diffusion steps
        schedule_type: Type of schedule
        beta_start: Starting beta value (for linear/exponential)
        beta_end: Ending beta value (for linear/exponential)
        s: Offset for cosine schedule (default 0.008)
    """

    def __init__(
        self,
        n_steps: int = 1000,
        schedule_type: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        s: float = 0.008,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.schedule_type = schedule_type
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.s = s

        # Compute schedule
        self._compute_schedule()

    def _compute_schedule(self):
        """Compute beta, alpha, and alpha_bar schedules."""
        if self.schedule_type == "linear":
            betas = self._linear_schedule()
        elif self.schedule_type == "cosine":
            betas = self._cosine_schedule()
        elif self.schedule_type == "sigmoid":
            betas = self._sigmoid_schedule()
        elif self.schedule_type == "exponential":
            betas = self._exponential_schedule()
        else:
            raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

        # Clamp betas to valid range
        betas = torch.clamp(betas, 0.0001, 0.999)

        # Compute derived quantities
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # For posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # Register buffers (not trainable parameters)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))
        self.register_buffer("_posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

    def _linear_schedule(self) -> Tensor:
        """Linear noise schedule."""
        return torch.linspace(self.beta_start, self.beta_end, self.n_steps)

    def _cosine_schedule(self) -> Tensor:
        """Cosine noise schedule from Nichol & Dhariwal."""
        steps = self.n_steps + 1
        t = torch.linspace(0, self.n_steps, steps) / self.n_steps
        alphas_cumprod = torch.cos((t + self.s) / (1 + self.s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas

    def _sigmoid_schedule(self) -> Tensor:
        """Sigmoid noise schedule."""
        t = torch.linspace(-6, 6, self.n_steps)
        betas = torch.sigmoid(t) * (self.beta_end - self.beta_start) + self.beta_start
        return betas

    def _exponential_schedule(self) -> Tensor:
        """Exponential noise schedule."""
        t = torch.linspace(0, 1, self.n_steps)
        betas = self.beta_start * (self.beta_end / self.beta_start) ** t
        return betas

    def _extract(self, a: Tensor, t: Tensor, x_shape: tuple[int, ...]) -> Tensor:
        """Extract values from a at timestep t.

        Args:
            a: Tensor to index into
            t: Timesteps to extract
            x_shape: Shape of x for broadcasting

        Returns:
            Extracted values broadcast to x_shape
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.view(batch_size, *((1,) * (len(x_shape) - 1)))

    def add_noise(
        self,
        x: Tensor,
        t: Tensor,
        noise: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Add noise to x at timestep t (forward diffusion).

        q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)

        Args:
            x: Original data of shape (batch, ...)
            t: Timesteps of shape (batch,)
            noise: Optional pre-generated noise

        Returns:
            Tuple of (noised_x, noise)
        """
        if noise is None:
            noise = torch.randn_like(x)

        sqrt_alpha_prod = self._extract(self.sqrt_alphas_cumprod, t, x.shape)
        sqrt_one_minus_alpha_prod = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)

        noised_x = sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise
        return noised_x, noise

    def remove_noise(
        self,
        x_t: Tensor,
        predicted_noise: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Remove noise from x_t using predicted noise.

        Predicts x_0 from x_t and noise prediction.

        Args:
            x_t: Noised data at timestep t
            predicted_noise: Model's noise prediction
            t: Timesteps

        Returns:
            Predicted x_0
        """
        sqrt_recip_alphas = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

        return sqrt_recip_alphas * x_t - sqrt_recipm1_alphas * predicted_noise

    def posterior_mean(
        self,
        x_start: Tensor,
        x_t: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Compute posterior mean for p(x_{t-1} | x_t, x_0).

        Args:
            x_start: Predicted x_0
            x_t: Current noised data
            t: Timesteps

        Returns:
            Posterior mean
        """
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        return coef1 * x_start + coef2 * x_t

    def get_posterior_variance(self, t: Tensor, x_shape: tuple[int, ...]) -> Tensor:
        """Get posterior variance at timestep t.

        Args:
            t: Timesteps
            x_shape: Shape for broadcasting

        Returns:
            Posterior variance
        """
        return self._extract(self._posterior_variance, t, x_shape)

    def step(
        self,
        model_output: Tensor,
        t: Tensor,
        x_t: Tensor,
        predict_epsilon: bool = True,
        clip_denoised: bool = True,
        clip_range: tuple[float, float] = (-1.0, 1.0),
    ) -> Tensor:
        """Perform one reverse diffusion step.

        Args:
            model_output: Model's prediction (noise or x_0)
            t: Current timesteps
            x_t: Current noised data
            predict_epsilon: Whether model predicts noise (True) or x_0 (False)
            clip_denoised: Whether to clip predicted x_0
            clip_range: Range for clipping

        Returns:
            x_{t-1}
        """
        # Get x_0 prediction
        x_start = self.remove_noise(x_t, model_output, t) if predict_epsilon else model_output

        # Clip if requested
        if clip_denoised:
            x_start = torch.clamp(x_start, clip_range[0], clip_range[1])

        # Compute mean and variance
        mean = self.posterior_mean(x_start, x_t, t)
        variance = self.get_posterior_variance(t, x_t.shape)

        # Sample (except for t=0)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))

        return mean + nonzero_mask * torch.sqrt(variance) * noise


class DiscreteNoiseScheduler(nn.Module):
    """Noise scheduler for discrete diffusion (e.g., codon sequences).

    Implements absorbing state diffusion from D3PM paper.

    Args:
        n_steps: Number of diffusion steps
        vocab_size: Size of vocabulary (e.g., 64 for codons)
        schedule_type: Type of schedule
    """

    def __init__(
        self,
        n_steps: int = 1000,
        vocab_size: int = 64,
        schedule_type: str = "cosine",
    ):
        super().__init__()
        self.n_steps = n_steps
        self.vocab_size = vocab_size
        self.schedule_type = schedule_type

        # Absorbing state is the last token
        self.absorbing_state = vocab_size - 1

        # Compute schedule
        self._compute_schedule()

    def _compute_schedule(self):
        """Compute transition probabilities."""
        # Probability of staying in current state at each timestep
        if self.schedule_type == "linear":
            stay_probs = 1 - torch.linspace(0, 1, self.n_steps)
        elif self.schedule_type == "cosine":
            t = torch.linspace(0, 1, self.n_steps)
            stay_probs = torch.cos(t * math.pi / 2) ** 2
        else:
            stay_probs = 1 - torch.linspace(0, 1, self.n_steps)

        # Cumulative stay probability
        stay_probs_cumprod = torch.cumprod(stay_probs, dim=0)

        self.register_buffer("stay_probs", stay_probs)
        self.register_buffer("stay_probs_cumprod", stay_probs_cumprod)

    def add_noise(
        self,
        x: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Add noise to discrete tokens.

        With probability (1 - stay_prob), replace token with absorbing state.

        Args:
            x: Token indices of shape (batch, seq_len)
            t: Timesteps of shape (batch,)

        Returns:
            Noised tokens
        """
        batch_size = x.shape[0]
        stay_prob = self.stay_probs_cumprod[t].view(batch_size, 1)

        # Bernoulli mask: True = keep, False = replace with absorbing
        mask = torch.rand_like(x.float()) < stay_prob

        noised = torch.where(mask, x, torch.full_like(x, self.absorbing_state))
        return noised

    def posterior_distribution(
        self,
        x_t: Tensor,
        logits: Tensor,
        t: Tensor,
    ) -> Tensor:
        """Compute posterior p(x_{t-1} | x_t, x_0).

        Args:
            x_t: Current noised tokens
            logits: Model's logits for x_0
            t: Timesteps

        Returns:
            Posterior logits
        """
        batch_size = x_t.shape[0]

        # Get transition probabilities
        if t.min() > 0:
            stay_t = self.stay_probs_cumprod[t].view(batch_size, 1, 1)
            stay_tm1 = self.stay_probs_cumprod[t - 1].view(batch_size, 1, 1)
        else:
            # Handle t=0 case
            stay_t = self.stay_probs_cumprod[t.clamp(min=0)].view(batch_size, 1, 1)
            stay_tm1 = torch.ones(batch_size, 1, 1, device=x_t.device)

        # Probability of transitioning from absorbing to non-absorbing
        p_x0 = torch.softmax(logits, dim=-1)  # (batch, seq, vocab)

        # Posterior combines x_0 prediction with transition model
        # Simplified: weight x_0 prediction by relative stay probability
        weight = stay_tm1 / (stay_t + 1e-8)
        posterior = p_x0 * weight.clamp(max=1.0)

        return posterior
