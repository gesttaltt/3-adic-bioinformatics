# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""D3PM: Discrete Denoising Diffusion Probabilistic Models.

Implements discrete diffusion for categorical data like protein sequences.

References:
    - Austin et al. (2021): Structured Denoising Diffusion Models
      in Discrete State-Spaces
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.diffusion.noise_schedule import (
    CosineSchedule,
    PAdicNoiseSchedule,
)


@dataclass
class D3PMConfig:
    """Configuration for D3PM model.

    Attributes:
        vocab_size: Size of discrete vocabulary (21 for amino acids + gap)
        max_length: Maximum sequence length
        hidden_dim: Hidden dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        n_timesteps: Number of diffusion timesteps
        noise_schedule: Type of noise schedule
        transition_type: Transition matrix type ('uniform', 'absorbing', 'padic')
    """

    vocab_size: int = 21  # 20 amino acids + gap/mask
    max_length: int = 512
    hidden_dim: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_timesteps: int = 1000
    noise_schedule: str = "cosine"
    transition_type: str = "absorbing"
    dropout: float = 0.1


class D3PM(nn.Module):
    """Discrete Denoising Diffusion Probabilistic Model.

    Learns to generate discrete sequences by reversing a
    corruption process that transitions between categorical states.

    Example:
        >>> config = D3PMConfig(vocab_size=21, max_length=100)
        >>> model = D3PM(config)
        >>> # Generate sequences
        >>> samples = model.sample(batch_size=16)
    """

    # Special tokens
    MASK_TOKEN = 0  # Absorbing state

    def __init__(
        self,
        config: D3PMConfig | None = None,
    ):
        """Initialize D3PM.

        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config or D3PMConfig()

        # Noise schedule
        self._init_noise_schedule()

        # Transition matrices
        self._init_transition_matrices()

        # Denoising network
        self._build_denoising_network()

    def _init_noise_schedule(self):
        """Initialize noise schedule based on config."""
        if self.config.noise_schedule == "cosine":
            self.schedule = CosineSchedule(self.config.n_timesteps)
        elif self.config.noise_schedule == "padic":
            self.schedule = PAdicNoiseSchedule(
                self.config.n_timesteps,
                p=3,
                max_length=self.config.max_length,
            )
        else:
            from src.models.diffusion.noise_schedule import LinearSchedule

            self.schedule = LinearSchedule(self.config.n_timesteps)

    def _init_transition_matrices(self):
        """Initialize forward transition matrices."""
        K = self.config.vocab_size

        if self.config.transition_type == "uniform":
            # Uniform transition: equal probability to all states
            Q = torch.ones(K, K) / K
        elif self.config.transition_type == "absorbing":
            # Absorbing state transition: transition to mask token
            Q = torch.zeros(K, K)
            Q[:, self.MASK_TOKEN] = 1.0
        else:
            # Identity with small uniform noise
            Q = torch.eye(K) * 0.9 + torch.ones(K, K) * 0.1 / K

        self.register_buffer("Q", Q)

    def _build_denoising_network(self):
        """Build the denoising neural network."""
        # Token embedding
        self.token_embedding = nn.Embedding(
            self.config.vocab_size,
            self.config.hidden_dim,
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.config.max_length, self.config.hidden_dim) * 0.02)

        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim * 4, self.config.hidden_dim),
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.config.hidden_dim,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.hidden_dim * 4,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.n_layers,
        )

        # Output projection to logits
        self.output_proj = nn.Linear(
            self.config.hidden_dim,
            self.config.vocab_size,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict clean token logits from noisy input.

        Args:
            x_t: Noisy tokens (batch, seq_len)
            t: Timesteps (batch,)
            mask: Attention mask (optional)

        Returns:
            Logits for clean tokens (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = x_t.shape

        # Embed tokens
        h = self.token_embedding(x_t)

        # Add positional encoding
        h = h + self.pos_encoding[:, :seq_len]

        # Add time embedding
        t_emb = self.time_embedding(t)  # (batch, hidden_dim)
        h = h + t_emb.unsqueeze(1)

        # Create attention mask if provided
        attn_mask = ~mask if mask is not None else None

        # Transformer
        h = self.transformer(h, src_key_padding_mask=attn_mask)

        # Project to logits
        logits = self.output_proj(h)

        return logits

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample from forward diffusion process q(x_t | x_0).

        Args:
            x_0: Clean tokens (batch, seq_len)
            t: Timesteps (batch,)

        Returns:
            Tuple of (noisy_tokens, transition_probs)
        """
        batch_size, seq_len = x_0.shape

        # Get cumulative alpha
        alpha_cumprod = self.schedule.get_cumulative(t)  # (batch,)

        # Transition probability matrix at time t
        # Q_t = alpha_t * I + (1 - alpha_t) * Q
        Q = self.Q.to(x_0.device)
        eye = torch.eye(self.config.vocab_size, device=x_0.device)

        Q_t = alpha_cumprod.view(-1, 1, 1) * eye + (1 - alpha_cumprod.view(-1, 1, 1)) * Q

        # Sample from categorical distribution
        # Get transition probs for each token
        x_0_onehot = F.one_hot(x_0, self.config.vocab_size).float()  # (batch, seq, K)

        # p(x_t | x_0) = x_0 @ Q_t
        probs = torch.einsum("bsk,bkj->bsj", x_0_onehot, Q_t)

        # Sample
        x_t = torch.multinomial(
            probs.view(-1, self.config.vocab_size),
            num_samples=1,
        ).view(batch_size, seq_len)

        return x_t, probs

    def compute_loss(
        self,
        x_0: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute D3PM training loss.

        Args:
            x_0: Clean tokens (batch, seq_len)
            mask: Attention mask (optional)

        Returns:
            Loss value
        """
        batch_size, seq_len = x_0.shape
        device = x_0.device

        # Sample random timesteps
        t = torch.randint(
            0,
            self.config.n_timesteps,
            (batch_size,),
            device=device,
        )

        # Forward diffusion
        x_t, _ = self.q_sample(x_0, t)

        # Predict clean tokens
        logits = self.forward(x_t, t, mask)

        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.view(-1, self.config.vocab_size),
            x_0.view(-1),
            reduction="none",
        ).view(batch_size, seq_len)

        # Apply mask if provided
        if mask is not None:
            loss = loss * mask.float()
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        seq_len: int | None = None,
        temperature: float = 1.0,
        guidance_scale: float = 1.0,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate samples via reverse diffusion.

        Args:
            batch_size: Number of samples
            seq_len: Sequence length (uses max_length if not provided)
            temperature: Sampling temperature
            guidance_scale: Classifier-free guidance scale
            condition: Optional conditioning tensor

        Returns:
            Generated tokens (batch_size, seq_len)
        """
        device = next(self.parameters()).device
        seq_len = seq_len or self.config.max_length

        # Start from noise (mask tokens for absorbing)
        if self.config.transition_type == "absorbing":
            x = torch.full(
                (batch_size, seq_len),
                self.MASK_TOKEN,
                dtype=torch.long,
                device=device,
            )
        else:
            # Random tokens
            x = torch.randint(
                0,
                self.config.vocab_size,
                (batch_size, seq_len),
                device=device,
            )

        # Reverse diffusion
        for t in reversed(range(self.config.n_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Predict clean tokens
            logits = self.forward(x, t_batch)

            # Apply temperature
            logits = logits / temperature

            # Sample
            if t > 0:
                # Get predicted probabilities
                probs = F.softmax(logits, dim=-1)

                # Re-noise according to q(x_{t-1} | x_t, x_0)
                x = self._posterior_sample(x, probs, t, t - 1)
            else:
                # Final step: argmax
                x = logits.argmax(dim=-1)

        return x

    def _posterior_sample(
        self,
        x_t: torch.Tensor,
        x_0_probs: torch.Tensor,
        t: int,
        t_prev: int,
    ) -> torch.Tensor:
        """Sample from posterior q(x_{t-1} | x_t, x_0).

        Args:
            x_t: Current tokens
            x_0_probs: Predicted clean token probabilities
            t: Current timestep
            t_prev: Previous timestep

        Returns:
            Sampled x_{t-1}
        """
        device = x_t.device
        batch_size, seq_len = x_t.shape

        # Get transition matrices
        t_tensor = torch.tensor([t], device=device)
        t_prev_tensor = torch.tensor([t_prev], device=device)

        alpha_t = self.schedule.get_cumulative(t_tensor)
        alpha_t_prev = self.schedule.get_cumulative(t_prev_tensor)

        # Approximate posterior
        # p(x_{t-1} | x_t, x_0) ∝ q(x_t | x_{t-1}) q(x_{t-1} | x_0)

        # Sample x_0 from predicted distribution
        x_0_sample = torch.multinomial(
            x_0_probs.view(-1, self.config.vocab_size),
            num_samples=1,
        ).view(batch_size, seq_len)

        # With probability (alpha_t_prev - alpha_t) / (1 - alpha_t),
        # set x_{t-1} = x_0
        # Otherwise keep x_t

        prob_x0 = ((alpha_t_prev - alpha_t) / (1 - alpha_t + 1e-8)).item()
        prob_x0 = min(max(prob_x0, 0), 1)

        mask = torch.bernoulli(torch.full((batch_size, seq_len), prob_x0, device=device)).bool()

        x_t_prev = torch.where(mask, x_0_sample, x_t)

        return x_t_prev


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for time encoding."""

    def __init__(self, dim: int):
        """Initialize embeddings.

        Args:
            dim: Embedding dimension
        """
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute embeddings.

        Args:
            t: Timesteps (batch,)

        Returns:
            Embeddings (batch, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t.unsqueeze(-1) * embeddings.unsqueeze(0)
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)

        return embeddings


class ConditionalD3PM(D3PM):
    """D3PM with conditioning support.

    Extends D3PM to support conditional generation based on
    embeddings (e.g., from VAE or property predictor).
    """

    def __init__(
        self,
        config: D3PMConfig | None = None,
        condition_dim: int = 64,
    ):
        """Initialize conditional D3PM.

        Args:
            config: Model configuration
            condition_dim: Dimension of conditioning vector
        """
        super().__init__(config)
        self.condition_dim = condition_dim

        # Condition projection
        self.condition_proj = nn.Sequential(
            nn.Linear(condition_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )

        # Null condition for classifier-free guidance
        self.null_condition = nn.Parameter(torch.zeros(condition_dim))

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with optional conditioning.

        Args:
            x_t: Noisy tokens
            t: Timesteps
            condition: Conditioning tensor (batch, condition_dim)
            mask: Attention mask

        Returns:
            Logits
        """
        batch_size, seq_len = x_t.shape

        # Embed tokens
        h = self.token_embedding(x_t)
        h = h + self.pos_encoding[:, :seq_len]

        # Add time embedding
        t_emb = self.time_embedding(t)
        h = h + t_emb.unsqueeze(1)

        # Add condition embedding
        if condition is not None:
            c_emb = self.condition_proj(condition)
            h = h + c_emb.unsqueeze(1)

        # Transformer
        attn_mask = ~mask if mask is not None else None

        h = self.transformer(h, src_key_padding_mask=attn_mask)
        logits = self.output_proj(h)

        return logits

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        seq_len: int | None = None,
        condition: torch.Tensor | None = None,
        guidance_scale: float = 2.0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate with classifier-free guidance.

        Args:
            batch_size: Number of samples
            seq_len: Sequence length
            condition: Conditioning tensor
            guidance_scale: CFG scale (1 = no guidance)
            temperature: Sampling temperature

        Returns:
            Generated tokens
        """
        device = next(self.parameters()).device
        seq_len = seq_len or self.config.max_length

        # Start from noise
        if self.config.transition_type == "absorbing":
            x = torch.full((batch_size, seq_len), self.MASK_TOKEN, dtype=torch.long, device=device)
        else:
            x = torch.randint(0, self.config.vocab_size, (batch_size, seq_len), device=device)

        # Null condition for CFG
        null_cond = self.null_condition.unsqueeze(0).expand(batch_size, -1)

        # Reverse diffusion
        for t in reversed(range(self.config.n_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            if condition is not None and guidance_scale != 1.0:
                # Classifier-free guidance
                logits_cond = self.forward(x, t_batch, condition)
                logits_uncond = self.forward(x, t_batch, null_cond)

                # Guided logits
                logits = logits_uncond + guidance_scale * (logits_cond - logits_uncond)
            else:
                logits = self.forward(x, t_batch, condition)

            # Apply temperature
            logits = logits / temperature

            # Sample
            if t > 0:
                probs = F.softmax(logits, dim=-1)
                x = self._posterior_sample(x, probs, t, t - 1)
            else:
                x = logits.argmax(dim=-1)

        return x
