# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""P-adic aware contrastive learning module.

This module implements contrastive learning methods that leverage p-adic
structure for intelligent positive/negative sampling and hierarchical
feature learning.

Key features:
- P-adic distance-based positive pair sampling
- Multi-scale hierarchical contrastive loss
- SimCLR-style encoder with p-adic awareness
- Momentum contrast (MoCo) integration

References:
- Chen et al. (2020): SimCLR
- He et al. (2020): MoCo
- Wang & Isola (2020): Understanding Contrastive Loss
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ContrastiveConfig:
    """Configuration for contrastive learning."""

    temperature: float = 0.07
    use_padic_sampling: bool = True
    hierarchy_levels: int = 3
    projection_dim: int = 128
    hidden_dim: int = 256
    momentum: float = 0.999
    queue_size: int = 65536


class PAdicContrastiveLoss(nn.Module):
    """Contrastive loss with p-adic positive sampling.

    Uses p-adic distance to determine positive pairs: samples that are
    p-adically close (high valuation) are treated as positives, while
    p-adically distant samples are negatives.

    This captures the intuition that hierarchically similar items
    (e.g., same gene family, same protein fold) should have similar
    representations.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        valuation_threshold: int = 2,
        use_hard_negatives: bool = True,
        prime: int = 3,
    ):
        """Initialize p-adic contrastive loss.

        Args:
            temperature: Temperature for softmax scaling
            valuation_threshold: Minimum valuation to consider positive
            use_hard_negatives: Whether to mine hard negatives
            prime: Prime base for p-adic valuation
        """
        super().__init__()
        self.temperature = temperature
        self.valuation_threshold = valuation_threshold
        self.use_hard_negatives = use_hard_negatives
        self.prime = prime
        self.max_valuation = 9  # 3^9 > 19683

    def _compute_valuation(self, diff: torch.Tensor) -> torch.Tensor:
        """Compute p-adic valuation of difference.

        v_p(n) = largest k such that p^k divides n
        """
        # Handle zero case (infinite valuation -> max_valuation)
        is_zero = diff == 0
        diff = torch.where(is_zero, torch.ones_like(diff), diff)

        valuation = torch.zeros_like(diff, dtype=torch.float32)
        remaining = diff.abs()

        for _k in range(self.max_valuation + 1):
            divisible = (remaining % self.prime) == 0
            valuation = torch.where(divisible, valuation + 1, valuation)
            remaining = torch.where(divisible, remaining // self.prime, remaining)

        # Set zero differences to max valuation
        valuation = torch.where(
            is_zero,
            torch.tensor(self.max_valuation, dtype=torch.float32, device=diff.device),
            valuation,
        )
        return valuation.clamp(max=self.max_valuation)

    def _get_positive_mask(self, indices: torch.Tensor) -> torch.Tensor:
        """Create mask for positive pairs based on p-adic distance.

        Returns:
            Boolean mask of shape (batch, batch) where True = positive pair
        """
        batch_size = indices.shape[0]
        device = indices.device

        # Compute pairwise differences
        i_expanded = indices.unsqueeze(1)  # (batch, 1)
        j_expanded = indices.unsqueeze(0)  # (1, batch)
        diff = i_expanded - j_expanded  # (batch, batch)

        # Compute valuation
        valuation = self._compute_valuation(diff)

        # Positive if valuation >= threshold (p-adically close)
        positive_mask = valuation >= self.valuation_threshold

        # Exclude self-pairs
        eye = torch.eye(batch_size, device=device, dtype=torch.bool)
        positive_mask = positive_mask & ~eye

        return positive_mask

    def forward(
        self,
        embeddings: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Compute p-adic contrastive loss.

        Args:
            embeddings: Latent representations (batch, dim)
            indices: P-adic indices for each sample (batch,)

        Returns:
            Scalar contrastive loss
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        similarity = torch.mm(embeddings, embeddings.t()) / self.temperature

        # Get positive mask
        positive_mask = self._get_positive_mask(indices)

        # If no positives, return zero loss
        n_positives = positive_mask.sum()
        if n_positives == 0:
            return torch.tensor(0.0, device=device)

        # Compute InfoNCE loss
        # For each anchor, compute log-sum-exp over all samples
        # Subtract positive similarities

        # Mask out self-similarity
        eye = torch.eye(batch_size, device=device)
        similarity = similarity - eye * 1e9

        # Log-sum-exp for denominator
        log_sum_exp = torch.logsumexp(similarity, dim=1)

        # Mean of positive similarities
        positive_sim = similarity * positive_mask.float()
        positive_counts = positive_mask.sum(dim=1).clamp(min=1)
        mean_positive = positive_sim.sum(dim=1) / positive_counts

        # Loss: -log(exp(pos) / sum(exp(all))) = log_sum_exp - pos
        loss = log_sum_exp - mean_positive

        # Average over samples with positives
        has_positive = positive_mask.any(dim=1)
        loss = loss[has_positive].mean() if has_positive.sum() > 0 else torch.tensor(0.0, device=device)

        return loss


class MultiScaleContrastive(nn.Module):
    """Hierarchical contrastive learning at multiple p-adic scales.

    Computes contrastive loss at different valuation thresholds,
    encouraging the model to learn both fine-grained and coarse-grained
    structure.
    """

    def __init__(
        self,
        n_levels: int = 3,
        base_temperature: float = 0.07,
        temperature_scale: float = 1.5,
        level_weights: list[float] | None = None,
        prime: int = 3,
    ):
        """Initialize multi-scale contrastive loss.

        Args:
            n_levels: Number of hierarchy levels
            base_temperature: Temperature for finest level
            temperature_scale: Multiply temperature by this per level
            level_weights: Optional weights for each level
            prime: Prime base for p-adic valuation
        """
        super().__init__()
        self.n_levels = n_levels
        self.base_temperature = base_temperature
        self.temperature_scale = temperature_scale
        self.prime = prime

        # Default equal weights
        if level_weights is None:
            level_weights = [1.0 / n_levels] * n_levels
        self.register_buffer(
            "level_weights",
            torch.tensor(level_weights, dtype=torch.float32),
        )

        # Create loss functions for each level
        self.losses = nn.ModuleList(
            [
                PAdicContrastiveLoss(
                    temperature=base_temperature * (temperature_scale**i),
                    valuation_threshold=i + 1,
                    prime=prime,
                )
                for i in range(n_levels)
            ]
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute multi-scale contrastive loss.

        Args:
            embeddings: Latent representations (batch, dim)
            indices: P-adic indices (batch,)

        Returns:
            Tuple of (total_loss, dict of per-level losses)
        """
        total_loss = torch.tensor(0.0, device=embeddings.device)
        level_losses = {}

        for i, (loss_fn, weight) in enumerate(zip(self.losses, self.level_weights, strict=False)):
            level_loss = loss_fn(embeddings, indices)
            level_losses[f"level_{i}"] = level_loss.item()
            total_loss = total_loss + weight * level_loss

        return total_loss, level_losses


class SimCLREncoder(nn.Module):
    """SimCLR-style encoder with projection head.

    Takes a base encoder and adds a projection head for contrastive
    learning. The projection head is discarded after training.

    Architecture:
        input -> base_encoder -> representation -> projection_head -> embedding
    """

    def __init__(
        self,
        base_encoder: nn.Module,
        representation_dim: int,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        use_bn: bool = True,
    ):
        """Initialize SimCLR encoder.

        Args:
            base_encoder: Backbone encoder network
            representation_dim: Output dimension of base encoder
            projection_dim: Output dimension of projection head
            hidden_dim: Hidden dimension in projection head
            use_bn: Whether to use batch normalization
        """
        super().__init__()
        self.base_encoder = base_encoder
        self.representation_dim = representation_dim
        self.projection_dim = projection_dim

        # Projection head: 2-layer MLP
        if use_bn:
            self.projection_head = nn.Sequential(
                nn.Linear(representation_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, projection_dim),
            )
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(representation_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, projection_dim),
            )

    def forward(
        self,
        x: torch.Tensor,
        return_representation: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor
            return_representation: If True, also return pre-projection features

        Returns:
            Projected embeddings, optionally with representations
        """
        representation = self.base_encoder(x)
        projection = self.projection_head(representation)

        if return_representation:
            return projection, representation
        return projection


class MomentumContrastEncoder(nn.Module):
    """Momentum Contrast (MoCo) encoder for contrastive learning.

    Maintains a momentum-updated key encoder and a queue of negative
    samples for efficient contrastive learning.

    Key features:
    - Momentum update of key encoder
    - Dictionary queue for negative samples
    - Supports p-adic positive sampling
    """

    def __init__(
        self,
        encoder: nn.Module,
        dim: int = 128,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.07,
    ):
        """Initialize MoCo encoder.

        Args:
            encoder: Encoder network
            dim: Embedding dimension
            queue_size: Size of negative sample queue
            momentum: Momentum for key encoder update
            temperature: Temperature for contrastive loss
        """
        super().__init__()

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        self.dim = dim

        # Query encoder
        self.encoder_q = encoder

        # Key encoder (momentum updated)
        self.encoder_k = copy.deepcopy(encoder)
        for param in self.encoder_k.parameters():
            param.requires_grad = False

        # Queue
        self.register_buffer(
            "queue",
            F.normalize(torch.randn(dim, queue_size), dim=0),
        )
        self.register_buffer(
            "queue_indices",
            torch.zeros(queue_size, dtype=torch.long),
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update(self):
        """Update key encoder with momentum."""
        for param_q, param_k in zip(
            self.encoder_q.parameters(),
            self.encoder_k.parameters(),
            strict=False,
        ):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data

    @torch.no_grad()
    def _dequeue_and_enqueue(
        self,
        keys: torch.Tensor,
        indices: torch.Tensor,
    ):
        """Update queue with new keys."""
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.queue_size:
            # Wrap around
            n_first = self.queue_size - ptr
            self.queue[:, ptr:] = keys[:n_first].T
            self.queue[:, : batch_size - n_first] = keys[n_first:].T
            self.queue_indices[ptr:] = indices[:n_first]
            self.queue_indices[: batch_size - n_first] = indices[n_first:]
        else:
            self.queue[:, ptr : ptr + batch_size] = keys.T
            self.queue_indices[ptr : ptr + batch_size] = indices

        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for MoCo.

        Args:
            x_q: Query samples
            x_k: Key samples (augmented views)
            indices: P-adic indices for samples

        Returns:
            Tuple of (logits, labels) for cross-entropy loss
        """
        # Compute query embeddings
        q = self.encoder_q(x_q)
        q = F.normalize(q, dim=1)

        # Compute key embeddings
        with torch.no_grad():
            self._momentum_update()
            k = self.encoder_k(x_k)
            k = F.normalize(k, dim=1)

        # Positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # Negative logits: NxK (from queue)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # Logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature

        # Labels: positives are at index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Update queue
        self._dequeue_and_enqueue(k, indices)

        return logits, labels


class PAdicPositiveSampler:
    """Sample positive pairs based on p-adic proximity.

    Used for data augmentation in contrastive learning:
    given an anchor, samples positives from p-adically close items.
    """

    def __init__(
        self,
        min_valuation: int = 2,
        max_valuation: int = 9,
        prime: int = 3,
    ):
        """Initialize sampler.

        Args:
            min_valuation: Minimum valuation for positive pair
            max_valuation: Maximum valuation to consider
            prime: Prime base
        """
        self.min_valuation = min_valuation
        self.max_valuation = max_valuation
        self.prime = prime

    def get_positive_candidates(
        self,
        anchor_idx: int,
        all_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Get indices of positive candidates for anchor.

        Args:
            anchor_idx: Index of anchor sample
            all_indices: All available p-adic indices

        Returns:
            Boolean mask of positive candidates
        """
        device = all_indices.device
        anchor_val = all_indices[anchor_idx]

        # Compute valuations
        diff = all_indices - anchor_val

        # Valuation computation
        is_zero = diff == 0
        abs_diff = diff.abs()

        valuation = torch.zeros_like(diff, dtype=torch.float32)
        remaining = abs_diff

        for _k in range(self.max_valuation + 1):
            divisible = (remaining % self.prime) == 0
            valuation = torch.where(divisible & ~is_zero, valuation + 1, valuation)
            remaining = torch.where(divisible, remaining // self.prime, remaining)

        valuation = torch.where(
            is_zero,
            torch.tensor(self.max_valuation, dtype=torch.float32, device=device),
            valuation,
        )

        # Create mask
        mask = (valuation >= self.min_valuation) & (torch.arange(len(all_indices), device=device) != anchor_idx)

        return mask

    def sample_positive(
        self,
        anchor_idx: int,
        all_indices: torch.Tensor,
    ) -> int | None:
        """Sample a single positive for anchor.

        Args:
            anchor_idx: Index of anchor sample
            all_indices: All available p-adic indices

        Returns:
            Index of sampled positive, or None if no positives available
        """
        candidates = self.get_positive_candidates(anchor_idx, all_indices)
        candidate_indices = torch.where(candidates)[0]

        if len(candidate_indices) == 0:
            return None

        # Random selection
        rand_idx = torch.randint(len(candidate_indices), (1,)).item()
        return int(candidate_indices[rand_idx])


class ContrastiveDataAugmentation:
    """Data augmentation strategies for contrastive learning.

    Provides various augmentation methods for biological sequences
    and structures.
    """

    def __init__(
        self,
        noise_scale: float = 0.1,
        mask_prob: float = 0.15,
        shuffle_window: int = 3,
    ):
        """Initialize augmentation.

        Args:
            noise_scale: Scale of Gaussian noise
            mask_prob: Probability of masking tokens
            shuffle_window: Window size for local shuffling
        """
        self.noise_scale = noise_scale
        self.mask_prob = mask_prob
        self.shuffle_window = shuffle_window
        self.training = True

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to input."""
        noise = torch.randn_like(x) * self.noise_scale
        return x + noise

    def random_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly mask positions in input."""
        mask = torch.rand_like(x) > self.mask_prob
        return x * mask.float()

    def dropout(self, x: torch.Tensor, p: float = 0.1) -> torch.Tensor:
        """Apply dropout augmentation."""
        if self.training:
            mask = torch.rand_like(x) > p
            return x * mask.float() / (1 - p)
        return x

    def __call__(
        self,
        x: torch.Tensor,
        augmentation: str = "noise",
    ) -> torch.Tensor:
        """Apply specified augmentation.

        Args:
            x: Input tensor
            augmentation: Type of augmentation

        Returns:
            Augmented tensor
        """
        if augmentation == "noise":
            return self.add_noise(x)
        elif augmentation == "mask":
            return self.random_mask(x)
        elif augmentation == "dropout":
            return self.dropout(x)
        else:
            return x
