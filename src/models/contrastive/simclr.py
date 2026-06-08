# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""SimCLR: Simple Contrastive Learning of Representations.

Contrastive learning with InfoNCE loss using in-batch negatives.

References:
    - Chen et al. (2020): A Simple Framework for Contrastive Learning
      of Visual Representations
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.contrastive.augmentations import SequenceAugmentations


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss.

    Also known as InfoNCE loss for contrastive learning.
    """

    def __init__(
        self,
        temperature: float = 0.5,
        reduction: str = "mean",
    ):
        """Initialize NT-Xent loss.

        Args:
            temperature: Temperature scaling factor
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        z_i: torch.Tensor,
        z_j: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NT-Xent loss.

        Args:
            z_i: First view embeddings (batch, embed_dim)
            z_j: Second view embeddings (batch, embed_dim)

        Returns:
            Loss value
        """
        batch_size = z_i.shape[0]
        device = z_i.device

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)

        # Concatenate views
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch, embed_dim)

        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # (2*batch, 2*batch)

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+batch) and (i+batch, i)
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool, device=device)
        pos_mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = True
        pos_mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = True

        # InfoNCE loss
        # log(exp(sim_pos) / sum(exp(sim_neg)))
        log_prob = F.log_softmax(sim, dim=1)

        # Extract positive log probabilities
        pos_log_prob = log_prob[pos_mask].view(2 * batch_size, 1)

        loss = -pos_log_prob.mean()

        return loss


class SimCLR(nn.Module):
    """Simple Contrastive Learning of Representations.

    Learns representations by maximizing agreement between
    differently augmented views of the same sample.

    Example:
        >>> encoder = nn.Sequential(nn.Linear(100, 256), nn.ReLU())
        >>> simclr = SimCLR(encoder)
        >>> loss = simclr.compute_loss(x1, x2)
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 256,
        proj_dim: int = 128,
        hidden_dim: int = 256,
        temperature: float = 0.5,
        augmentation: SequenceAugmentations | None = None,
    ):
        """Initialize SimCLR.

        Args:
            encoder: Base encoder network
            embed_dim: Encoder output dimension
            proj_dim: Projection output dimension
            hidden_dim: Projection hidden dimension
            temperature: NT-Xent temperature
            augmentation: Augmentation pipeline
        """
        super().__init__()

        self.encoder = encoder
        self.temperature = temperature

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

        # Loss
        self.criterion = NTXentLoss(temperature=temperature)

        # Augmentation
        self.augmentation = augmentation or SequenceAugmentations.default_for_simclr()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to projection space.

        Args:
            x: Input tensor

        Returns:
            Projected embeddings
        """
        h = self.encoder(x)
        z = self.projector(h)
        return z

    def compute_loss(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SimCLR contrastive loss.

        Args:
            x1: First augmented view
            x2: Second augmented view

        Returns:
            NT-Xent loss
        """
        z1 = self.forward(x1)
        z2 = self.forward(x2)

        return self.criterion(z1, z2)

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoder representation (for downstream).

        Args:
            x: Input tensor

        Returns:
            Encoder output (before projection)
        """
        return self.encoder(x)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    Extends contrastive loss to use label information
    for defining positive pairs.

    References:
        - Khosla et al. (2020): Supervised Contrastive Learning
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
    ):
        """Initialize supervised contrastive loss.

        Args:
            temperature: Temperature for scaling
            base_temperature: Base temperature for normalization
        """
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute supervised contrastive loss.

        Args:
            features: Feature embeddings (batch, n_views, embed_dim)
            labels: Class labels (batch,)

        Returns:
            Loss value
        """
        device = features.device
        batch_size = features.shape[0]

        # Flatten views
        if features.dim() == 3:
            n_views = features.shape[1]
            features = features.reshape(-1, features.shape[-1])
            labels = labels.repeat(n_views)
        else:
            n_views = 1

        # Normalize features
        features = F.normalize(features, dim=-1)

        # Compute similarity
        sim = torch.mm(features, features.t()) / self.temperature

        # Create positive mask from labels
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.t()).float().to(device)

        # Mask out self-contrast
        n_samples = features.shape[0]
        self_mask = torch.eye(n_samples, dtype=torch.bool, device=device)
        pos_mask = pos_mask.masked_fill(self_mask, 0)

        # Compute log softmax
        sim.masked_fill_(self_mask, float("-inf"))
        log_prob = F.log_softmax(sim, dim=1)

        # Compute mean of positive log-likelihoods
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + 1e-8)

        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(n_views, batch_size).mean()

        return loss


class MoCo(nn.Module):
    """Momentum Contrast.

    Maintains a queue of negative samples for contrastive learning.

    References:
        - He et al. (2020): Momentum Contrast for Unsupervised Visual
          Representation Learning
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int = 256,
        queue_size: int = 65536,
        momentum: float = 0.999,
        temperature: float = 0.07,
    ):
        """Initialize MoCo.

        Args:
            encoder: Base encoder
            embed_dim: Embedding dimension
            queue_size: Size of negative queue
            momentum: Momentum for key encoder
            temperature: Temperature for InfoNCE
        """
        super().__init__()

        self.encoder_q = encoder  # Query encoder
        self.encoder_k = encoder.__class__(*encoder.init_args)  # Key encoder

        # Initialize key encoder from query
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters(), strict=False):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.momentum = momentum
        self.temperature = temperature

        # Queue
        self.register_buffer("queue", F.normalize(torch.randn(embed_dim, queue_size), dim=0))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.queue_size = queue_size

    @torch.no_grad()
    def _momentum_update(self):
        """Update key encoder with momentum."""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters(), strict=False):
            param_k.data = self.momentum * param_k.data + (1 - self.momentum) * param_q.data

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        """Update queue with new keys."""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Replace oldest keys
        if ptr + batch_size > self.queue_size:
            self.queue[:, ptr:] = keys.t()[:, : self.queue_size - ptr]
            remaining = batch_size - (self.queue_size - ptr)
            self.queue[:, :remaining] = keys.t()[:, self.queue_size - ptr :]
            self.queue_ptr[0] = remaining
        else:
            self.queue[:, ptr : ptr + batch_size] = keys.t()
            self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute query and key embeddings.

        Args:
            x_q: Query input
            x_k: Key input

        Returns:
            Tuple of (query_embeddings, key_embeddings)
        """
        # Query embeddings
        q = F.normalize(self.encoder_q(x_q), dim=-1)

        # Key embeddings (no gradient)
        with torch.no_grad():
            self._momentum_update()
            k = F.normalize(self.encoder_k(x_k), dim=-1)

        return q, k

    def compute_loss(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MoCo contrastive loss.

        Args:
            x_q: Query view
            x_k: Key view

        Returns:
            InfoNCE loss
        """
        q, k = self.forward(x_q, x_k)

        # Positive logits
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # Negative logits from queue
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # Logits
        logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature

        # Labels: positive is index 0
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Cross entropy loss
        loss = F.cross_entropy(logits, labels)

        # Update queue
        self._dequeue_and_enqueue(k)

        return loss
