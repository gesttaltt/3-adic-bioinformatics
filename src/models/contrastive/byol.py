# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""BYOL: Bootstrap Your Own Latent.

Self-supervised learning without negative samples using
a momentum encoder and predictor network.

References:
    - Grill et al. (2020): Bootstrap Your Own Latent: A New Approach
      to Self-Supervised Learning
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.contrastive.augmentations import SequenceAugmentations


@dataclass
class BYOLConfig:
    """Configuration for BYOL.

    Attributes:
        embed_dim: Encoder output dimension
        proj_dim: Projection head output dimension
        hidden_dim: Projection head hidden dimension
        momentum: EMA momentum for target encoder
        use_predictor: Use predictor network (True for BYOL)
        symmetric: Use symmetric loss (both directions)
    """

    embed_dim: int = 256
    proj_dim: int = 128
    hidden_dim: int = 512
    momentum: float = 0.996
    use_predictor: bool = True
    symmetric: bool = True


class MLP(nn.Module):
    """Simple MLP for projection/prediction heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        use_bn: bool = True,
    ):
        """Initialize MLP.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            num_layers: Number of layers
            use_bn: Use batch normalization
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for _i in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)


class MomentumEncoder(nn.Module):
    """Momentum-updated target encoder.

    Maintains an exponential moving average of the online encoder.
    """

    def __init__(
        self,
        encoder: nn.Module,
        momentum: float = 0.996,
    ):
        """Initialize momentum encoder.

        Args:
            encoder: Online encoder to copy
            momentum: EMA momentum coefficient
        """
        super().__init__()
        self.momentum = momentum

        # Create target encoder as copy
        self.target_encoder = copy.deepcopy(encoder)

        # Freeze target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, online_encoder: nn.Module):
        """Update target encoder with EMA.

        Args:
            online_encoder: Current online encoder
        """
        for online_params, target_params in zip(
            online_encoder.parameters(),
            self.target_encoder.parameters(),
            strict=False,
        ):
            target_params.data = self.momentum * target_params.data + (1 - self.momentum) * online_params.data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with target encoder."""
        with torch.no_grad():
            return self.target_encoder(x)


class BYOL(nn.Module):
    """Bootstrap Your Own Latent.

    Self-supervised learning method that learns representations
    by predicting target network outputs without negative samples.

    Example:
        >>> encoder = nn.Sequential(nn.Linear(100, 256), nn.ReLU())
        >>> byol = BYOL(encoder)
        >>> # Training loop
        >>> for batch in dataloader:
        ...     loss = byol.compute_loss(batch)
        ...     loss.backward()
        ...     optimizer.step()
        ...     byol.update_target()
    """

    def __init__(
        self,
        encoder: nn.Module,
        config: BYOLConfig | None = None,
        augmentation: SequenceAugmentations | None = None,
    ):
        """Initialize BYOL.

        Args:
            encoder: Base encoder network
            config: BYOL configuration
            augmentation: Augmentation pipeline
        """
        super().__init__()
        self.config = config or BYOLConfig()

        # Online encoder
        self.encoder = encoder

        # Target encoder (momentum updated)
        self.momentum_encoder = MomentumEncoder(
            encoder,
            momentum=self.config.momentum,
        )

        # Projection head (online)
        self.projector = MLP(
            input_dim=self.config.embed_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.proj_dim,
        )

        # Target projection head
        self.target_projector = copy.deepcopy(self.projector)
        for param in self.target_projector.parameters():
            param.requires_grad = False

        # Predictor (only for online network)
        if self.config.use_predictor:
            self.predictor = MLP(
                input_dim=self.config.proj_dim,
                hidden_dim=self.config.hidden_dim,
                output_dim=self.config.proj_dim,
            )
        else:
            self.predictor = nn.Identity()

        # Augmentation
        self.augmentation = augmentation or SequenceAugmentations.default_for_byol()

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning online and target representations.

        Args:
            x: Input tensor

        Returns:
            Tuple of (online_prediction, target_projection)
        """
        # Online path: encoder -> projector -> predictor
        online_repr = self.encoder(x)
        online_proj = self.projector(online_repr)
        online_pred = self.predictor(online_proj)

        # Target path: encoder -> projector (no predictor)
        with torch.no_grad():
            target_repr = self.momentum_encoder(x)
            target_proj = self.target_projector(target_repr)

        return online_pred, target_proj

    def compute_loss(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute BYOL loss.

        Args:
            x1: First view (or batch if x2 not provided)
            x2: Second view (optional, will augment x1 if not provided)

        Returns:
            BYOL loss
        """
        if x2 is None:
            # Create augmented views
            # (In practice, augmentation is often done in dataloader)
            x2 = x1

        # Forward both views through online network
        online_pred_1, target_proj_1 = self.forward(x1)
        online_pred_2, target_proj_2 = self.forward(x2)

        # Cosine similarity loss
        loss_1 = self._cosine_loss(online_pred_1, target_proj_2.detach())

        if self.config.symmetric:
            loss_2 = self._cosine_loss(online_pred_2, target_proj_1.detach())
            loss = (loss_1 + loss_2) / 2
        else:
            loss = loss_1

        return loss

    def _cosine_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative cosine similarity loss.

        Args:
            pred: Prediction from online network
            target: Target from target network

        Returns:
            Loss value
        """
        pred = F.normalize(pred, dim=-1, p=2)
        target = F.normalize(target, dim=-1, p=2)

        # Negative cosine similarity
        loss = 2 - 2 * (pred * target).sum(dim=-1)

        return loss.mean()

    @torch.no_grad()
    def update_target(self):
        """Update target network with EMA."""
        # Update encoder
        self.momentum_encoder.update(self.encoder)

        # Update projector
        for online_params, target_params in zip(
            self.projector.parameters(),
            self.target_projector.parameters(),
            strict=False,
        ):
            target_params.data = (
                self.config.momentum * target_params.data + (1 - self.config.momentum) * online_params.data
            )

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get learned representation (for downstream tasks).

        Args:
            x: Input tensor

        Returns:
            Representation from encoder
        """
        return self.encoder(x)


class BYOLTrainer:
    """Training loop for BYOL."""

    def __init__(
        self,
        model: BYOL,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        device: str = "cuda",
    ):
        """Initialize trainer.

        Args:
            model: BYOL model
            optimizer: Optimizer for online network
            scheduler: Learning rate scheduler
            device: Computation device
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        augmentation: Callable | None = None,
    ) -> float:
        """Train for one epoch.

        Args:
            dataloader: Training data loader
            augmentation: Optional augmentation function

        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss = 0
        n_batches = 0

        for batch in dataloader:
            x = batch[0] if isinstance(batch, (tuple, list)) else batch

            x = x.to(self.device)

            # Create augmented views
            if augmentation is not None:
                x1 = augmentation(x)
                x2 = augmentation(x)
            else:
                x1 = x
                x2 = x

            # Compute loss
            loss = self.model.compute_loss(x1, x2)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update target network
            self.model.update_target()

            total_loss += loss.item()
            n_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / n_batches


class BYOL2(nn.Module):
    """BYOL v2 with additional improvements.

    Adds:
    - Multi-crop strategy
    - Stronger augmentations
    - Improved momentum schedule
    """

    def __init__(
        self,
        encoder: nn.Module,
        config: BYOLConfig | None = None,
        n_global_views: int = 2,
        n_local_views: int = 4,
    ):
        """Initialize BYOL v2.

        Args:
            encoder: Base encoder
            config: Configuration
            n_global_views: Number of global (large) crops
            n_local_views: Number of local (small) crops
        """
        super().__init__()
        self.config = config or BYOLConfig()
        self.n_global_views = n_global_views
        self.n_local_views = n_local_views

        # Online and target encoders
        self.encoder = encoder
        self.momentum_encoder = MomentumEncoder(encoder, self.config.momentum)

        # Projection heads
        self.projector = MLP(
            self.config.embed_dim,
            self.config.hidden_dim,
            self.config.proj_dim,
        )
        self.target_projector = copy.deepcopy(self.projector)

        # Predictor
        self.predictor = MLP(
            self.config.proj_dim,
            self.config.hidden_dim,
            self.config.proj_dim,
        )

        # Momentum schedule
        self.base_momentum = self.config.momentum
        self.current_step = 0
        self.total_steps = 100000  # Will be set during training

    def compute_loss(
        self,
        global_views: list[torch.Tensor],
        local_views: list[torch.Tensor],
    ) -> torch.Tensor:
        """Compute multi-crop BYOL loss.

        Args:
            global_views: List of global view tensors
            local_views: List of local view tensors

        Returns:
            Total loss
        """
        # Encode all views through online network
        online_global = [self.predictor(self.projector(self.encoder(v))) for v in global_views]

        # Encode global views through target network
        with torch.no_grad():
            target_global = [self.target_projector(self.momentum_encoder(v)) for v in global_views]

        # All views predict global targets
        total_loss = 0
        n_pairs = 0

        # Global-to-global
        for i, pred in enumerate(online_global):
            for j, target in enumerate(target_global):
                if i != j:
                    total_loss += self._cosine_loss(pred, target)
                    n_pairs += 1

        # Local-to-global
        for local_v in local_views:
            online_local = self.predictor(self.projector(self.encoder(local_v)))
            for target in target_global:
                total_loss += self._cosine_loss(online_local, target)
                n_pairs += 1

        return total_loss / n_pairs

    def _cosine_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Negative cosine similarity."""
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        return 2 - 2 * (pred * target).sum(dim=-1).mean()

    @torch.no_grad()
    def update_target(self):
        """Update target with cosine momentum schedule."""
        # Cosine momentum schedule
        momentum = (
            1
            - (1 - self.base_momentum)
            * (1 + torch.cos(torch.tensor(torch.pi * self.current_step / self.total_steps)))
            / 2
        )

        self.momentum_encoder.momentum = momentum.item()
        self.momentum_encoder.update(self.encoder)

        # Update projector
        for o, t in zip(self.projector.parameters(), self.target_projector.parameters(), strict=False):
            t.data = momentum * t.data + (1 - momentum) * o.data

        self.current_step += 1
