# Copyright 2024-2026 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""DDG Ensemble combining fuzzy VAE and transformer predictions.

This module implements the final ensemble that combines:
1. FuzzyDDGHead: VAE-based predictions with uncertainty quantification
2. DDGTransformer: Full-sequence transformer predictions
3. HierarchicalTransformer: Two-level attention predictions

The ensemble uses learned weights to combine predictions and
provides both point estimates and uncertainty quantification.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


@dataclass
class EnsembleConfig:
    """Configuration for DDG ensemble."""

    # Component dimensions
    fused_latent_dim: int = 128
    transformer_d_model: int = 128

    # Fuzzy head
    fuzzy_hidden_dim: int = 64
    min_sigma: float = 0.01  # Minimum uncertainty

    # Ensemble
    n_components: int = 3  # fuzzy + full_seq + hierarchical
    learn_weights: bool = True
    initial_weights: list[float] = None  # Default: equal weights

    def __post_init__(self):
        if self.initial_weights is None:
            self.initial_weights = [1.0 / self.n_components] * self.n_components


class FuzzyDDGHead(nn.Module):
    """Fuzzy DDG prediction head with uncertainty quantification.

    Returns a distribution over DDG values instead of point estimates,
    enabling principled uncertainty quantification.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        min_sigma: float = 0.01,
    ):
        """Initialize fuzzy head.

        Args:
            input_dim: Input dimension (fused latent)
            hidden_dim: Hidden layer dimension
            min_sigma: Minimum uncertainty (prevents collapse)
        """
        super().__init__()
        self.min_sigma = min_sigma

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Separate heads for mean and uncertainty
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensure positive
        )

    def forward(self, z_fused: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            z_fused: Fused latent representation (batch, input_dim)

        Returns:
            Dictionary with 'mu', 'sigma', 'distribution'
        """
        h = self.shared(z_fused)
        mu = self.mu_head(h)
        sigma = self.sigma_head(h) + self.min_sigma

        return {
            "mu": mu,
            "sigma": sigma,
            "distribution": Normal(mu, sigma),
        }

    def sample(
        self,
        z_fused: torch.Tensor,
        n_samples: int = 100,
    ) -> torch.Tensor:
        """Sample from the predictive distribution.

        Args:
            z_fused: Fused latent representation
            n_samples: Number of samples

        Returns:
            Samples (batch, n_samples)
        """
        output = self.forward(z_fused)
        dist = output["distribution"]
        samples = dist.sample((n_samples,))  # (n_samples, batch, 1)
        return samples.squeeze(-1).permute(1, 0)  # (batch, n_samples)

    def nll_loss(
        self,
        z_fused: torch.Tensor,
        y: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Negative log-likelihood loss.

        Args:
            z_fused: Fused latent
            y: Target DDG values
            reduction: Loss reduction

        Returns:
            NLL loss
        """
        output = self.forward(z_fused)
        dist = output["distribution"]

        if y.dim() == 1:
            y = y.unsqueeze(-1)

        nll = -dist.log_prob(y)

        if reduction == "mean":
            return nll.mean()
        elif reduction == "sum":
            return nll.sum()
        return nll


class DDGEnsemble(nn.Module):
    """Ensemble of fuzzy VAE and transformer models.

    Combines predictions from:
    1. FuzzyDDGHead (VAE-based with uncertainty)
    2. DDGTransformer (full-sequence)
    3. HierarchicalTransformer (local + global)

    Uses learned or fixed weights to combine predictions.
    """

    def __init__(
        self,
        fuzzy_head: FuzzyDDGHead,
        full_seq_transformer: nn.Module | None = None,
        hierarchical_transformer: nn.Module | None = None,
        config: EnsembleConfig | None = None,
    ):
        """Initialize ensemble.

        Args:
            fuzzy_head: Fuzzy DDG prediction head
            full_seq_transformer: Full-sequence transformer (optional)
            hierarchical_transformer: Hierarchical transformer (optional)
            config: EnsembleConfig
        """
        super().__init__()

        if config is None:
            config = EnsembleConfig()

        self.config = config

        # Store components
        self.fuzzy = fuzzy_head
        self.full_seq = full_seq_transformer
        self.hierarchical = hierarchical_transformer

        # Count active components
        self.n_active = 1  # fuzzy always active
        if full_seq_transformer is not None:
            self.n_active += 1
        if hierarchical_transformer is not None:
            self.n_active += 1

        # Learnable ensemble weights
        if config.learn_weights:
            self.weights = nn.Parameter(torch.tensor(config.initial_weights[: self.n_active]))
        else:
            self.register_buffer("weights", torch.tensor(config.initial_weights[: self.n_active]))

    def get_normalized_weights(self) -> torch.Tensor:
        """Get softmax-normalized ensemble weights."""
        return F.softmax(self.weights, dim=0)

    def forward(
        self,
        z_fused: torch.Tensor,
        sequence: torch.Tensor | None = None,
        mutation_pos: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass through ensemble.

        Args:
            z_fused: Fused latent from multimodal VAE (batch, fused_dim)
            sequence: Token indices for transformers (batch, seq_len)
            mutation_pos: Mutation positions (batch,)
            padding_mask: Sequence padding mask

        Returns:
            Dictionary with predictions and component outputs
        """
        weights = self.get_normalized_weights()
        predictions = []

        # Fuzzy VAE prediction
        fuzzy_output = self.fuzzy(z_fused)
        fuzzy_pred = fuzzy_output["mu"]
        predictions.append(fuzzy_pred)

        # Full-sequence transformer
        full_pred = None
        if self.full_seq is not None and sequence is not None:
            full_output = self.full_seq(sequence, mutation_pos, padding_mask)
            full_pred = full_output["ddg_pred"]
            predictions.append(full_pred)

        # Hierarchical transformer
        hier_pred = None
        if self.hierarchical is not None and sequence is not None and mutation_pos is not None:
            hier_output = self.hierarchical(sequence, mutation_pos, padding_mask)
            hier_pred = hier_output["ddg_pred"]
            predictions.append(hier_pred)

        # Weighted ensemble
        stacked = torch.stack(predictions, dim=-1)  # (batch, 1, n_components)
        ensemble_pred = (stacked * weights).sum(dim=-1)

        # Ensemble uncertainty (from fuzzy + prediction disagreement)
        fuzzy_sigma = fuzzy_output["sigma"]
        if len(predictions) > 1:
            pred_var = torch.var(stacked, dim=-1)
            ensemble_sigma = torch.sqrt(fuzzy_sigma**2 + pred_var)
        else:
            ensemble_sigma = fuzzy_sigma

        return {
            "ddg_pred": ensemble_pred,
            "sigma": ensemble_sigma,
            "fuzzy": {
                "mu": fuzzy_pred,
                "sigma": fuzzy_output["sigma"],
            },
            "full_sequence": full_pred,
            "hierarchical": hier_pred,
            "weights": weights,
        }

    def predict(
        self,
        z_fused: torch.Tensor,
        sequence: torch.Tensor | None = None,
        mutation_pos: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Make ensemble DDG predictions.

        Args:
            z_fused: Fused latent
            sequence: Token indices
            mutation_pos: Mutation positions
            padding_mask: Padding mask

        Returns:
            Ensemble DDG predictions (batch, 1)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(z_fused, sequence, mutation_pos, padding_mask)
            return output["ddg_pred"]

    def predict_with_uncertainty(
        self,
        z_fused: torch.Tensor,
        sequence: torch.Tensor | None = None,
        mutation_pos: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Make predictions with uncertainty estimates.

        Args:
            z_fused: Fused latent
            sequence: Token indices
            mutation_pos: Mutation positions
            padding_mask: Padding mask

        Returns:
            Tuple of (predictions, uncertainties)
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(z_fused, sequence, mutation_pos, padding_mask)
            return output["ddg_pred"], output["sigma"]

    def loss(
        self,
        z_fused: torch.Tensor,
        y: torch.Tensor,
        sequence: torch.Tensor | None = None,
        mutation_pos: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> dict[str, torch.Tensor]:
        """Compute ensemble loss.

        Args:
            z_fused: Fused latent
            y: Target DDG values
            sequence: Token indices
            mutation_pos: Mutation positions
            padding_mask: Padding mask
            reduction: Loss reduction

        Returns:
            Dictionary with loss components
        """
        output = self.forward(z_fused, sequence, mutation_pos, padding_mask)

        if y.dim() == 1:
            y = y.unsqueeze(-1)

        # Ensemble MSE loss
        ensemble_mse = F.mse_loss(output["ddg_pred"], y, reduction=reduction)

        # Fuzzy NLL loss (encourages calibrated uncertainty)
        fuzzy_nll = self.fuzzy.nll_loss(z_fused, y, reduction=reduction)

        # Component losses (for monitoring)
        component_losses = {
            "fuzzy_mse": F.mse_loss(output["fuzzy"]["mu"], y, reduction=reduction),
        }
        if output["full_sequence"] is not None:
            component_losses["full_seq_mse"] = F.mse_loss(output["full_sequence"], y, reduction=reduction)
        if output["hierarchical"] is not None:
            component_losses["hier_mse"] = F.mse_loss(output["hierarchical"], y, reduction=reduction)

        # Total loss: ensemble + uncertainty calibration
        total_loss = ensemble_mse + 0.1 * fuzzy_nll

        return {
            "loss": total_loss,
            "ensemble_mse": ensemble_mse,
            "fuzzy_nll": fuzzy_nll,
            **component_losses,
        }

    @classmethod
    def from_components(
        cls,
        fused_dim: int = 128,
        full_seq_config: dict | None = None,
        hier_config: dict | None = None,
    ) -> DDGEnsemble:
        """Create ensemble from component configurations.

        Args:
            fused_dim: Dimension of fused latent
            full_seq_config: Config for full-sequence transformer
            hier_config: Config for hierarchical transformer

        Returns:
            DDGEnsemble instance
        """
        from src.bioinformatics.models.ddg_transformer import (
            DDGTransformer,
            HierarchicalTransformer,
            TransformerConfig,
        )

        # Create fuzzy head
        fuzzy = FuzzyDDGHead(input_dim=fused_dim)

        # Create transformers if configs provided
        full_seq = None
        if full_seq_config is not None:
            config = TransformerConfig(**full_seq_config)
            full_seq = DDGTransformer(config)

        hier = None
        if hier_config is not None:
            config = TransformerConfig(**hier_config)
            hier = HierarchicalTransformer(config)

        return cls(fuzzy, full_seq, hier)


__all__ = [
    "EnsembleConfig",
    "FuzzyDDGHead",
    "DDGEnsemble",
]
