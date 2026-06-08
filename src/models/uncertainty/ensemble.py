# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Ensemble-based uncertainty quantification.

Uses disagreement between ensemble members to estimate epistemic
uncertainty. Multiple independently trained models capture model
uncertainty through prediction variance.

References:
    - Lakshminarayanan et al. (2017): Simple and Scalable Predictive Uncertainty
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsemblePredictor(nn.Module):
    """Ensemble of neural networks for uncertainty estimation.

    Trains multiple models with different initializations and/or
    data subsets to capture epistemic uncertainty through disagreement.

    Example:
        >>> base_model_fn = lambda: nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
        >>> ensemble = EnsemblePredictor(base_model_fn, n_members=5)
        >>> result = ensemble.predict_with_uncertainty(x)
        >>> print(result['epistemic_uncertainty'])
    """

    def __init__(
        self,
        model_fn: Callable[[], nn.Module],
        n_members: int = 5,
        aggregation: str = "mean",
    ):
        """Initialize ensemble.

        Args:
            model_fn: Factory function that creates a model instance
            n_members: Number of ensemble members
            aggregation: How to aggregate predictions ('mean', 'median')
        """
        super().__init__()
        self.n_members = n_members
        self.aggregation = aggregation

        # Create ensemble members
        self.members = nn.ModuleList([model_fn() for _ in range(n_members)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass with aggregated prediction.

        Args:
            x: Input tensor

        Returns:
            Aggregated prediction
        """
        predictions = torch.stack([member(x) for member in self.members], dim=0)

        if self.aggregation == "mean":
            return predictions.mean(dim=0)
        elif self.aggregation == "median":
            return predictions.median(dim=0).values
        else:
            return predictions.mean(dim=0)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
    ) -> dict:
        """Predict with uncertainty from ensemble disagreement.

        Args:
            x: Input tensor

        Returns:
            Dictionary with predictions and uncertainties
        """
        # Collect predictions from all members
        with torch.no_grad():
            predictions = torch.stack([member(x) for member in self.members], dim=0)  # (n_members, batch, ...)

        # Aggregate prediction
        mean_prediction = predictions.mean(dim=0) if self.aggregation == "mean" else predictions.median(dim=0).values

        # Epistemic uncertainty: variance across ensemble
        epistemic = predictions.var(dim=0)

        # Confidence based on agreement
        # Lower variance = higher confidence
        confidence = 1 / (1 + epistemic)

        return {
            "prediction": mean_prediction,
            "epistemic_uncertainty": epistemic,
            "total_uncertainty": epistemic,
            "confidence": confidence,
            "member_predictions": predictions,
        }

    def get_member_predictions(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Get individual predictions from all members.

        Args:
            x: Input tensor

        Returns:
            (n_members, batch, ...) predictions
        """
        with torch.no_grad():
            return torch.stack([member(x) for member in self.members], dim=0)


class DeepEnsemble(nn.Module):
    """Deep Ensemble with learned aleatoric uncertainty.

    Each member predicts both mean and variance (heteroscedastic),
    providing both epistemic (member disagreement) and aleatoric
    (learned data noise) uncertainty estimates.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: list[int] = None,
        n_members: int = 5,
    ):
        """Initialize deep ensemble.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            n_members: Number of ensemble members
        """
        if hidden_dims is None:
            hidden_dims = [256, 128]
        super().__init__()
        self.n_members = n_members
        self.output_dim = output_dim

        # Create ensemble members
        self.members = nn.ModuleList(
            [self._create_member(input_dim, output_dim, hidden_dims) for _ in range(n_members)]
        )

    def _create_member(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int],
    ) -> nn.Module:
        """Create a single ensemble member.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden dimensions

        Returns:
            Neural network module
        """
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            prev_dim = hidden_dim

        return nn.ModuleDict(
            {
                "backbone": nn.Sequential(*layers),
                "mean_head": nn.Linear(prev_dim, output_dim),
                "log_var_head": nn.Linear(prev_dim, output_dim),
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        member_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor
            member_idx: Specific member to use (None = all)

        Returns:
            Tuple of (mean, variance)
        """
        if member_idx is not None:
            # Single member
            member = self.members[member_idx]
            features = member["backbone"](x)
            mean = member["mean_head"](features)
            log_var = member["log_var_head"](features)
            return mean, torch.exp(log_var)
        else:
            # All members
            means = []
            variances = []

            for member in self.members:
                features = member["backbone"](x)
                mean = member["mean_head"](features)
                log_var = member["log_var_head"](features)
                means.append(mean)
                variances.append(torch.exp(log_var))

            # Stack: (n_members, batch, output_dim)
            means = torch.stack(means, dim=0)
            variances = torch.stack(variances, dim=0)

            # Mixture of Gaussians combination
            mixture_mean = means.mean(dim=0)
            mixture_var = (variances + means**2).mean(dim=0) - mixture_mean**2

            return mixture_mean, mixture_var

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
    ) -> dict:
        """Predict with decomposed uncertainty.

        Args:
            x: Input tensor

        Returns:
            Dictionary with predictions and uncertainties
        """
        with torch.no_grad():
            means = []
            variances = []

            for member in self.members:
                features = member["backbone"](x)
                mean = member["mean_head"](features)
                log_var = member["log_var_head"](features)
                means.append(mean)
                variances.append(torch.exp(log_var))

            means = torch.stack(means, dim=0)
            variances = torch.stack(variances, dim=0)

        # Epistemic uncertainty: variance of means
        epistemic = means.var(dim=0)

        # Aleatoric uncertainty: mean of variances
        aleatoric = variances.mean(dim=0)

        # Mean prediction
        mean_prediction = means.mean(dim=0)

        # Total uncertainty
        total = epistemic + aleatoric

        # Confidence
        confidence = 1 / (1 + total)

        return {
            "prediction": mean_prediction,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
            "total_uncertainty": total,
            "confidence": confidence,
            "member_means": means,
            "member_variances": variances,
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        member_idx: int | None = None,
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss.

        Args:
            x: Input features
            y: Targets
            member_idx: Train specific member (None = all)

        Returns:
            Loss value
        """
        if member_idx is not None:
            # Train single member
            mean, variance = self.forward(x, member_idx)
            nll = 0.5 * (torch.log(variance + 1e-8) + (y - mean) ** 2 / (variance + 1e-8))
            return nll.mean()
        else:
            # Train all members
            total_loss = 0
            for i in range(self.n_members):
                mean, variance = self.forward(x, i)
                nll = 0.5 * (torch.log(variance + 1e-8) + (y - mean) ** 2 / (variance + 1e-8))
                total_loss += nll.mean()
            return total_loss / self.n_members


class SnapshotEnsemble(nn.Module):
    """Snapshot Ensemble for efficient uncertainty.

    Collects model snapshots during training with cyclic learning rate
    to create an ensemble without training multiple models.

    References:
        - Huang et al. (2017): Snapshot Ensembles
    """

    def __init__(
        self,
        model: nn.Module,
        n_snapshots: int = 5,
    ):
        """Initialize snapshot ensemble.

        Args:
            model: Base model architecture
            n_snapshots: Maximum number of snapshots to keep
        """
        super().__init__()
        self.model = model
        self.n_snapshots = n_snapshots
        self.snapshots: list[dict] = []

    def add_snapshot(self):
        """Add current model state as a snapshot."""
        state = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.snapshots.append(state)

        # Keep only recent snapshots
        if len(self.snapshots) > self.n_snapshots:
            self.snapshots.pop(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with current model."""
        return self.model(x)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
    ) -> dict:
        """Predict using all snapshots.

        Args:
            x: Input tensor

        Returns:
            Dictionary with predictions and uncertainties
        """
        if len(self.snapshots) == 0:
            # No snapshots, use current model
            with torch.no_grad():
                pred = self.model(x)
            return {
                "prediction": pred,
                "epistemic_uncertainty": torch.zeros_like(pred),
                "confidence": torch.ones_like(pred),
            }

        # Collect predictions from all snapshots
        predictions = []
        current_state = {k: v.clone() for k, v in self.model.state_dict().items()}

        for snapshot_state in self.snapshots:
            self.model.load_state_dict(snapshot_state)
            with torch.no_grad():
                predictions.append(self.model(x))

        # Restore current state
        self.model.load_state_dict(current_state)

        predictions = torch.stack(predictions, dim=0)

        mean_prediction = predictions.mean(dim=0)
        epistemic = predictions.var(dim=0)
        confidence = 1 / (1 + epistemic)

        return {
            "prediction": mean_prediction,
            "epistemic_uncertainty": epistemic,
            "total_uncertainty": epistemic,
            "confidence": confidence,
            "snapshot_predictions": predictions,
        }

    def get_cyclic_lr_schedule(
        self,
        base_lr: float,
        max_lr: float,
        cycle_length: int,
    ) -> Callable[[int], float]:
        """Get cyclic learning rate schedule for training.

        Args:
            base_lr: Minimum learning rate
            max_lr: Maximum learning rate
            cycle_length: Steps per cycle

        Returns:
            Learning rate schedule function
        """
        import math

        def schedule(step: int) -> float:
            cycle_pos = (step % cycle_length) / cycle_length
            return base_lr + 0.5 * (max_lr - base_lr) * (1 + math.cos(math.pi * cycle_pos))

        return schedule


class BatchEnsemble(nn.Module):
    """Memory-efficient ensemble using rank-1 factors.

    Shares most parameters across ensemble members, using only
    rank-1 perturbations for diversity. Much more memory efficient
    than full ensembles.

    References:
        - Wen et al. (2020): BatchEnsemble
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] = None,
        n_members: int = 4,
    ):
        """Initialize batch ensemble.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            n_members: Number of ensemble members
        """
        if hidden_dims is None:
            hidden_dims = [256, 128]
        super().__init__()
        self.n_members = n_members

        # Shared backbone
        layers = []
        prev_dim = input_dim
        self.layer_dims = [input_dim]

        for hidden_dim in hidden_dims:
            layers.append(BatchEnsembleLinear(prev_dim, hidden_dim, n_members))
            layers.append(nn.SiLU())
            prev_dim = hidden_dim
            self.layer_dims.append(hidden_dim)

        self.backbone = nn.ModuleList(layers)

        # Output head
        self.output_head = BatchEnsembleLinear(prev_dim, output_dim, n_members)

    def forward(
        self,
        x: torch.Tensor,
        member_idx: int | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (batch, input_dim)
            member_idx: Specific member (None = all)

        Returns:
            Predictions
        """
        h = x

        for layer in self.backbone:
            h = layer(h, member_idx) if isinstance(layer, BatchEnsembleLinear) else layer(h)

        return self.output_head(h, member_idx)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
    ) -> dict:
        """Predict with all ensemble members.

        Args:
            x: Input tensor

        Returns:
            Dictionary with predictions and uncertainties
        """
        with torch.no_grad():
            predictions = torch.stack([self.forward(x, i) for i in range(self.n_members)], dim=0)

        mean_prediction = predictions.mean(dim=0)
        epistemic = predictions.var(dim=0)
        confidence = 1 / (1 + epistemic)

        return {
            "prediction": mean_prediction,
            "epistemic_uncertainty": epistemic,
            "total_uncertainty": epistemic,
            "confidence": confidence,
            "member_predictions": predictions,
        }


class BatchEnsembleLinear(nn.Module):
    """Linear layer with rank-1 ensemble factors.

    Implements W_i = W ⊙ (r_i ⊗ s_i) for each ensemble member i,
    where W is shared and r_i, s_i are learned rank-1 factors.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_members: int,
    ):
        """Initialize batch ensemble linear layer.

        Args:
            in_features: Input features
            out_features: Output features
            n_members: Number of ensemble members
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_members = n_members

        # Shared weight matrix
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Rank-1 factors for each member
        self.r = nn.Parameter(torch.ones(n_members, in_features))
        self.s = nn.Parameter(torch.ones(n_members, out_features))

    def forward(
        self,
        x: torch.Tensor,
        member_idx: int | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (batch, in_features)
            member_idx: Specific member (None = all, cycling through batch)

        Returns:
            Output tensor
        """
        if member_idx is not None:
            # Specific member
            r = self.r[member_idx]  # (in_features,)
            s = self.s[member_idx]  # (out_features,)

            # Apply rank-1 perturbation
            x_scaled = x * r
            out = F.linear(x_scaled, self.weight, self.bias)
            return out * s
        else:
            # Cycle through members across batch
            batch_size = x.shape[0]
            member_indices = torch.arange(batch_size, device=x.device) % self.n_members

            # Gather r and s for each sample
            r = self.r[member_indices]  # (batch, in_features)
            s = self.s[member_indices]  # (batch, out_features)

            x_scaled = x * r
            out = F.linear(x_scaled, self.weight, self.bias)
            return out * s
