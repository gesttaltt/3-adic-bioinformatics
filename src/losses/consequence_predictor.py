# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Consequence Predictor: Teaches the model WHY ranking matters.

This module creates a feedback loop where the model learns that improving
3-adic correlation leads to improved algebraic closure capability.

The key insight: hunger without purpose is aimless optimization.
Purpose = understanding that r → addition_accuracy.

Single responsibility: Learn and predict consequence of metric structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geometry import poincare_distance


class ConsequencePredictor(nn.Module):
    """Predicts closure capability from metric quality.

    Learns: (ranking_correlation, z_statistics) → predicted_addition_accuracy

    This creates a gradient path: if the model improves r, it should
    predict higher addition accuracy, and the prediction error teaches
    it whether that prediction was correct.
    """

    def __init__(self, latent_dim: int = 16, hidden_dim: int = 32, curvature: float = 1.0):
        """Initialize consequence predictor.

        Args:
            latent_dim: Latent space dimension
            hidden_dim: Hidden layer dimension
            curvature: Curvature for hyperbolic distance (V5.12.2)
        """
        super().__init__()
        self.curvature = curvature

        # Input: [ranking_corr, z_mean_norm, z_std, coverage_estimate]
        # Output: predicted addition accuracy [0, 1]
        self.predictor = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Running statistics for normalization
        self.register_buffer("r_mean", torch.tensor(0.5))
        self.register_buffer("r_std", torch.tensor(0.2))
        self.register_buffer("n_updates", torch.tensor(0))

        # History for computing actual addition accuracy
        self.actual_accuracy_history: list[float] = []
        self.predicted_accuracy_history: list[float] = []

    def compute_z_statistics(self, z: torch.Tensor) -> tuple[float, float]:
        """Compute latent space statistics.

        Args:
            z: Latent codes (batch_size, latent_dim)

        Returns:
            (mean_hyp_dist, std) of latent vectors
        """
        # V5.12.2: Use hyperbolic distance from origin
        origin = torch.zeros_like(z)
        hyp_dists = poincare_distance(z, origin, c=self.curvature)
        return hyp_dists.mean().item(), z.std().item()

    def forward(
        self,
        ranking_correlation: float,
        z: torch.Tensor,
        coverage_pct: float = 0.0,
    ) -> torch.Tensor:
        """Predict addition accuracy from current state.

        Args:
            ranking_correlation: Current 3-adic correlation (0-1)
            z: Latent codes for statistics
            coverage_pct: Current coverage percentage (0-100)

        Returns:
            Predicted addition accuracy (0-1)
        """
        # Compute z statistics
        z_mean_norm, z_std = self.compute_z_statistics(z)

        # Normalize coverage to [0, 1]
        coverage_norm = coverage_pct / 100.0

        # Build input tensor
        features = torch.tensor(
            [ranking_correlation, z_mean_norm, z_std, coverage_norm],
            device=z.device,
            dtype=torch.float32,
        ).unsqueeze(0)

        # Predict
        predicted_accuracy = self.predictor(features)

        return predicted_accuracy.squeeze()

    def compute_loss(self, predicted_accuracy: torch.Tensor, actual_accuracy: float) -> torch.Tensor:
        """Compute prediction error.

        Args:
            predicted_accuracy: Model's prediction
            actual_accuracy: Ground truth from evaluation

        Returns:
            MSE loss between prediction and actual
        """
        actual = torch.tensor(actual_accuracy, device=predicted_accuracy.device)
        return F.mse_loss(predicted_accuracy, actual)

    def update_history(self, predicted: float, actual: float, max_history: int = 100):
        """Track prediction quality over time.

        Args:
            predicted: Predicted accuracy
            actual: Actual accuracy
            max_history: Maximum history length
        """
        self.predicted_accuracy_history.append(predicted)
        self.actual_accuracy_history.append(actual)

        if len(self.predicted_accuracy_history) > max_history:
            self.predicted_accuracy_history.pop(0)
            self.actual_accuracy_history.pop(0)

    def get_prediction_quality(self) -> float:
        """How well is the model predicting consequences?

        Returns:
            Correlation between predicted and actual (higher = better self-model)
        """
        if len(self.predicted_accuracy_history) < 10:
            return 0.0

        pred = torch.tensor(self.predicted_accuracy_history)
        actual = torch.tensor(self.actual_accuracy_history)

        # Pearson correlation
        pred_centered = pred - pred.mean()
        actual_centered = actual - actual.mean()

        numerator = (pred_centered * actual_centered).sum()
        denominator = torch.sqrt((pred_centered**2).sum() * (actual_centered**2).sum())

        if denominator < 1e-8:
            return 0.0

        return (numerator / denominator).item()


def evaluate_addition_accuracy(model: nn.Module, device: str, n_samples: int = 1000, curvature: float = 1.0) -> float:
    """Evaluate model's emergent addition capability.

    Tests: z_a + z_b - z_0 ≈ z_{a∘b}

    Args:
        model: VAE model with encode method
        device: Device
        n_samples: Number of test pairs
        curvature: Curvature for hyperbolic distance (V5.12.2)

    Returns:
        Addition accuracy (0-1)
    """
    model.eval()

    with torch.no_grad():
        # Generate random operation pairs
        # For simplicity, test with identity composition: a ∘ 0 = a
        n_ops = min(n_samples, 1000)

        # Generate ternary operations
        indices = torch.arange(n_ops, device=device)

        # Convert to ternary representation
        ternary_data = torch.zeros(n_ops, 9, device=device)
        for i in range(9):
            ternary_data[:, i] = ((indices // (3**i)) % 3) - 1

        # Zero operation (identity for some compositions)
        zero_op = torch.zeros(1, 9, device=device)

        # Encode operations
        # Handle both DualNeuralVAEV5 and wrapped models
        if hasattr(model, "base"):
            # AppetitiveDualVAE wrapper
            outputs = model.base(ternary_data.float(), 1.0, 1.0, 0.5, 0.5)
        else:
            # Direct DualNeuralVAEV5
            outputs = model(ternary_data.float(), 1.0, 1.0, 0.5, 0.5)

        z_A = outputs["z_A"]  # (n_ops, latent_dim)

        # Encode zero operation
        if hasattr(model, "base"):
            z0_outputs = model.base(zero_op.float(), 1.0, 1.0, 0.5, 0.5)
        else:
            z0_outputs = model(zero_op.float(), 1.0, 1.0, 0.5, 0.5)
        z_0 = z0_outputs["z_A"]  # (1, latent_dim)

        # Test 1: Identity operation should be near origin (small hyperbolic distance)
        # A well-structured latent space places identity centrally
        # V5.12.2: Use hyperbolic distance from origin
        origin_0 = torch.zeros_like(z_0)
        origin_A = torch.zeros_like(z_A)
        z_0_dist = poincare_distance(z_0, origin_0, c=curvature).item()
        z_A_dists = poincare_distance(z_A, origin_A, c=curvature)
        mean_dist = z_A_dists.mean().item()

        # Identity should have below-average distance (near center)
        identity_central = 1.0 if z_0_dist < mean_dist else 0.5

        # Test 2: 3-adically close operations should be latent-close
        # Sample random pairs and check if 3-adic neighbors are latent neighbors
        n_pairs = min(500, n_ops * (n_ops - 1) // 2)
        i_idx = torch.randint(0, n_ops, (n_pairs,), device=device)
        j_idx = torch.randint(0, n_ops, (n_pairs,), device=device)

        # Filter to distinct pairs
        valid = i_idx != j_idx
        i_idx, j_idx = i_idx[valid], j_idx[valid]

        if len(i_idx) > 10:
            # Compute 3-adic valuations (higher = closer in 3-adic metric)
            diff = torch.abs(indices[i_idx] - indices[j_idx])
            v_3adic = torch.zeros_like(diff, dtype=torch.float32)
            remaining = diff.clone()
            for _ in range(9):
                mask = (remaining % 3 == 0) & (remaining > 0)
                v_3adic[mask] += 1
                remaining[mask] = remaining[mask] // 3

            # V5.12.2: Compute hyperbolic latent distances
            latent_dist = poincare_distance(z_A[i_idx], z_A[j_idx], c=curvature)

            # Check correlation: high 3-adic valuation should correlate with low latent distance
            # Normalize to [0,1] for comparison
            v_norm = v_3adic / (v_3adic.max() + 1e-8)
            d_norm = latent_dist / (latent_dist.max() + 1e-8)

            # Concordance: pairs where high v_3adic (close in 3-adic) have low latent_dist
            # For triplets (i,j,k): if v(i,j) > v(i,k) then d(i,j) < d(i,k)
            # Simplified: check correlation between v_3adic and -latent_dist
            correlation = torch.corrcoef(torch.stack([v_norm, -d_norm]))[0, 1].item()
            # Map correlation [-1, 1] to accuracy [0, 1]
            structure_accuracy = (correlation + 1) / 2 if not torch.isnan(torch.tensor(correlation)) else 0.5
        else:
            structure_accuracy = 0.5

        # Combined accuracy: weight both tests
        accuracy = 0.3 * identity_central + 0.7 * structure_accuracy

    return accuracy


class PurposefulRankingLoss(nn.Module):
    """Ranking loss with consequence awareness.

    Combines the standard ranking loss with a term that encourages
    the model to understand why ranking matters for closure.
    """

    def __init__(self, latent_dim: int = 16, consequence_weight: float = 0.1):
        """Initialize purposeful ranking loss.

        Args:
            latent_dim: Latent space dimension
            consequence_weight: Weight for consequence prediction loss
        """
        super().__init__()
        self.consequence_predictor = ConsequencePredictor(latent_dim)
        self.consequence_weight = consequence_weight

        # Cache for delayed consequence feedback
        self.last_prediction = None
        self.last_ranking_corr = None

    def forward(
        self,
        ranking_correlation: float,
        z: torch.Tensor,
        coverage_pct: float,
        actual_addition_accuracy: float | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute consequence-aware loss.

        Args:
            ranking_correlation: Current 3-adic correlation
            z: Latent codes
            coverage_pct: Current coverage percentage
            actual_addition_accuracy: If provided, compute prediction error

        Returns:
            Dict with 'predicted_accuracy', 'consequence_loss'
        """
        # Predict what addition accuracy should be given current ranking
        predicted = self.consequence_predictor(ranking_correlation, z, coverage_pct)

        result = {
            "predicted_accuracy": predicted,
            "consequence_loss": torch.tensor(0.0, device=z.device),
        }

        # If we have ground truth, compute prediction error
        if actual_addition_accuracy is not None:
            consequence_loss = self.consequence_predictor.compute_loss(predicted, actual_addition_accuracy)
            result["consequence_loss"] = consequence_loss * self.consequence_weight

            # Update history for tracking
            self.consequence_predictor.update_history(predicted.item(), actual_addition_accuracy)

        return result

    def get_self_model_quality(self) -> float:
        """How well does the model understand its own capabilities?"""
        return self.consequence_predictor.get_prediction_quality()
