# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Differentiable Controller for V5.11.

FIXES THE V5.10 STATENET GRADIENT FLOW PROBLEM.

The core problem in v5.10:
```python
optimizer.lr = lr * (1 + delta_lr.item())  # .item() breaks gradient chain!
self.lambda1 = self.lambda1 + delta.item()  # Python float, no gradients
```

The V5.11 solution: Move control signals INTO the forward pass as tensor operations.

Instead of modifying external hyperparameters, controller outputs become
MULTIPLIERS within the loss computation:

```python
# V5.10 (broken):
self.lambda1 = self.lambda1 + delta_lambda1.item()  # No gradient
total_loss = lambda1 * loss_A + ...  # lambda1 is Python float

# V5.11 (fixed):
control = self.controller(batch_features)  # Tensor output
weight_A = F.softplus(control['weight_recon'])  # Tensor, bounded positive
total_loss = weight_A * loss_A + ...  # Gradient flows through weight_A!
```

Single responsibility: Differentiable training dynamics control.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableController(nn.Module):
    """Lightweight controller that learns to modulate training dynamics.

    Key difference from StateNet: ALL outputs participate in tensor
    operations, so gradients flow back and the controller learns.

    Input: Batch statistics (computed as tensors, gradients intact)
    Output: Control signals (all tensors, gradients flow)

    The controller learns: "what weights minimize total_loss over time?"
    Gradients flow: loss → weights → controller → controller params
    """

    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        output_bounds: dict[str, tuple] | None = None,
    ):
        """Initialize DifferentiableController.

        Args:
            input_dim: Dimension of batch statistics input
            hidden_dim: Hidden layer dimension
            output_bounds: Dict of {name: (min, max)} for output clamping
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Small network - we want this to be lightweight
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),  # 6 control signals
        )

        # Output indices and their meanings:
        # 0: rho (cross-injection strength) [0, 0.5]
        # 1: weight_geodesic (geodesic loss importance) [0.1, inf)
        # 2: weight_radial (radial loss importance) [0, inf)
        # 3: beta_A (VAE-A KL weight) [0.5, inf)
        # 4: beta_B (VAE-B KL weight) [0.5, inf)
        # 5: tau (curriculum position) [0, 1]

        self.output_bounds = output_bounds or {
            "rho": (0.0, 0.5),
            "weight_geodesic": (0.1, 10.0),
            "weight_radial": (0.0, 5.0),
            "beta_A": (0.5, 5.0),
            "beta_B": (0.5, 5.0),
            "tau": (0.0, 1.0),
        }

        # Initialize for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize for stable starting values."""
        with torch.no_grad():
            # Initialize final layer to output near-zero (sigmoid/softplus will center)
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, batch_stats: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute control signals from batch statistics.

        CRITICAL: All outputs are tensors. Gradients flow through every output.

        Args:
            batch_stats: Tensor of batch statistics
                [mean_radius_A, mean_radius_B, H_A, H_B, kl_A, kl_B, geo_loss, rad_loss]
                All should be detached scalars converted to tensor, OR
                computed as part of forward pass with gradients.

        Returns:
            Dict of control signals (all tensors, gradients flow)
        """
        # Ensure batch dimension
        if batch_stats.dim() == 1:
            batch_stats = batch_stats.unsqueeze(0)

        raw = self.net(batch_stats)

        # Apply bounded activations (all tensors, all differentiable)
        return {
            "rho": torch.sigmoid(raw[:, 0]) * 0.5,  # [0, 0.5]
            "weight_geodesic": F.softplus(raw[:, 1]) + 0.1,  # [0.1, inf)
            "weight_radial": F.softplus(raw[:, 2]),  # [0, inf)
            "beta_A": F.softplus(raw[:, 3]) + 0.5,  # [0.5, inf)
            "beta_B": F.softplus(raw[:, 4]) + 0.5,  # [0.5, inf)
            "tau": torch.sigmoid(raw[:, 5]),  # [0, 1]
        }


__all__ = [
    "DifferentiableController",
]
