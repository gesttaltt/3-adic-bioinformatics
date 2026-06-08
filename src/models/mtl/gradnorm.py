# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""GradNorm for automatic task weighting in multi-task learning.

Dynamically adjusts task weights based on training progress
to balance learning across all tasks.

References:
    - Chen et al. (2018): GradNorm: Gradient Normalization for
      Adaptive Loss Balancing in Deep Multitask Networks
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GradNormOptimizer:
    """GradNorm optimizer for multi-task learning.

    Automatically adjusts task weights to balance gradient magnitudes
    across tasks, ensuring all tasks learn at similar rates.

    Example:
        >>> model = MultiTaskModel()
        >>> gradnorm = GradNormOptimizer(model, n_tasks=3, alpha=1.5)
        >>> for batch in dataloader:
        ...     losses = model.compute_losses(batch)
        ...     gradnorm.step(losses)
    """

    def __init__(
        self,
        model: nn.Module,
        n_tasks: int,
        alpha: float = 1.5,
        lr: float = 0.025,
        shared_layer_name: str | None = "shared_encoder",
    ):
        """Initialize GradNorm optimizer.

        Args:
            model: Multi-task model
            n_tasks: Number of tasks
            alpha: Asymmetry parameter (1.5 recommended)
            lr: Learning rate for task weights
            shared_layer_name: Name of shared layer for gradient computation
        """
        self.model = model
        self.n_tasks = n_tasks
        self.alpha = alpha
        self.lr = lr
        self.shared_layer_name = shared_layer_name

        # Initialize task weights
        self.task_weights = nn.Parameter(torch.ones(n_tasks, requires_grad=True))

        # Track initial losses for relative improvement
        self.initial_losses: torch.Tensor | None = None
        self.loss_ratios: list[torch.Tensor] = []

        # Get shared layer parameters
        self.shared_params = self._get_shared_params()

    def _get_shared_params(self) -> list[nn.Parameter]:
        """Get parameters of the shared layer."""
        shared_params = []

        for name, module in self.model.named_modules():
            if self.shared_layer_name in name:
                for param in module.parameters():
                    if param.requires_grad:
                        shared_params.append(param)
                break

        if not shared_params:
            # Fallback: use last layer of first sequential
            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    shared_params.extend(module.parameters())
                    break

        return shared_params

    def step(
        self,
        task_losses: dict[str, torch.Tensor],
        update_weights: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Perform GradNorm step.

        Args:
            task_losses: Dictionary of per-task losses
            update_weights: Whether to update task weights

        Returns:
            Tuple of (weighted_loss, task_weights_dict)
        """
        losses = list(task_losses.values())
        task_names = list(task_losses.keys())

        if len(losses) != self.n_tasks:
            # Pad with zeros if fewer losses
            while len(losses) < self.n_tasks:
                losses.append(torch.tensor(0.0, device=losses[0].device))
                task_names.append(f"task_{len(losses)}")

        losses_tensor = torch.stack(losses)

        # Store initial losses
        if self.initial_losses is None:
            self.initial_losses = losses_tensor.detach().clone()

        # Compute loss ratios (relative improvement)
        loss_ratios = losses_tensor / (self.initial_losses + 1e-8)
        avg_loss_ratio = loss_ratios.mean()

        # Compute relative inverse training rate
        r_i = loss_ratios / (avg_loss_ratio + 1e-8)

        if update_weights:
            # Compute gradients w.r.t. shared parameters
            grads = []
            for _i, loss in enumerate(losses):
                if loss.requires_grad:
                    grad = torch.autograd.grad(
                        loss,
                        self.shared_params,
                        retain_graph=True,
                        allow_unused=True,
                    )
                    grad_norm = sum(g.norm() ** 2 for g in grad if g is not None) ** 0.5
                    grads.append(grad_norm)
                else:
                    grads.append(torch.tensor(0.0, device=losses[0].device))

            grads_tensor = torch.stack(grads)

            # Compute weighted gradients
            weighted_grads = grads_tensor * self.task_weights

            # Target gradient magnitudes
            mean_grad = weighted_grads.mean()
            target_grads = mean_grad * (r_i**self.alpha)

            # GradNorm loss
            gradnorm_loss = (weighted_grads - target_grads).abs().sum()

            # Update task weights
            with torch.no_grad():
                grad_weights = torch.autograd.grad(
                    gradnorm_loss,
                    self.task_weights,
                    retain_graph=True,
                )[0]

                self.task_weights.data -= self.lr * grad_weights

                # Renormalize weights
                self.task_weights.data = self.task_weights.data * self.n_tasks / self.task_weights.data.sum()

                # Clamp to positive
                self.task_weights.data = self.task_weights.data.clamp(min=0.1)

        # Compute weighted loss
        weighted_loss = (losses_tensor * self.task_weights).sum()

        # Return weight dictionary
        weight_dict = {name: self.task_weights[i].item() for i, name in enumerate(task_names[: self.n_tasks])}

        return weighted_loss, weight_dict

    def get_weights(self) -> dict[str, float]:
        """Get current task weights."""
        return {f"task_{i}": self.task_weights[i].item() for i in range(self.n_tasks)}

    def reset(self):
        """Reset optimizer state."""
        self.initial_losses = None
        self.loss_ratios = []
        self.task_weights.data = torch.ones(self.n_tasks)


class UncertaintyWeighting(nn.Module):
    """Uncertainty-based task weighting.

    Learns task-specific homoscedastic uncertainty parameters
    to automatically weight losses.

    References:
        - Kendall et al. (2018): Multi-Task Learning Using Uncertainty
          to Weigh Losses for Scene Geometry and Semantics
    """

    def __init__(self, n_tasks: int):
        """Initialize uncertainty weighting.

        Args:
            n_tasks: Number of tasks
        """
        super().__init__()
        self.n_tasks = n_tasks

        # Log variance parameters (one per task)
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))

    def forward(
        self,
        task_losses: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute uncertainty-weighted loss.

        Args:
            task_losses: List of per-task losses

        Returns:
            Tuple of (weighted_loss, task_weights)
        """
        weighted_losses = []

        for i, loss in enumerate(task_losses):
            # Weight = 1 / (2 * sigma^2) = exp(-log_var) / 2
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + 0.5 * self.log_vars[i]
            weighted_losses.append(weighted_loss)

        total_loss = sum(weighted_losses)

        # Extract weights for logging
        weights = {f"task_{i}": torch.exp(-self.log_vars[i]).item() for i in range(self.n_tasks)}

        return total_loss, weights

    def get_uncertainties(self) -> dict[str, float]:
        """Get learned task uncertainties."""
        return {f"task_{i}": torch.exp(self.log_vars[i]).item() for i in range(self.n_tasks)}


class DynamicWeightAveraging(nn.Module):
    """Dynamic Weight Averaging for task balancing.

    Adjusts task weights based on rate of loss decrease.

    References:
        - Liu et al. (2019): End-to-End Multi-Task Learning with Attention
    """

    def __init__(
        self,
        n_tasks: int,
        temperature: float = 2.0,
    ):
        """Initialize DWA.

        Args:
            n_tasks: Number of tasks
            temperature: Temperature for softmax weighting
        """
        super().__init__()
        self.n_tasks = n_tasks
        self.temperature = temperature

        # Track loss history
        self.loss_history: list[torch.Tensor] = []
        self.max_history = 2

    def forward(
        self,
        task_losses: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute DWA-weighted loss.

        Args:
            task_losses: List of per-task losses

        Returns:
            Tuple of (weighted_loss, task_weights)
        """
        losses_tensor = torch.stack([l.detach() for l in task_losses])
        self.loss_history.append(losses_tensor)

        if len(self.loss_history) > self.max_history:
            self.loss_history.pop(0)

        if len(self.loss_history) < 2:
            # Equal weights initially
            weights = torch.ones(self.n_tasks, device=losses_tensor.device) / self.n_tasks
        else:
            # Rate of change
            prev_losses = self.loss_history[-2]
            curr_losses = self.loss_history[-1]

            # Relative change
            r = curr_losses / (prev_losses + 1e-8)

            # Softmax with temperature
            weights = torch.softmax(r / self.temperature, dim=0)

        # Weighted sum
        weighted_loss = (torch.stack(task_losses) * weights).sum()

        weight_dict = {f"task_{i}": weights[i].item() for i in range(self.n_tasks)}

        return weighted_loss, weight_dict


class PCGrad:
    """Projecting Conflicting Gradients for multi-task optimization.

    Modifies gradients to reduce conflicts between tasks.

    References:
        - Yu et al. (2020): Gradient Surgery for Multi-Task Learning
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        """Initialize PCGrad.

        Args:
            optimizer: Base optimizer
        """
        self.optimizer = optimizer

    def step(
        self,
        task_losses: list[torch.Tensor],
        model: nn.Module,
    ):
        """Perform PCGrad step.

        Args:
            task_losses: List of per-task losses
            model: Model to optimize
        """
        # Compute gradients for each task
        task_grads = []

        for loss in task_losses:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)

            grads = []
            for param in model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.clone().flatten())
                else:
                    grads.append(torch.zeros_like(param).flatten())

            task_grads.append(torch.cat(grads))

        # Project conflicting gradients
        projected_grads = self._project_gradients(task_grads)

        # Apply projected gradients
        self.optimizer.zero_grad()
        offset = 0
        for param in model.parameters():
            numel = param.numel()
            param.grad = projected_grads[offset : offset + numel].view_as(param)
            offset += numel

        self.optimizer.step()

    def _project_gradients(
        self,
        grads: list[torch.Tensor],
    ) -> torch.Tensor:
        """Project gradients to reduce conflicts.

        Args:
            grads: List of gradient vectors

        Returns:
            Projected gradient
        """
        import random

        n_tasks = len(grads)
        projected = grads[0].clone()

        # Randomly shuffle task order
        indices = list(range(1, n_tasks))
        random.shuffle(indices)

        for i in indices:
            grad_i = grads[i]

            # Check for conflict
            dot = torch.dot(projected, grad_i)

            if dot < 0:
                # Project out conflicting component
                projected = projected - (dot / (grad_i.norm() ** 2 + 1e-8)) * grad_i

        return projected
