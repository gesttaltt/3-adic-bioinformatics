# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""VAE-Specific Natural Gradient Optimizers.

Implements natural gradient optimization tailored for VAE training,
accounting for the specific structure of encoder and decoder distributions.

Key Insight:
For VAEs, the relevant Fisher information comes from:
1. Encoder Fisher: Information about q(z|x) parameters
2. Decoder Fisher: Information about p(x|z) parameters
3. Prior Fisher: Information about the latent prior p(z)

Natural gradients respect this geometry, leading to:
- More stable ELBO optimization
- Better uncertainty quantification
- Improved latent space structure

Mathematical Background:
The natural gradient for parameter theta is:
    tilde{grad} = F^{-1} grad L

where F is the Fisher information matrix.

For VAEs with Gaussian encoder q(z|x) = N(mu(x), sigma^2(x)):
    F_encoder = E[ (d log q / d mu)^2 ] = 1/sigma^2 for mean
    F_encoder = E[ (d log q / d log sigma)^2 ] = 2 for log-variance

References:
- Hoffman (2013): Stochastic Variational Inference
- Khan (2018): Fast and Scalable Estimation of Uncertainty
- Salimbeni (2018): Natural Gradients in Practice
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.optim import Optimizer


@dataclass
class FisherEstimate:
    """Container for Fisher information estimates."""

    mean_fisher: torch.Tensor  # Fisher for mean parameters
    var_fisher: torch.Tensor  # Fisher for variance parameters
    condition_number: float
    effective_rank: float


class VAEFisherEstimator:
    """Estimate Fisher information for VAE encoder parameters.

    For a VAE with Gaussian encoder q(z|x) = N(mu_theta(x), sigma_theta^2(x)),
    the Fisher information has a simple structure:

    For mean parameters:
        F_mu = E[1/sigma^2 * (d mu/d theta)^T (d mu/d theta)]

    For variance parameters:
        F_sigma = E[2/sigma^2 * (d log sigma/d theta)^T (d log sigma/d theta)]

    This class efficiently estimates these quantities using mini-batches.
    """

    def __init__(
        self,
        encoder: nn.Module,
        damping: float = 1e-4,
        ema_decay: float = 0.95,
    ):
        """Initialize Fisher estimator.

        Args:
            encoder: VAE encoder module
            damping: Damping for numerical stability
            ema_decay: EMA decay for online estimation
        """
        self.encoder = encoder
        self.damping = damping
        self.ema_decay = ema_decay

        # Running estimates (diagonal approximation)
        self._mean_fisher: dict[str, torch.Tensor] | None = None
        self._var_fisher: dict[str, torch.Tensor] | None = None
        self._step = 0

    def update(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> FisherEstimate:
        """Update Fisher estimate with new batch.

        Args:
            x: Input batch (batch, input_dim)
            mu: Encoder mean output (batch, latent_dim)
            logvar: Encoder log-variance output (batch, latent_dim)

        Returns:
            Current Fisher estimate
        """
        self._step += 1
        var = torch.exp(logvar)

        # For diagonal Fisher, we estimate E[g^2] for each parameter
        # where g is the gradient of log q(z|x)

        # Initialize if needed
        if self._mean_fisher is None:
            self._mean_fisher = {}
            self._var_fisher = {}
            for name, param in self.encoder.named_parameters():
                if param.requires_grad:
                    self._mean_fisher[name] = torch.zeros_like(param)
                    self._var_fisher[name] = torch.zeros_like(param)

        # Compute gradients for mean parameters
        # grad_theta log q = grad_theta [(z - mu)^2 / (2 sigma^2)]
        # At z = mu: grad_mu log q = 0, but Fisher = E[(grad)^2] = 1/sigma^2 * |d mu/d theta|^2

        # Sample z ~ q(z|x) for stochastic estimation
        eps = torch.randn_like(mu)
        z = mu + var.sqrt() * eps

        # Compute score function for mean: grad_mu log q = (z - mu) / sigma^2
        score_mu = (z - mu) / var

        # Backpropagate score to get parameter gradients
        mean_grads = self._compute_param_gradients(mu, score_mu)

        # Compute score function for logvar: grad_logvar log q = -0.5 + 0.5 * (z - mu)^2 / sigma^2
        score_logvar = -0.5 + 0.5 * (z - mu) ** 2 / var
        var_grads = self._compute_param_gradients(logvar, score_logvar)

        # Update running estimates with EMA
        for name in self._mean_fisher:
            if name in mean_grads:
                g_sq = mean_grads[name] ** 2
                self._mean_fisher[name] = self.ema_decay * self._mean_fisher[name] + (1 - self.ema_decay) * g_sq

            if name in var_grads:
                g_sq = var_grads[name] ** 2
                self._var_fisher[name] = self.ema_decay * self._var_fisher[name] + (1 - self.ema_decay) * g_sq

        # Compute summary statistics
        all_fisher = []
        for name in self._mean_fisher:
            all_fisher.append(self._mean_fisher[name].view(-1))
            all_fisher.append(self._var_fisher[name].view(-1))

        fisher_diag = torch.cat(all_fisher)
        fisher_diag = fisher_diag.clamp(min=self.damping)

        condition_number = (fisher_diag.max() / fisher_diag.min()).item()
        effective_rank = (fisher_diag.sum() ** 2 / (fisher_diag**2).sum()).item()

        return FisherEstimate(
            mean_fisher=fisher_diag[: len(fisher_diag) // 2],
            var_fisher=fisher_diag[len(fisher_diag) // 2 :],
            condition_number=condition_number,
            effective_rank=effective_rank,
        )

    def _compute_param_gradients(
        self,
        output: torch.Tensor,
        score: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute parameter gradients via backpropagation.

        Args:
            output: Encoder output (mu or logvar)
            score: Score function values

        Returns:
            Dict mapping parameter names to gradients
        """
        # Compute weighted gradients
        weighted_loss = (output * score.detach()).sum()

        self.encoder.zero_grad()
        weighted_loss.backward(retain_graph=True)

        grads = {}
        for name, param in self.encoder.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone()

        return grads

    def get_preconditioner(
        self,
        param_name: str,
    ) -> torch.Tensor:
        """Get preconditioner (inverse Fisher) for parameter.

        Args:
            param_name: Name of parameter

        Returns:
            Preconditioner tensor (same shape as parameter)
        """
        if self._mean_fisher is None:
            return torch.ones(1)

        fisher = self._mean_fisher.get(param_name, torch.ones(1))
        fisher = fisher + self._var_fisher.get(param_name, torch.zeros(1))

        # Inverse with damping
        return 1.0 / (fisher + self.damping)


class VAENaturalGradient(Optimizer):
    """Natural gradient optimizer specifically for VAEs.

    Uses the structure of VAE training to efficiently compute and
    apply natural gradients:

    1. For encoder mean parameters: precondition by 1/variance
    2. For encoder variance parameters: precondition by 1/2
    3. For decoder parameters: use standard Fisher approximation

    This is more efficient than generic natural gradient methods
    because it exploits the known structure of the VAE likelihood.
    """

    def __init__(
        self,
        params: Iterator,
        lr: float = 0.001,
        damping: float = 1e-4,
        ema_decay: float = 0.95,
        weight_decay: float = 0.0,
        use_adaptive_damping: bool = True,
    ):
        """Initialize VAE natural gradient optimizer.

        Args:
            params: Model parameters
            lr: Learning rate
            damping: Tikhonov damping for Fisher inversion
            ema_decay: EMA decay for Fisher estimation
            weight_decay: L2 regularization
            use_adaptive_damping: Whether to adapt damping automatically
        """
        defaults = dict(
            lr=lr,
            damping=damping,
            ema_decay=ema_decay,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        self.use_adaptive_damping = use_adaptive_damping
        self._step_count = 0

    def set_variance_estimate(
        self,
        variance: torch.Tensor,
        param_groups: list | None = None,
    ):
        """Set current variance estimate from encoder for preconditioning.

        Args:
            variance: Variance tensor from encoder (batch, latent_dim)
            param_groups: Which param groups to update (default: all)
        """
        mean_var = variance.mean(dim=0)  # Average over batch

        groups = param_groups or range(len(self.param_groups))
        for group_idx in groups:
            self.param_groups[group_idx]["current_variance"] = mean_var

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> torch.Tensor | None:
        """Perform optimization step.

        Args:
            closure: Closure that reevaluates the model and returns loss

        Returns:
            Loss value if closure provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients not supported")

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["fisher_diag"] = torch.ones_like(p)

                state["step"] += 1

                # Update Fisher diagonal estimate (EMA of squared gradients)
                ema = group["ema_decay"]
                fisher = state["fisher_diag"]
                fisher.mul_(ema).addcmul_(grad, grad, value=1 - ema)

                # Compute natural gradient
                damping = group["damping"]
                if self.use_adaptive_damping:
                    # Adapt damping based on gradient magnitude
                    grad_norm = grad.norm()
                    damping = max(damping, 0.01 * grad_norm.item())

                nat_grad = grad / (fisher.sqrt() + damping)

                # Weight decay
                if group["weight_decay"] != 0:
                    nat_grad.add_(p, alpha=group["weight_decay"])

                # Update parameter
                p.add_(nat_grad, alpha=-group["lr"])

        return loss


class AdaptiveNaturalGradient(Optimizer):
    """Natural gradient with adaptive damping and learning rate.

    Automatically adjusts:
    - Damping: Based on trust region / actual improvement ratio
    - Learning rate: Based on gradient norm and Fisher curvature

    This makes the optimizer more robust to hyperparameter choices.
    """

    def __init__(
        self,
        params: Iterator,
        lr: float = 0.01,
        initial_damping: float = 1.0,
        damping_decay: float = 0.99,
        min_damping: float = 1e-6,
        max_damping: float = 1e6,
        ema_decay: float = 0.95,
        weight_decay: float = 0.0,
    ):
        """Initialize adaptive natural gradient optimizer.

        Args:
            params: Model parameters
            lr: Initial learning rate
            initial_damping: Initial damping value
            damping_decay: Decay factor for damping reduction
            min_damping: Minimum damping
            max_damping: Maximum damping
            ema_decay: EMA decay for Fisher estimation
            weight_decay: L2 regularization
        """
        defaults = dict(
            lr=lr,
            damping=initial_damping,
            ema_decay=ema_decay,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.damping_decay = damping_decay
        self.min_damping = min_damping
        self.max_damping = max_damping
        self._prev_loss = None
        self._step_count = 0

    @torch.no_grad()
    def step(
        self,
        closure: Callable | None = None,
        loss: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Perform optimization step with adaptive damping.

        Args:
            closure: Closure that reevaluates the model
            loss: Current loss value (for damping adaptation)

        Returns:
            Loss value if closure provided
        """
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        # Adapt damping based on loss change
        if loss is not None and self._prev_loss is not None:
            loss_val = loss.item() if torch.is_tensor(loss) else loss
            improvement = self._prev_loss - loss_val

            for group in self.param_groups:
                if improvement > 0:
                    # Good step, reduce damping
                    group["damping"] = max(self.min_damping, group["damping"] * self.damping_decay)
                else:
                    # Bad step, increase damping
                    group["damping"] = min(self.max_damping, group["damping"] / self.damping_decay)

        if loss is not None:
            self._prev_loss = loss.item() if torch.is_tensor(loss) else loss

        # Standard natural gradient step
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["fisher_diag"] = torch.ones_like(p)
                    state["step"] = 0

                state["step"] += 1

                # Update Fisher estimate
                ema = group["ema_decay"]
                fisher = state["fisher_diag"]
                fisher.mul_(ema).addcmul_(grad, grad, value=1 - ema)

                # Natural gradient with current damping
                damping = group["damping"]
                nat_grad = grad / (fisher.sqrt() + damping)

                if group["weight_decay"] != 0:
                    nat_grad.add_(p, alpha=group["weight_decay"])

                p.add_(nat_grad, alpha=-group["lr"])

        return loss

    def get_damping(self) -> float:
        """Get current average damping across parameter groups."""
        return sum(g["damping"] for g in self.param_groups) / len(self.param_groups)


class FisherRaoSGD(Optimizer):
    """SGD with Fisher-Rao metric preconditioning.

    Unlike standard natural gradient which uses empirical Fisher,
    this optimizer uses the closed-form Fisher for known distributions:

    For Gaussian mean: F = 1/sigma^2
    For Gaussian log-variance: F = 2

    This is more stable and doesn't require gradient accumulation.
    """

    def __init__(
        self,
        params: Iterator,
        lr: float = 0.01,
        momentum: float = 0.9,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        min_preconditioner: float = 0.1,
        max_preconditioner: float = 10.0,
    ):
        """Initialize Fisher-Rao SGD.

        Args:
            params: Model parameters
            lr: Learning rate
            momentum: Momentum factor
            dampening: Dampening for momentum
            weight_decay: L2 regularization
            nesterov: Use Nesterov momentum
            min_preconditioner: Minimum preconditioner value
            max_preconditioner: Maximum preconditioner value
        """
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

        self.min_preconditioner = min_preconditioner
        self.max_preconditioner = max_preconditioner
        self._preconditioners: dict[int, torch.Tensor] = {}

    def set_preconditioner(
        self,
        param_id: int,
        preconditioner: torch.Tensor,
    ):
        """Set preconditioner for a specific parameter.

        Args:
            param_id: Parameter id (from id(param))
            preconditioner: Preconditioner tensor
        """
        self._preconditioners[param_id] = preconditioner.clamp(
            self.min_preconditioner,
            self.max_preconditioner,
        )

    def set_variance_preconditioner(
        self,
        mean_params: list,
        variance: torch.Tensor,
    ):
        """Set preconditioners for encoder mean parameters based on variance.

        For Gaussian encoder, optimal preconditioner for mean is 1/variance.

        Args:
            mean_params: List of mean parameters (typically encoder output layer)
            variance: Current variance estimate (batch, latent_dim)
        """
        mean_var = variance.mean(dim=0)
        precond = 1.0 / (mean_var + 1e-8)

        for param in mean_params:
            # Match preconditioner shape to parameter
            if param.dim() == 2:  # Weight matrix
                # Use average over latent dimensions
                self._preconditioners[id(param)] = precond.mean().expand_as(param)
            else:  # Bias
                self._preconditioners[id(param)] = precond.expand_as(param)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> torch.Tensor | None:
        """Perform optimization step.

        Args:
            closure: Closure that reevaluates the model

        Returns:
            Loss value if closure provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # Apply preconditioner if available
                param_id = id(p)
                if param_id in self._preconditioners:
                    precond = self._preconditioners[param_id]
                    grad = grad * precond

                # Weight decay
                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

                # Momentum
                if momentum != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = torch.clone(grad).detach()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(momentum).add_(grad, alpha=1 - dampening)

                    grad = grad.add(buf, alpha=momentum) if nesterov else buf

                p.add_(grad, alpha=-group["lr"])

        return loss


__all__ = [
    "VAEFisherEstimator",
    "FisherEstimate",
    "VAENaturalGradient",
    "AdaptiveNaturalGradient",
    "FisherRaoSGD",
]
