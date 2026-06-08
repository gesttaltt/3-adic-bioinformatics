# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Information Geometry module for neural network analysis.

This module implements tools from information geometry for analyzing
and optimizing neural networks:
- Fisher information matrix estimation
- Natural gradient optimization
- Geodesic analysis on statistical manifolds

Key concepts:
- The parameter space of a probabilistic model forms a Riemannian manifold
- The Fisher information matrix defines the metric tensor
- Natural gradients account for the geometry of this manifold

References:
- Amari (1998): Natural Gradient Works Efficiently in Learning
- Martens (2014): New Insights and Perspectives on NGNN
- Pascanu & Bengio (2013): Revisiting Natural Gradient
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


@dataclass
class FisherInfo:
    """Container for Fisher information matrix and related quantities."""

    matrix: torch.Tensor  # Fisher information matrix
    eigenvalues: torch.Tensor | None = None
    eigenvectors: torch.Tensor | None = None
    condition_number: float | None = None
    trace: float | None = None
    log_determinant: float | None = None

    @classmethod
    def from_matrix(cls, F_matrix: torch.Tensor) -> FisherInfo:
        """Create FisherInfo from Fisher matrix with computed properties."""
        # Compute eigendecomposition for analysis
        eigenvalues, eigenvectors = torch.linalg.eigh(F_matrix)

        # Ensure positive (numerical stability)
        eigenvalues = eigenvalues.clamp(min=1e-10)

        condition_number = (eigenvalues.max() / eigenvalues.min()).item()
        trace = eigenvalues.sum().item()
        log_determinant = eigenvalues.log().sum().item()

        return cls(
            matrix=F_matrix,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            condition_number=condition_number,
            trace=trace,
            log_determinant=log_determinant,
        )


class FisherInformationEstimator:
    """Estimate Fisher information matrix for neural networks.

    The Fisher information matrix F measures the curvature of the
    log-likelihood surface:
        F = E[grad log p(x|theta) grad log p(x|theta)^T]

    For neural networks, we approximate this using:
    1. Empirical Fisher: gradients from data samples
    2. True Fisher: gradients from model samples
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = "empirical",
        damping: float = 1e-4,
        block_diagonal: bool = True,
    ):
        """Initialize Fisher estimator.

        Args:
            model: Neural network model
            method: 'empirical' or 'true' Fisher
            damping: Regularization for numerical stability
            block_diagonal: Use block-diagonal approximation per layer
        """
        self.model = model
        self.method = method
        self.damping = damping
        self.block_diagonal = block_diagonal

        # Collect parameters
        self.params = list(model.parameters())
        self.n_params = sum(p.numel() for p in self.params)

    def estimate(
        self,
        data_loader: Iterator,
        n_samples: int = 100,
        loss_fn: Callable | None = None,
    ) -> FisherInfo | dict[str, FisherInfo]:
        """Estimate Fisher information matrix.

        Args:
            data_loader: Data iterator yielding (input, target) pairs
            n_samples: Number of samples for estimation
            loss_fn: Loss function (default: cross-entropy)

        Returns:
            FisherInfo or dict of per-layer FisherInfo
        """
        if loss_fn is None:
            loss_fn = F.cross_entropy

        if self.block_diagonal:
            return self._estimate_block_diagonal(data_loader, n_samples, loss_fn)
        else:
            return self._estimate_full(data_loader, n_samples, loss_fn)

    def _estimate_full(
        self,
        data_loader: Iterator,
        n_samples: int,
        loss_fn: Callable,
    ) -> FisherInfo:
        """Estimate full Fisher matrix (expensive for large models)."""
        device = next(self.model.parameters()).device

        # Initialize accumulator
        F_matrix = torch.zeros(self.n_params, self.n_params, device=device)

        sample_count = 0
        for _batch_idx, (inputs, targets) in enumerate(data_loader):
            if sample_count >= n_samples:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)

            for i in range(min(batch_size, n_samples - sample_count)):
                # Single sample gradient
                self.model.zero_grad()
                output = self.model(inputs[i : i + 1])

                if self.method == "empirical":
                    # Use actual target
                    loss = loss_fn(output, targets[i : i + 1])
                else:
                    # Sample from model distribution (true Fisher)
                    with torch.no_grad():
                        probs = F.softmax(output, dim=-1)
                        sampled = torch.multinomial(probs, 1).squeeze()
                    loss = loss_fn(output, sampled.unsqueeze(0))

                loss.backward()

                # Collect gradients
                grad_vector = self._params_to_vector()
                F_matrix += torch.outer(grad_vector, grad_vector)

                sample_count += 1

        # Normalize
        F_matrix = F_matrix / sample_count

        # Add damping
        F_matrix = F_matrix + self.damping * torch.eye(self.n_params, device=device)

        return FisherInfo.from_matrix(F_matrix)

    def _estimate_block_diagonal(
        self,
        data_loader: Iterator,
        n_samples: int,
        loss_fn: Callable,
    ) -> dict[str, FisherInfo]:
        """Estimate block-diagonal Fisher (one block per layer)."""
        device = next(self.model.parameters()).device

        # Initialize accumulators for each parameter
        fisher_blocks = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                size = param.numel()
                fisher_blocks[name] = torch.zeros(size, size, device=device)

        sample_count = 0
        for _batch_idx, (inputs, targets) in enumerate(data_loader):
            if sample_count >= n_samples:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)
            batch_size = inputs.size(0)

            for i in range(min(batch_size, n_samples - sample_count)):
                self.model.zero_grad()
                output = self.model(inputs[i : i + 1])

                if self.method == "empirical":
                    loss = loss_fn(output, targets[i : i + 1])
                else:
                    with torch.no_grad():
                        probs = F.softmax(output, dim=-1)
                        sampled = torch.multinomial(probs, 1).squeeze()
                    loss = loss_fn(output, sampled.unsqueeze(0))

                loss.backward()

                # Accumulate per-parameter Fisher blocks
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        g = param.grad.view(-1)
                        fisher_blocks[name] += torch.outer(g, g)

                sample_count += 1

        # Normalize and add damping
        result = {}
        for name, F_block in fisher_blocks.items():
            F_block = F_block / sample_count
            size = F_block.size(0)
            F_block = F_block + self.damping * torch.eye(size, device=device)
            result[name] = FisherInfo.from_matrix(F_block)

        return result

    def _params_to_vector(self) -> torch.Tensor:
        """Flatten all gradients to single vector."""
        grads = []
        for param in self.params:
            if param.grad is not None:
                grads.append(param.grad.view(-1))
            else:
                grads.append(torch.zeros(param.numel(), device=param.device))
        return torch.cat(grads)


class NaturalGradientOptimizer(Optimizer):
    """Optimizer using natural gradients.

    Natural gradients account for the information geometry of the
    parameter space by preconditioning with the inverse Fisher matrix:
        theta_{t+1} = theta_t - eta F^{-1} grad L

    This is equivalent to steepest descent in the space of distributions,
    not the space of parameters.
    """

    def __init__(
        self,
        params: Iterator,
        lr: float = 0.01,
        damping: float = 1e-4,
        cov_ema_decay: float = 0.95,
        weight_decay: float = 0.0,
    ):
        """Initialize natural gradient optimizer.

        Args:
            params: Model parameters
            lr: Learning rate
            damping: Tikhonov regularization for Fisher inversion
            cov_ema_decay: EMA decay for covariance estimation
            weight_decay: L2 regularization
        """
        defaults = dict(
            lr=lr,
            damping=damping,
            cov_ema_decay=cov_ema_decay,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

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

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("NaturalGradient does not support sparse gradients")

                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    # Initialize covariance estimate (diagonal approximation)
                    state["cov"] = torch.ones_like(p).view(-1)

                state["step"] += 1

                # Update covariance estimate (EMA of squared gradients)
                g_flat = grad.view(-1)
                cov = state["cov"]
                decay = group["cov_ema_decay"]
                cov.mul_(decay).addcmul_(g_flat, g_flat, value=1 - decay)

                # Compute natural gradient: F^{-1} g approx g / (cov + damping)
                damping = group["damping"]
                nat_grad = g_flat / (cov + damping)

                # Weight decay
                if group["weight_decay"] != 0:
                    nat_grad.add_(p.view(-1), alpha=group["weight_decay"])

                # Update parameter
                p.add_(nat_grad.view_as(p), alpha=-group["lr"])

        return loss


class KFACOptimizer(Optimizer):
    """Kronecker-Factored Approximate Curvature (K-FAC) optimizer.

    K-FAC approximates the Fisher information matrix for each layer
    as a Kronecker product: F approx A tensor S, where A is input covariance
    and S is output gradient covariance.

    This allows efficient inversion:
        F^{-1} approx A^{-1} tensor S^{-1}
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.01,
        damping: float = 1e-3,
        cov_ema_decay: float = 0.95,
        weight_decay: float = 0.0,
        update_freq: int = 10,
    ):
        """Initialize K-FAC optimizer.

        Args:
            model: Neural network model
            lr: Learning rate
            damping: Damping for matrix inversion
            cov_ema_decay: EMA decay for covariance estimation
            weight_decay: L2 regularization
            update_freq: Frequency of covariance updates
        """
        self.model = model
        self.damping = damping
        self.cov_ema_decay = cov_ema_decay
        self.update_freq = update_freq
        self.step_count = 0

        # State for each layer
        self.layer_state: dict[str, dict] = {}
        self._register_hooks()

        params = model.parameters()
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.register_forward_hook(self._save_input_hook(name))
                module.register_full_backward_hook(self._save_grad_hook(name))

                self.layer_state[name] = {
                    "input": None,
                    "grad_output": None,
                    "A": None,  # Input covariance
                    "S": None,  # Output gradient covariance
                }

    def _save_input_hook(self, name: str):
        """Create hook to save layer input."""

        def hook(module, input, output):
            if isinstance(input, tuple):
                input = input[0]
            self.layer_state[name]["input"] = input.detach()

        return hook

    def _save_grad_hook(self, name: str):
        """Create hook to save output gradient."""

        def hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                grad_output = grad_output[0]
            self.layer_state[name]["grad_output"] = grad_output.detach()

        return hook

    def _update_covariance(self, name: str, state: dict):
        """Update covariance estimates for a layer."""
        input_act = state["input"]
        grad_out = state["grad_output"]

        if input_act is None or grad_out is None:
            return

        # Flatten spatial dimensions if needed
        if input_act.dim() > 2:
            input_act = input_act.view(input_act.size(0), -1)
        if grad_out.dim() > 2:
            grad_out = grad_out.view(grad_out.size(0), -1)

        batch_size = input_act.size(0)

        # Append bias term
        input_act = F.pad(input_act, (0, 1), value=1.0)

        # Compute covariances
        A_new = (input_act.t() @ input_act) / batch_size
        S_new = (grad_out.t() @ grad_out) / batch_size

        # EMA update
        decay = self.cov_ema_decay
        if state["A"] is None:
            state["A"] = A_new
            state["S"] = S_new
        else:
            state["A"] = decay * state["A"] + (1 - decay) * A_new
            state["S"] = decay * state["S"] + (1 - decay) * S_new

    def _safe_inverse(
        self,
        matrix: torch.Tensor,
        damping: float,
        max_damping: float = 1.0,
        damping_factor: float = 10.0,
    ) -> torch.Tensor:
        """Numerically stable matrix inverse with adaptive damping.

        Args:
            matrix: Matrix to invert
            damping: Initial damping value
            max_damping: Maximum damping before giving up
            damping_factor: Factor to increase damping on failure

        Returns:
            Inverse matrix, or identity-scaled approximation if inversion fails
        """
        device = matrix.device
        n = matrix.size(0)
        current_damping = damping

        while current_damping <= max_damping:
            damped = matrix + current_damping * torch.eye(n, device=device)

            try:
                # Try Cholesky decomposition first (faster, more stable for SPD)
                L = torch.linalg.cholesky(damped)
                return torch.cholesky_inverse(L)
            except RuntimeError:
                pass

            try:
                # Fall back to standard inverse
                return torch.linalg.inv(damped)
            except RuntimeError:
                pass

            try:
                # Try pseudoinverse (SVD-based, most robust)
                return torch.linalg.pinv(damped)
            except RuntimeError:
                pass

            # Increase damping and retry
            current_damping *= damping_factor

        # Last resort: return scaled identity (equivalent to SGD step)
        return torch.eye(n, device=device) / damping

    def _compute_natural_grad(
        self,
        name: str,
        state: dict,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        """Compute natural gradient using Kronecker factorization."""
        A = state["A"]
        S = state["S"]

        if A is None or S is None:
            return grad  # Fall back to standard gradient

        # Check for numerical issues
        if torch.isnan(A).any() or torch.isnan(S).any():
            return grad  # Fall back on NaN

        if torch.isinf(A).any() or torch.isinf(S).any():
            return grad  # Fall back on Inf

        # Safe inversion with adaptive damping
        A_inv = self._safe_inverse(A, self.damping)
        S_inv = self._safe_inverse(S, self.damping)

        # Natural gradient: (A^{-1} tensor S^{-1}) vec(G) = vec(S^{-1} G A^{-1})
        try:
            if grad.dim() == 2:
                # Handle dimension mismatch safely
                a_size = A_inv.size(0) - 1  # Exclude bias
                if a_size == grad.size(1):
                    nat_grad = S_inv @ grad @ A_inv[:-1, :-1]
                else:
                    nat_grad = S_inv @ grad @ A_inv[: grad.size(1), : grad.size(1)]
            else:
                nat_grad = grad  # For bias

            # Sanity check
            if torch.isnan(nat_grad).any() or torch.isinf(nat_grad).any():
                return grad

            return nat_grad
        except RuntimeError:
            return grad  # Fall back on any error

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> torch.Tensor | None:
        """Perform optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_count += 1

        # Update covariances periodically
        if self.step_count % self.update_freq == 0:
            for name, state in self.layer_state.items():
                self._update_covariance(name, state)

        # Apply updates
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Find corresponding layer
                layer_name = None
                for name, module in self.model.named_modules():
                    if hasattr(module, "weight") and module.weight is p:
                        layer_name = name
                        break

                if layer_name and layer_name in self.layer_state:
                    grad = self._compute_natural_grad(
                        layer_name,
                        self.layer_state[layer_name],
                        p.grad,
                    )
                else:
                    grad = p.grad

                # Weight decay
                if group["weight_decay"] != 0:
                    grad = grad + group["weight_decay"] * p

                p.add_(grad, alpha=-group["lr"])

        return loss


class InformationGeometricAnalyzer:
    """Analyze neural network training through information geometry lens.

    Provides tools for:
    - Tracking Fisher information during training
    - Computing geodesic distances between model states
    - Analyzing loss landscape curvature
    """

    def __init__(
        self,
        model: nn.Module,
        track_eigenvalues: bool = True,
        n_eigenvalues: int = 10,
    ):
        """Initialize analyzer.

        Args:
            model: Neural network model
            track_eigenvalues: Whether to track Fisher eigenvalues
            n_eigenvalues: Number of eigenvalues to track
        """
        self.model = model
        self.track_eigenvalues = track_eigenvalues
        self.n_eigenvalues = n_eigenvalues

        self.history: dict[str, list] = {
            "condition_numbers": [],
            "trace": [],
            "log_det": [],
            "top_eigenvalues": [],
            "bottom_eigenvalues": [],
        }

    def analyze_step(
        self,
        data_loader: Iterator,
        n_samples: int = 50,
    ) -> dict[str, float]:
        """Analyze Fisher information at current training step.

        Args:
            data_loader: Data iterator
            n_samples: Samples for Fisher estimation

        Returns:
            Dict of metrics
        """
        estimator = FisherInformationEstimator(
            self.model,
            method="empirical",
            block_diagonal=False,
        )

        fisher_info = estimator.estimate(data_loader, n_samples)

        # Extract metrics
        metrics = {
            "condition_number": fisher_info.condition_number,
            "trace": fisher_info.trace,
            "log_determinant": fisher_info.log_determinant,
        }

        if self.track_eigenvalues and fisher_info.eigenvalues is not None:
            eigs = fisher_info.eigenvalues
            metrics["top_eigenvalue"] = eigs[-1].item()
            metrics["bottom_eigenvalue"] = eigs[0].item()

            # Store eigenvalue spectrum
            self.history["top_eigenvalues"].append(eigs[-self.n_eigenvalues :].cpu().numpy())
            self.history["bottom_eigenvalues"].append(eigs[: self.n_eigenvalues].cpu().numpy())

        # Update history
        self.history["condition_numbers"].append(metrics["condition_number"])
        self.history["trace"].append(metrics["trace"])
        self.history["log_det"].append(metrics["log_determinant"])

        return metrics

    def geodesic_distance(
        self,
        params1: dict[str, torch.Tensor],
        params2: dict[str, torch.Tensor],
        fisher_info: FisherInfo,
    ) -> float:
        """Compute geodesic distance between parameter configurations.

        The geodesic distance on the statistical manifold is:
            d(theta1, theta2) = sqrt((theta1 - theta2)^T F (theta1 - theta2))

        Args:
            params1: First parameter dict
            params2: Second parameter dict
            fisher_info: Fisher information at midpoint

        Returns:
            Geodesic distance
        """
        # Flatten parameters
        vec1 = torch.cat([p.view(-1) for p in params1.values()])
        vec2 = torch.cat([p.view(-1) for p in params2.values()])
        diff = vec1 - vec2

        # Compute distance
        F = fisher_info.matrix
        dist_sq = (diff @ F @ diff).item()

        return math.sqrt(max(dist_sq, 0))

    def effective_dimensionality(self, fisher_info: FisherInfo) -> float:
        """Compute effective dimensionality from Fisher information.

        This measures how many parameters are actually contributing
        to the model's function.

        eff_dim = tr(F)^2 / tr(F^2)
        """
        F = fisher_info.matrix
        trace = F.trace()
        trace_sq = (F @ F).trace()

        if trace_sq < 1e-10:
            return 0.0

        return (trace**2 / trace_sq).item()

    def flatness_measure(
        self,
        data_loader: Iterator,
        epsilon: float = 0.01,
        n_directions: int = 10,
    ) -> dict[str, float]:
        """Measure loss landscape flatness around current parameters.

        Uses random directions in parameter space to estimate local
        curvature.

        Args:
            data_loader: Data for loss computation
            epsilon: Perturbation magnitude
            n_directions: Number of random directions

        Returns:
            Dict with flatness metrics
        """
        # Get current parameters
        original_params = {name: p.clone() for name, p in self.model.named_parameters()}

        # Compute baseline loss
        baseline_loss = self._compute_loss(data_loader)

        # Sample random directions and compute loss variation
        losses = []
        for _ in range(n_directions):
            # Random direction
            for name, p in self.model.named_parameters():
                direction = torch.randn_like(p)
                direction = direction / direction.norm() * epsilon
                p.data = original_params[name] + direction

            perturbed_loss = self._compute_loss(data_loader)
            losses.append(perturbed_loss)

            # Restore original
            for name, p in self.model.named_parameters():
                p.data = original_params[name]

        losses = torch.tensor(losses)

        return {
            "baseline_loss": baseline_loss,
            "mean_perturbed_loss": losses.mean().item(),
            "std_perturbed_loss": losses.std().item(),
            "max_increase": (losses - baseline_loss).max().item(),
            "sharpness": (losses - baseline_loss).mean().item() / epsilon,
        }

    def _compute_loss(self, data_loader: Iterator) -> float:
        """Compute average loss on data."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                device = next(self.model.parameters()).device
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, targets)
                total_loss += loss.item()
                n_batches += 1

                if n_batches >= 10:  # Limit for efficiency
                    break

        self.model.train()
        return total_loss / max(n_batches, 1)
