# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Meta-Learning module for few-shot adaptation.

This module implements meta-learning algorithms that enable rapid
adaptation to new diseases and biological domains with limited data.

Key features:
- MAML (Model-Agnostic Meta-Learning)
- P-adic task sampling based on biological hierarchy
- Few-shot adaptation for new pathogens

References:
- Finn et al. (2017): MAML
- Nichol et al. (2018): Reptile
- Snell et al. (2017): Prototypical Networks
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Task:
    """Represents a meta-learning task (support + query sets)."""

    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor
    task_id: int | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def n_support(self) -> int:
        return len(self.support_x)

    @property
    def n_query(self) -> int:
        return len(self.query_x)

    def to(self, device: torch.device) -> Task:
        """Move task data to device."""
        return Task(
            support_x=self.support_x.to(device),
            support_y=self.support_y.to(device),
            query_x=self.query_x.to(device),
            query_y=self.query_y.to(device),
            task_id=self.task_id,
            metadata=self.metadata,
        )


class MAML(nn.Module):
    """Model-Agnostic Meta-Learning.

    MAML learns an initialization that can be quickly adapted
    to new tasks with a few gradient steps.

    Algorithm:
    1. Sample batch of tasks
    2. For each task, compute adapted parameters via gradient descent
    3. Evaluate on query set with adapted parameters
    4. Update initialization based on query loss

    Key insight: The meta-gradient flows through the inner loop adaptation,
    learning an initialization that is easy to adapt.
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        n_inner_steps: int = 5,
        first_order: bool = False,
    ):
        """Initialize MAML.

        Args:
            model: Base model to meta-learn
            inner_lr: Learning rate for inner loop adaptation
            n_inner_steps: Number of gradient steps in inner loop
            first_order: Use first-order approximation (faster)
        """
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.n_inner_steps = n_inner_steps
        self.first_order = first_order

    def adapt(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        loss_fn: Callable | None = None,
    ) -> nn.Module:
        """Adapt model to task using support set.

        Args:
            support_x: Support set inputs
            support_y: Support set labels
            loss_fn: Loss function (default: cross-entropy)

        Returns:
            Adapted model (copy with updated parameters)
        """
        if loss_fn is None:
            loss_fn = F.cross_entropy

        # Create a copy of the model
        adapted_model = copy.deepcopy(self.model)

        # Inner loop: gradient descent on support set
        for _ in range(self.n_inner_steps):
            output = adapted_model(support_x)
            loss = loss_fn(output, support_y)

            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_model.parameters(),
                create_graph=not self.first_order,
            )

            # Update parameters
            for param, grad in zip(adapted_model.parameters(), grads, strict=False):
                param.data = param.data - self.inner_lr * grad

        return adapted_model

    def forward(
        self,
        task: Task,
        loss_fn: Callable | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a single task.

        Args:
            task: Task with support and query sets
            loss_fn: Loss function

        Returns:
            Tuple of (query_loss, query_predictions)
        """
        if loss_fn is None:
            loss_fn = F.cross_entropy

        # Adapt to task
        adapted_model = self.adapt(task.support_x, task.support_y, loss_fn)

        # Evaluate on query set
        query_output = adapted_model(task.query_x)
        query_loss = loss_fn(query_output, task.query_y)

        return query_loss, query_output

    def meta_train_step(
        self,
        tasks: list[Task],
        meta_optimizer: torch.optim.Optimizer,
        loss_fn: Callable | None = None,
    ) -> dict[str, float]:
        """Perform one meta-training step.

        Args:
            tasks: Batch of tasks
            meta_optimizer: Optimizer for meta-parameters
            loss_fn: Loss function

        Returns:
            Dict with training metrics
        """
        if loss_fn is None:
            loss_fn = F.cross_entropy

        meta_optimizer.zero_grad()

        total_loss = 0.0
        total_acc = 0.0

        for task in tasks:
            query_loss, query_output = self.forward(task, loss_fn)
            total_loss += query_loss

            # Compute accuracy
            preds = query_output.argmax(dim=-1)
            acc = (preds == task.query_y).float().mean()
            total_acc += acc.item()

        # Average loss and backprop
        avg_loss = total_loss / len(tasks)
        avg_loss.backward()
        meta_optimizer.step()

        return {
            "meta_loss": avg_loss.item(),
            "meta_accuracy": total_acc / len(tasks),
        }


class PAdicTaskSampler:
    """Sample tasks based on p-adic hierarchical structure.

    Uses p-adic distance to create tasks where:
    - Support and query samples are p-adically close
    - Different tasks cover different p-adic neighborhoods

    This leverages biological hierarchy (taxonomies, gene families)
    for more meaningful task construction.
    """

    def __init__(
        self,
        data_x: torch.Tensor,
        data_y: torch.Tensor,
        padic_indices: torch.Tensor,
        n_support: int = 5,
        n_query: int = 15,
        prime: int = 3,
        valuation_threshold: int = 2,
    ):
        """Initialize sampler.

        Args:
            data_x: All input data
            data_y: All labels
            padic_indices: P-adic index for each sample
            n_support: Number of support samples per task
            n_query: Number of query samples per task
            prime: Prime for p-adic valuation
            valuation_threshold: Minimum valuation for same task
        """
        self.data_x = data_x
        self.data_y = data_y
        self.padic_indices = padic_indices
        self.n_support = n_support
        self.n_query = n_query
        self.prime = prime
        self.valuation_threshold = valuation_threshold
        self.max_valuation = 9

        # Precompute p-adic neighborhoods
        self._build_neighborhoods()

    def _compute_valuation(self, n: int) -> int:
        """Compute p-adic valuation of n."""
        if n == 0:
            return self.max_valuation
        v = 0
        while n % self.prime == 0:
            n //= self.prime
            v += 1
        return min(v, self.max_valuation)

    def _build_neighborhoods(self):
        """Build p-adic neighborhoods for efficient sampling."""
        n_samples = len(self.padic_indices)
        self.neighborhoods = {}

        for i in range(n_samples):
            idx_i = self.padic_indices[i].item()

            # Find samples in same p-adic neighborhood
            neighbors = []
            for j in range(n_samples):
                if i == j:
                    continue
                idx_j = self.padic_indices[j].item()
                v = self._compute_valuation(abs(idx_i - idx_j))
                if v >= self.valuation_threshold:
                    neighbors.append(j)

            self.neighborhoods[i] = neighbors

    def sample_task(self) -> Task:
        """Sample a single task based on p-adic structure."""
        # Select anchor sample
        anchor_idx = torch.randint(len(self.data_x), (1,)).item()

        # Get p-adic neighbors
        neighbors = self.neighborhoods.get(anchor_idx, [])

        if len(neighbors) < self.n_support + self.n_query:
            # Fall back to random sampling
            all_indices = list(range(len(self.data_x)))
            all_indices.remove(anchor_idx)
            selected = torch.tensor(all_indices)[torch.randperm(len(all_indices))[: self.n_support + self.n_query]]
        else:
            # Sample from p-adic neighbors
            neighbor_tensor = torch.tensor(neighbors)
            perm = torch.randperm(len(neighbors))
            selected = neighbor_tensor[perm[: self.n_support + self.n_query]]

        # Split into support and query
        support_idx = selected[: self.n_support]
        query_idx = selected[self.n_support : self.n_support + self.n_query]

        return Task(
            support_x=self.data_x[support_idx],
            support_y=self.data_y[support_idx],
            query_x=self.data_x[query_idx],
            query_y=self.data_y[query_idx],
            task_id=anchor_idx,
            metadata={"anchor_padic_idx": self.padic_indices[anchor_idx].item()},
        )

    def sample_batch(self, n_tasks: int) -> list[Task]:
        """Sample a batch of tasks."""
        return [self.sample_task() for _ in range(n_tasks)]


class FewShotAdapter(nn.Module):
    """Quick adaptation module for few-shot learning.

    Combines MAML-style adaptation with prototypical learning
    for efficient few-shot inference.
    """

    def __init__(
        self,
        encoder: nn.Module,
        prototype_dim: int = 128,
        n_adapt_steps: int = 3,
        adapt_lr: float = 0.1,
    ):
        """Initialize adapter.

        Args:
            encoder: Feature extractor
            prototype_dim: Dimension of prototype space
            n_adapt_steps: Number of adaptation steps
            adapt_lr: Adaptation learning rate
        """
        super().__init__()
        self.encoder = encoder
        self.prototype_dim = prototype_dim
        self.n_adapt_steps = n_adapt_steps
        self.adapt_lr = adapt_lr

        # Adaptation layer
        self.adapt_layer = nn.Linear(prototype_dim, prototype_dim)

    def compute_prototypes(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> torch.Tensor:
        """Compute class prototypes from support set.

        Args:
            support_x: Support set inputs
            support_y: Support set labels

        Returns:
            Prototypes tensor of shape (n_classes, prototype_dim)
        """
        # Encode support set
        embeddings = self.encoder(support_x)

        # Compute mean embedding per class
        classes = torch.unique(support_y)
        prototypes = []

        for c in classes:
            mask = support_y == c
            class_embeddings = embeddings[mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)

        return torch.stack(prototypes)

    def forward(
        self,
        query_x: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Classify query set using prototypes.

        Args:
            query_x: Query inputs
            prototypes: Class prototypes

        Returns:
            Class logits
        """
        # Encode query
        query_embeddings = self.encoder(query_x)

        # Adapt embeddings
        query_embeddings = self.adapt_layer(query_embeddings)

        # Compute distances to prototypes
        dists = torch.cdist(query_embeddings, prototypes)

        # Return negative distances as logits (closer = higher logit)
        return -dists

    def adapt_and_predict(
        self,
        task: Task,
    ) -> torch.Tensor:
        """Adapt to task and predict query labels.

        Args:
            task: Task with support and query sets

        Returns:
            Query set predictions
        """
        # Quick adaptation on support set
        for _ in range(self.n_adapt_steps):
            prototypes = self.compute_prototypes(task.support_x, task.support_y)
            logits = self.forward(task.support_x, prototypes)
            loss = F.cross_entropy(logits, task.support_y)

            # Update adaptation layer
            grad = torch.autograd.grad(loss, self.adapt_layer.parameters())
            for param, g in zip(self.adapt_layer.parameters(), grad, strict=False):
                param.data = param.data - self.adapt_lr * g

        # Predict on query set
        prototypes = self.compute_prototypes(task.support_x, task.support_y)
        logits = self.forward(task.query_x, prototypes)

        return logits.argmax(dim=-1)


class Reptile(nn.Module):
    """Reptile meta-learning algorithm.

    Simpler than MAML: just moves initialization toward
    adapted parameters, without computing meta-gradients.

    Algorithm:
    1. Sample task
    2. Train on task for k steps
    3. Move initialization toward final parameters:
       θ = θ + ε(θ' - θ)
    """

    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        n_inner_steps: int = 10,
        meta_step_size: float = 0.1,
    ):
        """Initialize Reptile.

        Args:
            model: Base model
            inner_lr: Inner loop learning rate
            n_inner_steps: Steps per task
            meta_step_size: Step size for meta-update
        """
        super().__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.n_inner_steps = n_inner_steps
        self.meta_step_size = meta_step_size

    def train_on_task(
        self,
        task: Task,
        loss_fn: Callable | None = None,
    ) -> dict[str, torch.Tensor]:
        """Train model on a single task.

        Returns the difference between initial and final parameters.
        """
        if loss_fn is None:
            loss_fn = F.cross_entropy

        # Store initial parameters
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}

        # Create optimizer for inner loop
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)

        # Train for k steps
        for _ in range(self.n_inner_steps):
            optimizer.zero_grad()
            output = self.model(task.support_x)
            loss = loss_fn(output, task.support_y)
            loss.backward()
            optimizer.step()

        # Compute parameter differences
        param_diffs = {}
        for name, param in self.model.named_parameters():
            param_diffs[name] = param.data - initial_params[name]

        # Restore initial parameters
        for name, param in self.model.named_parameters():
            param.data = initial_params[name]

        return param_diffs

    def meta_step(
        self,
        tasks: list[Task],
        loss_fn: Callable | None = None,
    ) -> dict[str, float]:
        """Perform meta-update on batch of tasks."""
        if loss_fn is None:
            loss_fn = F.cross_entropy

        # Accumulate parameter differences
        accumulated_diffs = None

        for task in tasks:
            diffs = self.train_on_task(task, loss_fn)

            if accumulated_diffs is None:
                accumulated_diffs = {k: v.clone() for k, v in diffs.items()}
            else:
                for k in accumulated_diffs:
                    accumulated_diffs[k] += diffs[k]

        # Average and apply
        for name, param in self.model.named_parameters():
            avg_diff = accumulated_diffs[name] / len(tasks)
            param.data += self.meta_step_size * avg_diff

        # Evaluate on query sets
        total_loss = 0.0
        total_acc = 0.0
        for task in tasks:
            with torch.no_grad():
                output = self.model(task.query_x)
                loss = loss_fn(output, task.query_y)
                total_loss += loss.item()
                acc = (output.argmax(dim=-1) == task.query_y).float().mean()
                total_acc += acc.item()

        return {
            "meta_loss": total_loss / len(tasks),
            "meta_accuracy": total_acc / len(tasks),
        }
