# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Task-specific heads for multi-task learning.

Provides specialized output heads for different prediction tasks
with support for cross-task information sharing.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskHead(nn.Module):
    """Base class for task-specific heads.

    Provides common interface for prediction heads.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] = None,
        dropout: float = 0.1,
    ):
        """Initialize task head.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        if hidden_dims is None:
            hidden_dims = [128, 64]
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features

        Returns:
            Task predictions
        """
        features = self.backbone(x)
        return self.output_layer(features)


class RegressionHead(TaskHead):
    """Head for regression tasks.

    Predicts continuous values with optional learned variance.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        learn_variance: bool = True,
        **kwargs,
    ):
        """Initialize regression head.

        Args:
            input_dim: Input dimension
            output_dim: Number of regression targets
            learn_variance: Learn heteroscedastic variance
            **kwargs: Arguments for TaskHead
        """
        super().__init__(input_dim, output_dim, **kwargs)
        self.learn_variance = learn_variance

        if learn_variance:
            self.variance_layer = nn.Linear(
                kwargs.get("hidden_dims", [128, 64])[-1],
                output_dim,
            )

    def forward(
        self,
        x: torch.Tensor,
        return_variance: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with optional variance.

        Args:
            x: Input features
            return_variance: Return learned variance

        Returns:
            Predictions and optionally variance
        """
        features = self.backbone(x)
        mean = self.output_layer(features)

        if return_variance and self.learn_variance:
            log_var = self.variance_layer(features)
            variance = torch.exp(log_var)
            return mean, variance

        return mean, None

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        variance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute regression loss.

        Args:
            predictions: Model predictions
            targets: Ground truth
            variance: Learned variance (optional)

        Returns:
            Loss value
        """
        if variance is not None:
            # Negative log-likelihood with learned variance
            nll = 0.5 * (torch.log(variance + 1e-8) + (targets - predictions) ** 2 / (variance + 1e-8))
            return nll.mean()
        else:
            return F.mse_loss(predictions, targets)


class ClassificationHead(TaskHead):
    """Head for classification tasks.

    Supports binary, multi-class, and multi-label classification.
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        task_type: str = "multiclass",
        class_weights: torch.Tensor | None = None,
        **kwargs,
    ):
        """Initialize classification head.

        Args:
            input_dim: Input dimension
            n_classes: Number of classes
            task_type: 'binary', 'multiclass', or 'multilabel'
            class_weights: Optional class weights for imbalanced data
            **kwargs: Arguments for TaskHead
        """
        output_dim = 1 if task_type == "binary" else n_classes
        super().__init__(input_dim, output_dim, **kwargs)

        self.n_classes = n_classes
        self.task_type = task_type

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        x: torch.Tensor,
        return_probs: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features
            return_probs: Return probabilities instead of logits

        Returns:
            Logits or probabilities
        """
        logits = super().forward(x)

        if return_probs:
            if self.task_type == "binary":
                return torch.sigmoid(logits)
            elif self.task_type == "multiclass":
                return F.softmax(logits, dim=-1)
            else:  # multilabel
                return torch.sigmoid(logits)

        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classification loss.

        Args:
            logits: Model logits
            targets: Ground truth labels

        Returns:
            Loss value
        """
        if self.task_type == "binary":
            return F.binary_cross_entropy_with_logits(
                logits.squeeze(-1),
                targets.float(),
                weight=self.class_weights,
            )
        elif self.task_type == "multiclass":
            return F.cross_entropy(
                logits,
                targets.long(),
                weight=self.class_weights,
            )
        else:  # multilabel
            return F.binary_cross_entropy_with_logits(
                logits,
                targets.float(),
                weight=self.class_weights,
            )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions.

        Args:
            x: Input features

        Returns:
            Predicted classes
        """
        logits = self.forward(x)

        if self.task_type == "binary":
            return (torch.sigmoid(logits) > 0.5).long().squeeze(-1)
        elif self.task_type == "multiclass":
            return logits.argmax(dim=-1)
        else:  # multilabel
            return (torch.sigmoid(logits) > 0.5).long()


class DrugResistanceHead(nn.Module):
    """Specialized head for drug resistance prediction.

    Handles multiple drugs with shared and drug-specific components.
    """

    def __init__(
        self,
        input_dim: int,
        n_drugs: int = 18,
        drug_embedding_dim: int = 32,
        hidden_dims: list[int] = None,
        drug_names: list[str] | None = None,
    ):
        """Initialize drug resistance head.

        Args:
            input_dim: Input feature dimension
            n_drugs: Number of drugs
            drug_embedding_dim: Drug embedding dimension
            hidden_dims: Hidden layer dimensions
            drug_names: Optional list of drug names
        """
        if hidden_dims is None:
            hidden_dims = [256, 128]
        super().__init__()
        self.n_drugs = n_drugs
        self.drug_names = drug_names or [f"Drug_{i}" for i in range(n_drugs)]

        # Drug embeddings (learned)
        self.drug_embeddings = nn.Embedding(n_drugs, drug_embedding_dim)

        # Shared encoder
        shared_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims[:-1]:
            shared_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            prev_dim = hidden_dim

        self.shared_encoder = nn.Sequential(*shared_layers)

        # Drug-specific layers
        self.drug_specific = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(prev_dim + drug_embedding_dim, hidden_dims[-1]),
                    nn.LayerNorm(hidden_dims[-1]),
                    nn.SiLU(),
                    nn.Linear(hidden_dims[-1], 1),
                )
                for _ in range(n_drugs)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        drug_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict resistance for drugs.

        Args:
            x: Sequence features (batch, input_dim)
            drug_indices: Which drugs to predict (None = all)

        Returns:
            Resistance predictions (batch, n_drugs) or (batch, len(drug_indices))
        """
        # Shared encoding
        shared = self.shared_encoder(x)

        if drug_indices is None:
            # Predict all drugs
            outputs = []
            for i, drug_layer in enumerate(self.drug_specific):
                drug_emb = self.drug_embeddings(torch.tensor([i], device=x.device)).expand(x.shape[0], -1)
                combined = torch.cat([shared, drug_emb], dim=-1)
                outputs.append(drug_layer(combined))

            return torch.cat(outputs, dim=-1)
        else:
            # Predict specific drugs
            outputs = []
            for idx in drug_indices:
                drug_emb = self.drug_embeddings(torch.tensor([idx], device=x.device)).expand(x.shape[0], -1)
                combined = torch.cat([shared, drug_emb], dim=-1)
                outputs.append(self.drug_specific[idx](combined))

            return torch.cat(outputs, dim=-1)

    def get_drug_correlations(self) -> torch.Tensor:
        """Compute drug similarity from embeddings.

        Returns:
            (n_drugs, n_drugs) correlation matrix
        """
        with torch.no_grad():
            all_emb = self.drug_embeddings.weight
            # Cosine similarity
            norm = all_emb.norm(dim=-1, keepdim=True)
            normed = all_emb / (norm + 1e-8)
            return torch.mm(normed, normed.t())


class CrossTaskAttention(nn.Module):
    """Cross-task attention for information sharing.

    Allows tasks to attend to each other's representations
    for improved multi-task learning.

    References:
        - Liu et al. (2019): Multi-Task Learning as Multi-Objective Optimization
    """

    def __init__(
        self,
        task_dim: int,
        n_tasks: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize cross-task attention.

        Args:
            task_dim: Dimension of task representations
            n_tasks: Number of tasks
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.n_tasks = n_tasks
        self.n_heads = n_heads
        self.head_dim = task_dim // n_heads

        assert task_dim % n_heads == 0, "task_dim must be divisible by n_heads"

        # Query, Key, Value projections for each task
        self.q_proj = nn.Linear(task_dim, task_dim)
        self.k_proj = nn.Linear(task_dim, task_dim)
        self.v_proj = nn.Linear(task_dim, task_dim)

        # Output projection
        self.out_proj = nn.Linear(task_dim, task_dim)

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(task_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        task_representations: torch.Tensor,
        task_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply cross-task attention.

        Args:
            task_representations: (batch, n_tasks, task_dim) task features
            task_mask: Optional mask for task availability

        Returns:
            Updated task representations with cross-task info
        """
        batch_size = task_representations.shape[0]
        residual = task_representations

        # Project Q, K, V
        Q = self.q_proj(task_representations)
        K = self.k_proj(task_representations)
        V = self.v_proj(task_representations)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.n_tasks, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, self.n_tasks, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, self.n_tasks, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        if task_mask is not None:
            scores = scores.masked_fill(~task_mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        context = torch.matmul(attn, V)

        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(batch_size, self.n_tasks, -1)
        output = self.out_proj(context)

        # Residual connection and layer norm
        return self.layer_norm(residual + self.dropout(output))


class TaskRouter(nn.Module):
    """Soft routing between tasks for mixture-of-experts style MTL.

    Learns to route shared representations to different expert pathways
    based on input features.
    """

    def __init__(
        self,
        input_dim: int,
        n_experts: int,
        n_tasks: int,
        top_k: int = 2,
    ):
        """Initialize task router.

        Args:
            input_dim: Input feature dimension
            n_experts: Number of expert networks
            n_tasks: Number of tasks
            top_k: Number of experts to route to
        """
        super().__init__()
        self.n_experts = n_experts
        self.n_tasks = n_tasks
        self.top_k = top_k

        # Router network
        self.router = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, n_experts),
        )

        # Load balancing loss coefficient
        self.balance_coefficient = 0.01

    def forward(
        self,
        x: torch.Tensor,
        task_id: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute routing weights.

        Args:
            x: Input features (batch, input_dim)
            task_id: Optional task identifier for task-specific routing

        Returns:
            Tuple of (routing_weights, expert_indices, load_balance_loss)
        """
        # Compute router logits
        router_logits = self.router(x)

        # Add noise during training for exploration
        if self.training:
            noise = torch.randn_like(router_logits) * 0.1
            router_logits = router_logits + noise

        # Top-k selection
        top_k_logits, top_k_indices = router_logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)

        # Load balancing loss (encourages even expert usage)
        probs = F.softmax(router_logits, dim=-1)
        avg_probs = probs.mean(dim=0)
        target = torch.ones_like(avg_probs) / self.n_experts
        balance_loss = self.balance_coefficient * F.mse_loss(avg_probs, target)

        return top_k_weights, top_k_indices, balance_loss
