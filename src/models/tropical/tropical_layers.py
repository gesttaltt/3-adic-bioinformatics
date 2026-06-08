# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tropical Neural Network Layers.

Implements neural network layers using tropical (max-plus) algebra:
- TropicalLinear: y_i = max_j(W_ij + x_j) + b_i
- TropicalConv1d: Tropical convolution
- TropicalLayerNorm: Normalization for tropical features

These layers naturally produce piecewise linear functions and
are well-suited for learning tree-like hierarchical structures.

Mathematical Properties:
1. Tropical matrix multiplication is associative
2. The result is always piecewise linear
3. Number of linear regions grows polynomially with depth
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TropicalLinear(nn.Module):
    """Tropical linear layer using max-plus algebra.

    Standard linear: y = Wx + b (matrix multiply)
    Tropical linear: y_i = max_j(W_ij + x_j) + b_i (max-plus)

    Properties:
    - Output is piecewise linear in input
    - Naturally encodes tree-like structures
    - Equivalent to max-pooling over linear functions
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        temperature: float = 1.0,
        soft_tropical: bool = True,
    ):
        """Initialize tropical linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to include bias
            temperature: Temperature for soft tropical (softmax approximation)
            soft_tropical: If True, use logsumexp (differentiable); else hard max
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.temperature = temperature
        self.soft_tropical = soft_tropical

        # Weight matrix (added to inputs, not multiplied)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters for tropical operations."""
        # Initialize weights with small values (since they're added)
        nn.init.normal_(self.weight, mean=0, std=0.1)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using tropical multiplication.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            Output tensor (..., out_features)
        """
        # Tropical matrix-vector multiplication:
        # y_i = max_j(W_ij + x_j)

        # Expand for broadcasting
        # x: (..., in_features) -> (..., 1, in_features)
        # W: (out_features, in_features)
        x_expanded = x.unsqueeze(-2)  # (..., 1, in_features)

        # Add weights to inputs
        # tropical_prod[..., i, j] = W[i, j] + x[j]
        tropical_prod = self.weight + x_expanded  # (..., out_features, in_features)

        if self.soft_tropical:
            # Soft tropical: logsumexp approximation to max
            # logsumexp(x/T) * T ≈ max(x) as T → 0
            y = self.temperature * torch.logsumexp(tropical_prod / self.temperature, dim=-1)
        else:
            # Hard tropical: exact max
            y = tropical_prod.max(dim=-1)[0]

        # Add bias
        if self.bias is not None:
            y = y + self.bias

        return y

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


class TropicalConv1d(nn.Module):
    """Tropical 1D convolution using max-plus algebra.

    Instead of weighted sum, uses max of sum:
    y[i] = max_k(W[k] + x[i+k])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        temperature: float = 1.0,
        soft_tropical: bool = True,
    ):
        """Initialize tropical convolution.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of convolution kernel
            stride: Stride of convolution
            padding: Padding size
            temperature: Temperature for soft tropical
            soft_tropical: Whether to use soft tropical
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.temperature = temperature
        self.soft_tropical = soft_tropical

        # Weight: (out_channels, in_channels, kernel_size)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels))

        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters."""
        nn.init.normal_(self.weight, mean=0, std=0.1)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input (batch, in_channels, length)

        Returns:
            Output (batch, out_channels, new_length)
        """
        batch_size, in_channels, length = x.shape

        # Add padding
        if self.padding > 0:
            x = F.pad(x, (self.padding, self.padding), value=float("-inf"))

        # Unfold to get sliding windows
        # (batch, in_channels, n_windows, kernel_size)
        x_unfolded = x.unfold(2, self.kernel_size, self.stride)
        n_windows = x_unfolded.size(2)

        # Reshape for tropical multiplication
        # x: (batch, 1, in_channels, n_windows, kernel_size)
        # W: (out_channels, in_channels, 1, kernel_size)
        x_exp = x_unfolded.unsqueeze(1)
        w_exp = self.weight.unsqueeze(2)

        # Tropical convolution: add weights to inputs
        tropical_prod = x_exp + w_exp  # (batch, out_channels, in_channels, n_windows, kernel_size)

        if self.soft_tropical:
            # Soft tropical over in_channels and kernel
            y = self.temperature * torch.logsumexp(
                tropical_prod.view(batch_size, self.out_channels, -1, n_windows) / self.temperature,
                dim=2,
            )
        else:
            # Hard tropical
            y = tropical_prod.view(batch_size, self.out_channels, -1, n_windows).max(dim=2)[0]

        # Add bias
        y = y + self.bias.view(1, -1, 1)

        return y


class TropicalLayerNorm(nn.Module):
    """Normalization for tropical features.

    Shifts features to have zero tropical mean (median-like centering).
    """

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
    ):
        """Initialize tropical layer norm.

        Args:
            normalized_shape: Size of normalized dimension
            eps: Epsilon for numerical stability
            elementwise_affine: Whether to include learnable affine
        """
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply tropical normalization.

        Args:
            x: Input tensor (..., normalized_shape)

        Returns:
            Normalized tensor
        """
        # Center by median (tropical mean equivalent)
        median = x.median(dim=-1, keepdim=True)[0]
        centered = x - median

        # Scale by inter-quartile range (robust scale)
        q75 = torch.quantile(x, 0.75, dim=-1, keepdim=True)
        q25 = torch.quantile(x, 0.25, dim=-1, keepdim=True)
        iqr = (q75 - q25).clamp(min=self.eps)

        normalized = centered / iqr

        if self.elementwise_affine:
            normalized = normalized * self.weight + self.bias

        return normalized


class TropicalActivation(nn.Module):
    """Activation functions for tropical networks.

    Several options:
    - Identity: No additional nonlinearity (tropical is already piecewise linear)
    - Tropical ReLU: max(x, threshold)
    - Tropical Sigmoid: Soft thresholding
    """

    def __init__(
        self,
        activation: str = "identity",
        threshold: float = 0.0,
        temperature: float = 1.0,
    ):
        """Initialize tropical activation.

        Args:
            activation: Activation type ('identity', 'relu', 'sigmoid')
            threshold: Threshold for ReLU
            temperature: Temperature for soft activations
        """
        super().__init__()
        self.activation = activation
        self.threshold = threshold
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        if self.activation == "identity":
            return x

        elif self.activation == "relu":
            # Tropical ReLU: max(x, threshold)
            return torch.maximum(x, torch.tensor(self.threshold, device=x.device))

        elif self.activation == "sigmoid":
            # Soft thresholding via sigmoid
            return x * torch.sigmoid((x - self.threshold) / self.temperature)

        elif self.activation == "softplus":
            # Softplus for smooth approximation
            return F.softplus(x - self.threshold) + self.threshold

        else:
            raise ValueError(f"Unknown activation: {self.activation}")


class TropicalMLP(nn.Module):
    """Multi-layer perceptron using tropical layers.

    Stacks tropical linear layers with optional normalization
    and activations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        temperature: float = 1.0,
        soft_tropical: bool = True,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        """Initialize tropical MLP.

        Args:
            input_dim: Input dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            n_layers: Number of hidden layers
            temperature: Temperature for soft tropical
            soft_tropical: Whether to use soft tropical
            activation: Activation type
            dropout: Dropout rate
        """
        super().__init__()

        layers = []

        # Input layer
        layers.append(TropicalLinear(input_dim, hidden_dim, temperature=temperature, soft_tropical=soft_tropical))
        layers.append(TropicalLayerNorm(hidden_dim))
        layers.append(TropicalActivation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(TropicalLinear(hidden_dim, hidden_dim, temperature=temperature, soft_tropical=soft_tropical))
            layers.append(TropicalLayerNorm(hidden_dim))
            layers.append(TropicalActivation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(TropicalLinear(hidden_dim, output_dim, temperature=temperature, soft_tropical=soft_tropical))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.layers(x)


class TropicalAttention(nn.Module):
    """Tropical self-attention mechanism.

    Attention scores computed using tropical operations:
    - Query-key similarity: tropical inner product
    - Softmax replaced with tropical normalization
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        temperature: float = 1.0,
        dropout: float = 0.0,
    ):
        """Initialize tropical attention.

        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            temperature: Temperature for tropical softmax
            dropout: Dropout rate
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.temperature = temperature

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Query, key, value projections (tropical)
        self.q_proj = TropicalLinear(embed_dim, embed_dim, temperature=temperature)
        self.k_proj = TropicalLinear(embed_dim, embed_dim, temperature=temperature)
        self.v_proj = TropicalLinear(embed_dim, embed_dim, temperature=temperature)
        self.out_proj = TropicalLinear(embed_dim, embed_dim, temperature=temperature)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply tropical attention.

        Args:
            x: Input (batch, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Output (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Tropical attention scores: Q + K^T (instead of Q @ K^T)
        # Use broadcasting: q[..., i, :, None] + k[..., j, None, :]
        scores = q.unsqueeze(-1) + k.unsqueeze(-2).transpose(-1, -2)
        scores = scores.sum(dim=-1)  # Sum over head_dim (tropical contraction)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Tropical softmax: logsumexp normalization
        attn_weights = F.softmax(scores / self.temperature, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply to values
        output = torch.matmul(attn_weights, v)

        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        return self.out_proj(output)


__all__ = [
    "TropicalLinear",
    "TropicalConv1d",
    "TropicalLayerNorm",
    "TropicalActivation",
    "TropicalMLP",
    "TropicalAttention",
]
