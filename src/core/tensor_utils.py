# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Tensor utilities - Centralized tensor operations.

This module provides optimized tensor manipulation patterns used across
the codebase, eliminating duplicate implementations.

Consolidated from:
- src/core/padic_math.py (padic_distance_batch broadcasting)
- src/utils/padic_shift.py (PAdicSequenceEncoder.compute_distances)
- src/contrastive/padic_contrastive.py (positive pair indexing)

Key Features:
- Pairwise broadcasting for distance matrices
- Batch indexing into precomputed matrices
- Safe normalization and clamping
- GPU-optimized operations

Usage:
    from src.core.tensor_utils import (
        pairwise_broadcast,
        batch_index_select,
        safe_normalize,
        clamp_norm,
    )

Examples:
    # Compute pairwise distances efficiently
    i_idx, j_idx = pairwise_broadcast(indices, seq_len)
    distances = batch_index_select(distance_matrix, i_idx, j_idx, batch_shape)

References:
    - PyTorch Broadcasting Semantics
    - Einstein Summation Conventions
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.config.constants import EPSILON, EPSILON_NORM

# ============================================================================
# Pairwise Operations
# ============================================================================


def pairwise_broadcast(
    indices: torch.Tensor,
    expand_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create pairwise broadcasted index tensors.

    Creates expanded index tensors for computing all pairwise operations
    efficiently. Replaces the common pattern:
        i_idx = indices.unsqueeze(2).expand(-1, -1, seq_len)
        j_idx = indices.unsqueeze(1).expand(-1, seq_len, -1)

    Args:
        indices: Input indices tensor of shape (batch, seq_len) or (seq_len,)
        expand_size: Size to expand to. If None, uses last dimension of indices.

    Returns:
        Tuple of (i_indices, j_indices) both of shape:
        - (batch, seq_len, seq_len) if input is 2D
        - (seq_len, seq_len) if input is 1D

    Examples:
        >>> indices = torch.tensor([[0, 1, 2], [3, 4, 5]])
        >>> i_idx, j_idx = pairwise_broadcast(indices)
        >>> i_idx.shape
        torch.Size([2, 3, 3])
    """
    if indices.dim() == 1:
        seq_len = len(indices)
        expand_size = expand_size or seq_len
        i_idx = indices.unsqueeze(1).expand(seq_len, expand_size)
        j_idx = indices.unsqueeze(0).expand(expand_size, seq_len)
        return i_idx, j_idx

    # 2D case: (batch, seq_len)
    batch_size, seq_len = indices.shape
    expand_size = expand_size or seq_len

    i_idx = indices.unsqueeze(2).expand(batch_size, seq_len, expand_size)
    j_idx = indices.unsqueeze(1).expand(batch_size, expand_size, seq_len)

    return i_idx, j_idx


def pairwise_difference(
    tensor: torch.Tensor,
    absolute: bool = True,
) -> torch.Tensor:
    """Compute all pairwise differences.

    Args:
        tensor: Input tensor of shape (..., n)
        absolute: If True, return absolute differences

    Returns:
        Difference matrix of shape (..., n, n)

    Examples:
        >>> x = torch.tensor([1, 2, 4])
        >>> pairwise_difference(x)
        tensor([[0, 1, 3],
                [1, 0, 2],
                [3, 2, 0]])
    """
    diff = tensor.unsqueeze(-1) - tensor.unsqueeze(-2)
    if absolute:
        diff = diff.abs()
    return diff


def batch_index_select(
    matrix: torch.Tensor,
    row_indices: torch.Tensor,
    col_indices: torch.Tensor,
    output_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Efficiently select elements from 2D matrix using batch indices.

    Replaces the common pattern:
        flat_i = i_idx.reshape(-1)
        flat_j = j_idx.reshape(-1)
        result = matrix[flat_i, flat_j].reshape(batch, seq, seq)

    Args:
        matrix: 2D matrix to index into, shape (M, N)
        row_indices: Row indices, any shape
        col_indices: Column indices, same shape as row_indices
        output_shape: Desired output shape. If None, uses row_indices.shape.

    Returns:
        Selected elements reshaped to output_shape

    Examples:
        >>> matrix = torch.randn(64, 64)
        >>> rows = torch.randint(0, 64, (8, 10, 10))
        >>> cols = torch.randint(0, 64, (8, 10, 10))
        >>> result = batch_index_select(matrix, rows, cols)
        >>> result.shape
        torch.Size([8, 10, 10])
    """
    original_shape = row_indices.shape
    output_shape = output_shape or original_shape

    # Flatten indices
    flat_rows = row_indices.reshape(-1).long()
    flat_cols = col_indices.reshape(-1).long()

    # Index and reshape
    selected = matrix[flat_rows, flat_cols]
    return selected.reshape(output_shape)


# ============================================================================
# Normalization Operations
# ============================================================================


def safe_normalize(
    tensor: torch.Tensor,
    dim: int = -1,
    eps: float = EPSILON_NORM,
) -> torch.Tensor:
    """Safely normalize tensor to unit norm.

    Handles zero-norm vectors by returning zero vector instead of NaN.

    Args:
        tensor: Input tensor
        dim: Dimension to normalize along
        eps: Small value to prevent division by zero

    Returns:
        Normalized tensor with unit norm along dim

    Examples:
        >>> x = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
        >>> safe_normalize(x, dim=-1)
        tensor([[0.6, 0.8],
                [0.0, 0.0]])
    """
    norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
    return tensor / (norm + eps)


def safe_normalize_l1(
    tensor: torch.Tensor,
    dim: int = -1,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Safely normalize tensor to L1 unit norm (sum to 1).

    Args:
        tensor: Input tensor
        dim: Dimension to normalize along
        eps: Small value to prevent division by zero

    Returns:
        Normalized tensor with L1 norm = 1 along dim
    """
    l1_norm = tensor.abs().sum(dim=dim, keepdim=True)
    return tensor / (l1_norm + eps)


def clamp_norm(
    tensor: torch.Tensor,
    max_norm: float,
    dim: int = -1,
    eps: float = EPSILON_NORM,
) -> torch.Tensor:
    """Clamp tensor norm to maximum value.

    Scales down vectors whose norm exceeds max_norm while preserving direction.

    Args:
        tensor: Input tensor
        max_norm: Maximum allowed norm
        dim: Dimension to compute norm along
        eps: Small value for numerical stability

    Returns:
        Tensor with norms clamped to max_norm

    Examples:
        >>> x = torch.tensor([[3.0, 4.0]])  # norm = 5
        >>> clamp_norm(x, max_norm=1.0)
        tensor([[0.6, 0.8]])  # norm = 1
    """
    norm = torch.norm(tensor, p=2, dim=dim, keepdim=True)
    scale = torch.where(
        norm > max_norm,
        max_norm / (norm + eps),
        torch.ones_like(norm),
    )
    return tensor * scale


def soft_clamp(
    tensor: torch.Tensor,
    min_val: float,
    max_val: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Soft clamping using sigmoid for differentiable bounds.

    Args:
        tensor: Input tensor
        min_val: Soft minimum value
        max_val: Soft maximum value
        temperature: Controls sharpness of clamping

    Returns:
        Soft-clamped tensor
    """
    range_val = max_val - min_val
    normalized = (tensor - min_val) / (range_val + EPSILON)
    clamped = torch.sigmoid(normalized / temperature)
    return min_val + clamped * range_val


# ============================================================================
# Masking Operations
# ============================================================================


def create_causal_mask(
    size: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Create causal (lower triangular) attention mask.

    Args:
        size: Sequence length
        device: Target device
        dtype: Output dtype (bool for mask, float for additive)

    Returns:
        Causal mask of shape (size, size)
    """
    mask = torch.tril(torch.ones(size, size, device=device, dtype=dtype))
    return mask


def create_padding_mask(
    lengths: torch.Tensor,
    max_length: int | None = None,
) -> torch.Tensor:
    """Create padding mask from sequence lengths.

    Args:
        lengths: Tensor of sequence lengths, shape (batch,)
        max_length: Maximum sequence length. If None, uses max(lengths).

    Returns:
        Boolean mask of shape (batch, max_length) where True = valid position
    """
    max_len = max_length or lengths.max().item()
    positions = torch.arange(max_len, device=lengths.device)
    mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
    return mask


def apply_mask(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Apply boolean mask to tensor.

    Args:
        tensor: Input tensor
        mask: Boolean mask (True = keep, False = replace)
        fill_value: Value to use for masked positions

    Returns:
        Masked tensor
    """
    return torch.where(mask, tensor, torch.full_like(tensor, fill_value))


# ============================================================================
# Reduction Operations
# ============================================================================


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    eps: float = EPSILON,
) -> torch.Tensor:
    """Compute mean over masked values.

    Args:
        tensor: Input tensor
        mask: Boolean mask (True = include in mean)
        dim: Dimension to reduce
        keepdim: Keep reduced dimension
        eps: Small value to prevent division by zero

    Returns:
        Masked mean
    """
    masked_tensor = tensor * mask.float()
    sum_val = masked_tensor.sum(dim=dim, keepdim=keepdim)
    count = mask.float().sum(dim=dim, keepdim=keepdim)
    return sum_val / (count + eps)


def masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """Compute softmax over masked positions.

    Sets masked positions to -inf before softmax so they get zero probability.

    Args:
        logits: Input logits
        mask: Boolean mask (True = include in softmax)
        dim: Dimension to apply softmax

    Returns:
        Softmax probabilities (masked positions = 0)
    """
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    return F.softmax(masked_logits, dim=dim)


# ============================================================================
# Gather/Scatter Operations
# ============================================================================


def gather_from_indices(
    source: torch.Tensor,
    indices: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """Gather values from source tensor using indices.

    Wrapper around torch.gather with automatic index expansion.

    Args:
        source: Source tensor
        indices: Index tensor
        dim: Dimension to gather from

    Returns:
        Gathered tensor
    """
    # Expand indices to match source dimensions
    expanded_indices = indices
    for d in range(source.dim()):
        if d != dim:
            expanded_indices = expanded_indices.unsqueeze(d)

    # Expand to match source shape in non-gather dimensions
    expand_shape = list(source.shape)
    expand_shape[dim] = indices.shape[0] if indices.dim() == 1 else indices.shape[dim]
    expanded_indices = expanded_indices.expand(*expand_shape)

    return torch.gather(source, dim, expanded_indices)


def scatter_mean(
    values: torch.Tensor,
    indices: torch.Tensor,
    dim_size: int,
    dim: int = 0,
) -> torch.Tensor:
    """Scatter values and compute mean for each index.

    Args:
        values: Values to scatter
        indices: Target indices for each value
        dim_size: Size of output dimension
        dim: Dimension to scatter to

    Returns:
        Tensor with scattered means
    """
    # Create output tensor and count tensor
    output = torch.zeros(dim_size, *values.shape[1:], device=values.device)
    counts = torch.zeros(dim_size, device=values.device)

    # Scatter sum and count
    output.scatter_add_(dim, indices.unsqueeze(-1).expand_as(values), values)
    counts.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float))

    # Compute mean (avoiding division by zero)
    counts = counts.clamp(min=1)
    output = output / counts.unsqueeze(-1)

    return output


# ============================================================================
# Shape Utilities
# ============================================================================


def flatten_batch(
    tensor: torch.Tensor,
    start_dim: int = 0,
    end_dim: int = 1,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Flatten batch dimensions, returning original shape for restoration.

    Args:
        tensor: Input tensor
        start_dim: First dimension to flatten
        end_dim: Last dimension to flatten (inclusive)

    Returns:
        Tuple of (flattened tensor, original shape tuple for restoration)
    """
    original_shape = tensor.shape
    flattened = tensor.flatten(start_dim, end_dim)
    return flattened, tuple(original_shape[start_dim : end_dim + 1])


def unflatten_batch(
    tensor: torch.Tensor,
    dim: int,
    original_shape: tuple[int, ...],
) -> torch.Tensor:
    """Restore original batch dimensions.

    Args:
        tensor: Flattened tensor
        dim: Dimension to unflatten
        original_shape: Original shape tuple from flatten_batch

    Returns:
        Unflattened tensor
    """
    return tensor.unflatten(dim, original_shape)


def ensure_4d(
    tensor: torch.Tensor,
    channel_dim: int = 1,
) -> torch.Tensor:
    """Ensure tensor is 4D for conv operations.

    Args:
        tensor: Input tensor (2D, 3D, or 4D)
        channel_dim: Position for channel dimension if adding

    Returns:
        4D tensor (batch, channels, height, width)
    """
    if tensor.dim() == 2:
        # (batch, features) -> (batch, 1, 1, features)
        return tensor.unsqueeze(1).unsqueeze(1)
    elif tensor.dim() == 3:
        # (batch, seq, features) -> (batch, 1, seq, features)
        return tensor.unsqueeze(channel_dim)
    elif tensor.dim() == 4:
        return tensor
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got {tensor.dim()}D")


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    # Pairwise operations
    "pairwise_broadcast",
    "pairwise_difference",
    "batch_index_select",
    # Normalization
    "safe_normalize",
    "safe_normalize_l1",
    "clamp_norm",
    "soft_clamp",
    # Masking
    "create_causal_mask",
    "create_padding_mask",
    "apply_mask",
    # Reduction
    "masked_mean",
    "masked_softmax",
    # Gather/Scatter
    "gather_from_indices",
    "scatter_mean",
    # Shape utilities
    "flatten_batch",
    "unflatten_batch",
    "ensure_4d",
]
