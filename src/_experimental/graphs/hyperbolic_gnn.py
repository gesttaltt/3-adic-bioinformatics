# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Hyperbolic Graph Neural Networks implementation.

This module implements graph neural networks that operate in hyperbolic
space, which naturally captures hierarchical structure in biological
networks (protein interaction networks, phylogenetic trees, etc.).

Key features:
- Poincare ball model operations
- Lorentz (hyperboloid) model operations
- Hyperbolic message passing
- Multi-scale wavelet decomposition

References:
- Chami et al. (2019): Hyperbolic Graph Convolutional Neural Networks
- Liu et al. (2019): Hyperbolic Graph Neural Networks
- Ganea et al. (2018): Hyperbolic Neural Networks
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Hyperbolic Math Utilities
# =============================================================================


class PoincareOperations:
    """Mathematical operations in the Poincare ball model.

    The Poincare ball B^n_c = {x in R^n : c||x||^2 < 1} is a model
    of hyperbolic space with curvature -c.

    Key operations:
    - Mobius addition
    - Exponential map (tangent space -> manifold)
    - Logarithmic map (manifold -> tangent space)
    - Parallel transport

    Args:
        curvature: Absolute value of negative curvature (default: 1.0)
        eps: Numerical stability constant (default: 1e-5)

    Example:
        >>> poincare = PoincareOperations(curvature=1.0)
        >>> x = torch.randn(32, 16) * 0.1  # Points near origin
        >>> y = torch.randn(32, 16) * 0.1
        >>> z = poincare.mobius_add(x, y)  # Hyperbolic addition
        >>> dist = poincare.distance(x, y)  # Hyperbolic distance
    """

    def __init__(self, curvature: float = 1.0, eps: float = 1e-5):
        """Initialize Poincare operations.

        Args:
            curvature: Absolute value of negative curvature
            eps: Numerical stability constant
        """
        self.c = curvature
        self.eps = eps

    def _lambda_x(self, x: torch.Tensor) -> torch.Tensor:
        """Conformal factor lambda_x = 2 / (1 - c||x||^2)."""
        c = self.c
        norm_sq = (x * x).sum(dim=-1, keepdim=True)
        return 2.0 / (1.0 - c * norm_sq).clamp(min=self.eps)

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mobius addition: x + y (hyperbolic).

        The gyrovector addition in the Poincare ball:
        x + y = ((1 + 2c<x,y> + c||y||^2)x + (1 - c||x||^2)y) /
                (1 + 2c<x,y> + c^2||x||^2||y||^2)

        Args:
            x: First point in Poincare ball
            y: Second point in Poincare ball

        Returns:
            Result of Mobius addition
        """
        c = self.c
        x_sq = (x * x).sum(dim=-1, keepdim=True)
        y_sq = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)

        num = (1 + 2 * c * xy + c * y_sq) * x + (1 - c * x_sq) * y
        denom = 1 + 2 * c * xy + c * c * x_sq * y_sq
        return num / denom.clamp(min=self.eps)

    def mobius_scalar(self, r: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Mobius scalar multiplication: r * x (hyperbolic).

        r * x = (1/sqrt(c)) tanh(r * arctanh(sqrt(c) ||x||)) * x / ||x||

        Args:
            r: Scalar multiplier
            x: Point in Poincare ball

        Returns:
            Result of Mobius scalar multiplication
        """
        c = self.c
        sqrt_c = math.sqrt(c)

        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        unit = x / norm

        arctanh_arg = (sqrt_c * norm).clamp(max=1 - self.eps)
        result_norm = (1.0 / sqrt_c) * torch.tanh(r * torch.atanh(arctanh_arg))

        return result_norm * unit

    def exp_map(
        self,
        v: torch.Tensor,
        base: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Exponential map: tangent vector -> manifold point.

        exp_x(v) = x + (tanh(sqrt(c) ||v|| lambda_x / 2) * v / (sqrt(c) ||v||))

        Args:
            v: Tangent vector
            base: Base point (origin if None)

        Returns:
            Point on manifold
        """
        c = self.c
        sqrt_c = math.sqrt(c)

        norm = v.norm(dim=-1, keepdim=True).clamp(min=self.eps)

        if base is None:
            # Exp at origin simplifies
            tanh_arg = sqrt_c * norm
            return torch.tanh(tanh_arg) * v / (sqrt_c * norm)

        lambda_x = self._lambda_x(base)
        tanh_arg = sqrt_c * norm * lambda_x / 2
        direction = torch.tanh(tanh_arg) * v / (sqrt_c * norm)

        return self.mobius_add(base, direction)

    def log_map(
        self,
        y: torch.Tensor,
        base: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Logarithmic map: manifold point -> tangent vector.

        log_x(y) = (2 / (sqrt(c) lambda_x)) arctanh(sqrt(c) ||-x + y||) * (-x + y) / ||-x + y||

        Args:
            y: Point on manifold
            base: Base point (origin if None)

        Returns:
            Tangent vector at base
        """
        c = self.c
        sqrt_c = math.sqrt(c)

        if base is None:
            # Log at origin
            norm = y.norm(dim=-1, keepdim=True).clamp(min=self.eps)
            arctanh_arg = (sqrt_c * norm).clamp(max=1 - self.eps)
            return torch.atanh(arctanh_arg) * y / (sqrt_c * norm)

        # -x in Mobius sense
        neg_base = -base
        diff = self.mobius_add(neg_base, y)
        norm = diff.norm(dim=-1, keepdim=True).clamp(min=self.eps)

        lambda_x = self._lambda_x(base)
        arctanh_arg = (sqrt_c * norm).clamp(max=1 - self.eps)

        return (2.0 / (sqrt_c * lambda_x)) * torch.atanh(arctanh_arg) * diff / norm

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Hyperbolic distance d(x, y).

        d(x, y) = (2/sqrt(c)) arctanh(sqrt(c) ||-x + y||)

        Args:
            x: First point
            y: Second point

        Returns:
            Hyperbolic distance
        """
        c = self.c
        sqrt_c = math.sqrt(c)

        diff = self.mobius_add(-x, y)
        norm = diff.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        arctanh_arg = (sqrt_c * norm).clamp(max=1 - self.eps)

        return (2.0 / sqrt_c) * torch.atanh(arctanh_arg)

    def project(self, x: torch.Tensor, max_norm: float = 0.95) -> torch.Tensor:
        """Project point onto Poincare ball (stay inside boundary).

        Args:
            x: Point to project
            max_norm: Maximum allowed norm (default: 0.95)

        Returns:
            Projected point inside the ball
        """
        c = self.c
        max_r = max_norm / math.sqrt(c)

        norm = x.norm(dim=-1, keepdim=True)
        cond = norm > max_r
        x = torch.where(cond, x * max_r / norm, x)
        return x


class LorentzOperations:
    """Mathematical operations in the Lorentz (hyperboloid) model.

    The Lorentz model represents hyperbolic space as the upper sheet
    of a hyperboloid: H^n = {x in R^{n+1} : <x,x>_L = -1/c, x_0 > 0}
    where <x,y>_L = -x_0 y_0 + sum_{i>0} x_i y_i.

    Args:
        curvature: Absolute value of curvature (default: 1.0)
        eps: Numerical stability constant (default: 1e-5)

    Example:
        >>> lorentz = LorentzOperations(curvature=1.0)
        >>> x = torch.randn(32, 17)  # 16D + 1 time component
        >>> x = lorentz.project_to_hyperboloid(x)
        >>> y = lorentz.project_to_hyperboloid(torch.randn(32, 17))
        >>> dist = lorentz.distance(x, y)
    """

    def __init__(self, curvature: float = 1.0, eps: float = 1e-5):
        """Initialize Lorentz operations.

        Args:
            curvature: Absolute value of curvature
            eps: Numerical stability constant
        """
        self.c = curvature
        self.eps = eps

    def minkowski_inner(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Minkowski inner product <x, y>_L = -x_0 y_0 + sum x_i y_i.

        Args:
            x: First vector
            y: Second vector

        Returns:
            Minkowski inner product
        """
        return -x[..., 0:1] * y[..., 0:1] + (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)

    def minkowski_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Minkowski norm ||x||_L = sqrt(|<x,x>_L|).

        Args:
            x: Vector

        Returns:
            Minkowski norm
        """
        inner = self.minkowski_inner(x, x)
        return inner.abs().clamp(min=self.eps).sqrt()

    def project_to_hyperboloid(self, x: torch.Tensor) -> torch.Tensor:
        """Project point onto hyperboloid.

        Given x = (x_0, x_1, ..., x_n), compute x_0 to satisfy <x,x>_L = -1/c.
        x_0 = sqrt(1/c + sum x_i^2)

        Args:
            x: Point to project

        Returns:
            Point on hyperboloid
        """
        c = self.c
        space_sq = (x[..., 1:] ** 2).sum(dim=-1, keepdim=True)
        x_0 = torch.sqrt((1.0 / c) + space_sq)
        return torch.cat([x_0, x[..., 1:]], dim=-1)

    def exp_map(
        self,
        v: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        """Exponential map on hyperboloid.

        exp_x(v) = cosh(||v||_L) * x + sinh(||v||_L) * v / ||v||_L

        Args:
            v: Tangent vector
            base: Base point on hyperboloid

        Returns:
            Point on hyperboloid
        """
        norm = self.minkowski_norm(v).clamp(min=self.eps)
        direction = v / norm

        return torch.cosh(norm) * base + torch.sinh(norm) * direction

    def log_map(
        self,
        y: torch.Tensor,
        base: torch.Tensor,
    ) -> torch.Tensor:
        """Logarithmic map on hyperboloid.

        log_x(y) = d(x,y) * (y - <x,y>_L * x) / ||y - <x,y>_L * x||_L

        Args:
            y: Target point
            base: Base point

        Returns:
            Tangent vector at base
        """
        inner_xy = self.minkowski_inner(base, y)
        dist = self.distance(base, y)

        numerator = y + inner_xy * base  # Note: <x,y>_L is negative for hyperboloid points
        norm = self.minkowski_norm(numerator).clamp(min=self.eps)

        return dist * numerator / norm

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Hyperbolic distance on hyperboloid.

        d(x, y) = (1/sqrt(c)) * arccosh(-c * <x, y>_L)

        Args:
            x: First point
            y: Second point

        Returns:
            Hyperbolic distance
        """
        c = self.c
        sqrt_c = math.sqrt(c)

        inner = self.minkowski_inner(x, y)
        # Clamp for numerical stability
        arccosh_arg = (-c * inner).clamp(min=1.0)

        return (1.0 / sqrt_c) * torch.acosh(arccosh_arg)

    def parallel_transport(
        self,
        v: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Parallel transport v from tangent space at x to y.

        Args:
            v: Tangent vector at x
            x: Source point
            y: Target point

        Returns:
            Transported vector at y
        """
        log_xy = self.log_map(y, x)
        log_yx = self.log_map(x, y)

        norm_log = self.minkowski_norm(log_xy).clamp(min=self.eps)

        # Transport formula
        inner_v_log = self.minkowski_inner(v, log_xy)
        term1 = v
        term2 = (inner_v_log / norm_log**2) * (log_xy + log_yx)

        return term1 - term2


# =============================================================================
# Hyperbolic Neural Network Layers
# =============================================================================


class HyperbolicLinear(nn.Module):
    """Linear layer operating in hyperbolic space.

    Maps from tangent space, applies linear transformation,
    then maps back to manifold.

    Architecture:
        1. Log map to tangent space at origin
        2. Apply Euclidean linear transformation
        3. Exp map back to manifold
        4. Project to stay inside ball

    Args:
        in_features: Input dimension
        out_features: Output dimension
        curvature: Hyperbolic curvature (default: 1.0)
        bias: Whether to use bias (default: True)

    Example:
        >>> layer = HyperbolicLinear(64, 32, curvature=1.0)
        >>> x = torch.randn(32, 64) * 0.1  # Points in Poincare ball
        >>> out = layer(x)  # Shape: (32, 32)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        curvature: float = 1.0,
        bias: bool = True,
    ):
        """Initialize hyperbolic linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            curvature: Hyperbolic curvature
            bias: Whether to use bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.curvature = curvature

        self.poincare = PoincareOperations(curvature)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor in Poincare ball (batch, in_features)

        Returns:
            Output tensor in Poincare ball (batch, out_features)
        """
        # To tangent space
        v = self.poincare.log_map(x)

        # Linear transform
        v = self.linear(v)

        # Back to manifold
        out = self.poincare.exp_map(v)

        # Project to stay in ball
        return self.poincare.project(out)


class HyperbolicGraphConv(nn.Module):
    """Graph convolution layer in hyperbolic space.

    Performs message passing in the Poincare ball model:
    1. Transform features with hyperbolic linear
    2. Aggregate neighbor features using hyperbolic mean
    3. Update with another hyperbolic linear transformation

    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        curvature: Hyperbolic curvature (default: 1.0)
        use_attention: Whether to use attention mechanism (default: False)
        heads: Number of attention heads (default: 1)
        dropout: Dropout rate (default: 0.0)

    Example:
        >>> conv = HyperbolicGraphConv(64, 64, use_attention=True)
        >>> x = torch.randn(100, 64) * 0.1  # 100 nodes
        >>> edge_index = torch.randint(0, 100, (2, 500))  # 500 edges
        >>> out = conv(x, edge_index)  # Shape: (100, 64)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        curvature: float = 1.0,
        use_attention: bool = False,
        heads: int = 1,
        dropout: float = 0.0,
    ):
        """Initialize hyperbolic graph convolution.

        Args:
            in_channels: Input feature dimension
            out_channels: Output feature dimension
            curvature: Hyperbolic curvature
            use_attention: Whether to use attention mechanism
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.curvature = curvature
        self.use_attention = use_attention
        self.heads = heads

        self.poincare = PoincareOperations(curvature)

        # Message transform
        self.msg_linear = HyperbolicLinear(in_channels, out_channels, curvature)

        # Attention (optional)
        if use_attention:
            self.att_linear = nn.Linear(2 * out_channels, heads)
            self.att_dropout = nn.Dropout(dropout)

        # Update
        self.update_linear = HyperbolicLinear(out_channels, out_channels, curvature)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (n_nodes, in_channels)
            edge_index: Edge indices (2, n_edges)
            edge_attr: Optional edge attributes

        Returns:
            Updated node features (n_nodes, out_channels)
        """
        n_nodes = x.shape[0]
        device = x.device

        # Transform features
        x = self.msg_linear(x)

        # Aggregate messages
        src, dst = edge_index[0], edge_index[1]

        if self.use_attention:
            # Compute attention weights
            edge_feat = torch.cat([x[src], x[dst]], dim=-1)
            att_raw = self.att_linear(edge_feat)
            att_raw = F.leaky_relu(att_raw, 0.2)

            # Softmax over neighbors
            torch.zeros(n_nodes, self.heads, device=device)
            att_sum = torch.zeros(n_nodes, self.heads, device=device)
            for i in range(len(src)):
                att_sum[dst[i]] += torch.exp(att_raw[i])
            att_weights = torch.exp(att_raw) / att_sum[dst].clamp(min=1e-8)
            att_weights = self.att_dropout(att_weights)
            att_weights = att_weights.mean(dim=-1, keepdim=True)  # Average heads
        else:
            # Equal weights
            degree = torch.zeros(n_nodes, 1, device=device)
            for d in dst:
                degree[d] += 1
            degree = degree.clamp(min=1)
            att_weights = 1.0 / degree[dst]

        # Hyperbolic mean aggregation using Frechet mean approximation
        # For simplicity, use weighted tangent space mean
        msg = torch.zeros_like(x)
        count = torch.zeros(n_nodes, 1, device=device)

        for i, (s, d) in enumerate(zip(src, dst, strict=False)):
            # Log map source to tangent space at destination
            v = self.poincare.log_map(x[s], x[d])
            msg[d] = msg[d] + att_weights[i] * v
            count[d] += 1

        # Normalize and exp map
        msg = msg / count.clamp(min=1)

        # Combine with self
        out = torch.zeros_like(x)
        for i in range(n_nodes):
            if count[i] > 0:
                out[i] = self.poincare.exp_map(msg[i], x[i])
            else:
                out[i] = x[i]

        out = self.poincare.project(out)
        out = self.update_linear(out)
        out = self.dropout(out)

        return out


class LorentzMLP(nn.Module):
    """Multi-layer perceptron in Lorentz model.

    Operates on the hyperboloid model of hyperbolic space.

    Args:
        in_features: Input dimension (excluding time component)
        hidden_features: Hidden dimension
        out_features: Output dimension
        curvature: Hyperbolic curvature (default: 1.0)
        n_layers: Number of layers (default: 2)
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> mlp = LorentzMLP(16, 32, 8, n_layers=3)
        >>> x = torch.randn(32, 17)  # 16 features + 1 time component
        >>> out = mlp(x)  # Shape: (32, 9)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        curvature: float = 1.0,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize Lorentz MLP.

        Args:
            in_features: Input dimension (excluding time component)
            hidden_features: Hidden dimension
            out_features: Output dimension
            curvature: Hyperbolic curvature
            n_layers: Number of layers
            dropout: Dropout rate
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.curvature = curvature

        self.lorentz = LorentzOperations(curvature)

        # Build layers
        layers = []
        dims = [in_features] + [hidden_features] * (n_layers - 1) + [out_features]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i] + 1, dims[i + 1] + 1))  # +1 for time component
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        self.layers = nn.ModuleList([layer for layer in layers if isinstance(layer, nn.Module)])
        self.activations = [isinstance(layer, nn.ReLU) for layer in layers]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input on hyperboloid (first component is time)

        Returns:
            Output on hyperboloid
        """
        # Project to ensure on hyperboloid
        x = self.lorentz.project_to_hyperboloid(x)

        # Process through layers
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # Map to tangent space, apply linear, map back
                v = x.clone()
                v[..., 0] = 0  # Zero out time component for tangent vector
                v = layer(x)
                x = self.lorentz.project_to_hyperboloid(v)
            else:
                x = layer(x)

        return x


# =============================================================================
# HyboWaveNet: Multi-scale Wavelet + Hyperbolic GNN
# =============================================================================


class SpectralWavelet(nn.Module):
    """Spectral wavelet decomposition for graphs.

    Computes multi-scale representations using graph wavelets
    based on the graph Laplacian.

    Args:
        n_scales: Number of wavelet scales (default: 4)
        min_scale: Minimum wavelet scale (default: 0.5)
        max_scale: Maximum wavelet scale (default: 4.0)

    Example:
        >>> wavelet = SpectralWavelet(n_scales=4)
        >>> x = torch.randn(100, 64)
        >>> edge_index = torch.randint(0, 100, (2, 500))
        >>> coeffs = wavelet(x, edge_index)  # List of 4 tensors
    """

    def __init__(
        self,
        n_scales: int = 4,
        min_scale: float = 0.5,
        max_scale: float = 4.0,
    ):
        """Initialize spectral wavelet.

        Args:
            n_scales: Number of wavelet scales
            min_scale: Minimum wavelet scale
            max_scale: Maximum wavelet scale
        """
        super().__init__()
        self.n_scales = n_scales

        # Log-spaced scales
        scales = torch.logspace(
            math.log10(min_scale),
            math.log10(max_scale),
            n_scales,
        )
        self.register_buffer("scales", scales)

    def _compute_laplacian(
        self,
        edge_index: torch.Tensor,
        n_nodes: int,
    ) -> torch.Tensor:
        """Compute normalized graph Laplacian.

        Args:
            edge_index: Edge indices (2, n_edges)
            n_nodes: Number of nodes

        Returns:
            Normalized Laplacian matrix (n_nodes, n_nodes)
        """
        device = edge_index.device

        # Degree
        src, dst = edge_index[0], edge_index[1]
        degree = torch.zeros(n_nodes, device=device)
        for d in dst:
            degree[d] += 1
        degree = degree.clamp(min=1)

        # Adjacency
        adj = torch.zeros(n_nodes, n_nodes, device=device)
        for s, d in zip(src, dst, strict=False):
            adj[s, d] = 1
            adj[d, s] = 1

        # Normalized Laplacian: I - D^{-1/2} A D^{-1/2}
        d_inv_sqrt = 1.0 / degree.sqrt()
        d_inv_sqrt = torch.diag(d_inv_sqrt)

        laplacian = torch.eye(n_nodes, device=device) - d_inv_sqrt @ adj @ d_inv_sqrt

        return laplacian

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Compute wavelet coefficients at multiple scales.

        Args:
            x: Node features (n_nodes, features)
            edge_index: Edge indices (2, n_edges)

        Returns:
            List of wavelet coefficients at each scale
        """
        n_nodes = x.shape[0]
        device = x.device

        # Compute Laplacian
        L = self._compute_laplacian(edge_index, n_nodes)

        # Eigendecomposition (for small graphs)
        if n_nodes <= 500:
            eigenvalues, eigenvectors = torch.linalg.eigh(L)
        else:
            # For larger graphs, use approximation
            eigenvalues = torch.ones(n_nodes, device=device)
            eigenvectors = torch.eye(n_nodes, device=device)

        # Compute wavelets at each scale
        wavelets = []
        for scale in self.scales:
            # Heat kernel: exp(-scale * lambda)
            kernel = torch.exp(-scale * eigenvalues)

            # Apply wavelet: V @ diag(kernel) @ V^T @ x
            wavelet_x = eigenvectors @ (kernel.unsqueeze(1) * (eigenvectors.t() @ x))
            wavelets.append(wavelet_x)

        return wavelets


class HyboWaveNet(nn.Module):
    """Multi-scale Wavelet + Hyperbolic GNN.

    Combines spectral wavelet decomposition with hyperbolic graph
    convolutions for hierarchical graph representation learning.

    Architecture:
        1. Spectral wavelet decomposition at multiple scales
        2. Scale-wise hyperbolic convolutions
        3. Cross-scale attention aggregation
        4. Final hyperbolic output

    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        out_channels: Output dimension
        n_scales: Number of wavelet scales (default: 4)
        n_layers: GNN layers per scale (default: 2)
        curvature: Hyperbolic curvature (default: 1.0)
        dropout: Dropout rate (default: 0.1)
        use_attention: Use cross-scale attention (default: True)

    Example:
        >>> model = HyboWaveNet(64, 128, 32, n_scales=4)
        >>> x = torch.randn(100, 64)
        >>> edge_index = torch.randint(0, 100, (2, 500))
        >>> node_emb = model(x, edge_index)  # (100, 32)
        >>> graph_emb = model.encode_graph(x, edge_index)  # (1, 32)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        n_scales: int = 4,
        n_layers: int = 2,
        curvature: float = 1.0,
        dropout: float = 0.1,
        use_attention: bool = True,
    ):
        """Initialize HyboWaveNet.

        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden dimension
            out_channels: Output dimension
            n_scales: Number of wavelet scales
            n_layers: GNN layers per scale
            curvature: Hyperbolic curvature
            dropout: Dropout rate
            use_attention: Use cross-scale attention
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_scales = n_scales
        self.curvature = curvature

        self.poincare = PoincareOperations(curvature)

        # Wavelet transform
        self.wavelet = SpectralWavelet(n_scales=n_scales)

        # Scale-wise encoders
        self.scale_encoders = nn.ModuleList(
            [HyperbolicLinear(in_channels, hidden_channels, curvature) for _ in range(n_scales)]
        )

        # Scale-wise GNNs
        self.scale_gnns = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        HyperbolicGraphConv(
                            hidden_channels,
                            hidden_channels,
                            curvature,
                            use_attention=True,
                            dropout=dropout,
                        )
                        for _ in range(n_layers)
                    ]
                )
                for _ in range(n_scales)
            ]
        )

        # Cross-scale attention
        self.use_attention = use_attention
        if use_attention:
            self.scale_attention = nn.Linear(hidden_channels, 1)

        # Output
        self.output = HyperbolicLinear(hidden_channels, out_channels, curvature)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (n_nodes, in_channels)
            edge_index: Edge indices (2, n_edges)
            edge_attr: Optional edge attributes

        Returns:
            Node embeddings (n_nodes, out_channels)
        """
        n_nodes = x.shape[0]
        device = x.device

        # Wavelet decomposition
        wavelet_features = self.wavelet(x, edge_index)

        # Process each scale
        scale_outputs = []
        for i, (wavelet_x, encoder, gnns) in enumerate(
            zip(wavelet_features, self.scale_encoders, self.scale_gnns, strict=False)
        ):
            # Encode
            h = encoder(wavelet_x)
            h = self.poincare.project(h)

            # GNN layers
            for gnn in gnns:
                h = gnn(h, edge_index, edge_attr)

            scale_outputs.append(h)

        # Aggregate scales
        if self.use_attention:
            # Attention-weighted aggregation
            attention_scores = []
            for h in scale_outputs:
                # Map to tangent space for attention
                v = self.poincare.log_map(h)
                score = self.scale_attention(v)
                attention_scores.append(score)

            attention = torch.softmax(torch.cat(attention_scores, dim=-1), dim=-1)

            # Weighted mean in tangent space
            mean_v = torch.zeros(n_nodes, self.hidden_channels, device=device)
            for i, h in enumerate(scale_outputs):
                v = self.poincare.log_map(h)
                mean_v = mean_v + attention[..., i : i + 1] * v

            out = self.poincare.exp_map(mean_v)
        else:
            # Simple mean in tangent space
            mean_v = torch.zeros(n_nodes, self.hidden_channels, device=device)
            for h in scale_outputs:
                v = self.poincare.log_map(h)
                mean_v = mean_v + v / self.n_scales

            out = self.poincare.exp_map(mean_v)

        out = self.poincare.project(out)
        out = self.dropout(out)
        out = self.output(out)

        return self.poincare.project(out)

    def encode_graph(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode entire graph to single embedding.

        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch assignment for multi-graph batching

        Returns:
            Graph-level embedding
        """
        # Node embeddings
        node_emb = self.forward(x, edge_index)

        # Pool to graph level
        if batch is None:
            # Single graph: hyperbolic mean
            v = self.poincare.log_map(node_emb)
            mean_v = v.mean(dim=0, keepdim=True)
            return self.poincare.exp_map(mean_v)
        else:
            # Multiple graphs: segment-wise mean
            n_graphs = batch.max().item() + 1
            graph_embs = []
            for g in range(n_graphs):
                mask = batch == g
                v = self.poincare.log_map(node_emb[mask])
                mean_v = v.mean(dim=0, keepdim=True)
                graph_embs.append(self.poincare.exp_map(mean_v))

            return torch.cat(graph_embs, dim=0)
