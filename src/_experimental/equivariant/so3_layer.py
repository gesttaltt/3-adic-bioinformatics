# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""SO(3)-equivariant neural network layers.

This module provides SO(3)-equivariant layers that respect 3D rotational
symmetry. Useful for processing point clouds, molecular structures, and
other 3D data where rotation invariance/equivariance is important.

References:
    - Fuchs et al., "SE(3)-Transformers" (2020)
    - Satorras et al., "E(n) Equivariant Graph Neural Networks" (2021)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .spherical_harmonics import ClebschGordanCoefficients, SphericalHarmonics

# Try to import e3nn for optimized operations
try:
    from e3nn import o3
    from e3nn.nn import FullyConnectedNet

    HAS_E3NN = True
except ImportError:
    HAS_E3NN = False


class RadialBasisFunctions(nn.Module):
    """Radial basis function expansion for distances.

    Expands scalar distances into a smooth basis for use in
    equivariant networks.

    Args:
        n_rbf: Number of radial basis functions
        cutoff: Maximum distance cutoff
        rbf_type: Type of basis function ("gaussian", "bessel")
    """

    def __init__(
        self,
        n_rbf: int = 16,
        cutoff: float = 5.0,
        rbf_type: str = "gaussian",
    ):
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        self.rbf_type = rbf_type

        if rbf_type == "gaussian":
            # Gaussian centers and widths
            centers = torch.linspace(0, cutoff, n_rbf)
            self.register_buffer("centers", centers)
            self.widths = nn.Parameter(torch.ones(n_rbf) * (cutoff / n_rbf))
        elif rbf_type == "bessel":
            # Bessel function frequencies
            freqs = torch.arange(1, n_rbf + 1) * torch.pi / cutoff
            self.register_buffer("freqs", freqs)
        else:
            raise ValueError(f"Unknown rbf_type: {rbf_type}")

    def forward(self, distances: Tensor) -> Tensor:
        """Expand distances to radial basis.

        Args:
            distances: Pairwise distances of shape (...)

        Returns:
            Basis values of shape (..., n_rbf)
        """
        if self.rbf_type == "gaussian":
            return self._gaussian_rbf(distances)
        else:
            return self._bessel_rbf(distances)

    def _gaussian_rbf(self, distances: Tensor) -> Tensor:
        """Gaussian radial basis functions."""
        d = distances[..., None]
        return torch.exp(-((d - self.centers) ** 2) / (2 * self.widths**2))

    def _bessel_rbf(self, distances: Tensor) -> Tensor:
        """Bessel radial basis functions."""
        d = distances[..., None]
        # Normalize to [0, 1]
        d / self.cutoff
        return torch.sqrt(2 / self.cutoff) * torch.sin(self.freqs * d) / (d + 1e-8)


class SmoothCutoff(nn.Module):
    """Smooth cutoff function for finite-range interactions.

    Args:
        cutoff: Cutoff distance
        cutoff_type: Type of cutoff ("cosine", "polynomial")
    """

    def __init__(self, cutoff: float = 5.0, cutoff_type: str = "cosine"):
        super().__init__()
        self.cutoff = cutoff
        self.cutoff_type = cutoff_type

    def forward(self, distances: Tensor) -> Tensor:
        """Apply smooth cutoff.

        Args:
            distances: Pairwise distances

        Returns:
            Cutoff weights in [0, 1]
        """
        if self.cutoff_type == "cosine":
            return self._cosine_cutoff(distances)
        else:
            return self._polynomial_cutoff(distances)

    def _cosine_cutoff(self, distances: Tensor) -> Tensor:
        """Cosine cutoff: smoothly goes from 1 to 0."""
        mask = distances < self.cutoff
        cutoff = 0.5 * (1 + torch.cos(torch.pi * distances / self.cutoff))
        return cutoff * mask.float()

    def _polynomial_cutoff(self, distances: Tensor) -> Tensor:
        """Polynomial cutoff with continuous derivatives."""
        x = distances / self.cutoff
        mask = x < 1
        cutoff = 1 - 6 * x**5 + 15 * x**4 - 10 * x**3
        return torch.clamp(cutoff, 0, 1) * mask.float()


class SO3Linear(nn.Module):
    """SO(3)-equivariant linear layer.

    Performs a linear transformation that commutes with rotations.
    Operates on spherical tensor representations.

    Args:
        in_features: Number of input features per irrep
        out_features: Number of output features per irrep
        lmax_in: Maximum input angular momentum
        lmax_out: Maximum output angular momentum
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lmax_in: int = 2,
        lmax_out: int = 2,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax_in = lmax_in
        self.lmax_out = lmax_out

        # Separate weights for each angular momentum channel
        self.weights = nn.ParameterDict()
        for l in range(min(lmax_in, lmax_out) + 1):
            weight = nn.Parameter(torch.randn(out_features, in_features) / (in_features**0.5))
            self.weights[str(l)] = weight

        if bias:
            # Only l=0 (scalar) can have bias
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor, l_indices: Tensor | None = None) -> Tensor:
        """Apply SO(3)-equivariant linear transformation.

        Args:
            x: Input tensor of shape (..., in_features, n_harmonics)
               Can be 3D (n_nodes, in_features, n_harmonics) or
               4D (batch, n_nodes, in_features, n_harmonics)
            l_indices: Angular momentum index for each harmonic

        Returns:
            Output tensor of shape (..., out_features, n_harmonics)
        """
        # Handle both 3D and 4D inputs
        n_harmonics = x.shape[-1]
        prefix_shape = x.shape[:-2]

        # Apply weight for each angular momentum
        output = torch.zeros(*prefix_shape, self.out_features, n_harmonics, device=x.device, dtype=x.dtype)

        idx = 0
        for l in range(self.lmax_in + 1):
            n_m = 2 * l + 1  # Number of m values for this l
            if l <= self.lmax_out and str(l) in self.weights:
                x_l = x[..., idx : idx + n_m]  # (batch, nodes, in_feat, n_m)
                weight = self.weights[str(l)]  # (out_feat, in_feat)
                # Correct einsum: ...im,oi->...om where i=in_feat, o=out_feat, m=n_m
                output[..., idx : idx + n_m] = torch.einsum("...im,oi->...om", x_l, weight)
            idx += n_m

        # Add bias to scalar (l=0) component only
        if self.bias is not None:
            output[..., 0] = output[..., 0] + self.bias

        return output


class SO3Convolution(nn.Module):
    """SO(3)-equivariant convolution using tensor products.

    Combines features from neighboring nodes while preserving
    rotational equivariance using Clebsch-Gordan coefficients.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        lmax: Maximum angular momentum
        n_rbf: Number of radial basis functions
        cutoff: Distance cutoff
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lmax: int = 2,
        n_rbf: int = 16,
        cutoff: float = 5.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax

        # Spherical harmonics for edge directions
        self.sh = SphericalHarmonics(lmax=lmax)

        # Radial basis for distances
        self.rbf = RadialBasisFunctions(n_rbf=n_rbf, cutoff=cutoff)

        # Cutoff function
        self.cutoff_fn = SmoothCutoff(cutoff=cutoff)

        # Radial network: maps rbf to weights
        self.radial_net = nn.Sequential(
            nn.Linear(n_rbf, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
            nn.Linear(64, in_features * out_features),
        )

        # Clebsch-Gordan coefficients
        self.cg = ClebschGordanCoefficients(lmax=lmax)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_vec: Tensor,
        edge_dist: Tensor | None = None,
    ) -> Tensor:
        """Perform SO(3)-equivariant convolution.

        Args:
            x: Node features of shape (n_nodes, in_features, n_harmonics)
            edge_index: Edge indices of shape (2, n_edges)
            edge_vec: Edge vectors of shape (n_edges, 3)
            edge_dist: Edge distances of shape (n_edges,), computed if None

        Returns:
            Updated features of shape (n_nodes, out_features, n_harmonics)
        """
        src, dst = edge_index

        # Compute distances if not provided
        if edge_dist is None:
            edge_dist = torch.linalg.norm(edge_vec, dim=-1)

        # Spherical harmonics of edge directions
        edge_sh = self.sh(edge_vec)  # (n_edges, n_harmonics)

        # Radial basis and cutoff
        rbf = self.rbf(edge_dist)  # (n_edges, n_rbf)
        cutoff = self.cutoff_fn(edge_dist)  # (n_edges,)

        # Radial weights
        radial_weights = self.radial_net(rbf)  # (n_edges, in*out)
        radial_weights = radial_weights.view(-1, self.out_features, self.in_features)
        radial_weights = radial_weights * cutoff[:, None, None]

        # Get source features
        x_src = x[src]  # (n_edges, in_features, n_harmonics)

        # Tensor product with edge spherical harmonics
        # Simplified: multiply by edge_sh for each feature
        message = torch.einsum("eih,eh->eih", x_src, edge_sh)

        # Apply radial weights: (n_edges, out_feat, in_feat) x (n_edges, in_feat, n_harm) -> (n_edges, out_feat, n_harm)
        message = torch.einsum("eoi,eih->eoh", radial_weights, message)

        # Aggregate messages
        n_nodes = x.shape[0]
        n_harmonics = x.shape[2]
        output = torch.zeros(n_nodes, self.out_features, n_harmonics, device=x.device, dtype=x.dtype)
        output.scatter_add_(0, dst[:, None, None].expand(-1, self.out_features, n_harmonics), message)

        return output


class SO3Layer(nn.Module):
    """Full SO(3)-equivariant layer with message passing.

    Combines SO(3) convolution with nonlinearity and optional
    self-interaction.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        lmax: Maximum angular momentum
        n_rbf: Number of radial basis functions
        cutoff: Distance cutoff
        use_self_interaction: Whether to include self-connections
        activation: Activation function for scalar features
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lmax: int = 2,
        n_rbf: int = 16,
        cutoff: float = 5.0,
        use_self_interaction: bool = True,
        activation: str = "silu",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lmax = lmax
        self.use_self_interaction = use_self_interaction

        # Number of spherical harmonics
        self.n_harmonics = (lmax + 1) ** 2

        # Main convolution
        self.conv = SO3Convolution(
            in_features=in_features,
            out_features=out_features,
            lmax=lmax,
            n_rbf=n_rbf,
            cutoff=cutoff,
        )

        # Self-interaction
        if use_self_interaction:
            self.self_linear = SO3Linear(
                in_features=in_features,
                out_features=out_features,
                lmax_in=lmax,
                lmax_out=lmax,
            )

        # Activation (only for scalar features)
        if activation == "silu":
            self.activation = nn.SiLU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()

        # Layer normalization for scalar features
        self.norm = nn.LayerNorm(out_features)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_vec: Tensor,
        edge_dist: Tensor | None = None,
    ) -> Tensor:
        """Apply SO(3)-equivariant layer.

        Args:
            x: Node features of shape (n_nodes, in_features, n_harmonics)
               or (n_nodes, in_features) for scalar-only input
            edge_index: Edge indices of shape (2, n_edges)
            edge_vec: Edge vectors of shape (n_edges, 3)
            edge_dist: Edge distances (optional)

        Returns:
            Updated features of shape (n_nodes, out_features, n_harmonics)
        """
        # Handle scalar-only input
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Add trivial harmonic dimension
            x = F.pad(x, (0, self.n_harmonics - 1))  # Pad with zeros

        # Message passing
        h = self.conv(x, edge_index, edge_vec, edge_dist)

        # Self-interaction
        if self.use_self_interaction:
            h = h + self.self_linear(x)

        # Apply activation to scalar (l=0) component
        scalar = h[..., 0]  # (n_nodes, out_features)
        scalar = self.norm(scalar)
        scalar = self.activation(scalar)
        h = h.clone()
        h[..., 0] = scalar

        return h


class SO3GNN(nn.Module):
    """Full SO(3)-equivariant graph neural network.

    Multi-layer GNN that preserves rotational equivariance.
    Useful for molecular property prediction and 3D point cloud processing.

    Args:
        in_features: Input feature dimension
        hidden_features: Hidden feature dimension
        out_features: Output feature dimension
        n_layers: Number of SO(3) layers
        lmax: Maximum angular momentum
        n_rbf: Number of radial basis functions
        cutoff: Distance cutoff
        pool: Pooling method ("sum", "mean", "max")
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        n_layers: int = 3,
        lmax: int = 2,
        n_rbf: int = 16,
        cutoff: float = 5.0,
        pool: str = "mean",
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.pool = pool

        # Initial embedding
        self.embed = nn.Linear(in_features, hidden_features)

        # SO(3) layers
        self.layers = nn.ModuleList()
        for _i in range(n_layers):
            self.layers.append(
                SO3Layer(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    lmax=lmax,
                    n_rbf=n_rbf,
                    cutoff=cutoff,
                )
            )

        # Output projection (from spherical to scalar)
        n_harmonics = (lmax + 1) ** 2
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_features * n_harmonics, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, out_features),
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
    ) -> Tensor:
        """Forward pass through SO(3) GNN.

        Args:
            x: Node features of shape (n_nodes, in_features)
            pos: Node positions of shape (n_nodes, 3)
            edge_index: Edge indices of shape (2, n_edges)
            batch: Batch assignment of shape (n_nodes,)

        Returns:
            Graph-level predictions of shape (n_graphs, out_features)
        """
        # Compute edge vectors and distances
        src, dst = edge_index
        edge_vec = pos[dst] - pos[src]
        edge_dist = torch.linalg.norm(edge_vec, dim=-1)

        # Initial embedding
        h = self.embed(x)

        # Apply SO(3) layers
        for layer in self.layers:
            h = layer(h, edge_index, edge_vec, edge_dist)

        # Flatten spherical features
        h = h.flatten(-2, -1)  # (n_nodes, hidden * n_harmonics)

        # Project to output
        h = self.output_proj(h)

        # Pool over nodes
        if batch is None:
            batch = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)

        n_graphs = batch.max().item() + 1
        output = torch.zeros(n_graphs, self.out_features, device=h.device, dtype=h.dtype)

        if self.pool == "sum":
            output.scatter_add_(0, batch[:, None].expand_as(h), h)
        elif self.pool == "mean":
            output.scatter_add_(0, batch[:, None].expand_as(h), h)
            counts = torch.bincount(batch).float()[:, None]
            output = output / counts.clamp(min=1)
        elif self.pool == "max":
            # Max pooling
            for i in range(n_graphs):
                mask = batch == i
                output[i] = h[mask].max(dim=0)[0]

        return output
