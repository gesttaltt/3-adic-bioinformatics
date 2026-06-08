# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Equivariant neural network layers.

Provides building blocks for SE(3)/E(3) equivariant networks.
Can use e3nn backend when available, otherwise uses portable implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Check for e3nn availability
try:
    import e3nn
    from e3nn import o3
    from e3nn.nn import FullyConnectedNet

    HAS_E3NN = True
except ImportError:
    HAS_E3NN = False


class RadialBasisFunctions(nn.Module):
    """Radial basis functions for distance encoding.

    Encodes pairwise distances using Gaussian or Bessel basis functions.
    This is invariant to rotations/translations by construction.
    """

    def __init__(
        self,
        n_rbf: int = 16,
        cutoff: float = 10.0,
        rbf_type: str = "gaussian",
    ):
        """Initialize RBF layer.

        Args:
            n_rbf: Number of basis functions
            cutoff: Distance cutoff
            rbf_type: 'gaussian' or 'bessel'
        """
        super().__init__()
        self.n_rbf = n_rbf
        self.cutoff = cutoff
        self.rbf_type = rbf_type

        if rbf_type == "gaussian":
            # Gaussian centers and widths
            centers = torch.linspace(0, cutoff, n_rbf)
            self.register_buffer("centers", centers)
            self.register_buffer("widths", torch.ones(n_rbf) * (cutoff / n_rbf))
        elif rbf_type == "bessel":
            # Bessel function frequencies
            freqs = torch.arange(1, n_rbf + 1) * math.pi / cutoff
            self.register_buffer("freqs", freqs)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Compute RBF encoding of distances.

        Args:
            distances: Pairwise distances (...,)

        Returns:
            RBF encodings (..., n_rbf)
        """
        d = distances.unsqueeze(-1)

        if self.rbf_type == "gaussian":
            # Gaussian RBF
            rbf = torch.exp(-((d - self.centers) ** 2) / (2 * self.widths**2))
        else:
            # Bessel RBF
            rbf = math.sqrt(2 / self.cutoff) * torch.sin(self.freqs * d) / d.clamp(min=1e-8)

        # Smooth cutoff
        cutoff_mask = (distances < self.cutoff).float().unsqueeze(-1)
        rbf = rbf * cutoff_mask

        return rbf


class InvariantReadout(nn.Module):
    """Convert equivariant features to invariant output.

    Pools node features while maintaining invariance to
    rotations and translations.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        aggregation: str = "mean",
    ):
        """Initialize readout layer.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            hidden_dim: Hidden layer dimension
            aggregation: 'mean', 'sum', or 'attention'
        """
        super().__init__()
        self.aggregation = aggregation

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        if aggregation == "attention":
            self.attention = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        node_features: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool node features to graph-level representation.

        Args:
            node_features: (N, input_dim) node features
            batch: (N,) batch assignment for each node

        Returns:
            (B, output_dim) pooled features
        """
        # Transform features
        x = self.mlp(node_features)

        if batch is None:
            # Single graph
            if self.aggregation == "mean":
                return x.mean(dim=0, keepdim=True)
            elif self.aggregation == "sum":
                return x.sum(dim=0, keepdim=True)
            elif self.aggregation == "attention":
                weights = F.softmax(self.attention(node_features), dim=0)
                return (x * weights).sum(dim=0, keepdim=True)
        else:
            # Multiple graphs
            n_graphs = batch.max().item() + 1
            output = torch.zeros(n_graphs, x.shape[-1], device=x.device)

            if self.aggregation == "mean":
                for i in range(n_graphs):
                    mask = batch == i
                    output[i] = x[mask].mean(dim=0)
            elif self.aggregation == "sum":
                for i in range(n_graphs):
                    mask = batch == i
                    output[i] = x[mask].sum(dim=0)
            elif self.aggregation == "attention":
                weights = self.attention(node_features)
                for i in range(n_graphs):
                    mask = batch == i
                    w = F.softmax(weights[mask], dim=0)
                    output[i] = (x[mask] * w).sum(dim=0)

            return output


class EquivariantBlock(nn.Module):
    """Equivariant message passing block.

    Performs message passing that respects SE(3) symmetry.
    Uses spherical harmonics for angular encoding when e3nn available,
    otherwise uses distance-only invariant features.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 16,
        hidden_dim: int = 128,
        n_rbf: int = 16,
        cutoff: float = 10.0,
        use_e3nn: bool = True,
    ):
        """Initialize equivariant block.

        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            hidden_dim: Hidden dimension
            n_rbf: Number of radial basis functions
            cutoff: Distance cutoff
            use_e3nn: Use e3nn if available
        """
        super().__init__()
        self.node_dim = node_dim
        self.use_e3nn = use_e3nn and HAS_E3NN

        # Radial basis for distance encoding
        self.rbf = RadialBasisFunctions(n_rbf=n_rbf, cutoff=cutoff)

        if self.use_e3nn:
            self._init_e3nn_layers(node_dim, edge_dim, hidden_dim, n_rbf)
        else:
            self._init_invariant_layers(node_dim, edge_dim, hidden_dim, n_rbf)

    def _init_e3nn_layers(self, node_dim, edge_dim, hidden_dim, n_rbf):
        """Initialize e3nn-based equivariant layers."""
        # Scalar and vector irreps
        self.irreps_in = o3.Irreps(f"{node_dim}x0e")
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=2)
        self.irreps_out = o3.Irreps(f"{node_dim}x0e")

        # Message MLP
        self.message_mlp = FullyConnectedNet(
            [n_rbf, hidden_dim, hidden_dim],
            act=torch.nn.SiLU(),
        )

        # Tensor product for equivariant messages
        self.tp = o3.FullyConnectedTensorProduct(
            self.irreps_in,
            self.irreps_sh,
            self.irreps_out,
            shared_weights=False,
        )

        # Update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def _init_invariant_layers(self, node_dim, edge_dim, hidden_dim, n_rbf):
        """Initialize invariant message passing layers."""
        # Message function
        self.message_net = nn.Sequential(
            nn.Linear(2 * node_dim + n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        # Update function
        self.update_net = nn.Sequential(
            nn.Linear(2 * node_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Equivariant message passing.

        Args:
            node_features: (N, node_dim) node features
            edge_index: (2, E) edge indices
            positions: (N, 3) node positions

        Returns:
            Updated node features (N, node_dim)
        """
        src, dst = edge_index

        # Compute edge vectors and distances
        edge_vec = positions[dst] - positions[src]
        distances = edge_vec.norm(dim=-1)

        # Radial basis encoding
        rbf = self.rbf(distances)

        if self.use_e3nn:
            return self._forward_e3nn(node_features, edge_index, edge_vec, rbf)
        else:
            return self._forward_invariant(node_features, edge_index, rbf)

    def _forward_e3nn(self, node_features, edge_index, edge_vec, rbf):
        """E3nn-based equivariant forward."""
        src, dst = edge_index

        # Spherical harmonics for angular encoding
        edge_sh = o3.spherical_harmonics(
            self.irreps_sh,
            edge_vec,
            normalize=True,
            normalization="component",
        )

        # Compute messages using tensor product
        weights = self.message_mlp(rbf)
        messages = self.tp(node_features[src], edge_sh, weights)

        # Aggregate messages
        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, dst, messages)

        # Update with residual
        combined = torch.cat([node_features, aggregated], dim=-1)
        update = self.update_mlp(combined)

        return node_features + update

    def _forward_invariant(self, node_features, edge_index, rbf):
        """Invariant message passing forward."""
        src, dst = edge_index

        # Compute messages
        message_input = torch.cat(
            [node_features[src], node_features[dst], rbf],
            dim=-1,
        )
        messages = self.message_net(message_input)

        # Aggregate messages
        aggregated = torch.zeros_like(node_features)
        aggregated.index_add_(0, dst, messages)

        # Update
        update_input = torch.cat([node_features, aggregated], dim=-1)
        update = self.update_net(update_input)

        # Residual + layer norm
        return self.layer_norm(node_features + update)


class VectorNeuronLayer(nn.Module):
    """Vector Neuron layer for SO(3)-equivariant processing.

    Processes 3D vectors while maintaining rotation equivariance.
    Based on Vector Neurons (Deng et al., 2021).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ):
        """Initialize vector neuron layer.

        Args:
            in_channels: Input channels (per vector)
            out_channels: Output channels
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Linear transform preserving equivariance
        self.linear = nn.Linear(in_channels, out_channels, bias=False)

        # Direction-preserving nonlinearity parameters
        self.q = nn.Linear(in_channels, out_channels, bias=False)
        self.k = nn.Parameter(torch.randn(out_channels) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Equivariant forward pass.

        Args:
            x: (batch, n_vectors, in_channels, 3) input vectors

        Returns:
            (batch, n_vectors, out_channels, 3) output vectors
        """
        # x shape: (B, N, C_in, 3)
        # Transpose for linear: (B, N, 3, C_in) -> (B, N, 3, C_out)
        x_t = x.transpose(-1, -2)
        out = self.linear(x_t).transpose(-1, -2)  # (B, N, C_out, 3)

        # Equivariant nonlinearity
        # Project onto learned directions
        q = self.q(x_t).transpose(-1, -2)  # (B, N, C_out, 3)
        q_norm = q / (q.norm(dim=-1, keepdim=True) + 1e-8)

        # Dot product with output
        proj = (out * q_norm).sum(dim=-1, keepdim=True)

        # ReLU-like nonlinearity preserving direction
        k = self.k.view(1, 1, -1, 1)
        scale = F.relu(proj - k) - F.relu(-proj - k)

        return out + scale * q_norm
