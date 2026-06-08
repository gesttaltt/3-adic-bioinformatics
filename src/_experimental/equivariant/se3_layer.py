# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""SE(3)-equivariant neural network layers.

This module provides SE(3)-equivariant layers that respect both 3D rotations
AND translations. SE(3) is the group of rigid transformations in 3D space.

These layers are particularly useful for:
- Protein structure prediction
- Molecular dynamics
- Point cloud processing where translation doesn't matter

References:
    - Fuchs et al., "SE(3)-Transformers" (2020)
    - Satorras et al., "E(n) Equivariant Graph Neural Networks" (2021)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .so3_layer import RadialBasisFunctions, SmoothCutoff


class SE3Linear(nn.Module):
    """SE(3)-equivariant linear layer.

    Operates on both scalar and vector features while preserving
    SE(3) equivariance.

    Args:
        scalar_in: Number of input scalar features
        scalar_out: Number of output scalar features
        vector_in: Number of input vector features (each is 3D)
        vector_out: Number of output vector features
    """

    def __init__(
        self,
        scalar_in: int,
        scalar_out: int,
        vector_in: int = 0,
        vector_out: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.scalar_in = scalar_in
        self.scalar_out = scalar_out
        self.vector_in = vector_in
        self.vector_out = vector_out

        # Scalar-to-scalar
        self.w_ss = nn.Linear(scalar_in, scalar_out, bias=bias)

        # Vector-to-scalar (via norm)
        if vector_in > 0:
            self.w_vs = nn.Linear(vector_in, scalar_out, bias=False)
        else:
            self.w_vs = None

        # Scalar-to-vector (not possible while preserving equivariance)
        # Vector-to-vector (via scalar weighting)
        if vector_out > 0 and vector_in > 0:
            self.w_vv = nn.Linear(vector_in, vector_out, bias=False)
        else:
            self.w_vv = None

    def forward(self, scalars: Tensor, vectors: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Apply SE(3)-equivariant linear transformation.

        Args:
            scalars: Scalar features of shape (..., scalar_in)
            vectors: Vector features of shape (..., vector_in, 3)

        Returns:
            Tuple of (output_scalars, output_vectors)
        """
        # Scalar transformation
        out_scalars = self.w_ss(scalars)

        # Vector norms contribute to scalars
        if vectors is not None and self.w_vs is not None:
            v_norms = torch.linalg.norm(vectors, dim=-1)  # (..., vector_in)
            out_scalars = out_scalars + self.w_vs(v_norms)

        # Vector transformation
        out_vectors = None
        if vectors is not None and self.w_vv is not None:
            # Reshape: vectors (..., vector_in, 3) -> (..., 3, vector_in)
            vectors_reshaped = vectors.transpose(-1, -2)

            # Apply weights per spatial dimension
            # w_vv.weight has shape (vector_out, vector_in), so we use 'oi' order
            out_vectors = torch.einsum("...xi,oi->...xo", vectors_reshaped, self.w_vv.weight)
            out_vectors = out_vectors.transpose(-1, -2)  # (..., vector_out, 3)

        return out_scalars, out_vectors


class SE3MessagePassing(nn.Module):
    """SE(3)-equivariant message passing layer.

    Aggregates information from neighboring nodes while preserving
    SE(3) equivariance. Uses EGNN-style updates.

    Args:
        hidden_dim: Hidden feature dimension
        n_rbf: Number of radial basis functions
        cutoff: Distance cutoff
        update_coords: Whether to update coordinates (for dynamics)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_rbf: int = 16,
        cutoff: float = 5.0,
        update_coords: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.update_coords = update_coords

        # Radial basis
        self.rbf = RadialBasisFunctions(n_rbf=n_rbf, cutoff=cutoff)
        self.cutoff_fn = SmoothCutoff(cutoff=cutoff)

        # Message network
        self.message_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Attention for aggregation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Coordinate update (if enabled)
        if update_coords:
            self.coord_net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1, bias=False),
            )

    def forward(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Perform SE(3)-equivariant message passing.

        Args:
            h: Node features of shape (n_nodes, hidden_dim)
            pos: Node positions of shape (n_nodes, 3)
            edge_index: Edge indices of shape (2, n_edges)

        Returns:
            Tuple of (updated_features, updated_positions)
        """
        src, dst = edge_index
        n_nodes = h.shape[0]

        # Edge vectors and distances
        edge_vec = pos[dst] - pos[src]  # (n_edges, 3)
        edge_dist = torch.linalg.norm(edge_vec, dim=-1)  # (n_edges,)

        # Radial features
        rbf = self.rbf(edge_dist)
        cutoff = self.cutoff_fn(edge_dist)

        # Message computation
        h_src = h[src]
        h_dst = h[dst]
        msg_input = torch.cat([h_src, h_dst, rbf], dim=-1)
        messages = self.message_net(msg_input)  # (n_edges, hidden_dim)

        # Attention weights
        att = self.attention(messages).squeeze(-1) * cutoff  # (n_edges,)

        # Aggregate messages
        h_new = torch.zeros_like(h)
        h_new.scatter_add_(0, dst[:, None].expand(-1, self.hidden_dim), messages * att[:, None])

        # Normalize by degree
        degree = torch.bincount(dst, minlength=n_nodes).float()[:, None]
        h_new = h_new / degree.clamp(min=1)

        # Update positions if enabled
        pos_new = pos
        if self.update_coords:
            # Coordinate update using edge vectors
            coord_weights = self.coord_net(messages).squeeze(-1) * cutoff
            edge_vec_normalized = edge_vec / (edge_dist[:, None] + 1e-8)

            pos_delta = torch.zeros_like(pos)
            pos_delta.scatter_add_(0, dst[:, None].expand(-1, 3), edge_vec_normalized * coord_weights[:, None])
            pos_new = pos + pos_delta / degree.clamp(min=1)

        return h_new, pos_new


class SE3Layer(nn.Module):
    """Full SE(3)-equivariant layer.

    Combines message passing with residual connections and normalization.

    Args:
        hidden_dim: Hidden feature dimension
        n_rbf: Number of radial basis functions
        cutoff: Distance cutoff
        update_coords: Whether to update coordinates
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int,
        n_rbf: int = 16,
        cutoff: float = 5.0,
        update_coords: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.update_coords = update_coords

        # Message passing
        self.message_passing = SE3MessagePassing(
            hidden_dim=hidden_dim,
            n_rbf=n_rbf,
            cutoff=cutoff,
            update_coords=update_coords,
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply SE(3)-equivariant layer.

        Args:
            h: Node features of shape (n_nodes, hidden_dim)
            pos: Node positions of shape (n_nodes, 3)
            edge_index: Edge indices of shape (2, n_edges)

        Returns:
            Tuple of (updated_features, updated_positions)
        """
        # Message passing with residual
        h_mp, pos_new = self.message_passing(h, pos, edge_index)
        h = self.norm1(h + h_mp)

        # Feed-forward with residual
        h = self.norm2(h + self.ff(h))

        return h, pos_new


class SE3Transformer(nn.Module):
    """SE(3)-equivariant Transformer architecture.

    Full transformer model that preserves SE(3) equivariance.
    Uses self-attention weighted by geometric information.

    Args:
        in_features: Input feature dimension
        hidden_dim: Hidden feature dimension
        out_features: Output feature dimension
        n_layers: Number of SE(3) layers
        n_heads: Number of attention heads
        cutoff: Distance cutoff
        update_coords: Whether to update coordinates
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        n_layers: int = 4,
        n_heads: int = 4,
        cutoff: float = 5.0,
        update_coords: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Input embedding
        self.embed = nn.Linear(in_features, hidden_dim)

        # SE(3) layers
        self.layers = nn.ModuleList(
            [
                SE3Layer(
                    hidden_dim=hidden_dim,
                    cutoff=cutoff,
                    update_coords=update_coords,
                )
                for _ in range(n_layers)
            ]
        )

        # Output projection
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Node features of shape (n_nodes, in_features)
            pos: Node positions of shape (n_nodes, 3)
            edge_index: Edge indices of shape (2, n_edges)
            batch: Batch assignment of shape (n_nodes,)

        Returns:
            Tuple of (node_predictions, final_positions)
        """
        # Embed input
        h = self.embed(x)

        # Apply SE(3) layers
        for layer in self.layers:
            h, pos = layer(h, pos, edge_index)

        # Output projection
        out = self.output(h)

        return out, pos


class EGNN(nn.Module):
    """E(n) Equivariant Graph Neural Network.

    Simplified implementation of EGNN from Satorras et al. (2021).
    Equivariant to rotations, translations, and reflections in n dimensions.

    Args:
        in_features: Input feature dimension
        hidden_dim: Hidden feature dimension
        out_features: Output feature dimension
        n_layers: Number of EGNN layers
        update_coords: Whether to update coordinates
        residual: Whether to use residual connections
    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        n_layers: int = 4,
        update_coords: bool = True,
        residual: bool = True,
    ):
        super().__init__()
        self.residual = residual

        # Input embedding
        self.embed = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.SiLU(),
        )

        # EGNN layers
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim, update_coords=update_coords) for _ in range(n_layers)])

        # Output
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_features),
        )

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Node features of shape (n_nodes, in_features)
            pos: Node positions of shape (n_nodes, 3)
            edge_index: Edge indices of shape (2, n_edges)
            batch: Batch assignment

        Returns:
            Tuple of (predictions, final_positions)
        """
        h = self.embed(x)

        for layer in self.layers:
            h_new, pos_new = layer(h, pos, edge_index)
            if self.residual:
                h = h + h_new
                pos = pos + (pos_new - pos)
            else:
                h = h_new
                pos = pos_new

        out = self.output(h)
        return out, pos


class EGNNLayer(nn.Module):
    """Single EGNN layer.

    Args:
        hidden_dim: Feature dimension
        update_coords: Whether to update coordinates
    """

    def __init__(self, hidden_dim: int, update_coords: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.update_coords = update_coords

        # Edge model
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Node model
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Coordinate model
        if update_coords:
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1, bias=False),
            )

    def forward(
        self,
        h: Tensor,
        pos: Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass."""
        src, dst = edge_index

        # Edge features
        edge_vec = pos[dst] - pos[src]
        edge_dist_sq = (edge_vec**2).sum(dim=-1, keepdim=True)

        h_src, h_dst = h[src], h[dst]
        edge_input = torch.cat([h_src, h_dst, edge_dist_sq], dim=-1)
        edge_feat = self.edge_mlp(edge_input)

        # Aggregate edge features
        agg = torch.zeros_like(h)
        agg.scatter_add_(0, dst[:, None].expand_as(edge_feat), edge_feat)

        # Node update
        node_input = torch.cat([h, agg], dim=-1)
        h_new = self.node_mlp(node_input)

        # Coordinate update
        pos_new = pos
        if self.update_coords:
            coord_weights = self.coord_mlp(edge_feat)
            edge_vec_weighted = edge_vec * coord_weights

            coord_delta = torch.zeros_like(pos)
            coord_delta.scatter_add_(0, dst[:, None].expand(-1, 3), edge_vec_weighted)

            # Normalize by degree
            degree = torch.bincount(dst, minlength=h.shape[0]).float()[:, None]
            pos_new = pos + coord_delta / degree.clamp(min=1)

        return h_new, pos_new
