# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""SE(3)-Equivariant encoder for protein 3D structure.

Encodes protein structures while respecting rotational and translational
symmetry. Uses equivariant message passing over atomic coordinates.

References:
    - Satorras et al. (2021): E(n) Equivariant Graph Neural Networks
    - Thomas et al. (2018): Tensor Field Networks
    - Brandstetter et al. (2022): Geometric and Physical Quantities
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.equivariant.layers import (
    EquivariantBlock,
    InvariantReadout,
)


@dataclass
class SE3Config:
    """Configuration for SE(3)-equivariant encoder.

    Attributes:
        node_dim: Node feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output embedding dimension
        n_layers: Number of equivariant layers
        n_rbf: Number of radial basis functions
        cutoff: Distance cutoff for edges
        max_neighbors: Maximum neighbors per node
        aggregation: Readout aggregation ('mean', 'sum', 'attention')
        use_e3nn: Use e3nn backend if available
    """

    node_dim: int = 128
    hidden_dim: int = 256
    output_dim: int = 64
    n_layers: int = 4
    n_rbf: int = 16
    cutoff: float = 10.0
    max_neighbors: int = 32
    aggregation: str = "mean"
    use_e3nn: bool = True


class SE3EquivariantEncoder(nn.Module):
    """SE(3)-Equivariant encoder for protein structures.

    Takes atomic coordinates and features, performs equivariant
    message passing, and outputs invariant graph-level embeddings.

    The encoder preserves symmetry under rotations and translations,
    meaning the output is the same regardless of the protein's
    orientation in 3D space.

    Example:
        >>> config = SE3Config(output_dim=64)
        >>> encoder = SE3EquivariantEncoder(config)
        >>> # coords: (N, 3), features: (N, 20)
        >>> embedding = encoder(coords, features)
        >>> print(embedding.shape)  # (1, 64)
    """

    # Standard atom features
    ATOM_TYPES = ["C", "N", "O", "S", "H", "OTHER"]
    RESIDUE_TYPES = 20  # Standard amino acids

    def __init__(
        self,
        config: SE3Config | None = None,
        device: str = "cuda",
    ):
        """Initialize SE(3) encoder.

        Args:
            config: Encoder configuration
            device: Computation device
        """
        super().__init__()
        self.config = config or SE3Config()
        self._device = device

        # Node embedding
        self.node_embedding = nn.Sequential(
            nn.Linear(self.RESIDUE_TYPES + len(self.ATOM_TYPES), self.config.node_dim),
            nn.SiLU(),
            nn.Linear(self.config.node_dim, self.config.node_dim),
        )

        # Equivariant message passing layers
        self.layers = nn.ModuleList(
            [
                EquivariantBlock(
                    node_dim=self.config.node_dim,
                    hidden_dim=self.config.hidden_dim,
                    n_rbf=self.config.n_rbf,
                    cutoff=self.config.cutoff,
                    use_e3nn=self.config.use_e3nn,
                )
                for _ in range(self.config.n_layers)
            ]
        )

        # Invariant readout
        self.readout = InvariantReadout(
            input_dim=self.config.node_dim,
            output_dim=self.config.output_dim,
            hidden_dim=self.config.hidden_dim,
            aggregation=self.config.aggregation,
        )

        self.to(device)

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    @property
    def output_dim(self) -> int:
        return self.config.output_dim

    def forward(
        self,
        positions: torch.Tensor,
        node_features: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode protein structure to embedding.

        Args:
            positions: (N, 3) atomic coordinates
            node_features: (N, F) node features (optional)
            edge_index: (2, E) edge indices (computed if not provided)
            batch: (N,) batch assignment for multiple proteins

        Returns:
            (B, output_dim) structure embeddings
        """
        # Ensure on correct device
        positions = positions.to(self.device)

        # Build edges if not provided
        if edge_index is None:
            edge_index = self._build_edges(positions, batch)

        # Initialize node features
        if node_features is None:
            # Use position-derived features
            node_features = self._compute_position_features(positions)
        else:
            node_features = node_features.to(self.device)

        # Embed node features
        h = self.node_embedding(node_features)

        # Equivariant message passing
        for layer in self.layers:
            h = layer(h, edge_index, positions)

        # Invariant readout
        embedding = self.readout(h, batch)

        return embedding

    def _build_edges(
        self,
        positions: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build edge index based on distance cutoff.

        Args:
            positions: (N, 3) coordinates
            batch: (N,) batch assignment

        Returns:
            (2, E) edge index
        """
        n_nodes = positions.shape[0]

        # Compute pairwise distances
        dist = torch.cdist(positions, positions)

        # Create mask for edges within cutoff
        mask = (dist < self.config.cutoff) & (dist > 0)

        # Limit neighbors
        if self.config.max_neighbors is not None:
            # Keep only top-k neighbors
            _, indices = dist.topk(
                min(self.config.max_neighbors + 1, n_nodes),
                dim=-1,
                largest=False,
            )
            neighbor_mask = torch.zeros_like(mask)
            for i in range(n_nodes):
                neighbor_mask[i, indices[i]] = True
            mask = mask & neighbor_mask

        # Don't connect nodes from different graphs
        if batch is not None:
            batch_mask = batch.unsqueeze(0) == batch.unsqueeze(1)
            mask = mask & batch_mask

        # Convert to edge index
        edge_index = mask.nonzero(as_tuple=False).t()

        return edge_index

    def _compute_position_features(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute invariant features from positions.

        Args:
            positions: (N, 3) coordinates

        Returns:
            (N, F) position-derived features
        """
        n_nodes = positions.shape[0]

        # Simple invariant features
        # - Distance from centroid
        centroid = positions.mean(dim=0, keepdim=True)
        dist_from_center = (positions - centroid).norm(dim=-1, keepdim=True)

        # - Local density (neighbors within cutoff)
        pairwise_dist = torch.cdist(positions, positions)
        local_density = (pairwise_dist < self.config.cutoff).sum(dim=-1, keepdim=True).float()
        local_density = local_density / local_density.max()

        # - Normalized position (still invariant under global translation)
        norm_pos = (positions - centroid) / (dist_from_center.max() + 1e-8)

        # Combine features
        features = torch.cat(
            [
                dist_from_center / (dist_from_center.max() + 1e-8),
                local_density,
                norm_pos,
            ],
            dim=-1,
        )

        # Pad to expected dimension
        n_features = features.shape[-1]
        if n_features < self.RESIDUE_TYPES + len(self.ATOM_TYPES):
            padding = torch.zeros(
                n_nodes,
                self.RESIDUE_TYPES + len(self.ATOM_TYPES) - n_features,
                device=positions.device,
            )
            features = torch.cat([features, padding], dim=-1)

        return features

    def encode_protein(
        self,
        coords: torch.Tensor,
        residue_types: torch.Tensor,
        atom_types: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode protein from standard representation.

        Args:
            coords: (N, 3) atomic coordinates
            residue_types: (N,) residue type indices (0-19)
            atom_types: (N,) atom type indices (optional)

        Returns:
            (1, output_dim) protein embedding
        """
        n_atoms = coords.shape[0]

        # One-hot encode residue types
        residue_onehot = F.one_hot(
            residue_types.long(),
            num_classes=self.RESIDUE_TYPES,
        ).float()

        # One-hot encode atom types if provided
        if atom_types is not None:
            atom_onehot = F.one_hot(
                atom_types.long(),
                num_classes=len(self.ATOM_TYPES),
            ).float()
        else:
            atom_onehot = torch.zeros(n_atoms, len(self.ATOM_TYPES), device=coords.device)

        # Combine features
        node_features = torch.cat([residue_onehot, atom_onehot], dim=-1)

        return self.forward(coords, node_features)


class SE3WithHyperbolic(nn.Module):
    """SE(3) encoder with hyperbolic output space.

    Combines equivariant structure encoding with hyperbolic geometry
    for hierarchical representation of protein structure.
    """

    def __init__(
        self,
        se3_config: SE3Config | None = None,
        hyperbolic_dim: int = 32,
        curvature: float = -1.0,
        device: str = "cuda",
    ):
        """Initialize combined encoder.

        Args:
            se3_config: SE(3) encoder configuration
            hyperbolic_dim: Dimension of hyperbolic output
            curvature: Poincaré ball curvature
            device: Computation device
        """
        super().__init__()

        self.se3_encoder = SE3EquivariantEncoder(se3_config, device)
        self.curvature = curvature

        # Project to hyperbolic space
        self.hyperbolic_proj = nn.Sequential(
            nn.Linear(self.se3_encoder.output_dim, hyperbolic_dim * 2),
            nn.SiLU(),
            nn.Linear(hyperbolic_dim * 2, hyperbolic_dim),
        )

        self.to(device)

    def forward(
        self,
        positions: torch.Tensor,
        node_features: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Encode structure to hyperbolic space.

        Args:
            positions: Atomic coordinates
            node_features: Node features

        Returns:
            Hyperbolic embeddings on Poincaré ball
        """
        # SE(3) encoding
        euclidean = self.se3_encoder(positions, node_features, **kwargs)

        # Project to hyperbolic
        pre_hyp = self.hyperbolic_proj(euclidean)

        # Map to Poincaré ball via exponential map
        c = abs(self.curvature)
        norm = pre_hyp.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sqrt_c = c**0.5

        hyperbolic = torch.tanh(sqrt_c * norm) * pre_hyp / (sqrt_c * norm)

        # Ensure inside ball
        max_norm = 1 - 1e-5
        hyperbolic = hyperbolic * max_norm / hyperbolic.norm(dim=-1, keepdim=True).clamp(min=max_norm)

        return hyperbolic
