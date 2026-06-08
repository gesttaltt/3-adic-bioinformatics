# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Holographic Encoder combining Spectral Graph Features with Poincare Embeddings.

This module implements the Spectral Bio-ML & Holographic Embeddings proposal,
combining graph spectral features (Laplacian eigenvectors) with hyperbolic
Poincare ball embeddings to capture hierarchical protein relationships.

Key Concepts:
    - Spectral encoding: Graph Laplacian eigenvectors for topology
    - Holographic projection: Poincare ball for hierarchical structure
    - Multi-scale features: Combine local and global graph properties
    - PPI network compatibility: Designed for protein-protein interactions

Research Reference:
    RESEARCH_PROPOSALS/Spectral_BioML_Holographic_Embeddings/proposal.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geometry.poincare import (
    poincare_distance,
    project_to_poincare,
)


class GraphLaplacianEncoder(nn.Module):
    """Compute graph Laplacian features from adjacency matrices.

    Extracts spectral features (eigenvectors) from the normalized
    graph Laplacian for topological encoding.
    """

    def __init__(
        self,
        n_eigenvectors: int = 16,
        normalize: bool = True,
    ):
        """Initialize Laplacian encoder.

        Args:
            n_eigenvectors: Number of eigenvectors to extract
            normalize: Whether to use normalized Laplacian
        """
        super().__init__()
        self.n_eigenvectors = n_eigenvectors
        self.normalize = normalize

    def compute_laplacian(self, adjacency: torch.Tensor) -> torch.Tensor:
        """Compute graph Laplacian from adjacency matrix.

        Args:
            adjacency: Adjacency matrix (B, N, N) or (N, N)

        Returns:
            Graph Laplacian
        """
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)

        B, N, _ = adjacency.shape

        # Degree matrix
        degree = adjacency.sum(dim=-1)  # (B, N)

        if self.normalize:
            # Normalized Laplacian: L_sym = I - D^{-1/2} A D^{-1/2}
            d_inv_sqrt = torch.pow(degree + 1e-8, -0.5)
            d_mat = torch.diag_embed(d_inv_sqrt)
            identity = torch.eye(N, device=adjacency.device).unsqueeze(0).expand(B, -1, -1)
            norm_adj = torch.bmm(torch.bmm(d_mat, adjacency), d_mat)
            laplacian = identity - norm_adj
        else:
            # Unnormalized Laplacian: L = D - A
            degree_mat = torch.diag_embed(degree)
            laplacian = degree_mat - adjacency

        return laplacian

    def forward(self, adjacency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract spectral features from graph.

        Args:
            adjacency: Adjacency matrix (B, N, N)

        Returns:
            Tuple of (eigenvalues, eigenvectors) for top-k
        """
        L = self.compute_laplacian(adjacency)

        # Eigendecomposition (symmetric)
        eigenvalues, eigenvectors = torch.linalg.eigh(L)

        # Select top-k non-trivial eigenvectors
        # Skip first (constant eigenvector for connected graphs)
        B, N, _ = eigenvectors.shape
        k = min(self.n_eigenvectors, N - 1)

        selected_vals = eigenvalues[:, 1 : k + 1]  # (B, k)
        selected_vecs = eigenvectors[:, :, 1 : k + 1]  # (B, N, k)

        return selected_vals, selected_vecs


class MultiScaleGraphFeatures(nn.Module):
    """Extract multi-scale features from graph structure.

    Combines local (node-level) and global (graph-level) features
    for comprehensive graph representation.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        n_scales: int = 3,
    ):
        """Initialize multi-scale feature extractor.

        Args:
            hidden_dim: Dimension of hidden features
            n_scales: Number of scales to consider
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_scales = n_scales

        # Graph convolution-like aggregation at multiple scales
        self.scale_weights = nn.Parameter(torch.ones(n_scales) / n_scales)

    def forward(
        self,
        adjacency: torch.Tensor,
        node_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute multi-scale graph features.

        Args:
            adjacency: Adjacency matrix (B, N, N)
            node_features: Optional node features (B, N, D)

        Returns:
            Multi-scale graph features (B, hidden_dim)
        """
        B, N, _ = adjacency.shape

        if node_features is None:
            # Use degree as default node features
            node_features = adjacency.sum(dim=-1, keepdim=True)  # (B, N, 1)

        # Normalize adjacency for message passing
        degree = adjacency.sum(dim=-1, keepdim=True) + 1e-8
        norm_adj = adjacency / degree

        # Multi-scale aggregation
        features = node_features
        scale_features = []

        for s in range(self.n_scales):
            # Aggregate at scale s
            if s > 0:
                features = torch.bmm(norm_adj, features)
            scale_features.append(features.mean(dim=1))  # Graph-level pooling

        # Weighted combination
        weights = F.softmax(self.scale_weights, dim=0)
        combined = sum(w * f for w, f in zip(weights, scale_features, strict=False))

        # Pad or project to hidden_dim
        if combined.shape[-1] < self.hidden_dim:
            padding = torch.zeros(B, self.hidden_dim - combined.shape[-1], device=combined.device)
            combined = torch.cat([combined, padding], dim=-1)
        elif combined.shape[-1] > self.hidden_dim:
            combined = combined[:, : self.hidden_dim]

        return combined


class HolographicEncoder(nn.Module):
    """Combine spectral graph features with Poincare hyperbolic embeddings.

    This encoder creates hierarchical-aware representations of graph structures
    by combining spectral features (topology) with hyperbolic geometry (hierarchy).
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 16,
        n_eigenvectors: int = 16,
        curvature: float = 1.0,
        max_norm: float = 0.95,
        use_multi_scale: bool = True,
    ):
        """Initialize holographic encoder.

        Args:
            input_dim: Dimension of input node features
            hidden_dim: Hidden layer dimension
            output_dim: Output latent dimension (on Poincare ball)
            n_eigenvectors: Number of spectral eigenvectors to use
            curvature: Curvature of Poincare ball
            max_norm: Maximum norm for Poincare projection
            use_multi_scale: Whether to use multi-scale features
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.curvature = curvature
        self.max_norm = max_norm

        # Spectral encoder
        self.laplacian_encoder = GraphLaplacianEncoder(n_eigenvectors=n_eigenvectors)

        # Multi-scale features
        self.use_multi_scale = use_multi_scale
        if use_multi_scale:
            self.multi_scale = MultiScaleGraphFeatures(hidden_dim=hidden_dim // 2)

        # Spectral feature projection
        self.spectral_proj = nn.Sequential(
            nn.Linear(n_eigenvectors, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Node feature projection
        self.node_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Combined projection to Poincare ball
        combined_dim = hidden_dim if use_multi_scale else hidden_dim // 2 + hidden_dim // 2
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Learnable curvature scaling
        self.curvature_scale = nn.Parameter(torch.tensor(1.0))

    def encode_spectral(
        self,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Encode spectral features from graph.

        Args:
            adjacency: Adjacency matrix (B, N, N)

        Returns:
            Spectral features (B, hidden_dim // 2)
        """
        eigenvalues, eigenvectors = self.laplacian_encoder(adjacency)

        # Pool eigenvectors to graph level
        # Use eigenvalue-weighted pooling
        eigenvectors.shape[0]
        weights = F.softmax(-eigenvalues, dim=-1)  # Lower eigenvalues = more important
        pooled = (eigenvectors * weights.unsqueeze(1)).sum(dim=1)  # (B, k)

        # Pad if needed
        if pooled.shape[-1] < self.laplacian_encoder.n_eigenvectors:
            pad_size = self.laplacian_encoder.n_eigenvectors - pooled.shape[-1]
            pooled = F.pad(pooled, (0, pad_size))

        return self.spectral_proj(pooled)

    def encode_nodes(
        self,
        node_features: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> torch.Tensor:
        """Encode node features with graph context.

        Args:
            node_features: Node features (B, N, D)
            adjacency: Adjacency matrix (B, N, N)

        Returns:
            Node-based graph features (B, hidden_dim // 2)
        """
        # Simple mean pooling with projection
        graph_features = node_features.mean(dim=1)  # (B, D)

        # Pad or truncate to input_dim
        if graph_features.shape[-1] < self.input_dim:
            pad_size = self.input_dim - graph_features.shape[-1]
            graph_features = F.pad(graph_features, (0, pad_size))
        elif graph_features.shape[-1] > self.input_dim:
            graph_features = graph_features[:, : self.input_dim]

        return self.node_proj(graph_features)

    def forward(
        self,
        adjacency: torch.Tensor,
        node_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode graph to holographic Poincare embedding.

        Args:
            adjacency: Adjacency matrix (B, N, N)
            node_features: Optional node features (B, N, D)

        Returns:
            Dictionary with:
                - z: Poincare ball embedding (B, output_dim)
                - z_euclidean: Pre-projection embedding
                - spectral_features: Spectral encoding
                - eigenvalues: Graph Laplacian eigenvalues
        """
        adjacency.shape[0]

        # Spectral features
        spectral_features = self.encode_spectral(adjacency)

        # Get eigenvalues for analysis
        eigenvalues, _ = self.laplacian_encoder(adjacency)

        # Node features
        if node_features is not None:
            node_features_enc = self.encode_nodes(node_features, adjacency)
        else:
            # Use degree as proxy
            degrees = adjacency.sum(dim=-1)  # (B, N)
            # Pad to input_dim
            if degrees.shape[-1] < self.input_dim:
                degrees = F.pad(degrees, (0, self.input_dim - degrees.shape[-1]))
            node_features_enc = self.node_proj(degrees)

        # Combine features
        if self.use_multi_scale:
            multi_scale_features = self.multi_scale(adjacency, node_features)
            combined = torch.cat([spectral_features, multi_scale_features], dim=-1)
        else:
            combined = torch.cat([spectral_features, node_features_enc], dim=-1)

        # Project to output dimension
        z_euclidean = self.output_proj(combined)

        # Map to Poincare ball
        scaled_curvature = self.curvature * torch.sigmoid(self.curvature_scale)
        z = project_to_poincare(z_euclidean, c=scaled_curvature.item(), max_norm=self.max_norm)

        return {
            "z": z,
            "z_euclidean": z_euclidean,
            "spectral_features": spectral_features,
            "eigenvalues": eigenvalues,
            "curvature": scaled_curvature,
        }

    def compute_hierarchy_score(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute hierarchy score based on Poincare distance from origin.

        Points near origin = high in hierarchy
        Points near boundary = low in hierarchy

        Args:
            z: Poincare ball embeddings (B, D)

        Returns:
            Hierarchy scores (B,) - higher = higher in hierarchy
        """
        # V5.12.2: Use hyperbolic distance from origin
        origin = torch.zeros_like(z)
        hyp_dist = poincare_distance(z, origin, c=self.curvature)
        # Invert: closer to origin = higher score
        # Normalize by approximate max hyperbolic distance at max_norm
        max_hyp_dist = poincare_distance(
            torch.full_like(z[:1], self.max_norm / z.shape[-1] ** 0.5),
            torch.zeros_like(z[:1]),
            c=self.curvature,
        )
        hierarchy = 1.0 - hyp_dist / (max_hyp_dist.item() + 1e-6)
        return hierarchy


class PPINetworkEncoder(nn.Module):
    """Specialized encoder for Protein-Protein Interaction networks.

    Combines holographic encoding with PPI-specific features like
    interaction confidence scores and protein properties.
    """

    def __init__(
        self,
        n_proteins: int = 1000,
        embedding_dim: int = 32,
        hidden_dim: int = 128,
        output_dim: int = 16,
        curvature: float = 1.0,
    ):
        """Initialize PPI encoder.

        Args:
            n_proteins: Maximum number of proteins in vocabulary
            embedding_dim: Dimension of protein embeddings
            hidden_dim: Hidden layer dimension
            output_dim: Output latent dimension
            curvature: Curvature for Poincare ball
        """
        super().__init__()
        self.n_proteins = n_proteins
        self.embedding_dim = embedding_dim

        # Protein embeddings
        self.protein_embedding = nn.Embedding(n_proteins, embedding_dim)

        # Holographic encoder
        self.holographic = HolographicEncoder(
            input_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            curvature=curvature,
        )

        # Confidence score projection
        self.confidence_proj = nn.Linear(1, embedding_dim)

    def forward(
        self,
        protein_ids: torch.Tensor,
        adjacency: torch.Tensor,
        confidence_scores: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode PPI network.

        Args:
            protein_ids: Protein IDs (B, N)
            adjacency: Interaction adjacency matrix (B, N, N)
            confidence_scores: Optional interaction confidence (B, N, N)

        Returns:
            Dictionary with embeddings and features
        """
        # Get protein embeddings
        protein_features = self.protein_embedding(protein_ids)  # (B, N, D)

        # Modulate adjacency by confidence if available
        if confidence_scores is not None:
            adjacency = adjacency * confidence_scores

        # Holographic encoding
        result = self.holographic(adjacency, protein_features)

        return result


class HierarchicalProteinEmbedding(nn.Module):
    """Create hierarchical protein embeddings using holographic encoding.

    Maps proteins to points on Poincare ball where:
    - Core/hub proteins are near origin
    - Peripheral proteins are near boundary
    - Similar proteins are close in hyperbolic space
    """

    def __init__(
        self,
        sequence_encoder: nn.Module | None = None,
        hidden_dim: int = 128,
        output_dim: int = 16,
        curvature: float = 1.0,
    ):
        """Initialize hierarchical embedding.

        Args:
            sequence_encoder: Optional sequence encoder for protein features
            hidden_dim: Hidden dimension
            output_dim: Output dimension on Poincare ball
            curvature: Ball curvature
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.curvature = curvature

        # Holographic encoder for network structure
        self.holographic = HolographicEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            curvature=curvature,
        )

        # Sequence feature projection (if encoder provided)
        self.sequence_encoder = sequence_encoder
        if sequence_encoder is not None:
            self.seq_proj = nn.Linear(hidden_dim, hidden_dim)

        # Hierarchy predictor
        self.hierarchy_head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        adjacency: torch.Tensor,
        sequences: torch.Tensor | None = None,
        node_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Embed proteins hierarchically.

        Args:
            adjacency: Interaction network (B, N, N)
            sequences: Optional protein sequences for encoding
            node_features: Optional pre-computed node features

        Returns:
            Dictionary with hierarchical embeddings
        """
        # Get sequence features if available
        if sequences is not None and self.sequence_encoder is not None:
            seq_features = self.sequence_encoder(sequences)
            seq_features = self.seq_proj(seq_features)
            node_features = seq_features

        # Holographic encoding
        result = self.holographic(adjacency, node_features)

        # Predict hierarchy levels
        hierarchy_scores = self.hierarchy_head(result["z"]).squeeze(-1)

        result["hierarchy_scores"] = hierarchy_scores

        return result

    def compute_hierarchical_loss(
        self,
        z: torch.Tensor,
        hierarchy_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for hierarchical structure preservation.

        Args:
            z: Poincare embeddings (B, D)
            hierarchy_labels: Ground truth hierarchy levels (B,)

        Returns:
            Scalar loss
        """
        # V5.12.2: Use hyperbolic distance from origin
        origin = torch.zeros_like(z)
        hyp_dist = poincare_distance(z, origin, c=self.curvature)

        # High hierarchy = low hyperbolic distance (near origin)
        # Compute rank correlation loss
        pred_ranks = torch.argsort(torch.argsort(hyp_dist))
        true_ranks = torch.argsort(torch.argsort(hierarchy_labels))

        # MSE on ranks (simple approximation)
        rank_loss = F.mse_loss(pred_ranks.float(), true_ranks.float())

        return rank_loss
