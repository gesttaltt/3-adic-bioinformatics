# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Structure-conditioned sequence generation using diffusion.

This module provides models for generating codon sequences conditioned
on protein structure, enabling inverse folding at the codon level.

References:
    - Ingraham et al., "Generative Models for Graph-Based Protein Design" (2019)
    - Dauparas et al., "Robust deep learning based protein sequence design" (2022)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .codon_diffusion import CodonDiffusion


class StructureEncoder(nn.Module):
    """Encoder for protein backbone structure.

    Encodes Ca coordinates and backbone geometry into per-residue
    features suitable for conditioning sequence generation.

    Args:
        hidden_dim: Hidden dimension
        n_layers: Number of GNN layers
        n_neighbors: Number of spatial neighbors to consider
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_neighbors: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_neighbors = n_neighbors

        # Initial features from geometry
        self.rbf = RadialBasisEmbedding(num_rbf=16, max_dist=20.0)
        self.angle_embed = nn.Linear(8, hidden_dim)  # sin/cos of backbone angles

        # Position embedding
        self.pos_embed = nn.Linear(hidden_dim + 16, hidden_dim)

        # Structure GNN layers
        self.layers = nn.ModuleList([StructureGNNLayer(hidden_dim, dropout) for _ in range(n_layers)])

        # Output projection
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        coords: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode backbone structure.

        Args:
            coords: Ca coordinates of shape (batch, n_residues, 3)
            mask: Valid residue mask of shape (batch, n_residues)

        Returns:
            Tuple of (structure_features, edge_index)
        """
        batch_size, n_residues, _ = coords.shape
        device = coords.device

        if mask is None:
            mask = torch.ones(batch_size, n_residues, dtype=torch.bool, device=device)

        # Compute pairwise distances
        dist = torch.cdist(coords, coords)  # (batch, n, n)

        # Build k-nearest neighbor graph
        edge_index, edge_attr = self._build_graph(dist, coords, mask)

        # Initial node features from local geometry
        h = self._initial_features(coords, mask)

        # Apply GNN layers
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, mask)

        # Output projection
        h = self.output(h)

        # Zero out masked positions
        h = h * mask.unsqueeze(-1).float()

        return h, edge_index

    def _build_graph(
        self,
        dist: Tensor,
        coords: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Build k-NN spatial graph.

        Args:
            dist: Pairwise distances
            coords: Coordinates
            mask: Valid residue mask

        Returns:
            Tuple of (edge_index, edge_features)
        """
        batch_size, n_residues, _ = coords.shape
        device = coords.device

        # Mask invalid distances
        large_val = 1e10
        dist_masked = dist.clone()
        dist_masked[~mask.unsqueeze(-1).expand_as(dist)] = large_val
        dist_masked[~mask.unsqueeze(-2).expand_as(dist)] = large_val

        # Exclude self-loops
        dist_masked += torch.eye(n_residues, device=device).unsqueeze(0) * large_val

        # Get k-nearest neighbors
        k = min(self.n_neighbors, n_residues - 1)
        _, indices = torch.topk(dist_masked, k, dim=-1, largest=False)

        # Build edge index (flattened for batching)
        src = torch.arange(n_residues, device=device).unsqueeze(-1).expand(-1, k)
        src = src.unsqueeze(0).expand(batch_size, -1, -1)

        # Stack as edge index
        edges = []
        edge_features_list = []

        for b in range(batch_size):
            b_src = src[b].flatten() + b * n_residues
            b_dst = indices[b].flatten() + b * n_residues
            edges.append(torch.stack([b_src, b_dst]))

            # Edge features: RBF of distance
            b_dist = dist[b, src[b].flatten() % n_residues, indices[b].flatten() % n_residues]
            edge_features_list.append(self.rbf(b_dist))

        edge_index = torch.cat(edges, dim=1)
        edge_attr = torch.cat(edge_features_list, dim=0)

        return edge_index, edge_attr

    def _initial_features(self, coords: Tensor, mask: Tensor) -> Tensor:
        """Compute initial node features from geometry."""
        batch_size, n_residues, _ = coords.shape

        # Compute backbone angles (phi, psi approximations)
        angles = self._compute_backbone_angles(coords)

        # Combine with positional encoding
        angle_embed = self.angle_embed(angles)

        # Add RBF of distance to neighbors
        dist = torch.cdist(coords, coords)
        mean_dist = dist.mean(dim=-1)
        dist_rbf = self.rbf(mean_dist)

        # Combine features
        features = torch.cat([angle_embed, dist_rbf], dim=-1)
        features = self.pos_embed(features)

        return features * mask.unsqueeze(-1).float()

    def _compute_backbone_angles(self, coords: Tensor) -> Tensor:
        """Compute backbone angle features."""
        batch_size, n_residues, _ = coords.shape

        # Vectors between consecutive residues
        v1 = coords[:, 1:] - coords[:, :-1]  # (batch, n-1, 3)
        v1 = F.normalize(v1, dim=-1)

        # Pad to maintain sequence length
        v1_padded = F.pad(v1, (0, 0, 0, 1))  # Forward
        v1_shifted = F.pad(v1, (0, 0, 1, 0))  # Backward

        # Compute angles from dot products
        cos_angle = (v1_padded * v1_shifted).sum(dim=-1).clamp(-1, 1)
        sin_angle = torch.sqrt(1 - cos_angle**2)

        # Stack sin/cos for multiple angle types
        angles = torch.stack(
            [
                torch.sin(torch.acos(cos_angle)),
                torch.cos(torch.acos(cos_angle)),
                cos_angle,
                sin_angle,
            ],
            dim=-1,
        )

        # Pad to 8 features
        angles = F.pad(angles, (0, 4))

        return angles


class RadialBasisEmbedding(nn.Module):
    """RBF embedding for distances."""

    def __init__(self, num_rbf: int = 16, max_dist: float = 20.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.max_dist = max_dist

        # RBF centers
        centers = torch.linspace(0, max_dist, num_rbf)
        self.register_buffer("centers", centers)

        # RBF widths
        widths = torch.ones(num_rbf) * (max_dist / num_rbf)
        self.register_buffer("widths", widths)

    def forward(self, dist: Tensor) -> Tensor:
        """Compute RBF embedding.

        Args:
            dist: Distances of shape (...)

        Returns:
            RBF values of shape (..., num_rbf)
        """
        dist = dist.unsqueeze(-1)
        return torch.exp(-((dist - self.centers) ** 2) / (2 * self.widths**2))


class StructureGNNLayer(nn.Module):
    """Single GNN layer for structure encoding."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Message network
        self.message_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + 16, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Update network
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        mask: Tensor,
    ) -> Tensor:
        """Forward pass.

        Args:
            h: Node features (batch * n_nodes, hidden_dim)
            edge_index: Edge indices (2, n_edges)
            edge_attr: Edge features (n_edges, 16)
            mask: Valid node mask (batch, n_nodes)

        Returns:
            Updated node features
        """
        batch_size, n_nodes, _ = h.shape
        h_flat = h.view(-1, self.hidden_dim)

        # Message passing
        src, dst = edge_index
        h_src = h_flat[src]
        h_dst = h_flat[dst]

        # Compute messages
        msg_input = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        messages = self.message_net(msg_input)

        # Aggregate
        aggr = torch.zeros_like(h_flat)
        aggr.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        # Update
        h_new = h_flat + self.update_net(torch.cat([h_flat, aggr], dim=-1))
        h_new = self.norm(h_new)

        # Reshape and mask
        h_new = h_new.view(batch_size, n_nodes, self.hidden_dim)
        h_new = h_new * mask.unsqueeze(-1).float()

        return h_new


class StructureConditionedGen(nn.Module):
    """Structure-to-sequence generation using conditional diffusion.

    Given protein backbone structure, generates compatible codon sequences.
    Useful for:
    - Inverse folding at codon level
    - Codon optimization for given structure
    - Exploring sequence space for a structure

    Args:
        hidden_dim: Hidden dimension
        n_diffusion_steps: Number of diffusion steps
        n_layers: Number of transformer layers
        vocab_size: Codon vocabulary size (64)
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_diffusion_steps: int = 1000,
        n_layers: int = 6,
        vocab_size: int = 64,
        n_encoder_layers: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # Structure encoder
        self.structure_encoder = StructureEncoder(
            hidden_dim=hidden_dim,
            n_layers=n_encoder_layers,
        )

        # Conditional diffusion model
        self.diffusion = CodonDiffusion(
            n_steps=n_diffusion_steps,
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

        # Cross-attention for conditioning
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        coords: Tensor,
        codons: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Training forward pass.

        Args:
            coords: Backbone Ca coordinates (batch, n_residues, 3)
            codons: Ground truth codon indices (batch, seq_len)
                    Note: seq_len = n_residues * 3 for all 3 codons per residue
            mask: Valid residue mask (batch, n_residues)

        Returns:
            Dictionary with loss and metrics
        """
        # Encode structure
        structure_features, _ = self.structure_encoder(coords, mask)

        # Expand structure features to match codon sequence length
        # Each residue has 1 codon, so we need to align
        # For now, assume 1 codon per residue
        context = structure_features

        # Train diffusion with structure context
        return self.diffusion.forward(codons, context=context)

    @torch.no_grad()
    def design(
        self,
        coords: Tensor,
        mask: Tensor | None = None,
        n_designs: int = 10,
        temperature: float = 1.0,
    ) -> Tensor:
        """Design sequences for given structure.

        Args:
            coords: Backbone coordinates (batch, n_residues, 3)
            mask: Valid residue mask
            n_designs: Number of sequences to generate per structure
            temperature: Sampling temperature

        Returns:
            Designed codon sequences (batch * n_designs, seq_len)
        """
        batch_size, n_residues, _ = coords.shape
        device = coords.device

        # Encode structure
        structure_features, _ = self.structure_encoder(coords, mask)

        # Expand for multiple designs
        structure_features = structure_features.repeat_interleave(n_designs, dim=0)

        # Generate sequences
        seq_len = n_residues  # 1 codon per residue
        sequences = self.diffusion.sample(
            n_samples=batch_size * n_designs,
            seq_length=seq_len,
            temperature=temperature,
            context=structure_features,
            device=device,
        )

        return sequences

    def sample(
        self,
        coords: Tensor,
        mask: Tensor | None = None,
        n_samples: int = 1,
        **kwargs,
    ) -> Tensor:
        """Alias for design method."""
        return self.design(coords, mask, n_designs=n_samples, **kwargs)


class MultiObjectiveDesigner(nn.Module):
    """Multi-objective codon sequence designer.

    Combines structure compatibility with other objectives like:
    - Codon usage bias
    - mRNA stability
    - Expression level
    - Avoiding unwanted motifs

    Args:
        structure_gen: Structure-conditioned generator
        use_codon_bias: Whether to incorporate codon usage bias
        use_mrna_stability: Whether to consider mRNA stability
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        vocab_size: int = 64,
        use_codon_bias: bool = True,
        use_mrna_stability: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.use_codon_bias = use_codon_bias
        self.use_mrna_stability = use_mrna_stability

        # Main structure-conditioned generator
        self.structure_gen = StructureConditionedGen(
            hidden_dim=hidden_dim,
            vocab_size=vocab_size,
        )

        # Objective predictors
        if use_codon_bias:
            self.codon_bias_predictor = nn.Sequential(
                nn.Linear(vocab_size, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

        if use_mrna_stability:
            self.stability_predictor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

    def forward(
        self,
        coords: Tensor,
        codons: Tensor,
        mask: Tensor | None = None,
        weights: dict[str, float] | None = None,
    ) -> dict[str, Tensor]:
        """Multi-objective training.

        Args:
            coords: Structure coordinates
            codons: Target codon sequences
            mask: Valid residue mask
            weights: Objective weights

        Returns:
            Dictionary with losses
        """
        if weights is None:
            weights = {"structure": 1.0, "codon_bias": 0.1, "stability": 0.1}

        # Main loss
        result = self.structure_gen.forward(coords, codons, mask)
        total_loss = weights["structure"] * result["loss"]

        # Codon bias loss
        if self.use_codon_bias:
            logits = result["logits"]
            bias_score = self.codon_bias_predictor(logits.softmax(dim=-1))
            result["codon_bias_loss"] = -bias_score.mean()  # Maximize bias score
            total_loss = total_loss + weights.get("codon_bias", 0.1) * result["codon_bias_loss"]

        result["total_loss"] = total_loss
        return result

    @torch.no_grad()
    def design_optimized(
        self,
        coords: Tensor,
        mask: Tensor | None = None,
        n_candidates: int = 100,
        n_select: int = 10,
        **kwargs,
    ) -> Tensor:
        """Design sequences with multi-objective selection.

        Args:
            coords: Structure coordinates
            mask: Valid mask
            n_candidates: Number of candidates to generate
            n_select: Number to select after scoring

        Returns:
            Best sequences according to objectives
        """
        # Generate candidates
        candidates = self.structure_gen.design(coords, mask, n_designs=n_candidates, **kwargs)

        # Score candidates
        scores = torch.zeros(n_candidates, device=coords.device)

        if self.use_codon_bias:
            one_hot = F.one_hot(candidates, self.vocab_size).float()
            bias_scores = self.codon_bias_predictor(one_hot).squeeze(-1).mean(dim=-1)
            scores = scores + bias_scores

        # Select best
        _, best_indices = torch.topk(scores, n_select)
        return candidates[best_indices]
