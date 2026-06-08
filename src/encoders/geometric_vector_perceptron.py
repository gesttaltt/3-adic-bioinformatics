# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""
Geometric Vector Perceptron (GVP) with P-adic Integration.

Implementation based on Jing et al. (2020) "Learning from Protein Structure
with Geometric Vector Perceptrons" with novel p-adic distance integration
for capturing hierarchical codon/amino acid relationships.

GVP operates on both scalar and vector features, maintaining SE(3) equivariance
while incorporating ultrametric structure from p-adic biology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.core.padic_math import padic_valuation_vectorized


@dataclass
class GVPOutput:
    """Output from GVP layer."""

    scalar_features: torch.Tensor  # (batch, n, s_out)
    vector_features: torch.Tensor  # (batch, n, v_out, 3)


class VectorLinear(nn.Module):
    """
    Linear transformation for vector features.

    Maps (n, v_in, 3) -> (n, v_out, 3) while maintaining equivariance.
    """

    def __init__(self, v_in: int, v_out: int, bias: bool = False):
        super().__init__()
        self.v_in = v_in
        self.v_out = v_out
        self.weight = nn.Parameter(torch.randn(v_out, v_in) / math.sqrt(v_in))
        self.bias = nn.Parameter(torch.zeros(v_out, 3)) if bias else None

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Apply linear transformation to vectors.

        Args:
            v: Vector features (batch, n, v_in, 3)

        Returns:
            Transformed vectors (batch, n, v_out, 3)
        """
        # v: (batch, n, v_in, 3) -> (batch, n, 3, v_in)
        v = v.transpose(-2, -1)
        # Linear: (batch, n, 3, v_in) @ (v_out, v_in)^T -> (batch, n, 3, v_out)
        out = torch.matmul(v, self.weight.T)
        # Back to (batch, n, v_out, 3)
        out = out.transpose(-2, -1)

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(0)

        return out


class GVPLayer(nn.Module):
    """
    Geometric Vector Perceptron Layer.

    Processes both scalar (s) and vector (v) features while
    maintaining SE(3) equivariance for vectors.
    """

    def __init__(
        self,
        s_in: int,
        s_out: int,
        v_in: int,
        v_out: int,
        activations: tuple[bool, bool] = (True, True),
        vector_gate: bool = True,
    ):
        """
        Initialize GVP layer.

        Args:
            s_in: Input scalar feature dimension
            s_out: Output scalar feature dimension
            v_in: Input vector feature channels
            v_out: Output vector feature channels
            activations: (scalar_activation, vector_activation)
            vector_gate: Use gating for vector outputs
        """
        super().__init__()
        self.s_in = s_in
        self.s_out = s_out
        self.v_in = v_in
        self.v_out = v_out
        self.activations = activations
        self.vector_gate = vector_gate

        # Scalar path
        self.scalar_net = nn.Sequential(
            nn.Linear(s_in + v_in, s_out),
            nn.SiLU() if activations[0] else nn.Identity(),
        )

        # Vector path
        self.vector_linear = VectorLinear(v_in, v_out) if v_out > 0 else None

        # Gating for vectors (depends on scalars)
        if vector_gate and v_out > 0:
            self.gate_linear = nn.Linear(s_in + v_in, v_out)
        else:
            self.gate_linear = None

    def forward(self, s: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply GVP transformation.

        Args:
            s: Scalar features (batch, n, s_in)
            v: Vector features (batch, n, v_in, 3)

        Returns:
            (scalar_out, vector_out)
        """
        # Compute vector norms for scalar path
        v_norm = torch.norm(v, dim=-1)  # (batch, n, v_in)

        # Concatenate scalar features with vector norms
        s_cat = torch.cat([s, v_norm], dim=-1)

        # Scalar transformation
        s_out = self.scalar_net(s_cat)

        # Vector transformation
        if self.vector_linear is not None:
            v_out = self.vector_linear(v)

            if self.activations[1]:
                # Vector activation: scale by norm
                v_norm_out = torch.norm(v_out, dim=-1, keepdim=True)
                v_out = v_out * torch.sigmoid(v_norm_out)

            if self.gate_linear is not None:
                # Gating based on scalar features
                gate = torch.sigmoid(self.gate_linear(s_cat))
                v_out = v_out * gate.unsqueeze(-1)
        else:
            v_out = torch.zeros(s.shape[0], s.shape[1], 0, 3, device=s.device)

        return s_out, v_out


class GVPMessage(nn.Module):
    """
    GVP-based message passing for graph neural networks.

    Aggregates information from neighboring nodes while
    maintaining geometric equivariance.
    """

    def __init__(
        self,
        node_s: int,
        node_v: int,
        edge_s: int,
        edge_v: int,
        hidden_s: int = 64,
        hidden_v: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize GVP message passing.

        Args:
            node_s: Node scalar feature dimension
            node_v: Node vector feature channels
            edge_s: Edge scalar feature dimension
            edge_v: Edge vector feature channels
            hidden_s: Hidden scalar dimension
            hidden_v: Hidden vector channels
            dropout: Dropout probability
        """
        super().__init__()
        self.node_s = node_s
        self.node_v = node_v
        self.edge_s = edge_s
        self.edge_v = edge_v

        # Message network
        self.message_net = nn.ModuleList(
            [
                GVPLayer(node_s * 2 + edge_s, hidden_s, node_v * 2 + edge_v, hidden_v),
                GVPLayer(hidden_s, hidden_s, hidden_v, hidden_v),
                GVPLayer(hidden_s, node_s, hidden_v, node_v, activations=(False, False)),
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm_s = nn.LayerNorm(node_s)
        self.layer_norm_v = nn.LayerNorm(node_v * 3)

    def forward(
        self,
        node_s: torch.Tensor,
        node_v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_s: torch.Tensor,
        edge_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Perform message passing.

        Args:
            node_s: Node scalar features (batch, n_nodes, node_s)
            node_v: Node vector features (batch, n_nodes, node_v, 3)
            edge_index: Edge indices (batch, 2, n_edges)
            edge_s: Edge scalar features (batch, n_edges, edge_s)
            edge_v: Edge vector features (batch, n_edges, edge_v, 3)

        Returns:
            Updated (node_s, node_v)
        """
        batch_size, n_nodes = node_s.shape[:2]
        edge_index.shape[-1]

        # Gather source and target node features
        src_idx = edge_index[:, 0, :]  # (batch, n_edges)
        tgt_idx = edge_index[:, 1, :]  # (batch, n_edges)

        # Expand indices for gathering
        src_idx_s = src_idx.unsqueeze(-1).expand(-1, -1, node_s.shape[-1])
        tgt_idx_s = tgt_idx.unsqueeze(-1).expand(-1, -1, node_s.shape[-1])

        src_s = torch.gather(node_s, 1, src_idx_s)
        tgt_s = torch.gather(node_s, 1, tgt_idx_s)

        src_idx_v = src_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, node_v.shape[-2], 3)
        tgt_idx_v = tgt_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, node_v.shape[-2], 3)

        src_v = torch.gather(node_v, 1, src_idx_v)
        tgt_v = torch.gather(node_v, 1, tgt_idx_v)

        # Concatenate for message
        msg_s = torch.cat([src_s, tgt_s, edge_s], dim=-1)
        msg_v = torch.cat([src_v, tgt_v, edge_v], dim=-2)

        # Process through message network
        for layer in self.message_net:
            msg_s, msg_v = layer(msg_s, msg_v)

        # Aggregate messages at target nodes
        agg_s = torch.zeros(batch_size, n_nodes, self.node_s, device=node_s.device)
        agg_v = torch.zeros(batch_size, n_nodes, self.node_v, 3, device=node_v.device)

        tgt_idx_s = tgt_idx.unsqueeze(-1).expand(-1, -1, self.node_s)
        tgt_idx_v = tgt_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.node_v, 3)

        agg_s.scatter_add_(1, tgt_idx_s, msg_s)
        agg_v.scatter_add_(1, tgt_idx_v, msg_v)

        # Residual and normalization
        node_s = self.layer_norm_s(node_s + self.dropout(agg_s))
        # Reshape for LayerNorm
        v_shape = node_v.shape
        node_v_flat = node_v.view(batch_size, n_nodes, -1)
        agg_v_flat = agg_v.view(batch_size, n_nodes, -1)
        node_v = self.layer_norm_v(node_v_flat + self.dropout(agg_v_flat))
        node_v = node_v.view(v_shape)

        return node_s, node_v


class PAdicGVP(nn.Module):
    """
    P-adic Geometric Vector Perceptron.

    Extends GVP with p-adic distance integration for biological
    applications. Incorporates ultrametric structure into the
    geometric representation.
    """

    def __init__(
        self,
        s_in: int = 64,
        s_out: int = 64,
        v_in: int = 8,
        v_out: int = 8,
        p: int = 3,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize P-adic GVP.

        Args:
            s_in: Input scalar dimension
            s_out: Output scalar dimension
            v_in: Input vector channels
            v_out: Output vector channels
            p: Prime for p-adic valuation (default 3 for codons)
            n_layers: Number of GVP layers
            dropout: Dropout probability
        """
        super().__init__()
        self.s_in = s_in
        self.s_out = s_out
        self.v_in = v_in
        self.v_out = v_out
        self.p = p
        self.n_layers = n_layers

        # P-adic feature encoder
        self.padic_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.SiLU(),
            nn.Linear(16, s_in),
        )

        # GVP layers
        self.gvp_layers = nn.ModuleList()
        for i in range(n_layers):
            s_layer_in = s_in if i == 0 else s_out
            v_layer_in = v_in if i == 0 else v_out
            is_last = i == n_layers - 1
            self.gvp_layers.append(
                GVPLayer(
                    s_layer_in,
                    s_out,
                    v_layer_in,
                    v_out,
                    activations=(not is_last, not is_last),
                )
            )

        self.dropout = nn.Dropout(dropout)

        # P-adic attention for distance-aware processing
        self.padic_attention = nn.MultiheadAttention(s_out, num_heads=4, dropout=dropout, batch_first=True)

    def compute_padic_valuation(self, x: torch.Tensor) -> torch.Tensor:
        """Compute p-adic valuation of pairwise differences.

        Uses centralized padic_valuation_vectorized from src.core.padic_math.
        """
        # x: (batch, n, d) -> pairwise differences
        diff = x.unsqueeze(-2) - x.unsqueeze(-3)  # (batch, n, n, d)
        diff_int = (diff.abs() * 1000).long()  # Scale and discretize

        # Use centralized vectorized valuation
        valuation = padic_valuation_vectorized(diff_int, self.p)

        # Average across dimensions
        return valuation.float().mean(dim=-1)

    def padic_distance(self, x: torch.Tensor) -> torch.Tensor:
        """Compute p-adic distances."""
        valuation = self.compute_padic_valuation(x)
        return torch.pow(torch.tensor(float(self.p), device=x.device), -valuation)

    def forward(self, s: torch.Tensor, v: torch.Tensor, coords: torch.Tensor | None = None) -> dict[str, Any]:
        """
        Apply P-adic GVP.

        Args:
            s: Scalar features (batch, n, s_in)
            v: Vector features (batch, n, v_in, 3)
            coords: Optional 3D coordinates for p-adic distance (batch, n, 3)

        Returns:
            Dictionary with processed features and p-adic metrics
        """
        # Compute p-adic distances if coordinates provided
        if coords is not None:
            padic_dist = self.padic_distance(coords)
            padic_features = self.padic_encoder(padic_dist.mean(dim=-1, keepdim=True))
            s = s + padic_features
        else:
            padic_dist = None

        # Apply GVP layers
        for i, layer in enumerate(self.gvp_layers):
            s_new, v_new = layer(s, v)
            if i > 0:  # Skip connection after first layer
                s = s + self.dropout(s_new) if s.shape == s_new.shape else s_new
                v = v + self.dropout(v_new) if v.shape == v_new.shape else v_new
            else:
                s, v = s_new, v_new

        # P-adic attention (uses distance as bias if available)
        if padic_dist is not None:
            # Convert p-adic distance to attention bias
            # Need to expand for multi-head attention: (batch * n_heads, seq, seq)
            batch_size, seq_len = padic_dist.shape[:2]
            n_heads = self.padic_attention.num_heads
            attn_bias = -padic_dist.unsqueeze(1).expand(-1, n_heads, -1, -1)
            attn_bias = attn_bias.reshape(batch_size * n_heads, seq_len, seq_len)
            s_attn, _ = self.padic_attention(s, s, s, attn_mask=attn_bias)
            s = s + self.dropout(s_attn)

        return {
            "scalar_features": s,
            "vector_features": v,
            "padic_distances": padic_dist,
        }


class ProteinGVPEncoder(nn.Module):
    """
    Complete GVP encoder for protein structure.

    Takes backbone coordinates and amino acid features,
    produces hierarchical representations.
    """

    def __init__(
        self,
        node_s_in: int = 20,  # Amino acid one-hot
        node_v_in: int = 1,  # CA direction
        hidden_s: int = 64,
        hidden_v: int = 16,
        output_dim: int = 128,
        n_layers: int = 3,
        use_padic: bool = True,
    ):
        """
        Initialize protein GVP encoder.

        Args:
            node_s_in: Input node scalar dimension (e.g., amino acid features)
            node_v_in: Input node vector channels
            hidden_s: Hidden scalar dimension
            hidden_v: Hidden vector channels
            output_dim: Output embedding dimension
            n_layers: Number of GVP layers
            use_padic: Whether to use p-adic distance integration
        """
        super().__init__()
        self.node_s_in = node_s_in
        self.node_v_in = node_v_in
        self.hidden_s = hidden_s
        self.hidden_v = hidden_v
        self.output_dim = output_dim
        self.use_padic = use_padic

        # Input projection
        self.input_proj_s = nn.Linear(node_s_in, hidden_s)
        self.input_proj_v = VectorLinear(node_v_in, hidden_v)

        # GVP backbone
        if use_padic:
            self.gvp_backbone = PAdicGVP(
                s_in=hidden_s,
                s_out=hidden_s,
                v_in=hidden_v,
                v_out=hidden_v,
                n_layers=n_layers,
            )
        else:
            self.gvp_layers = nn.ModuleList([GVPLayer(hidden_s, hidden_s, hidden_v, hidden_v) for _ in range(n_layers)])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_s + hidden_v * 3, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

        # Pooling attention
        self.pool_attention = nn.Sequential(
            nn.Linear(hidden_s, 1),
            nn.Softmax(dim=1),
        )

    def forward(
        self,
        node_s: torch.Tensor,
        node_v: torch.Tensor,
        coords: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """
        Encode protein structure.

        Args:
            node_s: Node scalar features (batch, n_residues, node_s_in)
            node_v: Node vector features (batch, n_residues, node_v_in, 3)
            coords: Optional CA coordinates (batch, n_residues, 3)
            mask: Optional mask for valid residues (batch, n_residues)

        Returns:
            Dictionary with protein embeddings
        """
        # Project inputs
        s = self.input_proj_s(node_s)
        v = self.input_proj_v(node_v)

        # Apply GVP backbone
        if self.use_padic:
            result = self.gvp_backbone(s, v, coords)
            s = result["scalar_features"]
            v = result["vector_features"]
            padic_dist = result["padic_distances"]
        else:
            for layer in self.gvp_layers:
                s, v = layer(s, v)
            padic_dist = None

        # Combine scalar and vector features
        v_flat = v.reshape(v.shape[0], v.shape[1], -1)
        features = torch.cat([s, v_flat], dim=-1)

        # Project to output dimension
        node_embeddings = self.output_proj(features)

        # Attention-weighted pooling
        attn_weights = self.pool_attention(s)
        if mask is not None:
            attn_weights = attn_weights * mask.unsqueeze(-1)
            attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-10)

        protein_embedding = (node_embeddings * attn_weights).sum(dim=1)

        return {
            "node_embeddings": node_embeddings,
            "protein_embedding": protein_embedding,
            "attention_weights": attn_weights.squeeze(-1),
            "padic_distances": padic_dist,
            "scalar_features": s,
            "vector_features": v,
        }


class CodonGVP(nn.Module):
    """
    GVP encoder specialized for codon sequences.

    Maps codons to geometric representations using
    p-adic ternary structure.
    """

    def __init__(
        self,
        n_codons: int = 64,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 16,
        n_layers: int = 2,
    ):
        """
        Initialize codon GVP.

        Args:
            n_codons: Number of codon types (64 standard)
            embedding_dim: Codon embedding dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            n_layers: Number of GVP layers
        """
        super().__init__()
        self.n_codons = n_codons
        self.embedding_dim = embedding_dim

        # Codon embeddings
        self.codon_embedding = nn.Embedding(n_codons, embedding_dim)

        # Initialize with p-adic structure
        self._init_padic_embeddings()

        # Create virtual vectors from ternary structure
        self.ternary_to_vector = nn.Linear(4, 3)  # 4 ternary digits -> 3D vector

        # GVP backbone
        self.gvp = PAdicGVP(
            s_in=embedding_dim,
            s_out=hidden_dim,
            v_in=1,  # Single vector per codon
            v_out=4,
            n_layers=n_layers,
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim + 4 * 3, output_dim)

    def _init_padic_embeddings(self):
        """Initialize embeddings with p-adic structure."""
        with torch.no_grad():
            for idx in range(self.n_codons):
                # Convert to ternary
                ternary = []
                n = idx
                for _ in range(4):
                    ternary.append(n % 3)
                    n //= 3

                # Use ternary digits to initialize
                torch.tensor(ternary, dtype=torch.float32)
                # Embed in higher dimension
                embedding = torch.zeros(self.embedding_dim)
                for i, digit in enumerate(ternary):
                    start = i * (self.embedding_dim // 4)
                    end = (i + 1) * (self.embedding_dim // 4)
                    embedding[start:end] = (digit - 1) / 2  # Center around 0

                self.codon_embedding.weight[idx] = embedding

    def codon_to_ternary(self, codon_idx: torch.Tensor) -> torch.Tensor:
        """Convert codon indices to ternary representation."""
        batch_shape = codon_idx.shape
        flat_idx = codon_idx.view(-1)

        ternary = torch.zeros(flat_idx.shape[0], 4, device=codon_idx.device)
        for i in range(4):
            ternary[:, i] = (flat_idx // (3**i)) % 3

        return ternary.view(*batch_shape, 4)

    def forward(self, codon_indices: torch.Tensor) -> dict[str, Any]:
        """
        Encode codon sequence.

        Args:
            codon_indices: Codon indices (batch, seq_len)

        Returns:
            Dictionary with codon embeddings
        """
        # Get scalar features from embedding
        s = self.codon_embedding(codon_indices)

        # Create vectors from ternary structure
        ternary = self.codon_to_ternary(codon_indices)
        v = self.ternary_to_vector(ternary)
        v = v.unsqueeze(-2)  # (batch, seq, 1, 3)

        # Apply GVP
        result = self.gvp(s, v)

        s_out = result["scalar_features"]
        v_out = result["vector_features"]

        # Combine for output
        v_flat = v_out.reshape(v_out.shape[0], v_out.shape[1], -1)
        combined = torch.cat([s_out, v_flat], dim=-1)
        output = self.output_proj(combined)

        return {
            "codon_embeddings": output,
            "scalar_features": s_out,
            "vector_features": v_out,
            "ternary_representation": ternary,
            "padic_distances": result["padic_distances"],
        }
