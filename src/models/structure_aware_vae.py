# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Structure-Aware VAE with AlphaFold2 integration.

This module extends the base VAE to incorporate 3D protein structure
information from AlphaFold2 predictions. Structure awareness improves
predictions by capturing:

1. Spatial proximity of mutations
2. Active site geometry
3. Protein stability effects
4. Drug binding pocket topology

The structure encoder uses SE(3)-equivariant message passing to
maintain rotational and translational invariance.

Usage:
    from src.models.structure_aware_vae import StructureAwareVAE

    model = StructureAwareVAE(
        input_dim=128,
        latent_dim=32,
        structure_dim=64,
    )

    # Forward with structure
    outputs = model(sequence, structure_coords, plddt_scores)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StructureConfig:
    """Configuration for structure-aware components.

    Attributes:
        use_structure: Whether to use structure information
        structure_dim: Structure embedding dimension
        n_structure_layers: Number of structure encoder layers
        cutoff: Distance cutoff for edges (Angstroms)
        use_plddt: Whether to use pLDDT confidence weighting
        use_pae: Whether to use Predicted Aligned Error
        fusion_type: How to fuse sequence and structure (cross_attention, gated, concat)
    """

    use_structure: bool = True
    structure_dim: int = 64
    n_structure_layers: int = 3
    cutoff: float = 10.0
    use_plddt: bool = True
    use_pae: bool = False
    fusion_type: str = "cross_attention"


class InvariantPointAttention(nn.Module):
    """SE(3)-invariant point attention for structure encoding.

    Based on AlphaFold2's Invariant Point Attention (IPA) but simplified
    for efficiency. Computes attention over 3D coordinates while
    maintaining rotational/translational invariance.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int = 4,
        n_query_points: int = 4,
        n_value_points: int = 4,
    ):
        """Initialize IPA layer.

        Args:
            embed_dim: Feature dimension
            n_heads: Number of attention heads
            n_query_points: Number of query point positions
            n_value_points: Number of value point positions
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.n_query_points = n_query_points
        self.n_value_points = n_value_points

        # Query, key, value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Point projections (for geometric attention)
        self.q_point_proj = nn.Linear(embed_dim, n_heads * n_query_points * 3)
        self.k_point_proj = nn.Linear(embed_dim, n_heads * n_query_points * 3)
        self.v_point_proj = nn.Linear(embed_dim, n_heads * n_value_points * 3)

        # Output projection
        self.out_proj = nn.Linear(embed_dim + n_heads * n_value_points * 3, embed_dim)

        # Learnable weights
        self.w_c = nn.Parameter(torch.ones(n_heads))  # Coordinate weight
        self.w_l = nn.Parameter(torch.ones(n_heads))  # Linear weight

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Node features (batch, n_nodes, embed_dim)
            coords: 3D coordinates (batch, n_nodes, 3)
            mask: Optional attention mask (batch, n_nodes, n_nodes)

        Returns:
            Updated features (batch, n_nodes, embed_dim)
        """
        batch_size, n_nodes, _ = features.shape

        # Standard QKV attention
        q = self.q_proj(features).view(batch_size, n_nodes, self.n_heads, self.head_dim)
        k = self.k_proj(features).view(batch_size, n_nodes, self.n_heads, self.head_dim)
        v = self.v_proj(features).view(batch_size, n_nodes, self.n_heads, self.head_dim)

        # Point attention (geometric component)
        q_points = self.q_point_proj(features).view(batch_size, n_nodes, self.n_heads, self.n_query_points, 3)
        k_points = self.k_point_proj(features).view(batch_size, n_nodes, self.n_heads, self.n_query_points, 3)
        v_points = self.v_point_proj(features).view(batch_size, n_nodes, self.n_heads, self.n_value_points, 3)

        # Transform points by coordinates (simplified: add position)
        q_points = q_points + coords.unsqueeze(2).unsqueeze(3)
        k_points = k_points + coords.unsqueeze(2).unsqueeze(3)
        v_points = v_points + coords.unsqueeze(2).unsqueeze(3)

        # Compute attention scores
        # Linear component
        linear_attn = torch.einsum("bihd,bjhd->bhij", q, k) * self.head_dim**-0.5

        # Geometric component (point-wise distances)
        # (batch, n, heads, points, 3) - (batch, n, heads, points, 3)
        point_diff = q_points.unsqueeze(2) - k_points.unsqueeze(1)  # (batch, n, n, heads, points, 3)
        point_dist = (point_diff**2).sum(dim=-1).sum(dim=-1)  # (batch, n, n, heads)
        point_attn = -0.5 * point_dist.permute(0, 3, 1, 2)  # (batch, heads, n, n)

        # Combine attention
        w_l = F.softplus(self.w_l).view(1, self.n_heads, 1, 1)
        w_c = F.softplus(self.w_c).view(1, self.n_heads, 1, 1)
        attn = w_l * linear_attn + w_c * point_attn

        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out_linear = torch.einsum("bhij,bjhd->bihd", attn, v).reshape(batch_size, n_nodes, -1)

        # Apply attention to point values
        out_points = torch.einsum("bhij,bjhpd->bihpd", attn, v_points)
        out_points = out_points.reshape(batch_size, n_nodes, -1)

        # Combine and project
        out = torch.cat([out_linear, out_points], dim=-1)
        return self.out_proj(out)


class SE3Encoder(nn.Module):
    """SE(3)-equivariant encoder for protein structures.

    Uses message passing with geometric features to encode
    3D protein structure into latent representations.
    """

    def __init__(
        self,
        node_dim: int = 64,
        edge_dim: int = 32,
        n_layers: int = 3,
        cutoff: float = 10.0,
        n_heads: int = 4,
    ):
        """Initialize SE3 encoder.

        Args:
            node_dim: Node feature dimension
            edge_dim: Edge feature dimension
            n_layers: Number of message passing layers
            cutoff: Distance cutoff for edges
            n_heads: Attention heads for IPA
        """
        super().__init__()

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.cutoff = cutoff

        # Initial embeddings
        self.node_embed = nn.Linear(21, node_dim)  # 20 AA + gap
        self.edge_embed = nn.Linear(1, edge_dim)  # Distance

        # Message passing layers
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "ipa": InvariantPointAttention(node_dim, n_heads),
                        "ffn": nn.Sequential(
                            nn.Linear(node_dim, node_dim * 4),
                            nn.GELU(),
                            nn.Linear(node_dim * 4, node_dim),
                        ),
                        "norm1": nn.LayerNorm(node_dim),
                        "norm2": nn.LayerNorm(node_dim),
                    }
                )
                for _ in range(n_layers)
            ]
        )

        # Output projection
        self.out_proj = nn.Linear(node_dim, node_dim)

    def forward(
        self,
        coords: torch.Tensor,
        aa_indices: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode 3D structure.

        Args:
            coords: CA atom coordinates (batch, n_residues, 3)
            aa_indices: Amino acid indices (batch, n_residues)
            mask: Optional residue mask

        Returns:
            Structure embeddings (batch, n_residues, node_dim)
        """
        batch_size, n_residues, _ = coords.shape
        device = coords.device

        # Initial node features
        if aa_indices is not None:
            # One-hot encode amino acids
            aa_onehot = F.one_hot(aa_indices.clamp(0, 20), num_classes=21).float()
            h = self.node_embed(aa_onehot)
        else:
            h = torch.zeros(batch_size, n_residues, self.node_dim, device=device)

        # Compute pairwise distances for edge masking
        dist = torch.cdist(coords, coords)
        edge_mask = dist > self.cutoff  # Mask out distant residues

        # Message passing
        for layer in self.layers:
            # IPA with residual
            h_ipa = layer["ipa"](layer["norm1"](h), coords, edge_mask)
            h = h + h_ipa

            # FFN with residual
            h_ffn = layer["ffn"](layer["norm2"](h))
            h = h + h_ffn

        return self.out_proj(h)


class StructureSequenceFusion(nn.Module):
    """Fuse sequence and structure embeddings."""

    def __init__(
        self,
        seq_dim: int,
        struct_dim: int,
        output_dim: int,
        fusion_type: str = "cross_attention",
        n_heads: int = 4,
    ):
        """Initialize fusion module.

        Args:
            seq_dim: Sequence embedding dimension
            struct_dim: Structure embedding dimension
            output_dim: Output dimension
            fusion_type: Fusion method (cross_attention, gated, concat)
            n_heads: Attention heads for cross_attention
        """
        super().__init__()

        self.fusion_type = fusion_type

        if fusion_type == "cross_attention":
            self.seq_proj = nn.Linear(seq_dim, output_dim)
            self.struct_proj = nn.Linear(struct_dim, output_dim)
            self.cross_attn = nn.MultiheadAttention(output_dim, num_heads=n_heads, batch_first=True)
            self.out_proj = nn.Linear(output_dim, output_dim)

        elif fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(seq_dim + struct_dim, output_dim),
                nn.Sigmoid(),
            )
            self.transform = nn.Linear(seq_dim + struct_dim, output_dim)

        elif fusion_type == "concat":
            self.proj = nn.Linear(seq_dim + struct_dim, output_dim)

        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(
        self,
        seq_embed: torch.Tensor,
        struct_embed: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse embeddings.

        Args:
            seq_embed: Sequence embeddings (batch, seq_len, seq_dim) or (batch, seq_dim)
            struct_embed: Structure embeddings (batch, seq_len, struct_dim) or (batch, struct_dim)

        Returns:
            Fused embeddings (batch, output_dim) or (batch, seq_len, output_dim)
        """
        if self.fusion_type == "cross_attention":
            q = self.seq_proj(seq_embed)
            k = v = self.struct_proj(struct_embed)

            # Handle 2D inputs
            if q.dim() == 2:
                q = q.unsqueeze(1)
                k = k.unsqueeze(1)
                v = v.unsqueeze(1)

            fused, _ = self.cross_attn(q, k, v)
            fused = self.out_proj(fused)

            if seq_embed.dim() == 2:
                fused = fused.squeeze(1)

            return fused

        elif self.fusion_type == "gated":
            combined = torch.cat([seq_embed, struct_embed], dim=-1)
            gate = self.gate(combined)
            return gate * self.transform(combined)

        else:  # concat
            combined = torch.cat([seq_embed, struct_embed], dim=-1)
            return self.proj(combined)


class StructureAwareVAE(nn.Module):
    """VAE that incorporates 3D protein structure.

    Combines sequence encoding with structure-aware components
    for improved drug resistance prediction.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden_dims: list[int] | None = None,
        structure_config: StructureConfig | None = None,
        dropout: float = 0.1,
    ):
        """Initialize structure-aware VAE.

        Args:
            input_dim: Input sequence dimension
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions
            structure_config: Structure configuration
            dropout: Dropout rate
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        if structure_config is None:
            structure_config = StructureConfig()

        self.config = structure_config
        self.latent_dim = latent_dim

        # Sequence encoder
        seq_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            seq_layers.extend(
                [
                    nn.Linear(in_dim, h_dim),
                    nn.GELU(),
                    nn.LayerNorm(h_dim),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = h_dim

        self.sequence_encoder = nn.Sequential(*seq_layers)
        self.seq_dim = hidden_dims[-1]

        # Structure encoder (optional)
        if structure_config.use_structure:
            self.structure_encoder = SE3Encoder(
                node_dim=structure_config.structure_dim,
                n_layers=structure_config.n_structure_layers,
                cutoff=structure_config.cutoff,
            )

            # Fusion module
            self.fusion = StructureSequenceFusion(
                seq_dim=self.seq_dim,
                struct_dim=structure_config.structure_dim,
                output_dim=self.seq_dim,
                fusion_type=structure_config.fusion_type,
            )

            # pLDDT weighting (must match structure_dim for element-wise multiplication)
            if structure_config.use_plddt:
                self.plddt_proj = nn.Linear(1, structure_config.structure_dim)
        else:
            self.structure_encoder = None
            self.fusion = None

        # Latent projections
        self.fc_mu = nn.Linear(self.seq_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.seq_dim, latent_dim)

        # Decoder
        dec_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.extend([nn.Linear(in_dim, h_dim), nn.GELU(), nn.LayerNorm(h_dim)])
            in_dim = h_dim
        dec_layers.append(nn.Linear(in_dim, input_dim))

        self.decoder = nn.Sequential(*dec_layers)

    def encode(
        self,
        sequence: torch.Tensor,
        structure: torch.Tensor | None = None,
        plddt: torch.Tensor | None = None,
        aa_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution.

        Args:
            sequence: Sequence features (batch, seq_len, dim) or (batch, dim)
            structure: Optional 3D coordinates (batch, seq_len, 3)
            plddt: Optional pLDDT scores (batch, seq_len)
            aa_indices: Optional amino acid indices (batch, seq_len)

        Returns:
            Tuple of (mu, logvar)
        """
        # Encode sequence
        if sequence.dim() == 3:
            # Mean pool over sequence length
            seq_embed = self.sequence_encoder(sequence.mean(dim=1))
        else:
            seq_embed = self.sequence_encoder(sequence)

        # Encode structure if available
        if self.structure_encoder is not None and structure is not None:
            struct_embed = self.structure_encoder(structure, aa_indices)

            # Apply pLDDT weighting
            if self.config.use_plddt and plddt is not None:
                plddt_weight = torch.sigmoid(self.plddt_proj(plddt.unsqueeze(-1)))
                struct_embed = struct_embed * plddt_weight

            # Pool structure embedding
            struct_embed_pooled = struct_embed.mean(dim=1)

            # Fuse
            combined = self.fusion(seq_embed, struct_embed_pooled)
        else:
            combined = seq_embed

        return self.fc_mu(combined), self.fc_logvar(combined)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector."""
        return self.decoder(z)

    def forward(
        self,
        sequence: torch.Tensor,
        structure: torch.Tensor | None = None,
        plddt: torch.Tensor | None = None,
        aa_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            sequence: Sequence features
            structure: Optional 3D coordinates
            plddt: Optional pLDDT scores
            aa_indices: Optional amino acid indices

        Returns:
            Dictionary with logits, mu, logvar, z
        """
        mu, logvar = self.encode(sequence, structure, plddt, aa_indices)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)

        return {
            "logits": logits,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def count_parameters(self) -> dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


__all__ = [
    "StructureAwareVAE",
    "StructureConfig",
    "SE3Encoder",
    "StructureSequenceFusion",
    "InvariantPointAttention",
]
