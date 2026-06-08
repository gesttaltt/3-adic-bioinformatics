# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Cross-modal fusion mechanisms.

Provides various strategies for combining representations
from different modalities (sequence, structure, properties).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FusionConfig:
    """Configuration for fusion layers.

    Attributes:
        modality_dims: Dictionary of modality -> dimension
        output_dim: Output embedding dimension
        hidden_dim: Hidden layer dimension
        n_heads: Number of attention heads
        dropout: Dropout rate
        fusion_type: Type of fusion ('attention', 'gated', 'concat')
    """

    modality_dims: dict[str, int] = field(
        default_factory=lambda: {
            "sequence": 64,
            "structure": 64,
        }
    )
    output_dim: int = 128
    hidden_dim: int = 256
    n_heads: int = 4
    dropout: float = 0.1
    fusion_type: str = "attention"


class ConcatFusion(nn.Module):
    """Simple concatenation-based fusion.

    Concatenates modality embeddings and projects to output dimension.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        output_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        """Initialize concatenation fusion.

        Args:
            modality_dims: Dictionary of modality name -> dimension
            output_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        total_dim = sum(modality_dims.values())

        # Projection network
        self.projection = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        modality_embeddings: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Fuse modality embeddings.

        Args:
            modality_embeddings: Dictionary of modality name -> embedding

        Returns:
            Fused embedding
        """
        # Concatenate in consistent order
        embeddings = [modality_embeddings[name] for name in self.modality_names]
        concatenated = torch.cat(embeddings, dim=-1)

        return self.projection(concatenated)


class GatedFusion(nn.Module):
    """Gated fusion mechanism.

    Learns modality-specific gates to control contribution.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        """Initialize gated fusion.

        Args:
            modality_dims: Dictionary of modality -> dimension
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.n_modalities = len(modality_dims)

        # Project each modality to common dimension
        self.projections = nn.ModuleDict({name: nn.Linear(dim, output_dim) for name, dim in modality_dims.items()})

        # Gate network
        total_dim = sum(modality_dims.values())
        self.gate_network = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, self.n_modalities),
            nn.Softmax(dim=-1),
        )

        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        modality_embeddings: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Fuse with learned gates.

        Args:
            modality_embeddings: Dictionary of embeddings

        Returns:
            Gated fusion result
        """
        # Project each modality
        projected = {name: self.projections[name](emb) for name, emb in modality_embeddings.items()}

        # Compute gates
        concatenated = torch.cat(
            [modality_embeddings[name] for name in self.modality_names],
            dim=-1,
        )
        gates = self.gate_network(concatenated)  # (batch, n_modalities)

        # Weighted sum
        stacked = torch.stack(
            [projected[name] for name in self.modality_names],
            dim=-1,
        )  # (batch, output_dim, n_modalities)

        fused = (stacked * gates.unsqueeze(1)).sum(dim=-1)
        fused = self.dropout(fused)
        fused = self.output_norm(fused)

        return fused

    def get_gate_weights(
        self,
        modality_embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Get gate weights for interpretability.

        Args:
            modality_embeddings: Dictionary of embeddings

        Returns:
            Dictionary of modality -> gate weight
        """
        concatenated = torch.cat(
            [modality_embeddings[name] for name in self.modality_names],
            dim=-1,
        )
        gates = self.gate_network(concatenated)

        return {name: gates[:, i] for i, name in enumerate(self.modality_names)}


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism.

    Each modality attends to all other modalities.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        output_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize cross-modal attention.

        Args:
            modality_dims: Dictionary of modality -> dimension
            output_dim: Output dimension (common space)
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.n_modalities = len(modality_dims)
        self.output_dim = output_dim
        self.n_heads = n_heads

        assert output_dim % n_heads == 0
        self.head_dim = output_dim // n_heads

        # Project to common dimension
        self.input_projections = nn.ModuleDict(
            {name: nn.Linear(dim, output_dim) for name, dim in modality_dims.items()}
        )

        # Multi-head attention components
        self.q_proj = nn.Linear(output_dim, output_dim)
        self.k_proj = nn.Linear(output_dim, output_dim)
        self.v_proj = nn.Linear(output_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        modality_embeddings: dict[str, torch.Tensor],
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Apply cross-modal attention.

        Args:
            modality_embeddings: Dictionary of embeddings
            return_attention: Return attention weights

        Returns:
            Fused embedding (and optionally attention weights)
        """
        batch_size = next(iter(modality_embeddings.values())).shape[0]

        # Project to common dimension
        projected = {name: self.input_projections[name](emb) for name, emb in modality_embeddings.items()}

        # Stack as sequence: (batch, n_modalities, output_dim)
        stacked = torch.stack(
            [projected[name] for name in self.modality_names],
            dim=1,
        )

        # Self-attention over modalities
        Q = self.q_proj(stacked)
        K = self.k_proj(stacked)
        V = self.v_proj(stacked)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.n_modalities, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, self.n_modalities, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, self.n_modalities, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        context = torch.matmul(attn, V)

        # Reshape back
        context = context.transpose(1, 2).contiguous().view(batch_size, self.n_modalities, self.output_dim)

        # Output projection
        output = self.out_proj(context)

        # Residual connection
        output = self.layer_norm(stacked + output)

        # Pool over modalities
        fused = output.mean(dim=1)

        if return_attention:
            return fused, attn

        return fused


class CrossModalFusion(nn.Module):
    """Comprehensive cross-modal fusion layer.

    Supports multiple fusion strategies with learned modality weights.

    Example:
        >>> config = FusionConfig(modality_dims={"seq": 64, "struct": 64})
        >>> fusion = CrossModalFusion(config)
        >>> embeddings = {"seq": seq_emb, "struct": struct_emb}
        >>> fused = fusion(embeddings)
    """

    def __init__(
        self,
        config: FusionConfig | None = None,
    ):
        """Initialize cross-modal fusion.

        Args:
            config: Fusion configuration
        """
        super().__init__()
        self.config = config or FusionConfig()

        # Choose fusion mechanism
        if self.config.fusion_type == "attention":
            self.fusion = CrossModalAttention(
                modality_dims=self.config.modality_dims,
                output_dim=self.config.output_dim,
                n_heads=self.config.n_heads,
                dropout=self.config.dropout,
            )
        elif self.config.fusion_type == "gated":
            self.fusion = GatedFusion(
                modality_dims=self.config.modality_dims,
                output_dim=self.config.output_dim,
                dropout=self.config.dropout,
            )
        else:  # concat
            self.fusion = ConcatFusion(
                modality_dims=self.config.modality_dims,
                output_dim=self.config.output_dim,
                hidden_dim=self.config.hidden_dim,
                dropout=self.config.dropout,
            )

        # Optional: modality-specific preprocessing
        self.modality_norms = nn.ModuleDict(
            {name: nn.LayerNorm(dim) for name, dim in self.config.modality_dims.items()}
        )

    def forward(
        self,
        modality_embeddings: dict[str, torch.Tensor],
        return_attention: bool = False,
    ) -> torch.Tensor:
        """Fuse modality embeddings.

        Args:
            modality_embeddings: Dictionary of modality -> embedding
            return_attention: Return attention weights (if applicable)

        Returns:
            Fused embedding
        """
        # Normalize each modality
        normalized = {name: self.modality_norms[name](emb) for name, emb in modality_embeddings.items()}

        # Apply fusion
        if hasattr(self.fusion, "forward") and return_attention:
            if hasattr(self.fusion, "get_gate_weights"):
                fused = self.fusion(normalized)
                weights = self.fusion.get_gate_weights(normalized)
                return fused, weights
            elif isinstance(self.fusion, CrossModalAttention):
                return self.fusion(normalized, return_attention=True)

        return self.fusion(normalized)


class HierarchicalFusion(nn.Module):
    """Hierarchical fusion for many modalities.

    Fuses modalities in a hierarchical manner for efficiency.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        output_dim: int = 128,
        hierarchy: list[list[str]] | None = None,
    ):
        """Initialize hierarchical fusion.

        Args:
            modality_dims: Dictionary of modality -> dimension
            output_dim: Final output dimension
            hierarchy: List of modality groups to fuse at each level
        """
        super().__init__()
        self.modality_dims = modality_dims

        if hierarchy is None:
            # Default: pair-wise fusion
            names = list(modality_dims.keys())
            hierarchy = [[names[i], names[i + 1] if i + 1 < len(names) else names[i]] for i in range(0, len(names), 2)]

        self.hierarchy = hierarchy

        # Build fusion layers for each level
        self.fusion_layers = nn.ModuleList()
        current_dims = modality_dims.copy()

        for level, group in enumerate(hierarchy):
            group_dims = {name: current_dims[name] for name in group if name in current_dims}

            if len(group_dims) > 1:
                fusion = GatedFusion(
                    modality_dims=group_dims,
                    output_dim=output_dim,
                )
                self.fusion_layers.append(fusion)

                # Update dims for next level
                fused_name = f"level_{level}_fused"
                current_dims[fused_name] = output_dim
                for name in group:
                    if name in current_dims:
                        del current_dims[name]

        # Final projection
        remaining_dim = sum(current_dims.values())
        self.final_proj = nn.Linear(remaining_dim, output_dim)

    def forward(
        self,
        modality_embeddings: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Apply hierarchical fusion.

        Args:
            modality_embeddings: Dictionary of embeddings

        Returns:
            Fused embedding
        """
        current = modality_embeddings.copy()

        for level, (group, fusion) in enumerate(zip(self.hierarchy, self.fusion_layers, strict=False)):
            group_emb = {name: current[name] for name in group if name in current}

            if len(group_emb) > 1:
                fused = fusion(group_emb)
                fused_name = f"level_{level}_fused"
                current[fused_name] = fused

                for name in group:
                    if name in current:
                        del current[name]

        # Final concatenation and projection
        remaining = torch.cat(list(current.values()), dim=-1)
        return self.final_proj(remaining)
