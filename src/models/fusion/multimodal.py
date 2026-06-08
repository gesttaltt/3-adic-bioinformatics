# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Multimodal encoder combining multiple input modalities.

Provides end-to-end encoding of sequences, structures, and
other modalities into unified representations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fusion.cross_modal import (
    CrossModalFusion,
    FusionConfig,
)


@dataclass
class MultimodalConfig:
    """Configuration for multimodal encoder.

    Attributes:
        sequence_dim: Dimension of sequence embeddings
        structure_dim: Dimension of structure embeddings
        property_dim: Dimension of property embeddings
        output_dim: Final output dimension
        hidden_dim: Hidden layer dimension
        fusion_type: Type of fusion ('attention', 'gated', 'concat')
        use_hyperbolic: Project to hyperbolic space
        curvature: Hyperbolic curvature if used
    """

    sequence_dim: int = 64
    structure_dim: int = 64
    property_dim: int = 32
    output_dim: int = 128
    hidden_dim: int = 256
    fusion_type: str = "attention"
    use_hyperbolic: bool = False
    curvature: float = -1.0


class MultimodalEncoder(nn.Module):
    """End-to-end multimodal encoder.

    Combines sequence, structure, and property encoders
    with cross-modal fusion for unified representations.

    Example:
        >>> config = MultimodalConfig()
        >>> encoder = MultimodalEncoder(config)
        >>> # With separate encoders
        >>> output = encoder(
        ...     sequence_emb=seq_encoder(sequences),
        ...     structure_emb=struct_encoder(coordinates),
        ... )
    """

    def __init__(
        self,
        config: MultimodalConfig | None = None,
        sequence_encoder: nn.Module | None = None,
        structure_encoder: nn.Module | None = None,
        property_encoder: nn.Module | None = None,
    ):
        """Initialize multimodal encoder.

        Args:
            config: Encoder configuration
            sequence_encoder: Pre-built sequence encoder
            structure_encoder: Pre-built structure encoder
            property_encoder: Pre-built property encoder
        """
        super().__init__()
        self.config = config or MultimodalConfig()

        # Store modality encoders
        self.sequence_encoder = sequence_encoder
        self.structure_encoder = structure_encoder
        self.property_encoder = property_encoder

        # Build fusion layer
        modality_dims = {}
        if True:
            modality_dims["sequence"] = self.config.sequence_dim
        if True:
            modality_dims["structure"] = self.config.structure_dim
        if property_encoder is not None:
            modality_dims["property"] = self.config.property_dim

        fusion_config = FusionConfig(
            modality_dims=modality_dims,
            output_dim=self.config.output_dim,
            hidden_dim=self.config.hidden_dim,
            fusion_type=self.config.fusion_type,
        )
        self.fusion = CrossModalFusion(fusion_config)

        # Optional hyperbolic projection
        if self.config.use_hyperbolic:
            self.hyperbolic_proj = nn.Sequential(
                nn.Linear(self.config.output_dim, self.config.output_dim * 2),
                nn.SiLU(),
                nn.Linear(self.config.output_dim * 2, self.config.output_dim),
            )

    @property
    def output_dim(self) -> int:
        return self.config.output_dim

    def forward(
        self,
        sequence_emb: torch.Tensor | None = None,
        structure_emb: torch.Tensor | None = None,
        property_emb: torch.Tensor | None = None,
        sequence_input: Any | None = None,
        structure_input: Any | None = None,
        property_input: Any | None = None,
        return_modality_embeddings: bool = False,
    ) -> torch.Tensor:
        """Encode multimodal inputs.

        Can accept either pre-computed embeddings or raw inputs
        (if encoders are provided).

        Args:
            sequence_emb: Pre-computed sequence embedding
            structure_emb: Pre-computed structure embedding
            property_emb: Pre-computed property embedding
            sequence_input: Raw sequence input for encoder
            structure_input: Raw structure input for encoder
            property_input: Raw property input for encoder
            return_modality_embeddings: Also return individual embeddings

        Returns:
            Fused multimodal embedding
        """
        modality_embeddings = {}

        # Handle sequence
        if sequence_emb is not None:
            modality_embeddings["sequence"] = sequence_emb
        elif sequence_input is not None and self.sequence_encoder is not None:
            modality_embeddings["sequence"] = self.sequence_encoder(sequence_input)

        # Handle structure
        if structure_emb is not None:
            modality_embeddings["structure"] = structure_emb
        elif structure_input is not None and self.structure_encoder is not None:
            modality_embeddings["structure"] = self.structure_encoder(structure_input)

        # Handle properties
        if property_emb is not None:
            modality_embeddings["property"] = property_emb
        elif property_input is not None and self.property_encoder is not None:
            modality_embeddings["property"] = self.property_encoder(property_input)

        # Ensure we have at least one modality
        if not modality_embeddings:
            raise ValueError("At least one modality embedding required")

        # Handle missing modalities
        modality_embeddings = self._handle_missing_modalities(modality_embeddings)

        # Fuse modalities
        fused = self.fusion(modality_embeddings)

        # Optional hyperbolic projection
        if self.config.use_hyperbolic:
            fused = self._to_hyperbolic(fused)

        if return_modality_embeddings:
            return fused, modality_embeddings

        return fused

    def _handle_missing_modalities(
        self,
        modality_embeddings: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Handle missing modalities with learned defaults.

        Args:
            modality_embeddings: Available embeddings

        Returns:
            Complete set of embeddings
        """
        # Get batch size from available embedding
        batch_size = next(iter(modality_embeddings.values())).shape[0]
        device = next(iter(modality_embeddings.values())).device

        # Create default embeddings for missing modalities
        if "sequence" not in modality_embeddings:
            modality_embeddings["sequence"] = torch.zeros(batch_size, self.config.sequence_dim, device=device)

        if "structure" not in modality_embeddings:
            modality_embeddings["structure"] = torch.zeros(batch_size, self.config.structure_dim, device=device)

        return modality_embeddings

    def _to_hyperbolic(self, x: torch.Tensor) -> torch.Tensor:
        """Project to Poincaré ball.

        Args:
            x: Euclidean embedding

        Returns:
            Hyperbolic embedding
        """
        x = self.hyperbolic_proj(x)

        # Exponential map to Poincaré ball
        c = abs(self.config.curvature)
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sqrt_c = c**0.5

        hyperbolic = torch.tanh(sqrt_c * norm) * x / (sqrt_c * norm)

        # Ensure inside ball
        max_norm = 1 - 1e-5
        current_norm = hyperbolic.norm(dim=-1, keepdim=True)
        hyperbolic = torch.where(
            current_norm > max_norm,
            hyperbolic * max_norm / current_norm,
            hyperbolic,
        )

        return hyperbolic


class ContrastiveMultimodalEncoder(nn.Module):
    """Multimodal encoder with contrastive alignment.

    Aligns modality representations in shared space using
    contrastive learning.
    """

    def __init__(
        self,
        config: MultimodalConfig | None = None,
        temperature: float = 0.07,
    ):
        """Initialize contrastive multimodal encoder.

        Args:
            config: Encoder configuration
            temperature: Contrastive loss temperature
        """
        super().__init__()
        self.config = config or MultimodalConfig()
        self.temperature = temperature

        # Individual modality projectors
        self.sequence_proj = nn.Sequential(
            nn.Linear(self.config.sequence_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.output_dim),
        )

        self.structure_proj = nn.Sequential(
            nn.Linear(self.config.structure_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.output_dim),
        )

        # Fusion for inference
        self.fusion = CrossModalFusion(
            FusionConfig(
                modality_dims={
                    "sequence": self.config.output_dim,
                    "structure": self.config.output_dim,
                },
                output_dim=self.config.output_dim,
                fusion_type="attention",
            )
        )

    def forward(
        self,
        sequence_emb: torch.Tensor,
        structure_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode and fuse modalities.

        Args:
            sequence_emb: Sequence embedding
            structure_emb: Structure embedding (optional)

        Returns:
            Fused embedding
        """
        seq_proj = F.normalize(self.sequence_proj(sequence_emb), dim=-1)

        if structure_emb is not None:
            struct_proj = F.normalize(self.structure_proj(structure_emb), dim=-1)
            fused = self.fusion(
                {
                    "sequence": seq_proj,
                    "structure": struct_proj,
                }
            )
            return fused
        else:
            return seq_proj

    def compute_alignment_loss(
        self,
        sequence_emb: torch.Tensor,
        structure_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contrastive alignment loss.

        Aligns sequence and structure representations of
        the same protein.

        Args:
            sequence_emb: Sequence embeddings
            structure_emb: Structure embeddings

        Returns:
            Alignment loss
        """
        # Project to shared space
        seq_proj = F.normalize(self.sequence_proj(sequence_emb), dim=-1)
        struct_proj = F.normalize(self.structure_proj(structure_emb), dim=-1)

        # Similarity matrix
        sim = torch.mm(seq_proj, struct_proj.t()) / self.temperature

        # Labels: diagonal (same protein)
        labels = torch.arange(sim.shape[0], device=sim.device)

        # Cross-entropy loss (both directions)
        loss_seq_to_struct = F.cross_entropy(sim, labels)
        loss_struct_to_seq = F.cross_entropy(sim.t(), labels)

        return (loss_seq_to_struct + loss_struct_to_seq) / 2


class LateFusion(nn.Module):
    """Late fusion: combine predictions from separate modality predictors.

    Each modality has its own predictor, outputs are combined.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        output_dim: int,
        hidden_dim: int = 128,
    ):
        """Initialize late fusion.

        Args:
            modality_dims: Dictionary of modality -> dimension
            output_dim: Final output dimension
            hidden_dim: Hidden dimension for predictors
        """
        super().__init__()

        # Separate predictors
        self.predictors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, output_dim),
                )
                for name, dim in modality_dims.items()
            }
        )

        # Learned combination weights
        self.weights = nn.Parameter(torch.ones(len(modality_dims)))

    def forward(
        self,
        modality_embeddings: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Combine modality predictions.

        Args:
            modality_embeddings: Dictionary of embeddings

        Returns:
            Combined prediction
        """
        # Get predictions from each modality
        predictions = []
        names = list(self.predictors.keys())

        for name in names:
            if name in modality_embeddings:
                pred = self.predictors[name](modality_embeddings[name])
                predictions.append(pred)
            else:
                # Handle missing modality
                batch_size = next(iter(modality_embeddings.values())).shape[0]
                device = next(iter(modality_embeddings.values())).device
                predictions.append(torch.zeros(batch_size, pred.shape[-1], device=device))

        # Weighted combination
        stacked = torch.stack(predictions, dim=-1)
        weights = F.softmax(self.weights, dim=0)
        combined = (stacked * weights).sum(dim=-1)

        return combined


class EarlyFusion(nn.Module):
    """Early fusion: combine inputs before any processing.

    Simple but can lose modality-specific information.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        output_dim: int,
        hidden_dims: list[int] = None,
    ):
        """Initialize early fusion.

        Args:
            modality_dims: Dictionary of modality -> dimension
            output_dim: Output dimension
            hidden_dims: Hidden layer dimensions
        """
        if hidden_dims is None:
            hidden_dims = [256, 128]
        super().__init__()

        total_dim = sum(modality_dims.values())
        self.modality_names = list(modality_dims.keys())

        # Shared network
        layers = []
        prev_dim = total_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.SiLU(),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        modality_embeddings: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Process concatenated inputs.

        Args:
            modality_embeddings: Dictionary of embeddings

        Returns:
            Processed output
        """
        # Concatenate in consistent order
        concatenated = torch.cat(
            [modality_embeddings[name] for name in self.modality_names],
            dim=-1,
        )

        return self.network(concatenated)
