# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Hybrid Encoder combining p-adic Codon Encoder with Protein Language Models.

This module implements a hybrid encoding strategy that combines:
1. P-adic CodonEncoder - Our custom hierarchical structure-aware embeddings
2. ESM-2 / ProtTrans - Pre-trained protein language model embeddings

The hybrid approach captures both:
- Codon-level p-adic hierarchical structure (unique to our approach)
- Protein-level evolutionary and functional features (from PLMs)

References:
- Lin et al. (2023): ESM-2 - Evolutionary Scale Modeling
- Elnaggar et al. (2021): ProtTrans - Transfer learning in proteins
- RiboNN (2024): Full-length mRNA integration
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.encoders.codon_encoder import CodonEncoder


class PLMBackend(Enum):
    """Available protein language model backends."""

    NONE = "none"
    ESM2_8M = "esm2_t6_8M_UR50D"
    ESM2_35M = "esm2_t12_35M_UR50D"
    ESM2_150M = "esm2_t30_150M_UR50D"
    ESM2_650M = "esm2_t33_650M_UR50D"
    PROT_BERT = "prot_bert"
    PROT_T5_XL = "prot_t5_xl_uniref50"


# ESM-2 model dimensions by size
ESM2_DIMS = {
    PLMBackend.ESM2_8M: 320,
    PLMBackend.ESM2_35M: 480,
    PLMBackend.ESM2_150M: 640,
    PLMBackend.ESM2_650M: 1280,
    PLMBackend.PROT_BERT: 1024,
    PLMBackend.PROT_T5_XL: 1024,
}


@dataclass
class HybridEncoderConfig:
    """Configuration for HybridCodonEncoder."""

    codon_dim: int = 64
    output_dim: int = 64
    plm_backend: PLMBackend = PLMBackend.NONE
    freeze_plm: bool = True
    fusion_type: str = "attention"  # 'concat', 'add', 'attention', 'gated'
    use_padic_init: bool = True
    dropout: float = 0.1


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between codon and PLM embeddings."""

    def __init__(
        self,
        codon_dim: int,
        plm_dim: int,
        output_dim: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.codon_dim = codon_dim
        self.plm_dim = plm_dim
        self.output_dim = output_dim

        # Project both to same dimension
        self.codon_proj = nn.Linear(codon_dim, output_dim)
        self.plm_proj = nn.Linear(plm_dim, output_dim)

        # Cross-attention: codon queries, PLM keys/values
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        codon_emb: torch.Tensor,
        plm_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse codon and PLM embeddings via cross-attention.

        Args:
            codon_emb: Codon embeddings (batch, seq_len, codon_dim)
            plm_emb: PLM embeddings (batch, seq_len, plm_dim)

        Returns:
            Fused embeddings (batch, seq_len, output_dim)
        """
        # Project to common dimension
        q = self.codon_proj(codon_emb)
        kv = self.plm_proj(plm_emb)

        # Cross-attention
        fused, _ = self.cross_attn(q, kv, kv)

        # Residual connection with codon embeddings
        output = self.output_proj(fused + q)

        return output


class GatedFusion(nn.Module):
    """Gated fusion between codon and PLM embeddings."""

    def __init__(
        self,
        codon_dim: int,
        plm_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project both to output dimension
        self.codon_proj = nn.Linear(codon_dim, output_dim)
        self.plm_proj = nn.Linear(plm_dim, output_dim)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(codon_dim + plm_dim, output_dim),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        codon_emb: torch.Tensor,
        plm_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse via learnable gating.

        Args:
            codon_emb: Codon embeddings (batch, seq_len, codon_dim)
            plm_emb: PLM embeddings (batch, seq_len, plm_dim)

        Returns:
            Gated fusion (batch, seq_len, output_dim)
        """
        # Project
        codon_proj = self.codon_proj(codon_emb)
        plm_proj = self.plm_proj(plm_emb)

        # Compute gate
        combined = torch.cat([codon_emb, plm_emb], dim=-1)
        gate = self.gate(combined)

        # Apply gate
        output = gate * codon_proj + (1 - gate) * plm_proj
        output = self.layer_norm(self.dropout(output))

        return output


class HybridCodonEncoder(nn.Module):
    """Hybrid encoder combining p-adic Codon Encoder with Protein Language Models.

    This encoder creates a unified representation by fusing:
    1. Codon-level embeddings with p-adic structure awareness
    2. Pre-trained protein language model embeddings (optional)

    The fusion strategy can be:
    - 'concat': Simple concatenation + linear projection
    - 'add': Element-wise addition after projection
    - 'attention': Cross-attention fusion
    - 'gated': Learnable gating mechanism

    For memory-efficient operation on RTX 2060 SUPER (8GB VRAM):
    - Use ESM2_8M or ESM2_35M backends
    - Set freeze_plm=True to avoid gradient memory
    - Use mixed precision training
    """

    def __init__(self, config: HybridEncoderConfig | None = None, **kwargs):
        """Initialize HybridCodonEncoder.

        Args:
            config: Configuration object
            **kwargs: Override config values
        """
        super().__init__()

        # Handle config
        if config is None:
            config = HybridEncoderConfig()

        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config
        self.codon_dim = config.codon_dim
        self.output_dim = config.output_dim
        self.plm_backend = config.plm_backend
        self.freeze_plm = config.freeze_plm
        self.fusion_type = config.fusion_type

        # Initialize codon encoder
        self.codon_encoder = CodonEncoder(
            embedding_dim=config.codon_dim,
            use_padic_init=config.use_padic_init,
        )

        # Initialize PLM if specified
        self.plm = None
        self.plm_dim = 0

        if config.plm_backend != PLMBackend.NONE:
            self._init_plm(config.plm_backend, config.freeze_plm)
            self.plm_dim = ESM2_DIMS.get(config.plm_backend, 0)

        # Initialize fusion layer
        if self.plm is not None:
            self._init_fusion(config)
        else:
            # No PLM, just project codon embeddings
            self.output_proj = nn.Linear(config.codon_dim, config.output_dim)

    def _init_plm(self, backend: PLMBackend, freeze: bool):
        """Initialize protein language model.

        Args:
            backend: Which PLM to use
            freeze: Whether to freeze PLM weights
        """
        try:
            if backend in [
                PLMBackend.ESM2_8M,
                PLMBackend.ESM2_35M,
                PLMBackend.ESM2_150M,
                PLMBackend.ESM2_650M,
            ]:
                import esm

                model, alphabet = esm.pretrained.load_model_and_alphabet(backend.value)
                self.plm = model
                self.plm_alphabet = alphabet
                self.plm_batch_converter = alphabet.get_batch_converter()

            elif backend == PLMBackend.PROT_BERT:
                from transformers import BertModel, BertTokenizer

                self.plm = BertModel.from_pretrained("Rostlab/prot_bert")
                self.plm_tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert")

            elif backend == PLMBackend.PROT_T5_XL:
                from transformers import T5EncoderModel, T5Tokenizer

                self.plm = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
                self.plm_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50")

            if freeze and self.plm is not None:
                for param in self.plm.parameters():
                    param.requires_grad = False
                self.plm.eval()

        except ImportError as e:
            print(f"Warning: Could not load PLM backend {backend.value}: {e}")
            print("Falling back to codon-only mode.")
            self.plm = None

    def _init_fusion(self, config: HybridEncoderConfig):
        """Initialize fusion layer based on config.

        Args:
            config: Encoder configuration
        """
        if config.fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(config.codon_dim + self.plm_dim, config.output_dim),
                nn.LayerNorm(config.output_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
        elif config.fusion_type == "add":
            self.codon_proj = nn.Linear(config.codon_dim, config.output_dim)
            self.plm_proj = nn.Linear(self.plm_dim, config.output_dim)
            self.fusion = nn.Sequential(
                nn.LayerNorm(config.output_dim),
                nn.Dropout(config.dropout),
            )
        elif config.fusion_type == "attention":
            self.fusion = CrossAttentionFusion(
                codon_dim=config.codon_dim,
                plm_dim=self.plm_dim,
                output_dim=config.output_dim,
                dropout=config.dropout,
            )
        elif config.fusion_type == "gated":
            self.fusion = GatedFusion(
                codon_dim=config.codon_dim,
                plm_dim=self.plm_dim,
                output_dim=config.output_dim,
                dropout=config.dropout,
            )
        else:
            raise ValueError(f"Unknown fusion type: {config.fusion_type}")

    def get_plm_embeddings(
        self,
        amino_acid_sequences: list[str],
    ) -> torch.Tensor:
        """Get embeddings from protein language model.

        Args:
            amino_acid_sequences: List of amino acid sequences

        Returns:
            PLM embeddings (batch, max_len, plm_dim)
        """
        if self.plm is None:
            raise ValueError("No PLM initialized")

        device = next(self.plm.parameters()).device

        if self.plm_backend in [
            PLMBackend.ESM2_8M,
            PLMBackend.ESM2_35M,
            PLMBackend.ESM2_150M,
            PLMBackend.ESM2_650M,
        ]:
            # ESM format
            data = [(f"seq_{i}", seq) for i, seq in enumerate(amino_acid_sequences)]
            _, _, batch_tokens = self.plm_batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad() if self.freeze_plm else torch.enable_grad():
                results = self.plm(batch_tokens, repr_layers=[self.plm.num_layers])
                embeddings = results["representations"][self.plm.num_layers]

            # Remove BOS/EOS tokens
            embeddings = embeddings[:, 1:-1, :]

        else:
            # HuggingFace format
            inputs = self.plm_tokenizer(
                amino_acid_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            with torch.no_grad() if self.freeze_plm else torch.enable_grad():
                outputs = self.plm(**inputs)
                embeddings = outputs.last_hidden_state

        return embeddings

    def forward(
        self,
        codon_indices: torch.Tensor,
        amino_acid_sequences: list[str] | None = None,
        plm_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass producing hybrid embeddings.

        Args:
            codon_indices: Codon indices (batch, seq_len), values 0-63
            amino_acid_sequences: Optional AA sequences for PLM
            plm_embeddings: Optional pre-computed PLM embeddings

        Returns:
            Hybrid embeddings (batch, seq_len, output_dim)
        """
        # Get codon embeddings
        codon_emb = self.codon_encoder(codon_indices)

        # If no PLM, just project codon embeddings
        if self.plm is None:
            return self.output_proj(codon_emb)

        # Get PLM embeddings
        if plm_embeddings is None:
            if amino_acid_sequences is None:
                raise ValueError("Must provide amino_acid_sequences or plm_embeddings when PLM is enabled")
            plm_emb = self.get_plm_embeddings(amino_acid_sequences)
        else:
            plm_emb = plm_embeddings

        # Ensure same sequence length (PLM may differ due to tokenization)
        codon_len = codon_emb.shape[1]
        plm_len = plm_emb.shape[1]

        if codon_len != plm_len:
            # Interpolate PLM embeddings to match codon length
            # Codon sequences are 3x longer than AA sequences
            if plm_len * 3 == codon_len:
                # Repeat each AA embedding for 3 codons
                plm_emb = plm_emb.unsqueeze(2).expand(-1, -1, 3, -1)
                plm_emb = plm_emb.reshape(plm_emb.shape[0], -1, plm_emb.shape[-1])
            else:
                # General interpolation
                plm_emb = F.interpolate(
                    plm_emb.transpose(1, 2),
                    size=codon_len,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

        # Apply fusion
        if self.fusion_type == "concat":
            combined = torch.cat([codon_emb, plm_emb], dim=-1)
            output = self.fusion(combined)
        elif self.fusion_type == "add":
            codon_proj = self.codon_proj(codon_emb)
            plm_proj = self.plm_proj(plm_emb)
            output = self.fusion(codon_proj + plm_proj)
        else:
            # attention or gated
            output = self.fusion(codon_emb, plm_emb)

        return output

    def get_codon_only_embeddings(self, codon_indices: torch.Tensor) -> torch.Tensor:
        """Get codon-only embeddings without PLM.

        Args:
            codon_indices: Codon indices (batch, seq_len)

        Returns:
            Codon embeddings (batch, seq_len, codon_dim)
        """
        return self.codon_encoder(codon_indices)

    def compute_padic_loss(self) -> torch.Tensor:
        """Compute p-adic structure preservation loss.

        Returns:
            Scalar loss tensor
        """
        return self.codon_encoder.compute_padic_loss()


class HybridEncoderFactory:
    """Factory for creating hybrid encoders with different configurations."""

    @staticmethod
    def create_codon_only(embedding_dim: int = 64) -> HybridCodonEncoder:
        """Create codon-only encoder (no PLM)."""
        config = HybridEncoderConfig(
            codon_dim=embedding_dim,
            output_dim=embedding_dim,
            plm_backend=PLMBackend.NONE,
        )
        return HybridCodonEncoder(config)

    @staticmethod
    def create_esm2_small(
        codon_dim: int = 64,
        output_dim: int = 128,
        freeze: bool = True,
    ) -> HybridCodonEncoder:
        """Create encoder with ESM-2 8M (fits on 8GB VRAM)."""
        config = HybridEncoderConfig(
            codon_dim=codon_dim,
            output_dim=output_dim,
            plm_backend=PLMBackend.ESM2_8M,
            freeze_plm=freeze,
            fusion_type="attention",
        )
        return HybridCodonEncoder(config)

    @staticmethod
    def create_esm2_medium(
        codon_dim: int = 64,
        output_dim: int = 128,
        freeze: bool = True,
    ) -> HybridCodonEncoder:
        """Create encoder with ESM-2 35M (fits on 8GB VRAM with gradient checkpointing)."""
        config = HybridEncoderConfig(
            codon_dim=codon_dim,
            output_dim=output_dim,
            plm_backend=PLMBackend.ESM2_35M,
            freeze_plm=freeze,
            fusion_type="gated",
        )
        return HybridCodonEncoder(config)


__all__ = [
    "HybridCodonEncoder",
    "HybridEncoderConfig",
    "HybridEncoderFactory",
    "PLMBackend",
    "CrossAttentionFusion",
    "GatedFusion",
]
