# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Multi-Scale Nucleotide Encoder for Sub-Codon Granularity.

This module implements encoders that operate at multiple levels of granularity
below the codon level, capturing:

1. **Nucleotide-level**: Individual base features (A, C, G, T/U)
2. **Dinucleotide-level**: Adjacent pair features (16 combinations)
3. **Position-within-codon**: 1st, 2nd, 3rd (wobble) position features
4. **Codon junction**: 6-mer spanning codon boundaries
5. **Local structure**: mRNA secondary structure features
6. **Ribosome site**: A-site, P-site, E-site context windows

Based on research from:
- RiboNN (2024): Dinucleotide position-dependent effects
- Riboformer (2024): Context-dependent translation dynamics
- CodonTransformer (2025): Multi-species codon context
- BPfold: Local mRNA secondary structure
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class NucleotideBase(Enum):
    """DNA/RNA nucleotide bases."""

    A = 0  # Adenine
    C = 1  # Cytosine
    G = 2  # Guanine
    T = 3  # Thymine (DNA)
    U = 3  # Uracil (RNA) - same encoding as T


# Dinucleotide encoding (16 combinations)
DINUCLEOTIDES = [
    "AA",
    "AC",
    "AG",
    "AT",
    "CA",
    "CC",
    "CG",
    "CT",
    "GA",
    "GC",
    "GG",
    "GT",
    "TA",
    "TC",
    "TG",
    "TT",
]

DINUCLEOTIDE_TO_IDX = {dn: i for i, dn in enumerate(DINUCLEOTIDES)}

# Codon position names
CODON_POSITIONS = ["first", "second", "wobble"]


@dataclass
class MultiScaleConfig:
    """Configuration for MultiScaleNucleotideEncoder."""

    # Feature dimensions
    nucleotide_dim: int = 16
    dinucleotide_dim: int = 32
    position_dim: int = 16
    junction_dim: int = 32
    structure_dim: int = 32
    output_dim: int = 64

    # Context windows
    ribosome_window: int = 10  # ±5 codons around A-site
    structure_window: int = 30  # nucleotides for local structure

    # Feature flags
    use_dinucleotide: bool = True
    use_position: bool = True
    use_junction: bool = True
    use_structure: bool = False  # Requires secondary structure predictor
    use_gc_content: bool = True
    use_cpg_density: bool = True

    # Aggregation
    aggregation: str = "concat"  # 'concat', 'attention', 'hierarchical'
    dropout: float = 0.1


class NucleotideEmbedding(nn.Module):
    """Embedding layer for individual nucleotides.

    Includes positional encoding for position within codon (1st, 2nd, wobble).
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        use_position_encoding: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_position = use_position_encoding

        # Base embedding (4 nucleotides + padding)
        self.base_embedding = nn.Embedding(5, embedding_dim, padding_idx=4)

        # Position within codon (1st, 2nd, wobble)
        if use_position_encoding:
            self.position_embedding = nn.Embedding(3, embedding_dim)

            # Learned combination
            self.combine = nn.Linear(embedding_dim * 2, embedding_dim)

    def forward(
        self,
        nucleotides: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed nucleotides with optional position encoding.

        Args:
            nucleotides: Nucleotide indices (batch, seq_len), 0-3 for ACGT, 4 for pad
            positions: Position within codon (batch, seq_len), 0-2

        Returns:
            Embeddings (batch, seq_len, embedding_dim)
        """
        base_emb = self.base_embedding(nucleotides)

        if self.use_position and positions is not None:
            pos_emb = self.position_embedding(positions)
            combined = torch.cat([base_emb, pos_emb], dim=-1)
            return self.combine(combined)

        return base_emb


class DinucleotideEncoder(nn.Module):
    """Encode dinucleotide features with position-dependent effects.

    Based on RiboNN findings that dinucleotides like AUG, GG, UU, AA, UA
    have strong position-dependent effects on translation efficiency.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        use_position_weights: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_position_weights = use_position_weights

        # 16 dinucleotides + padding
        self.dinucleotide_embedding = nn.Embedding(17, embedding_dim, padding_idx=16)

        # Position-dependent weights (learned)
        if use_position_weights:
            # Different weights for different regions: 5'UTR, CDS, 3'UTR
            self.region_weights = nn.Parameter(torch.ones(3, embedding_dim))

            # Distance from start/stop codon effect
            self.distance_proj = nn.Linear(1, embedding_dim)

    def encode_dinucleotides(
        self,
        nucleotides: torch.Tensor,
    ) -> torch.Tensor:
        """Convert nucleotide sequence to dinucleotide indices.

        Args:
            nucleotides: (batch, seq_len) nucleotide indices 0-3

        Returns:
            (batch, seq_len-1) dinucleotide indices 0-15
        """
        # Combine adjacent nucleotides: first * 4 + second
        first = nucleotides[:, :-1]
        second = nucleotides[:, 1:]
        dinuc_idx = first * 4 + second
        return dinuc_idx

    def forward(
        self,
        nucleotides: torch.Tensor,
        region_mask: torch.Tensor | None = None,
        start_codon_pos: int | None = None,
    ) -> torch.Tensor:
        """Encode sequence as dinucleotide features.

        Args:
            nucleotides: (batch, seq_len) nucleotide indices
            region_mask: (batch, seq_len) region labels (0=5'UTR, 1=CDS, 2=3'UTR)
            start_codon_pos: Position of start codon for distance weighting

        Returns:
            (batch, seq_len-1, embedding_dim) dinucleotide embeddings
        """
        dinuc_idx = self.encode_dinucleotides(nucleotides)
        embeddings = self.dinucleotide_embedding(dinuc_idx)

        if self.use_position_weights and region_mask is not None:
            # Apply region-specific weights
            region_mask = region_mask[:, :-1]  # Match dinucleotide length
            weights = self.region_weights[region_mask]
            embeddings = embeddings * weights

        return embeddings


class WobblePositionEncoder(nn.Module):
    """Special encoder for wobble (3rd) position features.

    The wobble position is special because:
    - 90% of synonymous mutations occur at position 3
    - Wobble pairing affects ribosome speed
    - tRNA availability varies by wobble base

    Based on research showing wobble-mediated decoding is slower
    than Watson-Crick decoding regardless of tRNA levels.
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        include_tRNA_weights: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.include_tRNA = include_tRNA_weights

        # Wobble base embedding
        self.wobble_embedding = nn.Embedding(4, embedding_dim)

        # Wobble pairing type (Watson-Crick vs non-standard)
        # WC: A-U, G-C; Wobble: G-U, I-U, I-A, I-C
        self.pairing_embedding = nn.Embedding(3, embedding_dim // 2)  # WC, wobble, other

        # tRNA abundance weights (organism-specific, default human)
        if include_tRNA_weights:
            # Relative tRNA abundance for each codon (simplified)
            self.register_buffer(
                "tRNA_weights",
                torch.ones(64),  # Will be loaded from data
            )

        # Combine features
        self.combine = nn.Linear(embedding_dim + embedding_dim // 2, embedding_dim)

    def forward(
        self,
        wobble_bases: torch.Tensor,
        first_two_bases: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode wobble position features.

        Args:
            wobble_bases: (batch, n_codons) wobble base indices 0-3
            first_two_bases: (batch, n_codons) first two bases as index (0-15)

        Returns:
            (batch, n_codons, embedding_dim) wobble features
        """
        wobble_emb = self.wobble_embedding(wobble_bases)

        # Determine pairing type (simplified)
        # 0 = strong WC (C, G at wobble), 1 = weak WC (A, T/U), 2 = other
        pairing_type = torch.where(
            (wobble_bases == 1) | (wobble_bases == 2),
            torch.zeros_like(wobble_bases),
            torch.ones_like(wobble_bases),
        )
        pairing_emb = self.pairing_embedding(pairing_type)

        combined = torch.cat([wobble_emb, pairing_emb], dim=-1)
        return self.combine(combined)


class CodonJunctionEncoder(nn.Module):
    """Encode codon junction (boundary) features.

    Captures 6-mer context spanning codon boundaries, which affects:
    - Ribosome processivity
    - mRNA secondary structure at junction
    - Translation elongation rate
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        context_size: int = 6,  # 3 nucleotides on each side
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_size = context_size

        # 6-mer encoder (4^6 = 4096 possible, too many - use learned projection)
        self.nucleotide_embedding = nn.Embedding(4, 8)

        # Convolutional encoder for context
        self.conv = nn.Conv1d(8, embedding_dim, kernel_size=context_size, padding=0)

        # Final projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        nucleotides: torch.Tensor,
    ) -> torch.Tensor:
        """Encode codon junction features.

        Args:
            nucleotides: (batch, seq_len) nucleotide indices

        Returns:
            (batch, n_junctions, embedding_dim) junction features
        """
        # Embed nucleotides
        emb = self.nucleotide_embedding(nucleotides)  # (batch, seq_len, 8)
        emb = emb.transpose(1, 2)  # (batch, 8, seq_len)

        # Apply convolution
        conv_out = self.conv(emb)  # (batch, embedding_dim, seq_len - context + 1)
        conv_out = conv_out.transpose(1, 2)  # (batch, n_junctions, embedding_dim)

        # Get junction positions (every 3rd position after context)
        # For simplicity, return all positions; filter to junctions externally
        return self.output_proj(F.gelu(conv_out))


class LocalStructureEncoder(nn.Module):
    """Encode local mRNA secondary structure features.

    Uses simplified structure prediction or pre-computed features.
    Based on BPfold and RhoFold+ approaches.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        window_size: int = 30,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.window_size = window_size

        # Structure feature encoder
        # Input: base pairing probabilities, MFE contribution, etc.
        self.structure_encoder = nn.Sequential(
            nn.Linear(window_size, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

        # Hairpin/stem indicator
        self.hairpin_embedding = nn.Embedding(3, embedding_dim // 4)  # none, stem, loop

    def forward(
        self,
        pairing_probs: torch.Tensor,
        structure_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode local structure features.

        Args:
            pairing_probs: (batch, seq_len, window_size) local pairing probabilities
            structure_labels: (batch, seq_len) structure labels (0=none, 1=stem, 2=loop)

        Returns:
            (batch, seq_len, embedding_dim) structure features
        """
        struct_emb = self.structure_encoder(pairing_probs)

        if structure_labels is not None:
            hairpin_emb = self.hairpin_embedding(structure_labels)
            struct_emb = struct_emb + F.pad(hairpin_emb, (0, self.embedding_dim - self.embedding_dim // 4))

        return struct_emb


class RibosomeSiteEncoder(nn.Module):
    """Encode ribosome site context (A-site, P-site, E-site).

    Based on Riboformer's codon-level resolution approach.
    The A-site is the primary decoding site, P-site holds peptidyl-tRNA,
    E-site is the exit site.
    """

    def __init__(
        self,
        codon_embedding_dim: int = 64,
        output_dim: int = 64,
        context_window: int = 10,  # ±5 codons
    ):
        super().__init__()
        self.context_window = context_window
        self.output_dim = output_dim

        # Site-specific embeddings
        self.site_embedding = nn.Embedding(3, output_dim // 4)  # A, P, E

        # Context encoder with attention
        self.context_attn = nn.MultiheadAttention(
            embed_dim=codon_embedding_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Linear(codon_embedding_dim + output_dim // 4, output_dim)

    def forward(
        self,
        codon_embeddings: torch.Tensor,
        site_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Encode ribosome site context.

        Args:
            codon_embeddings: (batch, n_codons, codon_dim) codon embeddings
            site_positions: (batch,) position of A-site codon

        Returns:
            (batch, 3, output_dim) embeddings for A, P, E sites
        """
        batch_size = codon_embeddings.shape[0]
        n_codons = codon_embeddings.shape[1]
        device = codon_embeddings.device

        outputs = []

        for site_offset, site_idx in [(-1, 2), (0, 0), (1, 1)]:  # E, A, P
            # Get context window around site
            center = site_positions + site_offset
            start = (center - self.context_window // 2).clamp(0, n_codons - self.context_window)
            start + self.context_window

            # Extract context (simplified: just use position embedding)
            site_emb = self.site_embedding(torch.tensor([site_idx], device=device).expand(batch_size))

            # For simplicity, use center codon embedding
            center_clamped = center.clamp(0, n_codons - 1)
            center_emb = codon_embeddings[torch.arange(batch_size, device=device), center_clamped]

            combined = torch.cat([center_emb, site_emb], dim=-1)
            outputs.append(self.output_proj(combined))

        return torch.stack(outputs, dim=1)


class MultiScaleNucleotideEncoder(nn.Module):
    """Multi-scale encoder capturing sub-codon granularity.

    Combines multiple levels of encoding:
    1. Nucleotide-level with position encoding
    2. Dinucleotide features (RiboNN-style)
    3. Wobble position special features
    4. Codon junction context
    5. Local secondary structure (optional)
    6. Ribosome site context

    This goes beyond codon-level to capture finer-grained biological signals.
    """

    def __init__(self, config: MultiScaleConfig | None = None, **kwargs):
        """Initialize multi-scale encoder.

        Args:
            config: Configuration object
            **kwargs: Override config values
        """
        super().__init__()

        if config is None:
            config = MultiScaleConfig()

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        self.config = config

        # Initialize sub-encoders
        self.nucleotide_encoder = NucleotideEmbedding(
            embedding_dim=config.nucleotide_dim,
            use_position_encoding=config.use_position,
        )

        if config.use_dinucleotide:
            self.dinucleotide_encoder = DinucleotideEncoder(
                embedding_dim=config.dinucleotide_dim,
            )

        if config.use_position:
            self.wobble_encoder = WobblePositionEncoder(
                embedding_dim=config.position_dim,
            )

        if config.use_junction:
            self.junction_encoder = CodonJunctionEncoder(
                embedding_dim=config.junction_dim,
            )

        if config.use_structure:
            self.structure_encoder = LocalStructureEncoder(
                embedding_dim=config.structure_dim,
                window_size=config.structure_window,
            )

        # GC content encoder
        if config.use_gc_content:
            self.gc_proj = nn.Linear(1, 8)

        # CpG density encoder
        if config.use_cpg_density:
            self.cpg_proj = nn.Linear(1, 8)

        # Compute total input dimension for aggregation
        total_dim = config.nucleotide_dim
        if config.use_dinucleotide:
            total_dim += config.dinucleotide_dim
        if config.use_position:
            total_dim += config.position_dim
        if config.use_junction:
            total_dim += config.junction_dim
        if config.use_structure:
            total_dim += config.structure_dim
        if config.use_gc_content:
            total_dim += 8
        if config.use_cpg_density:
            total_dim += 8

        # Aggregation layer
        if config.aggregation == "concat":
            self.aggregator = nn.Sequential(
                nn.Linear(total_dim, config.output_dim * 2),
                nn.LayerNorm(config.output_dim * 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.output_dim * 2, config.output_dim),
            )
        elif config.aggregation == "hierarchical":
            # Hierarchical aggregation: nt -> dinuc -> codon
            self.nt_to_dinuc = nn.Linear(config.nucleotide_dim * 2, config.dinucleotide_dim)
            self.dinuc_to_codon = nn.Linear(config.dinucleotide_dim * 3, config.output_dim)
            self.aggregator = None
        else:
            self.aggregator = nn.Linear(total_dim, config.output_dim)

    def compute_gc_content(
        self,
        nucleotides: torch.Tensor,
        window_size: int = 30,
    ) -> torch.Tensor:
        """Compute local GC content.

        Args:
            nucleotides: (batch, seq_len) nucleotide indices (G=2, C=1)
            window_size: Window for local GC computation

        Returns:
            (batch, seq_len, 1) GC content fractions
        """
        seq_len = nucleotides.shape[1]
        is_gc = ((nucleotides == 1) | (nucleotides == 2)).float()

        # Sliding window average with proper padding
        kernel = torch.ones(1, 1, window_size, device=nucleotides.device) / window_size
        pad_left = window_size // 2
        pad_right = window_size - pad_left - 1
        is_gc_padded = F.pad(is_gc.unsqueeze(1), (pad_left, pad_right), mode="replicate")
        gc_content = F.conv1d(is_gc_padded, kernel).squeeze(1)

        # Ensure exact size match
        gc_content = gc_content[:, :seq_len]

        return gc_content.unsqueeze(-1)

    def compute_cpg_density(
        self,
        nucleotides: torch.Tensor,
        window_size: int = 30,
    ) -> torch.Tensor:
        """Compute local CpG dinucleotide density.

        CpG dinucleotides are important for:
        - Immune recognition (unmethylated CpG activates TLR9)
        - mRNA stability
        - Codon optimization strategies

        Args:
            nucleotides: (batch, seq_len) nucleotide indices
            window_size: Window for local density

        Returns:
            (batch, seq_len, 1) CpG density
        """
        seq_len = nucleotides.shape[1]

        # CpG = C followed by G (indices 1, 2)
        is_c = (nucleotides[:, :-1] == 1).float()
        is_g = (nucleotides[:, 1:] == 2).float()
        is_cpg = is_c * is_g

        # Pad to original length
        is_cpg = F.pad(is_cpg, (0, 1))

        # Sliding window average with proper padding
        kernel = torch.ones(1, 1, window_size, device=nucleotides.device) / window_size
        pad_left = window_size // 2
        pad_right = window_size - pad_left - 1
        is_cpg_padded = F.pad(is_cpg.unsqueeze(1), (pad_left, pad_right), mode="replicate")
        cpg_density = F.conv1d(is_cpg_padded, kernel).squeeze(1)

        # Ensure exact size match
        cpg_density = cpg_density[:, :seq_len]

        return cpg_density.unsqueeze(-1)

    def forward(
        self,
        nucleotides: torch.Tensor,
        codon_positions: torch.Tensor | None = None,
        region_mask: torch.Tensor | None = None,
        structure_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Multi-scale encoding of nucleotide sequence.

        Args:
            nucleotides: (batch, seq_len) nucleotide indices 0-3
            codon_positions: (batch, seq_len) position within codon 0-2
            region_mask: (batch, seq_len) region labels for dinucleotide weighting
            structure_features: (batch, seq_len, window) pre-computed structure features

        Returns:
            Dictionary with:
                - 'output': (batch, n_features, output_dim) aggregated features
                - 'nucleotide': (batch, seq_len, nt_dim) nucleotide embeddings
                - 'dinucleotide': (batch, seq_len-1, dinuc_dim) dinucleotide embeddings
                - 'wobble': (batch, n_codons, pos_dim) wobble features
                - etc.
        """
        batch_size = nucleotides.shape[0]
        seq_len = nucleotides.shape[1]
        device = nucleotides.device

        outputs = {}

        # Generate codon positions if not provided
        if codon_positions is None:
            codon_positions = torch.arange(seq_len, device=device) % 3
            codon_positions = codon_positions.unsqueeze(0).expand(batch_size, -1)

        # 1. Nucleotide embeddings
        nt_emb = self.nucleotide_encoder(nucleotides, codon_positions)
        outputs["nucleotide"] = nt_emb

        features_to_aggregate = [nt_emb]

        # 2. Dinucleotide features
        if self.config.use_dinucleotide:
            dinuc_emb = self.dinucleotide_encoder(nucleotides, region_mask)
            # Pad to match nucleotide length (dinuc is seq_len-1)
            # Pad at the end to match seq_len
            pad_size = seq_len - dinuc_emb.shape[1]
            dinuc_emb = F.pad(dinuc_emb, (0, 0, 0, pad_size))
            outputs["dinucleotide"] = dinuc_emb
            features_to_aggregate.append(dinuc_emb)

        # 3. Wobble position features
        if self.config.use_position:
            # Extract wobble bases (every 3rd nucleotide starting at position 2)
            wobble_indices = torch.arange(2, seq_len, 3, device=device)
            if len(wobble_indices) > 0:
                wobble_bases = nucleotides[:, wobble_indices]
                wobble_emb = self.wobble_encoder(wobble_bases)
                # Expand to full sequence length
                full_wobble = torch.zeros(batch_size, seq_len, self.config.position_dim, device=device)
                full_wobble[:, wobble_indices] = wobble_emb
                outputs["wobble"] = full_wobble
                features_to_aggregate.append(full_wobble)

        # 4. Junction features
        if self.config.use_junction:
            junction_emb = self.junction_encoder(nucleotides)
            # Pad to match nucleotide length
            pad_size = seq_len - junction_emb.shape[1]
            junction_emb = F.pad(junction_emb, (0, 0, pad_size // 2, pad_size - pad_size // 2))
            outputs["junction"] = junction_emb
            features_to_aggregate.append(junction_emb)

        # 5. Structure features (if provided)
        if self.config.use_structure and structure_features is not None:
            struct_emb = self.structure_encoder(structure_features)
            outputs["structure"] = struct_emb
            features_to_aggregate.append(struct_emb)

        # 6. GC content
        if self.config.use_gc_content:
            gc = self.compute_gc_content(nucleotides)
            gc_emb = self.gc_proj(gc)
            outputs["gc_content"] = gc
            features_to_aggregate.append(gc_emb)

        # 7. CpG density
        if self.config.use_cpg_density:
            cpg = self.compute_cpg_density(nucleotides)
            cpg_emb = self.cpg_proj(cpg)
            outputs["cpg_density"] = cpg
            features_to_aggregate.append(cpg_emb)

        # Aggregate all features
        if self.config.aggregation == "hierarchical":
            # Hierarchical: nucleotides -> dinucleotides -> codons
            output = self._hierarchical_aggregate(outputs, nucleotides)
        else:
            combined = torch.cat(features_to_aggregate, dim=-1)
            output = self.aggregator(combined)

        outputs["output"] = output

        return outputs

    def _hierarchical_aggregate(
        self,
        features: dict[str, torch.Tensor],
        nucleotides: torch.Tensor,
    ) -> torch.Tensor:
        """Hierarchical aggregation: nt -> dinuc -> codon.

        Args:
            features: Dictionary of computed features
            nucleotides: Original nucleotide sequence

        Returns:
            (batch, n_codons, output_dim) codon-level features
        """
        batch_size = nucleotides.shape[0]
        seq_len = nucleotides.shape[1]
        n_codons = seq_len // 3

        # Get nucleotide features
        nt = features["nucleotide"]

        # Aggregate pairs to dinucleotides
        nt_pairs = nt.reshape(batch_size, seq_len // 2, -1)
        dinuc = self.nt_to_dinuc(nt_pairs.reshape(batch_size, seq_len // 2, -1))

        # Aggregate triplets to codons
        dinuc_triplets = dinuc[:, : n_codons * 3 // 2].reshape(batch_size, n_codons, -1)
        codon = self.dinuc_to_codon(dinuc_triplets.reshape(batch_size, n_codons, -1))

        return codon


class MultiScaleEncoderFactory:
    """Factory for creating multi-scale encoders."""

    @staticmethod
    def create_basic() -> MultiScaleNucleotideEncoder:
        """Create basic encoder with essential features."""
        config = MultiScaleConfig(
            use_structure=False,
            use_junction=False,
            aggregation="concat",
        )
        return MultiScaleNucleotideEncoder(config)

    @staticmethod
    def create_full() -> MultiScaleNucleotideEncoder:
        """Create full encoder with all features."""
        config = MultiScaleConfig(
            use_structure=True,
            use_junction=True,
            aggregation="hierarchical",
        )
        return MultiScaleNucleotideEncoder(config)

    @staticmethod
    def create_ribonn_style() -> MultiScaleNucleotideEncoder:
        """Create encoder inspired by RiboNN (dinucleotide focus)."""
        config = MultiScaleConfig(
            nucleotide_dim=8,
            dinucleotide_dim=64,
            position_dim=16,
            junction_dim=0,
            structure_dim=0,
            output_dim=64,
            use_dinucleotide=True,
            use_position=True,
            use_junction=False,
            use_structure=False,
            use_gc_content=True,
            use_cpg_density=True,
            aggregation="concat",
        )
        return MultiScaleNucleotideEncoder(config)


__all__ = [
    "MultiScaleNucleotideEncoder",
    "MultiScaleConfig",
    "MultiScaleEncoderFactory",
    "NucleotideEmbedding",
    "DinucleotideEncoder",
    "WobblePositionEncoder",
    "CodonJunctionEncoder",
    "LocalStructureEncoder",
    "RibosomeSiteEncoder",
]
