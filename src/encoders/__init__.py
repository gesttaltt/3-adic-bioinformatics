# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Specialized encoders for biological sequence and structure embedding.

This module provides a collection of encoders that transform biological data
into geometric representations suitable for the p-adic hyperbolic framework.

Encoder Categories:
    **Codon Encoding**:
        CodonEncoder: P-adic encoding of DNA/RNA codons

    **Post-Translational Modification (PTM)**:
        PTMGoldilocksEncoder: Encodes PTM sites with stability zones
        GoldilocksZone: Defines stability regions for modifications

    **Motor Proteins / Ternary Logic**:
        TernaryMotorEncoder: Ternary state encoding for molecular motors
        ATPSynthaseEncoder: Rotary motor state encoding

    **Circadian / Toroidal**:
        CircadianCycleEncoder: Temporal cycle encoding on torus
        KaiCClockEncoder: KaiC circadian clock protein encoding
        ToroidalEmbedding: General toroidal manifold embedding

    **Spectral / Holographic**:
        HolographicEncoder: Multi-scale spectral graph encoding
        GraphLaplacianEncoder: Laplacian eigenvector features
        PPINetworkEncoder: Protein-protein interaction networks

    **Diffusion Maps**:
        DiffusionMapEncoder: Nonlinear dimensionality reduction
        DiffusionPseudotime: Trajectory inference from diffusion

    **Geometric Vector Perceptron (GVP)**:
        GVPLayer: SE(3)-equivariant neural network layer
        PAdicGVP: P-adic enhanced GVP
        ProteinGVPEncoder: Protein structure encoding

    **Surface / MaSIF-style**:
        MaSIFEncoder: Molecular surface fingerprinting
        SurfacePatchEncoder: Local surface patch features
        GeodesicConv: Convolution on geodesic distances

Example:
    >>> from src.encoders import CodonEncoder, MaSIFEncoder
    >>> codon_enc = CodonEncoder(prime=3)
    >>> embeddings = codon_enc.encode_sequence("ATGCGA")
"""

from .alphafold_encoder import (
    AlphaFoldEncoder,
    AlphaFoldFeatureExtractor,
    AlphaFoldStructure,
    AlphaFoldStructureLoader,
)
from .circadian_encoder import CircadianCycleEncoder, KaiCClockEncoder, ToroidalEmbedding
from .codon_encoder import CodonEncoder
from .diffusion_encoder import (
    DiffusionMapEncoder,
    DiffusionMapResult,
    DiffusionPseudotime,
    KernelBuilder,
    MultiscaleDiffusion,
)
from .geometric_vector_perceptron import (
    CodonGVP,
    GVPLayer,
    GVPMessage,
    GVPOutput,
    PAdicGVP,
    ProteinGVPEncoder,
    VectorLinear,
)
from .holographic_encoder import (
    GraphLaplacianEncoder,
    HierarchicalProteinEmbedding,
    HolographicEncoder,
    MultiScaleGraphFeatures,
    PPINetworkEncoder,
)
from .hybrid_encoder import (
    CrossAttentionFusion,
    GatedFusion,
    HybridCodonEncoder,
    HybridEncoderConfig,
    HybridEncoderFactory,
    PLMBackend,
)
from .hyperbolic_codon_encoder import (
    HyperbolicCodonEncoder,
    compute_codon_padic_valuation,
    compute_target_radius,
    extract_codon_embeddings_from_vae,
)
from .motor_encoder import ATPSynthaseEncoder, RotaryPositionEncoder, TernaryMotorEncoder
from .multiscale_nucleotide_encoder import (
    CodonJunctionEncoder,
    DinucleotideEncoder,
    LocalStructureEncoder,
    MultiScaleConfig,
    MultiScaleEncoderFactory,
    MultiScaleNucleotideEncoder,
    NucleotideEmbedding,
    RibosomeSiteEncoder,
    WobblePositionEncoder,
)
from .padic_amino_acid_encoder import (
    AA_TO_GROUP,
    AA_TO_INDEX,
    AminoAcidGroup,
    FiveAdicAminoAcidEncoder,
    MultiPrimeAminoAcidEncoder,
    MutationType,
    MutationTypeEmbedding,
    SevenAdicSecondaryStructureEncoder,
    compute_5adic_distance,
    compute_5adic_distance_matrix,
)
from .ptm_encoder import GoldilocksZone, PTMDataset, PTMGoldilocksEncoder, PTMType
from .surface_encoder import (
    GeodesicConv,
    MaSIFEncoder,
    PAdicSurfaceAttention,
    SurfaceComplementarity,
    SurfaceEncoderOutput,
    SurfaceFeatureExtractor,
    SurfaceInteractionPredictor,
    SurfacePatchEncoder,
)
from .tam_aware_encoder import (
    TAM_PATHWAYS,
    NRTIFeatureExtractor,
    TAMAwareEncoder,
    detect_tam_patterns,
    extract_tam_features,
)
from .trainable_codon_encoder import (
    TrainableCodonEncoder,
    codon_to_onehot_12dim,
    compute_codon_hierarchy_level,
    train_codon_encoder,
)

__all__ = [
    # Codon encoding
    "CodonEncoder",
    # Hyperbolic codon encoding (p-adic native)
    "HyperbolicCodonEncoder",
    "compute_codon_padic_valuation",
    "compute_target_radius",
    "extract_codon_embeddings_from_vae",
    # Trainable codon encoder (12-dim input)
    "TrainableCodonEncoder",
    "train_codon_encoder",
    "codon_to_onehot_12dim",
    "compute_codon_hierarchy_level",
    # Hybrid encoder (ESM-2 + Codon)
    "HybridCodonEncoder",
    "HybridEncoderConfig",
    "HybridEncoderFactory",
    "PLMBackend",
    "CrossAttentionFusion",
    "GatedFusion",
    # Multi-scale nucleotide encoder
    "MultiScaleNucleotideEncoder",
    "MultiScaleConfig",
    "MultiScaleEncoderFactory",
    "NucleotideEmbedding",
    "DinucleotideEncoder",
    "WobblePositionEncoder",
    "CodonJunctionEncoder",
    "LocalStructureEncoder",
    "RibosomeSiteEncoder",
    # PTM encoding
    "PTMType",
    "GoldilocksZone",
    "PTMGoldilocksEncoder",
    "PTMDataset",
    # Motor/ternary encoding
    "TernaryMotorEncoder",
    "ATPSynthaseEncoder",
    "RotaryPositionEncoder",
    # Circadian/toroidal encoding
    "CircadianCycleEncoder",
    "KaiCClockEncoder",
    "ToroidalEmbedding",
    # Holographic/spectral encoding
    "HolographicEncoder",
    "GraphLaplacianEncoder",
    "MultiScaleGraphFeatures",
    "PPINetworkEncoder",
    "HierarchicalProteinEmbedding",
    # Diffusion map encoding
    "DiffusionMapEncoder",
    "DiffusionMapResult",
    "DiffusionPseudotime",
    "KernelBuilder",
    "MultiscaleDiffusion",
    # Geometric Vector Perceptron
    "GVPLayer",
    "GVPMessage",
    "GVPOutput",
    "VectorLinear",
    "PAdicGVP",
    "ProteinGVPEncoder",
    "CodonGVP",
    # MaSIF-style surface encoder
    "MaSIFEncoder",
    "SurfaceEncoderOutput",
    "SurfacePatchEncoder",
    "SurfaceFeatureExtractor",
    "GeodesicConv",
    "PAdicSurfaceAttention",
    "SurfaceInteractionPredictor",
    "SurfaceComplementarity",
    # TAM-aware NRTI encoding
    "TAMAwareEncoder",
    "TAM_PATHWAYS",
    "NRTIFeatureExtractor",
    "detect_tam_patterns",
    "extract_tam_features",
    # P-adic amino acid encoding
    "FiveAdicAminoAcidEncoder",
    "SevenAdicSecondaryStructureEncoder",
    "MultiPrimeAminoAcidEncoder",
    "MutationTypeEmbedding",
    "AminoAcidGroup",
    "MutationType",
    "AA_TO_GROUP",
    "AA_TO_INDEX",
    "compute_5adic_distance",
    "compute_5adic_distance_matrix",
    # AlphaFold structure encoder
    "AlphaFoldEncoder",
    "AlphaFoldStructureLoader",
    "AlphaFoldStructure",
    "AlphaFoldFeatureExtractor",
]
