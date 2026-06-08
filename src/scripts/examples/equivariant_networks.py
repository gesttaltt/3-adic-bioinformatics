# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Example: Equivariant Networks for Protein Structure.

This script demonstrates the use of SO(3)/SE(3)-equivariant networks
and codon symmetry layers for protein-aware neural networks.

Usage:
    python src/scripts/examples/equivariant_networks.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from src.equivariant import (
    EGNN,
    CodonEmbedding,
    CodonSymmetryLayer,
    CodonTransformer,
    SO3Layer,
    SphericalHarmonics,
)


def example_spherical_harmonics():
    """Demonstrate spherical harmonics computation."""
    print("=" * 60)
    print("Spherical Harmonics")
    print("=" * 60)

    # Create spherical harmonics up to l=2
    sh = SphericalHarmonics(lmax=2)

    # Convert 3D points to spherical coordinates
    points = torch.randn(10, 3)  # 10 random 3D points
    points = points / points.norm(dim=-1, keepdim=True)  # Normalize to unit sphere

    # Compute spherical harmonic features
    features = sh(points)

    print(f"Input: {points.shape[0]} points on unit sphere")
    print(f"Output: {features.shape} spherical harmonic features")
    print("  l=0: 1 feature, l=1: 3 features, l=2: 5 features = 9 total")


def example_so3_layer():
    """Demonstrate SO(3)-equivariant layers."""
    print("\n" + "=" * 60)
    print("SO(3)-Equivariant Layer")
    print("=" * 60)

    # Create SO(3) layer
    layer = SO3Layer(
        in_features=16,
        out_features=32,
        lmax=2,
        use_radial=True,
    )

    # Input features on nodes with positions
    batch_size = 2
    n_nodes = 20
    positions = torch.randn(batch_size, n_nodes, 3)
    features = torch.randn(batch_size, n_nodes, 16)

    # Create some edges (k-nearest neighbors style)
    edge_index = torch.stack(
        [
            torch.randint(0, n_nodes, (100,)),
            torch.randint(0, n_nodes, (100,)),
        ]
    )

    output = layer(features, positions, edge_index)

    print(f"Input features: {features.shape}")
    print(f"Output features: {output.shape}")
    print("Layer is equivariant to 3D rotations!")


def example_se3_transformer():
    """Demonstrate SE(3)-equivariant transformer."""
    print("\n" + "=" * 60)
    print("EGNN (SE(3)-Equivariant Graph Neural Network)")
    print("=" * 60)

    # Create EGNN
    model = EGNN(
        in_channels=16,
        hidden_channels=32,
        out_channels=16,
        n_layers=2,
    )

    # Protein-like input (positions + features)
    n_residues = 50
    positions = torch.randn(1, n_residues, 3) * 10  # Ca positions
    features = torch.randn(1, n_residues, 16)  # Node features

    # Create sequential edges (backbone connectivity)
    src = torch.arange(n_residues - 1)
    dst = torch.arange(1, n_residues)
    edge_index = torch.stack(
        [
            torch.cat([src, dst]),  # Bidirectional
            torch.cat([dst, src]),
        ]
    )

    # Forward pass
    new_features, new_positions = model(features, positions, edge_index)

    print(f"Input: {n_residues} residues with positions and features")
    print(f"Output features: {new_features.shape}")
    print(f"Output positions: {new_positions.shape}")
    print("Both rotations AND translations are equivariant!")


def example_codon_symmetry():
    """Demonstrate codon symmetry layer."""
    print("\n" + "=" * 60)
    print("Codon Symmetry Layer")
    print("=" * 60)

    # Create codon embedding
    embed = CodonEmbedding(embed_dim=64)

    # Create codon symmetry layer
    sym_layer = CodonSymmetryLayer(
        hidden_dim=64,
        respect_wobble=True,
        respect_synonymy=True,
    )

    # Input: batch of codon sequences
    batch_size = 4
    seq_length = 30
    codons = torch.randint(0, 64, (batch_size, seq_length))

    # Embed codons
    embeddings = embed(codons)
    print(f"Codon embeddings: {embeddings.shape}")

    # Apply symmetry-aware layer
    output = sym_layer(embeddings)
    print(f"Symmetry-aware output: {output.shape}")

    # The layer respects:
    print("\nBiological symmetries encoded:")
    print("  - Wobble position: 3rd codon position is more flexible")
    print("  - Synonymous codons: Codons for same amino acid are grouped")


def example_codon_transformer():
    """Demonstrate full codon transformer."""
    print("\n" + "=" * 60)
    print("Codon Transformer")
    print("=" * 60)

    # Create codon transformer
    model = CodonTransformer(
        hidden_dim=64,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
    )

    # Input: codon sequence
    codons = torch.randint(0, 64, (2, 50))

    # Forward pass
    output = model(codons)

    print(f"Input codons: {codons.shape}")
    print(f"Output features: {output.shape}")
    print("Transformer respects codon symmetries at every layer!")


if __name__ == "__main__":
    print("Equivariant Networks for Protein Structure")
    print("=" * 60)
    print()

    example_spherical_harmonics()
    example_so3_layer()
    example_se3_transformer()
    example_codon_symmetry()
    example_codon_transformer()

    print("\n" + "=" * 60)
    print("Examples completed!")
