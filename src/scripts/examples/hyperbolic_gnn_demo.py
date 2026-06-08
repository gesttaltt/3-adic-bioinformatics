# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Example: Hyperbolic Graph Neural Networks.

This script demonstrates the use of hyperbolic GNNs for learning
hierarchical representations of protein graphs.

Usage:
    python src/scripts/examples/hyperbolic_gnn_demo.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from src.graphs import (
    HyboWaveNet,
    HyperbolicGraphConv,
    HyperbolicLinear,
    LorentzOperations,
    PoincareOperations,
    SpectralWavelet,
)


def example_poincare_operations():
    """Demonstrate Poincare ball operations."""
    print("=" * 60)
    print("Poincare Ball Operations")
    print("=" * 60)

    # Create Poincare operations with curvature c=1
    poincare = PoincareOperations(curvature=1.0)

    # Points in the Poincare ball (must have norm < 1)
    x = torch.randn(10, 8) * 0.3  # 10 points in 8D
    y = torch.randn(10, 8) * 0.3

    # Mobius addition
    z = poincare.mobius_add(x, y)
    print("\nMobius addition: x + y (in hyperbolic space)")
    print(f"  Input norms: x={x.norm(dim=-1).mean():.3f}, y={y.norm(dim=-1).mean():.3f}")
    print(f"  Output norm: {z.norm(dim=-1).mean():.3f}")

    # Exponential map (Euclidean tangent -> hyperbolic)
    v = torch.randn(10, 8) * 0.1  # Tangent vector
    origin = torch.zeros(1, 8)
    exp_v = poincare.exp_map(v, origin)
    print("\nExponential map: tangent -> Poincare ball")
    print(f"  Tangent vector norm: {v.norm(dim=-1).mean():.3f}")
    print(f"  Mapped point norm: {exp_v.norm(dim=-1).mean():.3f}")

    # Hyperbolic distance
    dist = poincare.distance(x, y)
    print("\nHyperbolic distance:")
    print(f"  Mean distance: {dist.mean():.3f}")
    print("  (Grows faster near boundary of ball)")


def example_lorentz_operations():
    """Demonstrate Lorentz model operations."""
    print("\n" + "=" * 60)
    print("Lorentz (Hyperboloid) Model")
    print("=" * 60)

    # Create Lorentz operations
    lorentz = LorentzOperations(curvature=1.0)

    # Points on the hyperboloid
    # First coordinate is "time", rest is "space"
    # Must satisfy: -x0^2 + x1^2 + ... + xn^2 = -1/c
    space = torch.randn(10, 7) * 0.3
    time = torch.sqrt(1 + (space**2).sum(dim=-1, keepdim=True))
    x = torch.cat([time, space], dim=-1)

    print(f"Points on hyperboloid: {x.shape}")

    # Lorentzian inner product (should be -1 for unit hyperboloid)
    inner = lorentz.lorentzian_inner(x, x)
    print(f"Lorentzian norm (should be ~-1): {inner.mean():.4f}")


def example_hyperbolic_linear():
    """Demonstrate hyperbolic linear layer."""
    print("\n" + "=" * 60)
    print("Hyperbolic Linear Layer")
    print("=" * 60)

    # Create hyperbolic linear layer
    layer = HyperbolicLinear(
        in_features=32,
        out_features=64,
        curvature=1.0,
    )

    # Input in Poincare ball
    x = torch.randn(8, 32) * 0.3
    x = x / (x.norm(dim=-1, keepdim=True) + 1)  # Ensure in ball

    # Forward pass
    y = layer(x)

    print(f"Input: {x.shape} (in Poincare ball)")
    print(f"Output: {y.shape} (still in Poincare ball)")
    print(f"Output norm: {y.norm(dim=-1).mean():.3f} (< 1)")


def example_hyperbolic_graph_conv():
    """Demonstrate hyperbolic graph convolution."""
    print("\n" + "=" * 60)
    print("Hyperbolic Graph Convolution")
    print("=" * 60)

    # Create hyperbolic graph conv
    conv = HyperbolicGraphConv(
        in_channels=16,
        out_channels=32,
        curvature=1.0,
        use_attention=True,
    )

    # Graph structure
    n_nodes = 50
    n_edges = 200

    # Node features (in Poincare ball)
    x = torch.randn(n_nodes, 16) * 0.3

    # Random edges
    edge_index = torch.stack(
        [
            torch.randint(0, n_nodes, (n_edges,)),
            torch.randint(0, n_nodes, (n_edges,)),
        ]
    )

    # Forward pass
    out = conv(x, edge_index)

    print(f"Input: {n_nodes} nodes with 16 features")
    print(f"Graph: {n_edges} edges")
    print(f"Output: {out.shape}")


def example_spectral_wavelet():
    """Demonstrate spectral wavelet decomposition."""
    print("\n" + "=" * 60)
    print("Spectral Wavelet Decomposition")
    print("=" * 60)

    # Create wavelet module
    wavelet = SpectralWavelet(n_scales=4)

    # Graph Laplacian (symmetric, positive semi-definite)
    n_nodes = 30
    # Random adjacency matrix
    adj = torch.rand(n_nodes, n_nodes)
    adj = (adj + adj.T) / 2  # Symmetric
    adj = adj * (torch.rand(n_nodes, n_nodes) > 0.7).float()  # Sparse
    # Laplacian
    degree = adj.sum(dim=1)
    laplacian = torch.diag(degree) - adj

    # Node features
    x = torch.randn(n_nodes, 16)

    # Multi-scale wavelet features
    wavelets = wavelet(x, laplacian)

    print(f"Input: {n_nodes} nodes, 16 features")
    print(f"Scales: {wavelet.n_scales}")
    print(f"Output: {wavelets.shape}")
    print("Each scale captures different frequency information!")


def example_hybowave_net():
    """Demonstrate full HyboWaveNet model."""
    print("\n" + "=" * 60)
    print("HyboWaveNet: Wavelet + Hyperbolic GNN")
    print("=" * 60)

    # Create HyboWaveNet
    model = HyboWaveNet(
        in_channels=16,
        hidden_channels=32,
        out_channels=16,
        n_scales=3,
        curvature=1.0,
        n_layers=2,
    )

    # Protein-like graph
    n_nodes = 100
    x = torch.randn(n_nodes, 16)

    # Backbone connectivity + some long-range contacts
    edges = []
    for i in range(n_nodes - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    # Add some random contacts
    for _ in range(50):
        i, j = torch.randint(0, n_nodes, (2,)).tolist()
        edges.append([i, j])
        edges.append([j, i])

    edge_index = torch.tensor(edges).T

    # Forward pass
    node_embeddings = model(x, edge_index)
    graph_embedding = model.encode_graph(x, edge_index)

    print(f"Input: {n_nodes} residues, {edge_index.shape[1]} edges")
    print(f"Node embeddings: {node_embeddings.shape}")
    print(f"Graph embedding: {graph_embedding.shape}")
    print("\nHyperbolic space naturally captures:")
    print("  - Hierarchical protein structure")
    print("  - Tree-like domain organization")
    print("  - Evolutionary relationships")


if __name__ == "__main__":
    print("Hyperbolic Graph Neural Networks")
    print("=" * 60)
    print()

    example_poincare_operations()
    example_lorentz_operations()
    example_hyperbolic_linear()
    example_hyperbolic_graph_conv()
    example_spectral_wavelet()
    example_hybowave_net()

    print("\n" + "=" * 60)
    print("Examples completed!")
