#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Complete protein design workflow example.

This script demonstrates an end-to-end protein design pipeline:
1. Load protein structure (or generate example)
2. Compute topological features
3. Encode structure with equivariant networks
4. Generate codon sequences with diffusion
5. Analyze results with information geometry

Usage:
    python src/scripts/examples/protein_design_workflow.py
"""

from __future__ import annotations

import sys

import torch


def create_example_structure(n_residues: int = 50) -> tuple:
    """Create an example protein structure.

    Returns:
        Tuple of (ca_coords, residue_features)
    """
    # Simulate an alpha helix structure
    # Helix parameters: 3.6 residues per turn, 1.5A rise per residue
    t = torch.arange(n_residues).float()
    radius = 2.3  # Angstroms

    x = radius * torch.cos(2 * 3.14159 * t / 3.6)
    y = radius * torch.sin(2 * 3.14159 * t / 3.6)
    z = 1.5 * t

    ca_coords = torch.stack([x, y, z], dim=1)

    # Random residue features (would be amino acid embeddings in practice)
    residue_features = torch.randn(n_residues, 16)

    return ca_coords, residue_features


def build_contact_graph(coords: torch.Tensor, threshold: float = 10.0) -> torch.Tensor:
    """Build contact graph from Ca coordinates.

    Args:
        coords: (n_residues, 3) Ca coordinates
        threshold: Distance threshold in Angstroms

    Returns:
        edge_index: (2, n_edges) edge connectivity
    """
    dist = torch.cdist(coords, coords)
    contact_mask = (dist < threshold) & (dist > 0)
    edge_index = contact_mask.nonzero().T
    return edge_index


def compute_topological_features(coords: torch.Tensor) -> dict:
    """Compute topological features of the protein structure.

    Args:
        coords: (n_residues, 3) Ca coordinates

    Returns:
        Dictionary with topological features
    """
    from src.topology import PersistenceVectorizer, RipsFiltration

    # Compute persistent homology
    filtration = RipsFiltration(max_dimension=1, max_edge_length=15.0)
    fingerprint = filtration.build(coords)

    # Vectorize for ML
    vectorizer = PersistenceVectorizer(method="statistics")
    topo_vector = vectorizer.transform(fingerprint)

    return {
        "fingerprint": fingerprint,
        "vector": torch.from_numpy(topo_vector).float(),
        "total_features": fingerprint.total_features,
    }


def encode_structure(
    coords: torch.Tensor,
    features: torch.Tensor,
    edge_index: torch.Tensor,
) -> torch.Tensor:
    """Encode protein structure with equivariant network.

    Args:
        coords: (n_residues, 3) Ca coordinates
        features: (n_residues, in_features) residue features
        edge_index: (2, n_edges) graph connectivity

    Returns:
        (n_residues, out_features) structure embeddings
    """
    from src.equivariant import EGNN

    # Create EGNN encoder
    encoder = EGNN(
        in_features=features.shape[1],
        hidden_dim=64,
        out_features=32,
        n_layers=3,
    )
    encoder.eval()

    # Encode structure
    with torch.no_grad():
        embeddings, _ = encoder(features, coords, edge_index)

    return embeddings


def generate_codon_sequences(
    n_residues: int,
    n_designs: int = 5,
) -> torch.Tensor:
    """Generate codon sequences using diffusion.

    Args:
        n_residues: Number of residues (codons)
        n_designs: Number of sequence designs to generate

    Returns:
        (n_designs, n_residues) codon indices
    """
    from src.diffusion import CodonDiffusion

    # Create diffusion model
    model = CodonDiffusion(
        n_steps=100,
        vocab_size=64,  # 64 codons
        hidden_dim=128,
        n_layers=4,
    )
    model.eval()

    # Generate sequences
    with torch.no_grad():
        sequences = model.sample(n_samples=n_designs, seq_length=n_residues)

    return sequences


def analyze_embeddings(embeddings: torch.Tensor) -> dict:
    """Analyze embeddings using hyperbolic geometry.

    Args:
        embeddings: (n_residues, dim) structure embeddings

    Returns:
        Dictionary with analysis results
    """
    from src.graphs import PoincareOperations

    # Project to Poincare ball
    poincare = PoincareOperations(curvature=1.0)

    # Normalize to stay in ball
    embeddings_normalized = embeddings * 0.3 / (embeddings.norm(dim=-1, keepdim=True) + 1e-6)

    # Compute pairwise hyperbolic distances
    n = embeddings_normalized.shape[0]
    distances = torch.zeros(n, n)
    for i in range(n):
        dist = poincare.distance(
            embeddings_normalized[i : i + 1].expand(n, -1),
            embeddings_normalized,
        )
        # Handle different return shapes
        if dist.dim() > 1:
            dist = dist.squeeze()
        distances[i] = dist

    # Analyze distance distribution
    upper_tri = distances[torch.triu(torch.ones(n, n), diagonal=1) == 1]

    return {
        "mean_distance": upper_tri.mean().item(),
        "max_distance": upper_tri.max().item(),
        "min_distance": upper_tri[upper_tri > 0].min().item() if (upper_tri > 0).any() else 0,
        "distance_std": upper_tri.std().item(),
    }


def decode_codons(codon_indices: torch.Tensor) -> list:
    """Decode codon indices to nucleotide sequences.

    Args:
        codon_indices: (n_sequences, seq_length) codon indices

    Returns:
        List of nucleotide sequence strings
    """
    # Standard codon table (UCAG ordering)
    BASES = "UCAG"
    CODONS = [f"{b1}{b2}{b3}" for b1 in BASES for b2 in BASES for b3 in BASES]

    sequences = []
    for seq in codon_indices:
        codons = [CODONS[idx.item()] for idx in seq]
        sequences.append("-".join(codons))

    return sequences


def main():
    """Run complete protein design workflow."""
    print("=" * 60)
    print("PROTEIN DESIGN WORKFLOW EXAMPLE")
    print("=" * 60)

    # Step 1: Create/load structure
    print("\n1. Creating example protein structure...")
    n_residues = 30
    coords, features = create_example_structure(n_residues)
    print(f"   Structure: {n_residues} residues")
    print(f"   Coordinates shape: {coords.shape}")

    # Step 2: Compute topological features
    print("\n2. Computing topological features...")
    topo = compute_topological_features(coords)
    print(f"   Total persistent features: {topo['total_features']}")
    print(f"   Feature vector shape: {topo['vector'].shape}")

    # Step 3: Build contact graph
    print("\n3. Building contact graph...")
    edge_index = build_contact_graph(coords, threshold=8.0)
    print(f"   Number of contacts: {edge_index.shape[1]}")

    # Step 4: Encode structure
    print("\n4. Encoding structure with EGNN...")
    embeddings = encode_structure(coords, features, edge_index)
    print(f"   Embedding shape: {embeddings.shape}")

    # Step 5: Analyze embeddings
    print("\n5. Analyzing embeddings in hyperbolic space...")
    analysis = analyze_embeddings(embeddings)
    print(f"   Mean hyperbolic distance: {analysis['mean_distance']:.3f}")
    print(f"   Distance range: [{analysis['min_distance']:.3f}, {analysis['max_distance']:.3f}]")

    # Step 6: Generate codon sequences
    print("\n6. Generating codon sequences with diffusion...")
    n_designs = 3
    codon_sequences = generate_codon_sequences(n_residues, n_designs)
    print(f"   Generated {n_designs} sequence designs")

    # Step 7: Decode sequences
    print("\n7. Decoding codon sequences...")
    nucleotide_seqs = decode_codons(codon_sequences)
    for i, seq in enumerate(nucleotide_seqs):
        # Show first 10 codons
        codons = seq.split("-")[:10]
        print(f"   Design {i + 1}: {'-'.join(codons)}...")

    print("\n" + "=" * 60)
    print("Workflow complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
