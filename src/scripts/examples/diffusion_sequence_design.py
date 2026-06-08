# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Example: Codon Sequence Design with Diffusion Models.

This script demonstrates how to use the discrete diffusion model
to generate codon sequences, either unconditionally or conditioned
on protein backbone structure.

Usage:
    python src/scripts/examples/diffusion_sequence_design.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from src.diffusion import CodonDiffusion, StructureConditionedGen


def example_unconditional_generation():
    """Generate codon sequences unconditionally."""
    print("=" * 60)
    print("Unconditional Codon Sequence Generation")
    print("=" * 60)

    # Create diffusion model
    model = CodonDiffusion(
        n_steps=100,  # Diffusion steps (reduce for speed)
        vocab_size=64,  # 64 codons
        hidden_dim=128,  # Model hidden dimension
        n_layers=4,  # Transformer layers
    )
    model.eval()

    # Generate sequences
    print("\nGenerating 5 sequences of length 50 codons...")
    with torch.no_grad():
        sequences = model.sample(
            n_samples=5,
            seq_length=50,
            device="cpu",
        )

    print(f"Generated sequences shape: {sequences.shape}")
    print(f"Sample sequence (codon indices): {sequences[0, :10].tolist()}...")

    # Decode to codons
    CODONS = [f"{b1}{b2}{b3}" for b1 in "UCAG" for b2 in "UCAG" for b3 in "UCAG"]
    print("\nFirst sequence decoded:")
    decoded = [CODONS[idx] for idx in sequences[0, :10].tolist()]
    print(f"  {' '.join(decoded)}...")


def example_structure_conditioned_generation():
    """Generate codon sequences conditioned on backbone structure."""
    print("\n" + "=" * 60)
    print("Structure-Conditioned Sequence Design")
    print("=" * 60)

    # Create structure-conditioned generator
    model = StructureConditionedGen(
        hidden_dim=64,  # Smaller for demo
        n_diffusion_steps=50,
        n_layers=2,
        n_encoder_layers=1,
    )
    model.eval()

    # Create mock backbone coordinates (normally from PDB)
    # Shape: (batch, n_residues, 3) for Ca atoms
    n_residues = 30
    backbone = torch.randn(1, n_residues, 3) * 5  # Random structure

    print(f"\nInput backbone: {n_residues} residues")

    # Design sequences for this structure
    print("Designing 3 sequences...")
    with torch.no_grad():
        sequences = model.design(backbone, n_designs=3)

    print(f"Designed sequences shape: {sequences.shape}")
    print(f"Each sequence has {sequences.shape[1]} codons")


def example_training_loop():
    """Demonstrate training loop for diffusion model."""
    print("\n" + "=" * 60)
    print("Training Loop Example")
    print("=" * 60)

    # Create model
    model = CodonDiffusion(
        n_steps=100,
        vocab_size=64,
        hidden_dim=64,
        n_layers=2,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create mock training data
    batch_size = 8
    seq_length = 40
    train_data = torch.randint(0, 64, (batch_size, seq_length))

    print(f"\nTraining batch: {batch_size} sequences x {seq_length} codons")

    # Training step
    model.train()
    for step in range(3):
        optimizer.zero_grad()

        # Forward pass returns loss dict
        result = model.training_step(train_data)
        loss = result["loss"]

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Step {step + 1}: loss = {loss.item():.4f}, accuracy = {result['accuracy'].item():.4f}")


if __name__ == "__main__":
    print("Diffusion Models for Codon Sequence Design")
    print("=" * 60)
    print()

    example_unconditional_generation()
    example_structure_conditioned_generation()
    example_training_loop()

    print("\n" + "=" * 60)
    print("Examples completed!")
