"""
Sliding Window Embedder for Large Genomes (Arboviruses)

This script breaks large sequences (e.g., Dengue ~11kb) into overlapping windows,
embeds them using the Ternary VAE, and creates a "Hyperbolic Trajectory" tensor.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

try:
    from research.bioinformatics.codon_encoder_research.hiv.src.hyperbolic_utils import (
        AA_TO_CODON,
        encode_sequence_hyperbolic,
        load_hyperbolic_encoder,
    )
except ImportError:
    # Fail gracefully if utils not found (e.g., in CI)
    print("Warning: hyperbolic_utils not found. Mocking for infrastructure test.")
    AA_TO_CODON = {}


def sliding_window(seq: str, window_size: int, stride: int) -> list[str]:
    """Generator for sliding windows."""
    windows = []
    for i in range(0, len(seq) - window_size + 1, stride):
        windows.append(seq[i : i + window_size])
    return windows


def embed_genome_trajectory(
    fasta_path: str, output_path: str, window_size: int = 300, stride: int = 50, device: str = "cpu"
):
    """
    Embeds a genome into a sequence of hyperbolic vectors.

    Args:
        fasta_path: Path to input FASTA (dengue/zika genome)
        output_path: Path to save .pt tensor
        window_size: Size of window in nucleotides (or AA if translated)
        stride: Step size
    """
    input_file = Path(fasta_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"I: Loading genome from {input_file}")

    # Simple AA Translation Logic (Mock for now, assuming input is Protein or we handle translation later)
    # For MVP Phase 0, we assume the input is already an AA sequence or we treat raw nucleotides
    # if the encoder supports it. The current HIV encoder expects AA.
    # We will assume INPUT IS AMINO ACID for now (polyprotein).

    records = list(SeqIO.parse(input_file, "fasta"))
    if not records:
        print("E: No records found in FASTA.")
        return

    # Load Encoder
    print("I: Loading Hyperbolic Encoder...")
    try:
        encoder, _ = load_hyperbolic_encoder(version="3adic", device=device)
    except Exception as e:
        print(f"E: Failed to load encoder: {e}")
        return

    all_trajectories = {}

    for record in records:
        seq = str(record.seq).upper()
        # Filter non-AA
        seq = "".join([c for c in seq if c in AA_TO_CODON])

        windows = sliding_window(seq, window_size, stride)
        print(f"I: Processing {record.id}: {len(seq)} residues -> {len(windows)} windows")

        vectors = []
        for win in windows:
            # Embed window
            # encode_sequence_hyperbolic returns (L, 16)
            # We pool to (16,) to get the window's centroid
            emb = encode_sequence_hyperbolic(win, encoder, AA_TO_CODON)
            centroid = np.mean(emb, axis=0)
            vectors.append(centroid)

        if vectors:
            tensor = torch.tensor(np.stack(vectors), dtype=torch.float32)
            all_trajectories[record.id] = tensor

    if all_trajectories:
        torch.save(all_trajectories, output_file)
        print(f"S: Saved trajectories to {output_file}")
        # Print shape of first trajectory
        key = list(all_trajectories.keys())[0]
        print(f"   Shape of {key}: {all_trajectories[key].shape}")
    else:
        print("W: No valid trajectories generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input FASTA file")
    parser.add_argument("--output", required=True, help="Output .pt file")
    parser.add_argument("--window", type=int, default=100, help="Window size (AA residues)")
    parser.add_argument("--stride", type=int, default=10, help="Stride (AA residues)")
    args = parser.parse_args()

    embed_genome_trajectory(args.input, args.output, args.window, args.stride)
