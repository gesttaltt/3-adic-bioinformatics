"""
StarPepDB Ingestion and Processing Script

This script handles the ingestion of peptide sequences from StarPepDB (or compatible CSV formats)
and processes them for use in the Ternary VAE pipeline.

Functionality:
1. Loads raw peptide data (CSV).
2. Filters for valid amino acid sequences.
3. Encodes peptides into 3-adic hyperbolic embeddings using the project's CodonEncoder.
4. Saves the processed dataset (embeddings + metadata) for model training.

Usage:
    python ingest_starpep.py --input data/raw/starpep.csv --output data/processed/starpep_hyperbolic.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

try:
    # Attempt to import from the location found in research/
    from research.bioinformatics.codon_encoder_research.hiv.src.hyperbolic_utils import (
        AA_TO_CODON,
        encode_sequence_hyperbolic,
        load_hyperbolic_encoder,
    )
except ImportError:
    print("Error: Could not import hyperbolic_utils. Please ensure the project structure is correct.")
    sys.exit(1)


def ingest_starpep(input_path: str, output_path: str):
    """
    Process StarPepDB data.

    Args:
        input_path: Path to raw CSV file (columns: 'sequence', 'activity', etc.)
        output_path: Path to save processed .pt file
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    # create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_file}...")
    if not input_file.exists():
        print(f"Warning: Input file {input_file} not found. Creating dummy data for demonstration.")
        # Create dummy dataframe for testing/demo
        df = pd.DataFrame(
            {
                "sequence": ["AKRGG", "GGLKK", "AKR", "INVALIDX"],
                "activity": [1, 1, 0, 0],
            }
        )
    else:
        df = pd.read_csv(input_file)

    print(f"Raw data shape: {df.shape}")

    # load encoder
    print("Loading Hyperbolic Codon Encoder...")
    try:
        # Assuming '3adic' version is available
        encoder, _ = load_hyperbolic_encoder(version="3adic", device="cpu")
    except Exception as e:
        print(f"Failed to load 3adic encoder: {e}")
        print("Falling back to legacy/mock for structure verification.")
        return

    embeddings_list = []
    valid_indices = []

    print("Encoding sequences...")
    for idx, row in df.iterrows():
        seq = str(row.get("sequence", "")).strip()

        # Basic validation
        if not seq or any(c not in AA_TO_CODON for c in seq.upper()):
            continue

        # Encode
        try:
            # Returns (L, 16) array
            emb = encode_sequence_hyperbolic(seq, encoder, AA_TO_CODON)

            # For fixed-size input, we might want to pool (mean/max) or pad.
            # Here we compute the hyperbolic centroid (average embedding of the peptide)
            # effectively a "geometric mean" in the p-adic space.
            peptide_embedding = np.mean(emb, axis=0)

            embeddings_list.append(peptide_embedding)
            valid_indices.append(idx)
        except Exception as e:
            print(f"Error encoding sequence {seq}: {e}")
            continue

    if not embeddings_list:
        print("No valid sequences found/encoded.")
        return

    embeddings_tensor = torch.tensor(np.stack(embeddings_list), dtype=torch.float32)
    metadata = df.loc[valid_indices].reset_index(drop=True)

    print(f"Successfully encoded {len(embeddings_list)} peptides.")
    print(f"Embedding shape: {embeddings_tensor.shape}")

    # Save
    save_dict = {
        "embeddings": embeddings_tensor,
        "metadata": metadata,
        "encoder_version": "3adic",
    }

    torch.save(save_dict, output_file)
    print(f"Saved processed data to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest StarPepDB data")
    parser.add_argument(
        "--input",
        default=str(RAW_DATA_DIR / "starpep.csv"),
        help="Path to input CSV",
    )
    parser.add_argument(
        "--output",
        default=str(PROCESSED_DATA_DIR / "starpep_hyperbolic.pt"),
        help="Path to output .pt file",
    )

    args = parser.parse_args()

    ingest_starpep(args.input, args.output)
