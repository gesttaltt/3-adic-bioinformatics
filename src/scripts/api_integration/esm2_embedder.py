#!/usr/bin/env python3
"""
ESM-2 Protein Embedding Extractor
=================================

This module provides utilities for extracting ESM-2 embeddings
from protein sequences for HIV drug resistance prediction.

Usage:
    # Basic usage
    embedder = ESM2Embedder()
    embedding = embedder.embed("PQVTLWQRPL...")

    # Batch processing
    embeddings = embedder.embed_batch(sequences)

    # Pre-compute for dataset
    embedder.precompute_dataset(sequences_dict, output_path)

Author: Claude Code
Date: December 28, 2024
"""

import warnings
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")


class ESM2Embedder:
    """
    Extract protein embeddings using ESM-2 model.

    ESM-2 (Evolutionary Scale Modeling 2) is a protein language model
    that learns representations from protein sequences. These embeddings
    capture evolutionary and structural information that can improve
    drug resistance prediction.

    Attributes:
        model_size: Size of ESM-2 model ('small', 'medium', 'large')
        device: Device to run model on ('cuda' or 'cpu')
    """

    MODELS = {
        "small": {"name": "facebook/esm2_t6_8M_UR50D", "dim": 320, "params": "8M", "memory": "~500MB"},
        "medium": {"name": "facebook/esm2_t12_35M_UR50D", "dim": 480, "params": "35M", "memory": "~1GB"},
        "large": {"name": "facebook/esm2_t33_650M_UR50D", "dim": 1280, "params": "650M", "memory": "~3GB"},
    }

    def __init__(self, model_size: str = "small", device: str | None = None, half_precision: bool = False):
        """
        Initialize ESM-2 embedder.

        Args:
            model_size: Model size - 'small' (8M), 'medium' (35M), or 'large' (650M)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            half_precision: Use FP16 for reduced memory (large model only)
        """
        from transformers import AutoModel, AutoTokenizer

        if model_size not in self.MODELS:
            raise ValueError(f"Unknown model size: {model_size}. Choose from {list(self.MODELS.keys())}")

        self.model_size = model_size
        self.model_info = self.MODELS[model_size]

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading ESM-2 model: {self.model_info['name']}")
        print(f"  Parameters: {self.model_info['params']}")
        print(f"  Embedding dim: {self.model_info['dim']}")
        print(f"  Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_info["name"])
        self.model = AutoModel.from_pretrained(self.model_info["name"])

        if half_precision and self.device == "cuda":
            self.model = self.model.half()

        self.model = self.model.to(self.device)
        self.model.eval()

        print("  Model loaded successfully!")

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.model_info["dim"]

    def embed(self, sequence: str, pooling: str = "mean") -> np.ndarray:
        """
        Get embedding for a single sequence.

        Args:
            sequence: Protein sequence (amino acid string)
            pooling: Pooling strategy - 'mean', 'cls', or 'none'
                - 'mean': Average across all positions
                - 'cls': Use [CLS] token embedding
                - 'none': Return full per-position embeddings

        Returns:
            Embedding array with shape:
                - 'mean'/'cls': (embedding_dim,)
                - 'none': (seq_len, embedding_dim)
        """
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state  # [1, seq_len+2, dim]

        if pooling == "mean":
            # Mean over all positions (excluding special tokens)
            return embeddings[:, 1:-1, :].mean(dim=1).squeeze().cpu().numpy()
        elif pooling == "cls":
            # [CLS] token embedding
            return embeddings[:, 0, :].squeeze().cpu().numpy()
        elif pooling == "none":
            # Full per-position embeddings (excluding special tokens)
            return embeddings[:, 1:-1, :].squeeze().cpu().numpy()
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

    def embed_batch(
        self, sequences: list[str], pooling: str = "mean", batch_size: int = 8, show_progress: bool = True
    ) -> np.ndarray:
        """
        Get embeddings for multiple sequences.

        Args:
            sequences: List of protein sequences
            pooling: Pooling strategy ('mean', 'cls')
            batch_size: Number of sequences per batch
            show_progress: Show progress bar

        Returns:
            Array of embeddings with shape (n_sequences, embedding_dim)
        """
        embeddings = []

        iterator = range(0, len(sequences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding sequences")

        for i in iterator:
            batch = sequences[i : i + batch_size]

            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True, max_length=1024, add_special_tokens=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            batch_embeddings = outputs.last_hidden_state

            if pooling == "mean":
                # Mean pooling with attention mask
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                batch_embeddings = (batch_embeddings * attention_mask).sum(dim=1)
                batch_embeddings = batch_embeddings / attention_mask.sum(dim=1)
            elif pooling == "cls":
                batch_embeddings = batch_embeddings[:, 0, :]

            embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def embed_mutation(self, wt_sequence: str, mutation: str) -> dict[str, np.ndarray]:
        """
        Get embeddings for wild-type and mutant sequences.

        Args:
            wt_sequence: Wild-type sequence
            mutation: Mutation string (e.g., 'M46I')

        Returns:
            Dictionary with 'wt', 'mut', 'diff', and 'similarity'
        """
        # Parse mutation
        wt_aa = mutation[0]
        pos = int(mutation[1:-1]) - 1  # Convert to 0-indexed
        mut_aa = mutation[-1]

        # Verify wild-type
        if wt_sequence[pos] != wt_aa:
            raise ValueError(f"Position {pos + 1} is {wt_sequence[pos]}, not {wt_aa}")

        # Create mutant sequence
        mut_sequence = wt_sequence[:pos] + mut_aa + wt_sequence[pos + 1 :]

        # Get embeddings
        wt_emb = self.embed(wt_sequence)
        mut_emb = self.embed(mut_sequence)

        # Calculate difference and similarity
        diff = mut_emb - wt_emb
        similarity = np.dot(wt_emb, mut_emb) / (np.linalg.norm(wt_emb) * np.linalg.norm(mut_emb))

        return {
            "wt": wt_emb,
            "mut": mut_emb,
            "diff": diff,
            "cosine_similarity": similarity,
            "euclidean_distance": np.linalg.norm(diff),
        }

    def precompute_dataset(self, sequences: dict[str, str], output_path: str | Path, pooling: str = "mean"):
        """
        Pre-compute embeddings for entire dataset and save to disk.

        Args:
            sequences: Dictionary of {sequence_id: sequence}
            output_path: Path to save embeddings (.npz file)
            pooling: Pooling strategy
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        ids = list(sequences.keys())
        seqs = list(sequences.values())

        print(f"Computing embeddings for {len(seqs)} sequences...")
        embeddings = self.embed_batch(seqs, pooling=pooling)

        # Save as numpy archive
        np.savez(
            output_path, ids=np.array(ids), embeddings=embeddings, model=self.model_info["name"], dim=self.embedding_dim
        )

        print(f"Saved embeddings to: {output_path}")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    @staticmethod
    def load_precomputed(path: str | Path) -> dict[str, np.ndarray]:
        """
        Load pre-computed embeddings from disk.

        Args:
            path: Path to .npz file

        Returns:
            Dictionary of {sequence_id: embedding}
        """
        data = np.load(path, allow_pickle=True)
        return dict(zip(data["ids"], data["embeddings"], strict=False))


class ESM2VAEEncoder:
    """
    VAE encoder that uses ESM-2 embeddings as input.

    This class wraps ESM-2 embeddings into a format suitable
    for our existing VAE architecture.
    """

    def __init__(self, embedder: ESM2Embedder, latent_dim: int = 16):
        """
        Initialize ESM-2 VAE encoder.

        Args:
            embedder: ESM2Embedder instance
            latent_dim: Latent dimension for VAE
        """
        import torch.nn as nn

        self.embedder = embedder
        self.latent_dim = latent_dim

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Linear(embedder.embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def encode(self, sequence: str) -> tuple:
        """
        Encode sequence to latent space.

        Args:
            sequence: Protein sequence

        Returns:
            Tuple of (mu, logvar) tensors
        """
        import torch

        # Get ESM-2 embedding
        emb = self.embedder.embed(sequence)
        emb_tensor = torch.FloatTensor(emb).unsqueeze(0)

        # Pass through encoder
        h = self.encoder(emb_tensor)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


def demo():
    """Demonstrate ESM-2 embedder functionality."""
    print("=" * 60)
    print(" ESM-2 EMBEDDER DEMO")
    print("=" * 60)

    # HIV-1 Protease sequence
    hiv_protease = "PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"

    # Initialize embedder
    embedder = ESM2Embedder(model_size="small")

    # Single sequence embedding
    print("\n1. Single Sequence Embedding")
    print("-" * 40)
    emb = embedder.embed(hiv_protease)
    print(f"   Sequence length: {len(hiv_protease)}")
    print(f"   Embedding shape: {emb.shape}")
    print(f"   Mean: {emb.mean():.4f}")
    print(f"   Std: {emb.std():.4f}")

    # Mutation analysis
    print("\n2. Mutation Effect Analysis")
    print("-" * 40)
    mutations = ["M46I", "I54V", "V82A", "L90M"]
    for mut in mutations:
        try:
            result = embedder.embed_mutation(hiv_protease, mut)
            print(f"   {mut}: cos_sim={result['cosine_similarity']:.6f}, dist={result['euclidean_distance']:.4f}")
        except Exception as e:
            print(f"   {mut}: Error - {e}")

    # Per-position embeddings
    print("\n3. Per-Position Embeddings")
    print("-" * 40)
    pos_emb = embedder.embed(hiv_protease, pooling="none")
    print(f"   Shape: {pos_emb.shape}")
    print(f"   Position 46 (M46I site): mean={pos_emb[45].mean():.4f}")
    print(f"   Position 82 (V82A site): mean={pos_emb[81].mean():.4f}")

    print("\n" + "=" * 60)
    print(" DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo()
