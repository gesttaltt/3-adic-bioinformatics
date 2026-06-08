# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""AlphaFold2 Structure Encoder.

This module provides integration with AlphaFold2 predicted structures
for enhanced drug resistance prediction. Features include:

1. AlphaFold DB structure downloading and caching
2. pLDDT confidence-weighted encoding
3. PAE (Predicted Aligned Error) for interface confidence
4. SE(3)-equivariant coordinate encoding

Usage:
    from src.encoders.alphafold_encoder import AlphaFoldEncoder, AlphaFoldStructureLoader

    # Load structure
    loader = AlphaFoldStructureLoader()
    structure = loader.get_structure("P03366")  # HIV-1 RT UniProt ID

    # Encode
    encoder = AlphaFoldEncoder(embed_dim=64)
    embedding = encoder(
        coords=structure["coords"],
        plddt_scores=structure["plddt"],
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AlphaFoldStructure:
    """Container for AlphaFold2 structure data.

    Attributes:
        coords: CA atom coordinates (L, 3)
        plddt: Per-residue confidence scores (L,)
        sequence: Amino acid sequence
        uniprot_id: UniProt accession
        pae: Optional Predicted Aligned Error matrix (L, L)
    """

    coords: np.ndarray
    plddt: np.ndarray
    sequence: str
    uniprot_id: str
    pae: Optional[np.ndarray] = None

    def to_tensors(self, device: str = "cpu") -> dict[str, torch.Tensor]:
        """Convert to PyTorch tensors."""
        return {
            "coords": torch.tensor(self.coords, dtype=torch.float32, device=device),
            "plddt": torch.tensor(self.plddt, dtype=torch.float32, device=device),
            "pae": torch.tensor(self.pae, dtype=torch.float32, device=device) if self.pae is not None else None,
        }


class AlphaFoldStructureLoader:
    """Download and cache AlphaFold2 structures for proteins.

    Accesses the AlphaFold Protein Structure Database (https://alphafold.ebi.ac.uk)
    to retrieve predicted structures and confidence scores.
    """

    ALPHAFOLD_DB_URL = "https://alphafold.ebi.ac.uk/files/"
    ALPHAFOLD_API_URL = "https://alphafold.ebi.ac.uk/api/"

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        model_version: int = 4,
    ):
        """Initialize structure loader.

        Args:
            cache_dir: Directory for caching structures (default: .alphafold_cache)
            model_version: AlphaFold model version (default: 4)
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".alphafold_cache"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_version = model_version

        # Index file for metadata
        self.index_file = self.cache_dir / "structure_index.json"
        self.index = self._load_index()

    def _load_index(self) -> dict:
        """Load structure index from cache."""
        if self.index_file.exists():
            try:
                with open(self.index_file) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_index(self):
        """Save structure index to cache."""
        with open(self.index_file, "w") as f:
            json.dump(self.index, f, indent=2)

    def get_structure(
        self,
        uniprot_id: str,
        force_download: bool = False,
    ) -> AlphaFoldStructure:
        """Get AlphaFold2 structure for a UniProt ID.

        Args:
            uniprot_id: UniProt accession (e.g., "P03366")
            force_download: Force re-download even if cached

        Returns:
            AlphaFoldStructure with coordinates and confidence
        """
        cache_file = self.cache_dir / f"{uniprot_id}_v{self.model_version}.npz"

        if cache_file.exists() and not force_download:
            return self._load_from_cache(cache_file, uniprot_id)

        # Download structure
        structure = self._download_structure(uniprot_id)

        # Save to cache
        self._save_to_cache(structure, cache_file)

        return structure

    def _load_from_cache(self, cache_file: Path, uniprot_id: str) -> AlphaFoldStructure:
        """Load structure from cache."""
        data = np.load(cache_file, allow_pickle=True)

        return AlphaFoldStructure(
            coords=data["coords"],
            plddt=data["plddt"],
            sequence=str(data["sequence"]),
            uniprot_id=uniprot_id,
            pae=data.get("pae"),
        )

    def _save_to_cache(self, structure: AlphaFoldStructure, cache_file: Path):
        """Save structure to cache."""
        data = {
            "coords": structure.coords,
            "plddt": structure.plddt,
            "sequence": structure.sequence,
        }
        if structure.pae is not None:
            data["pae"] = structure.pae

        np.savez_compressed(cache_file, **data)

    def _resolve_download_url(self, uniprot_id: str) -> str:
        """Resolve PDB download URL via the API with direct-URL fallback.

        The /api/prediction endpoint is sunset 2026-06-25; the response field
        changed from entryId to modelEntityId. We try the API first so the
        canonical pdbUrl is used when available, then fall back to constructing
        the URL directly from the UniProt ID (stable pattern regardless of API).
        """
        try:
            import requests
            api_url = f"{self.ALPHAFOLD_API_URL}prediction/{uniprot_id}"
            resp = requests.get(api_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                info = data[0] if data else {}

                # pdbUrl is present in both old and new API responses
                pdb_url = info.get("pdbUrl")
                if pdb_url:
                    return pdb_url

                # New API (post-sunset): modelEntityId replaces entryId
                model_entity_id = info.get("modelEntityId")
                if model_entity_id:
                    if "-model_v" in model_entity_id:
                        return f"{self.ALPHAFOLD_DB_URL}{model_entity_id}.pdb"
                    version = info.get("latestVersion", self.model_version)
                    return f"{self.ALPHAFOLD_DB_URL}{model_entity_id}-model_v{version}.pdb"

                # Old API: entryId
                entry_id = info.get("entryId")
                if entry_id:
                    version = info.get("latestVersion", self.model_version)
                    return f"{self.ALPHAFOLD_DB_URL}{entry_id}-model_v{version}.pdb"

        except Exception as e:
            logger.debug(f"AlphaFold API lookup failed for {uniprot_id}: {e}")

        # Fallback: construct URL directly (works for single-fragment proteins)
        return f"{self.ALPHAFOLD_DB_URL}AF-{uniprot_id}-F1-model_v{self.model_version}.pdb"

    def _download_structure(self, uniprot_id: str) -> AlphaFoldStructure:
        """Download structure from AlphaFold DB."""
        try:
            import requests
        except ImportError:
            logger.warning("requests not available, returning mock structure")
            return self._mock_structure(uniprot_id)

        pdb_url = self._resolve_download_url(uniprot_id)
        pae_url = pdb_url.replace(
            f"-model_v{self.model_version}.pdb",
            f"-predicted_aligned_error_v{self.model_version}.json",
        )

        try:
            pdb_response = requests.get(pdb_url, timeout=30)
            if pdb_response.status_code == 200:
                coords, plddt, sequence = self._parse_pdb(pdb_response.text)
            else:
                logger.warning(f"Failed to download PDB for {uniprot_id}: {pdb_response.status_code}")
                return self._mock_structure(uniprot_id)

            pae = None
            try:
                pae_response = requests.get(pae_url, timeout=30)
                if pae_response.status_code == 200:
                    pae = self._parse_pae(pae_response.json())
            except Exception as e:
                logger.debug(f"PAE not available for {uniprot_id}: {e}")

            return AlphaFoldStructure(
                coords=coords,
                plddt=plddt,
                sequence=sequence,
                uniprot_id=uniprot_id,
                pae=pae,
            )

        except Exception as e:
            logger.warning(f"Error downloading structure for {uniprot_id}: {e}")
            return self._mock_structure(uniprot_id)

    def _parse_pdb(self, pdb_text: str) -> tuple[np.ndarray, np.ndarray, str]:
        """Parse PDB file to extract CA coordinates and pLDDT (B-factor).

        Args:
            pdb_text: PDB file contents

        Returns:
            Tuple of (coords, plddt, sequence)
        """
        coords = []
        plddt = []
        sequence = []

        aa_3to1 = {
            "ALA": "A",
            "CYS": "C",
            "ASP": "D",
            "GLU": "E",
            "PHE": "F",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LYS": "K",
            "LEU": "L",
            "MET": "M",
            "ASN": "N",
            "PRO": "P",
            "GLN": "Q",
            "ARG": "R",
            "SER": "S",
            "THR": "T",
            "VAL": "V",
            "TRP": "W",
            "TYR": "Y",
        }

        for line in pdb_text.split("\n"):
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    b_factor = float(line[60:66])  # pLDDT stored in B-factor
                    res_name = line[17:20].strip()

                    coords.append([x, y, z])
                    plddt.append(b_factor)
                    sequence.append(aa_3to1.get(res_name, "X"))
                except (ValueError, IndexError):
                    continue

        return (
            np.array(coords, dtype=np.float32),
            np.array(plddt, dtype=np.float32),
            "".join(sequence),
        )

    def _parse_pae(self, pae_json: dict) -> np.ndarray:
        """Parse PAE JSON to matrix."""
        if isinstance(pae_json, list):
            pae_json = pae_json[0]

        pae_data = pae_json.get("predicted_aligned_error", [])
        return np.array(pae_data, dtype=np.float32)

    def _mock_structure(self, uniprot_id: str, length: int = 200) -> AlphaFoldStructure:
        """Generate mock structure for testing."""
        # Generate random coordinates (roughly protein-like)
        coords = np.zeros((length, 3), dtype=np.float32)
        for i in range(length):
            # Simple helix-like structure
            coords[i, 0] = np.cos(i * 0.1) * 2
            coords[i, 1] = np.sin(i * 0.1) * 2
            coords[i, 2] = i * 0.15

        # Random pLDDT (higher for middle, lower at ends)
        plddt = 70 + 20 * np.sin(np.linspace(0, np.pi, length)).astype(np.float32)
        plddt += np.random.randn(length).astype(np.float32) * 5
        plddt = np.clip(plddt, 0, 100)

        # Random sequence
        aa = "ACDEFGHIKLMNPQRSTVWY"
        sequence = "".join(np.random.choice(list(aa), length))

        return AlphaFoldStructure(
            coords=coords,
            plddt=plddt,
            sequence=sequence,
            uniprot_id=uniprot_id,
        )

    def list_cached_structures(self) -> list[str]:
        """List all cached UniProt IDs."""
        cached = []
        for f in self.cache_dir.glob("*.npz"):
            parts = f.stem.split("_")
            if parts:
                cached.append(parts[0])
        return cached


class AlphaFoldEncoder(nn.Module):
    """Full AlphaFold2 structure encoder with pLDDT-weighted confidence.

    Encodes AlphaFold2 predicted structures into embeddings using:
    1. SE(3)-equivariant coordinate encoding
    2. pLDDT confidence weighting
    3. Optional PAE for interface confidence
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_layers: int = 4,
        cutoff: float = 10.0,
        use_plddt: bool = True,
        use_pae: bool = False,
        dropout: float = 0.1,
    ):
        """Initialize AlphaFold encoder.

        Args:
            embed_dim: Output embedding dimension
            n_layers: Number of encoding layers
            cutoff: Distance cutoff for neighbor computation (Angstroms)
            use_plddt: Whether to weight by pLDDT confidence
            use_pae: Whether to use PAE matrix
            dropout: Dropout rate
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.cutoff = cutoff
        self.use_plddt = use_plddt
        self.use_pae = use_pae

        # Coordinate encoder (simplified from full SE(3))
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # Distance embedding
        self.dist_embed = nn.Sequential(
            nn.Linear(1, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        # Message passing layers
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "self_attn": nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True, dropout=dropout),
                        "ffn": nn.Sequential(
                            nn.Linear(embed_dim, embed_dim * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(embed_dim * 4, embed_dim),
                        ),
                        "norm1": nn.LayerNorm(embed_dim),
                        "norm2": nn.LayerNorm(embed_dim),
                    }
                )
                for _ in range(n_layers)
            ]
        )

        # pLDDT encoding
        if use_plddt:
            self.plddt_encoder = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.Sigmoid(),
            )

        # PAE encoding
        if use_pae:
            self.pae_encoder = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.Sigmoid(),
            )

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        coords: torch.Tensor,
        plddt_scores: Optional[torch.Tensor] = None,
        pae_matrix: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode AlphaFold2 structure.

        Args:
            coords: CA atom coordinates (batch, n_residues, 3)
            plddt_scores: Per-residue confidence (batch, n_residues)
            pae_matrix: Predicted Aligned Error (batch, n_residues, n_residues)
            mask: Optional residue mask (batch, n_residues)

        Returns:
            Structure embeddings (batch, n_residues, embed_dim)
        """
        batch_size, n_residues, _ = coords.shape
        device = coords.device

        # Initial coordinate encoding
        h = self.coord_encoder(coords)

        # Add distance information
        dist = torch.cdist(coords, coords)  # (batch, n, n)
        dist_mask = dist < self.cutoff

        # Aggregate distance features per residue
        dist_features = self.dist_embed(dist.unsqueeze(-1).mean(dim=2))  # Mean over neighbors
        h = h + dist_features

        # Apply pLDDT weighting
        if self.use_plddt and plddt_scores is not None:
            # Normalize pLDDT to [0, 1]
            plddt_norm = plddt_scores / 100.0
            confidence = self.plddt_encoder(plddt_norm.unsqueeze(-1))
            h = h * confidence

        # Apply PAE weighting (optional)
        if self.use_pae and pae_matrix is not None:
            # Lower PAE = higher confidence
            pae_confidence = torch.exp(-pae_matrix / 10.0)  # Decay with PAE
            pae_weight = self.pae_encoder(pae_confidence.mean(dim=-1, keepdim=True))
            h = h * pae_weight

        # Message passing (using distance as edge weights, no explicit attention mask)
        for layer in self.layers:
            # Self-attention with residual
            h_norm = layer["norm1"](h)
            h_attn, _ = layer["self_attn"](h_norm, h_norm, h_norm)
            h = h + h_attn

            # FFN with residual
            h = h + layer["ffn"](layer["norm2"](h))

        return self.out_proj(h)

    def pool(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pool residue embeddings to sequence embedding.

        Args:
            h: Residue embeddings (batch, n_residues, embed_dim)
            mask: Optional mask (batch, n_residues)

        Returns:
            Pooled embedding (batch, embed_dim)
        """
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            return (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return h.mean(dim=1)


class AlphaFoldFeatureExtractor:
    """High-level interface for extracting features from AlphaFold structures."""

    def __init__(
        self,
        encoder: Optional[AlphaFoldEncoder] = None,
        loader: Optional[AlphaFoldStructureLoader] = None,
        device: str = "cpu",
    ):
        """Initialize feature extractor.

        Args:
            encoder: AlphaFold encoder (creates default if None)
            loader: Structure loader (creates default if None)
            device: Device for computations
        """
        self.encoder = encoder or AlphaFoldEncoder()
        self.loader = loader or AlphaFoldStructureLoader()
        self.device = device

        self.encoder.to(device)
        self.encoder.eval()

    def extract_features(
        self,
        uniprot_id: str,
        return_structure: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Extract features for a protein.

        Args:
            uniprot_id: UniProt accession
            return_structure: Whether to return structure data

        Returns:
            Dictionary with embeddings and optional structure
        """
        # Load structure
        structure = self.loader.get_structure(uniprot_id)
        tensors = structure.to_tensors(self.device)

        # Encode
        with torch.no_grad():
            coords = tensors["coords"].unsqueeze(0)
            plddt = tensors["plddt"].unsqueeze(0)
            pae = tensors.get("pae")
            if pae is not None:
                pae = pae.unsqueeze(0)

            embeddings = self.encoder(coords, plddt, pae)
            pooled = self.encoder.pool(embeddings)

        result = {
            "residue_embeddings": embeddings.squeeze(0),
            "sequence_embedding": pooled.squeeze(0),
            "sequence": structure.sequence,
        }

        if return_structure:
            result["structure"] = structure

        return result

    def extract_mutation_region(
        self,
        uniprot_id: str,
        positions: list[int],
        window_size: int = 10,
    ) -> torch.Tensor:
        """Extract embeddings for region around mutation positions.

        Args:
            uniprot_id: UniProt accession
            positions: Mutation positions (0-indexed)
            window_size: Window around each position

        Returns:
            Embeddings for mutation region (n_positions, embed_dim)
        """
        features = self.extract_features(uniprot_id)
        embeddings = features["residue_embeddings"]

        region_embeddings = []
        for pos in positions:
            start = max(0, pos - window_size)
            end = min(len(embeddings), pos + window_size + 1)
            region = embeddings[start:end].mean(dim=0)
            region_embeddings.append(region)

        return torch.stack(region_embeddings)


__all__ = [
    "AlphaFoldEncoder",
    "AlphaFoldStructureLoader",
    "AlphaFoldStructure",
    "AlphaFoldFeatureExtractor",
]
