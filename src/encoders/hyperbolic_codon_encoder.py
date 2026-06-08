# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Hyperbolic Codon Encoder - Naturally p-adic in Poincare Ball.

This module implements a codon encoder that lives natively in hyperbolic space,
preserving the ultrametric (p-adic) structure of the genetic code.

Key Features:
- 64 codon embeddings stored as ManifoldParameter on Poincare ball
- P-adic structure encoded in radial position (valuation -> radius)
- Amino acid properties encoded in angular position
- Frechet mean aggregation for synonymous codons
- Native hyperbolic distance computation

Mathematical Foundation:
- P-adic valuation maps to hyperbolic radius: v=0 -> r~0.9, v=2 -> r~0.1
- This preserves ultrametric property: d(x,z) <= max(d(x,y), d(y,z))
- Hyperbolic space naturally represents hierarchical data

Usage:
    from src.encoders.hyperbolic_codon_encoder import HyperbolicCodonEncoder

    encoder = HyperbolicCodonEncoder(embedding_dim=16)
    z_hyp = encoder(codon_indices)  # Already on Poincare ball

    # Get amino acid embedding via Frechet mean
    z_aa = encoder.get_amino_acid_embedding('A')  # Aggregates all Ala codons
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from geoopt import ManifoldParameter

from src.biology.codons import (
    AMINO_ACID_TO_CODONS,
    CODON_TO_INDEX,
    GENETIC_CODE,
    codon_index_to_triplet,
)
from src.encoders.codon_encoder import AA_PROPERTIES
from src.geometry import (
    exp_map_zero,
    get_manifold,
    log_map_zero,
    poincare_distance,
    project_to_poincare,
)


def compute_codon_padic_valuation(idx: int, p: int = 3) -> int:
    """Compute p-adic valuation based on codon position matching.

    For 64 codons organized as B1*16 + B2*4 + B3:
    - v=0: All codons (default, most common)
    - v=1: Codons with same first base as reference (16 per group)
    - v=2: Codons with same first+second base (4 per group, highest hierarchy)

    We use a simpler scheme: valuation based on index structure.
    """
    # For codon embeddings, we use a hierarchical scheme based on base positions
    # Group by first base (0-15, 16-31, 32-47, 48-63)
    # This gives a natural 4-adic structure that we approximate as 3-adic

    # Simplified: use index modular structure
    # v = number of trailing zeros in base-4 representation
    if idx == 0:
        return 2  # Special: stop codon or reference

    # Check divisibility by powers of 4 (approximating 3-adic in 4-base system)
    if idx % 16 == 0:
        return 2
    elif idx % 4 == 0:
        return 1
    else:
        return 0


def compute_target_radius(valuation: int, max_radius: float = 0.95, min_radius: float = 0.1) -> float:
    """Map p-adic valuation to hyperbolic radius.

    Higher valuation (more divisible) -> smaller radius (closer to center)
    This encodes hierarchy: v=0 at boundary, v=2 at center.

    Args:
        valuation: P-adic valuation (0, 1, or 2 typically)
        max_radius: Maximum radius for v=0
        min_radius: Minimum radius for max valuation

    Returns:
        Target radius in [min_radius, max_radius]
    """
    # Linear interpolation: v=0 -> max, v=2 -> min
    max_valuation = 2.0
    t = min(valuation / max_valuation, 1.0)
    return max_radius - t * (max_radius - min_radius)


class HyperbolicCodonEncoder(nn.Module):
    """Codon encoder with native hyperbolic embeddings.

    Embeddings live on the Poincare ball, preserving p-adic ultrametric structure.
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        curvature: float = 1.0,
        max_radius: float = 0.95,
        min_radius: float = 0.1,
        use_padic_init: bool = True,
        learnable: bool = True,
    ):
        """Initialize HyperbolicCodonEncoder.

        Args:
            embedding_dim: Dimension of hyperbolic embeddings
            curvature: Poincare ball curvature (c > 0)
            max_radius: Maximum radius for v=0 codons (boundary)
            min_radius: Minimum radius for high-valuation codons (center)
            use_padic_init: Initialize with p-adic radial structure
            learnable: If True, embeddings are trainable parameters
        """
        super().__init__()

        self.num_codons = 64
        self.embedding_dim = embedding_dim
        self.curvature = curvature
        self.max_radius = max_radius
        self.min_radius = min_radius

        # Get manifold
        self.manifold = get_manifold(c=curvature)

        # Initialize embeddings
        init_embeddings = self._create_padic_embeddings() if use_padic_init else self._create_random_embeddings()

        if learnable:
            self.embeddings = ManifoldParameter(
                init_embeddings,
                manifold=self.manifold,
            )
        else:
            self.register_buffer("embeddings", init_embeddings)

        # Cache amino acid to codon index mapping
        self._aa_to_indices = {}
        for aa, codons in AMINO_ACID_TO_CODONS.items():
            self._aa_to_indices[aa] = [CODON_TO_INDEX[c] for c in codons]

    def _create_random_embeddings(self) -> torch.Tensor:
        """Create random embeddings on the Poincare ball."""
        # Random tangent vectors, then exp map to ball
        tangent = torch.randn(self.num_codons, self.embedding_dim) * 0.1
        embeddings = exp_map_zero(tangent, c=self.curvature)
        return project_to_poincare(embeddings, max_norm=self.max_radius, c=self.curvature)

    def _create_padic_embeddings(self) -> torch.Tensor:
        """Create embeddings with p-adic radial structure.

        Strategy:
        1. Assign radius based on p-adic valuation
        2. Assign angle based on amino acid properties
        3. Map (r, theta) to Poincare ball coordinates
        """
        embeddings = torch.zeros(self.num_codons, self.embedding_dim)

        for idx in range(self.num_codons):
            triplet = codon_index_to_triplet(idx)
            aa = GENETIC_CODE.get(triplet, "*")

            # Compute target radius from valuation
            valuation = compute_codon_padic_valuation(idx)
            target_r = compute_target_radius(valuation, self.max_radius, self.min_radius)

            # Get amino acid properties for angular direction
            props = AA_PROPERTIES.get(aa, (0, 0, 0, 0))

            # Create direction vector from properties
            # Use first 4 dims for AA properties, rest for codon-specific info
            direction = torch.zeros(self.embedding_dim)

            # Amino acid properties (normalized to unit sphere in first 4 dims)
            if self.embedding_dim >= 4:
                direction[0] = props[0]  # hydrophobicity
                direction[1] = props[1]  # charge
                direction[2] = props[2]  # size
                direction[3] = props[3]  # polarity

            # Codon-specific: encode base positions in remaining dims
            b1 = (idx // 16) % 4
            b2 = (idx // 4) % 4
            b3 = idx % 4

            if self.embedding_dim >= 7:
                # One-hot-ish encoding of position within AA synonymous group
                direction[4] = (b1 - 1.5) / 2.0  # Center around 0
                direction[5] = (b2 - 1.5) / 2.0
                direction[6] = (b3 - 1.5) / 2.0

            # Add small noise to remaining dimensions
            if self.embedding_dim > 7:
                direction[7:] = torch.randn(self.embedding_dim - 7) * 0.1

            # Normalize direction
            dir_norm = torch.norm(direction)
            if dir_norm > 1e-6:
                direction = direction / dir_norm
            else:
                # Fallback: random direction
                direction = torch.randn(self.embedding_dim)
                direction = direction / torch.norm(direction)

            # Scale by target radius (this is the tangent vector magnitude)
            # For exp map: ||exp_0(v)|| ~ tanh(||v||) for small v
            # So we need ||v|| ~ arctanh(r)
            arctanh_r = 0.5 * math.log((1 + target_r) / (1 - target_r + 1e-10))
            tangent_vector = direction * arctanh_r

            # Map to Poincare ball
            embeddings[idx] = exp_map_zero(tangent_vector.unsqueeze(0), c=self.curvature).squeeze(0)

        return project_to_poincare(embeddings, max_norm=self.max_radius, c=self.curvature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed codon indices to Poincare ball.

        Args:
            x: Tensor of codon indices, shape (Batch, ...) with values in [0, 63]

        Returns:
            Hyperbolic embeddings, shape (Batch, ..., embedding_dim)
        """
        return self.embeddings[x]

    def get_hyperbolic_radius(self, idx: int | None = None) -> torch.Tensor:
        """Get hyperbolic radius (distance from origin) for codon(s).

        Args:
            idx: Codon index (0-63), or None for all codons

        Returns:
            Hyperbolic radius, shape () or (64,)
        """
        origin = torch.zeros(1, self.embedding_dim, device=self.embeddings.device)

        if idx is not None:
            emb = self.embeddings[idx : idx + 1]
            return poincare_distance(emb, origin, c=self.curvature).squeeze()
        else:
            radii = []
            for i in range(self.num_codons):
                emb = self.embeddings[i : i + 1]
                radii.append(poincare_distance(emb, origin, c=self.curvature))
            return torch.cat(radii)

    def get_hyperbolic_distance_matrix(self) -> torch.Tensor:
        """Compute pairwise hyperbolic distances between all codons.

        Returns:
            Distance matrix, shape (64, 64)
        """
        dist_matrix = torch.zeros(self.num_codons, self.num_codons, device=self.embeddings.device)

        for i in range(self.num_codons):
            for j in range(i + 1, self.num_codons):
                d = poincare_distance(self.embeddings[i : i + 1], self.embeddings[j : j + 1], c=self.curvature)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        return dist_matrix

    def frechet_mean(self, indices: list[int], max_iter: int = 10, tol: float = 1e-6) -> torch.Tensor:
        """Compute Frechet mean of codon embeddings.

        The Frechet mean minimizes sum of squared hyperbolic distances.
        Uses iterative algorithm in tangent space.

        Args:
            indices: List of codon indices to average
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Frechet mean embedding, shape (embedding_dim,)
        """
        if len(indices) == 0:
            return torch.zeros(self.embedding_dim, device=self.embeddings.device)

        if len(indices) == 1:
            return self.embeddings[indices[0]].clone()

        # Initialize with first point
        mean = self.embeddings[indices[0]].clone()

        for _ in range(max_iter):
            # Compute tangent vectors from mean to all points
            tangent_sum = torch.zeros_like(mean)
            for idx in indices:
                point = self.embeddings[idx]
                # Log map: mean -> point
                tangent = log_map_zero(point.unsqueeze(0) - mean.unsqueeze(0), c=self.curvature).squeeze(0)
                tangent_sum = tangent_sum + tangent

            # Average tangent vector
            tangent_avg = tangent_sum / len(indices)

            # Check convergence
            if torch.norm(tangent_avg) < tol:
                break

            # Update mean via exp map
            new_mean = exp_map_zero(tangent_avg.unsqueeze(0), c=self.curvature).squeeze(0)
            mean = project_to_poincare(mean + new_mean, max_norm=self.max_radius, c=self.curvature)

        return mean

    def get_amino_acid_embedding(self, aa: str, method: str = "frechet") -> torch.Tensor:
        """Get aggregated embedding for an amino acid.

        Aggregates embeddings of all synonymous codons encoding this AA.

        Args:
            aa: Single-letter amino acid code
            method: 'frechet' (hyperbolic mean) or 'centroid' (Euclidean, faster)

        Returns:
            Amino acid embedding, shape (embedding_dim,)
        """
        indices = self._aa_to_indices.get(aa, [])

        if len(indices) == 0:
            return torch.zeros(self.embedding_dim, device=self.embeddings.device)

        if method == "frechet":
            return self.frechet_mean(indices)
        else:
            # Euclidean centroid (faster but geometrically incorrect)
            embs = self.embeddings[indices]
            centroid = embs.mean(dim=0)
            return project_to_poincare(centroid.unsqueeze(0), self.max_radius, self.curvature).squeeze(0)

    def get_all_amino_acid_embeddings(self, method: str = "frechet") -> dict[str, torch.Tensor]:
        """Get embeddings for all 20 standard amino acids.

        Args:
            method: Aggregation method ('frechet' or 'centroid')

        Returns:
            Dictionary mapping AA letter to embedding tensor
        """
        return {aa: self.get_amino_acid_embedding(aa, method) for aa in "ACDEFGHIKLMNPQRSTVWY"}

    def compute_aa_hyperbolic_distance(self, aa1: str, aa2: str, method: str = "frechet") -> float:
        """Compute hyperbolic distance between amino acids.

        Args:
            aa1, aa2: Single-letter amino acid codes
            method: Aggregation method for AA embeddings

        Returns:
            Hyperbolic distance between AA embeddings
        """
        emb1 = self.get_amino_acid_embedding(aa1, method)
        emb2 = self.get_amino_acid_embedding(aa2, method)
        return poincare_distance(emb1.unsqueeze(0), emb2.unsqueeze(0), c=self.curvature).item()

    def compute_padic_structure_loss(self) -> torch.Tensor:
        """Compute loss encouraging radial ordering by p-adic valuation.

        Loss penalizes when:
        - High-valuation codons have larger radius than low-valuation
        - Same-valuation codons have very different radii

        Returns:
            Scalar loss tensor
        """
        radii = self.get_hyperbolic_radius()

        # Compute target radii
        target_radii = torch.tensor(
            [
                compute_target_radius(compute_codon_padic_valuation(i), self.max_radius, self.min_radius)
                for i in range(self.num_codons)
            ],
            device=radii.device,
        )

        # MSE between actual and target radii
        return torch.mean((radii - target_radii) ** 2)

    def compute_synonymous_cohesion_loss(self) -> torch.Tensor:
        """Compute loss encouraging synonymous codons to cluster.

        Synonymous codons (encoding same AA) should be close in hyperbolic space.

        Returns:
            Scalar loss tensor
        """
        total_loss = torch.tensor(0.0, device=self.embeddings.device)
        count = 0

        for _aa, indices in self._aa_to_indices.items():
            if len(indices) < 2:
                continue

            # Compute pairwise distances within synonymous group
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    d = poincare_distance(
                        self.embeddings[indices[i] : indices[i] + 1],
                        self.embeddings[indices[j] : indices[j] + 1],
                        c=self.curvature,
                    )
                    total_loss = total_loss + d
                    count += 1

        return total_loss / max(count, 1)

    @classmethod
    def from_vae_checkpoint(
        cls,
        checkpoint_path: str,
        embedding_dim: int = 16,
        curvature: float = 1.0,
        use_encoder: str = "B",
        device: str = "cpu",
    ) -> HyperbolicCodonEncoder:
        """Create encoder initialized from VAE checkpoint embeddings.

        Extracts the 64 codon embeddings by mapping codon indices through
        the trained VAE and storing the hyperbolic representations.

        Args:
            checkpoint_path: Path to VAE checkpoint
            embedding_dim: Target embedding dimension (may differ from VAE latent)
            curvature: Poincare ball curvature
            use_encoder: 'A' or 'B' to select which VAE encoder
            device: Device for loading checkpoint

        Returns:
            HyperbolicCodonEncoder initialized with VAE embeddings
        """
        import sys
        from pathlib import Path

        # Add project root if needed
        project_root = Path(__file__).parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Determine model architecture from config
        config = ckpt.get("config", {})
        latent_dim = config.get("latent_dim", 16)
        hidden_dim = config.get("hidden_dim", 64)

        # Import model class
        from src.models.ternary_vae import TernaryVAEV5_11_PartialFreeze

        # Create model
        model = TernaryVAEV5_11_PartialFreeze(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            max_radius=0.99,
            curvature=curvature,
            use_controller=config.get("use_controller", False),
            use_dual_projection=config.get("use_dual_projection", True),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        model.to(device)

        # Create encoder instance
        encoder = cls(
            embedding_dim=latent_dim,  # Use VAE latent dim
            curvature=curvature,
            use_padic_init=False,  # Will override with VAE embeddings
            learnable=False,  # Start frozen
        )

        # Extract embeddings for all 64 codons
        with torch.no_grad():
            all_embeddings = []

            for codon_idx in range(64):
                # Get ternary operation index for this codon
                # Use the codon index directly mapped to ternary space
                # This is a simplified mapping - in practice, might need codon->op mapping
                op_tensor = torch.tensor([[codon_idx]], device=device)

                # Forward through VAE (just encoding, no decoding)
                out = model(op_tensor, compute_control=False)

                # Get hyperbolic embedding from chosen encoder
                if use_encoder == "B":
                    z_hyp = out["z_B_hyp"][0]  # Shape: (latent_dim,)
                else:
                    z_hyp = out["z_A_hyp"][0]

                all_embeddings.append(z_hyp)

            embeddings = torch.stack(all_embeddings)

        # Override encoder embeddings
        encoder.embeddings = embeddings.to(device)
        encoder.embedding_dim = latent_dim

        return encoder


# Convenience function for extraction
def extract_codon_embeddings_from_vae(
    checkpoint_path: str,
    use_encoder: str = "B",
    device: str = "cpu",
) -> dict:
    """Extract all codon and amino acid embeddings from a VAE checkpoint.

    Args:
        checkpoint_path: Path to VAE checkpoint
        use_encoder: 'A' or 'B' encoder
        device: Device for computation

    Returns:
        Dictionary with:
        - 'codon_embeddings': (64, latent_dim) tensor
        - 'codon_radii': (64,) tensor of hyperbolic radii
        - 'aa_embeddings': dict of AA -> (latent_dim,) tensor
        - 'aa_radii': dict of AA -> float
        - 'metadata': dict with config info
    """
    encoder = HyperbolicCodonEncoder.from_vae_checkpoint(checkpoint_path, use_encoder=use_encoder, device=device)

    # Get AA embeddings
    aa_embeddings = encoder.get_all_amino_acid_embeddings(method="frechet")

    # Compute radii
    codon_radii = encoder.get_hyperbolic_radius()
    aa_radii = {}
    origin = torch.zeros(1, encoder.embedding_dim, device=encoder.embeddings.device)
    for aa, emb in aa_embeddings.items():
        aa_radii[aa] = poincare_distance(emb.unsqueeze(0), origin, c=encoder.curvature).item()

    return {
        "codon_embeddings": encoder.embeddings,
        "codon_radii": codon_radii,
        "aa_embeddings": aa_embeddings,
        "aa_radii": aa_radii,
        "metadata": {
            "embedding_dim": encoder.embedding_dim,
            "curvature": encoder.curvature,
            "encoder_used": use_encoder,
        },
    }
