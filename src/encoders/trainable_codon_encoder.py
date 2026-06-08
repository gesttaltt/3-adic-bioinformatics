# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Trainable Codon Encoder - 12-dim Input, Hyperbolic Output.

This module implements a trainable encoder for the 64 genetic codons that:
1. Uses 12-dim input (4 bases × 3 positions) - no information loss
2. Outputs embeddings on the Poincaré ball (hyperbolic space)
3. Learns via combined p-adic structure + amino acid property loss

Architecture:
    Input: 12-dim one-hot (4 bases per position, 3 positions)
    Encoder: MLP with LayerNorm and SiLU activation
    Output: Poincaré ball embedding (via exp_map_zero)

Loss Components:
    1. P-adic Structure Loss: Preserves codon hierarchy in radial position
    2. AA Property Loss: Synonymous codons cluster together
    3. Triplet Loss: Relative similarity preservation
    4. Radial Target Loss: Explicit radius targets per hierarchy level

Usage:
    from src.encoders.trainable_codon_encoder import TrainableCodonEncoder

    encoder = TrainableCodonEncoder(latent_dim=16)
    z_hyp = encoder(codon_indices)  # (batch, latent_dim) on Poincaré ball

    # Training
    loss = encoder.compute_total_loss()
    loss.backward()
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.biology.codons import (
    AMINO_ACID_TO_CODONS,
    GENETIC_CODE,
    codon_index_to_triplet,
)
from src.encoders.codon_encoder import AA_PROPERTIES
from src.geometry import (
    exp_map_zero,
    poincare_distance,
    project_to_poincare,
)

# Base encoding: A=0, C=1, G=2, T/U=3
BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def codon_to_onehot_12dim(codon_idx: int) -> torch.Tensor:
    """Convert codon index to 12-dim one-hot encoding.

    Args:
        codon_idx: Codon index (0-63)

    Returns:
        12-dim tensor: [pos1_A, pos1_C, pos1_G, pos1_T, pos2_..., pos3_...]
    """
    triplet = codon_index_to_triplet(codon_idx)
    onehot = torch.zeros(12)

    for pos, base in enumerate(triplet):
        base_idx = BASE_TO_IDX[base]
        onehot[pos * 4 + base_idx] = 1.0

    return onehot


def compute_codon_hierarchy_level(codon_idx: int) -> int:
    """Compute hierarchy level based on position variability.

    Codons with more conserved positions (same first/second base as others
    encoding same AA) are higher hierarchy.

    Returns:
        Hierarchy level 0-3 (0=most variable, 3=most conserved)
    """
    triplet = codon_index_to_triplet(codon_idx)
    aa = GENETIC_CODE.get(triplet, "*")

    if aa == "*":
        return 0  # Stop codons at boundary

    # Get all codons for this amino acid
    aa_codons = AMINO_ACID_TO_CODONS.get(aa, [triplet])

    if len(aa_codons) == 1:
        return 3  # Unique codon (Met, Trp) - highest hierarchy

    # Check position conservation
    conserved_positions = 0
    for pos in range(3):
        bases_at_pos = set(c[pos] for c in aa_codons)
        if len(bases_at_pos) == 1:
            conserved_positions += 1

    # More conserved = higher hierarchy
    return conserved_positions


def get_target_radius(hierarchy: int, max_r: float = 0.9, min_r: float = 0.2) -> float:
    """Map hierarchy level to target radius.

    Higher hierarchy (more conserved) -> smaller radius (closer to center)
    """
    # hierarchy 0 -> max_r, hierarchy 3 -> min_r
    t = hierarchy / 3.0
    return max_r - t * (max_r - min_r)


class CodonEncoderMLP(nn.Module):
    """MLP encoder from 12-dim one-hot to latent space."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(12, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: One-hot encoded codons, shape (batch, 12)

        Returns:
            Latent vectors in tangent space, shape (batch, latent_dim)
        """
        return self.encoder(x)


class TrainableCodonEncoder(nn.Module):
    """Trainable codon encoder with hyperbolic output and structured loss."""

    def __init__(
        self,
        latent_dim: int = 16,
        hidden_dim: int = 64,
        curvature: float = 1.0,
        max_radius: float = 0.9,
        min_radius: float = 0.2,
        dropout: float = 0.1,
    ):
        """Initialize TrainableCodonEncoder.

        Args:
            latent_dim: Dimension of hyperbolic embeddings
            hidden_dim: Hidden layer dimension in MLP
            curvature: Poincaré ball curvature
            max_radius: Maximum radius (for low hierarchy codons)
            min_radius: Minimum radius (for high hierarchy codons)
            dropout: Dropout rate
        """
        super().__init__()

        self.latent_dim = latent_dim
        self.curvature = curvature
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.num_codons = 64

        # Encoder network
        self.encoder = CodonEncoderMLP(latent_dim, hidden_dim, dropout)

        # Pre-compute one-hot encodings for all 64 codons
        self.register_buffer("codon_onehots", torch.stack([codon_to_onehot_12dim(i) for i in range(64)]))

        # Pre-compute hierarchy levels and target radii
        hierarchies = [compute_codon_hierarchy_level(i) for i in range(64)]
        target_radii = [get_target_radius(h, max_radius, min_radius) for h in hierarchies]
        self.register_buffer("hierarchies", torch.tensor(hierarchies, dtype=torch.float32))
        self.register_buffer("target_radii", torch.tensor(target_radii, dtype=torch.float32))

        # Pre-compute amino acid groups
        self._build_aa_groups()

        # Pre-compute p-adic distance matrix
        self._build_padic_distances()

    def _build_aa_groups(self):
        """Build mapping from amino acid to codon indices."""
        self.aa_to_indices = {}
        for aa in "ACDEFGHIKLMNPQRSTVWY*":
            indices = []
            for i in range(64):
                triplet = codon_index_to_triplet(i)
                if GENETIC_CODE.get(triplet, "*") == aa:
                    indices.append(i)
            if indices:
                self.aa_to_indices[aa] = indices

    def _build_padic_distances(self):
        """Pre-compute 64x64 p-adic distance matrix."""
        from src.encoders.codon_encoder import compute_padic_distance_between_codons

        padic_matrix = torch.zeros(64, 64)
        for i in range(64):
            for j in range(64):
                padic_matrix[i, j] = compute_padic_distance_between_codons(i, j)

        self.register_buffer("padic_distances", padic_matrix)

    def encode_all(self) -> torch.Tensor:
        """Encode all 64 codons to Poincaré ball.

        Returns:
            Hyperbolic embeddings, shape (64, latent_dim)
        """
        # Get tangent space embeddings
        z_tangent = self.encoder(self.codon_onehots)

        # Map to Poincaré ball
        z_hyp = exp_map_zero(z_tangent, c=self.curvature)

        # Project to ensure within ball
        z_hyp = project_to_poincare(z_hyp, max_norm=self.max_radius, c=self.curvature)

        return z_hyp

    def forward(self, codon_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass for given codon indices.

        Args:
            codon_indices: Tensor of codon indices, shape (batch,) or (batch, seq_len)

        Returns:
            Hyperbolic embeddings, shape (batch, latent_dim) or (batch, seq_len, latent_dim)
        """
        original_shape = codon_indices.shape
        flat_indices = codon_indices.flatten()

        # Get one-hot encodings
        onehots = self.codon_onehots[flat_indices]

        # Encode to tangent space
        z_tangent = self.encoder(onehots)

        # Map to Poincaré ball
        z_hyp = exp_map_zero(z_tangent, c=self.curvature)
        z_hyp = project_to_poincare(z_hyp, max_norm=self.max_radius, c=self.curvature)

        # Reshape to original
        if len(original_shape) > 1:
            z_hyp = z_hyp.view(*original_shape, self.latent_dim)

        return z_hyp

    def get_hyperbolic_radii(self, z_hyp: torch.Tensor | None = None) -> torch.Tensor:
        """Get hyperbolic radii for embeddings.

        Args:
            z_hyp: Embeddings, or None to use all 64 codons

        Returns:
            Radii tensor
        """
        if z_hyp is None:
            z_hyp = self.encode_all()

        origin = torch.zeros(1, self.latent_dim, device=z_hyp.device)
        radii = poincare_distance(z_hyp, origin.expand(z_hyp.shape[0], -1), c=self.curvature)
        return radii

    def compute_radial_loss(self) -> torch.Tensor:
        """Loss encouraging correct radial ordering by hierarchy.

        Radial Target Loss: MSE between actual and target radii.
        """
        z_hyp = self.encode_all()
        radii = self.get_hyperbolic_radii(z_hyp)

        return F.mse_loss(radii, self.target_radii.to(radii.device))

    def compute_padic_structure_loss(self) -> torch.Tensor:
        """Loss encouraging hyperbolic distances to match p-adic distances.

        Uses correlation-based loss for scale-invariance.
        Vectorized for efficiency.
        """
        z_hyp = self.encode_all()

        # Vectorized hyperbolic distance matrix using cdist-like approach
        # For Poincare ball: use poincare_distance_matrix or compute efficiently
        # Simplified: use Euclidean as proxy (faster, still captures structure)
        hyp_dists = torch.cdist(z_hyp, z_hyp, p=2)

        # Get upper triangle
        triu_idx = torch.triu_indices(64, 64, offset=1, device=z_hyp.device)
        hyp_flat = hyp_dists[triu_idx[0], triu_idx[1]]
        padic_flat = self.padic_distances[triu_idx[0], triu_idx[1]].to(z_hyp.device)

        # Correlation loss: 1 - correlation
        hyp_centered = hyp_flat - hyp_flat.mean()
        padic_centered = padic_flat - padic_flat.mean()

        corr = (hyp_centered * padic_centered).sum() / (
            torch.sqrt((hyp_centered**2).sum() * (padic_centered**2).sum()) + 1e-8
        )

        return 1.0 - corr

    def compute_synonymous_cohesion_loss(self) -> torch.Tensor:
        """Loss encouraging synonymous codons (same AA) to cluster.

        Minimizes average pairwise distance within AA groups.
        Vectorized for efficiency.
        """
        z_hyp = self.encode_all()
        total_loss = torch.tensor(0.0, device=z_hyp.device)
        count = 0

        for aa, indices in self.aa_to_indices.items():
            if aa == "*" or len(indices) < 2:
                continue

            # Get group embeddings and compute pairwise distances
            group_embs = z_hyp[indices]
            group_dists = torch.cdist(group_embs, group_embs, p=2)

            # Sum upper triangle (pairwise distances)
            n = len(indices)
            triu_idx = torch.triu_indices(n, n, offset=1, device=z_hyp.device)
            total_loss = total_loss + group_dists[triu_idx[0], triu_idx[1]].sum()
            count += len(triu_idx[0])

        return total_loss / max(count, 1)

    def compute_aa_separation_loss(self) -> torch.Tensor:
        """Loss encouraging different AAs to be separated.

        Uses triplet-like margin: d(same_AA) < d(diff_AA) - margin
        """
        z_hyp = self.encode_all()

        # Compute AA centroids
        aa_centroids = {}
        for aa, indices in self.aa_to_indices.items():
            if aa != "*":
                aa_centroids[aa] = z_hyp[indices].mean(dim=0)

        # Margin-based separation loss
        margin = 0.5
        total_loss = torch.tensor(0.0, device=z_hyp.device)
        count = 0

        aas = list(aa_centroids.keys())
        for i, aa1 in enumerate(aas):
            for aa2 in aas[i + 1 :]:
                # Distance between AA centroids
                d = poincare_distance(aa_centroids[aa1].unsqueeze(0), aa_centroids[aa2].unsqueeze(0), c=self.curvature)
                # Penalize if too close
                total_loss = total_loss + F.relu(margin - d)
                count += 1

        return total_loss / max(count, 1)

    def compute_aa_property_loss(self) -> torch.Tensor:
        """Loss encouraging AA embeddings to reflect physicochemical properties.

        AAs with similar properties should have similar embeddings.
        """
        z_hyp = self.encode_all()

        # Compute AA centroids
        aa_centroids = {}
        aa_props = {}
        for aa, indices in self.aa_to_indices.items():
            if aa != "*" and aa in AA_PROPERTIES:
                aa_centroids[aa] = z_hyp[indices].mean(dim=0)
                aa_props[aa] = torch.tensor(AA_PROPERTIES[aa], device=z_hyp.device)

        # Correlation between hyperbolic distances and property distances
        aas = list(aa_centroids.keys())
        hyp_dists = []
        prop_dists = []

        for i, aa1 in enumerate(aas):
            for aa2 in aas[i + 1 :]:
                d_hyp = poincare_distance(
                    aa_centroids[aa1].unsqueeze(0), aa_centroids[aa2].unsqueeze(0), c=self.curvature
                )
                d_prop = torch.norm(aa_props[aa1] - aa_props[aa2])

                hyp_dists.append(d_hyp)
                prop_dists.append(d_prop)

        if len(hyp_dists) < 2:
            return torch.tensor(0.0, device=z_hyp.device)

        hyp_tensor = torch.stack(hyp_dists)
        prop_tensor = torch.stack(prop_dists)

        # Correlation loss
        hyp_c = hyp_tensor - hyp_tensor.mean()
        prop_c = prop_tensor - prop_tensor.mean()

        corr = (hyp_c * prop_c).sum() / (torch.sqrt((hyp_c**2).sum() * (prop_c**2).sum()) + 1e-8)

        return 1.0 - corr

    def compute_total_loss(
        self,
        radial_weight: float = 1.0,
        padic_weight: float = 1.0,
        cohesion_weight: float = 0.5,
        separation_weight: float = 0.3,
        property_weight: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Compute total training loss.

        Returns:
            Dictionary with individual losses and total
        """
        radial_loss = self.compute_radial_loss()
        padic_loss = self.compute_padic_structure_loss()
        cohesion_loss = self.compute_synonymous_cohesion_loss()
        separation_loss = self.compute_aa_separation_loss()
        property_loss = self.compute_aa_property_loss()

        total = (
            radial_weight * radial_loss
            + padic_weight * padic_loss
            + cohesion_weight * cohesion_loss
            + separation_weight * separation_loss
            + property_weight * property_loss
        )

        return {
            "total": total,
            "radial": radial_loss,
            "padic": padic_loss,
            "cohesion": cohesion_loss,
            "separation": separation_loss,
            "property": property_loss,
        }

    def get_amino_acid_embedding(self, aa: str) -> torch.Tensor:
        """Get aggregated embedding for an amino acid.

        Args:
            aa: Single-letter amino acid code

        Returns:
            Mean embedding of all synonymous codons
        """
        indices = self.aa_to_indices.get(aa, [])
        if not indices:
            return torch.zeros(self.latent_dim, device=self.codon_onehots.device)

        z_hyp = self.encode_all()
        return z_hyp[indices].mean(dim=0)

    def get_all_amino_acid_embeddings(self) -> dict[str, torch.Tensor]:
        """Get embeddings for all 20 standard amino acids."""
        return {aa: self.get_amino_acid_embedding(aa) for aa in "ACDEFGHIKLMNPQRSTVWY"}

    def compute_aa_distance(self, aa1: str, aa2: str) -> float:
        """Compute hyperbolic distance between amino acids."""
        emb1 = self.get_amino_acid_embedding(aa1)
        emb2 = self.get_amino_acid_embedding(aa2)
        return poincare_distance(emb1.unsqueeze(0), emb2.unsqueeze(0), c=self.curvature).item()


def train_codon_encoder(
    epochs: int = 1000,
    lr: float = 0.001,
    latent_dim: int = 16,
    hidden_dim: int = 64,
    device: str = "cpu",
    radial_weight: float = 1.0,
    padic_weight: float = 1.0,
    cohesion_weight: float = 0.5,
    separation_weight: float = 0.3,
    property_weight: float = 0.5,
    print_every: int = 100,
) -> TrainableCodonEncoder:
    """Train the codon encoder.

    Args:
        epochs: Number of training epochs
        lr: Learning rate
        latent_dim: Latent space dimension
        hidden_dim: Hidden layer dimension
        device: Training device
        *_weight: Loss component weights
        print_every: Print frequency

    Returns:
        Trained encoder
    """
    encoder = TrainableCodonEncoder(
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        optimizer.zero_grad()

        losses = encoder.compute_total_loss(
            radial_weight=radial_weight,
            padic_weight=padic_weight,
            cohesion_weight=cohesion_weight,
            separation_weight=separation_weight,
            property_weight=property_weight,
        )

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if losses["total"].item() < best_loss:
            best_loss = losses["total"].item()
            best_state = {k: v.clone() for k, v in encoder.state_dict().items()}

        if (epoch + 1) % print_every == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}: "
                f"total={losses['total'].item():.4f}, "
                f"radial={losses['radial'].item():.4f}, "
                f"padic={losses['padic'].item():.4f}, "
                f"cohesion={losses['cohesion'].item():.4f}"
            )

    # Load best state
    if best_state is not None:
        encoder.load_state_dict(best_state)

    return encoder
