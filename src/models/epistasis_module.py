# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Epistasis Module for mutation interaction modeling.

Epistasis refers to the phenomenon where the phenotypic effect of one mutation
depends on the presence of other mutations. This module learns these
interaction patterns from sequence data.

Key concepts:
- Pairwise interactions: How two mutations combine
- Higher-order interactions: Three or more mutations
- Sign epistasis: When combined effect differs from sum of individual effects
- Magnitude epistasis: Combined effect is different magnitude than expected

This integrates with the existing coevolution_loss.py for evolutionary
constraint modeling, adding learned interaction patterns.

Usage:
    from src.models.epistasis_module import EpistasisModule

    epistasis = EpistasisModule(n_positions=300, embed_dim=64)

    # Get interaction score for mutation combination
    positions = torch.tensor([65, 184, 215])  # Mutation positions
    score = epistasis(positions)

    # Visualize learned interactions
    matrix = epistasis.get_epistasis_matrix()
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EpistasisResult:
    """Container for epistasis analysis results.

    Attributes:
        interaction_score: Overall interaction score
        pairwise_scores: Pairwise interaction matrix
        higher_order_score: Higher-order interaction contribution
        sign_epistasis: Whether sign epistasis detected
        antagonistic: Whether interactions are antagonistic
        synergistic: Whether interactions are synergistic
    """

    interaction_score: torch.Tensor
    pairwise_scores: torch.Tensor | None = None
    higher_order_score: torch.Tensor | None = None
    sign_epistasis: torch.Tensor | None = None
    antagonistic: torch.Tensor | None = None
    synergistic: torch.Tensor | None = None


class PairwiseInteractionModule(nn.Module):
    """Learn pairwise mutation interactions.

    Uses a factorized embedding approach where each position has
    an embedding, and pairwise interactions are computed as
    embedding similarities/products.
    """

    def __init__(
        self,
        n_positions: int,
        embed_dim: int = 64,
        n_amino_acids: int = 21,
        use_position_embedding: bool = True,
        use_aa_embedding: bool = True,
    ):
        """Initialize pairwise interaction module.

        Args:
            n_positions: Maximum sequence length / number of positions
            embed_dim: Embedding dimension
            n_amino_acids: Number of amino acids (20 + gap)
            use_position_embedding: Whether to use position embeddings
            use_aa_embedding: Whether to use amino acid embeddings
        """
        super().__init__()

        self.n_positions = n_positions
        self.embed_dim = embed_dim
        self.n_amino_acids = n_amino_acids

        # Position embeddings
        if use_position_embedding:
            self.position_embedding = nn.Embedding(n_positions, embed_dim)
        else:
            self.position_embedding = None

        # Amino acid embeddings (for mutation-specific interactions)
        if use_aa_embedding:
            self.aa_embedding = nn.Embedding(n_amino_acids, embed_dim)
        else:
            self.aa_embedding = None

        # Interaction network
        self.interaction_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

        # Learnable interaction bias for known important pairs
        self.register_buffer("known_pairs_mask", torch.zeros(n_positions, n_positions))

    def set_known_epistatic_pairs(self, pairs: list[tuple[int, int]]):
        """Set known epistatic position pairs for biasing.

        Args:
            pairs: List of (pos1, pos2) tuples for known epistatic pairs
        """
        mask = torch.zeros(self.n_positions, self.n_positions)
        for p1, p2 in pairs:
            if p1 < self.n_positions and p2 < self.n_positions:
                mask[p1, p2] = 1.0
                mask[p2, p1] = 1.0
        self.known_pairs_mask = mask.to(self.known_pairs_mask.device)

    def forward(
        self,
        positions: torch.Tensor,
        amino_acids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute pairwise interaction scores.

        Args:
            positions: Mutation positions (batch, n_mutations)
            amino_acids: Optional amino acid indices (batch, n_mutations)

        Returns:
            Pairwise interaction scores (batch, n_mutations, n_mutations)
        """
        batch_size, n_muts = positions.shape
        device = positions.device

        # Get position embeddings
        if self.position_embedding is not None:
            pos_emb = self.position_embedding(positions.clamp(0, self.n_positions - 1))
        else:
            pos_emb = torch.zeros(batch_size, n_muts, self.embed_dim, device=device)

        # Add amino acid embeddings if available
        if self.aa_embedding is not None and amino_acids is not None:
            aa_emb = self.aa_embedding(amino_acids.clamp(0, self.n_amino_acids - 1))
            combined_emb = pos_emb + aa_emb
        else:
            combined_emb = pos_emb

        # Compute all pairwise interactions
        pairwise_scores = torch.zeros(batch_size, n_muts, n_muts, device=device)

        for i in range(n_muts):
            for j in range(i + 1, n_muts):
                # Concatenate embeddings for pair
                pair_emb = torch.cat([combined_emb[:, i], combined_emb[:, j]], dim=-1)
                score = self.interaction_net(pair_emb).squeeze(-1)
                pairwise_scores[:, i, j] = score
                pairwise_scores[:, j, i] = score

        return pairwise_scores


class HigherOrderInteractionModule(nn.Module):
    """Learn higher-order (3+ way) mutation interactions.

    Uses a transformer-based approach to capture complex dependencies
    among multiple mutations simultaneously.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_mutations: int = 20,
    ):
        """Initialize higher-order interaction module.

        Args:
            embed_dim: Embedding dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            max_mutations: Maximum number of mutations to process
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.max_mutations = max_mutations

        # Transformer for mutation set
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Aggregation and scoring
        self.aggregate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1),
        )

        # Order-specific components
        self.order_embedding = nn.Embedding(max_mutations + 1, embed_dim)

    def forward(
        self,
        mutation_embeddings: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute higher-order interaction score.

        Args:
            mutation_embeddings: Embeddings of mutated positions (batch, n_muts, dim)
            mask: Optional mask for padding (batch, n_muts)

        Returns:
            Higher-order interaction score (batch,)
        """
        batch_size, n_muts, _ = mutation_embeddings.shape

        # Add order embedding
        order_idx = torch.tensor([n_muts], device=mutation_embeddings.device)
        order_emb = self.order_embedding(order_idx.clamp(0, self.max_mutations))
        mutation_embeddings = mutation_embeddings + order_emb.unsqueeze(1)

        # Create attention mask if needed
        if mask is not None:
            attn_mask = ~mask.bool()  # True = masked
        else:
            attn_mask = None

        # Apply transformer
        transformed = self.transformer(
            mutation_embeddings,
            src_key_padding_mask=attn_mask,
        )

        # Aggregate (mean pooling)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (transformed * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = transformed.mean(dim=1)

        # Score
        score = self.aggregate(pooled).squeeze(-1)

        return score


class EpistasisModule(nn.Module):
    """Complete epistasis module combining pairwise and higher-order interactions.

    This module learns mutation interaction patterns from sequence data,
    combining:
    1. Pairwise position/mutation interactions
    2. Higher-order (3+) interactions via attention
    3. Sign and magnitude epistasis detection

    The learned interactions can be used to:
    - Predict fitness/resistance effects of mutation combinations
    - Identify key epistatic pairs for mechanistic studies
    - Improve drug resistance predictions for multi-mutant sequences
    """

    def __init__(
        self,
        n_positions: int,
        embed_dim: int = 64,
        n_amino_acids: int = 21,
        n_heads: int = 4,
        n_layers: int = 2,
        use_higher_order: bool = True,
        temperature: float = 1.0,
    ):
        """Initialize epistasis module.

        Args:
            n_positions: Maximum sequence length
            embed_dim: Embedding dimension
            n_amino_acids: Number of amino acids
            n_heads: Attention heads for higher-order module
            n_layers: Transformer layers for higher-order module
            use_higher_order: Whether to use higher-order interactions
            temperature: Temperature for interaction scoring
        """
        super().__init__()

        self.n_positions = n_positions
        self.embed_dim = embed_dim
        self.use_higher_order = use_higher_order
        self.temperature = temperature

        # Position and AA embeddings
        self.position_embedding = nn.Embedding(n_positions, embed_dim)
        self.aa_embedding = nn.Embedding(n_amino_acids, embed_dim)

        # Pairwise interactions
        self.pairwise = PairwiseInteractionModule(
            n_positions=n_positions,
            embed_dim=embed_dim,
            n_amino_acids=n_amino_acids,
        )

        # Higher-order interactions
        if use_higher_order:
            self.higher_order = HigherOrderInteractionModule(
                embed_dim=embed_dim,
                n_heads=n_heads,
                n_layers=n_layers,
            )
        else:
            self.higher_order = None

        # Sign epistasis detection
        self.sign_detector = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.GELU(),
            nn.Linear(32, 2),  # [antagonistic, synergistic]
        )

        # Combination weights
        self.combination_weights = nn.Parameter(torch.tensor([0.6, 0.4]))

    def forward(
        self,
        positions: torch.Tensor,
        amino_acids: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> EpistasisResult:
        """Compute epistasis scores for mutation combinations.

        Args:
            positions: Mutation positions (batch, n_mutations)
            amino_acids: Optional mutant amino acid indices (batch, n_mutations)
            return_details: Whether to return detailed breakdown

        Returns:
            EpistasisResult with interaction scores
        """
        batch_size, n_muts = positions.shape
        device = positions.device

        # Get mutation embeddings
        pos_emb = self.position_embedding(positions.clamp(0, self.n_positions - 1))
        if amino_acids is not None:
            aa_emb = self.aa_embedding(amino_acids.clamp(0, self.aa_embedding.num_embeddings - 1))
            mut_emb = pos_emb + aa_emb
        else:
            mut_emb = pos_emb

        # Pairwise interactions
        pairwise_scores = self.pairwise(positions, amino_acids)

        # Aggregate pairwise (upper triangle sum)
        mask = torch.triu(torch.ones(n_muts, n_muts, device=device), diagonal=1)
        pairwise_agg = (pairwise_scores * mask).sum(dim=(1, 2))

        # Normalize by number of pairs
        n_pairs = n_muts * (n_muts - 1) / 2
        if n_pairs > 0:
            pairwise_agg = pairwise_agg / n_pairs

        # Higher-order interactions
        if self.use_higher_order and n_muts >= 3:
            higher_order_score = self.higher_order(mut_emb)
        else:
            higher_order_score = torch.zeros(batch_size, device=device)

        # Combine scores
        weights = F.softmax(self.combination_weights, dim=0)
        interaction_score = weights[0] * pairwise_agg + weights[1] * higher_order_score

        # Apply temperature
        interaction_score = interaction_score / self.temperature

        # Detect sign epistasis
        mean_emb = mut_emb.mean(dim=1)
        sign_logits = self.sign_detector(mean_emb)
        sign_probs = F.softmax(sign_logits, dim=-1)

        result = EpistasisResult(
            interaction_score=interaction_score,
            antagonistic=sign_probs[:, 0],
            synergistic=sign_probs[:, 1],
        )

        if return_details:
            result.pairwise_scores = pairwise_scores
            result.higher_order_score = higher_order_score

        return result

    def get_epistasis_matrix(
        self,
        amino_acid: int | None = None,
    ) -> torch.Tensor:
        """Get learned pairwise epistasis matrix.

        Useful for visualization and identifying key epistatic pairs.

        Args:
            amino_acid: Optional specific amino acid index

        Returns:
            Epistasis matrix (n_positions, n_positions)
        """
        device = self.position_embedding.weight.device

        # Create all position pairs
        torch.arange(self.n_positions, device=device).unsqueeze(0)

        # Get position embeddings
        pos_emb = self.position_embedding.weight

        # Compute pairwise scores using dot product similarity
        # as proxy for full interaction network
        matrix = torch.mm(pos_emb, pos_emb.t())

        # Normalize
        matrix = matrix / (torch.norm(pos_emb, dim=1, keepdim=True) + 1e-8)
        matrix = matrix / (torch.norm(pos_emb, dim=1, keepdim=True).t() + 1e-8)

        return matrix

    def get_top_epistatic_pairs(
        self,
        k: int = 20,
        min_distance: int = 5,
    ) -> list[tuple[int, int, float]]:
        """Get top-k epistatic position pairs.

        Args:
            k: Number of pairs to return
            min_distance: Minimum sequence distance between positions

        Returns:
            List of (pos1, pos2, score) tuples
        """
        matrix = self.get_epistasis_matrix()

        # Zero out diagonal and nearby positions
        for i in range(matrix.shape[0]):
            for j in range(max(0, i - min_distance), min(matrix.shape[1], i + min_distance + 1)):
                matrix[i, j] = float("-inf")

        # Get top-k
        values, indices = torch.topk(matrix.view(-1), k)
        pairs = []
        for val, idx in zip(values, indices, strict=False):
            pos1 = idx // matrix.shape[1]
            pos2 = idx % matrix.shape[1]
            if pos1 < pos2:  # Avoid duplicates
                pairs.append((int(pos1), int(pos2), float(val)))

        return pairs


class EpistasisPredictor(nn.Module):
    """Predict fitness/resistance effects including epistasis.

    Combines individual mutation effects with epistatic interactions
    to predict phenotypic outcomes.
    """

    def __init__(
        self,
        n_positions: int,
        n_amino_acids: int = 21,
        embed_dim: int = 64,
        n_outputs: int = 1,
    ):
        """Initialize predictor.

        Args:
            n_positions: Sequence length
            n_amino_acids: Number of amino acids
            embed_dim: Embedding dimension
            n_outputs: Number of output predictions (e.g., drugs)
        """
        super().__init__()

        self.epistasis = EpistasisModule(
            n_positions=n_positions,
            n_amino_acids=n_amino_acids,
            embed_dim=embed_dim,
        )

        # Individual mutation effects
        self.individual_effects = nn.Parameter(torch.zeros(n_positions, n_amino_acids, n_outputs))

        # Combination head
        self.output_head = nn.Sequential(
            nn.Linear(1 + n_outputs, 32),  # epistasis score + individual
            nn.GELU(),
            nn.Linear(32, n_outputs),
        )

    def forward(
        self,
        positions: torch.Tensor,
        amino_acids: torch.Tensor,
        wild_type_aa: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict effect of mutation combination.

        Args:
            positions: Mutation positions (batch, n_muts)
            amino_acids: Mutant amino acids (batch, n_muts)
            wild_type_aa: Optional wild-type amino acids (batch, n_muts)

        Returns:
            Dictionary with predictions and components
        """
        batch_size, n_muts = positions.shape

        # Get epistasis
        epistasis_result = self.epistasis(positions, amino_acids, return_details=True)

        # Get individual effects
        individual = torch.zeros(batch_size, self.individual_effects.shape[-1], device=positions.device)
        for b in range(batch_size):
            for m in range(n_muts):
                pos = positions[b, m].clamp(0, self.individual_effects.shape[0] - 1)
                aa = amino_acids[b, m].clamp(0, self.individual_effects.shape[1] - 1)
                individual[b] += self.individual_effects[pos, aa]

        # Combine
        combined = torch.cat([epistasis_result.interaction_score.unsqueeze(-1), individual], dim=-1)
        predictions = self.output_head(combined)

        return {
            "predictions": predictions,
            "epistasis_score": epistasis_result.interaction_score,
            "individual_effects": individual,
            "pairwise_scores": epistasis_result.pairwise_scores,
            "antagonistic": epistasis_result.antagonistic,
            "synergistic": epistasis_result.synergistic,
        }


__all__ = [
    "EpistasisModule",
    "EpistasisResult",
    "EpistasisPredictor",
    "PairwiseInteractionModule",
    "HigherOrderInteractionModule",
]
