# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Set-theoretic feature extraction for MTL resistance prediction.

Provides methods to extract set-based features from mutation data
for integration with neural multi-task learning.

Features include:
- Jaccard/Dice similarity to known resistance patterns
- Cross-resistance scores based on set overlap
- Lattice-based hierarchy features
- Minimal mutation set membership indicators
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.analysis.set_theory.lattice import ResistanceLattice, ResistanceLevel
from src.analysis.set_theory.mutation_sets import (
    MutationSet,
)
from src.analysis.set_theory.rough_sets import RoughClassifier


@dataclass
class SetFeatureConfig:
    """Configuration for set-theoretic feature extraction.

    Attributes:
        n_drugs: Number of drugs
        drug_names: Names of drugs
        use_jaccard: Include Jaccard similarity features
        use_dice: Include Dice similarity features
        use_cross_resistance: Include cross-resistance features
        use_lattice: Include lattice hierarchy features
        use_rough: Include rough set membership features
        n_reference_patterns: Number of reference patterns per drug
    """

    n_drugs: int = 18
    drug_names: list[str] | None = None
    use_jaccard: bool = True
    use_dice: bool = True
    use_cross_resistance: bool = True
    use_lattice: bool = True
    use_rough: bool = True
    n_reference_patterns: int = 10


class SetFeatureExtractor(nn.Module):
    """Extract set-theoretic features from mutation data.

    Converts mutation sets into fixed-dimensional feature vectors
    suitable for neural network input.

    Example:
        >>> extractor = SetFeatureExtractor(config)
        >>> extractor.register_reference_patterns("RIF", [
        ...     MutationSet.from_strings(["rpoB_S450L"]),
        ...     MutationSet.from_strings(["rpoB_H445Y"]),
        ... ])
        >>> mutations = MutationSet.from_strings(["rpoB_S450L", "katG_S315T"])
        >>> features = extractor(mutations)
    """

    def __init__(
        self,
        config: SetFeatureConfig | None = None,
        rough_classifiers: dict[str, RoughClassifier] | None = None,
        lattice: ResistanceLattice | None = None,
    ):
        """Initialize feature extractor.

        Args:
            config: Feature extraction configuration
            rough_classifiers: Pre-trained rough classifiers per drug
            lattice: Resistance lattice for hierarchy features
        """
        super().__init__()
        self.config = config or SetFeatureConfig()
        self.rough_classifiers = rough_classifiers or {}
        self.lattice = lattice or ResistanceLattice()

        # Reference patterns for similarity computation
        self.reference_patterns: dict[str, list[MutationSet]] = {}

        # Compute feature dimension
        self._compute_feature_dim()

    def _compute_feature_dim(self):
        """Compute total feature dimension."""
        dim = 0
        n_drugs = self.config.n_drugs
        n_ref = self.config.n_reference_patterns

        if self.config.use_jaccard:
            dim += n_drugs * n_ref  # Jaccard to each reference

        if self.config.use_dice:
            dim += n_drugs * n_ref  # Dice to each reference

        if self.config.use_cross_resistance:
            dim += n_drugs * (n_drugs - 1) // 2  # Pairwise cross-resistance

        if self.config.use_lattice:
            dim += 6  # Resistance level one-hot + level value

        if self.config.use_rough:
            dim += n_drugs * 3  # lower/boundary/outside for each drug

        self.feature_dim = dim

    def register_reference_patterns(
        self,
        drug: str,
        patterns: list[MutationSet],
    ):
        """Register reference mutation patterns for a drug.

        Args:
            drug: Drug name
            patterns: Reference mutation sets (known resistance patterns)
        """
        self.reference_patterns[drug] = patterns[: self.config.n_reference_patterns]

    def forward(
        self,
        mutations: MutationSet,
    ) -> torch.Tensor:
        """Extract features from mutation set.

        Args:
            mutations: Mutation set

        Returns:
            Feature tensor of shape (feature_dim,)
        """
        features = []

        # Jaccard similarity features
        if self.config.use_jaccard:
            jaccard_feats = self._compute_jaccard_features(mutations)
            features.append(jaccard_feats)

        # Dice similarity features
        if self.config.use_dice:
            dice_feats = self._compute_dice_features(mutations)
            features.append(dice_feats)

        # Cross-resistance features
        if self.config.use_cross_resistance:
            cross_feats = self._compute_cross_resistance_features(mutations)
            features.append(cross_feats)

        # Lattice hierarchy features
        if self.config.use_lattice:
            lattice_feats = self._compute_lattice_features(mutations)
            features.append(lattice_feats)

        # Rough set membership features
        if self.config.use_rough:
            rough_feats = self._compute_rough_features(mutations)
            features.append(rough_feats)

        return torch.cat(features, dim=-1)

    def _compute_jaccard_features(
        self,
        mutations: MutationSet,
    ) -> torch.Tensor:
        """Compute Jaccard similarity to reference patterns.

        Args:
            mutations: Input mutation set

        Returns:
            Jaccard features tensor
        """
        features = []
        n_ref = self.config.n_reference_patterns
        drugs = self.config.drug_names or list(self.reference_patterns.keys())

        for drug in drugs:
            patterns = self.reference_patterns.get(drug, [])

            for i in range(n_ref):
                sim = mutations.jaccard_similarity(patterns[i]) if i < len(patterns) else 0.0
                features.append(sim)

        return torch.tensor(features, dtype=torch.float32)

    def _compute_dice_features(
        self,
        mutations: MutationSet,
    ) -> torch.Tensor:
        """Compute Dice similarity to reference patterns.

        Args:
            mutations: Input mutation set

        Returns:
            Dice features tensor
        """
        features = []
        n_ref = self.config.n_reference_patterns
        drugs = self.config.drug_names or list(self.reference_patterns.keys())

        for drug in drugs:
            patterns = self.reference_patterns.get(drug, [])

            for i in range(n_ref):
                sim = mutations.dice_similarity(patterns[i]) if i < len(patterns) else 0.0
                features.append(sim)

        return torch.tensor(features, dtype=torch.float32)

    def _compute_cross_resistance_features(
        self,
        mutations: MutationSet,
    ) -> torch.Tensor:
        """Compute cross-resistance indicator features.

        Args:
            mutations: Input mutation set

        Returns:
            Cross-resistance features
        """
        features = []
        drugs = self.config.drug_names or list(self.reference_patterns.keys())

        for i, drug1 in enumerate(drugs):
            for drug2 in drugs[i + 1 :]:
                # Check if mutations overlap with both drug patterns
                patterns1 = self.reference_patterns.get(drug1, [])
                patterns2 = self.reference_patterns.get(drug2, [])

                # Compute overlap score
                if patterns1 and patterns2:
                    # Union of patterns for each drug
                    all_p1 = MutationSet.empty()
                    for p in patterns1:
                        all_p1 = all_p1 | p

                    all_p2 = MutationSet.empty()
                    for p in patterns2:
                        all_p2 = all_p2 | p

                    # Check if input overlaps with both
                    overlap1 = len(mutations & all_p1) > 0
                    overlap2 = len(mutations & all_p2) > 0

                    score = 1.0 if (overlap1 and overlap2) else 0.0
                else:
                    score = 0.0

                features.append(score)

        return torch.tensor(features, dtype=torch.float32)

    def _compute_lattice_features(
        self,
        mutations: MutationSet,
    ) -> torch.Tensor:
        """Compute lattice hierarchy features.

        Args:
            mutations: Input mutation set

        Returns:
            Lattice features (one-hot level + normalized level)
        """
        level = self.lattice.resistance_level(mutations)

        # One-hot encoding of resistance level
        one_hot = torch.zeros(len(ResistanceLevel))
        one_hot[level.value] = 1.0

        # Normalized level value
        normalized = torch.tensor([level.value / (len(ResistanceLevel) - 1)])

        return torch.cat([one_hot, normalized])

    def _compute_rough_features(
        self,
        mutations: MutationSet,
    ) -> torch.Tensor:
        """Compute rough set membership features.

        Args:
            mutations: Input mutation set

        Returns:
            Rough features (lower/boundary/outside for each drug)
        """
        features = []
        drugs = self.config.drug_names or list(self.rough_classifiers.keys())

        for drug in drugs:
            if drug in self.rough_classifiers:
                classifier = self.rough_classifiers[drug]

                # Check membership for each mutation
                in_lower = 0
                in_boundary = 0
                outside = 0

                for mut in mutations:
                    if classifier.positive_mutations.definitely_in(mut):
                        in_lower += 1
                    elif classifier.positive_mutations.uncertain(mut):
                        in_boundary += 1
                    else:
                        outside += 1

                # Normalize by set size
                n = len(mutations) if len(mutations) > 0 else 1
                features.extend([in_lower / n, in_boundary / n, outside / n])
            else:
                features.extend([0.0, 0.0, 1.0])  # All outside if no classifier

        return torch.tensor(features, dtype=torch.float32)

    def extract_batch(
        self,
        mutation_sets: list[MutationSet],
    ) -> torch.Tensor:
        """Extract features for a batch of mutation sets.

        Args:
            mutation_sets: List of mutation sets

        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        features = [self(ms) for ms in mutation_sets]
        return torch.stack(features)


class SetEnhancedMTL(nn.Module):
    """Multi-task predictor enhanced with set-theoretic features.

    Combines neural embeddings with set-based features for
    improved resistance prediction.

    Example:
        >>> # Create base MTL model and set feature extractor
        >>> mtl_model = MultiTaskResistancePredictor(config)
        >>> set_extractor = SetFeatureExtractor(set_config)
        >>>
        >>> # Create enhanced model
        >>> enhanced = SetEnhancedMTL(mtl_model, set_extractor)
        >>>
        >>> # Forward pass
        >>> embeddings = encoder(sequences)  # From PLM/VAE
        >>> mutations = [MutationSet.from_strings(m) for m in batch_mutations]
        >>> outputs = enhanced(embeddings, mutations)
    """

    def __init__(
        self,
        base_model: nn.Module,
        set_extractor: SetFeatureExtractor,
        fusion_method: str = "concat",
        fusion_hidden_dim: int = 128,
    ):
        """Initialize enhanced MTL.

        Args:
            base_model: Base MTL predictor
            set_extractor: Set feature extractor
            fusion_method: How to combine features ('concat', 'gate', 'attention')
            fusion_hidden_dim: Hidden dimension for fusion layers
        """
        super().__init__()
        self.base_model = base_model
        self.set_extractor = set_extractor
        self.fusion_method = fusion_method

        # Get input dimensions
        set_dim = set_extractor.feature_dim

        # Build fusion layer based on method
        if fusion_method == "concat":
            # Simple concatenation - need to modify base model input
            self.fusion = nn.Identity()
            self.set_projection = nn.Linear(set_dim, fusion_hidden_dim)

        elif fusion_method == "gate":
            # Gated fusion
            self.set_projection = nn.Linear(set_dim, fusion_hidden_dim)
            self.gate = nn.Sequential(
                nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
                nn.Sigmoid(),
            )
            self.fusion = nn.Linear(fusion_hidden_dim, fusion_hidden_dim)

        elif fusion_method == "attention":
            # Cross-attention between neural and set features
            self.set_projection = nn.Linear(set_dim, fusion_hidden_dim)
            self.attention = nn.MultiheadAttention(
                embed_dim=fusion_hidden_dim,
                num_heads=4,
                batch_first=True,
            )
            self.fusion = nn.Linear(fusion_hidden_dim, fusion_hidden_dim)

    def forward(
        self,
        embeddings: torch.Tensor,
        mutations: list[MutationSet],
    ) -> dict[str, torch.Tensor]:
        """Forward pass with combined features.

        Args:
            embeddings: Neural embeddings (batch_size, embed_dim)
            mutations: List of mutation sets

        Returns:
            Prediction outputs
        """
        # Extract set features
        set_features = self.set_extractor.extract_batch(mutations)
        set_features = set_features.to(embeddings.device)
        set_projected = self.set_projection(set_features)

        # Fuse features
        if self.fusion_method == "concat":
            # Concatenate and pass to base model
            combined = torch.cat([embeddings, set_projected], dim=-1)
            # Note: Base model needs to handle larger input
            return self.base_model(combined)

        elif self.fusion_method == "gate":
            # Gated combination
            gate_input = torch.cat([embeddings, set_projected], dim=-1)
            gate = self.gate(gate_input)
            fused = gate * embeddings + (1 - gate) * set_projected
            return self.base_model(fused)

        elif self.fusion_method == "attention":
            # Cross-attention
            # Query: neural, Key/Value: set
            attended, _ = self.attention(
                embeddings.unsqueeze(1),
                set_projected.unsqueeze(1),
                set_projected.unsqueeze(1),
            )
            fused = embeddings + attended.squeeze(1)
            return self.base_model(fused)

        return self.base_model(embeddings)


class SetAwareTaskWeighting(nn.Module):
    """Task weighting informed by set-theoretic relationships.

    Uses cross-resistance patterns to inform task weighting,
    giving higher weight to related drug predictions.
    """

    def __init__(
        self,
        n_drugs: int,
        cross_resistance_matrix: dict[tuple[str, str], float] | None = None,
    ):
        """Initialize task weighting.

        Args:
            n_drugs: Number of drugs
            cross_resistance_matrix: Pairwise cross-resistance scores
        """
        super().__init__()
        self.n_drugs = n_drugs

        # Initialize correlation matrix
        if cross_resistance_matrix:
            # Build correlation matrix from cross-resistance
            corr = torch.eye(n_drugs)
            # Would need drug name mapping to fill in
        else:
            corr = torch.eye(n_drugs)

        self.register_buffer("correlation", corr)

        # Learnable task weights
        self.log_weights = nn.Parameter(torch.zeros(n_drugs))

    def forward(
        self,
        task_losses: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted task loss.

        Args:
            task_losses: Per-task losses (n_drugs,)

        Returns:
            Weighted total loss
        """
        weights = torch.softmax(self.log_weights, dim=0)

        # Incorporate correlation - upweight correlated tasks together
        corr_weights = torch.mv(self.correlation, weights)
        corr_weights = corr_weights / corr_weights.sum()

        return (corr_weights * task_losses).sum()
