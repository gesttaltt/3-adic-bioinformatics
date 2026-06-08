# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Off-target activity prediction for CRISPR analysis.

This module provides neural network prediction of Cas9 cleavage
activity at potential off-target sites.

Single responsibility: Activity prediction.
"""

import torch
import torch.nn as nn


class OfftargetActivityPredictor(nn.Module):
    """Predicts cleavage activity at off-target sites.

    Uses learned embeddings and mismatch patterns to predict
    the probability of Cas9 cleavage at a given off-target.

    Attributes:
        seq_len: Length of guide sequence
    """

    def __init__(
        self,
        seq_len: int = 20,
        hidden_dim: int = 128,
    ):
        """Initialize predictor.

        Args:
            seq_len: Length of guide sequence
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.seq_len = seq_len

        # Mismatch encoding (one-hot for each position)
        self.mismatch_encoder = nn.Sequential(
            nn.Linear(seq_len * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )

        # Sequence similarity features
        self.similarity_encoder = nn.Sequential(
            nn.Linear(seq_len, hidden_dim // 2),
            nn.ReLU(),
        )

        # Final predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def compute_mismatch_features(
        self,
        target: str,
        offtarget: str,
    ) -> torch.Tensor:
        """Compute mismatch feature vector.

        Args:
            target: Target sequence
            offtarget: Off-target sequence

        Returns:
            Mismatch features (seq_len * 4,)
        """
        features = torch.zeros(self.seq_len, 4)

        for i, (t, o) in enumerate(zip(target.upper(), offtarget.upper(), strict=False)):
            if t == o:
                features[i, 0] = 1.0  # Match
            elif (t in "AG" and o in "AG") or (t in "CT" and o in "CT"):
                features[i, 1] = 1.0  # Transition
            else:
                features[i, 2] = 1.0  # Transversion

        return features.flatten()

    def forward(
        self,
        targets: list[str],
        offtargets: list[str],
    ) -> torch.Tensor:
        """Predict off-target activity.

        Args:
            targets: List of target sequences
            offtargets: List of corresponding off-target sequences

        Returns:
            Predicted activity probabilities (batch,)
        """
        batch_size = len(targets)

        # Compute mismatch features
        mismatch_features = torch.stack(
            [self.compute_mismatch_features(t, o) for t, o in zip(targets, offtargets, strict=False)]
        )

        # Compute similarity (1 - mismatch fraction)
        similarity = torch.zeros(batch_size, self.seq_len)
        for i, (t, o) in enumerate(zip(targets, offtargets, strict=False)):
            for j, (nt1, nt2) in enumerate(zip(t.upper(), o.upper(), strict=False)):
                similarity[i, j] = 1.0 if nt1 == nt2 else 0.0

        # Encode features
        mismatch_encoded = self.mismatch_encoder(mismatch_features)
        similarity_encoded = self.similarity_encoder(similarity)

        # Combine and predict
        combined = torch.cat([mismatch_encoded, similarity_encoded], dim=-1)
        activity = self.predictor(combined).squeeze(-1)

        return activity


__all__ = ["OfftargetActivityPredictor"]
