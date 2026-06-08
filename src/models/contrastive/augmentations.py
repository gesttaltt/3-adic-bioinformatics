# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Sequence augmentation strategies for contrastive learning.

Provides various augmentations for protein sequences to create
positive pairs for self-supervised learning.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# Amino acid groups for biologically plausible substitutions
AA_GROUPS = {
    "hydrophobic": "AILMFVPGW",
    "polar": "STNQCY",
    "positive": "RKH",
    "negative": "DE",
    "aromatic": "FYW",
    "small": "AGST",
}

# Substitution matrix based on biochemical similarity
SIMILAR_AAS = {
    "A": "GVLS",
    "C": "STMA",
    "D": "EN",
    "E": "DQ",
    "F": "YW",
    "G": "AS",
    "H": "NQR",
    "I": "LVM",
    "K": "RH",
    "L": "IVM",
    "M": "ILV",
    "N": "DQSH",
    "P": "AG",
    "Q": "ENH",
    "R": "KH",
    "S": "TNGA",
    "T": "SNV",
    "V": "ILMA",
    "W": "FY",
    "Y": "FWH",
}


class SequenceAugmentation(ABC):
    """Abstract base class for sequence augmentations."""

    @abstractmethod
    def __call__(
        self,
        sequence: str | torch.Tensor,
    ) -> str | torch.Tensor:
        """Apply augmentation.

        Args:
            sequence: Input sequence (string or token tensor)

        Returns:
            Augmented sequence
        """
        pass


class MutationAugmentation(SequenceAugmentation):
    """Random mutation augmentation.

    Applies random amino acid substitutions, optionally
    using biochemically similar replacements.
    """

    def __init__(
        self,
        mutation_rate: float = 0.15,
        use_similar: bool = True,
        mask_token: int = 0,
    ):
        """Initialize mutation augmentation.

        Args:
            mutation_rate: Probability of mutating each position
            use_similar: Use biochemically similar substitutions
            mask_token: Token index for masked positions
        """
        self.mutation_rate = mutation_rate
        self.use_similar = use_similar
        self.mask_token = mask_token

    def __call__(
        self,
        sequence: str | torch.Tensor,
    ) -> str | torch.Tensor:
        """Apply mutation augmentation."""
        if isinstance(sequence, str):
            return self._mutate_string(sequence)
        else:
            return self._mutate_tensor(sequence)

    def _mutate_string(self, sequence: str) -> str:
        """Mutate string sequence."""
        result = list(sequence)
        all_aas = "ACDEFGHIKLMNPQRSTVWY"

        for i in range(len(result)):
            if random.random() < self.mutation_rate:
                if self.use_similar and result[i] in SIMILAR_AAS:
                    # Use similar amino acid
                    similar = SIMILAR_AAS[result[i]]
                    result[i] = random.choice(similar)
                else:
                    # Random substitution
                    result[i] = random.choice(all_aas)

        return "".join(result)

    def _mutate_tensor(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mutate token tensor."""
        result = tokens.clone()
        mask = torch.rand(tokens.shape, device=tokens.device) < self.mutation_rate

        # Random tokens (1-20 for amino acids)
        random_tokens = torch.randint(1, 21, tokens.shape, device=tokens.device)
        result = torch.where(mask, random_tokens, result)

        return result


class MaskingAugmentation(SequenceAugmentation):
    """Random masking augmentation.

    Masks random positions in the sequence, similar to
    masked language modeling.
    """

    def __init__(
        self,
        mask_rate: float = 0.15,
        mask_token: int = 0,
        random_token_rate: float = 0.1,
        keep_rate: float = 0.1,
    ):
        """Initialize masking augmentation.

        Args:
            mask_rate: Probability of selecting each position
            mask_token: Token index for [MASK]
            random_token_rate: Probability of random replacement
            keep_rate: Probability of keeping original token
        """
        self.mask_rate = mask_rate
        self.mask_token = mask_token
        self.random_token_rate = random_token_rate
        self.keep_rate = keep_rate

    def __call__(
        self,
        sequence: str | torch.Tensor,
    ) -> str | torch.Tensor:
        """Apply masking augmentation."""
        if isinstance(sequence, str):
            return self._mask_string(sequence)
        else:
            return self._mask_tensor(sequence)

    def _mask_string(self, sequence: str) -> str:
        """Mask string sequence."""
        result = list(sequence)
        all_aas = "ACDEFGHIKLMNPQRSTVWY"

        for i in range(len(result)):
            if random.random() < self.mask_rate:
                rand = random.random()
                if rand < 1 - self.random_token_rate - self.keep_rate:
                    result[i] = "X"  # Mask token
                elif rand < 1 - self.keep_rate:
                    result[i] = random.choice(all_aas)
                # else: keep original

        return "".join(result)

    def _mask_tensor(self, tokens: torch.Tensor) -> torch.Tensor:
        """Mask token tensor."""
        result = tokens.clone()
        device = tokens.device

        # Select positions to modify
        mask = torch.rand(tokens.shape, device=device) < self.mask_rate

        # Decide action for each masked position
        action_rand = torch.rand(tokens.shape, device=device)
        mask_action = action_rand < (1 - self.random_token_rate - self.keep_rate)
        random_action = (action_rand >= (1 - self.random_token_rate - self.keep_rate)) & (
            action_rand < (1 - self.keep_rate)
        )

        # Apply mask token
        result = torch.where(mask & mask_action, self.mask_token, result)

        # Apply random token
        random_tokens = torch.randint(1, 21, tokens.shape, device=device)
        result = torch.where(mask & random_action, random_tokens, result)

        return result


class CropAugmentation(SequenceAugmentation):
    """Random cropping augmentation.

    Crops a random contiguous subsequence.
    """

    def __init__(
        self,
        min_length: int = 50,
        max_length: int | None = None,
        crop_ratio: float = 0.8,
    ):
        """Initialize crop augmentation.

        Args:
            min_length: Minimum crop length
            max_length: Maximum crop length (None = sequence length)
            crop_ratio: Fraction of sequence to keep
        """
        self.min_length = min_length
        self.max_length = max_length
        self.crop_ratio = crop_ratio

    def __call__(
        self,
        sequence: str | torch.Tensor,
    ) -> str | torch.Tensor:
        """Apply crop augmentation."""
        if isinstance(sequence, str):
            return self._crop_string(sequence)
        else:
            return self._crop_tensor(sequence)

    def _crop_string(self, sequence: str) -> str:
        """Crop string sequence."""
        seq_len = len(sequence)
        crop_len = max(self.min_length, int(seq_len * self.crop_ratio))

        if self.max_length:
            crop_len = min(crop_len, self.max_length)

        crop_len = min(crop_len, seq_len)

        # Random start position
        max_start = seq_len - crop_len
        start = random.randint(0, max(0, max_start))

        return sequence[start : start + crop_len]

    def _crop_tensor(self, tokens: torch.Tensor) -> torch.Tensor:
        """Crop token tensor."""
        seq_len = tokens.shape[0] if tokens.dim() == 1 else tokens.shape[-1]

        crop_len = max(self.min_length, int(seq_len * self.crop_ratio))

        if self.max_length:
            crop_len = min(crop_len, self.max_length)

        crop_len = min(crop_len, seq_len)

        max_start = seq_len - crop_len
        start = random.randint(0, max(0, max_start))

        if tokens.dim() == 1:
            return tokens[start : start + crop_len]
        else:
            return tokens[..., start : start + crop_len]


class ShuffleAugmentation(SequenceAugmentation):
    """Local shuffling augmentation.

    Shuffles amino acids within local windows, preserving
    approximate sequence composition.
    """

    def __init__(
        self,
        window_size: int = 5,
        shuffle_prob: float = 0.3,
    ):
        """Initialize shuffle augmentation.

        Args:
            window_size: Size of local shuffle windows
            shuffle_prob: Probability of shuffling each window
        """
        self.window_size = window_size
        self.shuffle_prob = shuffle_prob

    def __call__(
        self,
        sequence: str | torch.Tensor,
    ) -> str | torch.Tensor:
        """Apply shuffle augmentation."""
        if isinstance(sequence, str):
            return self._shuffle_string(sequence)
        else:
            return self._shuffle_tensor(sequence)

    def _shuffle_string(self, sequence: str) -> str:
        """Shuffle string sequence."""
        result = list(sequence)

        for i in range(0, len(result), self.window_size):
            if random.random() < self.shuffle_prob:
                window = result[i : i + self.window_size]
                random.shuffle(window)
                result[i : i + self.window_size] = window

        return "".join(result)

    def _shuffle_tensor(self, tokens: torch.Tensor) -> torch.Tensor:
        """Shuffle token tensor."""
        result = tokens.clone()
        seq_len = tokens.shape[-1] if tokens.dim() > 1 else tokens.shape[0]

        for i in range(0, seq_len, self.window_size):
            if random.random() < self.shuffle_prob:
                end = min(i + self.window_size, seq_len)
                perm = torch.randperm(end - i, device=tokens.device)

                if tokens.dim() == 1:
                    result[i:end] = result[i:end][perm]
                else:
                    result[..., i:end] = result[..., i:end][..., perm]

        return result


class SequenceAugmentations(nn.Module):
    """Compose multiple augmentations.

    Applies a sequence of augmentations with specified probabilities.
    """

    def __init__(
        self,
        augmentations: list[tuple[SequenceAugmentation, float]] | None = None,
    ):
        """Initialize augmentation pipeline.

        Args:
            augmentations: List of (augmentation, probability) pairs
        """
        super().__init__()

        if augmentations is None:
            # Default augmentation pipeline
            augmentations = [
                (MutationAugmentation(mutation_rate=0.1), 0.5),
                (MaskingAugmentation(mask_rate=0.1), 0.5),
                (CropAugmentation(crop_ratio=0.9), 0.3),
                (ShuffleAugmentation(window_size=3), 0.2),
            ]

        self.augmentations = augmentations

    def forward(
        self,
        sequence: str | torch.Tensor,
    ) -> str | torch.Tensor:
        """Apply augmentation pipeline.

        Args:
            sequence: Input sequence

        Returns:
            Augmented sequence
        """
        result = sequence

        for aug, prob in self.augmentations:
            if random.random() < prob:
                result = aug(result)

        return result

    def create_pair(
        self,
        sequence: str | torch.Tensor,
    ) -> tuple[str | torch.Tensor, str | torch.Tensor]:
        """Create augmented pair for contrastive learning.

        Args:
            sequence: Input sequence

        Returns:
            Tuple of two differently augmented versions
        """
        view1 = self.forward(sequence)
        view2 = self.forward(sequence)

        return view1, view2

    @staticmethod
    def default_for_byol() -> SequenceAugmentations:
        """Get default augmentations for BYOL training."""
        return SequenceAugmentations(
            [
                (MutationAugmentation(mutation_rate=0.15, use_similar=True), 0.7),
                (MaskingAugmentation(mask_rate=0.15), 0.5),
                (CropAugmentation(crop_ratio=0.85, min_length=32), 0.4),
            ]
        )

    @staticmethod
    def default_for_simclr() -> SequenceAugmentations:
        """Get default augmentations for SimCLR training."""
        return SequenceAugmentations(
            [
                (MutationAugmentation(mutation_rate=0.2), 0.8),
                (MaskingAugmentation(mask_rate=0.2), 0.6),
                (CropAugmentation(crop_ratio=0.7, min_length=24), 0.5),
                (ShuffleAugmentation(window_size=5), 0.3),
            ]
        )
