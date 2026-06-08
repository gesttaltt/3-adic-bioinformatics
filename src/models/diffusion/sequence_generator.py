# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""High-level sequence generation interface.

Provides easy-to-use APIs for generating protein sequences
using discrete diffusion models.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.diffusion.d3pm import D3PM, ConditionalD3PM, D3PMConfig

# Standard amino acid vocabulary
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}  # 0 reserved for mask
IDX_TO_AA = {i + 1: aa for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AA[0] = "-"  # Mask/gap character


class SequenceGenerator(nn.Module):
    """High-level interface for sequence generation.

    Wraps D3PM model with convenient methods for generating
    protein sequences.

    Example:
        >>> generator = SequenceGenerator()
        >>> sequences = generator.generate(n_samples=10, length=100)
        >>> print(sequences[0])  # "MKLLVVLLFV..."
    """

    def __init__(
        self,
        model: D3PM | None = None,
        config: D3PMConfig | None = None,
        device: str = "cuda",
    ):
        """Initialize sequence generator.

        Args:
            model: Pre-trained D3PM model (creates new if None)
            config: Configuration for new model
            device: Computation device
        """
        super().__init__()

        if model is not None:
            self.model = model
        else:
            config = config or D3PMConfig()
            self.model = D3PM(config)

        self._device = device
        self.to(device)

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    def generate(
        self,
        n_samples: int = 1,
        length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        return_tokens: bool = False,
    ) -> list[str] | tuple[list[str], torch.Tensor]:
        """Generate protein sequences.

        Args:
            n_samples: Number of sequences to generate
            length: Sequence length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            return_tokens: Also return token tensor

        Returns:
            List of sequence strings, optionally with tokens
        """
        tokens = self.model.sample(
            batch_size=n_samples,
            seq_len=length,
            temperature=temperature,
        )

        # Convert to sequences
        sequences = self._tokens_to_sequences(tokens)

        if return_tokens:
            return sequences, tokens
        return sequences

    def _tokens_to_sequences(self, tokens: torch.Tensor) -> list[str]:
        """Convert token tensor to sequence strings.

        Args:
            tokens: (batch, length) token indices

        Returns:
            List of sequence strings
        """
        sequences = []
        for seq_tokens in tokens:
            seq = "".join(IDX_TO_AA.get(t.item(), "X") for t in seq_tokens)
            # Remove gap characters
            seq = seq.replace("-", "")
            sequences.append(seq)

        return sequences

    def _sequences_to_tokens(self, sequences: list[str]) -> torch.Tensor:
        """Convert sequence strings to tokens.

        Args:
            sequences: List of sequence strings

        Returns:
            Token tensor
        """
        max_len = max(len(s) for s in sequences)
        tokens = torch.zeros(len(sequences), max_len, dtype=torch.long, device=self.device)

        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq):
                tokens[i, j] = AA_TO_IDX.get(aa, 0)

        return tokens

    def compute_likelihood(
        self,
        sequences: list[str],
    ) -> torch.Tensor:
        """Compute log-likelihood of sequences.

        Args:
            sequences: List of sequence strings

        Returns:
            Log-likelihood for each sequence
        """
        tokens = self._sequences_to_tokens(sequences)
        mask = tokens != 0

        # Compute loss (negative log-likelihood)
        with torch.no_grad():
            nll = self.model.compute_loss(tokens, mask)

        return -nll


class ConditionalGenerator(nn.Module):
    """Conditional sequence generator.

    Generates sequences conditioned on embeddings or properties.

    Example:
        >>> generator = ConditionalGenerator(condition_dim=64)
        >>> # Generate resistant sequences
        >>> embedding = resistance_encoder(target_profile)
        >>> sequences = generator.generate(embedding, n_samples=10)
    """

    def __init__(
        self,
        condition_dim: int = 64,
        config: D3PMConfig | None = None,
        device: str = "cuda",
    ):
        """Initialize conditional generator.

        Args:
            condition_dim: Dimension of conditioning vector
            config: D3PM configuration
            device: Computation device
        """
        super().__init__()

        config = config or D3PMConfig()
        self.model = ConditionalD3PM(config, condition_dim)
        self.condition_dim = condition_dim

        self._device = device
        self.to(device)

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    def generate(
        self,
        condition: torch.Tensor,
        n_samples: int | None = None,
        length: int = 100,
        guidance_scale: float = 2.0,
        temperature: float = 1.0,
    ) -> list[str]:
        """Generate conditioned sequences.

        Args:
            condition: Conditioning tensor (batch, condition_dim) or (condition_dim,)
            n_samples: Number of samples (inferred from condition if None)
            length: Sequence length
            guidance_scale: Classifier-free guidance strength
            temperature: Sampling temperature

        Returns:
            List of generated sequences
        """
        # Handle single condition
        if condition.dim() == 1:
            condition = condition.unsqueeze(0)

        if n_samples is not None and n_samples != condition.shape[0]:
            # Expand condition to match n_samples
            condition = condition.expand(n_samples, -1)

        condition = condition.to(self.device)

        tokens = self.model.sample(
            batch_size=condition.shape[0],
            seq_len=length,
            condition=condition,
            guidance_scale=guidance_scale,
            temperature=temperature,
        )

        return self._tokens_to_sequences(tokens)

    def _tokens_to_sequences(self, tokens: torch.Tensor) -> list[str]:
        """Convert tokens to sequences."""
        sequences = []
        for seq_tokens in tokens:
            seq = "".join(IDX_TO_AA.get(t.item(), "X") for t in seq_tokens)
            seq = seq.replace("-", "")
            sequences.append(seq)
        return sequences

    def generate_with_properties(
        self,
        target_properties: dict[str, float],
        property_encoder: nn.Module,
        n_samples: int = 10,
        length: int = 100,
        **kwargs,
    ) -> list[str]:
        """Generate sequences with target properties.

        Args:
            target_properties: Dictionary of target property values
            property_encoder: Module to encode properties to condition
            n_samples: Number of samples
            length: Sequence length
            **kwargs: Additional generation arguments

        Returns:
            Generated sequences
        """
        # Encode properties to condition vector
        prop_tensor = torch.tensor(
            list(target_properties.values()),
            device=self.device,
        ).unsqueeze(0)

        condition = property_encoder(prop_tensor)
        condition = condition.expand(n_samples, -1)

        return self.generate(condition, length=length, **kwargs)


class InfillGenerator(nn.Module):
    """Sequence infilling generator.

    Generates sequences by filling in masked positions
    while respecting constraints from unmasked positions.
    """

    def __init__(
        self,
        model: D3PM,
        device: str = "cuda",
    ):
        """Initialize infilling generator.

        Args:
            model: D3PM model
            device: Computation device
        """
        super().__init__()
        self.model = model
        self._device = device
        self.to(device)

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    def infill(
        self,
        partial_sequence: str,
        mask_positions: list[int],
        temperature: float = 1.0,
        n_samples: int = 1,
    ) -> list[str]:
        """Fill in masked positions in a sequence.

        Args:
            partial_sequence: Sequence with gaps to fill
            mask_positions: Positions to generate
            temperature: Sampling temperature
            n_samples: Number of alternative fills

        Returns:
            List of completed sequences
        """
        # Convert to tokens
        tokens = torch.zeros(n_samples, len(partial_sequence), dtype=torch.long, device=self.device)

        for i, aa in enumerate(partial_sequence):
            if i in mask_positions:
                tokens[:, i] = 0  # Mask token
            else:
                tokens[:, i] = AA_TO_IDX.get(aa, 0)

        # Create mask for constrained positions
        constraint_mask = torch.ones(n_samples, len(partial_sequence), dtype=torch.bool, device=self.device)
        constraint_mask[:, mask_positions] = False

        # Generate with constraints
        filled = self._constrained_sample(tokens, constraint_mask, temperature)

        # Convert back to sequences
        return self._tokens_to_sequences(filled)

    def _constrained_sample(
        self,
        x: torch.Tensor,
        constraint_mask: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Sample while respecting constraints.

        Args:
            x: Token tensor with masked positions
            constraint_mask: True for constrained (fixed) positions
            temperature: Sampling temperature

        Returns:
            Completed token tensor
        """
        # Store original constrained values
        original = x.clone()

        for t in reversed(range(self.model.config.n_timesteps)):
            t_batch = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)

            # Predict
            logits = self.model.forward(x, t_batch)
            logits = logits / temperature

            if t > 0:
                probs = torch.softmax(logits, dim=-1)
                x = self.model._posterior_sample(x, probs, t, t - 1)
            else:
                x = logits.argmax(dim=-1)

            # Restore constraints
            x = torch.where(constraint_mask, original, x)

        return x

    def _tokens_to_sequences(self, tokens: torch.Tensor) -> list[str]:
        """Convert tokens to sequences."""
        sequences = []
        for seq_tokens in tokens:
            seq = "".join(IDX_TO_AA.get(t.item(), "X") for t in seq_tokens)
            sequences.append(seq)
        return sequences


class MutationGenerator(nn.Module):
    """Generate mutations using diffusion.

    Creates mutations by partially corrupting a wild-type sequence
    and denoising to explore the sequence space.
    """

    def __init__(
        self,
        model: D3PM,
        device: str = "cuda",
    ):
        """Initialize mutation generator.

        Args:
            model: D3PM model
            device: Computation device
        """
        super().__init__()
        self.model = model
        self._device = device
        self.to(device)

    @property
    def device(self) -> torch.device:
        return torch.device(self._device)

    def generate_mutations(
        self,
        wild_type: str,
        mutation_rate: float = 0.1,
        n_variants: int = 10,
        temperature: float = 1.0,
    ) -> list[tuple[str, list[tuple[int, str, str]]]]:
        """Generate mutations from wild-type sequence.

        Args:
            wild_type: Wild-type sequence
            mutation_rate: Fraction of positions to mutate
            n_variants: Number of variants to generate
            temperature: Sampling temperature

        Returns:
            List of (variant_sequence, [(pos, wt_aa, mut_aa), ...])
        """
        # Convert to tokens
        wt_tokens = torch.zeros(n_variants, len(wild_type), dtype=torch.long, device=self.device)
        for i, aa in enumerate(wild_type):
            wt_tokens[:, i] = AA_TO_IDX.get(aa, 0)

        # Select positions to mutate
        n_positions = int(len(wild_type) * mutation_rate)
        mutation_positions = torch.randint(
            0,
            len(wild_type),
            (n_variants, n_positions),
            device=self.device,
        )

        # Corrupt selected positions
        corrupted = wt_tokens.clone()
        for i in range(n_variants):
            for pos in mutation_positions[i]:
                corrupted[i, pos] = 0  # Mask

        # Determine timestep for partial corruption
        t_corrupt = int(self.model.config.n_timesteps * mutation_rate)

        # Denoise from corruption level
        variants = self._denoise_from_timestep(corrupted, t_corrupt, temperature)

        # Extract mutations
        results = []
        for i, var_tokens in enumerate(variants):
            var_seq = "".join(IDX_TO_AA.get(t.item(), "X") for t in var_tokens)
            mutations = []

            for j, (wt_aa, var_aa) in enumerate(zip(wild_type, var_seq, strict=False)):
                if wt_aa != var_aa:
                    mutations.append((j, wt_aa, var_aa))

            results.append((var_seq, mutations))

        return results

    def _denoise_from_timestep(
        self,
        x: torch.Tensor,
        start_t: int,
        temperature: float,
    ) -> torch.Tensor:
        """Denoise from a specific timestep.

        Args:
            x: Partially corrupted tokens
            start_t: Starting timestep
            temperature: Sampling temperature

        Returns:
            Denoised tokens
        """
        for t in reversed(range(start_t)):
            t_batch = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)

            logits = self.model.forward(x, t_batch)
            logits = logits / temperature

            if t > 0:
                probs = torch.softmax(logits, dim=-1)
                x = self.model._posterior_sample(x, probs, t, t - 1)
            else:
                x = logits.argmax(dim=-1)

        return x
