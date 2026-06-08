# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Set-theoretic data augmentation for mutation profiles.

Provides augmentation strategies based on set operations:
- Union: Combine mutation profiles (creates MDR-like samples)
- Intersection: Extract common mutations
- Difference: Create subset profiles
- Power set sampling: Generate all possible subsets
- Lattice-aware augmentation: Respect resistance hierarchy

These augmentations are biologically motivated and preserve
meaningful relationships between mutations and resistance.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset

from src.analysis.set_theory.formal_concepts import ConceptLattice, FormalContext
from src.analysis.set_theory.lattice import ResistanceLattice, ResistanceLevel
from src.analysis.set_theory.mutation_sets import (
    Mutation,
    MutationSet,
)


@dataclass
class AugmentationConfig:
    """Configuration for set-theoretic augmentation.

    Attributes:
        union_prob: Probability of union augmentation
        intersection_prob: Probability of intersection augmentation
        subset_prob: Probability of subset augmentation
        power_set_max_size: Max size for power set sampling
        preserve_hierarchy: Only create valid hierarchy transitions
        min_mutations: Minimum mutations in augmented sample
        max_mutations: Maximum mutations in augmented sample
    """

    union_prob: float = 0.3
    intersection_prob: float = 0.2
    subset_prob: float = 0.3
    power_set_max_size: int = 5
    preserve_hierarchy: bool = True
    min_mutations: int = 1
    max_mutations: int = 10


class SetAugmenter:
    """Data augmenter using set operations on mutations.

    Generates synthetic samples by combining existing mutation profiles
    using set-theoretic operations. This is useful for:
    - Expanding training data
    - Creating balanced MDR/XDR samples
    - Exploring mutation combination space

    Example:
        >>> augmenter = SetAugmenter(config)
        >>> augmenter.add_samples(training_mutation_sets)
        >>> augmented = augmenter.augment(original_sample)
    """

    def __init__(
        self,
        config: AugmentationConfig | None = None,
        lattice: ResistanceLattice | None = None,
    ):
        """Initialize augmenter.

        Args:
            config: Augmentation configuration
            lattice: Optional resistance lattice for hierarchy-aware augmentation
        """
        self.config = config or AugmentationConfig()
        self.lattice = lattice or ResistanceLattice()

        # Store samples for combination
        self.samples: list[MutationSet] = []
        self.by_level: dict[ResistanceLevel, list[MutationSet]] = {level: [] for level in ResistanceLevel}

    def add_samples(self, samples: list[MutationSet]):
        """Add samples for augmentation source.

        Args:
            samples: List of mutation sets
        """
        for sample in samples:
            self.samples.append(sample)

            # Index by resistance level
            level = self.lattice.resistance_level(sample)
            self.by_level[level].append(sample)

    def augment(
        self,
        sample: MutationSet,
        n_augmentations: int = 1,
    ) -> list[MutationSet]:
        """Generate augmented versions of a sample.

        Args:
            sample: Original sample
            n_augmentations: Number of augmentations to generate

        Returns:
            List of augmented mutation sets
        """
        augmented = []

        for _ in range(n_augmentations):
            aug = self._augment_single(sample)
            if aug and len(aug) >= self.config.min_mutations:
                augmented.append(aug)

        return augmented

    def _augment_single(self, sample: MutationSet) -> MutationSet | None:
        """Generate a single augmentation.

        Args:
            sample: Original sample

        Returns:
            Augmented sample or None
        """
        if not self.samples:
            return None

        r = random.random()

        if r < self.config.union_prob:
            return self._union_augment(sample)
        elif r < self.config.union_prob + self.config.intersection_prob:
            return self._intersection_augment(sample)
        elif r < self.config.union_prob + self.config.intersection_prob + self.config.subset_prob:
            return self._subset_augment(sample)
        else:
            return self._mutation_swap(sample)

    def _union_augment(self, sample: MutationSet) -> MutationSet:
        """Create augmentation via union with another sample.

        Args:
            sample: Original sample

        Returns:
            Union of sample with random other sample
        """
        other = random.choice(self.samples)
        combined = sample | other

        # Respect max mutations
        if len(combined) > self.config.max_mutations:
            # Randomly select subset
            mutations = list(combined._mutations)
            random.shuffle(mutations)
            combined = MutationSet(mutations[: self.config.max_mutations])

        return combined

    def _intersection_augment(self, sample: MutationSet) -> MutationSet | None:
        """Create augmentation via intersection.

        Args:
            sample: Original sample

        Returns:
            Intersection with another sample (if non-empty)
        """
        # Find samples with overlap
        candidates = [s for s in self.samples if not s.isdisjoint(sample) and s != sample]

        if not candidates:
            return sample

        other = random.choice(candidates)
        intersection = sample & other

        if len(intersection) >= self.config.min_mutations:
            return intersection

        return sample

    def _subset_augment(self, sample: MutationSet) -> MutationSet:
        """Create augmentation via random subset.

        Args:
            sample: Original sample

        Returns:
            Random subset of sample
        """
        mutations = list(sample._mutations)

        if len(mutations) <= self.config.min_mutations:
            return sample

        # Random subset size
        size = random.randint(self.config.min_mutations, len(mutations) - 1)

        random.shuffle(mutations)
        return MutationSet(mutations[:size])

    def _mutation_swap(self, sample: MutationSet) -> MutationSet:
        """Swap some mutations with mutations from another sample.

        Args:
            sample: Original sample

        Returns:
            Sample with swapped mutations
        """
        other = random.choice(self.samples)

        mutations = list(sample._mutations)
        other_mutations = list(other._mutations)

        if not mutations or not other_mutations:
            return sample

        # Swap 1-2 mutations
        n_swap = min(2, len(mutations), len(other_mutations))

        result = list(mutations)
        for _ in range(n_swap):
            if result and other_mutations:
                idx = random.randint(0, len(result) - 1)
                result[idx] = random.choice(other_mutations)

        return MutationSet(result)

    def generate_hierarchy_samples(
        self,
        target_level: ResistanceLevel,
        n_samples: int = 10,
    ) -> list[MutationSet]:
        """Generate samples at a specific resistance level.

        Args:
            target_level: Target resistance level
            n_samples: Number of samples to generate

        Returns:
            List of generated samples
        """
        generated = []

        # Get existing samples at adjacent levels
        level_samples = self.by_level[target_level]

        lower_samples = self.by_level[ResistanceLevel(target_level.value - 1)] if target_level.value > 0 else []

        if target_level.value < len(ResistanceLevel) - 1:
            higher_samples = self.by_level[ResistanceLevel(target_level.value + 1)]
        else:
            higher_samples = []

        for _ in range(n_samples):
            if level_samples:
                # Start with existing sample at level
                base = random.choice(level_samples)
                generated.append(self._augment_single(base))
            elif lower_samples and higher_samples:
                # Combine lower and higher to reach target
                low = random.choice(lower_samples)
                high = random.choice(higher_samples)

                # Union to increase, intersection to decrease
                if random.random() > 0.5:
                    candidate = low | MutationSet(random.sample(list(high._mutations), min(2, len(high))))
                else:
                    candidate = MutationSet(random.sample(list(high._mutations), max(1, len(high) - 1)))

                # Check if at target level
                if self.lattice.resistance_level(candidate) == target_level:
                    generated.append(candidate)

        return [g for g in generated if g is not None]


class ConceptBasedAugmenter:
    """Augment data based on FCA concepts.

    Uses formal concepts to generate semantically meaningful
    augmentations that preserve concept membership.
    """

    def __init__(
        self,
        concept_lattice: ConceptLattice,
        context: FormalContext,
    ):
        """Initialize concept-based augmenter.

        Args:
            concept_lattice: Concept lattice
            context: Formal context
        """
        self.lattice = concept_lattice
        self.context = context

    def augment_to_concept(
        self,
        sample: MutationSet,
        target_concept: int,
    ) -> MutationSet | None:
        """Augment sample to match a target concept.

        Args:
            sample: Original sample
            target_concept: Index of target concept

        Returns:
            Augmented sample matching concept intent
        """
        if target_concept >= len(self.lattice.concepts):
            return None

        concept = self.lattice.concepts[target_concept]

        # Get mutations from concept intent
        concept_mutations = {attr for attr in concept.intent if not attr.endswith("_R")}

        # Create sample with these mutations
        mutations = []
        for mut_str in concept_mutations:
            try:
                mutations.append(Mutation.from_string(mut_str))
            except ValueError:
                continue

        if mutations:
            return MutationSet(mutations)

        return None

    def generate_concept_samples(
        self,
        n_per_concept: int = 5,
    ) -> dict[int, list[MutationSet]]:
        """Generate samples for each concept.

        Args:
            n_per_concept: Samples per concept

        Returns:
            Concept index -> samples mapping
        """
        samples = {}

        for idx, concept in enumerate(self.lattice.concepts):
            concept_samples = []

            # Get base mutations from intent
            base_mutations = []
            for attr in concept.intent:
                if not attr.endswith("_R"):
                    try:
                        base_mutations.append(Mutation.from_string(attr))
                    except ValueError:
                        continue

            if not base_mutations:
                continue

            base = MutationSet(base_mutations)

            # Generate variations
            for _ in range(n_per_concept):
                if random.random() > 0.5 and len(base) > 1:
                    # Subset
                    muts = list(base._mutations)
                    random.shuffle(muts)
                    sample = MutationSet(muts[: max(1, len(muts) - 1)])
                else:
                    sample = base

                concept_samples.append(sample)

            samples[idx] = concept_samples

        return samples


class AugmentedDataset(Dataset):
    """Dataset wrapper with set-theoretic augmentation.

    Wraps an existing dataset and applies augmentation on-the-fly.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        augmenter: SetAugmenter,
        mutation_key: str = "mutations",
        augment_prob: float = 0.5,
    ):
        """Initialize augmented dataset.

        Args:
            base_dataset: Base dataset to wrap
            augmenter: Set augmenter
            mutation_key: Key for mutations in sample dict
            augment_prob: Probability of applying augmentation
        """
        self.base_dataset = base_dataset
        self.augmenter = augmenter
        self.mutation_key = mutation_key
        self.augment_prob = augment_prob

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.base_dataset[idx]

        # Apply augmentation with probability
        if random.random() < self.augment_prob:
            mutations = sample.get(self.mutation_key)

            if isinstance(mutations, list):
                ms = MutationSet.from_strings(mutations)
            elif isinstance(mutations, MutationSet):
                ms = mutations
            else:
                return sample

            augmented = self.augmenter.augment(ms, n_augmentations=1)

            if augmented:
                sample = dict(sample)  # Copy
                sample[self.mutation_key] = augmented[0].to_list()
                sample["augmented"] = True

        return sample


class BalancedSampler:
    """Sample to balance resistance levels using augmentation.

    Ensures each resistance level is equally represented
    by generating synthetic samples for underrepresented levels.
    """

    def __init__(
        self,
        samples: list[MutationSet],
        lattice: ResistanceLattice,
    ):
        """Initialize balanced sampler.

        Args:
            samples: Original samples
            lattice: Resistance lattice
        """
        self.lattice = lattice

        # Group by level
        self.by_level: dict[ResistanceLevel, list[MutationSet]] = {level: [] for level in ResistanceLevel}

        for sample in samples:
            level = lattice.resistance_level(sample)
            self.by_level[level].append(sample)

        # Create augmenter
        self.augmenter = SetAugmenter(lattice=lattice)
        self.augmenter.add_samples(samples)

    def sample_balanced(
        self,
        n_per_level: int,
    ) -> list[tuple[MutationSet, ResistanceLevel]]:
        """Sample with balanced levels.

        Args:
            n_per_level: Samples per resistance level

        Returns:
            List of (sample, level) tuples
        """
        balanced = []

        for level in ResistanceLevel:
            level_samples = self.by_level[level]

            if len(level_samples) >= n_per_level:
                # Enough samples - random sample
                selected = random.sample(level_samples, n_per_level)
            else:
                # Need augmentation
                selected = list(level_samples)

                while len(selected) < n_per_level:
                    if level_samples:
                        # Augment existing
                        base = random.choice(level_samples)
                        augmented = self.augmenter.augment(base, 1)

                        if augmented:
                            aug = augmented[0]
                            # Check level is still correct
                            if self.lattice.resistance_level(aug) == level:
                                selected.append(aug)
                    else:
                        # Generate from scratch
                        generated = self.augmenter.generate_hierarchy_samples(level, n_per_level - len(selected))
                        selected.extend(generated)
                        break

            for sample in selected[:n_per_level]:
                balanced.append((sample, level))

        return balanced
