# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Utilities for generating synthetic datasets with proper correlation.

This module provides functions to generate synthetic datasets where
the target variable is properly correlated with features, enabling
meaningful model evaluation even without real clinical data.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def generate_correlated_targets(
    X: np.ndarray,
    signal_strength: float = 0.5,
    noise_level: float = 0.2,
    n_causal_fraction: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Generate targets that correlate with features.

    Uses a subset of features as "causal" features and generates
    targets as a linear combination with added noise.

    Args:
        X: Feature matrix (n_samples, n_features)
        signal_strength: Magnitude of feature weights
        noise_level: Standard deviation of added noise
        n_causal_fraction: Fraction of features to use as causal
        seed: Random seed for reproducibility

    Returns:
        Target array (n_samples,) with values in [0, 1]
    """
    np.random.seed(seed)
    n_samples, n_features = X.shape

    # Select subset of features as "causal"
    n_causal = max(1, int(n_features * n_causal_fraction))
    causal_idx = np.random.choice(n_features, n_causal, replace=False)

    # Generate weights for causal features
    weights = np.zeros(n_features)
    weights[causal_idx] = np.random.randn(n_causal) * signal_strength

    # Compute targets
    y = np.dot(X, weights)
    y += np.random.normal(0, noise_level, n_samples)

    # Normalize to [0, 1]
    y_min, y_max = y.min(), y.max()
    y = (y - y_min) / (y_max - y_min) if y_max > y_min else np.full(n_samples, 0.5)

    return y.astype(np.float32)


def augment_synthetic_dataset(
    X: np.ndarray,
    y: np.ndarray,
    n_augment: int = 100,
    mutation_rate: float = 0.05,
    noise_scale: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment a small dataset by creating noisy variations.

    For each sample, creates variations by:
    1. Adding small random noise to features
    2. Slightly perturbing the target value
    3. Randomly flipping some feature values (simulating mutations)

    Args:
        X: Original feature matrix (n_samples, n_features)
        y: Original target array (n_samples,)
        n_augment: Number of augmented samples to create
        mutation_rate: Fraction of features to mutate per sample
        noise_scale: Scale of noise to add to targets
        seed: Random seed

    Returns:
        (X_augmented, y_augmented) with n_samples + n_augment samples
    """
    np.random.seed(seed)
    n_samples, n_features = X.shape

    # Select random base samples to augment from
    base_indices = np.random.choice(n_samples, n_augment, replace=True)

    X_new = []
    y_new = []

    for i in base_indices:
        x_base = X[i].copy()
        y_base = y[i]

        # Mutate some features (flip 0->1 or 1->0 for one-hot)
        n_mutate = max(1, int(n_features * mutation_rate))
        mutate_idx = np.random.choice(n_features, n_mutate, replace=False)

        for idx in mutate_idx:
            # For one-hot encoded features, this simulates mutations
            if x_base[idx] > 0.5:
                x_base[idx] = 0.0
            else:
                x_base[idx] = 1.0

        # Add small noise to target (but keep in [0,1])
        y_augmented = y_base + np.random.normal(0, noise_scale)
        y_augmented = np.clip(y_augmented, 0, 1)

        X_new.append(x_base)
        y_new.append(y_augmented)

    # Combine original and augmented
    X_combined = np.vstack([X, np.array(X_new)])
    y_combined = np.concatenate([y, np.array(y_new)])

    return X_combined.astype(np.float32), y_combined.astype(np.float32)


def create_mutation_based_dataset(
    reference_sequence: str,
    mutation_db: dict,
    encode_fn: Callable[[str, int], np.ndarray],
    max_length: int = 500,
    n_random_mutants: int = 50,
    effect_scores: dict = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create a dataset from a mutation database.

    Generates:
    1. Wild-type (reference) sequence
    2. All single mutants from the mutation database
    3. Random combinations of mutations

    Args:
        reference_sequence: Reference protein sequence
        mutation_db: Dictionary mapping positions to mutation info
            Format: {pos: {ref_aa: {mutations: [...], effect: ...}}}
        encode_fn: Function to encode sequence (seq, max_len) -> np.ndarray
        max_length: Maximum sequence length for encoding
        n_random_mutants: Number of random mutant combinations to generate
        effect_scores: Mapping of effect labels to numeric scores
        seed: Random seed

    Returns:
        (X, y, ids) tuple
    """
    np.random.seed(seed)

    if effect_scores is None:
        effect_scores = {"high": 0.9, "moderate": 0.5, "low": 0.2}

    sequences = [reference_sequence]
    resistances = [0.0]
    ids = ["WT"]

    # Generate all single mutants
    all_mutations = []
    for pos, info in mutation_db.items():
        if pos <= len(reference_sequence):
            ref_aa = list(info.keys())[0]
            if "/" in ref_aa:
                ref_aa = ref_aa.split("/")[0]

            for mut_aa in info[list(info.keys())[0]].get("mutations", []):
                mutant = list(reference_sequence)
                if pos - 1 < len(mutant):
                    mutant[pos - 1] = mut_aa
                    mutant_seq = "".join(mutant)

                    effect = info[list(info.keys())[0]].get("effect", "moderate")
                    score = effect_scores.get(effect, 0.3)

                    sequences.append(mutant_seq)
                    resistances.append(score)
                    ids.append(f"{ref_aa}{pos}{mut_aa}")

                    all_mutations.append((pos, ref_aa, mut_aa, score))

    # Generate random combinations of mutations
    if all_mutations and n_random_mutants > 0:
        for _i in range(n_random_mutants):
            # Pick 2-4 mutations
            n_muts = np.random.randint(2, min(5, len(all_mutations) + 1))
            selected = np.random.choice(len(all_mutations), n_muts, replace=False)

            mutant = list(reference_sequence)
            total_score = 0.0
            mut_ids = []

            for idx in selected:
                pos, ref_aa, mut_aa, score = all_mutations[idx]
                if pos - 1 < len(mutant):
                    mutant[pos - 1] = mut_aa
                    total_score += score
                    mut_ids.append(f"{ref_aa}{pos}{mut_aa}")

            # Cap score at 1.0
            combined_score = min(total_score / len(selected) + 0.1 * len(selected), 1.0)

            sequences.append("".join(mutant))
            resistances.append(combined_score)
            ids.append("+".join(mut_ids))

    # Encode all sequences
    X = np.array([encode_fn(s, max_length) for s in sequences])
    y = np.array(resistances, dtype=np.float32)

    return X, y, ids


def ensure_minimum_samples(
    X: np.ndarray,
    y: np.ndarray,
    ids: list[str],
    min_samples: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Ensure dataset has minimum number of samples by augmentation.

    If the dataset has fewer than min_samples, augments to reach that number.

    Args:
        X: Feature matrix
        y: Target array
        ids: Sample identifiers
        min_samples: Minimum number of samples required
        seed: Random seed

    Returns:
        (X, y, ids) with at least min_samples samples
    """
    n_samples = X.shape[0]

    if n_samples >= min_samples:
        return X, y, ids

    n_augment = min_samples - n_samples
    X_aug, y_aug = augment_synthetic_dataset(X, y, n_augment=n_augment, seed=seed)

    # Generate IDs for augmented samples
    ids_aug = ids.copy()
    for i in range(n_augment):
        base_id = ids[i % n_samples]
        ids_aug.append(f"{base_id}_aug{i}")

    return X_aug, y_aug, ids_aug
