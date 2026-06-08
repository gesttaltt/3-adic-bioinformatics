"""Temporal validation for drug resistance prediction.

Random train/test splits may leak information when similar sequences
appear in both sets. Temporal validation simulates real-world use
by training on historical data and testing on future data.

This provides a more realistic estimate of clinical utility.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class TemporalSplit:
    """Result of temporal split."""

    train_indices: np.ndarray
    test_indices: np.ndarray
    train_dates: pd.Series
    test_dates: pd.Series
    cutoff_date: datetime
    n_train: int
    n_test: int


def parse_date(date_str: str) -> datetime | None:
    """Parse date string to datetime."""
    if pd.isna(date_str):
        return None

    # Try various formats
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y",
        "%m/%d/%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(str(date_str), fmt)
        except ValueError:
            continue

    return None


def temporal_split(
    df: pd.DataFrame,
    date_column: str = "IsolateDate",
    test_year: int = 2020,
    fallback_random: bool = True,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TemporalSplit:
    """Split data temporally.

    Args:
        df: DataFrame with date column
        date_column: Name of date column
        test_year: Year to use as cutoff (test = this year and later)
        fallback_random: If dates unavailable, use random split
        test_size: Fraction for test if using random fallback
        random_state: Random seed for fallback

    Returns:
        TemporalSplit with train/test indices
    """
    # Check if date column exists
    if date_column not in df.columns:
        if fallback_random:
            print(f"  Warning: {date_column} not found, using random split")
            return random_split_as_temporal(df, test_size, random_state)
        else:
            raise ValueError(f"Date column {date_column} not found")

    # Parse dates
    dates = df[date_column].apply(parse_date)
    valid_dates = dates.notna()

    if valid_dates.sum() < len(df) * 0.5 and fallback_random:
        print(f"  Warning: Only {valid_dates.sum()}/{len(df)} valid dates, using random split")
        return random_split_as_temporal(df, test_size, random_state)

    # Split by cutoff date
    cutoff = datetime(test_year, 1, 1)

    train_mask = (dates < cutoff) | dates.isna()  # Include missing dates in train
    test_mask = dates >= cutoff

    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]

    # Ensure we have enough test data
    if len(test_indices) < 50:
        print(f"  Warning: Only {len(test_indices)} test samples, adjusting cutoff")
        # Try earlier cutoff
        for year in range(test_year - 1, test_year - 5, -1):
            cutoff = datetime(year, 1, 1)
            test_mask = dates >= cutoff
            test_indices = np.where(test_mask)[0]
            if len(test_indices) >= 50:
                break

    train_indices = np.where(~test_mask)[0]

    return TemporalSplit(
        train_indices=train_indices,
        test_indices=test_indices,
        train_dates=dates.iloc[train_indices],
        test_dates=dates.iloc[test_indices],
        cutoff_date=cutoff,
        n_train=len(train_indices),
        n_test=len(test_indices),
    )


def random_split_as_temporal(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TemporalSplit:
    """Create random split in TemporalSplit format."""
    n = len(df)
    np.random.seed(random_state)
    indices = np.random.permutation(n)

    n_test = int(n * test_size)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return TemporalSplit(
        train_indices=train_indices,
        test_indices=test_indices,
        train_dates=pd.Series([None] * len(train_indices)),
        test_dates=pd.Series([None] * len(test_indices)),
        cutoff_date=datetime.now(),
        n_train=len(train_indices),
        n_test=len(test_indices),
    )


def sequence_similarity_split(
    sequences: np.ndarray,
    test_size: float = 0.2,
    min_distance: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Split ensuring minimum sequence distance between train and test.

    This prevents data leakage from near-identical sequences.

    Args:
        sequences: One-hot encoded sequences (n, features)
        test_size: Fraction for test set
        min_distance: Minimum Hamming distance (fraction)
        random_state: Random seed

    Returns:
        (train_indices, test_indices)
    """
    n = len(sequences)
    n_test = int(n * test_size)

    np.random.seed(random_state)

    # Start with random test candidates
    test_indices = list(np.random.choice(n, size=min(n_test * 2, n), replace=False))
    train_indices = [i for i in range(n) if i not in test_indices]

    # Compute pairwise distances (approximate for efficiency)
    def hamming_distance(a: np.ndarray, b: np.ndarray) -> float:
        return np.mean(a != b)

    # Filter test set to ensure minimum distance
    final_test = []
    for test_idx in test_indices:
        if len(final_test) >= n_test:
            break

        # Check distance to all train samples (sample for efficiency)
        sample_train = np.random.choice(train_indices, size=min(100, len(train_indices)), replace=False)
        distances = [hamming_distance(sequences[test_idx], sequences[t]) for t in sample_train]

        if min(distances) >= min_distance:
            final_test.append(test_idx)
        else:
            train_indices.append(test_idx)

    # If not enough test samples, relax constraint
    if len(final_test) < n_test // 2:
        print(f"  Warning: Only {len(final_test)} samples meet distance threshold, relaxing")
        remaining = [i for i in test_indices if i not in final_test and i not in train_indices]
        final_test.extend(remaining[: n_test - len(final_test)])

    # Update train indices
    train_indices = [i for i in range(n) if i not in final_test]

    return np.array(train_indices), np.array(final_test)


def cross_validation_temporal(
    df: pd.DataFrame,
    date_column: str = "IsolateDate",
    n_splits: int = 5,
    gap_years: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-series cross-validation with gap.

    Creates expanding window splits where training always
    precedes testing, with a gap between them.

    Args:
        df: DataFrame with date column
        date_column: Name of date column
        n_splits: Number of CV folds
        gap_years: Gap between train and test

    Returns:
        List of (train_indices, test_indices) tuples
    """
    dates = df[date_column].apply(parse_date)
    valid_mask = dates.notna()

    if valid_mask.sum() < 100:
        print("  Warning: Not enough dated samples for temporal CV")
        return []

    valid_dates = dates[valid_mask]
    min_date = valid_dates.min()
    max_date = valid_dates.max()

    total_years = (max_date - min_date).days / 365
    if total_years < n_splits + gap_years:
        n_splits = max(2, int(total_years) - gap_years)

    splits = []
    for i in range(n_splits):
        # Each split uses progressively more training data
        train_end_year = min_date.year + int((i + 1) * (total_years - gap_years) / n_splits)
        test_start_year = train_end_year + gap_years

        train_cutoff = datetime(train_end_year, 12, 31)
        test_start = datetime(test_start_year, 1, 1)

        train_mask = dates <= train_cutoff
        test_mask = dates >= test_start

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        if len(train_indices) > 50 and len(test_indices) > 20:
            splits.append((train_indices, test_indices))

    return splits


def analyze_temporal_distribution(
    df: pd.DataFrame,
    date_column: str = "IsolateDate",
) -> dict:
    """Analyze temporal distribution of data.

    Returns:
        Dict with temporal statistics
    """
    dates = df[date_column].apply(parse_date)
    valid_mask = dates.notna()

    if valid_mask.sum() == 0:
        return {"error": "No valid dates found"}

    valid_dates = dates[valid_mask]

    # Year distribution
    years = valid_dates.apply(lambda x: x.year)
    year_counts = years.value_counts().sort_index()

    return {
        "n_total": len(df),
        "n_dated": valid_mask.sum(),
        "pct_dated": valid_mask.mean() * 100,
        "min_date": valid_dates.min(),
        "max_date": valid_dates.max(),
        "date_range_years": (valid_dates.max() - valid_dates.min()).days / 365,
        "samples_per_year": year_counts.to_dict(),
        "median_year": int(years.median()),
    }


if __name__ == "__main__":
    print("Testing Temporal Validation")
    print("=" * 60)

    # Create sample data
    n = 1000
    np.random.seed(42)

    # Simulate dates from 2010 to 2023
    dates = pd.to_datetime(np.random.choice(pd.date_range("2010-01-01", "2023-12-31"), n))
    df = pd.DataFrame({"IsolateDate": dates, "value": np.random.randn(n)})

    # Test temporal split
    split = temporal_split(df, test_year=2020)
    print("Temporal split:")
    print(f"  Train: {split.n_train}, Test: {split.n_test}")
    print(f"  Cutoff: {split.cutoff_date}")

    # Test distribution analysis
    analysis = analyze_temporal_distribution(df)
    print("\nTemporal distribution:")
    print(f"  Date range: {analysis['min_date']} to {analysis['max_date']}")
    print(f"  Median year: {analysis['median_year']}")

    # Test CV
    cv_splits = cross_validation_temporal(df, n_splits=3)
    print(f"\nTemporal CV splits: {len(cv_splits)}")
    for i, (train, test) in enumerate(cv_splits):
        print(f"  Split {i + 1}: {len(train)} train, {len(test)} test")

    print("\n" + "=" * 60)
    print("Temporal validation working!")
