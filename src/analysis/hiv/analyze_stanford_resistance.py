#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Stanford HIVDB Drug Resistance Analysis.

Analyzes 7,154 drug resistance records using p-adic hyperbolic geometry:
- Fold-change vs hyperbolic distance correlation
- Primary vs accessory mutation classification
- Cross-resistance pattern mapping
- Drug class-specific geometric signatures

Usage:
    python src/scripts/hiv/analysis/analyze_stanford_resistance.py
    python src/scripts/hiv/analysis/analyze_stanford_resistance.py --drug-class PI
    python src/scripts/hiv/analysis/analyze_stanford_resistance.py --output results/stanford_resistance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.biology.codons import AMINO_ACID_TO_CODONS, codon_to_index
from src.data.hiv import (
    get_stanford_drug_columns,
    load_stanford_hivdb,
    parse_mutation_list,
)


def compute_hyperbolic_distance(codon1: str, codon2: str, curvature: float = 1.0) -> float:
    """
    Compute hyperbolic distance between two codons in Poincare ball.

    Uses p-adic valuation to determine radial position.
    """
    if codon1 == codon2:
        return 0.0

    try:
        idx1 = codon_to_index(codon1)
        idx2 = codon_to_index(codon2)
    except (KeyError, ValueError):
        return float("nan")

    # Compute p-adic distance (3-adic)
    diff = abs(idx1 - idx2)
    if diff == 0:
        return 0.0

    v = 0
    while diff % 3 == 0:
        v += 1
        diff //= 3

    padic_dist = 3.0 ** (-v)

    # Map to hyperbolic distance
    # Higher p-adic valuation -> closer to origin -> smaller hyperbolic distance
    return np.arctanh(min(padic_dist, 0.99)) * 2 / curvature


def analyze_mutation_distances(df: pd.DataFrame, drug_class: str) -> pd.DataFrame:
    """Analyze hyperbolic distances for mutations in a drug class."""
    results = []

    for _, row in df.iterrows():
        mutations = parse_mutation_list(row.get("CompMutList", ""))

        for mut in mutations:
            wt = mut["wild_type"]
            mutant = mut["mutant"]
            position = mut["position"]

            # Get representative codons for amino acids
            wt_codons = AMINO_ACID_TO_CODONS.get(wt, [])
            mut_codons = AMINO_ACID_TO_CODONS.get(mutant, [])

            if wt_codons and mut_codons:
                # Use first codon as representative
                distance = compute_hyperbolic_distance(wt_codons[0], mut_codons[0])

                # Get fold-change for each drug
                for drug in get_stanford_drug_columns(drug_class):
                    if drug in row:
                        fold_change = row[drug]
                        if pd.notna(fold_change) and fold_change > 0:
                            results.append(
                                {
                                    "position": position,
                                    "wild_type": wt,
                                    "mutant": mutant,
                                    "drug": drug,
                                    "fold_change": fold_change,
                                    "hyperbolic_distance": distance,
                                    "log_fold_change": np.log10(fold_change) if fold_change > 0 else 0,
                                }
                            )

    return pd.DataFrame(results)


def classify_mutations(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Classify mutations as primary or accessory based on resistance level."""
    if df.empty:
        return df

    df = df.copy()

    # Primary mutations: cause > threshold fold-change
    df["is_primary"] = df["fold_change"] >= threshold

    # Calculate geometric features
    df["radial_position"] = df["hyperbolic_distance"].apply(lambda d: 1 - np.exp(-d) if pd.notna(d) else np.nan)

    return df


def compute_cross_resistance(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-resistance patterns between drugs."""
    if df.empty:
        return pd.DataFrame()

    # Group by mutation
    mutation_drugs = (
        df.groupby(["position", "wild_type", "mutant"])
        .agg(
            {
                "drug": lambda x: list(x.unique()),
                "fold_change": "mean",
                "hyperbolic_distance": "first",
            }
        )
        .reset_index()
    )

    mutation_drugs["n_drugs_affected"] = mutation_drugs["drug"].apply(len)
    mutation_drugs["is_cross_resistant"] = mutation_drugs["n_drugs_affected"] > 1

    return mutation_drugs


def main():
    parser = argparse.ArgumentParser(description="Analyze Stanford HIVDB drug resistance data")
    parser.add_argument(
        "--drug-class",
        type=str,
        default="all",
        choices=["pi", "nrti", "nnrti", "ini", "all"],
        help="Drug class to analyze",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "stanford_resistance",
        help="Output directory for results",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="Fold-change threshold for primary mutations",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Stanford HIVDB Drug Resistance Analysis")
    print("=" * 60)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\nLoading {args.drug_class.upper()} data...")
    try:
        df = load_stanford_hivdb(args.drug_class)
        print(f"Loaded {len(df):,} records")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run: ternary-vae data download to fetch the datasets")
        return 1

    # Analyze by drug class
    drug_classes = ["pi", "nrti", "nnrti", "ini"] if args.drug_class == "all" else [args.drug_class]

    all_results = []
    for dc in drug_classes:
        print(f"\nAnalyzing {dc.upper()}...")
        dc_df = df[df["drug_class"] == dc.upper()] if args.drug_class == "all" else df

        # Analyze mutation distances
        distances = analyze_mutation_distances(dc_df, dc)
        if not distances.empty:
            distances["drug_class"] = dc.upper()
            all_results.append(distances)
            print(f"  Found {len(distances):,} mutation-drug pairs")

    if all_results:
        results_df = pd.concat(all_results, ignore_index=True)

        # Classify mutations
        print("\nClassifying mutations...")
        results_df = classify_mutations(results_df, args.threshold)
        n_primary = results_df["is_primary"].sum()
        print(f"  Primary mutations: {n_primary:,}")
        print(f"  Accessory mutations: {len(results_df) - n_primary:,}")

        # Compute cross-resistance
        print("\nComputing cross-resistance patterns...")
        cross_resistance = compute_cross_resistance(results_df)
        n_cross = cross_resistance["is_cross_resistant"].sum() if not cross_resistance.empty else 0
        print(f"  Cross-resistant mutations: {n_cross:,}")

        # Compute correlation
        valid = results_df.dropna(subset=["hyperbolic_distance", "log_fold_change"])
        if len(valid) > 10:
            from scipy.stats import pearsonr, spearmanr

            r_pearson, p_pearson = pearsonr(valid["hyperbolic_distance"], valid["log_fold_change"])
            r_spearman, p_spearman = spearmanr(valid["hyperbolic_distance"], valid["log_fold_change"])
            print("\nDistance-Resistance Correlation:")
            print(f"  Pearson r = {r_pearson:.4f} (p = {p_pearson:.4e})")
            print(f"  Spearman rho = {r_spearman:.4f} (p = {p_spearman:.4e})")

        # Save results
        print(f"\nSaving results to {args.output}...")
        results_df.to_csv(args.output / "mutation_distances.csv", index=False)
        cross_resistance.to_csv(args.output / "cross_resistance.csv", index=False)

        # Save summary
        summary = {
            "total_records": len(df),
            "mutation_drug_pairs": len(results_df),
            "primary_mutations": int(n_primary),
            "cross_resistant_mutations": int(n_cross),
        }
        pd.DataFrame([summary]).to_csv(args.output / "summary.csv", index=False)

        print("\nAnalysis complete!")
        print(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
