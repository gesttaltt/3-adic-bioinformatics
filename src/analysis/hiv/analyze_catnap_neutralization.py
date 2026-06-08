#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""CATNAP Neutralization Analysis.

Analyzes 189,879 antibody-virus neutralization records:
- bnAb sensitivity geometric signatures
- Neutralization breadth vs epitope centrality
- Escape pathway prediction
- Antibody potency clustering

Usage:
    python src/scripts/hiv/analysis/analyze_catnap_neutralization.py
    python src/scripts/hiv/analysis/analyze_catnap_neutralization.py --antibody VRC01
    python src/scripts/hiv/analysis/analyze_catnap_neutralization.py --top-antibodies 20
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

from src.data.hiv import (
    calculate_antibody_breadth,
    classify_antibody,
    get_bnab_classes,
    get_catnap_by_antibody,
    get_catnap_resistant_viruses,
    get_catnap_sensitive_viruses,
    load_catnap,
)


def compute_potency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute potency metrics for each antibody."""
    results = []

    for antibody in df["Antibody"].unique():
        ab_data = df[df["Antibody"] == antibody]
        valid_ic50 = ab_data["IC50_numeric"].dropna()

        if len(valid_ic50) < 5:
            continue

        # Compute metrics
        metrics = {
            "Antibody": antibody,
            "n_tested": len(ab_data),
            "n_valid": len(valid_ic50),
            "median_ic50": valid_ic50.median(),
            "mean_ic50": valid_ic50.mean(),
            "std_ic50": valid_ic50.std(),
            "min_ic50": valid_ic50.min(),
            "max_ic50": valid_ic50.max(),
            "geometric_mean_ic50": np.exp(np.log(valid_ic50 + 1e-10).mean()),
            "epitope_class": classify_antibody(antibody) or "Unknown",
        }

        # Potency score (lower IC50 = higher potency)
        metrics["potency_score"] = 1.0 / (1.0 + metrics["median_ic50"])

        results.append(metrics)

    return pd.DataFrame(results).sort_values("potency_score", ascending=False)


def analyze_sensitivity_signatures(df: pd.DataFrame, ic50_threshold: float = 1.0) -> pd.DataFrame:
    """Analyze geometric signatures of antibody sensitivity."""
    results = []

    for antibody in df["Antibody"].unique():
        sensitive = get_catnap_sensitive_viruses(df, antibody, ic50_threshold)
        resistant = get_catnap_resistant_viruses(df, antibody, ic50_threshold * 50)

        if len(sensitive) < 5 or len(resistant) < 5:
            continue

        # Compute separation metrics
        sensitive_ic50 = sensitive["IC50_numeric"].dropna()
        resistant_ic50 = resistant["IC50_numeric"].dropna()

        if len(sensitive_ic50) == 0 or len(resistant_ic50) == 0:
            continue

        # Log-scale separation
        log_sensitive = np.log10(sensitive_ic50 + 1e-10)
        log_resistant = np.log10(resistant_ic50 + 1e-10)

        separation = log_resistant.mean() - log_sensitive.mean()
        pooled_std = np.sqrt((log_sensitive.var() + log_resistant.var()) / 2)

        results.append(
            {
                "Antibody": antibody,
                "n_sensitive": len(sensitive),
                "n_resistant": len(resistant),
                "mean_sensitive_ic50": sensitive_ic50.mean(),
                "mean_resistant_ic50": resistant_ic50.mean(),
                "log_separation": separation,
                "effect_size": separation / pooled_std if pooled_std > 0 else 0,
                "epitope_class": classify_antibody(antibody) or "Unknown",
            }
        )

    return pd.DataFrame(results).sort_values("effect_size", ascending=False)


def cluster_antibodies_by_profile(potency_df: pd.DataFrame) -> pd.DataFrame:
    """Cluster antibodies by their neutralization profiles."""
    if potency_df.empty or len(potency_df) < 3:
        return potency_df

    # Use simple clustering based on epitope class and potency
    potency_df = potency_df.copy()

    # Assign cluster based on epitope class
    class_to_cluster = {cls: i for i, cls in enumerate(get_bnab_classes().keys())}
    potency_df["cluster"] = potency_df["epitope_class"].map(lambda x: class_to_cluster.get(x, -1))

    # Sub-cluster by potency within each class
    potency_df["potency_rank"] = potency_df.groupby("cluster")["potency_score"].rank(ascending=False)

    return potency_df


def analyze_breadth_centrality(breadth_df: pd.DataFrame, potency_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze relationship between breadth and potency."""
    if breadth_df.empty or potency_df.empty:
        return pd.DataFrame()

    # Merge breadth and potency data
    merged = breadth_df.merge(
        potency_df[["Antibody", "median_ic50", "potency_score", "epitope_class"]],
        on="Antibody",
        how="inner",
    )

    if len(merged) < 5:
        return merged

    # Compute breadth-potency correlation
    from scipy.stats import spearmanr

    rho, p_value = spearmanr(merged["breadth_pct"], merged["potency_score"])

    merged["breadth_potency_corr"] = rho
    merged["breadth_potency_pvalue"] = p_value

    return merged


def main():
    parser = argparse.ArgumentParser(description="Analyze CATNAP neutralization data")
    parser.add_argument(
        "--antibody",
        type=str,
        default=None,
        help="Filter to specific antibody (e.g., VRC01)",
    )
    parser.add_argument(
        "--top-antibodies",
        type=int,
        default=50,
        help="Number of top antibodies to analyze",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "catnap_neutralization",
        help="Output directory for results",
    )
    parser.add_argument(
        "--ic50-threshold",
        type=float,
        default=1.0,
        help="IC50 threshold for sensitivity (ug/mL)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CATNAP Neutralization Analysis")
    print("=" * 60)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading CATNAP data...")
    try:
        df = load_catnap()
        print(f"Loaded {len(df):,} neutralization records")
        print(f"  Unique antibodies: {df['Antibody'].nunique():,}")
        print(f"  Unique viruses: {df['Virus'].nunique():,}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Filter to specific antibody if requested
    if args.antibody:
        df = get_catnap_by_antibody(df, args.antibody)
        print(f"Filtered to {len(df):,} records for {args.antibody}")

    # Calculate antibody breadth
    print("\nCalculating antibody breadth...")
    breadth_df = calculate_antibody_breadth(df)
    print(f"  Analyzed {len(breadth_df)} antibodies")

    if not breadth_df.empty:
        top_5 = breadth_df.head(5)
        print("\nTop 5 Broadest Antibodies:")
        for _, row in top_5.iterrows():
            print(f"  {row['Antibody']}: {row['breadth_pct']:.1f}% ({row['n_neutralized']}/{row['n_tested']})")

    # Compute potency metrics
    print("\nComputing potency metrics...")
    potency_df = compute_potency_metrics(df)
    print(f"  Analyzed {len(potency_df)} antibodies")

    if not potency_df.empty:
        top_potent = potency_df.head(5)
        print("\nTop 5 Most Potent Antibodies:")
        for _, row in top_potent.iterrows():
            print(f"  {row['Antibody']}: median IC50 = {row['median_ic50']:.3f} ug/mL ({row['epitope_class']})")

    # Analyze sensitivity signatures
    print("\nAnalyzing sensitivity signatures...")
    signatures = analyze_sensitivity_signatures(df, args.ic50_threshold)
    print(f"  Found {len(signatures)} antibodies with clear signatures")

    # Cluster antibodies
    print("\nClustering antibodies by profile...")
    clustered = cluster_antibodies_by_profile(potency_df)

    # Analyze breadth-centrality relationship
    print("\nAnalyzing breadth-potency relationship...")
    breadth_centrality = analyze_breadth_centrality(breadth_df, potency_df)

    if not breadth_centrality.empty and "breadth_potency_corr" in breadth_centrality.columns:
        corr = breadth_centrality["breadth_potency_corr"].iloc[0]
        pval = breadth_centrality["breadth_potency_pvalue"].iloc[0]
        print(f"  Breadth-Potency correlation: rho = {corr:.4f} (p = {pval:.4e})")

    # Save results
    print(f"\nSaving results to {args.output}...")
    breadth_df.to_csv(args.output / "antibody_breadth.csv", index=False)
    potency_df.to_csv(args.output / "antibody_potency.csv", index=False)
    signatures.to_csv(args.output / "sensitivity_signatures.csv", index=False)
    clustered.to_csv(args.output / "antibody_clusters.csv", index=False)

    if not breadth_centrality.empty:
        breadth_centrality.to_csv(args.output / "breadth_centrality.csv", index=False)

    # Save summary
    summary = {
        "total_records": len(df),
        "unique_antibodies": df["Antibody"].nunique(),
        "unique_viruses": df["Virus"].nunique(),
        "antibodies_analyzed": len(potency_df),
        "antibodies_with_signatures": len(signatures),
    }
    pd.DataFrame([summary]).to_csv(args.output / "summary.csv", index=False)

    print("\nAnalysis complete!")
    print(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
