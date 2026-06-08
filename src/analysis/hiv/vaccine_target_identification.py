#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Vaccine Target Identification.

Multi-constraint optimization for identifying optimal HIV vaccine targets:
- Conservation analysis across subtypes
- Escape mutation resistance scoring
- Broadly neutralizing antibody epitope mapping
- HLA population coverage optimization

Usage:
    python src/scripts/hiv/analysis/vaccine_target_identification.py
    python src/scripts/hiv/analysis/vaccine_target_identification.py --output results/vaccine_targets
    python src/scripts/hiv/analysis/vaccine_target_identification.py --min-conservation 0.8
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def load_datasets_for_vaccine_analysis() -> dict:
    """Load datasets needed for vaccine target analysis."""
    datasets = {}

    try:
        from src.data.hiv import load_lanl_ctl

        datasets["ctl"] = load_lanl_ctl()
    except FileNotFoundError:
        pass

    try:
        from src.data.hiv import calculate_antibody_breadth, load_catnap

        datasets["catnap"] = load_catnap()
        datasets["bnab_breadth"] = calculate_antibody_breadth(datasets["catnap"])
    except FileNotFoundError:
        pass

    try:
        from src.data.hiv import load_stanford_hivdb

        datasets["stanford"] = load_stanford_hivdb("all")
    except FileNotFoundError:
        pass

    return datasets


def compute_epitope_conservation(ctl_df: pd.DataFrame) -> pd.DataFrame:
    """Compute conservation score for each epitope."""
    from src.biology.codons import AMINO_ACID_TO_CODONS, codon_to_index

    results = []

    for _, row in ctl_df.iterrows():
        epitope = row.get("Epitope", "")
        if not epitope or not isinstance(epitope, str):
            continue

        # Compute sequence-based conservation
        radial_positions = []
        for aa in epitope:
            codons = AMINO_ACID_TO_CODONS.get(aa, [])
            if codons:
                try:
                    idx = codon_to_index(codons[0])
                    # P-adic valuation indicates evolutionary constraint
                    valuation = 0
                    temp_idx = idx
                    while temp_idx > 0 and temp_idx % 3 == 0:
                        valuation += 1
                        temp_idx //= 3
                    radial_positions.append(valuation)
                except (KeyError, ValueError):
                    pass

        if radial_positions:
            conservation = np.mean(radial_positions) / 5.0  # Normalize
        else:
            conservation = 0.5

        results.append(
            {
                "Epitope": epitope,
                "Protein": row.get("Protein", ""),
                "HXB2_start": row.get("HXB2_start", ""),
                "HXB2_end": row.get("HXB2_end", ""),
                "HLA": row.get("HLA", ""),
                "conservation_score": conservation,
                "sequence_length": len(epitope),
            }
        )

    return pd.DataFrame(results)


def compute_hla_coverage(ctl_df: pd.DataFrame) -> pd.DataFrame:
    """Compute HLA population coverage for epitopes."""
    from src.data.hiv import parse_hla_restrictions

    # HLA frequencies in major populations (simplified)
    hla_frequencies = {
        "A*02": 0.25,
        "A*01": 0.15,
        "A*03": 0.12,
        "A*24": 0.10,
        "B*07": 0.12,
        "B*08": 0.08,
        "B*27": 0.05,
        "B*57": 0.03,
        "B*58": 0.03,
    }

    results = []

    for _, row in ctl_df.iterrows():
        hla_string = row.get("HLA", "")
        hla_list = parse_hla_restrictions(hla_string)

        coverage = 0.0
        matched_hlas = []
        for hla in hla_list:
            for hla_prefix, freq in hla_frequencies.items():
                if hla.startswith(hla_prefix) or hla_prefix in hla:
                    coverage += freq
                    matched_hlas.append(hla_prefix)
                    break

        # Cap at 1.0
        coverage = min(coverage, 1.0)

        results.append(
            {
                "Epitope": row.get("Epitope", ""),
                "HLA": hla_string,
                "matched_hlas": ",".join(set(matched_hlas)),
                "hla_coverage": coverage,
            }
        )

    return pd.DataFrame(results)


def compute_escape_resistance(ctl_df: pd.DataFrame, stanford_df: pd.DataFrame) -> pd.DataFrame:
    """Compute resistance to escape mutations based on drug resistance data."""
    from src.data.hiv import parse_mutation_list
    from src.data.hiv.position_mapper import protein_position_to_hxb2

    # Build mutation density map from resistance data
    mutation_density = defaultdict(int)
    for _, row in stanford_df.iterrows():
        drug_class = row.get("drug_class", "").lower()
        mutations = parse_mutation_list(row.get("CompMutList", ""))

        protein = {"pi": "pr", "nrti": "rt", "nnrti": "rt", "ini": "in"}.get(drug_class, "pr")

        for mut in mutations:
            try:
                hxb2_pos = protein_position_to_hxb2(protein, mut["position"])
                mutation_density[hxb2_pos] += 1
            except ValueError:
                pass

    # Score epitopes by mutation density in their region
    results = []
    for _, row in ctl_df.iterrows():
        start = row.get("HXB2_start")
        end = row.get("HXB2_end")

        if pd.isna(start) or pd.isna(end):
            continue

        total_mutations = sum(mutation_density[pos] for pos in range(int(start), int(end) + 1))

        # Low mutation density = high escape resistance
        escape_resistance = 1.0 / (1.0 + total_mutations * 0.1)

        results.append(
            {
                "Epitope": row.get("Epitope", ""),
                "HXB2_start": start,
                "HXB2_end": end,
                "mutation_count": total_mutations,
                "escape_resistance": escape_resistance,
            }
        )

    return pd.DataFrame(results)


def rank_vaccine_targets(
    conservation_df: pd.DataFrame,
    hla_df: pd.DataFrame,
    escape_df: pd.DataFrame,
    weights: dict = None,
) -> pd.DataFrame:
    """Rank vaccine targets using multi-objective optimization."""
    if weights is None:
        weights = {
            "conservation": 0.35,
            "hla_coverage": 0.35,
            "escape_resistance": 0.30,
        }

    # Merge all scores
    merged = conservation_df.merge(
        hla_df[["Epitope", "hla_coverage", "matched_hlas"]],
        on="Epitope",
        how="left",
    )

    if not escape_df.empty:
        merged = merged.merge(
            escape_df[["Epitope", "escape_resistance", "mutation_count"]],
            on="Epitope",
            how="left",
        )
    else:
        merged["escape_resistance"] = 0.5
        merged["mutation_count"] = 0

    # Fill missing values
    merged["hla_coverage"] = merged["hla_coverage"].fillna(0)
    merged["escape_resistance"] = merged["escape_resistance"].fillna(0.5)

    # Compute composite score
    merged["vaccine_score"] = (
        weights["conservation"] * merged["conservation_score"]
        + weights["hla_coverage"] * merged["hla_coverage"]
        + weights["escape_resistance"] * merged["escape_resistance"]
    )

    # Rank
    merged = merged.sort_values("vaccine_score", ascending=False).reset_index(drop=True)
    merged["rank"] = range(1, len(merged) + 1)

    return merged


def main():
    parser = argparse.ArgumentParser(description="Identify optimal vaccine targets")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "vaccine_targets",
        help="Output directory for results",
    )
    parser.add_argument(
        "--min-conservation",
        type=float,
        default=0.5,
        help="Minimum conservation score threshold",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.1,
        help="Minimum HLA population coverage",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50,
        help="Number of top targets to report",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Vaccine Target Identification")
    print("=" * 60)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("\nLoading datasets...")
    datasets = load_datasets_for_vaccine_analysis()

    if "ctl" not in datasets:
        print("Error: CTL epitope data required for vaccine target analysis")
        return 1

    ctl_df = datasets["ctl"]
    print(f"  CTL Epitopes: {len(ctl_df):,}")

    stanford_df = datasets.get("stanford", pd.DataFrame())
    if not stanford_df.empty:
        print(f"  Stanford HIVDB: {len(stanford_df):,}")

    # Compute conservation scores
    print("\nComputing epitope conservation...")
    conservation_df = compute_epitope_conservation(ctl_df)
    print(f"  Analyzed {len(conservation_df)} epitopes")

    # Compute HLA coverage
    print("\nComputing HLA population coverage...")
    hla_df = compute_hla_coverage(ctl_df)
    high_coverage = (hla_df["hla_coverage"] >= args.min_coverage).sum()
    print(f"  High coverage epitopes (>={args.min_coverage}): {high_coverage}")

    # Compute escape resistance
    print("\nComputing escape mutation resistance...")
    escape_df = pd.DataFrame()
    if not stanford_df.empty:
        escape_df = compute_escape_resistance(ctl_df, stanford_df)
        print(f"  Analyzed {len(escape_df)} epitope-resistance pairs")

    # Rank vaccine targets
    print("\nRanking vaccine targets...")
    ranked = rank_vaccine_targets(conservation_df, hla_df, escape_df)

    # Apply filters
    filtered = ranked[
        (ranked["conservation_score"] >= args.min_conservation) & (ranked["hla_coverage"] >= args.min_coverage)
    ].copy()

    print(f"  Total candidates: {len(filtered)}")

    # Print top targets
    print(f"\nTop {min(args.top_n, len(filtered))} Vaccine Targets:")
    print("-" * 80)

    for _, row in filtered.head(args.top_n).iterrows():
        print(
            f"  #{row['rank']:3d} | {row['Epitope'][:15]:15s} | "
            f"{row['Protein']:6s} | "
            f"cons={row['conservation_score']:.3f} | "
            f"HLA={row['hla_coverage']:.2f} | "
            f"escape={row['escape_resistance']:.3f} | "
            f"score={row['vaccine_score']:.3f}"
        )

    # Save results
    print(f"\nSaving results to {args.output}...")

    conservation_df.to_csv(args.output / "conservation_scores.csv", index=False)
    hla_df.to_csv(args.output / "hla_coverage.csv", index=False)

    if not escape_df.empty:
        escape_df.to_csv(args.output / "escape_resistance.csv", index=False)

    ranked.to_csv(args.output / "ranked_targets.csv", index=False)
    filtered.head(args.top_n).to_csv(args.output / "top_targets.csv", index=False)

    # Save summary
    summary = {
        "total_epitopes_analyzed": len(conservation_df),
        "high_conservation_count": (conservation_df["conservation_score"] >= args.min_conservation).sum(),
        "high_coverage_count": high_coverage,
        "final_candidates": len(filtered),
        "top_target_epitope": filtered.iloc[0]["Epitope"] if not filtered.empty else "",
        "top_target_score": float(filtered.iloc[0]["vaccine_score"]) if not filtered.empty else 0,
    }
    pd.DataFrame([summary]).to_csv(args.output / "summary.csv", index=False)

    print("\nAnalysis complete!")
    print(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
