#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Cross-Dataset Integration Analysis.

Integrates multiple HIV datasets for combined analyses:
- Drug resistance vs immune escape trade-offs
- Multi-pressure constraint mapping
- Universal vaccine target identification
- Geographic correlation analysis

Usage:
    python src/scripts/hiv/analysis/cross_dataset_integration.py
    python src/scripts/hiv/analysis/cross_dataset_integration.py --output results/integration
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


def load_all_datasets() -> dict:
    """Load all available HIV datasets."""
    datasets = {}

    # Stanford HIVDB
    try:
        from src.data.hiv import load_stanford_hivdb

        datasets["stanford"] = load_stanford_hivdb("all")
        print(f"  Stanford HIVDB: {len(datasets['stanford']):,} records")
    except FileNotFoundError:
        print("  Stanford HIVDB: Not available")

    # CTL Epitopes
    try:
        from src.data.hiv import load_lanl_ctl

        datasets["ctl"] = load_lanl_ctl()
        print(f"  CTL Epitopes: {len(datasets['ctl']):,} records")
    except FileNotFoundError:
        print("  CTL Epitopes: Not available")

    # CATNAP
    try:
        from src.data.hiv import load_catnap

        datasets["catnap"] = load_catnap()
        print(f"  CATNAP: {len(datasets['catnap']):,} records")
    except FileNotFoundError:
        print("  CATNAP: Not available")

    return datasets


def find_overlapping_positions(stanford_df: pd.DataFrame, ctl_df: pd.DataFrame) -> pd.DataFrame:
    """Find positions that appear in both drug resistance and CTL epitope data."""
    from src.data.hiv import parse_mutation_list
    from src.data.hiv.position_mapper import protein_position_to_hxb2

    # Extract resistance positions
    resistance_positions = defaultdict(list)
    for _, row in stanford_df.iterrows():
        drug_class = row.get("drug_class", "").lower()
        mutations = parse_mutation_list(row.get("CompMutList", ""))

        for mut in mutations:
            pos = mut["position"]
            # Map to HXB2 coordinates based on drug class
            protein = {"pi": "pr", "nrti": "rt", "nnrti": "rt", "ini": "in"}.get(drug_class, "pr")
            try:
                hxb2_pos = protein_position_to_hxb2(protein, pos)
                resistance_positions[hxb2_pos].append(
                    {
                        "drug_class": drug_class,
                        "mutation": f"{mut['wild_type']}{pos}{mut['mutant']}",
                    }
                )
            except ValueError:
                pass

    # Extract epitope positions
    epitope_positions = defaultdict(list)
    for _, row in ctl_df.iterrows():
        start = row.get("HXB2_start")
        end = row.get("HXB2_end")
        if pd.notna(start) and pd.notna(end):
            for pos in range(int(start), int(end) + 1):
                epitope_positions[pos].append(
                    {
                        "epitope": row.get("Epitope", ""),
                        "protein": row.get("Protein", ""),
                        "hla": row.get("HLA", ""),
                    }
                )

    # Find overlap
    overlapping = []
    for pos in set(resistance_positions.keys()) & set(epitope_positions.keys()):
        overlapping.append(
            {
                "hxb2_position": pos,
                "n_resistance_mutations": len(resistance_positions[pos]),
                "resistance_mutations": [m["mutation"] for m in resistance_positions[pos]],
                "drug_classes": list(set(m["drug_class"] for m in resistance_positions[pos])),
                "n_epitopes": len(epitope_positions[pos]),
                "epitopes": [e["epitope"][:10] for e in epitope_positions[pos][:3]],
                "hla_restrictions": list(set(e["hla"] for e in epitope_positions[pos] if e["hla"])),
            }
        )

    return pd.DataFrame(overlapping).sort_values("hxb2_position")


def analyze_resistance_escape_tradeoffs(
    stanford_df: pd.DataFrame,
    ctl_df: pd.DataFrame,
) -> pd.DataFrame:
    """Analyze trade-offs between drug resistance and immune escape."""
    overlapping = find_overlapping_positions(stanford_df, ctl_df)

    if overlapping.empty:
        return pd.DataFrame()

    # Score each position by constraint level
    overlapping = overlapping.copy()

    # Higher constraint = more pressures from both drug and immune selection
    overlapping["resistance_pressure"] = overlapping["n_resistance_mutations"]
    overlapping["immune_pressure"] = overlapping["n_epitopes"]
    overlapping["total_constraint"] = overlapping["resistance_pressure"] + overlapping["immune_pressure"]

    # Identify trade-off positions
    overlapping["is_tradeoff"] = (overlapping["resistance_pressure"] > 1) & (overlapping["immune_pressure"] > 1)

    return overlapping.sort_values("total_constraint", ascending=False)


def identify_multi_constraint_targets(tradeoffs_df: pd.DataFrame) -> pd.DataFrame:
    """Identify positions under multiple selective constraints (vaccine targets)."""
    if tradeoffs_df.empty:
        return pd.DataFrame()

    # Positions under high constraint from multiple sources are good vaccine targets
    # (virus cannot easily escape without fitness cost)
    targets = tradeoffs_df[tradeoffs_df["total_constraint"] >= 3].copy()

    # Score based on constraint diversity
    targets["constraint_score"] = targets["resistance_pressure"] * 0.5 + targets["immune_pressure"] * 0.5

    # Normalize
    if targets["constraint_score"].max() > 0:
        targets["normalized_score"] = targets["constraint_score"] / targets["constraint_score"].max()
    else:
        targets["normalized_score"] = 0

    return targets.sort_values("normalized_score", ascending=False)


def compute_dataset_overlap_stats(datasets: dict) -> pd.DataFrame:
    """Compute statistics on dataset overlap and coverage."""
    stats = []

    dataset_names = list(datasets.keys())
    for i, name1 in enumerate(dataset_names):
        for name2 in dataset_names[i + 1 :]:
            df1 = datasets[name1]
            df2 = datasets[name2]

            # Count overlapping positions (simplified)
            stats.append(
                {
                    "dataset1": name1,
                    "dataset2": name2,
                    "records1": len(df1),
                    "records2": len(df2),
                    "combined_records": len(df1) + len(df2),
                }
            )

    return pd.DataFrame(stats)


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset integration analysis")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "integration",
        help="Output directory for results",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Cross-Dataset Integration Analysis")
    print("=" * 60)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load all datasets
    print("\nLoading datasets...")
    datasets = load_all_datasets()

    if len(datasets) < 2:
        print("\nInsufficient datasets for integration analysis")
        print("Need at least 2 datasets (Stanford HIVDB + CTL or CATNAP)")
        return 1

    # Compute overlap statistics
    print("\nComputing dataset overlap statistics...")
    overlap_stats = compute_dataset_overlap_stats(datasets)
    print(f"  Analyzed {len(overlap_stats)} dataset pairs")

    # Analyze resistance-escape tradeoffs
    tradeoffs_df = pd.DataFrame()
    if "stanford" in datasets and "ctl" in datasets:
        print("\nAnalyzing resistance-escape tradeoffs...")
        tradeoffs_df = analyze_resistance_escape_tradeoffs(
            datasets["stanford"],
            datasets["ctl"],
        )
        n_tradeoffs = tradeoffs_df["is_tradeoff"].sum() if not tradeoffs_df.empty else 0
        print(f"  Found {len(tradeoffs_df)} overlapping positions")
        print(f"  Trade-off positions: {n_tradeoffs}")

    # Identify vaccine targets
    targets_df = pd.DataFrame()
    if not tradeoffs_df.empty:
        print("\nIdentifying multi-constraint vaccine targets...")
        targets_df = identify_multi_constraint_targets(tradeoffs_df)
        print(f"  Found {len(targets_df)} high-constraint positions")

        if not targets_df.empty:
            print("\nTop 10 Vaccine Target Positions:")
            for _, row in targets_df.head(10).iterrows():
                print(
                    f"  HXB2 {row['hxb2_position']}: "
                    f"resistance={row['resistance_pressure']}, "
                    f"immune={row['immune_pressure']}, "
                    f"score={row['normalized_score']:.3f}"
                )

    # Save results
    print(f"\nSaving results to {args.output}...")

    overlap_stats.to_csv(args.output / "dataset_overlap.csv", index=False)

    if not tradeoffs_df.empty:
        tradeoffs_df.to_csv(args.output / "resistance_escape_tradeoffs.csv", index=False)

    if not targets_df.empty:
        targets_df.to_csv(args.output / "vaccine_targets.csv", index=False)

    # Save summary
    summary = {
        "datasets_loaded": len(datasets),
        "total_records": sum(len(df) for df in datasets.values()),
        "overlapping_positions": len(tradeoffs_df),
        "tradeoff_positions": int(tradeoffs_df["is_tradeoff"].sum()) if not tradeoffs_df.empty else 0,
        "vaccine_targets": len(targets_df),
    }
    pd.DataFrame([summary]).to_csv(args.output / "summary.csv", index=False)

    print("\nAnalysis complete!")
    print(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
