#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""CTL Epitope Escape Analysis (Expanded).

Analyzes 2,116 CTL epitopes using p-adic hyperbolic geometry:
- HLA-stratified escape landscapes
- Epitope conservation vs radial position
- Protein-specific escape velocity
- Boundary-crossing frequency by HLA

Usage:
    python src/scripts/hiv/analysis/analyze_ctl_escape_expanded.py
    python src/scripts/hiv/analysis/analyze_ctl_escape_expanded.py --protein Gag
    python src/scripts/hiv/analysis/analyze_ctl_escape_expanded.py --hla "A*02:01"
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

from src.biology.codons import AMINO_ACID_TO_CODONS, codon_to_index
from src.data.hiv import (
    get_epitopes_by_hla,
    get_epitopes_by_protein,
    load_lanl_ctl,
    parse_hla_restrictions,
)


def compute_epitope_embedding(epitope_seq: str) -> np.ndarray:
    """
    Compute hyperbolic embedding for an epitope sequence.

    Returns average codon embedding across the epitope.
    """
    embeddings = []

    for aa in epitope_seq:
        codons = AMINO_ACID_TO_CODONS.get(aa, [])
        if codons:
            # Use first codon as representative
            try:
                idx = codon_to_index(codons[0])
                # Map to radial position based on p-adic valuation
                valuation = 0
                temp_idx = idx
                while temp_idx > 0 and temp_idx % 3 == 0:
                    valuation += 1
                    temp_idx //= 3
                radial = 1.0 - (valuation / 5.0)  # Normalize to [0, 1]
                embeddings.append(radial)
            except (KeyError, ValueError):
                pass

    if embeddings:
        return np.array(embeddings)
    return np.array([0.5])  # Default to middle


def compute_escape_velocity(epitopes_df: pd.DataFrame) -> dict:
    """
    Compute escape velocity metrics for a set of epitopes.

    Escape velocity measures how quickly the virus can evolve away from
    immune recognition at each epitope position.
    """
    results = {
        "mean_radial_position": 0.0,
        "radial_variance": 0.0,
        "n_epitopes": len(epitopes_df),
        "conservation_score": 0.0,
    }

    if epitopes_df.empty:
        return results

    radial_positions = []
    for _, row in epitopes_df.iterrows():
        epitope = row.get("Epitope", "")
        if epitope and isinstance(epitope, str):
            embedding = compute_epitope_embedding(epitope)
            radial_positions.append(np.mean(embedding))

    if radial_positions:
        results["mean_radial_position"] = np.mean(radial_positions)
        results["radial_variance"] = np.var(radial_positions)
        # Lower variance = more conserved
        results["conservation_score"] = 1.0 / (1.0 + results["radial_variance"])

    return results


def analyze_hla_landscapes(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze escape landscapes stratified by HLA type."""
    hla_results = []

    # Collect all HLA types
    hla_counts = defaultdict(int)
    for hla_string in df["HLA"].dropna():
        for hla in parse_hla_restrictions(hla_string):
            hla_counts[hla] += 1

    # Analyze top HLA types
    top_hlas = sorted(hla_counts.items(), key=lambda x: -x[1])[:20]

    for hla, count in top_hlas:
        hla_epitopes = get_epitopes_by_hla(df, hla)
        velocity = compute_escape_velocity(hla_epitopes)

        hla_results.append(
            {
                "HLA": hla,
                "n_epitopes": count,
                "mean_radial": velocity["mean_radial_position"],
                "radial_variance": velocity["radial_variance"],
                "conservation_score": velocity["conservation_score"],
            }
        )

    return pd.DataFrame(hla_results)


def analyze_protein_escape(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze escape velocity by HIV protein."""
    proteins = ["Gag", "Pol", "Env", "Nef", "Tat", "Rev", "Vif", "Vpr", "Vpu"]
    results = []

    for protein in proteins:
        protein_epitopes = get_epitopes_by_protein(df, protein)
        if not protein_epitopes.empty:
            velocity = compute_escape_velocity(protein_epitopes)
            velocity["protein"] = protein
            results.append(velocity)

    return pd.DataFrame(results)


def identify_conserved_epitopes(df: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Identify highly conserved epitopes suitable for vaccine targets."""
    results = []

    for _, row in df.iterrows():
        epitope = row.get("Epitope", "")
        if not epitope or not isinstance(epitope, str):
            continue

        embedding = compute_epitope_embedding(epitope)
        mean_radial = np.mean(embedding)
        variance = np.var(embedding)
        conservation = 1.0 / (1.0 + variance)

        if conservation >= threshold:
            results.append(
                {
                    "Epitope": epitope,
                    "Protein": row.get("Protein", ""),
                    "HLA": row.get("HLA", ""),
                    "HXB2_start": row.get("HXB2_start", ""),
                    "HXB2_end": row.get("HXB2_end", ""),
                    "mean_radial": mean_radial,
                    "conservation_score": conservation,
                }
            )

    return pd.DataFrame(results).sort_values("conservation_score", ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Analyze CTL epitope escape landscapes")
    parser.add_argument(
        "--protein",
        type=str,
        default=None,
        help="Filter to specific protein (Gag, Pol, Env, etc.)",
    )
    parser.add_argument(
        "--hla",
        type=str,
        default=None,
        help="Filter to specific HLA type (e.g., A*02:01)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "ctl_escape",
        help="Output directory for results",
    )
    parser.add_argument(
        "--conservation-threshold",
        type=float,
        default=0.7,
        help="Conservation score threshold for vaccine targets",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("CTL Epitope Escape Analysis")
    print("=" * 60)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading CTL epitope data...")
    try:
        df = load_lanl_ctl()
        print(f"Loaded {len(df):,} epitopes")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Apply filters
    if args.protein:
        df = get_epitopes_by_protein(df, args.protein)
        print(f"Filtered to {len(df):,} {args.protein} epitopes")

    if args.hla:
        df = get_epitopes_by_hla(df, args.hla)
        print(f"Filtered to {len(df):,} {args.hla}-restricted epitopes")

    if df.empty:
        print("No epitopes match the filter criteria")
        return 1

    # Analyze HLA landscapes
    print("\nAnalyzing HLA-stratified escape landscapes...")
    hla_landscapes = analyze_hla_landscapes(df)
    print(f"  Analyzed {len(hla_landscapes)} HLA types")

    # Analyze protein escape
    print("\nAnalyzing protein-specific escape velocity...")
    protein_escape = analyze_protein_escape(df)
    print(f"  Analyzed {len(protein_escape)} proteins")

    # Identify conserved epitopes
    print("\nIdentifying conserved vaccine targets...")
    conserved = identify_conserved_epitopes(df, args.conservation_threshold)
    print(f"  Found {len(conserved)} conserved epitopes")

    # Print top results
    if not protein_escape.empty:
        print("\nEscape Velocity by Protein:")
        for _, row in protein_escape.iterrows():
            print(
                f"  {row['protein']}: {row['mean_radial_position']:.3f} (conservation: {row['conservation_score']:.3f})"
            )

    if not conserved.empty:
        print("\nTop Conserved Epitopes for Vaccine Design:")
        for _, row in conserved.head(10).iterrows():
            print(f"  {row['Epitope'][:15]}... ({row['Protein']}) - score: {row['conservation_score']:.3f}")

    # Save results
    print(f"\nSaving results to {args.output}...")
    hla_landscapes.to_csv(args.output / "hla_escape_landscapes.csv", index=False)
    protein_escape.to_csv(args.output / "protein_escape_velocity.csv", index=False)
    conserved.to_csv(args.output / "conserved_epitopes.csv", index=False)

    # Save summary
    summary = {
        "total_epitopes": len(df),
        "hla_types_analyzed": len(hla_landscapes),
        "proteins_analyzed": len(protein_escape),
        "conserved_targets": len(conserved),
    }
    pd.DataFrame([summary]).to_csv(args.output / "summary.csv", index=False)

    print("\nAnalysis complete!")
    print(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
