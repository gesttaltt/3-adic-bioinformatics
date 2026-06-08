#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""V3 Coreceptor Tropism Analysis.

Analyzes ~3,647 V3 loop sequences for CCR5/CXCR4 tropism:
- CCR5 vs CXCR4 hyperbolic separation
- Tropism-switching trajectory mapping
- Glycan shield correlation with tropism
- ML tropism predictor using hyperbolic features

Usage:
    python src/scripts/hiv/analysis/analyze_tropism_switching.py
    python src/scripts/hiv/analysis/analyze_tropism_switching.py --output results/tropism
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


def load_available_tropism_data() -> pd.DataFrame:
    """Load tropism data from available sources."""
    try:
        from src.data.hiv import load_v3_coreceptor

        return load_v3_coreceptor()
    except (ImportError, FileNotFoundError):
        pass

    try:
        from src.data.hiv import load_gp120_alignments

        alignments = load_gp120_alignments()
        # Convert to DataFrame format
        return pd.DataFrame([{"sequence_id": k, "sequence": v, "tropism": "unknown"} for k, v in alignments.items()])
    except (ImportError, FileNotFoundError):
        pass

    raise FileNotFoundError(
        "No tropism data available. Download using:\n"
        "  ternary-vae data download huggingface --dataset HIV_V3_coreceptor"
    )


def encode_sequence_hyperbolic(sequence: str) -> np.ndarray:
    """
    Encode amino acid sequence to hyperbolic features.

    Returns array of p-adic-derived radial positions for each residue.
    """
    from src.biology.codons import AMINO_ACID_TO_CODONS, codon_to_index

    features = []

    for aa in sequence:
        codons = AMINO_ACID_TO_CODONS.get(aa, [])
        if codons:
            try:
                idx = codon_to_index(codons[0])
                # P-adic valuation feature
                valuation = 0
                temp_idx = idx
                while temp_idx > 0 and temp_idx % 3 == 0:
                    valuation += 1
                    temp_idx //= 3
                features.append(valuation / 5.0)
            except (KeyError, ValueError):
                features.append(0.5)
        else:
            features.append(0.5)

    return np.array(features)


def compute_tropism_separation(df: pd.DataFrame) -> dict:
    """Compute hyperbolic separation between CCR5 and CXCR4 sequences."""
    ccr5_embeddings = []
    cxcr4_embeddings = []

    tropism_col = None
    for col in ["tropism", "Tropism", "label", "Label"]:
        if col in df.columns:
            tropism_col = col
            break

    seq_col = None
    for col in ["sequence", "Sequence", "v3_sequence", "V3"]:
        if col in df.columns:
            seq_col = col
            break

    if tropism_col is None or seq_col is None:
        return {"error": "Could not find tropism or sequence columns"}

    for _, row in df.iterrows():
        tropism = str(row[tropism_col]).upper()
        sequence = str(row[seq_col])

        if len(sequence) < 5:
            continue

        embedding = encode_sequence_hyperbolic(sequence)
        mean_radial = np.mean(embedding)

        if "CCR5" in tropism or "R5" in tropism:
            ccr5_embeddings.append(mean_radial)
        elif "CXCR4" in tropism or "X4" in tropism:
            cxcr4_embeddings.append(mean_radial)

    if not ccr5_embeddings or not cxcr4_embeddings:
        return {
            "n_ccr5": len(ccr5_embeddings),
            "n_cxcr4": len(cxcr4_embeddings),
            "error": "Insufficient data for separation analysis",
        }

    ccr5_arr = np.array(ccr5_embeddings)
    cxcr4_arr = np.array(cxcr4_embeddings)

    return {
        "n_ccr5": len(ccr5_arr),
        "n_cxcr4": len(cxcr4_arr),
        "ccr5_mean_radial": float(np.mean(ccr5_arr)),
        "ccr5_std_radial": float(np.std(ccr5_arr)),
        "cxcr4_mean_radial": float(np.mean(cxcr4_arr)),
        "cxcr4_std_radial": float(np.std(cxcr4_arr)),
        "separation": float(np.mean(cxcr4_arr) - np.mean(ccr5_arr)),
        "effect_size": float(
            (np.mean(cxcr4_arr) - np.mean(ccr5_arr)) / np.sqrt((np.var(ccr5_arr) + np.var(cxcr4_arr)) / 2)
        )
        if np.var(ccr5_arr) + np.var(cxcr4_arr) > 0
        else 0,
    }


def identify_glycan_sites(sequence: str) -> list[int]:
    """Identify N-glycosylation sites (N-X-S/T motif)."""
    sites = []
    for i in range(len(sequence) - 2):
        if sequence[i] == "N" and sequence[i + 1] != "P" and sequence[i + 2] in "ST":
            sites.append(i)
    return sites


def analyze_glycan_tropism_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze correlation between glycan shield and tropism."""
    results = []

    tropism_col = None
    for col in ["tropism", "Tropism", "label", "Label"]:
        if col in df.columns:
            tropism_col = col
            break

    seq_col = None
    for col in ["sequence", "Sequence", "v3_sequence", "V3"]:
        if col in df.columns:
            seq_col = col
            break

    if tropism_col is None or seq_col is None:
        return pd.DataFrame()

    for _, row in df.iterrows():
        tropism = str(row[tropism_col]).upper()
        sequence = str(row[seq_col])

        if len(sequence) < 5:
            continue

        glycan_sites = identify_glycan_sites(sequence)

        results.append(
            {
                "tropism": "CCR5"
                if "CCR5" in tropism or "R5" in tropism
                else ("CXCR4" if "CXCR4" in tropism or "X4" in tropism else "Unknown"),
                "n_glycans": len(glycan_sites),
                "sequence_length": len(sequence),
                "glycan_density": len(glycan_sites) / len(sequence),
            }
        )

    return pd.DataFrame(results)


def build_tropism_classifier(df: pd.DataFrame) -> dict:
    """Build simple tropism classifier using hyperbolic features."""
    tropism_col = None
    for col in ["tropism", "Tropism", "label", "Label"]:
        if col in df.columns:
            tropism_col = col
            break

    seq_col = None
    for col in ["sequence", "Sequence", "v3_sequence", "V3"]:
        if col in df.columns:
            seq_col = col
            break

    if tropism_col is None or seq_col is None:
        return {"error": "Could not find required columns"}

    X = []
    y = []

    for _, row in df.iterrows():
        tropism = str(row[tropism_col]).upper()
        sequence = str(row[seq_col])

        if len(sequence) < 5:
            continue

        # Extract features
        embedding = encode_sequence_hyperbolic(sequence)
        glycan_sites = identify_glycan_sites(sequence)

        features = [
            np.mean(embedding),
            np.std(embedding),
            np.min(embedding),
            np.max(embedding),
            len(glycan_sites),
            len(glycan_sites) / len(sequence),
            len(sequence),
        ]

        if "CCR5" in tropism or "R5" in tropism:
            X.append(features)
            y.append(0)
        elif "CXCR4" in tropism or "X4" in tropism:
            X.append(features)
            y.append(1)

    if len(X) < 50:
        return {"error": "Insufficient labeled data for classifier"}

    X = np.array(X)
    y = np.array(y)

    # Simple train/test split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        "n_samples": len(X),
        "n_ccr5": int(np.sum(y == 0)),
        "n_cxcr4": int(np.sum(y == 1)),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "accuracy": float(accuracy),
        "feature_importance": {
            "mean_radial": float(clf.feature_importances_[0]),
            "std_radial": float(clf.feature_importances_[1]),
            "min_radial": float(clf.feature_importances_[2]),
            "max_radial": float(clf.feature_importances_[3]),
            "n_glycans": float(clf.feature_importances_[4]),
            "glycan_density": float(clf.feature_importances_[5]),
            "seq_length": float(clf.feature_importances_[6]),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze V3 coreceptor tropism")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "tropism",
        help="Output directory for results",
    )
    parser.add_argument(
        "--build-classifier",
        action="store_true",
        help="Build and evaluate tropism classifier",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("V3 Coreceptor Tropism Analysis")
    print("=" * 60)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading tropism data...")
    try:
        df = load_available_tropism_data()
        print(f"Loaded {len(df):,} sequences")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Compute tropism separation
    print("\nComputing CCR5 vs CXCR4 hyperbolic separation...")
    separation = compute_tropism_separation(df)

    if "error" not in separation:
        print(f"  CCR5 sequences: {separation['n_ccr5']:,}")
        print(f"  CXCR4 sequences: {separation['n_cxcr4']:,}")
        print(f"  Mean radial CCR5: {separation['ccr5_mean_radial']:.4f}")
        print(f"  Mean radial CXCR4: {separation['cxcr4_mean_radial']:.4f}")
        print(f"  Separation: {separation['separation']:.4f}")
        print(f"  Effect size: {separation['effect_size']:.4f}")
    else:
        print(f"  Warning: {separation.get('error', 'Unknown error')}")

    # Analyze glycan-tropism correlation
    print("\nAnalyzing glycan shield correlation...")
    glycan_df = analyze_glycan_tropism_correlation(df)

    if not glycan_df.empty:
        for tropism in ["CCR5", "CXCR4"]:
            subset = glycan_df[glycan_df["tropism"] == tropism]
            if not subset.empty:
                print(f"  {tropism}: mean glycans = {subset['n_glycans'].mean():.2f}")

    # Build classifier
    classifier_results = {}
    if args.build_classifier:
        print("\nBuilding tropism classifier...")
        try:
            classifier_results = build_tropism_classifier(df)
            if "error" not in classifier_results:
                print(f"  Training samples: {classifier_results['train_size']}")
                print(f"  Test samples: {classifier_results['test_size']}")
                print(f"  Accuracy: {classifier_results['accuracy']:.1%}")
                print("\n  Feature Importance:")
                for feat, imp in classifier_results["feature_importance"].items():
                    print(f"    {feat}: {imp:.3f}")
            else:
                print(f"  Warning: {classifier_results['error']}")
        except ImportError:
            print("  scikit-learn not installed. Run: pip install scikit-learn")

    # Save results
    print(f"\nSaving results to {args.output}...")

    pd.DataFrame([separation]).to_csv(args.output / "tropism_separation.csv", index=False)
    glycan_df.to_csv(args.output / "glycan_tropism.csv", index=False)

    if classifier_results and "error" not in classifier_results:
        pd.DataFrame([classifier_results]).to_csv(args.output / "classifier_results.csv", index=False)

    # Save summary
    summary = {
        "total_sequences": len(df),
        "n_ccr5": separation.get("n_ccr5", 0),
        "n_cxcr4": separation.get("n_cxcr4", 0),
        "separation_effect_size": separation.get("effect_size", 0),
        "classifier_accuracy": classifier_results.get("accuracy", 0),
    }
    pd.DataFrame([summary]).to_csv(args.output / "summary.csv", index=False)

    print("\nAnalysis complete!")
    print(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
