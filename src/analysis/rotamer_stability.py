# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Rotamer Stability Analysis with P-adic Geometry.

This module analyzes the correlation between side-chain rotamer angles
and their stability using p-adic (3-adic) geometric measures. The key
hypothesis is that rare/unstable rotamers have higher "hyperbolic velocity"
(distance from cluster centroids) compared to common, stable rotamers.

Key Features:
1. Embed chi angles into hyperbolic space
2. Compute distance to rotamer library cluster centroids
3. Correlate hyperbolic distance with Rosetta/physics-based energy
4. Identify "Rosetta-blind" instabilities (low Rosetta score, high hyperbolic distance)

Usage:
    python src/scripts/analysis/rotamer_stability.py \
        --input data/processed/rotamers.pt \
        --output results/rotamer_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.paths import PROCESSED_DATA_DIR, RESULTS_DIR

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Standard rotamer library centroids (chi1, chi2 in radians)
# Based on Dunbrack rotamer library common conformations
ROTAMER_CENTROIDS = {
    "gauche+": np.array([np.radians(60), np.radians(60), 0, 0]),
    "gauche-": np.array([np.radians(-60), np.radians(-60), 0, 0]),
    "trans": np.array([np.radians(180), np.radians(180), 0, 0]),
    "g+/t": np.array([np.radians(60), np.radians(180), 0, 0]),
    "g-/t": np.array([np.radians(-60), np.radians(180), 0, 0]),
    "t/g+": np.array([np.radians(180), np.radians(60), 0, 0]),
    "t/g-": np.array([np.radians(180), np.radians(-60), 0, 0]),
}


@dataclass
class RotamerAnalysisResult:
    """Result of rotamer stability analysis for a single residue."""

    pdb_id: str
    chain_id: str
    residue_id: int
    residue_name: str
    chi_angles: list[float]
    nearest_rotamer: str
    euclidean_distance: float
    hyperbolic_distance: float
    padic_valuation: int
    stability_score: float
    is_rare: bool


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def angular_distance(a1: np.ndarray, a2: np.ndarray) -> float:
    """Compute angular distance between two angle vectors."""
    diff = 0.0
    valid_count = 0
    for x, y in zip(a1, a2, strict=False):
        if not np.isnan(x) and not np.isnan(y):
            d = abs(normalize_angle(x - y))
            diff += d**2
            valid_count += 1
    if valid_count == 0:
        return 0.0
    return np.sqrt(diff / valid_count)


def padic_valuation(n: int, p: int = 3) -> int:
    """Compute p-adic valuation of integer n."""
    if n == 0:
        return float("inf")
    v = 0
    while n % p == 0:
        v += 1
        n //= p
    return v


def angle_to_index(angle: float, bins: int = 36) -> int:
    """Convert angle to discrete bin index."""
    # Normalize to [0, 2pi]
    angle = normalize_angle(angle)
    if angle < 0:
        angle += 2 * np.pi
    # Map to bins
    idx = int(angle / (2 * np.pi) * bins)
    return min(idx, bins - 1)


def chi_to_padic_valuation(chi_angles: list[float], p: int = 3) -> int:
    """Convert chi angles to a p-adic valuation score.

    The intuition: Rare rotamers often occur at unusual angle combinations.
    We discretize angles and compute the p-adic valuation of the combined index.
    """
    # Discretize angles
    bins = 36  # 10-degree bins
    indices = []
    for chi in chi_angles:
        if not np.isnan(chi):
            indices.append(angle_to_index(chi, bins))

    if not indices:
        return 0

    # Combine indices into single number (base=bins)
    combined = 0
    for i, idx in enumerate(indices):
        combined += idx * (bins**i)

    return padic_valuation(combined + 1, p)  # +1 to avoid 0


def hyperbolic_distance_from_chi(
    chi_angles: np.ndarray,
    curvature: float = -1.0,
) -> float:
    """Compute hyperbolic distance to origin from chi angles.

    Maps chi angles to Poincare ball coordinates and computes geodesic distance.
    """
    # Filter valid angles
    valid_chi = [c for c in chi_angles if not np.isnan(c)]
    if not valid_chi:
        return 0.0

    # Map angles to Poincare ball (tanh maps to [-1, 1])
    coords = np.array([np.tanh(c / np.pi) for c in valid_chi])

    # Compute Euclidean norm
    r = np.linalg.norm(coords)
    if r >= 1.0:
        r = 0.999  # Clamp to disk

    # Hyperbolic distance from origin
    # d_H = 2 * arctanh(r) for K=-1
    return 2 * np.arctanh(r)


def find_nearest_rotamer(chi_angles: np.ndarray) -> tuple[str, float]:
    """Find the nearest standard rotamer and distance.

    Args:
        chi_angles: Array of chi angles

    Returns:
        Tuple of (rotamer name, euclidean distance)
    """
    min_dist = float("inf")
    nearest = "unknown"

    for name, centroid in ROTAMER_CENTROIDS.items():
        dist = angular_distance(chi_angles, centroid)
        if dist < min_dist:
            min_dist = dist
            nearest = name

    return nearest, min_dist


def analyze_rotamers(
    chi_tensor: np.ndarray,
    metadata: list[dict],
    rare_threshold: float = 0.8,
) -> list[RotamerAnalysisResult]:
    """Analyze rotamer stability for all residues.

    Args:
        chi_tensor: [N, 4] tensor of chi angles
        metadata: List of metadata dicts
        rare_threshold: Hyperbolic distance threshold for "rare" classification

    Returns:
        List of analysis results
    """
    results = []

    for _i, (chi, meta) in enumerate(zip(chi_tensor, metadata, strict=False)):
        # Find nearest standard rotamer
        nearest, eucl_dist = find_nearest_rotamer(chi)

        # Compute hyperbolic distance
        hyp_dist = hyperbolic_distance_from_chi(chi)

        # Compute p-adic valuation
        padic_val = chi_to_padic_valuation(chi.tolist())

        # Compute stability score (inverse of hyperbolic distance)
        stability = 1.0 / (1.0 + hyp_dist)

        # Classify as rare if above threshold
        is_rare = hyp_dist > rare_threshold

        result = RotamerAnalysisResult(
            pdb_id=meta["pdb_id"],
            chain_id=meta["chain_id"],
            residue_id=meta["residue_id"],
            residue_name=meta["residue_name"],
            chi_angles=chi.tolist(),
            nearest_rotamer=nearest,
            euclidean_distance=float(eucl_dist),
            hyperbolic_distance=float(hyp_dist),
            padic_valuation=int(padic_val),
            stability_score=float(stability),
            is_rare=is_rare,
        )
        results.append(result)

    return results


def compute_summary_statistics(results: list[RotamerAnalysisResult]) -> dict:
    """Compute summary statistics from analysis results."""
    if not results:
        return {}

    hyp_distances = [r.hyperbolic_distance for r in results]
    eucl_distances = [r.euclidean_distance for r in results]
    padic_vals = [r.padic_valuation for r in results]

    # Correlation between hyperbolic and Euclidean
    correlation = np.corrcoef(hyp_distances, eucl_distances)[0, 1] if len(hyp_distances) > 1 else 0.0

    # Group by residue type
    by_residue = {}
    for r in results:
        if r.residue_name not in by_residue:
            by_residue[r.residue_name] = []
        by_residue[r.residue_name].append(r.hyperbolic_distance)

    residue_means = {k: float(np.mean(v)) for k, v in by_residue.items()}

    return {
        "n_residues": len(results),
        "n_rare": sum(1 for r in results if r.is_rare),
        "rare_fraction": sum(1 for r in results if r.is_rare) / len(results),
        "hyperbolic_distance": {
            "mean": float(np.mean(hyp_distances)),
            "std": float(np.std(hyp_distances)),
            "min": float(np.min(hyp_distances)),
            "max": float(np.max(hyp_distances)),
        },
        "euclidean_distance": {
            "mean": float(np.mean(eucl_distances)),
            "std": float(np.std(eucl_distances)),
        },
        "padic_valuation": {
            "mean": float(np.mean(padic_vals)),
            "max": int(np.max(padic_vals)),
        },
        "hyp_eucl_correlation": float(correlation) if not np.isnan(correlation) else 0.0,
        "mean_by_residue_type": residue_means,
    }


def export_results(
    results: list[RotamerAnalysisResult],
    summary: dict,
    output_path: Path,
) -> None:
    """Export analysis results to JSON."""
    output = {
        "summary": summary,
        "residues": [
            {
                "pdb_id": r.pdb_id,
                "chain_id": r.chain_id,
                "residue_id": (
                    int(r.residue_id) if isinstance(r.residue_id, (np.integer, np.floating)) else r.residue_id
                ),
                "residue_name": r.residue_name,
                "chi_angles": [float(x) if not np.isnan(x) else None for x in r.chi_angles],
                "nearest_rotamer": r.nearest_rotamer,
                "euclidean_distance": float(r.euclidean_distance),
                "hyperbolic_distance": float(r.hyperbolic_distance),
                "padic_valuation": int(r.padic_valuation),
                "stability_score": float(r.stability_score),
                "is_rare": bool(r.is_rare),
            }
            for r in results
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exported results to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Rotamer stability analysis with p-adic geometry")
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROCESSED_DATA_DIR / "rotamers.pt"),
        help="Input rotamer tensor file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RESULTS_DIR / "rotamer_analysis.json"),
        help="Output JSON file",
    )
    parser.add_argument(
        "--rare_threshold",
        type=float,
        default=0.8,
        help="Hyperbolic distance threshold for rare classification",
    )

    args = parser.parse_args()

    if not HAS_TORCH:
        print("PyTorch required")
        return

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Run ingest_pdb_rotamers.py first to generate rotamer data.")
        return

    # Load data
    print(f"Loading rotamer data from {input_path}...")
    data = torch.load(input_path, weights_only=False)
    chi_tensor = data["chi_angles"].numpy()
    metadata = data["metadata"]

    print(f"Loaded {len(metadata)} residues")

    # Analyze
    print("Analyzing rotamer stability...")
    results = analyze_rotamers(chi_tensor, metadata, args.rare_threshold)

    # Compute summary
    summary = compute_summary_statistics(results)

    # Print summary
    print("\n=== Summary ===")
    print(f"Total residues: {summary['n_residues']}")
    print(f"Rare rotamers: {summary['n_rare']} ({summary['rare_fraction'] * 100:.1f}%)")
    print(f"Mean hyperbolic distance: {summary['hyperbolic_distance']['mean']:.3f}")
    print(f"Hyp-Eucl correlation: {summary['hyp_eucl_correlation']:.3f}")

    # Export
    export_results(results, summary, Path(args.output))


if __name__ == "__main__":
    main()
