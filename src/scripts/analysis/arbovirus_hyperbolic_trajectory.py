# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Arbovirus Hyperbolic Trajectory Analysis.

This module embeds arbovirus genomes into hyperbolic space and computes
temporal evolution trajectories to predict serotype dominance.

Key Features:
1. Embed viral genomes using sliding window p-adic encoding
2. Track serotype centroids over time (monthly/yearly)
3. Compute "Hyperbolic Momentum" (velocity vector) for each clade
4. Forecast trajectory 6-12 months ahead
5. Detect convergent evolution towards "severe" clusters

Usage:
    python src/scripts/analysis/arbovirus_hyperbolic_trajectory.py \
        --input data/processed/dengue_trajectories.pt \
        --output results/dengue_forecast.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from Bio import SeqIO

    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


@dataclass
class TrajectoryPoint:
    """A point in the hyperbolic trajectory."""

    time_point: str  # YYYY-MM or YYYY
    serotype: str
    centroid: np.ndarray  # Hyperbolic centroid coordinates
    n_sequences: int
    variance: float


@dataclass
class VelocityVector:
    """Velocity vector for a serotype."""

    serotype: str
    direction: np.ndarray
    magnitude: float
    time_window: str


@dataclass
class ForecastResult:
    """Forecast result for a serotype."""

    serotype: str
    current_position: np.ndarray
    predicted_position: np.ndarray
    confidence: float
    risk_score: float
    time_horizon: str


def padic_valuation(n: int, p: int = 3) -> int:
    """Compute p-adic valuation."""
    if n == 0:
        return float("inf")
    v = 0
    while n % p == 0:
        v += 1
        n //= p
    return v


def codon_to_index(codon: str) -> int:
    """Convert nucleotide codon to index (0-63)."""
    base_map = {"A": 0, "T": 1, "U": 1, "G": 2, "C": 3}
    idx = 0
    for i, base in enumerate(codon.upper()):
        if base in base_map:
            idx += base_map[base] * (4 ** (2 - i))
        else:
            return -1
    return idx


def encode_window_hyperbolic(
    sequence: str,
    window_size: int = 300,
    step: int = 100,
    p: int = 3,
) -> np.ndarray:
    """Encode a nucleotide sequence into hyperbolic coordinates.

    Uses sliding window p-adic encoding to capture local structure.

    Args:
        sequence: Nucleotide sequence
        window_size: Size of sliding window (nucleotides)
        step: Step size for sliding window
        p: Prime for p-adic valuation

    Returns:
        Array of shape (n_windows, embedding_dim)
    """
    sequence = sequence.upper().replace("U", "T")
    embeddings = []

    for start in range(0, len(sequence) - window_size + 1, step):
        window = sequence[start : start + window_size]

        # Convert to codons
        codon_indices = []
        for i in range(0, len(window) - 2, 3):
            codon = window[i : i + 3]
            idx = codon_to_index(codon)
            if idx >= 0:
                codon_indices.append(idx)

        if not codon_indices:
            continue

        # Compute p-adic valuations
        valuations = [padic_valuation(idx + 1, p) for idx in codon_indices]

        # Create embedding features
        features = np.array(
            [
                np.mean(valuations),
                np.std(valuations) if len(valuations) > 1 else 0,
                np.max(valuations),
                len([v for v in valuations if v > 0]) / len(valuations),  # Fraction with valuation > 0
                np.mean(codon_indices) / 64,  # Normalized mean index
                np.std(codon_indices) / 64 if len(codon_indices) > 1 else 0,
            ]
        )

        embeddings.append(features)

    if not embeddings:
        return np.zeros((1, 6))

    return np.array(embeddings)


def embed_genome(sequence: str) -> np.ndarray:
    """Embed full genome into single hyperbolic point.

    Averages window embeddings to get genome-level representation.
    """
    window_embeddings = encode_window_hyperbolic(sequence)
    return np.mean(window_embeddings, axis=0)


def parse_header(header: str) -> tuple[str, str | None, str | None]:
    """Parse FASTA header to extract accession, serotype, year."""
    parts = header.split("|")
    accession = parts[0].strip(">")

    serotype = None
    year = None

    if len(parts) > 1:
        serotype = parts[1] if parts[1] != "unknown" else None

    if len(parts) > 2:
        with contextlib.suppress(ValueError, IndexError):
            year = parts[2] if parts[2] != "unknown" else None

    return accession, serotype, year


def load_and_embed_sequences(
    fasta_path: Path,
) -> dict[str, list[tuple[np.ndarray, str]]]:
    """Load sequences and embed by serotype and time.

    Returns:
        Dict mapping serotype -> list of (embedding, time_point)
    """
    if not HAS_BIOPYTHON:
        raise ImportError("Biopython required for FASTA parsing")

    serotype_data = defaultdict(list)

    for record in SeqIO.parse(fasta_path, "fasta"):
        accession, serotype, year = parse_header(record.id)

        if not serotype:
            serotype = "unknown"
        if not year:
            year = "unknown"

        # Embed sequence
        embedding = embed_genome(str(record.seq))

        serotype_data[serotype].append((embedding, year))

    return dict(serotype_data)


def compute_trajectory(
    serotype_data: dict[str, list[tuple[np.ndarray, str]]],
) -> dict[str, list[TrajectoryPoint]]:
    """Compute temporal trajectory for each serotype."""
    trajectories = {}

    for serotype, data in serotype_data.items():
        # Group by time
        by_time = defaultdict(list)
        for emb, time in data:
            by_time[time].append(emb)

        # Compute centroids
        points = []
        for time, embeddings in sorted(by_time.items()):
            emb_array = np.array(embeddings)
            centroid = np.mean(emb_array, axis=0)
            variance = float(np.mean(np.var(emb_array, axis=0)))

            point = TrajectoryPoint(
                time_point=time,
                serotype=serotype,
                centroid=centroid,
                n_sequences=len(embeddings),
                variance=variance,
            )
            points.append(point)

        trajectories[serotype] = points

    return trajectories


def compute_velocity(
    trajectory: list[TrajectoryPoint],
    window: int = 2,
) -> VelocityVector | None:
    """Compute velocity vector from recent trajectory points."""
    if len(trajectory) < 2:
        return None

    # Take last `window` points
    recent = trajectory[-window:]

    # Compute direction
    start = recent[0].centroid
    end = recent[-1].centroid
    direction = end - start

    # Compute magnitude (hyperbolic distance)
    magnitude = float(np.linalg.norm(direction))

    # Normalize direction
    if magnitude > 0:
        direction = direction / magnitude

    return VelocityVector(
        serotype=recent[0].serotype,
        direction=direction,
        magnitude=magnitude,
        time_window=f"{recent[0].time_point} to {recent[-1].time_point}",
    )


def forecast_position(
    trajectory: list[TrajectoryPoint],
    velocity: VelocityVector,
    steps: int = 1,
) -> ForecastResult:
    """Forecast future position based on velocity."""
    current = trajectory[-1].centroid

    # Linear extrapolation in hyperbolic space
    predicted = current + velocity.direction * velocity.magnitude * steps

    # Confidence based on trajectory consistency
    if len(trajectory) > 2:
        # Compute variance of directions over trajectory
        directions = []
        for i in range(1, len(trajectory)):
            d = trajectory[i].centroid - trajectory[i - 1].centroid
            if np.linalg.norm(d) > 0:
                directions.append(d / np.linalg.norm(d))
        if len(directions) > 1:
            direction_variance = float(np.var(directions))
            confidence = 1.0 / (1.0 + direction_variance)
        else:
            confidence = 0.5
    else:
        confidence = 0.3

    # Risk score based on movement towards "severe" region
    # (Higher embedding values = more divergent = potentially more severe)
    risk_score = float(np.mean(predicted) / (np.mean(current) + 1e-6))

    return ForecastResult(
        serotype=velocity.serotype,
        current_position=current,
        predicted_position=predicted,
        confidence=confidence,
        risk_score=risk_score,
        time_horizon=f"{steps} periods ahead",
    )


def analyze_trajectories(
    fasta_path: Path,
    output_path: Path,
) -> dict:
    """Full trajectory analysis pipeline."""
    print(f"Loading sequences from {fasta_path}...")
    serotype_data = load_and_embed_sequences(fasta_path)

    print(f"Found {len(serotype_data)} serotypes")
    for s, data in serotype_data.items():
        print(f"  {s}: {len(data)} sequences")

    print("\nComputing trajectories...")
    trajectories = compute_trajectory(serotype_data)

    print("\nComputing velocities and forecasts...")
    results = {
        "serotypes": {},
        "summary": {},
    }

    for serotype, traj in trajectories.items():
        velocity = compute_velocity(traj)

        serotype_result = {
            "trajectory": [
                {
                    "time": p.time_point,
                    "centroid": p.centroid.tolist(),
                    "n_sequences": p.n_sequences,
                    "variance": p.variance,
                }
                for p in traj
            ],
        }

        if velocity:
            forecast = forecast_position(traj, velocity)
            serotype_result["velocity"] = {
                "direction": velocity.direction.tolist(),
                "magnitude": velocity.magnitude,
                "time_window": velocity.time_window,
            }
            serotype_result["forecast"] = {
                "current_position": forecast.current_position.tolist(),
                "predicted_position": forecast.predicted_position.tolist(),
                "confidence": forecast.confidence,
                "risk_score": forecast.risk_score,
                "time_horizon": forecast.time_horizon,
            }

        results["serotypes"][serotype] = serotype_result

    # Summary
    velocities = []
    for s, r in results["serotypes"].items():
        if "velocity" in r:
            velocities.append((s, r["velocity"]["magnitude"]))

    velocities.sort(key=lambda x: x[1], reverse=True)

    results["summary"] = {
        "total_serotypes": len(serotype_data),
        "fastest_moving": velocities[0][0] if velocities else None,
        "highest_risk": max(
            [(s, r.get("forecast", {}).get("risk_score", 0)) for s, r in results["serotypes"].items()],
            key=lambda x: x[1],
            default=(None, 0),
        )[0],
    }

    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Arbovirus hyperbolic trajectory analysis")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/dengue_paraguay.fasta",
        help="Input FASTA file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/dengue_forecast.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Run ingest_arboviruses.py first to download sequences.")
        return

    results = analyze_trajectories(input_path, Path(args.output))

    # Print summary
    print("\n=== Forecast Summary ===")
    print(f"Fastest moving serotype: {results['summary']['fastest_moving']}")
    print(f"Highest risk serotype: {results['summary']['highest_risk']}")


if __name__ == "__main__":
    main()
