# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Primer Stability Scanner for Arbovirus Diagnostics.

This module scans viral genomes to identify regions with minimal hyperbolic
movement over time - ideal candidates for stable RT-PCR primers that won't
fail due to viral evolution.

Key Features:
1. Sliding window analysis across genome
2. Track positional variance in hyperbolic space over time
3. Identify windows with minimum variance (conserved regions)
4. Export primer candidates with stability scores

Usage:
    python src/scripts/analysis/primer_stability_scanner.py \
        --input data/raw/dengue_paraguay.fasta \
        --output results/primer_candidates.csv \
        --window_size 20
"""

from __future__ import annotations

import argparse
import csv
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
class PrimerCandidate:
    """A candidate primer region."""

    position: int  # Start position in genome
    sequence: str  # Nucleotide sequence
    length: int
    stability_score: float  # Higher = more stable (less variance)
    conservation_score: float  # Fraction of sequences with this exact region
    gc_content: float
    tm_estimate: float  # Estimated melting temperature
    n_sequences: int  # Number of sequences analyzed
    variance_over_time: float


def compute_gc_content(sequence: str) -> float:
    """Compute GC content of sequence."""
    sequence = sequence.upper()
    gc = sum(1 for b in sequence if b in "GC")
    return gc / len(sequence) if sequence else 0


def estimate_tm(sequence: str) -> float:
    """Estimate melting temperature using Wallace rule.

    Tm = 2(A+T) + 4(G+C) for oligos < 14bp
    Tm = 64.9 + 41*(G+C-16.4)/(A+T+G+C) for longer
    """
    sequence = sequence.upper()
    a = sequence.count("A")
    t = sequence.count("T")
    g = sequence.count("G")
    c = sequence.count("C")

    if len(sequence) < 14:
        return 2 * (a + t) + 4 * (g + c)
    else:
        return 64.9 + 41 * (g + c - 16.4) / (a + t + g + c)


def padic_window_embedding(window: str, p: int = 3) -> np.ndarray:
    """Compute p-adic embedding for a window."""
    base_map = {"A": 0, "T": 1, "U": 1, "G": 2, "C": 3}
    window = window.upper()

    # Convert to numeric representation
    indices = []
    for base in window:
        if base in base_map:
            indices.append(base_map[base])
        else:
            indices.append(0)

    if not indices:
        return np.zeros(4)

    # Compute p-adic features
    def padic_val(n: int) -> int:
        if n == 0:
            return 0
        v = 0
        while n % p == 0:
            v += 1
            n //= p
        return v

    combined = sum(idx * (4**i) for i, idx in enumerate(indices[:10]))
    valuation = padic_val(combined + 1)

    features = np.array(
        [
            np.mean(indices),
            np.std(indices) if len(indices) > 1 else 0,
            valuation,
            sum(1 for i in indices if i in [2, 3]) / len(indices),  # GC ratio
        ]
    )

    return features


def scan_genome_windows(
    sequence: str,
    window_size: int = 20,
    step: int = 1,
) -> list[tuple[int, str, np.ndarray]]:
    """Scan genome with sliding window and compute embeddings.

    Returns:
        List of (position, window_sequence, embedding)
    """
    results = []
    sequence = sequence.upper()

    for pos in range(0, len(sequence) - window_size + 1, step):
        window = sequence[pos : pos + window_size]

        # Skip windows with Ns or other ambiguous bases
        if any(b not in "ATGCU" for b in window):
            continue

        embedding = padic_window_embedding(window)
        results.append((pos, window, embedding))

    return results


def compute_window_stability(
    all_windows: dict[int, list[tuple[str, np.ndarray, str]]],
    min_sequences: int = 5,
) -> dict[int, tuple[float, float, str]]:
    """Compute stability score for each window position.

    Args:
        all_windows: Dict mapping position -> list of (sequence, embedding, year)
        min_sequences: Minimum sequences to consider a position

    Returns:
        Dict mapping position -> (stability_score, conservation_score, consensus)
    """
    stability_scores = {}

    for pos, data in all_windows.items():
        if len(data) < min_sequences:
            continue

        sequences = [d[0] for d in data]
        embeddings = np.array([d[1] for d in data])

        # Variance of embeddings (lower = more stable in hyperbolic space)
        embedding_variance = float(np.mean(np.var(embeddings, axis=0)))
        stability_score = 1.0 / (1.0 + embedding_variance)

        # Conservation score (fraction with consensus sequence)
        seq_counts = defaultdict(int)
        for s in sequences:
            seq_counts[s] += 1
        consensus = max(seq_counts.items(), key=lambda x: x[1])[0]
        conservation = seq_counts[consensus] / len(sequences)

        stability_scores[pos] = (stability_score, conservation, consensus)

    return stability_scores


def find_primer_candidates(
    fasta_path: Path,
    window_size: int = 20,
    top_n: int = 50,
    min_gc: float = 0.4,
    max_gc: float = 0.6,
    min_tm: float = 55.0,
    max_tm: float = 65.0,
) -> list[PrimerCandidate]:
    """Find top primer candidates from multi-sequence alignment.

    Args:
        fasta_path: Path to FASTA file
        window_size: Primer length to scan
        top_n: Number of top candidates to return
        min_gc, max_gc: GC content bounds
        min_tm, max_tm: Melting temperature bounds

    Returns:
        List of PrimerCandidate objects
    """
    if not HAS_BIOPYTHON:
        raise ImportError("Biopython required for FASTA parsing")

    # Collect windows from all sequences
    all_windows = defaultdict(list)  # position -> [(sequence, embedding, year)]

    print("Scanning sequences...")
    n_sequences = 0
    for record in SeqIO.parse(fasta_path, "fasta"):
        seq = str(record.seq)
        windows = scan_genome_windows(seq, window_size)

        # Parse year from header
        parts = record.id.split("|")
        year = parts[2] if len(parts) > 2 else "unknown"

        for pos, window_seq, embedding in windows:
            all_windows[pos].append((window_seq, embedding, year))

        n_sequences += 1

    print(f"Processed {n_sequences} sequences, {len(all_windows)} window positions")

    # Compute stability scores
    print("Computing stability scores...")
    stability = compute_window_stability(all_windows)

    # Create candidates
    candidates = []
    for pos, (stab_score, cons_score, consensus) in stability.items():
        # Filter by GC content
        gc = compute_gc_content(consensus)
        if gc < min_gc or gc > max_gc:
            continue

        # Filter by Tm
        tm = estimate_tm(consensus)
        if tm < min_tm or tm > max_tm:
            continue

        # Compute variance over time
        data = all_windows[pos]
        years = set(d[2] for d in data)
        if len(years) > 1:
            by_year = defaultdict(list)
            for seq, emb, year in data:
                by_year[year].append(emb)
            year_centroids = [np.mean(embs, axis=0) for embs in by_year.values()]
            variance_over_time = float(np.mean(np.var(year_centroids, axis=0)))
        else:
            variance_over_time = 0.0

        candidate = PrimerCandidate(
            position=pos,
            sequence=consensus,
            length=window_size,
            stability_score=stab_score,
            conservation_score=cons_score,
            gc_content=gc,
            tm_estimate=tm,
            n_sequences=len(data),
            variance_over_time=variance_over_time,
        )
        candidates.append(candidate)

    # Sort by combined score (stability * conservation)
    candidates.sort(key=lambda x: x.stability_score * x.conservation_score, reverse=True)

    return candidates[:top_n]


def export_candidates(candidates: list[PrimerCandidate], output_path: Path) -> None:
    """Export primer candidates to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rank",
                "position",
                "sequence",
                "length",
                "stability_score",
                "conservation_score",
                "combined_score",
                "gc_content",
                "tm_estimate",
                "variance_over_time",
                "n_sequences",
            ]
        )

        for i, c in enumerate(candidates, 1):
            writer.writerow(
                [
                    i,
                    c.position,
                    c.sequence,
                    c.length,
                    f"{c.stability_score:.4f}",
                    f"{c.conservation_score:.4f}",
                    f"{c.stability_score * c.conservation_score:.4f}",
                    f"{c.gc_content:.3f}",
                    f"{c.tm_estimate:.1f}",
                    f"{c.variance_over_time:.6f}",
                    c.n_sequences,
                ]
            )

    print(f"Exported {len(candidates)} primer candidates to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Primer stability scanner for arbovirus diagnostics")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/dengue_paraguay.fasta",
        help="Input FASTA file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/primer_candidates.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=20,
        help="Primer window size (nt)",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=50,
        help="Number of top candidates to export",
    )
    parser.add_argument(
        "--min_gc",
        type=float,
        default=0.4,
        help="Minimum GC content",
    )
    parser.add_argument(
        "--max_gc",
        type=float,
        default=0.6,
        help="Maximum GC content",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Run ingest_arboviruses.py first to download sequences.")
        return

    candidates = find_primer_candidates(
        fasta_path=input_path,
        window_size=args.window_size,
        top_n=args.top_n,
        min_gc=args.min_gc,
        max_gc=args.max_gc,
    )

    if not candidates:
        print("No primer candidates found matching criteria.")
        return

    # Print top candidates
    print("\n=== Top 10 Primer Candidates ===")
    for i, c in enumerate(candidates[:10], 1):
        print(f"{i:2d}. Pos {c.position:5d}: {c.sequence}")
        print(
            f"    Stability={c.stability_score:.3f} "
            f"Conservation={c.conservation_score:.3f} "
            f"GC={c.gc_content:.2f} Tm={c.tm_estimate:.1f}"
        )

    # Export
    export_candidates(candidates, Path(args.output))


if __name__ == "__main__":
    main()
