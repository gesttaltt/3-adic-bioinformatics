#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
"""
Validate HIV Analysis Setup

Checks that all prerequisites are correctly configured and validates
the quality of the codon encoder and embeddings.

Usage:
    python src/scripts/validate_hiv_setup.py
    python src/scripts/validate_hiv_setup.py --verbose
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def check_file_exists(path, name, required=True):
    """Check if a file exists and report status."""
    exists = path.exists()
    status = "OK" if exists else ("MISSING" if required else "OPTIONAL")
    symbol = "[+]" if exists else ("[-]" if required else "[?]")
    print(f"  {symbol} {name}: {status}")
    if exists:
        size_kb = path.stat().st_size / 1024
        print(f"      Size: {size_kb:.1f} KB")
    return exists


def validate_embeddings(verbose=False):
    """Validate the extracted embeddings."""
    print("\n[2] Validating Embeddings...")

    embeddings_path = PROJECT_ROOT / "research" / "bioinformatics" / "genetic_code" / "data" / "v5_11_3_embeddings.pt"

    if not embeddings_path.exists():
        print("  [-] Embeddings file not found!")
        return False

    try:
        embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=False)

        # Check required keys
        required_keys = ["z_A_hyp", "z_B_hyp", "valuations", "metadata"]
        for key in required_keys:
            if key not in embeddings:
                print(f"  [-] Missing key: {key}")
                return False

        z_B = embeddings["z_B_hyp"]
        valuations = embeddings["valuations"].numpy()
        metadata = embeddings["metadata"]

        # Validate shape
        if z_B.shape[0] != 19683:
            print(f"  [-] Wrong number of embeddings: {z_B.shape[0]} (expected 19683)")
            return False

        print(f"  [+] Embeddings shape: {z_B.shape}")

        # Validate radii
        radii = torch.norm(z_B, dim=1).numpy()
        print(f"  [+] Radii range: [{radii.min():.4f}, {radii.max():.4f}]")

        if radii.max() > 1.0:
            print("  [-] Warning: Some radii exceed 1.0 (outside Poincaré ball)")

        # Validate hierarchy correlation
        corr, p = spearmanr(valuations, radii)
        print(f"  [+] Hierarchy correlation: {corr:.4f} (p={p:.2e})")

        if corr > -0.5:
            print("  [-] Warning: Weak hierarchy correlation (expected < -0.5)")

        # Check valuation distribution
        if verbose:
            print("\n  Valuation distribution:")
            for v in range(10):
                mask = valuations == v
                count = mask.sum()
                mean_r = radii[mask].mean() if count > 0 else 0
                print(f"    v={v}: n={count:5d}, mean_radius={mean_r:.4f}")

        print(f"  [+] Metadata: {metadata.get('model_version', 'unknown')}")

        return True

    except Exception as e:
        print(f"  [-] Error loading embeddings: {e}")
        return False


def validate_natural_positions(verbose=False):
    """Validate natural positions JSON."""
    print("\n[3] Validating Natural Positions...")

    positions_path = (
        PROJECT_ROOT / "research" / "bioinformatics" / "genetic_code" / "data" / "natural_positions_v5_11_3.json"
    )

    if not positions_path.exists():
        print("  [-] Natural positions file not found!")
        return False

    try:
        with open(positions_path) as f:
            data = json.load(f)

        positions = data["positions"]
        data["labels"]
        clusters = data["clusters"]

        print(f"  [+] Positions: {len(positions)}")
        print(f"  [+] Clusters: {len(clusters)}")

        # Validate cluster sizes match genetic code
        expected_sizes = [6, 6, 6, 4, 4, 4, 4, 4, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1]
        actual_sizes = [len(v) for v in clusters.values()]

        if sorted(actual_sizes, reverse=True) == expected_sizes:
            print("  [+] Cluster sizes match genetic code degeneracy")
        else:
            print("  [-] Warning: Cluster sizes don't match expected pattern")

        # Check separation ratio
        sep_ratio = data["metadata"].get("separation_ratio", 0)
        print(f"  [+] Separation ratio: {sep_ratio:.2f}x")

        if sep_ratio < 1.5:
            print("  [-] Warning: Low separation ratio (clusters may overlap)")

        if verbose:
            print("\n  Cluster details:")
            for cid, pos_list in list(clusters.items())[:5]:
                print(f"    Cluster {cid}: {len(pos_list)} positions")

        return True

    except Exception as e:
        print(f"  [-] Error loading positions: {e}")
        return False


def validate_codon_encoder(verbose=False):
    """Validate the trained codon encoder."""
    print("\n[4] Validating Codon Encoder...")

    encoder_paths = [
        PROJECT_ROOT / "research" / "bioinformatics" / "genetic_code" / "data" / "codon_encoder_3adic.pt",
        PROJECT_ROOT
        / "research"
        / "bioinformatics"
        / "codon_encoder_research"
        / "hiv"
        / "data"
        / "codon_encoder_3adic.pt",
    ]

    encoder_path = None
    for path in encoder_paths:
        if path.exists():
            encoder_path = path
            break

    if encoder_path is None:
        print("  [-] Codon encoder not found!")
        return False

    try:
        checkpoint = torch.load(encoder_path, map_location="cpu", weights_only=False)

        # Check required keys
        required_keys = ["model_state", "codon_to_position", "metadata"]
        for key in required_keys:
            if key not in checkpoint:
                print(f"  [-] Missing key: {key}")
                return False

        metadata = checkpoint["metadata"]
        codon_map = checkpoint["codon_to_position"]

        print(f"  [+] Codon mappings: {len(codon_map)}")

        cluster_acc = metadata.get("cluster_accuracy", 0)
        synonymous_acc = metadata.get("synonymous_accuracy", 0)

        print(f"  [+] Cluster accuracy: {cluster_acc * 100:.1f}%")
        print(f"  [+] Synonymous accuracy: {synonymous_acc * 100:.1f}%")

        if cluster_acc < 0.7:
            print("  [-] Warning: Low cluster accuracy (< 70%)")

        if synonymous_acc < 0.95:
            print("  [-] Warning: Low synonymous accuracy (< 95%)")

        # Validate codon mappings
        expected_codons = set(
            [
                "TTT",
                "TTC",
                "TTA",
                "TTG",
                "TCT",
                "TCC",
                "TCA",
                "TCG",
                "TAT",
                "TAC",
                "TAA",
                "TAG",
                "TGT",
                "TGC",
                "TGA",
                "TGG",
                "CTT",
                "CTC",
                "CTA",
                "CTG",
                "CCT",
                "CCC",
                "CCA",
                "CCG",
                "CAT",
                "CAC",
                "CAA",
                "CAG",
                "CGT",
                "CGC",
                "CGA",
                "CGG",
                "ATT",
                "ATC",
                "ATA",
                "ATG",
                "ACT",
                "ACC",
                "ACA",
                "ACG",
                "AAT",
                "AAC",
                "AAA",
                "AAG",
                "AGT",
                "AGC",
                "AGA",
                "AGG",
                "GTT",
                "GTC",
                "GTA",
                "GTG",
                "GCT",
                "GCC",
                "GCA",
                "GCG",
                "GAT",
                "GAC",
                "GAA",
                "GAG",
                "GGT",
                "GGC",
                "GGA",
                "GGG",
            ]
        )

        actual_codons = set(codon_map.keys())
        if actual_codons == expected_codons:
            print("  [+] All 64 codons mapped")
        else:
            missing = expected_codons - actual_codons
            extra = actual_codons - expected_codons
            if missing:
                print(f"  [-] Missing codons: {missing}")
            if extra:
                print(f"  [-] Extra codons: {extra}")

        if verbose:
            print("\n  Sample mappings:")
            for codon in list(codon_map.keys())[:5]:
                print(f"    {codon} -> position {codon_map[codon]}")

        return True

    except Exception as e:
        print(f"  [-] Error loading encoder: {e}")
        return False


def validate_results(verbose=False):
    """Validate existing analysis results."""
    print("\n[5] Validating Analysis Results...")

    results_dir = PROJECT_ROOT / "research" / "bioinformatics" / "codon_encoder_research" / "hiv" / "results"

    if not results_dir.exists():
        print("  [?] Results directory not found (run analysis first)")
        return True  # Not a failure, just not run yet

    escape_results = results_dir / "hiv_escape_results.json"
    resistance_results = results_dir / "hiv_resistance_results.json"

    results_found = 0

    if escape_results.exists():
        try:
            with open(escape_results) as f:
                data = json.load(f)
            n_mutations = data["summary"]["total_mutations"]
            crossing_rate = data["summary"]["boundary_crossing_rate"]
            print(f"  [+] Escape results: {n_mutations} mutations, {crossing_rate:.1%} boundary crossing")
            results_found += 1
        except Exception as e:
            print(f"  [-] Error reading escape results: {e}")
    else:
        print("  [?] Escape results not found (run: python src/scripts/run_hiv_analysis.py --escape)")

    if resistance_results.exists():
        try:
            with open(resistance_results) as f:
                data = json.load(f)
            n_mutations = len(data["results"])
            classes = list(data["summary_by_class"].keys())
            print(f"  [+] Resistance results: {n_mutations} mutations across {len(classes)} drug classes")
            results_found += 1
        except Exception as e:
            print(f"  [-] Error reading resistance results: {e}")
    else:
        print("  [?] Resistance results not found (run: python src/scripts/run_hiv_analysis.py --drug-resistance)")

    return True


def main():
    parser = argparse.ArgumentParser(description="Validate HIV Analysis Setup")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    args = parser.parse_args()

    print("=" * 60)
    print("HIV ANALYSIS SETUP VALIDATION")
    print("=" * 60)

    all_passed = True

    # 1. Check file existence
    print("\n[1] Checking Required Files...")

    files_to_check = [
        (PROJECT_ROOT / "results" / "checkpoints" / "v5_5" / "best.pt", "v5.5 checkpoint", True),
        (PROJECT_ROOT / "results" / "checkpoints" / "v5_11" / "best.pt", "v5.11 checkpoint", False),
        (PROJECT_ROOT / "results" / "checkpoints" / "v5_11_overnight" / "best.pt", "v5.11_overnight checkpoint", False),
        (
            PROJECT_ROOT / "research" / "bioinformatics" / "genetic_code" / "data" / "v5_11_3_embeddings.pt",
            "Embeddings",
            True,
        ),
        (
            PROJECT_ROOT / "research" / "bioinformatics" / "genetic_code" / "data" / "natural_positions_v5_11_3.json",
            "Natural positions",
            True,
        ),
        (
            PROJECT_ROOT / "research" / "bioinformatics" / "genetic_code" / "data" / "codon_encoder_3adic.pt",
            "Codon encoder",
            True,
        ),
    ]

    for path, name, required in files_to_check:
        exists = check_file_exists(path, name, required)
        if required and not exists:
            all_passed = False

    # 2. Validate embeddings
    if not validate_embeddings(args.verbose):
        all_passed = False

    # 3. Validate natural positions
    if not validate_natural_positions(args.verbose):
        all_passed = False

    # 4. Validate codon encoder
    if not validate_codon_encoder(args.verbose):
        all_passed = False

    # 5. Validate results (optional)
    validate_results(args.verbose)

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("VALIDATION PASSED")
        print("\nYou can now run:")
        print("  python src/scripts/run_hiv_analysis.py")
    else:
        print("VALIDATION FAILED")
        print("\nPlease run setup first:")
        print("  python src/scripts/setup/setup_hiv_analysis.py")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
