# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Download HIV drug resistance data from Stanford HIVDB.

This script downloads mutation-resistance data from the Stanford HIV
Drug Resistance Database (HIVDB), which provides curated datasets
linking genotype (mutations) to phenotype (drug resistance).

Usage:
    python src/scripts/ingest/download_hivdb.py
    python src/scripts/ingest/download_hivdb.py --output-dir data/hiv/raw

Data Sources:
    - Stanford HIVDB: https://hivdb.stanford.edu/
    - HIVdb API: https://hivdb.stanford.edu/graphql

Genes:
    - RT: Reverse Transcriptase (targets: NRTIs, NNRTIs)
    - PR: Protease (targets: PIs)
    - IN: Integrase (targets: INSTIs)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import requests

# Add project root to path
root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Stanford HIVDB API endpoint
HIVDB_API_URL = "https://hivdb.stanford.edu/graphql"

# Drug class mappings
DRUG_CLASSES = {
    "RT": {
        "NRTI": ["3TC", "ABC", "AZT", "D4T", "DDI", "FTC", "TDF"],
        "NNRTI": ["DOR", "EFV", "ETR", "NVP", "RPV"],
    },
    "PR": {
        "PI": ["ATV", "DRV", "FPV", "IDV", "LPV", "NFV", "SQV", "TPV"],
    },
    "IN": {
        "INSTI": ["BIC", "CAB", "DTG", "EVG", "RAL"],
    },
}

# Known resistance mutations (from HIVDB)
# This is a subset - the full database has thousands
KNOWN_MUTATIONS = {
    "RT": {
        # NRTI resistance
        "M184V": {"3TC": "high", "FTC": "high", "ABC": "low"},
        "M184I": {"3TC": "high", "FTC": "high"},
        "K65R": {"TDF": "high", "ABC": "moderate", "3TC": "low"},
        "K70E": {"TDF": "moderate"},
        "L74V": {"ABC": "moderate", "DDI": "moderate"},
        "Y115F": {"ABC": "low"},
        "Q151M": {"AZT": "high", "D4T": "high", "ABC": "high", "DDI": "high"},
        "T69ins": {"all_nrti": "high"},
        # TAMs (Thymidine Analog Mutations)
        "M41L": {"AZT": "moderate", "D4T": "moderate"},
        "D67N": {"AZT": "moderate", "D4T": "moderate"},
        "K70R": {"AZT": "moderate", "D4T": "moderate"},
        "L210W": {"AZT": "moderate", "D4T": "moderate"},
        "T215Y": {"AZT": "high", "D4T": "high"},
        "T215F": {"AZT": "high", "D4T": "high"},
        "K219Q": {"AZT": "moderate", "D4T": "moderate"},
        # NNRTI resistance
        "K103N": {"EFV": "high", "NVP": "high"},
        "K103S": {"EFV": "moderate", "NVP": "moderate"},
        "Y181C": {"NVP": "high", "EFV": "low"},
        "Y188L": {"EFV": "high", "NVP": "high"},
        "G190A": {"NVP": "high", "EFV": "moderate"},
        "E138K": {"RPV": "high", "ETR": "moderate"},
    },
    "PR": {
        # Major PI resistance
        "D30N": {"NFV": "high"},
        "V32I": {"DRV": "moderate", "ATV": "moderate"},
        "M46I": {"ATV": "moderate", "LPV": "moderate"},
        "M46L": {"ATV": "moderate", "LPV": "moderate"},
        "I47A": {"LPV": "high", "DRV": "moderate"},
        "I47V": {"LPV": "moderate"},
        "G48V": {"SQV": "high"},
        "I50L": {"ATV": "high"},
        "I50V": {"DRV": "high", "LPV": "high"},
        "I54L": {"DRV": "moderate", "ATV": "moderate"},
        "I54M": {"DRV": "moderate", "ATV": "moderate"},
        "L76V": {"DRV": "moderate", "LPV": "moderate"},
        "V82A": {"IDV": "high", "LPV": "moderate"},
        "V82F": {"IDV": "high", "LPV": "moderate"},
        "V82T": {"IDV": "high", "LPV": "moderate"},
        "I84V": {"all_pi": "moderate"},
        "N88S": {"ATV": "high", "NFV": "high"},
        "L90M": {"NFV": "high", "SQV": "moderate"},
    },
    "IN": {
        # INSTI resistance
        "T66I": {"EVG": "high"},
        "T66K": {"EVG": "high"},
        "E92Q": {"EVG": "high", "RAL": "moderate"},
        "G118R": {"RAL": "moderate", "DTG": "moderate"},
        "E138K": {"EVG": "low", "RAL": "low"},
        "G140S": {"RAL": "high", "EVG": "moderate"},
        "Q148H": {"RAL": "high", "EVG": "high", "DTG": "moderate"},
        "Q148K": {"RAL": "high", "EVG": "high", "DTG": "moderate"},
        "Q148R": {"RAL": "high", "EVG": "high", "DTG": "moderate"},
        "N155H": {"RAL": "high", "EVG": "high"},
        "R263K": {"DTG": "moderate"},
    },
}

# Reference sequences (HXB2 positions)
REFERENCE_SEQUENCES = {
    "RT": "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDKEFRKYTAFTIPSTNNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGLTTPDKKHQKEPPFLWMGYELHPDKWTVQPIMLPEKDSWTVNDIQKLVGKLNWASQIYPGIKVRQLCKLLRGAKALTEVIPLTEEAELELAENREILKEPVHGVYYDPSKDLIAEIQKQGQGQWTYQIYQEPFKNLKTGKYARTRGAHTNDVKQLTEAVQKIATESIVIWGKTPKFRLPIQKETWETWWTEYWQATWIPEWEFVNTPPLVKLWYQLEKEPIIGAETFYVDGAANRETKLGKAGYVTDRGRQKVVPLTDTTNQKTELQAIHLALQDSGLEVNIVTDSQYALGIIQAQPDRSESEVVNQIIEE",
    "PR": "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF",
    "IN": "FLDGIDKAQDEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGKVILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTVHTDNGSNFTSTTVKAACWWAGIKQEFGIPYNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGIGGYSAGERIVDIIATDIQTKELQKQITKIQNFRVYYRDSRDPLWKGPAKLLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASRQDED",
}


def query_hivdb(query: str, variables: dict | None = None) -> dict[str, Any]:
    """Execute a GraphQL query against HIVDB API.

    Args:
        query: GraphQL query string
        variables: Optional query variables

    Returns:
        JSON response data
    """
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    response = requests.post(HIVDB_API_URL, json=payload, timeout=30)
    response.raise_for_status()

    data = response.json()
    if "errors" in data:
        raise ValueError(f"HIVDB API error: {data['errors']}")

    return data.get("data", {})


def get_mutation_scores() -> dict[str, dict]:
    """Get mutation resistance scores from HIVDB.

    Returns:
        Dictionary of gene -> mutation -> drug -> score
    """
    query = """
    query MutationScores {
        mutationScoresAll {
            gene { name }
            mutations { text }
            drugScores {
                drug { name }
                score
                level
            }
        }
    }
    """

    try:
        data = query_hivdb(query)
        scores = {}

        for item in data.get("mutationScoresAll", []):
            gene = item["gene"]["name"]
            if gene not in scores:
                scores[gene] = {}

            for mut in item["mutations"]:
                mut_text = mut["text"]
                if mut_text not in scores[gene]:
                    scores[gene][mut_text] = {}

                for drug_score in item["drugScores"]:
                    drug = drug_score["drug"]["name"]
                    scores[gene][mut_text][drug] = {
                        "score": drug_score["score"],
                        "level": drug_score["level"],
                    }

        return scores

    except Exception as e:
        logger.warning(f"Failed to query HIVDB API: {e}")
        logger.info("Using local mutation database instead")
        return KNOWN_MUTATIONS


def generate_synthetic_sequences(
    gene: str,
    n_samples: int = 1000,
    max_mutations: int = 5,
    seed: int = 42,
) -> list[tuple[str, list[str], dict[str, float]]]:
    """Generate synthetic sequences with mutations.

    Args:
        gene: Gene name (RT, PR, IN)
        n_samples: Number of sequences to generate
        max_mutations: Maximum mutations per sequence
        seed: Random seed

    Returns:
        List of (sequence, mutations, resistance_scores) tuples
    """
    import numpy as np

    np.random.seed(seed)

    reference = REFERENCE_SEQUENCES.get(gene, "")
    mutations = KNOWN_MUTATIONS.get(gene, {})

    if not reference or not mutations:
        return []

    samples = []
    mutation_list = list(mutations.keys())

    # Wild-type
    samples.append((reference, [], {drug: 0.0 for drug in get_drugs_for_gene(gene)}))

    for _i in range(n_samples - 1):
        # Pick random number of mutations
        n_muts = np.random.randint(1, min(max_mutations + 1, len(mutation_list) + 1))
        selected_muts = np.random.choice(mutation_list, n_muts, replace=False)

        # Apply mutations to sequence
        seq = list(reference)
        applied_muts = []
        resistance = {drug: 0.0 for drug in get_drugs_for_gene(gene)}

        for mut in selected_muts:
            # Parse mutation (e.g., "M184V" -> position 184, M->V)
            pos = int("".join(c for c in mut if c.isdigit()))
            new_aa = mut[-1]

            if pos <= len(seq):
                seq[pos - 1] = new_aa
                applied_muts.append(mut)

                # Add resistance scores
                for drug, effect in mutations[mut].items():
                    if drug.startswith("all_"):
                        # Apply to all drugs of that class
                        for d in get_drugs_for_gene(gene):
                            score = {"high": 0.9, "moderate": 0.5, "low": 0.2}.get(effect, 0.3)
                            resistance[d] = max(resistance[d], score)
                    else:
                        score = {"high": 0.9, "moderate": 0.5, "low": 0.2}.get(effect, 0.3)
                        resistance[drug] = max(resistance.get(drug, 0), score)

        samples.append(("".join(seq), applied_muts, resistance))

    return samples


def get_drugs_for_gene(gene: str) -> list[str]:
    """Get list of drugs targeting a gene."""
    drugs = []
    for _drug_class, drug_list in DRUG_CLASSES.get(gene, {}).items():
        drugs.extend(drug_list)
    return drugs


def save_dataset(
    samples: list[tuple[str, list[str], dict[str, float]]],
    gene: str,
    output_dir: Path,
) -> Path:
    """Save dataset to files.

    Args:
        samples: List of (sequence, mutations, resistance) tuples
        gene: Gene name
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    import csv

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all drugs
    drugs = get_drugs_for_gene(gene)

    # Save as CSV
    csv_path = output_dir / f"{gene.lower()}_resistance.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # Header
        header = ["sequence_id", "sequence", "mutations"] + [f"{d}_resistance" for d in drugs]
        writer.writerow(header)

        for i, (seq, muts, resistance) in enumerate(samples):
            row = [
                f"{gene}_{i:04d}",
                seq,
                ",".join(muts) if muts else "WT",
            ]
            for drug in drugs:
                row.append(f"{resistance.get(drug, 0.0):.3f}")
            writer.writerow(row)

    logger.info(f"Saved {len(samples)} samples to {csv_path}")
    return csv_path


def create_train_test_split(
    csv_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, Path]:
    """Create train/val/test splits.

    Args:
        csv_path: Path to full dataset CSV
        output_dir: Output directory for splits
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        seed: Random seed

    Returns:
        Dictionary of split name -> file path
    """
    import numpy as np
    import pandas as pd

    np.random.seed(seed)

    df = pd.read_csv(csv_path)
    n = len(df)

    # Shuffle indices
    indices = np.random.permutation(n)

    # Split
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": df.iloc[indices[:train_end]],
        "val": df.iloc[indices[train_end:val_end]],
        "test": df.iloc[indices[val_end:]],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for name, split_df in splits.items():
        path = output_dir / f"{name}.csv"
        split_df.to_csv(path, index=False)
        paths[name] = path
        logger.info(f"  {name}: {len(split_df)} samples -> {path}")

    return paths


def main():
    parser = argparse.ArgumentParser(description="Download HIVDB resistance data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/hiv"),
        help="Output directory (default: data/hiv)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples per gene (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--genes",
        nargs="+",
        choices=["RT", "PR", "IN"],
        default=["RT", "PR", "IN"],
        help="Genes to process (default: all)",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("HIVDB Data Download & Generation")
    logger.info("=" * 60)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Samples per gene: {args.n_samples}")
    logger.info(f"Genes: {', '.join(args.genes)}")

    # Try to get real mutation scores from API
    logger.info("\nFetching mutation scores from HIVDB...")
    get_mutation_scores()

    for gene in args.genes:
        logger.info(f"\nProcessing {gene}...")

        # Generate synthetic sequences
        samples = generate_synthetic_sequences(
            gene=gene,
            n_samples=args.n_samples,
            seed=args.seed,
        )

        if not samples:
            logger.warning(f"  No samples generated for {gene}")
            continue

        # Save full dataset
        csv_path = save_dataset(
            samples=samples,
            gene=gene,
            output_dir=args.output_dir / "processed",
        )

        # Create splits
        logger.info("  Creating train/val/test splits...")
        create_train_test_split(
            csv_path=csv_path,
            output_dir=args.output_dir / "splits" / gene.lower(),
            seed=args.seed,
        )

    # Save reference sequences
    ref_path = args.output_dir / "raw" / "reference_sequences.json"
    ref_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ref_path, "w") as f:
        json.dump(REFERENCE_SEQUENCES, f, indent=2)
    logger.info(f"\nSaved reference sequences to {ref_path}")

    # Save mutation database
    mut_path = args.output_dir / "raw" / "known_mutations.json"
    with open(mut_path, "w") as f:
        json.dump(KNOWN_MUTATIONS, f, indent=2)
    logger.info(f"Saved mutation database to {mut_path}")

    logger.info("\n" + "=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
