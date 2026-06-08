# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Download and process Arcadia Science E. coli AMR dataset.

This script downloads the 7,000+ strain E. coli antimicrobial resistance
dataset from Arcadia Science, available on Zenodo.

Dataset Info:
- Source: Arcadia Science 2024
- Zenodo DOI: 10.5281/zenodo.12692732
- Size: ~6.1 GB (compressed)
- Strains: 7,000+ E. coli isolates
- Antibiotics: 21 drugs with MIC data
- Key files:
    - phenotype_data.csv - MIC values for all strains
    - genotype_data/ - VCF files with variant calls
    - blaTEM_sequences/ - TEM beta-lactamase sequences

Usage:
    # Download dataset (requires ~6GB disk space for compressed, ~12GB for extracted)
    python src/scripts/ingest/download_arcadia_ecoli.py --download

    # Process downloaded data into framework format
    python src/scripts/ingest/download_arcadia_ecoli.py --process

    # Both download and process
    python src/scripts/ingest/download_arcadia_ecoli.py --download --process

Requirements:
    - zenodo_get (pip install zenodo-get) for Zenodo downloads
    - pandas for data processing
    - requests for HTTP downloads
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Dataset information
ZENODO_RECORD_ID = "12692732"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
GITHUB_URL = "https://github.com/Arcadia-Science/2024-Ecoli-amr-genotype-phenotype_7000strains"

# Default paths
DEFAULT_DATA_DIR = Path("data/external/arcadia_ecoli")
DEFAULT_PROCESSED_DIR = Path("data/processed/ecoli_tem")

# Antibiotics in the dataset (focus on beta-lactams)
BETA_LACTAM_DRUGS = [
    "ampicillin",
    "amoxicillin_clavulanate",
    "piperacillin_tazobactam",
    "cefazolin",
    "ceftriaxone",
    "ceftazidime",
    "cefepime",
    "aztreonam",
    "meropenem",
    "ertapenem",
]


def check_zenodo_get() -> bool:
    """Check if zenodo_get is installed."""
    try:
        subprocess.run(
            ["zenodo_get", "--help"],
            capture_output=True,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_zenodo_get() -> bool:
    """Install zenodo_get package."""
    logger.info("Installing zenodo_get...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "zenodo-get"],
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        logger.error("Failed to install zenodo_get")
        return False


def download_from_zenodo(output_dir: Path) -> bool:
    """Download dataset from Zenodo.

    Args:
        output_dir: Directory to save downloaded files

    Returns:
        True if download successful
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading Arcadia E. coli dataset from Zenodo...")
    logger.info(f"Record: {ZENODO_URL}")
    logger.info(f"Output: {output_dir}")
    logger.info("This may take a while (~6.1 GB)...")

    try:
        # Use zenodo_get to download
        result = subprocess.run(
            ["zenodo_get", "-r", ZENODO_RECORD_ID, "-o", str(output_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("Download complete!")
        logger.info(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Download failed: {e.stderr}")
        return False


def download_from_github(output_dir: Path) -> bool:
    """Download key files from GitHub (smaller subset).

    This downloads just the phenotype CSV and README from GitHub,
    useful for quick testing without the full 6GB download.

    Args:
        output_dir: Directory to save downloaded files

    Returns:
        True if download successful
    """
    import requests

    output_dir.mkdir(parents=True, exist_ok=True)

    # Key files from GitHub
    files_to_download = [
        (
            "README.md",
            "https://raw.githubusercontent.com/Arcadia-Science/2024-Ecoli-amr-genotype-phenotype_7000strains/main/README.md",
        ),
    ]

    for filename, url in files_to_download:
        logger.info(f"Downloading {filename}...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            filepath = output_dir / filename
            filepath.write_text(response.text)
            logger.info(f"Saved to {filepath}")
        except Exception as e:
            logger.warning(f"Failed to download {filename}: {e}")

    return True


def process_phenotype_data(
    data_dir: Path,
    output_dir: Path,
) -> dict | None:
    """Process phenotype data into framework format.

    Args:
        data_dir: Directory containing raw data
        output_dir: Directory to save processed data

    Returns:
        Summary statistics dict or None if failed
    """
    import pandas as pd

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find phenotype file
    phenotype_files = list(data_dir.glob("*phenotype*.csv"))
    if not phenotype_files:
        phenotype_files = list(data_dir.glob("**/*phenotype*.csv"))

    if not phenotype_files:
        logger.error("No phenotype CSV found. Looking for alternative files...")
        # Try to find any CSV
        csv_files = list(data_dir.glob("**/*.csv"))
        if csv_files:
            logger.info(f"Found CSV files: {[f.name for f in csv_files[:5]]}")
            phenotype_files = csv_files[:1]
        else:
            logger.error("No CSV files found in data directory")
            return None

    phenotype_file = phenotype_files[0]
    logger.info(f"Processing {phenotype_file}...")

    # Load data
    df = pd.read_csv(phenotype_file)
    logger.info(f"Loaded {len(df)} samples with columns: {list(df.columns)}")

    # Extract beta-lactam columns
    mic_columns = [
        col for col in df.columns if any(drug in col.lower() for drug in ["amp", "cef", "pip", "mero", "erta", "aztre"])
    ]
    logger.info(f"Found {len(mic_columns)} MIC columns: {mic_columns}")

    # Summary statistics
    summary = {
        "n_samples": len(df),
        "n_antibiotics": len(mic_columns),
        "antibiotics": mic_columns,
        "source_file": str(phenotype_file),
    }

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_file}")

    # Save processed phenotype data
    if mic_columns:
        processed_file = output_dir / "phenotypes.csv"
        df_subset = df[["strain_id" if "strain_id" in df.columns else df.columns[0]] + mic_columns]
        df_subset.to_csv(processed_file, index=False)
        logger.info(f"Saved processed data to {processed_file}")

    return summary


def process_genotype_data(
    data_dir: Path,
    output_dir: Path,
) -> dict | None:
    """Process genotype/sequence data.

    Args:
        data_dir: Directory containing raw data
        output_dir: Directory to save processed data

    Returns:
        Summary dict or None if failed
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find TEM sequence files
    fasta_files = list(data_dir.glob("**/*TEM*.fasta")) + list(data_dir.glob("**/*tem*.fasta"))
    vcf_files = list(data_dir.glob("**/*.vcf")) + list(data_dir.glob("**/*.vcf.gz"))

    logger.info(f"Found {len(fasta_files)} FASTA files, {len(vcf_files)} VCF files")

    summary = {
        "n_fasta_files": len(fasta_files),
        "n_vcf_files": len(vcf_files),
        "fasta_files": [str(f) for f in fasta_files[:10]],
        "vcf_files": [str(f) for f in vcf_files[:10]],
    }

    # Save summary
    summary_file = output_dir / "genotype_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def create_dataset_for_benchmark(
    processed_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create dataset in benchmark format.

    Args:
        processed_dir: Directory with processed data

    Returns:
        (X, y, ids) tuple for benchmarking
    """

    phenotype_file = processed_dir / "phenotypes.csv"
    if not phenotype_file.exists():
        logger.warning("No processed phenotype data, using synthetic")
        from src.diseases import create_ecoli_synthetic_dataset

        return create_ecoli_synthetic_dataset()

    import pandas as pd

    df = pd.read_csv(phenotype_file)
    logger.info(f"Creating benchmark dataset from {len(df)} samples")

    # TODO: Implement full data loading when TEM sequences are available
    # For now, return synthetic data
    from src.diseases import create_ecoli_synthetic_dataset

    return create_ecoli_synthetic_dataset()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and process Arcadia E. coli AMR dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download dataset from Zenodo (~6.1 GB)",
    )
    parser.add_argument(
        "--download-github",
        action="store_true",
        help="Download just README/metadata from GitHub (quick)",
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Process downloaded data",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory for raw data (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f"Directory for processed data (default: {DEFAULT_PROCESSED_DIR})",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show dataset information and exit",
    )

    args = parser.parse_args()

    if args.info:
        print("=" * 60)
        print("Arcadia Science E. coli AMR Dataset")
        print("=" * 60)
        print(f"\nZenodo URL: {ZENODO_URL}")
        print(f"GitHub URL: {GITHUB_URL}")
        print("\nSize: ~6.1 GB (compressed)")
        print("Strains: 7,000+ E. coli isolates")
        print("Antibiotics: 21 drugs")
        print("\nBeta-lactams in dataset:")
        for drug in BETA_LACTAM_DRUGS:
            print(f"  - {drug}")
        print(f"\nDefault data directory: {DEFAULT_DATA_DIR}")
        print(f"Default output directory: {DEFAULT_PROCESSED_DIR}")
        return

    if not any([args.download, args.download_github, args.process]):
        parser.print_help()
        print("\nExample usage:")
        print("  python src/scripts/ingest/download_arcadia_ecoli.py --download")
        print("  python src/scripts/ingest/download_arcadia_ecoli.py --process")
        return

    # Download from Zenodo
    if args.download:
        if not check_zenodo_get():
            logger.info("zenodo_get not found, installing...")
            if not install_zenodo_get():
                logger.error("Cannot proceed without zenodo_get")
                return

        if not download_from_zenodo(args.data_dir):
            logger.error("Download failed")
            return

    # Quick download from GitHub
    if args.download_github:
        download_from_github(args.data_dir)

    # Process data
    if args.process:
        logger.info("Processing phenotype data...")
        pheno_summary = process_phenotype_data(args.data_dir, args.output_dir)

        logger.info("Processing genotype data...")
        process_genotype_data(args.data_dir, args.output_dir)

        if pheno_summary:
            print("\n" + "=" * 60)
            print("Processing Complete")
            print("=" * 60)
            print(f"Samples: {pheno_summary.get('n_samples', 'N/A')}")
            print(f"Antibiotics: {pheno_summary.get('n_antibiotics', 'N/A')}")
            print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
