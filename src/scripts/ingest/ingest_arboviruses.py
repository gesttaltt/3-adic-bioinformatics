# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Arbovirus Genome Ingestion from NCBI Virus.

This script downloads arbovirus genomes (Dengue, Zika, Chikungunya) from NCBI
and processes them for hyperbolic trajectory analysis.

Data Sources:
- NCBI Datasets CLI for bulk genome download
- Entrez for metadata queries

Key Features:
1. Download genomes by taxon and geographic location
2. Extract collection dates for temporal analysis
3. Parse serotype information for Dengue
4. Output processed FASTA with standardized headers

Usage:
    python src/scripts/ingest/ingest_arboviruses.py \
        --virus dengue \
        --geo_location "Paraguay" \
        --output data/raw/dengue_paraguay.fasta
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.paths import RAW_DATA_DIR

try:
    from Bio import SeqIO
    from Bio.SeqRecord import SeqRecord

    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


# NCBI Taxonomy IDs for arboviruses
TAXON_IDS = {
    "dengue": 12637,  # Dengue virus (all serotypes)
    "dengue1": 11053,  # DENV-1
    "dengue2": 11060,  # DENV-2
    "dengue3": 11069,  # DENV-3
    "dengue4": 11070,  # DENV-4
    "zika": 64320,  # Zika virus
    "chikungunya": 37124,  # Chikungunya virus
}


@dataclass
class VirusMetadata:
    """Metadata for a virus sequence."""

    accession: str
    organism: str
    serotype: str | None
    collection_date: datetime | None
    geo_location: str | None
    host: str | None
    sequence_length: int


def check_ncbi_datasets_installed() -> bool:
    """Check if NCBI datasets CLI is installed."""
    try:
        result = subprocess.run(
            ["datasets", "--version"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def download_via_ncbi_datasets(
    taxon_id: int,
    output_dir: Path,
    geo_location: str | None = None,
    date_range: tuple[str, str] | None = None,
) -> Path | None:
    """Download virus genomes using NCBI datasets CLI.

    Args:
        taxon_id: NCBI taxonomy ID
        output_dir: Directory for downloaded data
        geo_location: Geographic filter (e.g., "Paraguay")
        date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format

    Returns:
        Path to downloaded FASTA or None if failed
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / "ncbi_dataset.zip"

    cmd = [
        "datasets",
        "download",
        "virus",
        "genome",
        "taxon",
        str(taxon_id),
        "--include",
        "genome",
        "--filename",
        str(zip_path),
    ]

    if geo_location:
        cmd.extend(["--geo-location", geo_location])

    print(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return None
    except FileNotFoundError:
        print("Error: NCBI datasets CLI not found.")
        print("Install with: winget install NCBI.Datasets")
        return None

    # Extract FASTA from ZIP
    if not zip_path.exists():
        print("Download failed - no ZIP file created")
        return None

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Find the genomic.fna file
            fasta_files = [f for f in zf.namelist() if f.endswith(".fna")]
            if not fasta_files:
                print("No FASTA files in download")
                return None

            # Extract to output directory
            fasta_path = output_dir / "genomes.fasta"
            with open(fasta_path, "wb") as out:
                for fname in fasta_files:
                    with zf.open(fname) as f:
                        out.write(f.read())

            return fasta_path
    except Exception as e:
        print(f"Error extracting: {e}")
        return None


def parse_collection_date(date_str: str) -> datetime | None:
    """Parse collection date from various formats."""
    if not date_str:
        return None

    # Try various formats
    formats = [
        "%Y-%m-%d",
        "%Y-%m",
        "%Y",
        "%d-%b-%Y",
        "%b-%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    return None


def extract_serotype(header: str) -> str | None:
    """Extract Dengue serotype from sequence header."""
    patterns = [
        r"DENV[-_]?(\d)",
        r"dengue virus (\d)",
        r"serotype (\d)",
        r"type (\d)",
    ]

    for pattern in patterns:
        match = re.search(pattern, header, re.IGNORECASE)
        if match:
            return f"DENV-{match.group(1)}"

    return None


def process_fasta(
    input_fasta: Path,
    output_fasta: Path,
    min_length: int = 5000,
    max_length: int = 15000,
) -> list[VirusMetadata]:
    """Process FASTA file and extract metadata.

    Args:
        input_fasta: Input FASTA path
        output_fasta: Output FASTA path (cleaned)
        min_length: Minimum sequence length filter
        max_length: Maximum sequence length filter

    Returns:
        List of metadata for accepted sequences
    """
    if not HAS_BIOPYTHON:
        print("Biopython required for FASTA processing")
        return []

    metadata_list = []
    accepted_records = []

    for record in SeqIO.parse(input_fasta, "fasta"):
        seq_len = len(record.seq)

        # Length filter
        if seq_len < min_length or seq_len > max_length:
            continue

        # Parse header for metadata
        header = record.description

        # Extract accession (first word)
        accession = record.id.split(".")[0]

        # Extract serotype
        serotype = extract_serotype(header)

        # Extract date (look for date patterns)
        date_match = re.search(r"(\d{4}[-/]\d{2}[-/]\d{2})", header)
        if not date_match:
            date_match = re.search(r"(\d{4})", header)
        collection_date = parse_collection_date(date_match.group(1) if date_match else "")

        # Extract location
        loc_patterns = [
            r"Paraguay",
            r"South America",
            r"Brazil",
            r"Argentina",
        ]
        geo_location = None
        for pat in loc_patterns:
            if re.search(pat, header, re.IGNORECASE):
                geo_location = pat
                break

        meta = VirusMetadata(
            accession=accession,
            organism=serotype or "Dengue virus",
            serotype=serotype,
            collection_date=collection_date,
            geo_location=geo_location,
            host="Homo sapiens",  # Default assumption
            sequence_length=seq_len,
        )
        metadata_list.append(meta)

        # Create standardized record
        new_header = f"{accession}|{serotype or 'unknown'}|{collection_date.year if collection_date else 'unknown'}"
        new_record = SeqRecord(
            record.seq,
            id=new_header,
            description="",
        )
        accepted_records.append(new_record)

    # Write processed FASTA
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(accepted_records, output_fasta, "fasta")

    print(f"Processed {len(accepted_records)} sequences")
    return metadata_list


def create_demo_data(output_path: Path, virus: str = "dengue") -> None:
    """Create demo FASTA data for testing when NCBI download fails."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate synthetic demo sequences
    demo_sequences = []
    for i in range(10):
        serotype = (i % 4) + 1
        year = 2015 + (i % 10)
        accession = f"DEMO{i:04d}"

        # Generate random-ish sequence
        import random

        random.seed(42 + i)
        seq = "".join(random.choices("ATGC", k=10000))

        header = f">{accession}|DENV-{serotype}|{year}"
        demo_sequences.append(f"{header}\n{seq}")

    with open(output_path, "w") as f:
        f.write("\n".join(demo_sequences))

    print(f"Created demo data at {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Download arbovirus genomes from NCBI")
    parser.add_argument(
        "--virus",
        type=str,
        choices=list(TAXON_IDS.keys()),
        default="dengue",
        help="Virus type to download",
    )
    parser.add_argument(
        "--geo_location",
        type=str,
        default=None,
        help="Geographic location filter (e.g., 'Paraguay')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RAW_DATA_DIR / "arboviruses.fasta"),
        help="Output FASTA path",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Create demo data instead of downloading",
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    if args.demo:
        create_demo_data(output_path, args.virus)
        return

    # Check for NCBI datasets CLI
    if not check_ncbi_datasets_installed():
        print("NCBI datasets CLI not found. Creating demo data instead.")
        print("Install with: winget install NCBI.Datasets")
        create_demo_data(output_path, args.virus)
        return

    # Get taxon ID
    taxon_id = TAXON_IDS[args.virus]
    print(f"Downloading {args.virus} (taxon {taxon_id})...")

    # Download
    cache_dir = RAW_DATA_DIR / "ncbi_cache"
    raw_fasta = download_via_ncbi_datasets(
        taxon_id=taxon_id,
        output_dir=cache_dir,
        geo_location=args.geo_location,
    )

    if raw_fasta is None:
        print("Download failed. Creating demo data.")
        create_demo_data(output_path, args.virus)
        return

    # Process
    print("Processing sequences...")
    metadata = process_fasta(raw_fasta, output_path)

    # Summary
    print("\n=== Summary ===")
    print(f"Total sequences: {len(metadata)}")

    if metadata:
        serotypes = {}
        years = {}
        for m in metadata:
            if m.serotype:
                serotypes[m.serotype] = serotypes.get(m.serotype, 0) + 1
            if m.collection_date:
                year = m.collection_date.year
                years[year] = years.get(year, 0) + 1

        print(f"By serotype: {serotypes}")
        print(f"By year: {dict(sorted(years.items()))}")


if __name__ == "__main__":
    main()
