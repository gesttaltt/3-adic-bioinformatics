# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""GISAID Integration Client for Viral Sequence Data.

GISAID (Global Initiative on Sharing All Influenza Data) hosts the world's
largest collection of influenza and SARS-CoV-2 sequences with associated
metadata.

This module provides:
- Authentication with GISAID EpiFlu/EpiCoV APIs
- Sequence download and caching
- Metadata parsing and normalization
- Integration with disease analyzers

Note:
    GISAID requires registration and adherence to their data sharing agreement.
    This client expects valid credentials to be configured.
    https://gisaid.org

Supported Databases:
- EpiFlu: Influenza sequences and metadata
- EpiCoV: SARS-CoV-2 sequences and metadata

Usage:
    from src.data.gisaid_client import GISAIDClient

    client = GISAIDClient(username="...", password="...")

    # Download influenza sequences
    sequences = client.download_influenza_sequences(
        subtype="H3N2",
        start_date="2023-01-01",
        end_date="2024-01-01",
    )

    # Download SARS-CoV-2 sequences
    sequences = client.download_sarscov2_sequences(
        lineage="BA.2.86",
        region="Europe",
    )
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class GISAIDDatabase(Enum):
    """GISAID database types."""

    EPIFLU = "epiflu"  # Influenza
    EPICOV = "epicov"  # SARS-CoV-2


class InfluenzaLineage(Enum):
    """Influenza lineage types."""

    H1N1 = "A/H1N1"
    H1N1_PDM09 = "A/H1N1pdm09"
    H3N2 = "A/H3N2"
    B_VICTORIA = "B/Victoria"
    B_YAMAGATA = "B/Yamagata"
    H5N1 = "A/H5N1"
    H5N2 = "A/H5N2"
    H7N9 = "A/H7N9"
    H9N2 = "A/H9N2"


class CoVLineage(Enum):
    """SARS-CoV-2 lineage types (selected)."""

    ALPHA = "B.1.1.7"
    BETA = "B.1.351"
    GAMMA = "P.1"
    DELTA = "B.1.617.2"
    OMICRON_BA1 = "BA.1"
    OMICRON_BA2 = "BA.2"
    OMICRON_BA4 = "BA.4"
    OMICRON_BA5 = "BA.5"
    OMICRON_XBB = "XBB"
    OMICRON_XBB15 = "XBB.1.5"
    OMICRON_JN1 = "JN.1"
    OMICRON_BA286 = "BA.2.86"


@dataclass
class SequenceRecord:
    """GISAID sequence record with metadata."""

    accession_id: str
    strain_name: str
    sequence: str
    collection_date: str | None = None
    submission_date: str | None = None
    location: str | None = None
    host: str | None = None
    lineage: str | None = None
    clade: str | None = None
    gene_segment: str | None = None
    originating_lab: str | None = None
    submitting_lab: str | None = None
    authors: str | None = None
    passage: str | None = None
    age: str | None = None
    gender: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accession_id": self.accession_id,
            "strain_name": self.strain_name,
            "sequence": self.sequence,
            "collection_date": self.collection_date,
            "submission_date": self.submission_date,
            "location": self.location,
            "host": self.host,
            "lineage": self.lineage,
            "clade": self.clade,
            "gene_segment": self.gene_segment,
            "originating_lab": self.originating_lab,
            "submitting_lab": self.submitting_lab,
            "authors": self.authors,
            "passage": self.passage,
            "age": self.age,
            "gender": self.gender,
            **self.metadata,
        }


@dataclass
class GISAIDConfig:
    """Configuration for GISAID client.

    Attributes:
        username: GISAID account username
        password: GISAID account password
        cache_dir: Directory for caching downloaded sequences
        rate_limit: Maximum requests per minute
        timeout: Request timeout in seconds
    """

    username: str | None = None
    password: str | None = None
    cache_dir: str = ".gisaid_cache"
    rate_limit: int = 30
    timeout: int = 60


class GISAIDClient:
    """Client for GISAID EpiFlu and EpiCoV databases.

    Provides methods to search, download, and cache viral sequences
    from GISAID databases.

    Note:
        This is a mock implementation demonstrating the interface.
        Actual GISAID API access requires registration and approval.
        In production, this would use the official GISAID API.
    """

    # GISAID API endpoints (mock - real endpoints require registration)
    BASE_URL = "https://gisaid.org/api"
    EPIFLU_ENDPOINT = f"{BASE_URL}/epiflu"
    EPICOV_ENDPOINT = f"{BASE_URL}/epicov"

    def __init__(
        self,
        config: GISAIDConfig | None = None,
        username: str | None = None,
        password: str | None = None,
    ):
        """Initialize GISAID client.

        Args:
            config: Configuration object
            username: GISAID username (overrides config)
            password: GISAID password (overrides config)
        """
        self.config = config or GISAIDConfig()

        # Override with explicit credentials
        if username:
            self.config.username = username
        if password:
            self.config.password = password

        # Try environment variables
        if not self.config.username:
            self.config.username = os.environ.get("GISAID_USERNAME")
        if not self.config.password:
            self.config.password = os.environ.get("GISAID_PASSWORD")

        # Setup cache directory
        self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Rate limiting
        self._last_request_time = 0.0
        self._min_interval = 60.0 / self.config.rate_limit

        # Session token (mock)
        self._token: str | None = None

    def authenticate(self) -> bool:
        """Authenticate with GISAID.

        Returns:
            True if authentication successful
        """
        if not self.config.username or not self.config.password:
            print("Warning: GISAID credentials not configured.")
            print("Set GISAID_USERNAME and GISAID_PASSWORD environment variables.")
            return False

        # Mock authentication - real implementation would use GISAID API
        # In production: POST to auth endpoint with credentials
        self._token = hashlib.sha256(f"{self.config.username}:{self.config.password}".encode()).hexdigest()[:32]

        return True

    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _get_cache_key(self, **kwargs) -> str:
        """Generate cache key from query parameters."""
        key_str = json.dumps(kwargs, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _check_cache(self, cache_key: str) -> list[SequenceRecord] | None:
        """Check if results are cached."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                return [SequenceRecord(**rec) for rec in data]
            except Exception:
                return None
        return None

    def _save_cache(self, cache_key: str, records: list[SequenceRecord]):
        """Save results to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump([rec.to_dict() for rec in records], f)

    def download_influenza_sequences(
        self,
        subtype: str | InfluenzaLineage = "H3N2",
        gene: str = "HA",
        start_date: str | None = None,
        end_date: str | None = None,
        location: str | None = None,
        host: str = "Human",
        max_sequences: int = 1000,
        min_length: int = 500,
        use_cache: bool = True,
    ) -> list[SequenceRecord]:
        """Download influenza sequences from GISAID EpiFlu.

        Args:
            subtype: Influenza subtype (H3N2, H1N1, etc.)
            gene: Gene segment (HA, NA, PA, etc.)
            start_date: Start of date range (YYYY-MM-DD)
            end_date: End of date range (YYYY-MM-DD)
            location: Geographic location filter
            host: Host species
            max_sequences: Maximum sequences to download
            min_length: Minimum sequence length
            use_cache: Whether to use cached results

        Returns:
            List of sequence records
        """
        if isinstance(subtype, InfluenzaLineage):
            subtype = subtype.value

        # Check cache
        cache_params = {
            "db": "epiflu",
            "subtype": subtype,
            "gene": gene,
            "start_date": start_date,
            "end_date": end_date,
            "location": location,
            "host": host,
        }
        cache_key = self._get_cache_key(**cache_params)

        if use_cache:
            cached = self._check_cache(cache_key)
            if cached:
                return cached[:max_sequences]

        # Rate limit
        self._rate_limit()

        # Mock implementation - in production would call GISAID API
        # Real implementation:
        # response = requests.post(
        #     self.EPIFLU_ENDPOINT,
        #     headers={"Authorization": f"Bearer {self._token}"},
        #     json={"subtype": subtype, "gene": gene, ...}
        # )

        print(f"Note: Mock implementation - would download {subtype}/{gene} sequences from GISAID")
        print("For real data, configure GISAID credentials and use official API")

        # Return mock sequences for demonstration
        records = self._generate_mock_influenza_sequences(
            subtype=subtype,
            gene=gene,
            n_sequences=min(50, max_sequences),
            start_date=start_date,
        )

        if use_cache:
            self._save_cache(cache_key, records)

        return records

    def download_sarscov2_sequences(
        self,
        lineage: str | CoVLineage | None = None,
        gene: str = "S",  # Spike protein
        start_date: str | None = None,
        end_date: str | None = None,
        region: str | None = None,
        country: str | None = None,
        max_sequences: int = 1000,
        complete_only: bool = True,
        high_coverage: bool = True,
        use_cache: bool = True,
    ) -> list[SequenceRecord]:
        """Download SARS-CoV-2 sequences from GISAID EpiCoV.

        Args:
            lineage: Pango lineage (BA.2.86, JN.1, etc.)
            gene: Gene/protein (S, N, M, E, ORF1ab, etc.)
            start_date: Start of date range
            end_date: End of date range
            region: Geographic region (Africa, Asia, Europe, etc.)
            country: Specific country
            max_sequences: Maximum sequences to download
            complete_only: Only download complete genomes
            high_coverage: Only high coverage sequences
            use_cache: Whether to use cached results

        Returns:
            List of sequence records
        """
        if isinstance(lineage, CoVLineage):
            lineage = lineage.value

        # Check cache
        cache_params = {
            "db": "epicov",
            "lineage": lineage,
            "gene": gene,
            "start_date": start_date,
            "end_date": end_date,
            "region": region,
            "country": country,
            "complete_only": complete_only,
        }
        cache_key = self._get_cache_key(**cache_params)

        if use_cache:
            cached = self._check_cache(cache_key)
            if cached:
                return cached[:max_sequences]

        self._rate_limit()

        # Mock implementation
        print(f"Note: Mock implementation - would download {lineage or 'all'}/{gene} sequences from GISAID")
        print("For real data, configure GISAID credentials and use official API")

        records = self._generate_mock_sarscov2_sequences(
            lineage=lineage,
            gene=gene,
            n_sequences=min(50, max_sequences),
            start_date=start_date,
        )

        if use_cache:
            self._save_cache(cache_key, records)

        return records

    def search_sequences(
        self,
        database: GISAIDDatabase,
        query: str,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Search for sequences in GISAID.

        Args:
            database: Database to search (EpiFlu or EpiCoV)
            query: Search query string
            max_results: Maximum results to return

        Returns:
            List of matching sequence metadata
        """
        self._rate_limit()

        # Mock search results
        return [
            {
                "accession_id": f"EPI_ISL_{1000000 + i}",
                "strain_name": f"Mock/Query/{i}",
                "collection_date": "2024-01-01",
                "location": "Mock Location",
            }
            for i in range(min(10, max_results))
        ]

    def download_by_accession(
        self,
        accession_ids: list[str],
        database: GISAIDDatabase = GISAIDDatabase.EPICOV,
    ) -> list[SequenceRecord]:
        """Download specific sequences by accession ID.

        Args:
            accession_ids: List of GISAID accession IDs
            database: Database to query

        Returns:
            List of sequence records
        """
        self._rate_limit()

        # Mock implementation
        records = []
        for acc_id in accession_ids:
            records.append(
                SequenceRecord(
                    accession_id=acc_id,
                    strain_name=f"Mock/Accession/{acc_id}",
                    sequence="M" + "A" * 100,
                    collection_date="2024-01-01",
                )
            )

        return records

    def export_to_fasta(
        self,
        records: list[SequenceRecord],
        output_path: str | Path,
        include_metadata: bool = True,
    ) -> Path:
        """Export sequences to FASTA format.

        Args:
            records: Sequence records to export
            output_path: Output file path
            include_metadata: Include metadata in headers

        Returns:
            Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for rec in records:
                if include_metadata:
                    header = f">{rec.accession_id}|{rec.strain_name}|{rec.collection_date}|{rec.lineage}"
                else:
                    header = f">{rec.accession_id}"
                f.write(f"{header}\n{rec.sequence}\n")

        return output_path

    def export_to_dataframe(
        self,
        records: list[SequenceRecord],
    ) -> pd.DataFrame:
        """Export sequences to pandas DataFrame.

        Args:
            records: Sequence records to export

        Returns:
            DataFrame with sequence data
        """
        return pd.DataFrame([rec.to_dict() for rec in records])

    def _generate_mock_influenza_sequences(
        self,
        subtype: str,
        gene: str,
        n_sequences: int,
        start_date: str | None = None,
    ) -> list[SequenceRecord]:
        """Generate mock influenza sequences for testing."""
        np.random.seed(42)
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"

        # Typical sequence lengths by gene
        gene_lengths = {
            "HA": 566,  # Hemagglutinin
            "NA": 469,  # Neuraminidase
            "PA": 716,  # Polymerase acidic
            "PB1": 757,  # Polymerase basic 1
            "PB2": 759,  # Polymerase basic 2
            "NP": 498,  # Nucleoprotein
            "M1": 252,  # Matrix 1
            "M2": 97,  # Matrix 2
            "NS1": 230,  # Non-structural 1
            "NS2": 121,  # Nuclear export protein
        }

        seq_length = gene_lengths.get(gene, 500)
        base_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime(2024, 1, 1)

        records = []
        for i in range(n_sequences):
            # Generate random sequence
            seq = "".join(np.random.choice(list(amino_acids), seq_length))

            # Random date offset
            date_offset = np.random.randint(0, 365)
            collection_date = base_date.replace(day=1) + pd.Timedelta(days=int(date_offset))

            # Mock location
            locations = ["USA/California", "China/Guangdong", "UK/England", "Japan/Tokyo", "Australia/Victoria"]

            records.append(
                SequenceRecord(
                    accession_id=f"EPI_ISL_{2000000 + i:07d}",
                    strain_name=f"A/{locations[i % len(locations)]}/{i:04d}/{collection_date.year}({subtype})",
                    sequence=seq,
                    collection_date=collection_date.strftime("%Y-%m-%d"),
                    submission_date=collection_date.strftime("%Y-%m-%d"),
                    location=locations[i % len(locations)],
                    host="Human",
                    lineage=subtype,
                    gene_segment=gene,
                    passage="Original",
                )
            )

        return records

    def _generate_mock_sarscov2_sequences(
        self,
        lineage: str | None,
        gene: str,
        n_sequences: int,
        start_date: str | None = None,
    ) -> list[SequenceRecord]:
        """Generate mock SARS-CoV-2 sequences for testing."""
        np.random.seed(42)
        amino_acids = "ACDEFGHIKLMNPQRSTVWY"

        # Typical protein lengths
        gene_lengths = {
            "S": 1273,  # Spike
            "N": 419,  # Nucleocapsid
            "M": 222,  # Membrane
            "E": 75,  # Envelope
            "ORF1ab": 7096,  # Full polyprotein
            "RdRp": 932,  # RNA-dependent RNA polymerase
        }

        seq_length = gene_lengths.get(gene, 500)
        base_date = datetime.strptime(start_date, "%Y-%m-%d") if start_date else datetime(2024, 1, 1)
        lineage = lineage or "BA.2.86"

        records = []
        for i in range(n_sequences):
            seq = "".join(np.random.choice(list(amino_acids), seq_length))

            date_offset = np.random.randint(0, 365)
            collection_date = base_date.replace(day=1) + pd.Timedelta(days=int(date_offset))

            countries = ["USA", "UK", "Germany", "France", "Japan", "India", "Brazil", "South Africa"]

            records.append(
                SequenceRecord(
                    accession_id=f"EPI_ISL_{3000000 + i:07d}",
                    strain_name=f"hCoV-19/{countries[i % len(countries)]}/{i:06d}/{collection_date.year}",
                    sequence=seq,
                    collection_date=collection_date.strftime("%Y-%m-%d"),
                    submission_date=collection_date.strftime("%Y-%m-%d"),
                    location=countries[i % len(countries)],
                    host="Human",
                    lineage=lineage,
                    clade=f"{lineage}_clade",
                    gene_segment=gene,
                )
            )

        return records


class GISAIDSequenceProcessor:
    """Processor for GISAID sequence data.

    Provides utilities for cleaning, filtering, and preparing
    sequences for analysis.
    """

    @staticmethod
    def filter_quality(
        records: list[SequenceRecord],
        min_length: int = 100,
        max_ambiguous_fraction: float = 0.05,
    ) -> list[SequenceRecord]:
        """Filter sequences by quality criteria.

        Args:
            records: Input sequence records
            min_length: Minimum sequence length
            max_ambiguous_fraction: Maximum fraction of X/N characters

        Returns:
            Filtered records
        """
        filtered = []
        for rec in records:
            seq = rec.sequence.upper()

            # Check length
            if len(seq) < min_length:
                continue

            # Check ambiguous characters
            n_ambiguous = seq.count("X") + seq.count("N") + seq.count("-")
            if n_ambiguous / len(seq) > max_ambiguous_fraction:
                continue

            filtered.append(rec)

        return filtered

    @staticmethod
    def align_sequences(
        records: list[SequenceRecord],
        reference: str | None = None,
    ) -> list[SequenceRecord]:
        """Align sequences to reference.

        Note: This is a placeholder. Real implementation would use
        tools like MAFFT or MUSCLE.

        Args:
            records: Input sequence records
            reference: Optional reference sequence

        Returns:
            Aligned records
        """
        # Placeholder - would use alignment tool
        return records

    @staticmethod
    def extract_mutations(
        records: list[SequenceRecord],
        reference: str,
    ) -> list[dict[str, Any]]:
        """Extract mutations relative to reference.

        Args:
            records: Sequence records
            reference: Reference sequence

        Returns:
            List of mutation annotations per record
        """
        results = []
        for rec in records:
            mutations = []
            seq = rec.sequence

            for i, (ref_aa, seq_aa) in enumerate(zip(reference, seq, strict=False)):
                if ref_aa != seq_aa and seq_aa not in "-X":
                    mutations.append(
                        {
                            "position": i + 1,
                            "ref": ref_aa,
                            "alt": seq_aa,
                            "notation": f"{ref_aa}{i + 1}{seq_aa}",
                        }
                    )

            results.append(
                {
                    "accession_id": rec.accession_id,
                    "mutations": mutations,
                    "n_mutations": len(mutations),
                }
            )

        return results


def create_gisaid_client(
    username: str | None = None,
    password: str | None = None,
    cache_dir: str = ".gisaid_cache",
) -> GISAIDClient:
    """Create a configured GISAID client.

    Args:
        username: GISAID username (or set GISAID_USERNAME env var)
        password: GISAID password (or set GISAID_PASSWORD env var)
        cache_dir: Directory for caching sequences

    Returns:
        Configured GISAIDClient instance
    """
    config = GISAIDConfig(
        username=username,
        password=password,
        cache_dir=cache_dir,
    )
    return GISAIDClient(config)
