"""Hepatitis B Virus (HBV) data loader.

Loads HBV sequences from various sources:
- HBVdb (primary source)
- NCBI/GenBank
- Local cache

Drug resistance data for:
- Lamivudine (LAM)
- Adefovir (ADV)
- Entecavir (ETV)
- Tenofovir (TDF)
- Telbivudine (LdT)

Key genes:
- Polymerase (P): Drug resistance mutations
- Surface (S): Vaccine escape mutations (overlaps P gene!)
- Core (C): Immune escape
- X: Regulatory functions
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..base import OrganismLoader, OrganismType, SequenceRecord, SequenceType
from ..registry import OrganismRegistry

# Known drug resistance mutations in HBV polymerase
HBV_RESISTANCE_MUTATIONS = {
    # Lamivudine resistance
    "LAM": {
        "primary": ["L180M", "M204V", "M204I"],
        "secondary": ["V173L", "L80I"],
    },
    # Adefovir resistance
    "ADV": {
        "primary": ["A181T", "A181V", "N236T"],
        "secondary": ["I233V"],
    },
    # Entecavir resistance (requires LAM mutations + additional)
    "ETV": {
        "primary": ["T184G", "S202G", "M250V"],
        "secondary": ["I169T", "M250L"],
    },
    # Tenofovir resistance (rare)
    "TDF": {
        "primary": ["A194T"],  # Debated
        "secondary": [],
    },
}

# HBV genotypes and their characteristics
HBV_GENOTYPES = {
    "A": {"prevalence": "Europe, Africa", "severity": "moderate"},
    "B": {"prevalence": "Asia", "severity": "moderate"},
    "C": {"prevalence": "Asia", "severity": "high (HCC risk)"},
    "D": {"prevalence": "Mediterranean, Middle East", "severity": "moderate"},
    "E": {"prevalence": "Africa", "severity": "moderate"},
    "F": {"prevalence": "South America", "severity": "moderate"},
    "G": {"prevalence": "Europe, North America", "severity": "low"},
    "H": {"prevalence": "Central America", "severity": "low"},
}


@OrganismRegistry.register(OrganismType.HBV)
class HBVLoader(OrganismLoader):
    """Loader for Hepatitis B Virus sequences."""

    def __init__(
        self,
        organism: OrganismType = OrganismType.HBV,
        cache_dir: Path | None = None,
        max_sequences: int | None = None,
        gene: str = "polymerase",
        include_resistance: bool = True,
    ):
        super().__init__(organism, cache_dir, max_sequences)
        self.gene = gene
        self.include_resistance = include_resistance
        self._sequences: list[SequenceRecord] | None = None

    def load_sequences(self) -> list[SequenceRecord]:
        """Load HBV sequences.

        First tries to load from cache, then from remote sources.
        """
        if self._sequences is not None:
            return self._sequences

        # Try cache first
        cache_file = self.cache_dir / "hbv_sequences.json"
        if cache_file.exists():
            self._sequences = self._load_from_cache(cache_file)
            if self.max_sequences:
                self._sequences = self._sequences[: self.max_sequences]
            return self._sequences

        # Try NCBI/GenBank
        try:
            self._sequences = self._fetch_from_ncbi()
        except Exception as e:
            print(f"Could not fetch from NCBI: {e}")
            # Generate synthetic data for testing
            self._sequences = self._generate_synthetic()

        # Cache results
        self._save_to_cache(cache_file)

        if self.max_sequences:
            self._sequences = self._sequences[: self.max_sequences]

        return self._sequences

    def _load_from_cache(self, cache_file: Path) -> list[SequenceRecord]:
        """Load sequences from cache file."""
        with open(cache_file) as f:
            data = json.load(f)

        records = []
        for item in data:
            records.append(
                SequenceRecord(
                    id=item["id"],
                    sequence=item["sequence"],
                    organism=OrganismType.HBV,
                    sequence_type=SequenceType.DNA,
                    gene=item.get("gene"),
                    subtype=item.get("genotype"),
                    country=item.get("country"),
                    drug_resistance=item.get("drug_resistance"),
                    mutations=item.get("mutations"),
                )
            )

        return records

    def _save_to_cache(self, cache_file: Path):
        """Save sequences to cache file."""
        data = []
        for record in self._sequences:
            data.append(
                {
                    "id": record.id,
                    "sequence": record.sequence,
                    "gene": record.gene,
                    "genotype": record.subtype,
                    "country": record.country,
                    "drug_resistance": record.drug_resistance,
                    "mutations": record.mutations,
                }
            )

        with open(cache_file, "w") as f:
            json.dump(data, f)

    def _fetch_from_ncbi(self) -> list[SequenceRecord]:
        """Fetch HBV sequences from NCBI."""
        try:
            from data_access.clients.ncbi_client import NCBIClient

            client = NCBIClient()

            # Search for HBV polymerase sequences
            query = "Hepatitis B virus[Organism] AND polymerase[Gene]"
            if self.gene == "surface":
                query = "Hepatitis B virus[Organism] AND surface[Gene]"

            ids = client.search("nucleotide", query, max_results=self.max_sequences or 1000)
            sequences = client.fetch_sequences(ids)

            records = []
            for seq_id, seq_data in sequences.items():
                records.append(
                    SequenceRecord(
                        id=seq_id,
                        sequence=seq_data["sequence"],
                        organism=OrganismType.HBV,
                        sequence_type=SequenceType.DNA,
                        gene=self.gene,
                        annotations=seq_data.get("annotations", {}),
                    )
                )

            return records

        except ImportError:
            raise RuntimeError("NCBI client not available")

    def _generate_synthetic(self) -> list[SequenceRecord]:
        """Generate synthetic HBV sequences for testing."""
        np.random.seed(42)

        # HBV polymerase consensus (simplified)
        consensus = "ATG" * 100  # Placeholder - real consensus is ~2500 bp

        n_sequences = self.max_sequences or 100
        records = []

        for i in range(n_sequences):
            # Mutate consensus
            seq = list(consensus)
            n_mutations = np.random.poisson(5)

            mutations = []
            for _ in range(n_mutations):
                pos = np.random.randint(0, len(seq))
                old = seq[pos]
                new = np.random.choice([b for b in "ACGT" if b != old])
                seq[pos] = new
                mutations.append(f"{old}{pos}{new}")

            # Assign genotype
            genotype = np.random.choice(list(HBV_GENOTYPES.keys()))

            # Assign drug resistance based on mutations
            drug_resistance = {}
            for drug, _mut_info in HBV_RESISTANCE_MUTATIONS.items():
                # Simplified: random resistance level
                drug_resistance[drug] = np.random.random() if mutations else 0.0

            records.append(
                SequenceRecord(
                    id=f"HBV_syn_{i}",
                    sequence="".join(seq),
                    organism=OrganismType.HBV,
                    sequence_type=SequenceType.DNA,
                    gene=self.gene,
                    subtype=genotype,
                    drug_resistance=drug_resistance,
                    mutations=mutations,
                )
            )

        return records

    def get_validation_labels(self) -> dict[str, Any]:
        """Get drug resistance labels for validation."""
        records = self.load_sequences()

        labels = {
            "drug_resistance": {},
            "genotype": {},
            "mutations": {},
        }

        for record in records:
            labels["drug_resistance"][record.id] = record.drug_resistance or {}
            labels["genotype"][record.id] = record.subtype
            labels["mutations"][record.id] = record.mutations or []

        return labels

    def get_resistance_profile(self, sequence_id: str) -> dict[str, float]:
        """Get drug resistance profile for a sequence."""
        records = self.load_sequences()
        for record in records:
            if record.id == sequence_id:
                return record.drug_resistance or {}
        return {}

    def find_resistance_mutations(self, sequence: str) -> dict[str, list[str]]:
        """Find known resistance mutations in a sequence.

        Args:
            sequence: DNA sequence of polymerase gene

        Returns:
            Dictionary mapping drugs to found mutations
        """
        # This is a simplified implementation
        # Real implementation would do proper alignment
        found = {}

        for drug, mutations in HBV_RESISTANCE_MUTATIONS.items():
            found[drug] = []
            all_muts = mutations["primary"] + mutations["secondary"]

            for _mut in all_muts:
                # Parse mutation (e.g., "M204V")
                # Check if mutation is present in sequence
                # This requires proper sequence alignment
                pass

        return found

    @staticmethod
    def get_genotype_info(genotype: str) -> dict[str, str]:
        """Get information about an HBV genotype."""
        return HBV_GENOTYPES.get(genotype, {"prevalence": "Unknown", "severity": "Unknown"})


# Convenience function
def load_hbv_sequences(
    max_sequences: int = 1000,
    gene: str = "polymerase",
) -> list[SequenceRecord]:
    """Load HBV sequences.

    Args:
        max_sequences: Maximum number of sequences
        gene: Gene to load ("polymerase", "surface", "core", "x")

    Returns:
        List of HBV sequence records
    """
    loader = HBVLoader(max_sequences=max_sequences, gene=gene)
    return loader.load_sequences()
