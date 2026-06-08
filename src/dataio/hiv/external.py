# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""External HIV Dataset Loaders.

Loads datasets from external sources:
- HuggingFace: HIV V3 coreceptor (2,935 sequences), Human-HIV PPI (16,179 interactions)
- Zenodo: cview gp120 alignments (712 sequences)
- GitHub: HIV-data sequences (~9,000 sequences)
- Kaggle: HIV-AIDS epidemiological statistics
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from src.config.paths import PROJECT_ROOT


def _get_external_dir() -> Path:
    """Get the external data directory."""
    return PROJECT_ROOT / "data" / "external"


def load_v3_coreceptor() -> pd.DataFrame:
    """
    Load HIV V3 coreceptor tropism dataset from HuggingFace.

    Contains 2,935 V3 loop sequences with CCR5/CXCR4 coreceptor usage labels.

    Returns:
        DataFrame with columns:
        - sequence: V3 loop amino acid sequence
        - tropism: Coreceptor usage (CCR5, CXCR4, or dual)

    Raises:
        FileNotFoundError: If dataset not downloaded
        ImportError: If required dependencies missing

    Example:
        >>> df = load_v3_coreceptor()
        >>> print(df["tropism"].value_counts())
    """
    data_dir = _get_external_dir() / "huggingface" / "HIV_V3_coreceptor"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"V3 coreceptor dataset not found: {data_dir}\n"
            "Run: ternary-vae data download huggingface --dataset HIV_V3_coreceptor"
        )

    # Try loading from disk (saved HuggingFace dataset)
    try:
        from datasets import load_from_disk

        ds = load_from_disk(str(data_dir))
        if hasattr(ds, "to_pandas"):
            return ds.to_pandas()
        # Handle DatasetDict
        if "train" in ds:
            return ds["train"].to_pandas()
        return pd.DataFrame(ds)
    except ImportError:
        warnings.warn("HuggingFace datasets not installed. Trying parquet fallback.", stacklevel=2)

    # Fallback to parquet files
    parquet_files = list(data_dir.glob("*.parquet"))
    if parquet_files:
        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True)

    raise FileNotFoundError(f"No valid data files found in {data_dir}")


def load_hiv_ppi() -> pd.DataFrame:
    """
    Load Human-HIV protein-protein interaction dataset from HuggingFace.

    Contains 16,179 interactions between HIV and human proteins.

    Returns:
        DataFrame with columns:
        - hiv_protein: HIV protein name
        - human_protein: Human protein name/ID
        - interaction_type: Type of interaction

    Raises:
        FileNotFoundError: If dataset not downloaded

    Example:
        >>> df = load_hiv_ppi()
        >>> print(df["hiv_protein"].value_counts())
    """
    data_dir = _get_external_dir() / "huggingface" / "human_hiv_ppi"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Human-HIV PPI dataset not found: {data_dir}\n"
            "Run: ternary-vae data download huggingface --dataset human_hiv_ppi"
        )

    try:
        from datasets import load_from_disk

        ds = load_from_disk(str(data_dir))
        if hasattr(ds, "to_pandas"):
            return ds.to_pandas()
        if "train" in ds:
            return ds["train"].to_pandas()
        return pd.DataFrame(ds)
    except ImportError:
        pass

    # Fallback to parquet
    parquet_files = list(data_dir.glob("*.parquet"))
    if parquet_files:
        dfs = [pd.read_parquet(f) for f in parquet_files]
        return pd.concat(dfs, ignore_index=True)

    raise FileNotFoundError(f"No valid data files found in {data_dir}")


def load_gp120_alignments() -> dict[str, str]:
    """
    Load gp120 sequence alignments from Zenodo cview dataset.

    Contains 712 aligned gp120 sequences with tropism labels.

    Returns:
        Dictionary mapping sequence ID to aligned sequence

    Raises:
        FileNotFoundError: If dataset not downloaded

    Example:
        >>> alignments = load_gp120_alignments()
        >>> print(f"Loaded {len(alignments)} gp120 sequences")
    """
    data_dir = _get_external_dir() / "zenodo" / "cview_gp120"

    if not data_dir.exists():
        raise FileNotFoundError(
            f"gp120 alignment dataset not found: {data_dir}\n"
            "Download from Zenodo and extract to data/external/zenodo/cview_gp120/"
        )

    alignments = {}

    # Try FASTA files
    fasta_files = list(data_dir.glob("*.fasta")) + list(data_dir.glob("*.fa"))
    for fasta_file in fasta_files:
        alignments.update(_parse_fasta(fasta_file))

    if not alignments:
        raise FileNotFoundError(f"No FASTA files found in {data_dir}")

    return alignments


def _parse_fasta(filepath: Path) -> dict[str, str]:
    """Parse a FASTA file into a dictionary."""
    sequences = {}
    current_id = None
    current_seq = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_id:
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id:
            sequences[current_id] = "".join(current_seq)

    return sequences


def load_hiv_sequences(source: str = "github") -> dict[str, str]:
    """
    Load HIV sequences from GitHub repositories.

    Args:
        source: Data source ("github" for HIV-data repo)

    Returns:
        Dictionary mapping sequence ID to sequence

    Raises:
        FileNotFoundError: If dataset not downloaded

    Example:
        >>> seqs = load_hiv_sequences()
        >>> print(f"Loaded {len(seqs)} HIV sequences")
    """
    if source == "github":
        data_dir = _get_external_dir() / "github" / "HIV-data"
    else:
        raise ValueError(f"Unknown source: {source}")

    if not data_dir.exists():
        raise FileNotFoundError(
            f"HIV sequence data not found: {data_dir}\nClone the repository to data/external/github/HIV-data/"
        )

    sequences = {}

    # Find all FASTA files (including gzipped)
    for pattern in ["*.fasta", "*.fa", "*.fasta.gz", "*.fa.gz"]:
        for fasta_file in data_dir.rglob(pattern):
            if fasta_file.suffix == ".gz":
                import gzip

                with gzip.open(fasta_file, "rt") as f:
                    content = f.read()
                    sequences.update(_parse_fasta_content(content))
            else:
                sequences.update(_parse_fasta(fasta_file))

    return sequences


def _parse_fasta_content(content: str) -> dict[str, str]:
    """Parse FASTA content from a string."""
    sequences = {}
    current_id = None
    current_seq = []

    for line in content.split("\n"):
        line = line.strip()
        if line.startswith(">"):
            if current_id:
                sequences[current_id] = "".join(current_seq)
            current_id = line[1:].split()[0]
            current_seq = []
        elif line:
            current_seq.append(line)

    if current_id:
        sequences[current_id] = "".join(current_seq)

    return sequences


def load_epidemiological_data() -> pd.DataFrame:
    """
    Load HIV/AIDS epidemiological statistics from Kaggle.

    Returns:
        DataFrame with country-level HIV statistics

    Raises:
        FileNotFoundError: If dataset not downloaded

    Example:
        >>> df = load_epidemiological_data()
        >>> print(df.columns)
    """
    data_dir = _get_external_dir() / "kaggle" / "hiv-aids-dataset"

    if not data_dir.exists():
        # Try alternative location
        data_dir = _get_external_dir() / "csv"

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"Epidemiological data not found in {data_dir}\nDownload from Kaggle: hiv-aids-dataset")

    # Load all CSV files
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df["source_file"] = csv_file.name
            dfs.append(df)
        except Exception as e:
            warnings.warn(f"Failed to load {csv_file}: {e}", stacklevel=2)

    if not dfs:
        raise FileNotFoundError("No valid CSV files found")

    return pd.concat(dfs, ignore_index=True)


def list_available_datasets() -> dict[str, dict]:
    """
    List all available external datasets and their status.

    Returns:
        Dictionary with dataset info and availability status

    Example:
        >>> available = list_available_datasets()
        >>> for name, info in available.items():
        ...     print(f"{name}: {'Available' if info['exists'] else 'Missing'}")
    """
    external_dir = _get_external_dir()

    datasets = {
        "HIV_V3_coreceptor": {
            "path": external_dir / "huggingface" / "HIV_V3_coreceptor",
            "source": "HuggingFace",
            "records": "~2,935 sequences",
            "description": "V3 loop sequences with CCR5/CXCR4 tropism labels",
        },
        "human_hiv_ppi": {
            "path": external_dir / "huggingface" / "human_hiv_ppi",
            "source": "HuggingFace",
            "records": "~16,179 interactions",
            "description": "Human-HIV protein-protein interactions",
        },
        "cview_gp120": {
            "path": external_dir / "zenodo" / "cview_gp120",
            "source": "Zenodo",
            "records": "~712 sequences",
            "description": "Aligned gp120 sequences with tropism labels",
        },
        "HIV-data": {
            "path": external_dir / "github" / "HIV-data",
            "source": "GitHub",
            "records": "~9,000 sequences",
            "description": "HIV-1 envelope and genome sequences",
        },
        "hiv-aids-dataset": {
            "path": external_dir / "kaggle" / "hiv-aids-dataset",
            "source": "Kaggle",
            "records": "~170 countries",
            "description": "Epidemiological statistics",
        },
    }

    for _name, info in datasets.items():
        info["exists"] = info["path"].exists()

    return datasets
