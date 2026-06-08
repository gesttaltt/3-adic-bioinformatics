# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""LANL CTL Epitope Database Loaders.

Loads CTL (Cytotoxic T-Lymphocyte) epitope data from Los Alamos National Laboratory.
Contains 2,116 characterized epitopes with HLA restrictions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.paths import PROJECT_ROOT


def _get_data_dir() -> Path:
    """Get the research datasets directory."""
    return PROJECT_ROOT / "data" / "research" / "datasets"


def load_lanl_ctl() -> pd.DataFrame:
    """
    Load LANL CTL epitope database.

    Returns:
        DataFrame with columns:
        - Epitope: Peptide sequence
        - Protein: HIV protein (Gag, Pol, Env, etc.)
        - HXB2_start: Start position in HXB2 reference
        - HXB2_end: End position in HXB2 reference
        - Subprotein: Subprotein region (e.g., p17, RT)
        - Subtype: HIV subtype(s)
        - Species: Host species
        - HLA: HLA restriction(s)

    Raises:
        FileNotFoundError: If the data file doesn't exist

    Example:
        >>> df = load_lanl_ctl()
        >>> print(f"Loaded {len(df)} epitopes")
        >>> print(df["Protein"].value_counts().head())
    """
    filepath = _get_data_dir() / "ctl_summary.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"CTL summary file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Standardize column names
    column_mapping = {
        "HXB2 start": "HXB2_start",
        "HXB2 end": "HXB2_end",
        "HXB2 DNA Contig": "HXB2_DNA_Contig",
    }
    df = df.rename(columns=column_mapping)

    # Parse numeric positions
    df["HXB2_start"] = pd.to_numeric(df["HXB2_start"], errors="coerce")
    df["HXB2_end"] = pd.to_numeric(df["HXB2_end"], errors="coerce")

    return df


def parse_hla_restrictions(hla_string: str) -> list[str]:
    """
    Parse HLA restriction string into individual HLA types.

    Args:
        hla_string: Comma-separated HLA string like "A*02:01, B*57, B27"

    Returns:
        List of standardized HLA alleles

    Example:
        >>> parse_hla_restrictions("A*02:01, B*57")
        ['A*02:01', 'B*57']
    """
    if pd.isna(hla_string) or not hla_string:
        return []

    hla_list = []
    for hla in str(hla_string).split(","):
        hla = hla.strip()
        if hla:
            hla_list.append(hla)
    return hla_list


def get_epitopes_by_protein(df: pd.DataFrame | None = None, protein: str = "Gag") -> pd.DataFrame:
    """
    Get epitopes for a specific HIV protein.

    Args:
        df: CTL DataFrame (if None, loads from file)
        protein: Protein name (Gag, Pol, Env, Nef, Tat, Rev, Vif, Vpr, Vpu)

    Returns:
        Filtered DataFrame

    Example:
        >>> gag_epitopes = get_epitopes_by_protein(protein="Gag")
        >>> print(f"Found {len(gag_epitopes)} Gag epitopes")
    """
    if df is None:
        df = load_lanl_ctl()
    return df[df["Protein"].str.contains(protein, case=False, na=False)].copy()


def get_epitopes_by_hla(df: pd.DataFrame | None = None, hla: str = "A*02:01") -> pd.DataFrame:
    """
    Get epitopes restricted by a specific HLA type.

    Args:
        df: CTL DataFrame (if None, loads from file)
        hla: HLA allele (e.g., "A*02:01", "B*57", "B27")

    Returns:
        Filtered DataFrame

    Example:
        >>> a2_epitopes = get_epitopes_by_hla(hla="A*02:01")
        >>> print(f"Found {len(a2_epitopes)} A*02:01-restricted epitopes")
    """
    if df is None:
        df = load_lanl_ctl()
    return df[df["HLA"].str.contains(hla, case=False, na=False)].copy()


def get_epitope_coverage_by_hla(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Calculate epitope coverage for each HLA type.

    Args:
        df: CTL DataFrame (if None, loads from file)

    Returns:
        DataFrame with HLA type, count, and percentage

    Example:
        >>> coverage = get_epitope_coverage_by_hla()
        >>> print(coverage.head(10))
    """
    if df is None:
        df = load_lanl_ctl()

    # Count epitopes per HLA
    hla_counts = {}
    for hla_string in df["HLA"].dropna():
        for hla in parse_hla_restrictions(hla_string):
            hla_counts[hla] = hla_counts.get(hla, 0) + 1

    result = pd.DataFrame(
        [{"HLA": hla, "epitope_count": count} for hla, count in sorted(hla_counts.items(), key=lambda x: -x[1])]
    )

    result["percentage"] = 100 * result["epitope_count"] / len(df)
    return result


def get_conserved_epitopes(
    df: pd.DataFrame | None = None,
    min_subtypes: int = 3,
) -> pd.DataFrame:
    """
    Get epitopes conserved across multiple HIV subtypes.

    Args:
        df: CTL DataFrame (if None, loads from file)
        min_subtypes: Minimum number of subtypes for conservation

    Returns:
        DataFrame with conserved epitopes
    """
    if df is None:
        df = load_lanl_ctl()

    def count_subtypes(subtype_str):
        if pd.isna(subtype_str) or not subtype_str:
            return 0
        return len(str(subtype_str).split(","))

    df = df.copy()
    df["subtype_count"] = df["Subtype"].apply(count_subtypes)
    return df[df["subtype_count"] >= min_subtypes].copy()
