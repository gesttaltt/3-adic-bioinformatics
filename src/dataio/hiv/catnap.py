# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""CATNAP Antibody Neutralization Data Loaders.

Loads CATNAP (Compile, Analyze and Tally NAb Panels) neutralization assay data.
Contains 189,879 antibody-virus neutralization records.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.paths import PROJECT_ROOT


def _get_data_dir() -> Path:
    """Get the research datasets directory."""
    return PROJECT_ROOT / "data" / "research" / "datasets"


def load_catnap() -> pd.DataFrame:
    """
    Load CATNAP antibody neutralization data.

    Returns:
        DataFrame with columns:
        - Antibody: Antibody name (e.g., VRC01, PG9)
        - Virus: Virus strain identifier
        - Reference: Publication reference
        - Pubmed_ID: PubMed identifier
        - IC50: 50% inhibitory concentration (ug/mL)
        - IC80: 80% inhibitory concentration
        - ID50: 50% neutralization dilution
        - IC50_numeric: Parsed numeric IC50 value
        - IC50_censored: Whether the value was censored (> or <)

    Raises:
        FileNotFoundError: If the data file doesn't exist

    Example:
        >>> df = load_catnap()
        >>> print(f"Loaded {len(df):,} neutralization records")
        >>> print(f"Unique antibodies: {df['Antibody'].nunique()}")
    """
    filepath = _get_data_dir() / "catnap_assay.txt"
    if not filepath.exists():
        raise FileNotFoundError(f"CATNAP file not found: {filepath}")

    df = pd.read_csv(filepath, sep="\t", low_memory=False)

    # Standardize column names
    column_mapping = {
        "Pubmed ID": "Pubmed_ID",
    }
    df = df.rename(columns=column_mapping)

    # Parse IC50 values (handle ">", "<" prefixes)
    df["IC50_numeric"] = df["IC50"].apply(_parse_ic_value)
    df["IC50_censored"] = df["IC50"].apply(lambda x: ">" in str(x) or "<" in str(x))

    return df


def _parse_ic_value(value) -> float | None:
    """Parse IC50/IC80 values, handling censored values."""
    if pd.isna(value) or value == "":
        return None
    val_str = str(value).strip()
    # Remove > or < prefix
    val_str = val_str.lstrip(">< ")
    try:
        return float(val_str)
    except ValueError:
        return None


def get_catnap_by_antibody(df: pd.DataFrame | None = None, antibody: str = "VRC01") -> pd.DataFrame:
    """
    Get neutralization data for a specific antibody.

    Args:
        df: CATNAP DataFrame (if None, loads from file)
        antibody: Antibody name (e.g., "VRC01", "PG9", "10E8")

    Returns:
        Filtered DataFrame

    Example:
        >>> vrc01_data = get_catnap_by_antibody(antibody="VRC01")
        >>> print(f"VRC01 tested against {len(vrc01_data)} viruses")
    """
    if df is None:
        df = load_catnap()
    return df[df["Antibody"].str.contains(antibody, case=False, na=False)].copy()


def get_catnap_sensitive_viruses(
    df: pd.DataFrame | None = None,
    antibody: str = "VRC01",
    ic50_threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Get viruses sensitive to an antibody (IC50 below threshold).

    Args:
        df: CATNAP DataFrame (if None, loads from file)
        antibody: Antibody name
        ic50_threshold: Maximum IC50 for sensitivity (ug/mL)

    Returns:
        DataFrame with sensitive viruses
    """
    if df is None:
        df = load_catnap()
    ab_data = df[df["Antibody"].str.contains(antibody, case=False, na=False)].copy()
    return ab_data[(ab_data["IC50_numeric"].notna()) & (ab_data["IC50_numeric"] <= ic50_threshold)]


def get_catnap_resistant_viruses(
    df: pd.DataFrame | None = None,
    antibody: str = "VRC01",
    ic50_threshold: float = 50.0,
) -> pd.DataFrame:
    """
    Get viruses resistant to an antibody (IC50 above threshold).

    Args:
        df: CATNAP DataFrame (if None, loads from file)
        antibody: Antibody name
        ic50_threshold: Minimum IC50 for resistance (ug/mL)

    Returns:
        DataFrame with resistant viruses
    """
    if df is None:
        df = load_catnap()
    ab_data = df[df["Antibody"].str.contains(antibody, case=False, na=False)].copy()
    return ab_data[(ab_data["IC50_numeric"].notna()) & (ab_data["IC50_numeric"] >= ic50_threshold)]


def calculate_antibody_breadth(
    df: pd.DataFrame | None = None,
    ic50_threshold: float = 50.0,
) -> pd.DataFrame:
    """
    Calculate neutralization breadth for each antibody.

    Breadth is defined as the percentage of viruses neutralized at IC50 below threshold.

    Args:
        df: CATNAP DataFrame (if None, loads from file)
        ic50_threshold: IC50 threshold for considering a virus neutralized

    Returns:
        DataFrame with columns:
        - Antibody: Antibody name
        - n_tested: Number of viruses tested
        - n_neutralized: Number of viruses neutralized
        - breadth_pct: Percentage neutralized

    Example:
        >>> breadth = calculate_antibody_breadth()
        >>> top_10 = breadth.head(10)
        >>> print(top_10)
    """
    if df is None:
        df = load_catnap()

    # Filter to records with valid IC50
    valid_df = df[df["IC50_numeric"].notna()].copy()

    results = []
    for antibody in valid_df["Antibody"].unique():
        ab_data = valid_df[valid_df["Antibody"] == antibody]
        n_tested = len(ab_data)
        n_neutralized = len(ab_data[ab_data["IC50_numeric"] <= ic50_threshold])
        breadth_pct = 100 * n_neutralized / n_tested if n_tested > 0 else 0

        results.append(
            {
                "Antibody": antibody,
                "n_tested": n_tested,
                "n_neutralized": n_neutralized,
                "breadth_pct": breadth_pct,
            }
        )

    result_df = pd.DataFrame(results)
    return result_df.sort_values("breadth_pct", ascending=False).reset_index(drop=True)


def get_bnab_classes() -> dict[str, list[str]]:
    """
    Get broadly neutralizing antibody classifications by epitope target.

    Returns:
        Dictionary mapping epitope class to antibody names
    """
    return {
        "CD4bs": ["VRC01", "VRC03", "VRC-PG04", "b12", "NIH45-46", "3BNC117"],
        "V1V2_glycan": ["PG9", "PG16", "PGT145", "VRC26", "PGDM1400"],
        "V3_glycan": ["PGT121", "PGT128", "PGT135", "10-1074"],
        "gp120_gp41": ["35O22", "8ANC195"],
        "MPER": ["10E8", "4E10", "2F5"],
        "fusion_peptide": ["VRC34", "ACS202"],
    }


def classify_antibody(antibody: str) -> str | None:
    """
    Classify an antibody by its epitope target.

    Args:
        antibody: Antibody name

    Returns:
        Epitope class name or None if unknown
    """
    for epitope_class, antibodies in get_bnab_classes().items():
        for ab in antibodies:
            if ab.lower() in antibody.lower():
                return epitope_class
    return None
