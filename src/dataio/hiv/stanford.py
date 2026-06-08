# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Stanford HIVDB Drug Resistance Data Loaders.

Loads drug resistance data from Stanford HIV Drug Resistance Database:
- Protease Inhibitors (PI): 2,171 records
- NRTIs: 1,867 records
- NNRTIs: 2,270 records
- Integrase Inhibitors (INI): 846 records
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.config.paths import PROJECT_ROOT


def _get_data_dir() -> Path:
    """Get the research datasets directory."""
    return PROJECT_ROOT / "data" / "research" / "datasets"


def load_stanford_hivdb(drug_class: str = "all") -> pd.DataFrame:
    """
    Load Stanford HIV Drug Resistance Database data.

    Args:
        drug_class: One of 'pi', 'nrti', 'nnrti', 'ini', or 'all'

    Returns:
        DataFrame with columns:
        - SeqID: Sequence identifier
        - Drug columns: Fold-change values (e.g., FPV, ATV, etc.)
        - Position columns: Amino acid at each position
        - CompMutList: Composite mutation list string
        - drug_class: Added column indicating source

    Raises:
        FileNotFoundError: If the data file doesn't exist
        ValueError: If invalid drug_class specified

    Example:
        >>> df = load_stanford_hivdb("pi")
        >>> print(f"Loaded {len(df)} protease inhibitor records")
    """
    valid_classes = {"pi", "nrti", "nnrti", "ini", "all"}
    drug_class = drug_class.lower()

    if drug_class not in valid_classes:
        raise ValueError(f"Invalid drug_class '{drug_class}'. Must be one of: {valid_classes}")

    file_mapping = {
        "pi": "stanford_hivdb_pi.txt",
        "nrti": "stanford_hivdb_nrti.txt",
        "nnrti": "stanford_hivdb_nnrti.txt",
        "ini": "stanford_hivdb_ini.txt",
    }

    data_dir = _get_data_dir()

    if drug_class == "all":
        dfs = []
        for cls, filename in file_mapping.items():
            filepath = data_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath, sep="\t", low_memory=False)
                df["drug_class"] = cls.upper()
                dfs.append(df)
        if not dfs:
            raise FileNotFoundError(f"No Stanford HIVDB files found in {data_dir}")
        return pd.concat(dfs, ignore_index=True)

    filepath = data_dir / file_mapping[drug_class]
    if not filepath.exists():
        raise FileNotFoundError(f"Stanford HIVDB file not found: {filepath}")

    df = pd.read_csv(filepath, sep="\t", low_memory=False)
    df["drug_class"] = drug_class.upper()
    return df


def get_stanford_drug_columns(drug_class: str) -> list[str]:
    """
    Get drug column names for a specific drug class.

    Args:
        drug_class: One of 'pi', 'nrti', 'nnrti', 'ini'

    Returns:
        List of drug column names

    Example:
        >>> get_stanford_drug_columns("pi")
        ['FPV', 'ATV', 'IDV', 'LPV', 'NFV', 'SQV', 'TPV', 'DRV']
    """
    drug_columns = {
        "pi": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"],
        "nrti": ["ABC", "AZT", "D4T", "DDI", "FTC", "3TC", "TDF"],
        "nnrti": ["DOR", "EFV", "ETR", "NVP", "RPV"],
        "ini": ["BIC", "CAB", "DTG", "EVG", "RAL"],
    }
    return drug_columns.get(drug_class.lower(), [])


def parse_mutation_list(mut_string: str) -> list[dict]:
    """
    Parse Stanford HIVDB CompMutList format.

    Args:
        mut_string: Comma-separated mutation string like "D30N, M46I, R57G"

    Returns:
        List of dicts with 'position', 'wild_type', 'mutant' keys

    Example:
        >>> parse_mutation_list("D30N, M46I")
        [{'wild_type': 'D', 'position': 30, 'mutant': 'N'},
         {'wild_type': 'M', 'position': 46, 'mutant': 'I'}]
    """
    if pd.isna(mut_string) or not mut_string:
        return []

    mutations = []
    pattern = r"([A-Z])(\d+)([A-Z*])"

    for mut in str(mut_string).split(","):
        mut = mut.strip()
        match = re.match(pattern, mut)
        if match:
            mutations.append(
                {
                    "wild_type": match.group(1),
                    "position": int(match.group(2)),
                    "mutant": match.group(3),
                }
            )
    return mutations


def extract_stanford_positions(df: pd.DataFrame, protein: str = "PR") -> pd.DataFrame:
    """
    Extract position columns from Stanford data as a clean matrix.

    Args:
        df: Stanford HIVDB DataFrame
        protein: 'PR' (protease), 'RT' (reverse transcriptase), or 'IN' (integrase)

    Returns:
        DataFrame with SeqID and position columns only

    Example:
        >>> df = load_stanford_hivdb("pi")
        >>> positions = extract_stanford_positions(df, "PR")
        >>> print(positions.columns[:5])  # ['SeqID', 'P1', 'P2', 'P3', 'P4']
    """
    prefix_map = {"PR": "P", "RT": "RT", "IN": "IN"}
    prefix = prefix_map.get(protein.upper(), "P")

    position_cols = [col for col in df.columns if col.startswith(prefix) and col[len(prefix) :].isdigit()]
    position_cols = sorted(position_cols, key=lambda x: int(x[len(prefix) :]))

    return df[["SeqID"] + position_cols].copy()


def get_resistance_mutations(df: pd.DataFrame, drug: str, threshold: float = 3.0) -> pd.DataFrame:
    """
    Get mutations associated with resistance to a specific drug.

    Args:
        df: Stanford HIVDB DataFrame
        drug: Drug name (e.g., "DRV", "TDF")
        threshold: Fold-change threshold for resistance

    Returns:
        DataFrame with resistant sequences and their mutations
    """
    if drug not in df.columns:
        raise ValueError(f"Drug '{drug}' not found in DataFrame. Available: {get_stanford_drug_columns('all')}")

    # Filter to resistant sequences
    resistant = df[df[drug] >= threshold].copy()

    # Parse mutations
    resistant["mutations"] = resistant["CompMutList"].apply(parse_mutation_list)

    return resistant[["SeqID", drug, "CompMutList", "mutations", "drug_class"]]
