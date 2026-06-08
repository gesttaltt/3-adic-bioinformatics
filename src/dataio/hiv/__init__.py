# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""HIV Data Loading Module.

Provides unified loading interfaces for all HIV-related datasets:

**Research Datasets (data/research/datasets/):**
- Stanford HIVDB drug resistance data (7,154 records across 4 drug classes)
- LANL CTL epitope database (2,116 epitopes)
- CATNAP antibody neutralization data (189,879 records)

**External Datasets (data/external/):**
- HuggingFace: HIV V3 coreceptor (2,935 sequences), Human-HIV PPI (16,179 interactions)
- Zenodo: cview gp120 alignments (712 sequences)
- GitHub: HIV-data sequences (~9,000 sequences)
- Kaggle: HIV-AIDS epidemiological statistics

Usage:
    from src.data.hiv import (
        load_stanford_hivdb,
        load_lanl_ctl,
        load_catnap,
        load_v3_coreceptor,
        load_hiv_ppi,
    )

    # Load all drug resistance data
    df = load_stanford_hivdb("all")

    # Load specific drug class
    pi_df = load_stanford_hivdb("pi")

    # Load CTL epitopes
    ctl_df = load_lanl_ctl()

    # Load neutralization data
    catnap_df = load_catnap()
"""

from .catnap import (
    calculate_antibody_breadth,
    get_catnap_by_antibody,
    get_catnap_resistant_viruses,
    get_catnap_sensitive_viruses,
    load_catnap,
)
from .ctl import (
    get_epitopes_by_hla,
    get_epitopes_by_protein,
    load_lanl_ctl,
    parse_hla_restrictions,
)
from .external import (
    load_epidemiological_data,
    load_gp120_alignments,
    load_hiv_ppi,
    load_hiv_sequences,
    load_v3_coreceptor,
)
from .position_mapper import (
    PositionMapper,
    codon_position_to_hxb2,
    hxb2_to_protein_position,
    protein_position_to_hxb2,
)
from .stanford import (
    extract_stanford_positions,
    get_stanford_drug_columns,
    load_stanford_hivdb,
    parse_mutation_list,
)

__all__ = [
    # Stanford HIVDB
    "load_stanford_hivdb",
    "get_stanford_drug_columns",
    "parse_mutation_list",
    "extract_stanford_positions",
    # LANL CTL
    "load_lanl_ctl",
    "parse_hla_restrictions",
    "get_epitopes_by_protein",
    "get_epitopes_by_hla",
    # CATNAP
    "load_catnap",
    "get_catnap_by_antibody",
    "get_catnap_sensitive_viruses",
    "get_catnap_resistant_viruses",
    "calculate_antibody_breadth",
    # External datasets
    "load_v3_coreceptor",
    "load_hiv_ppi",
    "load_gp120_alignments",
    "load_hiv_sequences",
    "load_epidemiological_data",
    # Position mapping
    "PositionMapper",
    "hxb2_to_protein_position",
    "protein_position_to_hxb2",
    "codon_position_to_hxb2",
]
