# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""HIV Position Mapping Utilities.

Maps between different HIV sequence position numbering systems:
- HXB2 reference coordinates (K03455)
- Protein-specific positions (e.g., PR 1-99, RT 1-440)
- Codon positions for p-adic analysis

Reference: HXB2 (K03455) nucleotide positions:
- Gag: 790-2292
- Pol: 2085-5096
  - Protease (PR): 2253-2549 (aa 1-99)
  - Reverse Transcriptase (RT): 2550-3869 (aa 1-440)
  - RNase H: 3870-4229 (aa 441-560)
  - Integrase (IN): 4230-5096 (aa 1-288)
- Env: 6225-8795
  - gp120: 6225-7758
    - V3 loop: 7110-7217 (aa 296-331)
  - gp41: 7759-8795
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HXB2Region:
    """HXB2 reference region definition."""

    name: str
    nt_start: int
    nt_end: int
    aa_start: int
    aa_end: int


# HXB2 reference regions (nucleotide and amino acid positions)
HXB2_REGIONS = {
    "gag": HXB2Region("Gag", 790, 2292, 1, 500),
    "pol": HXB2Region("Pol", 2085, 5096, 1, 1003),
    "pr": HXB2Region("Protease", 2253, 2549, 1, 99),
    "rt": HXB2Region("Reverse Transcriptase", 2550, 3869, 1, 440),
    "rnaseh": HXB2Region("RNase H", 3870, 4229, 441, 560),
    "in": HXB2Region("Integrase", 4230, 5096, 1, 288),
    "env": HXB2Region("Env", 6225, 8795, 1, 856),
    "gp120": HXB2Region("gp120", 6225, 7758, 1, 511),
    "v3": HXB2Region("V3 loop", 7110, 7217, 296, 331),
    "gp41": HXB2Region("gp41", 7759, 8795, 512, 856),
    "nef": HXB2Region("Nef", 8797, 9417, 1, 206),
    "tat": HXB2Region("Tat", 5831, 8469, 1, 101),  # Spliced
    "rev": HXB2Region("Rev", 5970, 8653, 1, 116),  # Spliced
}


class PositionMapper:
    """
    Map between different HIV sequence position numbering systems.

    Example:
        >>> mapper = PositionMapper()
        >>> hxb2_pos = mapper.protein_to_hxb2("PR", 30)
        >>> print(f"PR position 30 = HXB2 nucleotide {hxb2_pos}")

        >>> protein, pos = mapper.hxb2_to_protein(2340)
        >>> print(f"HXB2 2340 = {protein} position {pos}")
    """

    def __init__(self):
        self.regions = HXB2_REGIONS

    def protein_to_hxb2(self, protein: str, aa_position: int) -> int:
        """
        Convert protein amino acid position to HXB2 nucleotide position.

        Args:
            protein: Protein name (PR, RT, IN, gp120, etc.)
            aa_position: Amino acid position within protein

        Returns:
            HXB2 nucleotide position (start of codon)

        Raises:
            ValueError: If protein unknown or position out of range
        """
        protein = protein.lower()
        if protein not in self.regions:
            raise ValueError(f"Unknown protein: {protein}. Valid: {list(self.regions.keys())}")

        region = self.regions[protein]
        if aa_position < region.aa_start or aa_position > region.aa_end:
            raise ValueError(
                f"Position {aa_position} out of range for {region.name} (valid: {region.aa_start}-{region.aa_end})"
            )

        # Calculate nucleotide position
        aa_offset = aa_position - region.aa_start
        nt_position = region.nt_start + (aa_offset * 3)

        return nt_position

    def hxb2_to_protein(self, nt_position: int) -> tuple[str, int]:
        """
        Convert HXB2 nucleotide position to protein and amino acid position.

        Args:
            nt_position: HXB2 nucleotide position

        Returns:
            Tuple of (protein_name, aa_position)

        Raises:
            ValueError: If position not in any coding region
        """
        for name, region in self.regions.items():
            if region.nt_start <= nt_position <= region.nt_end:
                nt_offset = nt_position - region.nt_start
                aa_position = region.aa_start + (nt_offset // 3)
                return name.upper(), aa_position

        raise ValueError(f"HXB2 position {nt_position} not in any coding region")

    def codon_to_hxb2(self, protein: str, codon_index: int) -> int:
        """
        Convert codon index to HXB2 nucleotide position.

        Codon index is 0-based within the protein coding sequence.

        Args:
            protein: Protein name
            codon_index: 0-based codon index

        Returns:
            HXB2 nucleotide position (start of codon)
        """
        protein = protein.lower()
        if protein not in self.regions:
            raise ValueError(f"Unknown protein: {protein}")

        region = self.regions[protein]
        return region.nt_start + (codon_index * 3)

    def get_region_info(self, protein: str) -> HXB2Region:
        """Get region information for a protein."""
        protein = protein.lower()
        if protein not in self.regions:
            raise ValueError(f"Unknown protein: {protein}")
        return self.regions[protein]

    def list_regions(self) -> list[str]:
        """List all available region names."""
        return list(self.regions.keys())


# Module-level convenience functions
_mapper = PositionMapper()


def hxb2_to_protein_position(nt_position: int) -> tuple[str, int]:
    """
    Convert HXB2 nucleotide position to protein and amino acid position.

    Args:
        nt_position: HXB2 nucleotide position

    Returns:
        Tuple of (protein_name, aa_position)

    Example:
        >>> protein, pos = hxb2_to_protein_position(2340)
        >>> print(f"{protein} position {pos}")
    """
    return _mapper.hxb2_to_protein(nt_position)


def protein_position_to_hxb2(protein: str, aa_position: int) -> int:
    """
    Convert protein amino acid position to HXB2 nucleotide position.

    Args:
        protein: Protein name (PR, RT, IN, gp120, etc.)
        aa_position: Amino acid position within protein

    Returns:
        HXB2 nucleotide position

    Example:
        >>> hxb2 = protein_position_to_hxb2("PR", 30)
        >>> print(f"HXB2 position: {hxb2}")
    """
    return _mapper.protein_to_hxb2(protein, aa_position)


def codon_position_to_hxb2(protein: str, codon_index: int) -> int:
    """
    Convert codon index to HXB2 nucleotide position.

    Args:
        protein: Protein name
        codon_index: 0-based codon index within protein

    Returns:
        HXB2 nucleotide position

    Example:
        >>> hxb2 = codon_position_to_hxb2("RT", 0)  # First codon of RT
        >>> print(f"RT codon 0 starts at HXB2 {hxb2}")
    """
    return _mapper.codon_to_hxb2(protein, codon_index)


def get_v3_positions() -> tuple[int, int]:
    """Get HXB2 nucleotide positions for V3 loop."""
    region = _mapper.get_region_info("v3")
    return region.nt_start, region.nt_end


def get_drug_target_positions(drug_class: str) -> tuple[str, int, int]:
    """
    Get protein and position range for a drug class.

    Args:
        drug_class: Drug class (PI, NRTI, NNRTI, INI)

    Returns:
        Tuple of (protein, start_aa, end_aa)
    """
    mapping = {
        "pi": ("pr", 1, 99),
        "nrti": ("rt", 1, 240),  # NRTI binding region
        "nnrti": ("rt", 100, 250),  # NNRTI binding pocket
        "ini": ("in", 1, 288),
    }

    drug_class = drug_class.lower()
    if drug_class not in mapping:
        raise ValueError(f"Unknown drug class: {drug_class}. Valid: PI, NRTI, NNRTI, INI")

    return mapping[drug_class]
