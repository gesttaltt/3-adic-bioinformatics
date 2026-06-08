# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""PDB Rotamer Ingestion Script.

This script downloads protein structures from RCSB PDB and extracts
side-chain dihedral angles (chi angles) for rotamer analysis.

Key Features:
1. Downloads PDB/CIF files from RCSB API
2. Parses atomic coordinates using Biopython
3. Computes chi1-chi4 dihedral angles for all residues
4. Outputs a tensor for use in hyperbolic embedding analysis

Usage:
    python src/scripts/ingest/ingest_pdb_rotamers.py \
        --pdb_ids "1CRN,1TIM,4LZT" \
        --output data/processed/rotamers.pt
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from Bio.PDB import MMCIFParser, PDBParser
    from Bio.PDB.Polypeptide import is_aa
    from Bio.PDB.vectors import calc_dihedral

    HAS_BIOPYTHON = True
except ImportError:
    HAS_BIOPYTHON = False


# Chi angle atom definitions for each amino acid
CHI_ATOMS = {
    "ARG": [
        ("N", "CA", "CB", "CG"),  # chi1
        ("CA", "CB", "CG", "CD"),  # chi2
        ("CB", "CG", "CD", "NE"),  # chi3
        ("CG", "CD", "NE", "CZ"),  # chi4
    ],
    "ASN": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "OD1"),
    ],
    "ASP": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "OD1"),
    ],
    "CYS": [
        ("N", "CA", "CB", "SG"),
    ],
    "GLN": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ],
    "GLU": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "OE1"),
    ],
    "HIS": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "ND1"),
    ],
    "ILE": [
        ("N", "CA", "CB", "CG1"),
        ("CA", "CB", "CG1", "CD1"),
    ],
    "LEU": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ],
    "LYS": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
        ("CB", "CG", "CD", "CE"),
        ("CG", "CD", "CE", "NZ"),
    ],
    "MET": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "SD"),
        ("CB", "CG", "SD", "CE"),
    ],
    "PHE": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ],
    "PRO": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD"),
    ],
    "SER": [
        ("N", "CA", "CB", "OG"),
    ],
    "THR": [
        ("N", "CA", "CB", "OG1"),
    ],
    "TRP": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ],
    "TYR": [
        ("N", "CA", "CB", "CG"),
        ("CA", "CB", "CG", "CD1"),
    ],
    "VAL": [
        ("N", "CA", "CB", "CG1"),
    ],
}


@dataclass
class RotamerData:
    """Stores rotamer data for a single residue."""

    pdb_id: str
    chain_id: str
    residue_id: int
    residue_name: str
    chi_angles: list[float]  # chi1, chi2, chi3, chi4 (nan if not applicable)
    sequence_context: str  # 5-residue context window


def download_pdb(pdb_id: str, output_dir: Path, format: str = "pdb") -> Path | None:
    """Download PDB file from RCSB.

    Args:
        pdb_id: 4-letter PDB identifier
        output_dir: Directory to save files
        format: "pdb" or "cif"

    Returns:
        Path to downloaded file or None if failed
    """
    pdb_id = pdb_id.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    if format == "cif":
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        output_file = output_dir / f"{pdb_id}.cif"
    else:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        output_file = output_dir / f"{pdb_id}.pdb"

    if output_file.exists():
        return output_file

    try:
        print(f"Downloading {pdb_id}...")
        urllib.request.urlretrieve(url, output_file)
        return output_file
    except Exception as e:
        print(f"Error downloading {pdb_id}: {e}")
        return None


def compute_dihedral(
    residue,
    atom_names: tuple[str, str, str, str],
) -> float | None:
    """Compute dihedral angle from four atoms.

    Args:
        residue: Biopython Residue object
        atom_names: Tuple of 4 atom names

    Returns:
        Dihedral angle in radians or None if atoms missing
    """
    try:
        atoms = [residue[name] for name in atom_names]
        vectors = [a.get_vector() for a in atoms]
        angle = calc_dihedral(*vectors)
        return float(angle)
    except KeyError:
        return None


def extract_chi_angles(residue) -> list[float]:
    """Extract chi angles for a residue.

    Args:
        residue: Biopython Residue object

    Returns:
        List of chi angles [chi1, chi2, chi3, chi4] with np.nan for missing
    """
    resname = residue.get_resname()
    chi_angles = [np.nan, np.nan, np.nan, np.nan]

    if resname not in CHI_ATOMS:
        return chi_angles

    atom_defs = CHI_ATOMS[resname]
    for i, atom_names in enumerate(atom_defs):
        angle = compute_dihedral(residue, atom_names)
        if angle is not None:
            chi_angles[i] = angle

    return chi_angles


def parse_structure(
    pdb_file: Path,
    pdb_id: str,
) -> list[RotamerData]:
    """Parse PDB/CIF file and extract rotamer data.

    Args:
        pdb_file: Path to structure file
        pdb_id: PDB identifier

    Returns:
        List of RotamerData objects
    """
    if not HAS_BIOPYTHON:
        raise ImportError("Biopython required for structure parsing")

    # Select parser based on file extension
    parser = MMCIFParser(QUIET=True) if pdb_file.suffix == ".cif" else PDBParser(QUIET=True)

    structure = parser.get_structure(pdb_id, pdb_file)
    rotamer_data = []

    for model in structure:
        for chain in model:
            # Build sequence for context
            residues = [r for r in chain if is_aa(r)]
            sequence = "".join([r.get_resname()[0] if len(r.get_resname()) == 3 else "X" for r in residues])

            for i, residue in enumerate(residues):
                # Skip non-standard amino acids
                if not is_aa(residue):
                    continue

                # Get sequence context (2 before, 2 after)
                start = max(0, i - 2)
                end = min(len(sequence), i + 3)
                context = sequence[start:end]

                # Extract chi angles
                chi_angles = extract_chi_angles(residue)

                rotamer = RotamerData(
                    pdb_id=pdb_id.upper(),
                    chain_id=chain.id,
                    residue_id=residue.id[1],
                    residue_name=residue.get_resname(),
                    chi_angles=chi_angles,
                    sequence_context=context,
                )
                rotamer_data.append(rotamer)

    return rotamer_data


def rotamer_data_to_tensor(
    data: list[RotamerData],
) -> tuple[np.ndarray, list[dict]]:
    """Convert rotamer data to tensor format.

    Args:
        data: List of RotamerData objects

    Returns:
        Tuple of (chi_angles tensor, metadata list)
    """
    chi_tensor = np.array([d.chi_angles for d in data], dtype=np.float32)

    metadata = [
        {
            "pdb_id": d.pdb_id,
            "chain_id": d.chain_id,
            "residue_id": d.residue_id,
            "residue_name": d.residue_name,
            "sequence_context": d.sequence_context,
        }
        for d in data
    ]

    return chi_tensor, metadata


def ingest_pdb_rotamers(
    pdb_ids: list[str],
    output_path: Path,
    cache_dir: Path = RAW_DATA_DIR / "pdb_cache",
) -> None:
    """Main ingestion function.

    Args:
        pdb_ids: List of PDB identifiers
        output_path: Path to save output tensor
        cache_dir: Directory for downloaded PDB files
    """
    if not HAS_TORCH:
        print("PyTorch required for saving tensors")
        return

    all_rotamer_data = []

    for pdb_id in pdb_ids:
        pdb_id = pdb_id.strip().upper()
        if len(pdb_id) != 4:
            print(f"Invalid PDB ID: {pdb_id}")
            continue

        # Try PDB format first, then CIF
        pdb_file = download_pdb(pdb_id, cache_dir, format="pdb")
        if pdb_file is None:
            pdb_file = download_pdb(pdb_id, cache_dir, format="cif")

        if pdb_file is None:
            continue

        try:
            rotamer_data = parse_structure(pdb_file, pdb_id)
            all_rotamer_data.extend(rotamer_data)
            print(f"  {pdb_id}: {len(rotamer_data)} residues processed")
        except Exception as e:
            print(f"Error parsing {pdb_id}: {e}")

    if not all_rotamer_data:
        print("No rotamer data extracted")
        return

    # Convert to tensor
    chi_tensor, metadata = rotamer_data_to_tensor(all_rotamer_data)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "chi_angles": torch.tensor(chi_tensor),
        "metadata": metadata,
        "pdb_ids": list(set(pdb_ids)),
        "n_residues": len(all_rotamer_data),
    }

    torch.save(save_dict, output_path)
    print(f"\nSaved {len(all_rotamer_data)} residues to {output_path}")
    print(f"Chi angles tensor shape: {chi_tensor.shape}")


def create_demo_rotamer_data(output_path: Path) -> None:
    """Create demo rotamer data for testing."""
    if not HAS_TORCH:
        print("PyTorch required for saving demo data")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate synthetic rotamer data
    np.random.seed(42)
    n_residues = 500

    # Common rotamer angles (degrees) - biased towards common conformations
    chi_angles = np.zeros((n_residues, 4))
    residue_names = []
    positions = []

    rotameric_residues = [
        "LEU",
        "ILE",
        "VAL",
        "PHE",
        "TYR",
        "TRP",
        "HIS",
        "ASN",
        "ASP",
        "GLU",
        "GLN",
        "LYS",
        "ARG",
        "MET",
        "SER",
        "THR",
        "CYS",
    ]

    for i in range(n_residues):
        res_name = np.random.choice(rotameric_residues)
        residue_names.append(res_name)
        positions.append(i + 1)

        # Chi1 - usually around -60 (gauche-), 180 (trans), or 60 (gauche+)
        chi1_center = np.random.choice([-60, 180, 60], p=[0.4, 0.4, 0.2])
        chi_angles[i, 0] = chi1_center + np.random.randn() * 15

        # Chi2 - similar distribution
        if res_name in ["LEU", "ILE", "PHE", "TYR", "TRP", "HIS", "ASN", "ASP", "GLU", "GLN", "LYS", "ARG", "MET"]:
            chi2_center = np.random.choice([-60, 180, 60], p=[0.35, 0.45, 0.2])
            chi_angles[i, 1] = chi2_center + np.random.randn() * 15
        else:
            chi_angles[i, 1] = np.nan

        # Chi3 - fewer residues have it
        if res_name in ["GLU", "GLN", "LYS", "ARG", "MET"]:
            chi_angles[i, 2] = np.random.choice([-60, 180, 60]) + np.random.randn() * 20
        else:
            chi_angles[i, 2] = np.nan

        # Chi4 - only LYS and ARG
        if res_name in ["LYS", "ARG"]:
            chi_angles[i, 3] = np.random.choice([-60, 180, 60]) + np.random.randn() * 20
        else:
            chi_angles[i, 3] = np.nan

    metadata = [
        {
            "pdb_id": f"DEMO{i // 100:02d}",
            "chain_id": "A",
            "residue_id": positions[i],
            "residue_name": residue_names[i],
            "sequence_context": "XXX" + residue_names[i][:1] + "XXX",  # Simplified context
        }
        for i in range(n_residues)
    ]

    save_dict = {
        "chi_angles": torch.tensor(chi_angles, dtype=torch.float32),
        "metadata": metadata,
        "pdb_ids": [f"DEMO{i:02d}" for i in range(5)],
        "n_residues": n_residues,
    }

    torch.save(save_dict, output_path)
    print(f"Created demo rotamer data at {output_path}")
    print(f"  {n_residues} residues from 5 demo structures")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ingest PDB structures and extract rotamer angles")
    parser.add_argument(
        "--pdb_ids",
        type=str,
        default=None,
        help='Comma-separated PDB IDs (e.g., "1CRN,1TIM,4LZT")',
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROCESSED_DATA_DIR / "rotamers.pt"),
        help="Output path for tensor",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(RAW_DATA_DIR / "pdb_cache"),
        help="Cache directory for PDB files",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Create demo data instead of downloading",
    )

    args = parser.parse_args()

    if args.demo:
        create_demo_rotamer_data(Path(args.output))
        return

    if not args.pdb_ids:
        print("Error: --pdb_ids required unless using --demo mode")
        return

    pdb_ids = [p.strip() for p in args.pdb_ids.split(",")]
    print(f"Processing {len(pdb_ids)} PDB structures...")

    ingest_pdb_rotamers(
        pdb_ids=pdb_ids,
        output_path=Path(args.output),
        cache_dir=Path(args.cache_dir),
    )


if __name__ == "__main__":
    main()
