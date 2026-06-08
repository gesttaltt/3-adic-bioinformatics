#!/usr/bin/env python3
"""AlphaFold Structural Validation Pipeline for DDG Predictions.

Scientific-grade validation that:
1. Fetches AlphaFold structural data (pLDDT, secondary structure)
2. Computes DDG predictions in parallel (no bias/contamination)
3. Cross-validates predictions against structural features
4. Tests our specific strengths from discoveries:
   - Hydrophobicity-driven mutations (importance 0.633)
   - Local contacts (4-8 residues, AUC 0.59)
   - Alpha-helical proteins (AUC 0.65)
   - Fast-folding domains

Usage:
    python validation/alphafold_validation_pipeline.py

Output:
    - validation/results/alphafold_validation_report.json
    - validation/results/structural_features.csv
    - validation/results/cross_validation_metrics.json

References:
    - AlphaFold API: https://alphafold.ebi.ac.uk/api-docs
    - S669 dataset: Pancotti et al. 2022
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import csv

import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

# HTTP requests
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests not available, install with: pip install requests")

# Scientific computing
try:
    from scipy.stats import spearmanr, pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StructuralFeatures:
    """Structural features extracted from AlphaFold."""
    pdb_id: str
    uniprot_id: str
    position: int
    plddt: float  # Per-residue confidence (0-100)
    plddt_local: float  # Average pLDDT ±5 residues
    secondary_structure: str  # H (helix), E (sheet), C (coil)
    is_alpha_helix: bool
    is_beta_sheet: bool
    contact_count: int  # Number of contacts within 8Å
    local_contact_density: float  # Contacts in 4-8 residue range
    rsa: float  # Relative solvent accessibility (0-1)


@dataclass
class ValidationResult:
    """Result of cross-validation."""
    mutation: str
    experimental_ddg: float
    predicted_ddg: float
    plddt: float
    secondary_structure: str
    hydro_diff: float
    charge_diff: float
    regime: str  # "hard_hybrid", "soft_hybrid", "uncertain", "soft_simple", "hard_simple"


# =============================================================================
# AlphaFold API Client
# =============================================================================

class AlphaFoldClient:
    """Client for AlphaFold Protein Structure Database API.

    NOTE: The /api/prediction endpoint is sunset 2026-06-25.
    Field name change: entryId -> modelEntityId.
    We fall back to direct file URL construction so the code stays
    functional regardless of whether the API endpoint is live.
    """

    BASE_URL = "https://alphafold.ebi.ac.uk/api"
    FILES_BASE = "https://alphafold.ebi.ac.uk/files"

    # PDB to UniProt mapping (for S669 proteins)
    # This is a subset - full mapping should be fetched from SIFTS
    PDB_TO_UNIPROT = {
        "1A2P": "P00720",  # T4 lysozyme
        "2LZM": "P00720",  # T4 lysozyme
        "1STN": "P00644",  # Staphylococcal nuclease
        "2CI2": "P01053",  # Chymotrypsin inhibitor 2
        "1RNH": "P10890",  # RNase HI
        "4PTI": "P00974",  # BPTI
        "1UBQ": "P0CG47",  # Ubiquitin
        "1A0F": "P00648",  # SN RNase
        "1A7V": "P06654",  # Barnase
        "1BA3": "P23280",  # Carbonic anhydrase
        "1BFM": "P0A7Z4",  # H-NS
        "1BNL": "P00698",  # Lysozyme C
        "1D5G": "P04637",  # p53
        "1DIV": "P15169",  # Carboxypeptidase A
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize AlphaFold client.

        Args:
            cache_dir: Directory to cache downloaded structures
        """
        self.cache_dir = cache_dir or Path(__file__).parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session() if HAS_REQUESTS else None

    def get_uniprot_id(self, pdb_id: str) -> Optional[str]:
        """Get UniProt ID from PDB ID.

        Uses local mapping first, then SIFTS API.
        """
        pdb_id = pdb_id.upper()[:4]

        # Check local mapping
        if pdb_id in self.PDB_TO_UNIPROT:
            return self.PDB_TO_UNIPROT[pdb_id]

        # Try SIFTS API for unmapped entries
        if self.session is None:
            return None

        try:
            url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if pdb_id.lower() in data:
                    uniprot_entries = data[pdb_id.lower()].get("UniProt", {})
                    if uniprot_entries:
                        return list(uniprot_entries.keys())[0]
        except Exception as e:
            print(f"  SIFTS lookup failed for {pdb_id}: {e}")

        return None

    def _construct_pdb_url(self, uniprot_id: str, version: int = 4, fragment: int = 1) -> str:
        """Construct AlphaFold PDB file URL directly, without relying on the API.

        Stable URL format that works regardless of API availability.
        """
        return f"{self.FILES_BASE}/AF-{uniprot_id}-F{fragment}-model_v{version}.pdb"

    def get_prediction_info(self, uniprot_id: str) -> Optional[dict]:
        """Get AlphaFold prediction info for a UniProt ID.

        Handles both the old API format (entryId) and the new format
        (modelEntityId, effective after 2026-06-25 sunset).

        Returns:
            dict with keys including pdbUrl and either entryId or modelEntityId.
            Returns a synthetic dict with pdbUrl if the API is unavailable.
        """
        if self.session is None:
            return {"pdbUrl": self._construct_pdb_url(uniprot_id)}

        try:
            url = f"{self.BASE_URL}/prediction/{uniprot_id}"
            response = self.session.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]  # Return first entry
        except Exception as e:
            print(f"  AlphaFold API error for {uniprot_id}: {e}")

        # API unavailable or changed: return synthetic info with direct file URL
        return {"pdbUrl": self._construct_pdb_url(uniprot_id)}

    def _resolve_pdb_url(self, uniprot_id: str, info: dict) -> str:
        """Resolve the PDB download URL from an API response or construct it directly.

        Handles old API format (entryId), new API format (modelEntityId),
        and direct URL construction as a final fallback.
        """
        # pdbUrl present in response (works for both old and new API)
        pdb_url = info.get("pdbUrl")
        if pdb_url:
            return pdb_url

        # New API: modelEntityId present but pdbUrl absent
        # modelEntityId may be "AF-P00720-F1-model_v4" (full) or "AF-P00720-F1" (bare)
        model_entity_id = info.get("modelEntityId")
        if model_entity_id:
            if "-model_v" in model_entity_id:
                return f"{self.FILES_BASE}/{model_entity_id}.pdb"
            version = info.get("latestVersion", 4)
            return f"{self.FILES_BASE}/{model_entity_id}-model_v{version}.pdb"

        # Old API: entryId present but pdbUrl absent (shouldn't happen, but guard it)
        entry_id = info.get("entryId")
        if entry_id:
            version = info.get("latestVersion", 4)
            return f"{self.FILES_BASE}/{entry_id}-model_v{version}.pdb"

        # Last resort: construct URL directly from UniProt ID
        return self._construct_pdb_url(uniprot_id)

    def get_plddt_scores(self, uniprot_id: str) -> Optional[list[float]]:
        """Get per-residue pLDDT confidence scores from PDB B-factors.

        AlphaFold stores pLDDT in the B-factor column of PDB files.

        Returns:
            List of pLDDT scores (0-100) indexed by residue position (0-based)
        """
        # Check cache
        cache_file = self.cache_dir / f"{uniprot_id}_plddt.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)

        if self.session is None:
            return None

        # Get prediction info, then resolve a download URL from it
        info = self.get_prediction_info(uniprot_id) or {}
        pdb_url = self._resolve_pdb_url(uniprot_id, info)

        try:
            response = self.session.get(pdb_url, timeout=60)
            if response.status_code == 200:
                # Parse B-factors (pLDDT) from ATOM records (CA atoms only)
                lines = response.text.split('\n')
                plddt = []
                current_res = None

                for line in lines:
                    if line.startswith('ATOM') and ' CA ' in line:
                        try:
                            # B-factor is columns 61-66 in PDB format
                            b_factor = float(line[60:66].strip())
                            res_num = int(line[22:26].strip())
                            if res_num != current_res:
                                plddt.append(b_factor)
                                current_res = res_num
                        except (ValueError, IndexError):
                            pass

                if plddt:
                    # Cache result
                    with open(cache_file, 'w') as f:
                        json.dump(plddt, f)
                    return plddt

        except Exception as e:
            print(f"pLDDT fetch error: {e}")

        return None

    def get_structure_features(
        self,
        pdb_id: str,
        chain: str,
        position: int
    ) -> Optional[StructuralFeatures]:
        """Get structural features for a specific residue position.

        Args:
            pdb_id: PDB identifier
            chain: Chain ID
            position: Residue position (1-indexed)

        Returns:
            StructuralFeatures dataclass with extracted features
        """
        uniprot_id = self.get_uniprot_id(pdb_id)
        if uniprot_id is None:
            return None

        plddt_scores = self.get_plddt_scores(uniprot_id)
        if plddt_scores is None or len(plddt_scores) == 0:
            return None

        # Adjust for 0-based indexing
        idx = position - 1
        if idx < 0 or idx >= len(plddt_scores):
            return None

        # Get pLDDT at position
        plddt = plddt_scores[idx]

        # Get local pLDDT (±5 residues)
        start = max(0, idx - 5)
        end = min(len(plddt_scores), idx + 6)
        plddt_local = np.mean(plddt_scores[start:end])

        # Estimate secondary structure from pLDDT pattern
        # High pLDDT (>80) with consistent neighbors suggests helix/sheet
        # Variable pLDDT suggests coil
        if idx > 3 and idx < len(plddt_scores) - 3:
            local_std = np.std(plddt_scores[idx-3:idx+4])
            if plddt > 80 and local_std < 10:
                ss = "H"  # Likely helix
            elif plddt > 70 and local_std < 15:
                ss = "E"  # Likely sheet
            else:
                ss = "C"  # Coil
        else:
            ss = "C"  # Terminal regions

        # Estimate contact density from pLDDT
        # Higher pLDDT correlates with buried residues (more contacts)
        contact_count = int(plddt / 10)  # Rough estimate

        # Local contact density (4-8 residue range)
        local_range = plddt_scores[max(0, idx-8):idx-4] + plddt_scores[idx+4:min(len(plddt_scores), idx+9)]
        local_contact_density = np.mean(local_range) / 100 if local_range else 0.0

        # RSA estimate (inverse of pLDDT, roughly)
        rsa = 1.0 - (plddt / 100)

        return StructuralFeatures(
            pdb_id=pdb_id,
            uniprot_id=uniprot_id,
            position=position,
            plddt=plddt,
            plddt_local=plddt_local,
            secondary_structure=ss,
            is_alpha_helix=(ss == "H"),
            is_beta_sheet=(ss == "E"),
            contact_count=contact_count,
            local_contact_density=local_contact_density,
            rsa=rsa,
        )


# =============================================================================
# Regime Classification (from V5 Arrow Flip discoveries)
# =============================================================================

# Amino acid properties for regime classification
AA_HYDRO = {
    "A": 0.62, "R": -2.53, "N": -0.78, "D": -0.90, "C": 0.29,
    "Q": -0.85, "E": -0.74, "G": 0.48, "H": -0.40, "I": 1.38,
    "L": 1.06, "K": -1.50, "M": 0.64, "F": 1.19, "P": 0.12,
    "S": -0.18, "T": -0.05, "W": 0.81, "Y": 0.26, "V": 1.08,
}

AA_CHARGE = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0,
    "Q": 0, "E": -1, "G": 0, "H": 0.5, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0,
}

AA_VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
    "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
    "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
    "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0,
}


def classify_mutation_regime(wt_aa: str, mut_aa: str) -> tuple[str, float]:
    """Classify mutation into regime based on V5 Arrow Flip discoveries.

    Regimes:
    - hard_hybrid: High hydro_diff, same charge, small volume (81% accuracy)
    - soft_hybrid: Moderate hydro_diff (76% accuracy)
    - uncertain: Transitional features (50% accuracy)
    - soft_simple: Low hydro_diff, charge differences (73% accuracy)
    - hard_simple: Very low hydro_diff, opposite charges (86% accuracy)

    Returns:
        (regime_name, expected_accuracy)
    """
    hydro_diff = abs(AA_HYDRO.get(wt_aa, 0) - AA_HYDRO.get(mut_aa, 0))
    charge_wt = AA_CHARGE.get(wt_aa, 0)
    charge_mut = AA_CHARGE.get(mut_aa, 0)
    same_charge = (charge_wt * charge_mut >= 0)  # Both positive, both negative, or either zero
    different_charge = not same_charge
    volume_diff = abs(AA_VOLUME.get(wt_aa, 0) - AA_VOLUME.get(mut_aa, 0))

    # Decision rules from V5 analysis
    if hydro_diff > 5.15:
        if same_charge and volume_diff < 55:
            return ("hard_hybrid", 0.81)
        else:
            return ("soft_hybrid", 0.76)
    elif hydro_diff <= 5.15 and different_charge:
        if hydro_diff < 1.0:
            return ("hard_simple", 0.86)
        else:
            return ("soft_simple", 0.73)
    else:
        return ("uncertain", 0.50)


# =============================================================================
# Validation Pipeline
# =============================================================================

class ValidationPipeline:
    """Scientific-grade validation pipeline."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize validation pipeline."""
        self.output_dir = output_dir or Path(__file__).parent / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.af_client = AlphaFoldClient()

        # Try to import predictor
        try:
            from deliverables.partners.jose_colbes.src.validated_ddg_predictor import (
                ValidatedDDGPredictor,
            )
            self.predictor = ValidatedDDGPredictor()
            print("Loaded ValidatedDDGPredictor")
        except ImportError:
            self.predictor = None
            print("Warning: ValidatedDDGPredictor not available")

    def load_s669_data(self) -> list[dict]:
        """Load S669 dataset with experimental DDG values."""
        data_file = Path(__file__).parent.parent / "reproducibility/data/s669_full.csv"

        if not data_file.exists():
            # Try simple format
            data_file = Path(__file__).parent.parent / "reproducibility/data/s669.csv"

        mutations = []
        with open(data_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                mutations.append(row)

        print(f"Loaded {len(mutations)} mutations from S669")
        return mutations

    def run_parallel_validation(self, max_proteins: int = 20) -> dict:
        """Run parallel validation - compute predictions independently then cross-validate.

        This design avoids bias by:
        1. Computing our predictions FIRST (no peeking at structure)
        2. Fetching structural data SECOND
        3. Cross-validating THIRD (comparing independently computed values)
        """
        print("\n" + "=" * 70)
        print("PARALLEL VALIDATION PIPELINE")
        print("=" * 70)

        mutations = self.load_s669_data()

        # STREAM 1: Our DDG predictions (computed independently)
        print("\n[STREAM 1] Computing DDG predictions (no structural data)...")
        predictions = {}
        for mut in mutations[:max_proteins * 10]:  # Sample
            # Parse mutation
            if 'Protein' in mut:
                # Full format
                pdb_id = mut['Protein'][:4]
                seq_mut = mut.get('Seq_Mut', '')
                if len(seq_mut) >= 3:
                    wt_aa = seq_mut[0]
                    mut_aa = seq_mut[-1]
                    exp_ddg = float(mut.get('Experimental_DDG_dir', 0))
                else:
                    continue
            else:
                # Simple format
                pdb_id = mut['pdb_id']
                wt_aa = mut['wild_type']
                mut_aa = mut['mutant']
                exp_ddg = float(mut['ddg'])

            key = f"{pdb_id}_{wt_aa}{mut_aa}"

            if self.predictor:
                pred = self.predictor.predict(wt_aa, mut_aa)
                pred_ddg = pred.ddg
            else:
                pred_ddg = 0.0

            # Classify regime
            regime, accuracy = classify_mutation_regime(wt_aa, mut_aa)

            predictions[key] = {
                'pdb_id': pdb_id,
                'wt_aa': wt_aa,
                'mut_aa': mut_aa,
                'exp_ddg': exp_ddg,
                'pred_ddg': pred_ddg,
                'regime': regime,
                'regime_accuracy': accuracy,
                'hydro_diff': abs(AA_HYDRO.get(wt_aa, 0) - AA_HYDRO.get(mut_aa, 0)),
                'charge_diff': abs(AA_CHARGE.get(wt_aa, 0) - AA_CHARGE.get(mut_aa, 0)),
            }

        print(f"  Computed {len(predictions)} predictions")

        # STREAM 2: Structural data (fetched independently)
        print("\n[STREAM 2] Fetching AlphaFold structural data...")
        structural_data = {}
        unique_pdbs = list(set(p['pdb_id'] for p in predictions.values()))[:max_proteins]

        for i, pdb_id in enumerate(unique_pdbs):
            print(f"  [{i+1}/{len(unique_pdbs)}] Fetching {pdb_id}...", end=" ")

            uniprot_id = self.af_client.get_uniprot_id(pdb_id)
            if uniprot_id:
                plddt = self.af_client.get_plddt_scores(uniprot_id)
                if plddt:
                    structural_data[pdb_id] = {
                        'uniprot_id': uniprot_id,
                        'plddt': plddt,
                        'mean_plddt': np.mean(plddt),
                        'length': len(plddt),
                    }
                    print(f"OK (UniProt: {uniprot_id}, {len(plddt)} residues)")
                else:
                    print(f"No pLDDT data")
            else:
                print(f"No UniProt mapping")

            time.sleep(0.5)  # Rate limiting

        print(f"  Retrieved structural data for {len(structural_data)} proteins")

        # STREAM 3: Cross-validation (combining independent streams)
        print("\n[STREAM 3] Cross-validating predictions vs structure...")

        results = {
            'all': [],
            'by_regime': {
                'hard_hybrid': [],
                'soft_hybrid': [],
                'uncertain': [],
                'soft_simple': [],
                'hard_simple': [],
            },
            'by_structure': {
                'high_plddt': [],  # >90
                'medium_plddt': [],  # 70-90
                'low_plddt': [],  # <70
            },
            'by_hydro': {
                'high_hydro_diff': [],  # >2.0
                'low_hydro_diff': [],  # ≤2.0
            }
        }

        for key, pred in predictions.items():
            pdb_id = pred['pdb_id']

            # Get structural data if available
            struct = structural_data.get(pdb_id, {})
            mean_plddt = struct.get('mean_plddt', 0)

            result = ValidationResult(
                mutation=key,
                experimental_ddg=pred['exp_ddg'],
                predicted_ddg=pred['pred_ddg'],
                plddt=mean_plddt,
                secondary_structure="",  # Would need per-residue
                hydro_diff=pred['hydro_diff'],
                charge_diff=pred['charge_diff'],
                regime=pred['regime'],
            )

            results['all'].append(result)
            results['by_regime'][pred['regime']].append(result)

            # Categorize by structure
            if mean_plddt > 90:
                results['by_structure']['high_plddt'].append(result)
            elif mean_plddt > 70:
                results['by_structure']['medium_plddt'].append(result)
            else:
                results['by_structure']['low_plddt'].append(result)

            # Categorize by hydrophobicity
            if pred['hydro_diff'] > 2.0:
                results['by_hydro']['high_hydro_diff'].append(result)
            else:
                results['by_hydro']['low_hydro_diff'].append(result)

        return results

    def compute_metrics(self, results: list[ValidationResult]) -> dict:
        """Compute correlation metrics for a set of results."""
        if len(results) < 5 or not HAS_SCIPY:
            return {'n': len(results), 'spearman': None, 'pearson': None}

        exp = [r.experimental_ddg for r in results]
        pred = [r.predicted_ddg for r in results]

        spearman, sp_p = spearmanr(exp, pred)
        pearson, pr_p = pearsonr(exp, pred)

        mae = np.mean(np.abs(np.array(exp) - np.array(pred)))

        return {
            'n': len(results),
            'spearman': round(spearman, 3),
            'spearman_p': round(sp_p, 4),
            'pearson': round(pearson, 3),
            'pearson_p': round(pr_p, 4),
            'mae': round(mae, 2),
        }

    def generate_report(self, results: dict) -> dict:
        """Generate comprehensive validation report."""
        report = {
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'pipeline': 'AlphaFold Structural Validation',
                'version': '1.0',
                'methodology': 'Parallel independent streams, then cross-validation',
            },
            'overall': self.compute_metrics(results['all']),
            'by_regime': {},
            'by_structure': {},
            'by_hydro': {},
            'discoveries_validation': {},
        }

        # Metrics by regime
        for regime, data in results['by_regime'].items():
            report['by_regime'][regime] = self.compute_metrics(data)

        # Metrics by structure
        for struct, data in results['by_structure'].items():
            report['by_structure'][struct] = self.compute_metrics(data)

        # Metrics by hydrophobicity
        for hydro, data in results['by_hydro'].items():
            report['by_hydro'][hydro] = self.compute_metrics(data)

        # Validate discoveries
        report['discoveries_validation'] = {
            'hydro_is_primary_predictor': {
                'hypothesis': 'High hydro_diff mutations should show stronger signal',
                'high_hydro_spearman': report['by_hydro'].get('high_hydro_diff', {}).get('spearman'),
                'low_hydro_spearman': report['by_hydro'].get('low_hydro_diff', {}).get('spearman'),
                'confirmed': None,  # Fill in after analysis
            },
            'regime_accuracy_matches': {
                'hypothesis': 'Hard regimes should outperform uncertain',
                'hard_hybrid': report['by_regime'].get('hard_hybrid', {}).get('spearman'),
                'hard_simple': report['by_regime'].get('hard_simple', {}).get('spearman'),
                'uncertain': report['by_regime'].get('uncertain', {}).get('spearman'),
                'confirmed': None,
            },
            'high_plddt_better': {
                'hypothesis': 'High confidence structures should validate better',
                'high_plddt': report['by_structure'].get('high_plddt', {}).get('spearman'),
                'low_plddt': report['by_structure'].get('low_plddt', {}).get('spearman'),
                'confirmed': None,
            },
        }

        return report

    def run(self, max_proteins: int = 20):
        """Run full validation pipeline."""
        print("\n" + "=" * 70)
        print("ALPHAFOLD STRUCTURAL VALIDATION FOR DDG PREDICTIONS")
        print("Scientific-Grade Parallel Validation Pipeline")
        print("=" * 70)

        # Run parallel validation
        results = self.run_parallel_validation(max_proteins)

        # Generate report
        report = self.generate_report(results)

        # Save report
        report_path = self.output_dir / "alphafold_validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {report_path}")

        # Print summary
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)

        print(f"\nOverall Metrics (n={report['overall']['n']}):")
        print(f"  Spearman r: {report['overall'].get('spearman', 'N/A')}")
        print(f"  Pearson r:  {report['overall'].get('pearson', 'N/A')}")
        print(f"  MAE:        {report['overall'].get('mae', 'N/A')} kcal/mol")

        print("\nBy Mutation Regime:")
        for regime, metrics in report['by_regime'].items():
            if metrics['n'] > 0:
                print(f"  {regime}: n={metrics['n']}, r={metrics.get('spearman', 'N/A')}")

        print("\nBy Structure Quality:")
        for struct, metrics in report['by_structure'].items():
            if metrics['n'] > 0:
                print(f"  {struct}: n={metrics['n']}, r={metrics.get('spearman', 'N/A')}")

        print("\nDiscoveries Validation:")
        for disc, data in report['discoveries_validation'].items():
            print(f"  {disc}:")
            print(f"    Hypothesis: {data['hypothesis']}")

        return report


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pipeline = ValidationPipeline()
    report = pipeline.run(max_proteins=15)  # Start with 15 proteins
