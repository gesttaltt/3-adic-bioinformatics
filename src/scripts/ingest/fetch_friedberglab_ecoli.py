# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Fetch and process FriedbergLab E. coli AMR dataset from Figshare.

This script downloads the small (1.88 MB) E. coli AMR dataset from Figshare
and processes it for validation with our TEM beta-lactamase analyzer.

Dataset Info:
- Source: FriedbergLab (PLOS One 2023)
- Figshare DOI: 10.6084/m9.figshare.21737288.v1
- Size: 1.88 MB (3 CSV files)
- GitHub: https://github.com/FriedbergLab/AMREcoli/

Files:
- fg_metadata.csv - Sample metadata (host, tissue, diagnosis)
- fg_genotype.csv - Genotype data (genes, AMR families)
- fg_phenotype.csv - Phenotype data (Sensitive/Resistant)

Usage:
    # Download and process data
    python src/scripts/ingest/fetch_friedberglab_ecoli.py

    # Just download
    python src/scripts/ingest/fetch_friedberglab_ecoli.py --download-only

    # Process existing data
    python src/scripts/ingest/fetch_friedberglab_ecoli.py --process-only
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Figshare API endpoints
FIGSHARE_ARTICLE_ID = "21737288"
FIGSHARE_API_URL = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}"

# Default paths
DEFAULT_DATA_DIR = Path("data/external/friedberglab_ecoli")
DEFAULT_PROCESSED_DIR = Path("data/processed/ecoli_friedberg")

# Beta-lactam related antibiotics in the dataset
BETA_LACTAM_ANTIBIOTICS = [
    "ampicillin",
    "amoxicillin",
    "amoxicillin-clavulanic acid",
    "cefazolin",
    "cefovecin",
    "cefoxitin",
    "cefpodoxime",
    "ceftiofur",
    "ceftriaxone",
    "cephalothin",
    "imipenem",
    "ticarcillin-clavulanic acid",
]

# TEM-related gene families
TEM_GENE_FAMILIES = [
    "TEM",
    "beta-lactamase",
    "class A beta-lactamase",
]


def get_figshare_files() -> list[dict]:
    """Get list of files from Figshare article via API.

    Returns:
        List of file metadata dicts with name, size, download_url
    """
    logger.info(f"Fetching file list from Figshare article {FIGSHARE_ARTICLE_ID}...")

    response = requests.get(FIGSHARE_API_URL, timeout=30)
    response.raise_for_status()

    article = response.json()
    files = article.get("files", [])

    logger.info(f"Found {len(files)} files:")
    for f in files:
        size_kb = f.get("size", 0) / 1024
        logger.info(f"  - {f['name']} ({size_kb:.1f} KB)")

    return files


def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL.

    Args:
        url: Download URL
        output_path: Path to save file

    Returns:
        True if successful
    """
    # Headers to mimic browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    try:
        logger.info(f"Downloading {output_path.name}...")
        response = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
        response.raise_for_status()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(response.content)

        logger.info(f"Saved to {output_path} ({len(response.content) / 1024:.1f} KB)")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def download_all_files(output_dir: Path) -> bool:
    """Download all files from Figshare.

    Args:
        output_dir: Directory to save files

    Returns:
        True if all files downloaded successfully
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    files = get_figshare_files()
    success = True

    for file_info in files:
        name = file_info["name"]
        url = file_info["download_url"]
        output_path = output_dir / name

        if not download_file(url, output_path):
            success = False

    return success


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the three CSV files.

    Args:
        data_dir: Directory containing CSV files

    Returns:
        Tuple of (metadata_df, genotype_df, phenotype_df)
    """
    metadata_path = data_dir / "fg_metadata.csv"
    # Note: actual filenames use plural forms
    genotype_path = data_dir / "fg_genotypes.csv"
    phenotype_path = data_dir / "fg_phenotypes.csv"

    logger.info("Loading data files...")

    metadata_df = pd.read_csv(metadata_path)
    genotype_df = pd.read_csv(genotype_path)
    phenotype_df = pd.read_csv(phenotype_path)

    logger.info(f"Metadata: {len(metadata_df)} samples")
    logger.info(f"Genotype: {len(genotype_df)} gene entries")
    logger.info(f"Phenotype: {len(phenotype_df)} resistance entries")

    return metadata_df, genotype_df, phenotype_df


def analyze_data(
    metadata_df: pd.DataFrame,
    genotype_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
) -> dict:
    """Analyze the dataset structure.

    Args:
        metadata_df: Sample metadata
        genotype_df: Genotype data (columns: sample_id, gene, gene_type, gene_identifier, gene_name_family, resistance_class)
        phenotype_df: Phenotype data (columns: sample_id, mic_id, breakpoint, phenotype)

    Returns:
        Analysis summary dict
    """
    logger.info("\n" + "=" * 60)
    logger.info("DATASET ANALYSIS")
    logger.info("=" * 60)

    # Basic stats
    n_samples = len(metadata_df)
    logger.info(f"\nTotal samples: {n_samples}")

    # Host animals
    if "host_animal_common" in metadata_df.columns:
        hosts = metadata_df["host_animal_common"].value_counts()
        logger.info("\nHost animals:")
        for host, count in hosts.items():
            logger.info(f"  {host}: {count}")

    # Resistance classes
    if "resistance_class" in genotype_df.columns:
        classes = genotype_df["resistance_class"].value_counts()
        logger.info("\nResistance classes:")
        for cls, count in classes.head(10).items():
            logger.info(f"  {cls}: {count}")

    # Beta-lactamase genes (look in gene_name_family and gene columns)
    gene_col = "gene_name_family" if "gene_name_family" in genotype_df.columns else "gene"
    beta_lac_mask = genotype_df[gene_col].str.contains("TEM|SHV|CTX|OXA|ampC|bla", case=False, na=False) | genotype_df[
        "resistance_class"
    ].str.contains("beta-lactam", case=False, na=False)
    beta_lac_genes = genotype_df[beta_lac_mask]
    logger.info(f"\nBeta-lactam resistance genes found: {len(beta_lac_genes)}")

    if len(beta_lac_genes) > 0:
        bl_families = beta_lac_genes[gene_col].value_counts()
        logger.info("Top beta-lactam genes:")
        for family, count in bl_families.head(15).items():
            logger.info(f"  {family}: {count}")

    # Phenotype data - get unique antibiotics from mic_id column
    antibiotics = phenotype_df["mic_id"].unique()
    logger.info(f"\nAntibiotics tested: {len(antibiotics)}")

    # Beta-lactam antibiotics
    beta_lactam_keywords = [
        "ampicillin",
        "amox",
        "cef",
        "ceph",
        "imipenem",
        "ticarcillin",
        "piperacillin",
        "penicillin",
    ]
    beta_lactam_abs = [ab for ab in antibiotics if any(kw in ab.lower() for kw in beta_lactam_keywords)]
    logger.info(f"Beta-lactam antibiotics: {len(beta_lactam_abs)}")
    for ab in sorted(beta_lactam_abs):
        logger.info(f"  - {ab}")

    # Resistance distribution for beta-lactams
    logger.info("\nResistance distribution (beta-lactams):")
    for ab in sorted(beta_lactam_abs):
        ab_data = phenotype_df[phenotype_df["mic_id"] == ab]
        resistant = (ab_data["phenotype"] == "R").sum()
        sensitive = (ab_data["phenotype"] == "S").sum()
        total = resistant + sensitive
        if total > 0:
            pct = resistant / total * 100
            logger.info(f"  {ab}: {resistant}/{total} resistant ({pct:.1f}%)")

    return {
        "n_samples": n_samples,
        "n_antibiotics": len(antibiotics),
        "n_beta_lactams": len(beta_lactam_abs),
        "beta_lactam_abs": beta_lactam_abs,
        "n_beta_lactamase_genes": len(beta_lac_genes),
    }


def create_tem_dataset(
    genotype_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
    drug: str = "ampicillin",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Create dataset for TEM beta-lactamase validation.

    Args:
        genotype_df: Genotype data with gene info
        phenotype_df: Phenotype data (columns: sample_id, mic_id, breakpoint, phenotype)
        drug: Drug to predict resistance for

    Returns:
        (X, y, sample_ids) tuple
    """
    logger.info(f"\nCreating dataset for {drug}...")

    # Filter phenotype data for this drug
    drug_data = phenotype_df[phenotype_df["mic_id"].str.lower().str.contains(drug.lower())]

    if len(drug_data) == 0:
        logger.error(f"Drug '{drug}' not found in phenotype data")
        # List available drugs
        available = phenotype_df["mic_id"].unique()
        logger.info(f"Available antibiotics: {sorted(available)}")
        return np.array([]), np.array([]), []

    logger.info(f"Found {len(drug_data)} phenotype records for {drug}")

    # Filter for valid S/R phenotypes
    valid_mask = drug_data["phenotype"].isin(["R", "S"])
    valid_data = drug_data[valid_mask]

    logger.info(f"Samples with valid S/R phenotype: {len(valid_data)}")

    # Get unique samples
    sample_ids = valid_data["sample_id"].unique().tolist()
    logger.info(f"Unique samples: {len(sample_ids)}")

    # Create labels (take first phenotype if duplicates)
    y_labels = []
    for sample_id in sample_ids:
        sample_pheno = valid_data[valid_data["sample_id"] == sample_id]["phenotype"].iloc[0]
        y_labels.append(1.0 if sample_pheno == "R" else 0.0)

    y = np.array(y_labels)

    # Create feature matrix based on beta-lactamase gene presence
    logger.info("Creating feature matrix from genotype data...")

    # Get unique beta-lactamase genes
    gene_col = "gene_name_family" if "gene_name_family" in genotype_df.columns else "gene"
    beta_lac_mask = genotype_df[gene_col].str.contains("TEM|SHV|CTX|OXA|ampC|bla", case=False, na=False) | genotype_df[
        "resistance_class"
    ].str.contains("beta-lactam", case=False, na=False)
    beta_lac_genes = genotype_df[beta_lac_mask]

    # Get unique gene identifiers
    unique_genes = beta_lac_genes["gene_identifier"].unique()
    logger.info(f"Unique beta-lactam resistance genes: {len(unique_genes)}")

    # Create gene presence matrix
    X = np.zeros((len(sample_ids), len(unique_genes)))

    for i, sample_id in enumerate(sample_ids):
        sample_genes = genotype_df[genotype_df["sample_id"] == sample_id]
        sample_beta_lac = sample_genes[
            (sample_genes[gene_col].str.contains("TEM|SHV|CTX|OXA|ampC|bla", case=False, na=False))
            | (sample_genes["resistance_class"].str.contains("beta-lactam", case=False, na=False))
        ]

        for j, gene in enumerate(unique_genes):
            if gene in sample_beta_lac["gene_identifier"].values:
                X[i, j] = 1.0

    logger.info(f"Feature matrix shape: {X.shape}")
    n_with_genes = (X.sum(axis=1) > 0).sum()
    logger.info(f"Samples with any beta-lactam gene: {n_with_genes} ({n_with_genes / len(sample_ids) * 100:.1f}%)")

    return X, y, sample_ids


def validate_with_analyzer(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: list[str],
) -> dict:
    """Validate using Ridge regression baseline.

    Args:
        X: Feature matrix
        y: Target labels
        sample_ids: Sample IDs

    Returns:
        Validation results dict
    """
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import cross_val_predict

    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)

    if len(X) == 0 or len(y) == 0:
        logger.error("Empty dataset, cannot validate")
        return {}

    # Basic stats
    n_resistant = (y == 1).sum()
    n_sensitive = (y == 0).sum()
    logger.info(f"\nDataset: {len(y)} samples")
    logger.info(f"Resistant: {n_resistant} ({n_resistant / len(y) * 100:.1f}%)")
    logger.info(f"Sensitive: {n_sensitive} ({n_sensitive / len(y) * 100:.1f}%)")

    # Ridge classifier with cross-validation
    model = RidgeClassifier(alpha=1.0)

    # Use fewer folds if small dataset
    n_folds = min(5, len(y) // 2)
    if n_folds < 2:
        logger.warning("Dataset too small for cross-validation")
        return {"error": "dataset_too_small"}

    logger.info(f"\nRunning {n_folds}-fold cross-validation...")

    # Predict probabilities (decision function)
    y_pred = cross_val_predict(model, X, y, cv=n_folds)
    y_scores = cross_val_predict(model, X, y, cv=n_folds, method="decision_function")

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="weighted")

    # Spearman correlation
    spearman_corr, spearman_p = stats.spearmanr(y, y_scores)

    # ROC AUC if binary
    try:
        roc_auc = roc_auc_score(y, y_scores)
    except Exception:
        roc_auc = None

    logger.info("\nResults:")
    logger.info(f"  Accuracy: {accuracy:.3f}")
    logger.info(f"  F1 Score: {f1:.3f}")
    logger.info(f"  Spearman: {spearman_corr:.3f} (p={spearman_p:.2e})")
    if roc_auc:
        logger.info(f"  ROC AUC: {roc_auc:.3f}")

    # Feature importance
    model.fit(X, y)
    np.abs(model.coef_).flatten()

    return {
        "n_samples": len(y),
        "n_resistant": int(n_resistant),
        "n_sensitive": int(n_sensitive),
        "accuracy": accuracy,
        "f1_score": f1,
        "spearman": spearman_corr,
        "spearman_p": spearman_p,
        "roc_auc": roc_auc,
        "n_features": X.shape[1],
    }


def validate_multiple_drugs(
    genotype_df: pd.DataFrame,
    phenotype_df: pd.DataFrame,
) -> dict:
    """Validate across multiple beta-lactam antibiotics.

    Args:
        genotype_df: Genotype data
        phenotype_df: Phenotype data (columns: sample_id, mic_id, breakpoint, phenotype)

    Returns:
        Results dict for all drugs
    """
    # Get unique antibiotics from mic_id column
    all_antibiotics = phenotype_df["mic_id"].unique()

    # Find beta-lactam antibiotics
    beta_lactam_keywords = [
        "ampicillin",
        "amox",
        "cef",
        "ceph",
        "imipenem",
        "ticarcillin",
        "piperacillin",
        "penicillin",
    ]
    beta_lactam_abs = [ab for ab in all_antibiotics if any(kw.lower() in ab.lower() for kw in beta_lactam_keywords)]

    logger.info(f"\nValidating across {len(beta_lactam_abs)} beta-lactam antibiotics...")

    all_results = {}

    for drug in sorted(beta_lactam_abs):
        X, y, ids = create_tem_dataset(genotype_df, phenotype_df, drug)

        if len(X) > 10:  # Minimum samples
            results = validate_with_analyzer(X, y, ids)
            all_results[drug] = results
        else:
            logger.warning(f"Skipping {drug}: too few samples ({len(X)})")

    return all_results


def print_summary(all_results: dict) -> None:
    """Print summary of all validation results.

    Args:
        all_results: Results dict for all drugs
    """
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY - Beta-Lactam Resistance Prediction")
    logger.info("=" * 60)

    print("\n| Antibiotic | Samples | Spearman | Accuracy | F1 | ROC AUC |")
    print("|------------|---------|----------|----------|-----|---------|")

    spearmans = []

    for drug, results in sorted(all_results.items()):
        if "error" in results:
            continue

        n = results.get("n_samples", 0)
        spearman = results.get("spearman", 0)
        accuracy = results.get("accuracy", 0)
        f1 = results.get("f1_score", 0)
        roc = results.get("roc_auc", 0)

        spearmans.append(spearman)

        print(f"| {drug[:30]:30} | {n:7} | {spearman:8.3f} | {accuracy:8.3f} | {f1:.3f} | {roc if roc else 'N/A':7} |")

    if spearmans:
        avg_spearman = np.mean(spearmans)
        print(f"\n**Average Spearman**: {avg_spearman:.3f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Fetch and process FriedbergLab E. coli AMR dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download data, don't process",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Only process existing data",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory for raw data (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help=f"Directory for processed data (default: {DEFAULT_PROCESSED_DIR})",
    )
    parser.add_argument(
        "--drug",
        type=str,
        default="ampicillin",
        help="Specific drug to analyze (default: ampicillin)",
    )
    parser.add_argument(
        "--all-drugs",
        action="store_true",
        help="Analyze all beta-lactam antibiotics",
    )

    args = parser.parse_args()

    # Download if needed
    if not args.process_only and not download_all_files(args.data_dir):
        logger.error("Download failed")
        return

    if args.download_only:
        logger.info("Download complete. Use --process-only to analyze.")
        return

    # Load data
    metadata_df, genotype_df, phenotype_df = load_data(args.data_dir)

    # Analyze data structure
    analyze_data(metadata_df, genotype_df, phenotype_df)

    # Validate
    if args.all_drugs:
        all_results = validate_multiple_drugs(genotype_df, phenotype_df)
        print_summary(all_results)

        # Save results
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(args.output_dir / "validation_results.json", "w") as f:
            # Convert numpy types to Python types for JSON
            serializable_results = {}
            for drug, results in all_results.items():
                serializable_results[drug] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in results.items()
                }
            json.dump(serializable_results, f, indent=2)
        logger.info(f"\nResults saved to {args.output_dir / 'validation_results.json'}")
    else:
        X, y, ids = create_tem_dataset(genotype_df, phenotype_df, args.drug)
        if len(X) > 0:
            results = validate_with_analyzer(X, y, ids)


if __name__ == "__main__":
    main()
