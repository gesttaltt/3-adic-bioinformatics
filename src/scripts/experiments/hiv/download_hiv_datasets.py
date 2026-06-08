#!/usr/bin/env python3
# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
"""
Download HIV Datasets from Various Sources

This script downloads freely available HIV datasets from:
- Kaggle (requires kaggle CLI)
- GitHub repositories
- Hugging Face datasets
- Zenodo
- Direct CSV/JSON downloads

Usage:
    python src/scripts/download_hiv_datasets.py --all
    python src/scripts/download_hiv_datasets.py --github --csv
    python src/scripts/download_hiv_datasets.py --huggingface
"""

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "external"


# =============================================================================
# Dataset Definitions
# =============================================================================

GITHUB_REPOS = [
    {
        "name": "HIV-data",
        "url": "https://github.com/malabz/HIV-data.git",
        "description": "HIV sequence data stored by sequence length",
    },
    {
        "name": "HIV-DRM-machine-learning",
        "url": "https://github.com/lucblassel/HIV-DRM-machine-learning.git",
        "description": "Drug resistance mutation ML data (African & UK datasets)",
    },
    {
        "name": "HIV-1_Paper",
        "url": "https://github.com/pauloluniyi/HIV-1_Paper.git",
        "description": "HIV-1 Drug Resistance and Genetic Diversity in Nigeria",
    },
]

CSV_DOWNLOADS = [
    {
        "name": "corgis_aids.csv",
        "url": "https://corgis-edu.github.io/corgis/datasets/csv/aids/aids.csv",
        "description": "CORGIS AIDS dataset - UNAIDS global statistics",
    },
]

ZENODO_DATASETS = [
    {
        "name": "hiv_genome_to_genome",
        "record_id": "7139",
        "description": "HIV Genome-to-Genome Study supplementary data",
    },
    {
        "name": "cview_gp120_sequences",
        "record_id": "6475667",
        "description": "North American subtype B gp120 sequences (CCR5/CXCR4)",
    },
    {
        "name": "hiv_virion_morphology",
        "record_id": "5149062",
        "description": "TEM images for HIV-1 virion classification",
    },
]

HUGGINGFACE_DATASETS = [
    {
        "name": "human_hiv_ppi",
        "repo": "damlab/human_hiv_ppi",
        "description": "Human-HIV Protein-Protein Interactions (16k+ pairs)",
    },
    {
        "name": "HIV_V3_coreceptor",
        "repo": "damlab/HIV_V3_coreceptor",
        "description": "HIV V3 loop coreceptor usage (LANL derived)",
    },
    {
        "name": "Protease_Hiv_drug",
        "repo": "rebe121314/Protease_Hiv_drug",
        "description": "HIV Protease drug resistance data",
    },
]

KAGGLE_DATASETS = [
    {
        "name": "hiv-aids-dataset",
        "slug": "imdevskp/hiv-aids-dataset",
        "description": "WHO/UNESCO HIV statistics",
    },
    {
        "name": "hiv-1-and-hiv-2-rna-sequences",
        "slug": "protobioengineering/hiv-1-and-hiv-2-rna-sequences",
        "description": "HIV-1 and HIV-2 FASTA/GenBank sequences",
    },
    {
        "name": "hivaids-annual-report",
        "slug": "mostafafaramin/hivaids-annual-report",
        "description": "HIV Surveillance Annual Report",
    },
]


# =============================================================================
# Download Functions
# =============================================================================


def download_file(url: str, dest_path: Path, description: str = "") -> bool:
    """Download a file from URL."""
    try:
        print(f"  Downloading: {description or url}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest_path)
        size_kb = dest_path.stat().st_size / 1024
        print(f"    -> {dest_path.name} ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"    [ERROR] {e}")
        return False


def download_github_repos(output_dir: Path) -> int:
    """Clone GitHub repositories."""
    print("\n" + "=" * 60)
    print("DOWNLOADING GITHUB REPOSITORIES")
    print("=" * 60)

    success_count = 0
    for repo in GITHUB_REPOS:
        repo_dir = output_dir / repo["name"]
        if repo_dir.exists():
            print(f"  [SKIP] {repo['name']} already exists")
            success_count += 1
            continue

        print(f"\n  Cloning: {repo['name']}")
        print(f"    {repo['description']}")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo["url"], str(repo_dir)],
                check=True,
                capture_output=True,
            )
            print("    -> Success")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"    [ERROR] {e}")
        except FileNotFoundError:
            print("    [ERROR] git not found in PATH")

    return success_count


def download_csv_files(output_dir: Path) -> int:
    """Download CSV files from direct URLs."""
    print("\n" + "=" * 60)
    print("DOWNLOADING CSV FILES")
    print("=" * 60)

    success_count = 0
    for dataset in CSV_DOWNLOADS:
        dest_path = output_dir / dataset["name"]
        if dest_path.exists():
            print(f"  [SKIP] {dataset['name']} already exists")
            success_count += 1
            continue

        if download_file(dataset["url"], dest_path, dataset["description"]):
            success_count += 1

    return success_count


def download_zenodo_datasets(output_dir: Path) -> int:
    """Download datasets from Zenodo."""
    print("\n" + "=" * 60)
    print("DOWNLOADING ZENODO DATASETS")
    print("=" * 60)

    success_count = 0
    for dataset in ZENODO_DATASETS:
        dataset_dir = output_dir / dataset["name"]
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            print(f"  [SKIP] {dataset['name']} already exists")
            success_count += 1
            continue

        print(f"\n  Dataset: {dataset['name']}")
        print(f"    {dataset['description']}")

        # Zenodo API to get files
        api_url = f"https://zenodo.org/api/records/{dataset['record_id']}"
        try:
            with urllib.request.urlopen(api_url) as response:
                record = json.loads(response.read().decode())

            dataset_dir.mkdir(parents=True, exist_ok=True)
            files_downloaded = 0

            for file_info in record.get("files", []):
                file_url = file_info["links"]["self"]
                file_name = file_info["key"]
                file_path = dataset_dir / file_name

                if file_path.exists():
                    print(f"    [SKIP] {file_name}")
                    files_downloaded += 1
                    continue

                # Skip very large files (>100MB)
                file_size_mb = file_info.get("size", 0) / (1024 * 1024)
                if file_size_mb > 100:
                    print(f"    [SKIP] {file_name} ({file_size_mb:.1f} MB - too large)")
                    continue

                if download_file(file_url, file_path, file_name):
                    files_downloaded += 1

            if files_downloaded > 0:
                success_count += 1
                print(f"    -> Downloaded {files_downloaded} files")

        except Exception as e:
            print(f"    [ERROR] {e}")

    return success_count


def download_huggingface_datasets(output_dir: Path) -> int:
    """Download datasets from Hugging Face."""
    print("\n" + "=" * 60)
    print("DOWNLOADING HUGGING FACE DATASETS")
    print("=" * 60)

    # Check if huggingface_hub is available
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("  [WARN] huggingface_hub not installed. Run: pip install huggingface_hub")
        print("  Attempting alternative download method...")
        return download_huggingface_alternative(output_dir)

    success_count = 0
    for dataset in HUGGINGFACE_DATASETS:
        dataset_dir = output_dir / dataset["name"]
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            print(f"  [SKIP] {dataset['name']} already exists")
            success_count += 1
            continue

        print(f"\n  Dataset: {dataset['name']}")
        print(f"    {dataset['description']}")

        try:
            snapshot_download(
                repo_id=dataset["repo"],
                repo_type="dataset",
                local_dir=str(dataset_dir),
                local_dir_use_symlinks=False,
            )
            print("    -> Success")
            success_count += 1
        except Exception as e:
            print(f"    [ERROR] {e}")

    return success_count


def download_huggingface_alternative(output_dir: Path) -> int:
    """Alternative HuggingFace download using git."""
    success_count = 0
    for dataset in HUGGINGFACE_DATASETS:
        dataset_dir = output_dir / dataset["name"]
        if dataset_dir.exists():
            print(f"  [SKIP] {dataset['name']} already exists")
            success_count += 1
            continue

        git_url = f"https://huggingface.co/datasets/{dataset['repo']}"
        print(f"\n  Cloning: {dataset['name']}")

        try:
            # Try git lfs clone
            subprocess.run(
                ["git", "clone", "--depth", "1", git_url, str(dataset_dir)],
                check=True,
                capture_output=True,
            )
            print("    -> Success")
            success_count += 1
        except Exception as e:
            print(f"    [ERROR] {e}")

    return success_count


def download_kaggle_datasets(output_dir: Path) -> int:
    """Download datasets from Kaggle."""
    print("\n" + "=" * 60)
    print("DOWNLOADING KAGGLE DATASETS")
    print("=" * 60)

    # Check if kaggle CLI is available
    try:
        result = subprocess.run(
            ["kaggle", "--version"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise FileNotFoundError
    except FileNotFoundError:
        print("  [WARN] Kaggle CLI not found or not configured.")
        print("  To install: pip install kaggle")
        print("  Then configure: https://www.kaggle.com/docs/api")
        print("\n  Manual download URLs:")
        for ds in KAGGLE_DATASETS:
            print(f"    - https://www.kaggle.com/datasets/{ds['slug']}")
        return 0

    success_count = 0
    for dataset in KAGGLE_DATASETS:
        dataset_dir = output_dir / dataset["name"]
        if dataset_dir.exists() and any(dataset_dir.iterdir()):
            print(f"  [SKIP] {dataset['name']} already exists")
            success_count += 1
            continue

        print(f"\n  Dataset: {dataset['name']}")
        print(f"    {dataset['description']}")

        try:
            dataset_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", dataset["slug"], "-p", str(dataset_dir), "--unzip"],
                check=True,
                capture_output=True,
            )
            print("    -> Success")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"    [ERROR] {e}")

    return success_count


def create_dataset_index(data_dir: Path):
    """Create an index of downloaded datasets."""
    print("\n" + "=" * 60)
    print("CREATING DATASET INDEX")
    print("=" * 60)

    index = {
        "description": "HIV External Datasets Index",
        "sources": {},
    }

    for source_dir in data_dir.iterdir():
        if not source_dir.is_dir():
            continue

        source_name = source_dir.name
        datasets = []

        for item in source_dir.iterdir():
            if item.is_dir():
                file_count = sum(1 for _ in item.rglob("*") if _.is_file())
                total_size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                datasets.append(
                    {
                        "name": item.name,
                        "type": "directory",
                        "file_count": file_count,
                        "size_mb": round(total_size / (1024 * 1024), 2),
                    }
                )
            else:
                datasets.append(
                    {
                        "name": item.name,
                        "type": "file",
                        "size_mb": round(item.stat().st_size / (1024 * 1024), 2),
                    }
                )

        if datasets:
            index["sources"][source_name] = datasets

    index_path = data_dir / "dataset_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"  Index saved to: {index_path}")

    # Print summary
    print("\n  Downloaded datasets:")
    for source, datasets in index["sources"].items():
        print(f"\n  [{source}]")
        for ds in datasets:
            if ds["type"] == "directory":
                print(f"    - {ds['name']}: {ds['file_count']} files ({ds['size_mb']} MB)")
            else:
                print(f"    - {ds['name']}: {ds['size_mb']} MB")


def main():
    parser = argparse.ArgumentParser(description="Download HIV Datasets")
    parser.add_argument("--all", action="store_true", help="Download from all sources")
    parser.add_argument("--github", action="store_true", help="Download GitHub repos")
    parser.add_argument("--csv", action="store_true", help="Download CSV files")
    parser.add_argument("--zenodo", action="store_true", help="Download Zenodo datasets")
    parser.add_argument("--huggingface", action="store_true", help="Download HuggingFace datasets")
    parser.add_argument("--kaggle", action="store_true", help="Download Kaggle datasets")
    args = parser.parse_args()

    # If no specific source, show help
    if not any([args.all, args.github, args.csv, args.zenodo, args.huggingface, args.kaggle]):
        parser.print_help()
        print("\nExample: python src/scripts/download_hiv_datasets.py --all")
        return

    print("=" * 60)
    print("HIV DATASET DOWNLOADER")
    print("=" * 60)
    print(f"Output directory: {DATA_DIR}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total_success = 0

    if args.all or args.github:
        total_success += download_github_repos(DATA_DIR / "github")

    if args.all or args.csv:
        total_success += download_csv_files(DATA_DIR / "csv")

    if args.all or args.zenodo:
        total_success += download_zenodo_datasets(DATA_DIR / "zenodo")

    if args.all or args.huggingface:
        total_success += download_huggingface_datasets(DATA_DIR / "huggingface")

    if args.all or args.kaggle:
        total_success += download_kaggle_datasets(DATA_DIR / "kaggle")

    # Create index
    create_dataset_index(DATA_DIR)

    print("\n" + "=" * 60)
    print(f"DOWNLOAD COMPLETE: {total_success} datasets downloaded")
    print("=" * 60)


if __name__ == "__main__":
    main()
