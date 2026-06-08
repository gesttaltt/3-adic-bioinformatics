# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Data management commands for Ternary VAE CLI."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Data management commands")
console = Console()


@app.command("status")
def data_status():
    """Show status of all available datasets.

    Example:
        ternary-vae data status
    """
    console.print("[bold blue]Dataset Status[/bold blue]\n")

    project_root = Path(__file__).resolve().parents[2]

    # Define all expected datasets
    datasets = {
        "Research Datasets": {
            "Stanford HIVDB PI": "data/research/datasets/stanford_hivdb_pi.txt",
            "Stanford HIVDB NRTI": "data/research/datasets/stanford_hivdb_nrti.txt",
            "Stanford HIVDB NNRTI": "data/research/datasets/stanford_hivdb_nnrti.txt",
            "Stanford HIVDB INI": "data/research/datasets/stanford_hivdb_ini.txt",
            "CATNAP Assay": "data/research/datasets/catnap_assay.txt",
            "CTL Epitopes": "data/research/datasets/ctl_summary.csv",
        },
        "HuggingFace": {
            "HIV V3 Coreceptor": "data/external/huggingface/HIV_V3_coreceptor",
            "Human-HIV PPI": "data/external/huggingface/human_hiv_ppi",
        },
        "GitHub": {
            "HIV-data": "data/external/github/HIV-data",
            "HIV-1 Paper": "data/external/github/HIV-1_Paper",
        },
        "Zenodo": {
            "cview gp120": "data/external/zenodo/cview_gp120",
        },
        "Kaggle": {
            "HIV-AIDS Dataset": "data/external/kaggle/hiv-aids-dataset",
        },
    }

    for category, items in datasets.items():
        table = Table(title=category)
        table.add_column("Dataset", style="cyan")
        table.add_column("Path", style="white")
        table.add_column("Status", style="green")

        for name, path in items.items():
            full_path = project_root / path
            if full_path.exists():
                if full_path.is_file():
                    size = full_path.stat().st_size
                    status = f"[green]{size // 1024:,} KB[/green]"
                else:
                    n_files = len(list(full_path.rglob("*")))
                    status = f"[green]{n_files} files[/green]"
            else:
                status = "[red]Missing[/red]"

            table.add_row(name, path, status)

        console.print(table)
        console.print()


@app.command("download")
def data_download(
    source: str = typer.Argument(
        ...,
        help="Data source to download from (huggingface, kaggle, zenodo, github)",
    ),
    dataset: str | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help="Specific dataset to download",
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory (defaults to data/external/<source>)",
    ),
):
    """Download datasets from external sources.

    Example:
        ternary-vae data download huggingface
        ternary-vae data download kaggle --dataset hiv-aids-dataset
    """
    console.print(f"[bold blue]Downloading from {source}[/bold blue]")

    project_root = Path(__file__).resolve().parents[2]
    if output_dir is None:
        output_dir = project_root / "data" / "external" / source

    output_dir.mkdir(parents=True, exist_ok=True)

    if source == "huggingface":
        _download_huggingface(dataset, output_dir)
    elif source == "kaggle":
        _download_kaggle(dataset, output_dir)
    elif source == "zenodo":
        _download_zenodo(dataset, output_dir)
    elif source == "github":
        _download_github(dataset, output_dir)
    else:
        console.print(f"[red]Unknown source: {source}[/red]")
        console.print("Valid sources: huggingface, kaggle, zenodo, github")
        raise typer.Exit(1)


def _download_huggingface(dataset: str | None, output_dir: Path):
    """Download from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[red]HuggingFace datasets not installed[/red]")
        console.print("Run: pip install datasets")
        raise typer.Exit(1)

    datasets_to_download = {
        "HIV_V3_coreceptor": "anirudhprabhakaran/HIV_V3_coreceptor",
        "human_hiv_ppi": "anirudhprabhakaran/human_hiv_ppi",
    }

    if dataset:
        if dataset not in datasets_to_download:
            console.print(f"[red]Unknown dataset: {dataset}[/red]")
            console.print(f"Available: {list(datasets_to_download.keys())}")
            raise typer.Exit(1)
        datasets_to_download = {dataset: datasets_to_download[dataset]}

    for name, hf_path in datasets_to_download.items():
        console.print(f"Downloading {name} from {hf_path}...")
        try:
            ds = load_dataset(hf_path)
            save_path = output_dir / name
            save_path.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(save_path))
            console.print(f"[green]Saved to {save_path}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to download {name}: {e}[/red]")


def _download_kaggle(dataset: str | None, output_dir: Path):
    """Download from Kaggle."""
    console.print("[yellow]Kaggle download requires kaggle CLI setup[/yellow]")
    console.print("1. pip install kaggle")
    console.print("2. Set up ~/.kaggle/kaggle.json with your API key")
    console.print("3. Run: kaggle datasets download -d <dataset-name> -p <output-dir>")


def _download_zenodo(dataset: str | None, output_dir: Path):
    """Download from Zenodo."""
    console.print("[yellow]Zenodo download implementation pending[/yellow]")
    console.print("Use src/scripts/hiv/download_hiv_datasets.py for Zenodo data")


def _download_github(dataset: str | None, output_dir: Path):
    """Download from GitHub."""
    repos = {
        "HIV-data": "https://github.com/user/HIV-data.git",
        "HIV-1_Paper": "https://github.com/user/HIV-1_Paper.git",
    }

    console.print("[yellow]GitHub download via git clone[/yellow]")
    for name, url in repos.items():
        console.print(f"  git clone {url} {output_dir / name}")


@app.command("validate")
def data_validate(
    dataset: str = typer.Argument(
        ...,
        help="Dataset to validate (stanford, catnap, ctl, v3, all)",
    ),
):
    """Validate dataset integrity and format.

    Example:
        ternary-vae data validate stanford
        ternary-vae data validate all
    """
    console.print(f"[bold blue]Validating {dataset}[/bold blue]")

    validators = {
        "stanford": _validate_stanford,
        "catnap": _validate_catnap,
        "ctl": _validate_ctl,
        "v3": _validate_v3,
    }

    if dataset == "all":
        for name, validator in validators.items():
            console.print(f"\n[bold]Validating {name}...[/bold]")
            validator()
    elif dataset in validators:
        validators[dataset]()
    else:
        console.print(f"[red]Unknown dataset: {dataset}[/red]")
        console.print(f"Valid options: {list(validators.keys())}, all")
        raise typer.Exit(1)


def _validate_stanford():
    """Validate Stanford HIVDB files."""
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "research" / "datasets"

    files = ["stanford_hivdb_pi.txt", "stanford_hivdb_nrti.txt", "stanford_hivdb_nnrti.txt", "stanford_hivdb_ini.txt"]

    for f in files:
        path = data_dir / f
        if path.exists():
            with open(path) as fp:
                lines = fp.readlines()
            console.print(f"[green]{f}: {len(lines):,} records[/green]")
        else:
            console.print(f"[red]{f}: NOT FOUND[/red]")


def _validate_catnap():
    """Validate CATNAP assay file."""
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "data" / "research" / "datasets" / "catnap_assay.txt"

    if path.exists():
        with open(path) as fp:
            lines = fp.readlines()
        console.print(f"[green]catnap_assay.txt: {len(lines):,} records[/green]")
    else:
        console.print("[red]catnap_assay.txt: NOT FOUND[/red]")


def _validate_ctl():
    """Validate CTL summary file."""
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "data" / "research" / "datasets" / "ctl_summary.csv"

    if path.exists():
        import csv

        with open(path) as fp:
            reader = csv.reader(fp)
            rows = list(reader)
        console.print(f"[green]ctl_summary.csv: {len(rows):,} rows[/green]")
    else:
        console.print("[red]ctl_summary.csv: NOT FOUND[/red]")


def _validate_v3():
    """Validate V3 coreceptor dataset."""
    project_root = Path(__file__).resolve().parents[2]
    path = project_root / "data" / "external" / "huggingface" / "HIV_V3_coreceptor"

    if path.exists():
        n_files = len(list(path.rglob("*")))
        console.print(f"[green]HIV_V3_coreceptor: {n_files} files[/green]")
    else:
        console.print("[red]HIV_V3_coreceptor: NOT FOUND[/red]")


@app.callback()
def callback():
    """Data management commands."""
    pass
