# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Analysis commands for Ternary VAE CLI."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.config.paths import DATA_DIR, EXTERNAL_DATA_DIR, RESULTS_DIR

app = typer.Typer(help="Analysis commands for HIV and bioinformatics data")
console = Console()


@app.command("stanford")
def analyze_stanford(
    drug_class: str | None = typer.Option(
        None,
        "--drug-class",
        "-d",
        help="Drug class to analyze (PI, NRTI, NNRTI, INI, or 'all')",
    ),
    output_dir: Path = typer.Option(
        RESULTS_DIR / "stanford_resistance",
        "--output",
        "-o",
        help="Output directory for results",
    ),
):
    """Analyze Stanford HIVDB drug resistance data.

    Example:
        ternary-vae analyze stanford --drug-class PI
        ternary-vae analyze stanford --drug-class all
    """
    console.print("[bold blue]Stanford HIVDB Drug Resistance Analysis[/bold blue]")

    # Check data availability
    data_dir = DATA_DIR / "research" / "datasets"
    drug_classes = {
        "PI": "stanford_hivdb_pi.txt",
        "NRTI": "stanford_hivdb_nrti.txt",
        "NNRTI": "stanford_hivdb_nnrti.txt",
        "INI": "stanford_hivdb_ini.txt",
    }

    if drug_class and drug_class.upper() != "ALL":
        classes_to_analyze = [drug_class.upper()]
    else:
        classes_to_analyze = list(drug_classes.keys())

    table = Table(title="Dataset Status")
    table.add_column("Drug Class", style="cyan")
    table.add_column("File", style="white")
    table.add_column("Status", style="green")

    for dc in classes_to_analyze:
        file_path = data_dir / drug_classes.get(dc, "")
        status = "[green]Found[/green]" if file_path.exists() else "[red]Missing[/red]"
        table.add_row(dc, drug_classes.get(dc, "N/A"), status)

    console.print(table)

    # Run analysis
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print("\n[bold]Running analysis...[/bold]")

    try:
        from src.biology.codons import codon_to_index
        from src.data.hiv import load_stanford_hivdb

        for dc in classes_to_analyze:
            file_path = data_dir / drug_classes[dc]
            if not file_path.exists():
                console.print(f"[yellow]Skipping {dc}: file not found[/yellow]")
                continue

            console.print(f"Analyzing {dc}...")
            df = load_stanford_hivdb(dc.lower())
            console.print(f"  Loaded {len(df)} records")

            # Save summary
            summary_path = output_dir / f"{dc.lower()}_summary.csv"
            df.describe().to_csv(summary_path)
            console.print(f"  Summary saved to {summary_path}")

        console.print("[bold green]Analysis complete![/bold green]")

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("Run: pip install -e .[bio]")


@app.command("catnap")
def analyze_catnap(
    antibody: str | None = typer.Option(
        None,
        "--antibody",
        "-a",
        help="Specific antibody to analyze (e.g., VRC01)",
    ),
    output_dir: Path = typer.Option(
        RESULTS_DIR / "catnap_neutralization",
        "--output",
        "-o",
        help="Output directory for results",
    ),
):
    """Analyze CATNAP neutralization assay data.

    Example:
        ternary-vae analyze catnap
        ternary-vae analyze catnap --antibody VRC01
    """
    console.print("[bold blue]CATNAP Neutralization Analysis[/bold blue]")

    data_file = DATA_DIR / "research" / "datasets" / "catnap_assay.txt"
    if not data_file.exists():
        console.print(f"[red]Data file not found: {data_file}[/red]")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from src.data.hiv import load_catnap

        console.print("Loading CATNAP data...")
        df = load_catnap()
        console.print(f"[green]Loaded {len(df):,} neutralization records[/green]")

        # Show antibody summary
        if antibody:
            df = df[df["Antibody"] == antibody]
            console.print(f"Filtered to {len(df):,} records for {antibody}")

        # Basic statistics
        table = Table(title="Neutralization Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Records", f"{len(df):,}")
        table.add_row("Unique Antibodies", str(df["Antibody"].nunique()))
        table.add_row("Unique Viruses", str(df["Virus"].nunique()))

        console.print(table)
        console.print("[bold green]Analysis complete![/bold green]")

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")


@app.command("ctl")
def analyze_ctl(
    protein: str | None = typer.Option(
        None,
        "--protein",
        "-p",
        help="Protein to analyze (Gag, Pol, Env, Nef, etc.)",
    ),
    output_dir: Path = typer.Option(
        RESULTS_DIR / "ctl_escape",
        "--output",
        "-o",
        help="Output directory for results",
    ),
):
    """Analyze CTL epitope escape data.

    Example:
        ternary-vae analyze ctl
        ternary-vae analyze ctl --protein Gag
    """
    console.print("[bold blue]CTL Epitope Escape Analysis[/bold blue]")

    data_file = DATA_DIR / "research" / "datasets" / "ctl_summary.csv"
    if not data_file.exists():
        console.print(f"[red]Data file not found: {data_file}[/red]")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from src.data.hiv import load_lanl_ctl

        console.print("Loading CTL epitope data...")
        df = load_lanl_ctl()
        console.print(f"[green]Loaded {len(df):,} epitopes[/green]")

        if protein:
            df = df[df["Protein"] == protein]
            console.print(f"Filtered to {len(df):,} epitopes for {protein}")

        # Show protein distribution
        table = Table(title="Epitopes by Protein")
        table.add_column("Protein", style="cyan")
        table.add_column("Count", style="green")

        for prot, count in df["Protein"].value_counts().head(10).items():
            table.add_row(prot, str(count))

        console.print(table)
        console.print("[bold green]Analysis complete![/bold green]")

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")


@app.command("tropism")
def analyze_tropism(
    output_dir: Path = typer.Option(
        RESULTS_DIR / "tropism",
        "--output",
        "-o",
        help="Output directory for results",
    ),
):
    """Analyze V3 coreceptor tropism data.

    Example:
        ternary-vae analyze tropism
    """
    console.print("[bold blue]V3 Coreceptor Tropism Analysis[/bold blue]")

    data_dir = EXTERNAL_DATA_DIR / "huggingface" / "HIV_V3_coreceptor"
    if not data_dir.exists():
        console.print(f"[red]Data directory not found: {data_dir}[/red]")
        console.print("Run: ternary-vae data download --source huggingface")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from src.data.hiv import load_v3_coreceptor

        console.print("Loading V3 coreceptor data...")
        df = load_v3_coreceptor()
        console.print(f"[green]Loaded {len(df):,} sequences[/green]")

        # Show tropism distribution
        table = Table(title="Tropism Distribution")
        table.add_column("Coreceptor", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")

        total = len(df)
        for tropism, count in df["tropism"].value_counts().items():
            pct = f"{100 * count / total:.1f}%"
            table.add_row(tropism, str(count), pct)

        console.print(table)
        console.print("[bold green]Analysis complete![/bold green]")

    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")


@app.command("all")
def analyze_all(
    output_dir: Path = typer.Option(
        RESULTS_DIR / "comprehensive_analysis",
        "--output",
        "-o",
        help="Output directory for results",
    ),
):
    """Run comprehensive analysis across all datasets.

    Example:
        ternary-vae analyze all
    """
    console.print("[bold blue]Comprehensive HIV Dataset Analysis[/bold blue]")
    console.print("This will run all available analyses...")

    import subprocess
    import sys

    # Run the comprehensive analysis script
    cmd = [
        sys.executable,
        "src/scripts/analyze_all_datasets.py",
        "--output_dir",
        str(output_dir),
    ]

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode == 0:
        console.print("[bold green]Comprehensive analysis complete![/bold green]")
    else:
        console.print("[red]Analysis failed[/red]")
        raise typer.Exit(1)


@app.callback()
def callback():
    """Analysis commands for HIV and bioinformatics data."""
    pass
