# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""CTL escape visualization functions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import seaborn  # noqa: F401

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for visualization")


def plot_hla_escape_landscape(
    df: pd.DataFrame,
    top_n: int = 15,
    save_path: str | Path | None = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot HLA-stratified escape landscapes.

    Args:
        df: DataFrame with HLA and escape metrics
        top_n: Number of top HLA types to show
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    df_top = df.head(top_n)

    # Left: Mean radial position by HLA
    if "HLA" in df_top.columns and "mean_radial" in df_top.columns:
        colors = plt.cm.viridis(np.linspace(0, 1, len(df_top)))

        axes[0].barh(
            df_top["HLA"],
            df_top["mean_radial"],
            color=colors,
        )

        axes[0].set_xlabel("Mean Radial Position")
        axes[0].set_ylabel("HLA Type")
        axes[0].set_title("HLA Escape Landscape")
        axes[0].invert_yaxis()

    # Right: Conservation score by HLA
    if "HLA" in df_top.columns and "conservation_score" in df_top.columns:
        colors = plt.cm.RdYlGn(df_top["conservation_score"])

        axes[1].barh(
            df_top["HLA"],
            df_top["conservation_score"],
            color=colors,
        )

        axes[1].set_xlabel("Conservation Score")
        axes[1].set_title("Epitope Conservation by HLA")
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, 1)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_protein_escape_velocity(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot escape velocity by HIV protein.

    Args:
        df: DataFrame with protein and escape velocity data
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    if "protein" not in df.columns:
        ax.text(0.5, 0.5, "No protein data available", ha="center", va="center", transform=ax.transAxes)
        return fig

    # Sort by mean radial position (escape velocity proxy)
    df_sorted = df.sort_values("mean_radial_position", ascending=True)

    # Create bar plot
    colors = plt.cm.plasma(np.linspace(0, 1, len(df_sorted)))

    x = np.arange(len(df_sorted))
    width = 0.35

    if "mean_radial_position" in df_sorted.columns:
        ax.bar(
            x - width / 2,
            df_sorted["mean_radial_position"],
            width,
            label="Mean Radial",
            color=colors,
            alpha=0.8,
        )

    if "radial_variance" in df_sorted.columns:
        ax.bar(
            x + width / 2,
            df_sorted["radial_variance"],
            width,
            label="Variance",
            color=colors,
            alpha=0.5,
        )

    ax.set_xlabel("Protein")
    ax.set_ylabel("Escape Velocity Metrics")
    ax.set_title("Escape Velocity by HIV Protein")
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["protein"], rotation=45, ha="right")
    ax.legend()

    # Add epitope counts as text
    if "n_epitopes" in df_sorted.columns:
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            ax.text(i, row["mean_radial_position"] + 0.02, f"n={row['n_epitopes']}", ha="center", fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_epitope_conservation(
    df: pd.DataFrame,
    top_n: int = 20,
    save_path: str | Path | None = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Plot epitope conservation vs radial position.

    Args:
        df: DataFrame with epitope conservation data
        top_n: Number of top epitopes to label
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    if "conservation_score" not in df.columns or "mean_radial" not in df.columns:
        ax.text(0.5, 0.5, "Required columns not found", ha="center", va="center", transform=ax.transAxes)
        return fig

    # Color by protein if available
    if "Protein" in df.columns and HAS_SEABORN:
        proteins = df["Protein"].unique()
        colors = {p: plt.cm.tab10(i) for i, p in enumerate(proteins)}

        for protein in proteins:
            subset = df[df["Protein"] == protein]
            ax.scatter(
                subset["mean_radial"],
                subset["conservation_score"],
                c=[colors[protein]],
                label=protein,
                alpha=0.6,
                s=50,
            )
    else:
        ax.scatter(
            df["mean_radial"],
            df["conservation_score"],
            alpha=0.6,
            s=50,
            c=df["conservation_score"],
            cmap="RdYlGn",
        )

    # Label top conserved epitopes
    top_conserved = df.nlargest(top_n, "conservation_score")
    for _, row in top_conserved.iterrows():
        epitope_label = row["Epitope"][:8] + "..." if len(str(row["Epitope"])) > 8 else row["Epitope"]
        ax.annotate(
            epitope_label,
            (row["mean_radial"], row["conservation_score"]),
            fontsize=7,
            alpha=0.8,
        )

    ax.set_xlabel("Mean Radial Position", fontsize=12)
    ax.set_ylabel("Conservation Score", fontsize=12)
    ax.set_title("Epitope Conservation vs Geometric Position", fontsize=14)

    if "Protein" in df.columns:
        ax.legend(title="Protein", loc="lower right")

    # Add quadrant labels
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)

    ax.text(0.25, 0.95, "Central\n(Conserved)", transform=ax.transAxes, ha="center", fontsize=9, alpha=0.7)
    ax.text(0.75, 0.95, "Peripheral\n(Variable)", transform=ax.transAxes, ha="center", fontsize=9, alpha=0.7)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
