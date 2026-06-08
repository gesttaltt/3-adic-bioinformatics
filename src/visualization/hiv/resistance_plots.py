# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Drug resistance visualization functions."""

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
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for visualization")


def plot_resistance_correlation(
    df: pd.DataFrame,
    distance_col: str = "hyperbolic_distance",
    resistance_col: str = "log_fold_change",
    hue_col: str | None = "drug_class",
    save_path: str | Path | None = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Plot correlation between hyperbolic distance and resistance.

    Args:
        df: DataFrame with distance and resistance columns
        distance_col: Column name for hyperbolic distance
        resistance_col: Column name for resistance (log fold-change)
        hue_col: Column for color grouping
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    if HAS_SEABORN and hue_col and hue_col in df.columns:
        sns.scatterplot(
            data=df,
            x=distance_col,
            y=resistance_col,
            hue=hue_col,
            alpha=0.6,
            ax=ax,
        )
    else:
        ax.scatter(
            df[distance_col],
            df[resistance_col],
            alpha=0.6,
            s=30,
        )

    # Add regression line
    valid = df.dropna(subset=[distance_col, resistance_col])
    if len(valid) > 10:
        z = np.polyfit(valid[distance_col], valid[resistance_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid[distance_col].min(), valid[distance_col].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label="Linear fit")

        # Calculate correlation
        from scipy.stats import spearmanr

        r, pval = spearmanr(valid[distance_col], valid[resistance_col])
        ax.text(
            0.05,
            0.95,
            f"Spearman r = {r:.3f}\np = {pval:.2e}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel("Hyperbolic Distance", fontsize=12)
    ax.set_ylabel("Log₁₀(Fold Change)", fontsize=12)
    ax.set_title("Drug Resistance vs Hyperbolic Distance", fontsize=14)
    ax.legend(loc="lower right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_mutation_classification(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Plot primary vs accessory mutation classification.

    Args:
        df: DataFrame with mutation data and classification
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Radial distribution by classification
    if "is_primary" in df.columns and "radial_position" in df.columns:
        primary = df[df["is_primary"]]["radial_position"].dropna()
        accessory = df[~df["is_primary"]]["radial_position"].dropna()

        axes[0].hist(primary, bins=30, alpha=0.7, label=f"Primary (n={len(primary)})")
        axes[0].hist(accessory, bins=30, alpha=0.7, label=f"Accessory (n={len(accessory)})")
        axes[0].set_xlabel("Radial Position")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Radial Distribution by Mutation Type")
        axes[0].legend()

    # Right: Fold-change distribution
    if "is_primary" in df.columns and "fold_change" in df.columns:
        primary_fc = df[df["is_primary"]]["fold_change"].dropna()
        accessory_fc = df[~df["is_primary"]]["fold_change"].dropna()

        data = [np.log10(primary_fc + 0.01), np.log10(accessory_fc + 0.01)]
        labels = ["Primary", "Accessory"]

        bp = axes[1].boxplot(data, labels=labels, patch_artist=True)
        colors = ["#ff7f0e", "#1f77b4"]
        for patch, color in zip(bp["boxes"], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        axes[1].set_ylabel("Log₁₀(Fold Change)")
        axes[1].set_title("Resistance by Mutation Type")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_cross_resistance_heatmap(
    df: pd.DataFrame,
    drugs: list[str] | None = None,
    save_path: str | Path | None = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Plot cross-resistance heatmap between drugs.

    Args:
        df: DataFrame with mutation and drug resistance data
        drugs: List of drugs to include (default: infer from data)
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    if not HAS_SEABORN:
        raise ImportError("seaborn required for heatmap")

    # Build cross-resistance matrix
    if drugs is None:
        drug_cols = [
            c
            for c in df.columns
            if c
            not in [
                "position",
                "wild_type",
                "mutant",
                "drug",
                "fold_change",
                "hyperbolic_distance",
                "log_fold_change",
                "drug_class",
                "is_primary",
                "radial_position",
            ]
        ]
        drugs = drug_cols[:15]  # Limit to 15 drugs

    if not drugs:
        raise ValueError("No drug columns found in data")

    # Create correlation matrix
    drug_data = df[drugs].dropna(how="all")
    corr_matrix = drug_data.corr()

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        center=0,
        square=True,
        ax=ax,
        cbar_kws={"label": "Correlation"},
    )

    ax.set_title("Cross-Resistance Correlation Matrix", fontsize=14)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_drug_class_embeddings(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Plot drug class-specific geometric signatures.

    Args:
        df: DataFrame with drug class and embedding information
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    drug_classes = df["drug_class"].unique() if "drug_class" in df.columns else []

    if len(drug_classes) == 0:
        ax.text(0.5, 0.5, "No drug class data available", ha="center", va="center", transform=ax.transAxes)
        return fig

    colors = plt.cm.Set2(np.linspace(0, 1, len(drug_classes)))

    for i, dc in enumerate(drug_classes):
        subset = df[df["drug_class"] == dc]

        if "hyperbolic_distance" in subset.columns and "radial_position" in subset.columns:
            ax.scatter(
                subset["hyperbolic_distance"],
                subset["radial_position"],
                c=[colors[i]],
                label=dc,
                alpha=0.6,
                s=50,
            )

    ax.set_xlabel("Hyperbolic Distance", fontsize=12)
    ax.set_ylabel("Radial Position", fontsize=12)
    ax.set_title("Drug Class Geometric Signatures", fontsize=14)
    ax.legend(title="Drug Class")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
