# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Antibody neutralization visualization functions."""

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


def plot_bnab_sensitivity(
    df: pd.DataFrame,
    top_n: int = 20,
    save_path: str | Path | None = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot bnAb sensitivity signatures.

    Args:
        df: DataFrame with antibody sensitivity data
        top_n: Number of top antibodies to show
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    df_top = df.head(top_n)

    # Left: Effect size (separation between sensitive/resistant)
    if "Antibody" in df_top.columns and "effect_size" in df_top.columns:
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(df_top)))

        axes[0].barh(
            df_top["Antibody"],
            df_top["effect_size"],
            color=colors,
        )

        axes[0].set_xlabel("Effect Size (Cohen's d)")
        axes[0].set_ylabel("Antibody")
        axes[0].set_title("Sensitivity Separation")
        axes[0].invert_yaxis()
        axes[0].axvline(x=0.8, color="green", linestyle="--", alpha=0.5, label="Large effect")
        axes[0].axvline(x=0.5, color="orange", linestyle="--", alpha=0.5, label="Medium effect")
        axes[0].legend(fontsize=8)

    # Right: Sensitive vs Resistant counts
    if all(c in df_top.columns for c in ["Antibody", "n_sensitive", "n_resistant"]):
        x = np.arange(len(df_top))
        width = 0.35

        axes[1].barh(x - width / 2, df_top["n_sensitive"], width, label="Sensitive", color="green", alpha=0.7)
        axes[1].barh(x + width / 2, df_top["n_resistant"], width, label="Resistant", color="red", alpha=0.7)

        axes[1].set_xlabel("Count")
        axes[1].set_yticks(x)
        axes[1].set_yticklabels(df_top["Antibody"])
        axes[1].set_title("Virus Counts")
        axes[1].invert_yaxis()
        axes[1].legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_breadth_potency(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Plot antibody breadth vs potency relationship.

    Args:
        df: DataFrame with breadth and potency data
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    if not all(c in df.columns for c in ["breadth_pct", "potency_score"]):
        ax.text(0.5, 0.5, "Required columns not found", ha="center", va="center", transform=ax.transAxes)
        return fig

    # Color by epitope class if available
    if "epitope_class" in df.columns and HAS_SEABORN:
        classes = df["epitope_class"].unique()
        colors = {c: plt.cm.Set1(i) for i, c in enumerate(classes)}

        for cls in classes:
            subset = df[df["epitope_class"] == cls]
            ax.scatter(
                subset["breadth_pct"],
                subset["potency_score"],
                c=[colors[cls]],
                label=cls,
                alpha=0.7,
                s=100,
            )

        ax.legend(title="Epitope Class", loc="upper left")
    else:
        scatter = ax.scatter(
            df["breadth_pct"],
            df["potency_score"],
            c=df["median_ic50"] if "median_ic50" in df.columns else "blue",
            cmap="viridis_r",
            alpha=0.7,
            s=100,
        )
        if "median_ic50" in df.columns:
            plt.colorbar(scatter, label="Median IC50 (μg/mL)")

    # Add correlation line
    from scipy.stats import spearmanr

    r, p = spearmanr(df["breadth_pct"], df["potency_score"])

    z = np.polyfit(df["breadth_pct"], df["potency_score"], 1)
    poly = np.poly1d(z)
    x_line = np.linspace(df["breadth_pct"].min(), df["breadth_pct"].max(), 100)
    ax.plot(x_line, poly(x_line), "r--", alpha=0.5, label=f"r={r:.2f}, p={p:.2e}")

    # Label top antibodies
    if "Antibody" in df.columns:
        top_breadth = df.nlargest(5, "breadth_pct")
        for _, row in top_breadth.iterrows():
            ax.annotate(
                row["Antibody"],
                (row["breadth_pct"], row["potency_score"]),
                fontsize=8,
                alpha=0.8,
            )

    ax.set_xlabel("Breadth (%)", fontsize=12)
    ax.set_ylabel("Potency Score", fontsize=12)
    ax.set_title("Antibody Breadth vs Potency", fontsize=14)

    # Mark ideal quadrant (high breadth, high potency)
    ax.axhline(y=df["potency_score"].median(), color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=df["breadth_pct"].median(), color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_antibody_clusters(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Plot antibody clusters by epitope class and potency.

    Args:
        df: DataFrame with clustering data
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    if "epitope_class" not in df.columns:
        ax.text(0.5, 0.5, "No epitope class data", ha="center", va="center", transform=ax.transAxes)
        return fig

    # Group by epitope class
    classes = df["epitope_class"].value_counts()

    colors = plt.cm.Set2(np.linspace(0, 1, len(classes)))

    # Create grouped bar chart
    x = np.arange(len(classes))

    ax.bar(x, classes.values, color=colors)

    ax.set_xlabel("Epitope Class", fontsize=12)
    ax.set_ylabel("Number of Antibodies", fontsize=12)
    ax.set_title("Antibody Distribution by Epitope Class", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(classes.index, rotation=45, ha="right")

    # Add counts on bars
    for i, v in enumerate(classes.values):
        ax.text(i, v + 0.5, str(v), ha="center", fontsize=10)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
