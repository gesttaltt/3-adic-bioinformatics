# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.

"""Cross-dataset integration visualization functions."""

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


def plot_constraint_landscape(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """Plot multi-pressure constraint landscape.

    Args:
        df: DataFrame with position and constraint data
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    required_cols = ["resistance_pressure", "immune_pressure"]
    if not all(c in df.columns for c in required_cols):
        ax.text(0.5, 0.5, "Required columns not found", ha="center", va="center", transform=ax.transAxes)
        return fig

    # Size by total constraint
    sizes = df["total_constraint"] * 50 if "total_constraint" in df.columns else 100

    # Color by tradeoff status
    if "is_tradeoff" in df.columns:
        colors = df["is_tradeoff"].map({True: "red", False: "blue"})
        ax.scatter(
            df["resistance_pressure"],
            df["immune_pressure"],
            c=colors,
            s=sizes,
            alpha=0.6,
        )

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Trade-off position"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=10, label="Single pressure"),
        ]
        ax.legend(handles=legend_elements, loc="upper right")
    else:
        ax.scatter(
            df["resistance_pressure"],
            df["immune_pressure"],
            c=sizes,
            cmap="viridis",
            s=sizes,
            alpha=0.6,
        )

    # Label high-constraint positions
    if "hxb2_position" in df.columns:
        high_constraint = df.nlargest(10, "total_constraint") if "total_constraint" in df.columns else df.head(10)
        for _, row in high_constraint.iterrows():
            ax.annotate(
                f"HXB2:{int(row['hxb2_position'])}",
                (row["resistance_pressure"], row["immune_pressure"]),
                fontsize=8,
                alpha=0.8,
            )

    ax.set_xlabel("Drug Resistance Pressure", fontsize=12)
    ax.set_ylabel("Immune Selection Pressure", fontsize=12)
    ax.set_title("Multi-Pressure Constraint Landscape", fontsize=14)

    # Add quadrant labels
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=1, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_vaccine_targets(
    df: pd.DataFrame,
    top_n: int = 20,
    save_path: str | Path | None = None,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Plot ranked vaccine target positions.

    Args:
        df: DataFrame with vaccine target rankings
        top_n: Number of top targets to show
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    df_top = df.head(top_n)

    # Left: Composite score bar chart
    if "vaccine_score" in df_top.columns:
        colors = plt.cm.RdYlGn(df_top["vaccine_score"] / df_top["vaccine_score"].max())

        if "Epitope" in df_top.columns:
            labels = [e[:15] + "..." if len(str(e)) > 15 else str(e) for e in df_top["Epitope"]]
        elif "hxb2_position" in df_top.columns:
            labels = [f"HXB2:{int(p)}" for p in df_top["hxb2_position"]]
        else:
            labels = [f"Target {i + 1}" for i in range(len(df_top))]

        axes[0].barh(labels, df_top["vaccine_score"], color=colors)
        axes[0].set_xlabel("Vaccine Score")
        axes[0].set_title("Top Vaccine Targets")
        axes[0].invert_yaxis()

    # Right: Component breakdown (stacked bar)
    component_cols = ["conservation_score", "hla_coverage", "escape_resistance"]
    available_cols = [c for c in component_cols if c in df_top.columns]

    if available_cols:
        x = np.arange(len(df_top))
        bottom = np.zeros(len(df_top))

        color_map = {
            "conservation_score": "#2ecc71",
            "hla_coverage": "#3498db",
            "escape_resistance": "#e74c3c",
        }
        labels_map = {
            "conservation_score": "Conservation",
            "hla_coverage": "HLA Coverage",
            "escape_resistance": "Escape Resist.",
        }

        for col in available_cols:
            values = df_top[col].fillna(0).values
            axes[1].bar(
                x,
                values,
                bottom=bottom,
                label=labels_map.get(col, col),
                color=color_map.get(col, "gray"),
                alpha=0.8,
            )
            bottom += values

        axes[1].set_xlabel("Target Rank")
        axes[1].set_ylabel("Score Components")
        axes[1].set_title("Score Breakdown")
        axes[1].set_xticks(x[::2])
        axes[1].set_xticklabels([str(i + 1) for i in x[::2]])
        axes[1].legend(loc="upper right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_tradeoff_map(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Plot resistance-immunity trade-off map.

    Args:
        df: DataFrame with trade-off data
        save_path: Optional path to save figure
        figsize: Figure size

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    if not HAS_SEABORN:
        raise ImportError("seaborn required for this plot")

    fig, ax = plt.subplots(figsize=figsize)

    # Create a grid for the trade-off visualization
    if "resistance_pressure" in df.columns and "immune_pressure" in df.columns:
        # Pivot to create heatmap data
        pivot_data = df.pivot_table(
            values="total_constraint" if "total_constraint" in df.columns else "resistance_pressure",
            index=pd.cut(df["immune_pressure"], bins=10),
            columns=pd.cut(df["resistance_pressure"], bins=10),
            aggfunc="count",
        ).fillna(0)

        sns.heatmap(
            pivot_data,
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": "Position Count"},
        )

        ax.set_xlabel("Drug Resistance Pressure (binned)")
        ax.set_ylabel("Immune Selection Pressure (binned)")
        ax.set_title("Trade-off Density Map")

        # Rotate x labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    else:
        ax.text(0.5, 0.5, "Required columns not found", ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
