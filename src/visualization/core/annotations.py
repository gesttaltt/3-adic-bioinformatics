# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Annotation utilities for scientific figures.

This module provides functions for adding statistical annotations,
significance brackets, and other common figure annotations.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from ..styles.palettes import SEMANTIC

# =============================================================================
# Statistical Annotations
# =============================================================================


def add_significance_bracket(
    ax: Axes,
    x1: float,
    x2: float,
    y: float,
    p_value: float,
    height: float = 0.02,
    line_width: float = 1.0,
    color: str = "black",
    fontsize: int | None = None,
) -> None:
    """Add significance bracket with p-value annotation.

    Args:
        ax: Matplotlib axes
        x1: Left x position
        x2: Right x position
        y: Y position of bracket base
        p_value: P-value for significance stars
        height: Height of bracket in data units
        line_width: Width of bracket lines
        color: Bracket color
        fontsize: Font size for annotation
    """
    # Determine significance stars
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
    else:
        stars = "ns"

    # Draw bracket
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + height, y + height, y],
        color=color,
        linewidth=line_width,
        clip_on=False,
    )

    # Add annotation
    if fontsize is None:
        fontsize = plt.rcParams.get("font.size", 10)

    ax.text(
        (x1 + x2) / 2,
        y + height,
        stars,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color=color,
    )


def add_pvalue_annotation(
    ax: Axes,
    x: float,
    y: float,
    p_value: float,
    prefix: str = "p = ",
    fontsize: int | None = None,
    color: str = "black",
    **kwargs,
) -> None:
    """Add p-value text annotation.

    Args:
        ax: Matplotlib axes
        x: X position
        y: Y position
        p_value: P-value to display
        prefix: Text prefix
        fontsize: Font size
        color: Text color
        **kwargs: Additional text arguments
    """
    if p_value < 0.001:
        text = f"{prefix}< 0.001"
    elif p_value < 0.01:
        text = f"{prefix}{p_value:.3f}"
    else:
        text = f"{prefix}{p_value:.2f}"

    ax.text(x, y, text, fontsize=fontsize, color=color, **kwargs)


def add_correlation_annotation(
    ax: Axes,
    r: float,
    p_value: float | None = None,
    loc: Literal["upper left", "upper right", "lower left", "lower right"] = "upper right",
    fontsize: int | None = None,
    **kwargs,
) -> None:
    """Add correlation coefficient annotation.

    Args:
        ax: Matplotlib axes
        r: Correlation coefficient
        p_value: Optional p-value
        loc: Annotation location
        fontsize: Font size
        **kwargs: Additional text arguments
    """
    if p_value is not None:
        if p_value < 0.001:
            text = f"r = {r:.3f}***"
        elif p_value < 0.01:
            text = f"r = {r:.3f}**"
        elif p_value < 0.05:
            text = f"r = {r:.3f}*"
        else:
            text = f"r = {r:.3f}"
    else:
        text = f"r = {r:.3f}"

    # Determine position
    if "upper" in loc:
        y = 0.95
        va = "top"
    else:
        y = 0.05
        va = "bottom"

    if "right" in loc:
        x = 0.95
        ha = "right"
    else:
        x = 0.05
        ha = "left"

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha=ha,
        va=va,
        **kwargs,
    )


# =============================================================================
# Zone Annotations
# =============================================================================


def add_goldilocks_zones(
    ax: Axes,
    ymin: float | None = None,
    ymax: float | None = None,
    alpha: float = 0.15,
    show_labels: bool = True,
) -> None:
    """Add Goldilocks zone shading (self -> modified-self -> foreign).

    Args:
        ax: Matplotlib axes
        ymin: Minimum y value for shading
        ymax: Maximum y value for shading
        alpha: Fill transparency
        show_labels: Add zone labels to legend
    """
    # Get current y limits if not provided
    if ymin is None or ymax is None:
        current_ymin, current_ymax = ax.get_ylim()
        if ymin is None:
            ymin = current_ymin
        if ymax is None:
            ymax = current_ymax

    # Zone 1: Self (0-15%)
    ax.axvspan(
        0,
        0.15,
        color=SEMANTIC.goldilocks_self,
        alpha=alpha,
        label="Self (no response)" if show_labels else None,
        zorder=0,
    )

    # Zone 2: Modified self / Goldilocks (15-30%)
    ax.axvspan(
        0.15,
        0.30,
        color=SEMANTIC.goldilocks_modified,
        alpha=alpha,
        label="Modified self (autoimmunity)" if show_labels else None,
        zorder=0,
    )

    # Zone 3: Foreign (>30%)
    ax.axvspan(
        0.30,
        1.0,
        color=SEMANTIC.goldilocks_foreign,
        alpha=alpha,
        label="Foreign (cleared)" if show_labels else None,
        zorder=0,
    )


def add_threshold_line(
    ax: Axes,
    value: float,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    label: str | None = None,
    color: str = "red",
    linestyle: str = "--",
    linewidth: float = 1.5,
    alpha: float = 0.8,
    **kwargs,
) -> None:
    """Add threshold line with optional label.

    Args:
        ax: Matplotlib axes
        value: Threshold value
        orientation: Line orientation
        label: Optional label
        color: Line color
        linestyle: Line style
        linewidth: Line width
        alpha: Line transparency
        **kwargs: Additional line arguments
    """
    if orientation == "horizontal":
        ax.axhline(
            y=value,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            **kwargs,
        )
    else:
        ax.axvline(
            x=value,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=label,
            **kwargs,
        )


def add_reference_region(
    ax: Axes,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    color: str = "#CCCCCC",
    alpha: float = 0.3,
    label: str | None = None,
    **kwargs,
) -> None:
    """Add shaded reference region.

    Args:
        ax: Matplotlib axes
        x_range: (xmin, xmax) for vertical band
        y_range: (ymin, ymax) for horizontal band
        color: Fill color
        alpha: Fill transparency
        label: Optional label
        **kwargs: Additional fill arguments
    """
    if x_range is not None:
        ax.axvspan(
            x_range[0],
            x_range[1],
            color=color,
            alpha=alpha,
            label=label,
            zorder=0,
            **kwargs,
        )
    if y_range is not None:
        ax.axhspan(
            y_range[0],
            y_range[1],
            color=color,
            alpha=alpha,
            label=label,
            zorder=0,
            **kwargs,
        )


# =============================================================================
# Legend Helpers
# =============================================================================


def create_legend_handles(
    labels_colors: dict[str, str],
    shape: Literal["patch", "line", "circle"] = "patch",
    **kwargs,
) -> list:
    """Create legend handles from label-color mapping.

    Args:
        labels_colors: Dictionary mapping labels to colors
        shape: Handle shape type
        **kwargs: Additional handle arguments

    Returns:
        List of legend handles
    """
    handles = []

    for label, color in labels_colors.items():
        if shape == "patch":
            handle = mpatches.Patch(color=color, label=label, **kwargs)
        elif shape == "line":
            handle = plt.Line2D(
                [0],
                [0],
                color=color,
                label=label,
                linewidth=2,
                **kwargs,
            )
        elif shape == "circle":
            handle = plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=8,
                label=label,
                **kwargs,
            )
        else:
            handle = mpatches.Patch(color=color, label=label, **kwargs)

        handles.append(handle)

    return handles


def add_category_legend(
    ax: Axes,
    labels_colors: dict[str, str],
    title: str | None = None,
    loc: str = "best",
    **kwargs,
) -> None:
    """Add categorical legend with custom colors.

    Args:
        ax: Matplotlib axes
        labels_colors: Dictionary mapping labels to colors
        title: Legend title
        loc: Legend location
        **kwargs: Additional legend arguments
    """
    handles = create_legend_handles(labels_colors)
    ax.legend(
        handles=handles,
        title=title,
        loc=loc,
        **kwargs,
    )


# =============================================================================
# Arrow Annotations
# =============================================================================


def add_annotation_arrow(
    ax: Axes,
    text: str,
    xy: tuple[float, float],
    xytext: tuple[float, float],
    fontsize: int | None = None,
    arrowprops: dict | None = None,
    **kwargs,
) -> None:
    """Add text annotation with arrow.

    Args:
        ax: Matplotlib axes
        text: Annotation text
        xy: Point to annotate
        xytext: Text position
        fontsize: Font size
        arrowprops: Arrow properties
        **kwargs: Additional annotation arguments
    """
    if arrowprops is None:
        arrowprops = {
            "arrowstyle": "->",
            "color": "black",
            "connectionstyle": "arc3,rad=0.2",
        }

    ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        fontsize=fontsize,
        arrowprops=arrowprops,
        **kwargs,
    )


# =============================================================================
# Scale Bars and Indicators
# =============================================================================


def add_scale_bar(
    ax: Axes,
    length: float,
    label: str,
    loc: Literal["lower right", "lower left", "upper right", "upper left"] = "lower right",
    color: str = "black",
    fontsize: int | None = None,
    pad: float = 0.1,
) -> None:
    """Add scale bar to axes.

    Args:
        ax: Matplotlib axes
        length: Scale bar length in data units
        label: Scale bar label
        loc: Scale bar location
        color: Bar color
        fontsize: Label font size
        pad: Padding from edge
    """
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

    scalebar = AnchoredSizeBar(
        ax.transData,
        length,
        label,
        loc=loc,
        pad=pad,
        color=color,
        frameon=False,
        size_vertical=length * 0.05,
        fontproperties={"size": fontsize or plt.rcParams["font.size"]},
    )

    ax.add_artist(scalebar)


def add_colorbar_annotation(
    cbar,
    positions: Sequence[float],
    labels: Sequence[str],
    fontsize: int | None = None,
) -> None:
    """Add text annotations to colorbar.

    Args:
        cbar: Colorbar object
        positions: Positions for annotations
        labels: Labels for each position
        fontsize: Font size
    """
    for pos, label in zip(positions, labels, strict=False):
        cbar.ax.text(
            1.1,
            pos,
            label,
            transform=cbar.ax.transAxes,
            va="center",
            fontsize=fontsize,
        )
