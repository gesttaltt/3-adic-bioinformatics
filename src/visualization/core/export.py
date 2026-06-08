# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Figure export utilities for multiple formats.

This module provides functions for exporting figures to various formats
with consistent quality settings for publication and presentation.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..config import DPI_PRESENTATION, DPI_PUBLICATION, ExportFormat, get_config

# =============================================================================
# Export Functions
# =============================================================================


def save_figure(
    fig: Figure,
    name: str,
    output_dir: Path | str | None = None,
    formats: Sequence[str | ExportFormat] = ("png", "svg"),
    dpi: int | None = None,
    transparent: bool = False,
    close: bool = True,
    **kwargs,
) -> list[Path]:
    """Save figure in multiple formats.

    Args:
        fig: Matplotlib figure to save
        name: Base filename (without extension)
        output_dir: Output directory. If None, uses config default.
        formats: Sequence of export formats ('png', 'svg', 'pdf', 'eps')
        dpi: Resolution for raster formats. If None, uses format-appropriate default.
        transparent: Save with transparent background
        close: Close figure after saving
        **kwargs: Additional arguments passed to fig.savefig()

    Returns:
        List of saved file paths
    """
    config = get_config()

    output_dir = config.output_dir if output_dir is None else Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for fmt in formats:
        fmt_str = fmt.value if isinstance(fmt, ExportFormat) else fmt

        filepath = output_dir / f"{name}.{fmt_str}"

        # Determine DPI based on format
        save_dpi = dpi
        if save_dpi is None:
            if fmt_str in ("png", "jpg", "jpeg"):
                save_dpi = DPI_PUBLICATION
            else:
                save_dpi = None  # Vector formats don't need DPI

        # Build savefig kwargs
        save_kwargs = {
            "format": fmt_str,
            "bbox_inches": "tight",
            "pad_inches": 0.1,
            "transparent": transparent,
            **kwargs,
        }

        if save_dpi is not None:
            save_kwargs["dpi"] = save_dpi

        # Handle format-specific settings
        if fmt_str == "pdf":
            save_kwargs.setdefault("backend", "pdf")
        elif fmt_str == "svg":
            save_kwargs.setdefault("metadata", {"Creator": "Ternary VAE Visualization"})

        fig.savefig(filepath, **save_kwargs)
        saved_paths.append(filepath)

    if close:
        plt.close(fig)

    return saved_paths


def save_publication_figure(
    fig: Figure,
    name: str,
    output_dir: Path | str | None = None,
    close: bool = True,
    **kwargs,
) -> list[Path]:
    """Save figure in publication-quality formats (PNG, SVG, PDF).

    Args:
        fig: Matplotlib figure to save
        name: Base filename
        output_dir: Output directory
        close: Close figure after saving
        **kwargs: Additional arguments

    Returns:
        List of saved file paths
    """
    return save_figure(
        fig=fig,
        name=name,
        output_dir=output_dir,
        formats=("png", "svg", "pdf"),
        dpi=DPI_PUBLICATION,
        close=close,
        **kwargs,
    )


def save_presentation_figure(
    fig: Figure,
    name: str,
    output_dir: Path | str | None = None,
    close: bool = True,
    **kwargs,
) -> list[Path]:
    """Save figure in presentation-quality formats (PNG, SVG).

    Args:
        fig: Matplotlib figure to save
        name: Base filename
        output_dir: Output directory
        close: Close figure after saving
        **kwargs: Additional arguments

    Returns:
        List of saved file paths
    """
    return save_figure(
        fig=fig,
        name=name,
        output_dir=output_dir,
        formats=("png", "svg"),
        dpi=DPI_PRESENTATION,
        close=close,
        **kwargs,
    )


def save_web_figure(
    fig: Figure,
    name: str,
    output_dir: Path | str | None = None,
    close: bool = True,
    **kwargs,
) -> list[Path]:
    """Save figure in web-optimized format (PNG only).

    Args:
        fig: Matplotlib figure to save
        name: Base filename
        output_dir: Output directory
        close: Close figure after saving
        **kwargs: Additional arguments

    Returns:
        List of saved file paths
    """
    return save_figure(
        fig=fig,
        name=name,
        output_dir=output_dir,
        formats=("png",),
        dpi=150,
        close=close,
        **kwargs,
    )


# =============================================================================
# Batch Export
# =============================================================================


def save_figure_batch(
    figures: dict[str, Figure],
    output_dir: Path | str | None = None,
    formats: Sequence[str | ExportFormat] = ("png", "svg"),
    dpi: int | None = None,
    close: bool = True,
    **kwargs,
) -> dict[str, list[Path]]:
    """Save multiple figures in batch.

    Args:
        figures: Dictionary mapping names to figures
        output_dir: Output directory
        formats: Export formats
        dpi: Resolution for raster formats
        close: Close figures after saving
        **kwargs: Additional arguments

    Returns:
        Dictionary mapping names to lists of saved paths
    """
    results = {}

    for name, fig in figures.items():
        paths = save_figure(
            fig=fig,
            name=name,
            output_dir=output_dir,
            formats=formats,
            dpi=dpi,
            close=close,
            **kwargs,
        )
        results[name] = paths

    return results


# =============================================================================
# Export Utilities
# =============================================================================


def get_figure_size_inches(fig: Figure) -> tuple[float, float]:
    """Get figure size in inches.

    Args:
        fig: Matplotlib figure

    Returns:
        (width, height) tuple in inches
    """
    return tuple(fig.get_size_inches())


def set_figure_size_inches(
    fig: Figure,
    width: float,
    height: float,
) -> None:
    """Set figure size in inches.

    Args:
        fig: Matplotlib figure
        width: Width in inches
        height: Height in inches
    """
    fig.set_size_inches(width, height)


def figure_to_array(fig: Figure, dpi: int = 150) -> np.ndarray:
    """Convert figure to numpy array.

    Args:
        fig: Matplotlib figure
        dpi: Resolution

    Returns:
        RGB array of shape (height, width, 3)
    """
    import io

    import numpy as np
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)

    img = Image.open(buf)
    return np.array(img)[:, :, :3]  # Drop alpha channel if present


def figure_to_base64(fig: Figure, format: str = "png", dpi: int = 150) -> str:
    """Convert figure to base64-encoded string.

    Args:
        fig: Matplotlib figure
        format: Image format
        dpi: Resolution

    Returns:
        Base64-encoded string
    """
    import base64
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches="tight")
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


# =============================================================================
# Plotly Export (for interactive figures)
# =============================================================================


def save_plotly_figure(
    fig,  # plotly.graph_objects.Figure
    name: str,
    output_dir: Path | str | None = None,
    formats: Sequence[str] = ("html", "png"),
    width: int = 1200,
    height: int = 800,
    scale: float = 2.0,
) -> list[Path]:
    """Save Plotly figure in multiple formats.

    Args:
        fig: Plotly figure object
        name: Base filename
        output_dir: Output directory
        formats: Export formats ('html', 'png', 'svg', 'pdf')
        width: Image width in pixels
        height: Image height in pixels
        scale: Scale factor for raster export

    Returns:
        List of saved file paths
    """
    config = get_config()

    output_dir = config.output_dir if output_dir is None else Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for fmt in formats:
        filepath = output_dir / f"{name}.{fmt}"

        if fmt == "html":
            fig.write_html(filepath, include_plotlyjs=True)
        else:
            fig.write_image(
                filepath,
                format=fmt,
                width=width,
                height=height,
                scale=scale,
            )

        saved_paths.append(filepath)

    return saved_paths
