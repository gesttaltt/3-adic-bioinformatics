# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Matplotlib theme configurations for scientific and presentation figures.

This module provides pre-configured themes that can be applied to matplotlib
to create publication-quality and presentation-ready visualizations.

Themes:
    - scientific: Clean, minimal style for journal publications
    - pitch: Bold, high-contrast style for presentations
    - dark: Dark background for demos and digital displays
    - notebook: Optimized for Jupyter notebook viewing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import matplotlib.pyplot as plt

from ..config import FONT_FAMILY_SANS, FONT_SIZES, Context, Theme, get_config
from .palettes import NEUTRALS, SEMANTIC

# =============================================================================
# Theme Definitions
# =============================================================================


@dataclass
class ThemeConfig:
    """Configuration for a visualization theme.

    Attributes:
        name: Theme identifier
        font_family: List of font family preferences
        font_sizes: Dict of font sizes by element
        colors: Dict of theme colors
        rcparams: Additional matplotlib rcParams
    """

    name: str
    font_family: list[str] = field(default_factory=lambda: FONT_FAMILY_SANS.copy())
    font_sizes: dict[str, int] = field(default_factory=dict)
    colors: dict[str, str | bool] = field(default_factory=dict)
    rcparams: dict[str, Any] = field(default_factory=dict)

    def to_rcparams(self) -> dict[str, Any]:
        """Convert theme to matplotlib rcParams dictionary."""
        params = {
            # Font configuration
            "font.family": "sans-serif",
            "font.sans-serif": self.font_family,
            "font.size": self.font_sizes.get("label", 10),
            # Axes configuration
            "axes.titlesize": self.font_sizes.get("title", 12),
            "axes.labelsize": self.font_sizes.get("label", 10),
            "axes.titleweight": "bold",
            "axes.labelweight": "normal",
            "axes.facecolor": self.colors.get("axes_bg", "white"),
            "axes.edgecolor": self.colors.get("axes_edge", NEUTRALS["gray_600"]),
            "axes.linewidth": 0.8,
            "axes.grid": self.colors.get("show_grid", False),
            "axes.axisbelow": True,
            # Tick configuration
            "xtick.labelsize": self.font_sizes.get("tick", 9),
            "ytick.labelsize": self.font_sizes.get("tick", 9),
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.color": self.colors.get("tick", NEUTRALS["gray_600"]),
            "ytick.color": self.colors.get("tick", NEUTRALS["gray_600"]),
            # Legend configuration
            "legend.fontsize": self.font_sizes.get("legend", 9),
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.facecolor": self.colors.get("legend_bg", "white"),
            "legend.edgecolor": self.colors.get("legend_edge", NEUTRALS["gray_200"]),
            # Figure configuration
            "figure.facecolor": self.colors.get("figure_bg", "white"),
            "figure.edgecolor": self.colors.get("figure_edge", "white"),
            "figure.dpi": 100,
            "figure.autolayout": False,
            "figure.constrained_layout.use": True,
            # Saving configuration
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "savefig.facecolor": self.colors.get("figure_bg", "white"),
            "savefig.edgecolor": "none",
            # Grid configuration
            "grid.color": self.colors.get("grid", NEUTRALS["gray_100"]),
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            # Line configuration
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
            # Patch configuration (bars, etc.)
            "patch.linewidth": 0.5,
            "patch.edgecolor": self.colors.get("patch_edge", NEUTRALS["gray_400"]),
            # Text configuration
            "text.color": self.colors.get("text", NEUTRALS["gray_700"]),
            # Spine configuration
            "axes.spines.top": self.colors.get("spine_top", True),
            "axes.spines.right": self.colors.get("spine_right", True),
        }

        # Add any custom rcparams
        params.update(self.rcparams)

        return params


# =============================================================================
# Pre-defined Themes
# =============================================================================


def _create_scientific_theme(context: Context = Context.NOTEBOOK) -> ThemeConfig:
    """Create scientific publication theme."""
    sizes = FONT_SIZES[context.value]

    return ThemeConfig(
        name="scientific",
        font_family=FONT_FAMILY_SANS.copy(),
        font_sizes=sizes,
        colors={
            "axes_bg": "white",
            "axes_edge": NEUTRALS["gray_600"],
            "figure_bg": "white",
            "text": NEUTRALS["gray_700"],
            "tick": NEUTRALS["gray_600"],
            "grid": NEUTRALS["gray_200"],
            "legend_bg": "white",
            "legend_edge": NEUTRALS["gray_200"],
            "patch_edge": NEUTRALS["gray_400"],
            "spine_top": True,
            "spine_right": True,
            "show_grid": False,
        },
        rcparams={
            "axes.titleweight": "normal",
            "axes.linewidth": 0.5,
            "lines.linewidth": 1.0,
        },
    )


def _create_pitch_theme(context: Context = Context.TALK) -> ThemeConfig:
    """Create presentation/pitch theme with bold styling."""
    sizes = FONT_SIZES[context.value]

    return ThemeConfig(
        name="pitch",
        font_family=FONT_FAMILY_SANS.copy(),
        font_sizes=sizes,
        colors={
            "axes_bg": SEMANTIC.background,
            "axes_edge": NEUTRALS["gray_400"],
            "figure_bg": "white",
            "text": NEUTRALS["gray_700"],
            "tick": NEUTRALS["gray_500"],
            "grid": NEUTRALS["gray_200"],
            "legend_bg": "white",
            "legend_edge": NEUTRALS["gray_100"],
            "patch_edge": NEUTRALS["gray_300"],
            "spine_top": False,
            "spine_right": False,
            "show_grid": False,
        },
        rcparams={
            "axes.titleweight": "bold",
            "axes.linewidth": 1.2,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
        },
    )


def _create_dark_theme(context: Context = Context.TALK) -> ThemeConfig:
    """Create dark theme for demos and digital displays."""
    sizes = FONT_SIZES[context.value]

    return ThemeConfig(
        name="dark",
        font_family=FONT_FAMILY_SANS.copy(),
        font_sizes=sizes,
        colors={
            "axes_bg": "#1E1E1E",
            "axes_edge": "#444444",
            "figure_bg": "#121212",
            "text": "#E0E0E0",
            "tick": "#AAAAAA",
            "grid": "#333333",
            "legend_bg": "#1E1E1E",
            "legend_edge": "#444444",
            "patch_edge": "#555555",
            "spine_top": False,
            "spine_right": False,
            "show_grid": True,
        },
        rcparams={
            "axes.titleweight": "bold",
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.0,
            "savefig.facecolor": "#121212",
        },
    )


def _create_notebook_theme(context: Context = Context.NOTEBOOK) -> ThemeConfig:
    """Create theme optimized for Jupyter notebooks."""
    sizes = FONT_SIZES[context.value]

    return ThemeConfig(
        name="notebook",
        font_family=FONT_FAMILY_SANS.copy(),
        font_sizes=sizes,
        colors={
            "axes_bg": "white",
            "axes_edge": NEUTRALS["gray_300"],
            "figure_bg": "white",
            "text": NEUTRALS["gray_700"],
            "tick": NEUTRALS["gray_500"],
            "grid": NEUTRALS["gray_100"],
            "legend_bg": "white",
            "legend_edge": NEUTRALS["gray_100"],
            "patch_edge": NEUTRALS["gray_200"],
            "spine_top": False,
            "spine_right": False,
            "show_grid": True,
        },
        rcparams={
            "figure.dpi": 100,
            "savefig.dpi": 150,
        },
    )


# Theme registry
_THEMES: dict[Theme, ThemeConfig] = {}


def _get_theme_config(theme: Theme, context: Context) -> ThemeConfig:
    """Get or create theme configuration."""
    if theme == Theme.SCIENTIFIC:
        return _create_scientific_theme(context)
    elif theme == Theme.PITCH:
        return _create_pitch_theme(context)
    elif theme == Theme.DARK:
        return _create_dark_theme(context)
    elif theme == Theme.NOTEBOOK:
        return _create_notebook_theme(context)
    else:
        return _create_scientific_theme(context)


# =============================================================================
# Theme Application Functions
# =============================================================================


def apply_theme(
    theme: Theme | str | None = None,
    context: Context | str | None = None,
) -> None:
    """Apply a visualization theme to matplotlib.

    Args:
        theme: Theme to apply (scientific, pitch, dark, notebook)
        context: Size context (paper, notebook, talk, poster)
    """
    config = get_config()

    if theme is None:
        theme = config.theme
    elif isinstance(theme, str):
        theme = Theme(theme)

    if context is None:
        context = config.context
    elif isinstance(context, str):
        context = Context(context)

    theme_config = _get_theme_config(theme, context)
    rcparams = theme_config.to_rcparams()

    plt.rcParams.update(rcparams)


def reset_theme() -> None:
    """Reset matplotlib to default settings."""
    plt.rcdefaults()


def use_scientific_style() -> None:
    """Apply scientific publication style."""
    apply_theme(Theme.SCIENTIFIC, Context.PAPER)


def use_pitch_style() -> None:
    """Apply presentation/pitch style."""
    apply_theme(Theme.PITCH, Context.TALK)


def use_dark_style() -> None:
    """Apply dark theme for demos."""
    apply_theme(Theme.DARK, Context.TALK)


def use_notebook_style() -> None:
    """Apply notebook-optimized style."""
    apply_theme(Theme.NOTEBOOK, Context.NOTEBOOK)


# =============================================================================
# Context Manager for Temporary Theme
# =============================================================================


class theme_context:
    """Context manager for temporarily applying a theme.

    Usage:
        with theme_context(Theme.SCIENTIFIC, Context.PAPER):
            plt.plot(...)
            plt.savefig(...)
    """

    def __init__(
        self,
        theme: Theme | str = Theme.SCIENTIFIC,
        context: Context | str = Context.NOTEBOOK,
    ):
        self.theme = Theme(theme) if isinstance(theme, str) else theme
        self.context = Context(context) if isinstance(context, str) else context
        self._original_rcparams: dict[str, Any] = {}

    def __enter__(self) -> theme_context:
        # Store current rcParams
        self._original_rcparams = plt.rcParams.copy()
        # Apply new theme
        apply_theme(self.theme, self.context)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Restore original rcParams
        plt.rcParams.update(self._original_rcparams)


# =============================================================================
# Style Presets for Common Use Cases
# =============================================================================


def setup_publication_style(journal: str = "nature") -> None:
    """Configure matplotlib for specific journal requirements.

    Args:
        journal: Target journal ('nature', 'science', 'ieee', 'cell')
    """
    apply_theme(Theme.SCIENTIFIC, Context.PAPER)

    # Journal-specific overrides
    if journal.lower() == "nature":
        plt.rcParams.update(
            {
                "font.size": 7,
                "axes.titlesize": 8,
                "axes.labelsize": 7,
                "xtick.labelsize": 6,
                "ytick.labelsize": 6,
                "legend.fontsize": 6,
                "figure.figsize": (3.5, 2.5),
            }
        )
    elif journal.lower() == "science":
        plt.rcParams.update(
            {
                "font.size": 8,
                "axes.titlesize": 9,
                "axes.labelsize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "figure.figsize": (3.5, 2.625),
            }
        )
    elif journal.lower() == "ieee":
        plt.rcParams.update(
            {
                "font.size": 8,
                "axes.titlesize": 9,
                "axes.labelsize": 8,
                "xtick.labelsize": 7,
                "ytick.labelsize": 7,
                "legend.fontsize": 7,
                "figure.figsize": (3.5, 2.5),
            }
        )
    elif journal.lower() == "cell":
        plt.rcParams.update(
            {
                "font.size": 7,
                "axes.titlesize": 8,
                "axes.labelsize": 7,
                "xtick.labelsize": 6,
                "ytick.labelsize": 6,
                "legend.fontsize": 6,
                "figure.figsize": (3.4, 2.5),
            }
        )


def setup_poster_style() -> None:
    """Configure matplotlib for poster presentations."""
    apply_theme(Theme.PITCH, Context.POSTER)


def setup_slide_style() -> None:
    """Configure matplotlib for slide presentations."""
    apply_theme(Theme.PITCH, Context.TALK)
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
        }
    )
