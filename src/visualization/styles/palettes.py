# Copyright 2024-2025 AI Whisperers (https://github.com/Ai-Whisperers)
#
# Licensed under the PolyForm Noncommercial License 1.0.0
# See LICENSE file in the repository root for full license text.
#
# For commercial licensing inquiries: support@aiwhisperers.com

"""Color palettes for scientific and presentation visualizations.

This module provides carefully curated color palettes that are:
- Perceptually uniform
- Colorblind-friendly (WCAG 2.1 compliant where possible)
- Print-safe (readable in grayscale)
- Semantically meaningful for bioinformatics contexts
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgba

# =============================================================================
# Core Color Definitions
# =============================================================================

# Primary brand colors
PRIMARY = "#1976D2"  # Blue - primary accent
SECONDARY = "#424242"  # Dark gray - secondary elements
ACCENT = "#E91E63"  # Pink - highlights

# Neutral palette
NEUTRALS = {
    "white": "#FFFFFF",
    "off_white": "#FAFAFA",
    "light_gray": "#F5F5F5",
    "gray_100": "#E0E0E0",
    "gray_200": "#BDBDBD",
    "gray_300": "#9E9E9E",
    "gray_400": "#757575",
    "gray_500": "#616161",
    "gray_600": "#424242",
    "gray_700": "#212121",
    "black": "#000000",
}

# =============================================================================
# Semantic Color Palettes
# =============================================================================


@dataclass(frozen=True)
class SemanticPalette:
    """Semantic color palette for biological/medical contexts."""

    # Risk gradient (protective -> high risk)
    risk_protective: str = "#2196F3"  # Blue
    risk_low: str = "#4CAF50"  # Green
    risk_moderate: str = "#FF9800"  # Orange
    risk_high: str = "#D32F2F"  # Red

    # Safety gradient
    safe: str = "#4CAF50"  # Green
    partial: str = "#FF9800"  # Orange
    unsafe: str = "#D32F2F"  # Red

    # Biological pathways
    parasympathetic: str = "#3F51B5"  # Indigo
    sympathetic: str = "#F44336"  # Red
    regeneration: str = "#4CAF50"  # Green
    gut_barrier: str = "#9C27B0"  # Purple
    inflammation: str = "#FF5722"  # Deep orange

    # Immunology
    immunodominant: str = "#E91E63"  # Pink
    silent: str = "#9E9E9E"  # Gray

    # Goldilocks zones (self -> modified-self -> foreign)
    goldilocks_self: str = "#90CAF9"  # Light blue
    goldilocks_modified: str = "#FFEB3B"  # Yellow
    goldilocks_foreign: str = "#EF9A9A"  # Light red

    # HLA risk categories
    hla_high_risk: str = "#D32F2F"  # Red
    hla_moderate_risk: str = "#FF9800"  # Orange
    hla_neutral: str = "#9E9E9E"  # Gray
    hla_protective: str = "#2196F3"  # Blue

    # General UI
    primary: str = "#1976D2"
    secondary: str = "#424242"
    background: str = "#FAFAFA"
    surface: str = "#FFFFFF"
    grid: str = "#E0E0E0"
    text: str = "#212121"
    text_secondary: str = "#757575"
    error: str = "#D32F2F"
    warning: str = "#FF9800"
    success: str = "#4CAF50"
    info: str = "#2196F3"


# Global semantic palette instance
SEMANTIC = SemanticPalette()

# Legacy compatibility - dict version
PALETTE = {
    "risk_high": SEMANTIC.risk_high,
    "risk_moderate": SEMANTIC.risk_moderate,
    "risk_low": SEMANTIC.risk_low,
    "risk_protective": SEMANTIC.risk_protective,
    "safe": SEMANTIC.safe,
    "partial": SEMANTIC.partial,
    "unsafe": SEMANTIC.unsafe,
    "parasympathetic": SEMANTIC.parasympathetic,
    "sympathetic": SEMANTIC.sympathetic,
    "regeneration": SEMANTIC.regeneration,
    "gut_barrier": SEMANTIC.gut_barrier,
    "inflammation": SEMANTIC.inflammation,
    "immunodominant": SEMANTIC.immunodominant,
    "silent": SEMANTIC.silent,
    "goldilocks_low": SEMANTIC.goldilocks_self,
    "goldilocks_zone": SEMANTIC.goldilocks_modified,
    "goldilocks_high": SEMANTIC.goldilocks_foreign,
    "primary": SEMANTIC.primary,
    "secondary": SEMANTIC.secondary,
    "background": SEMANTIC.background,
    "grid": SEMANTIC.grid,
    "text": SEMANTIC.text,
    "text_light": SEMANTIC.text_secondary,
}

# =============================================================================
# Categorical Color Palettes
# =============================================================================

# Paul Tol's colorblind-safe qualitative palette (vibrant)
TOLVIBRANT = [
    "#EE7733",  # Orange
    "#0077BB",  # Blue
    "#33BBEE",  # Cyan
    "#EE3377",  # Magenta
    "#CC3311",  # Red
    "#009988",  # Teal
    "#BBBBBB",  # Grey
]

# Paul Tol's muted palette (for dense plots)
TOLMUTED = [
    "#332288",  # Indigo
    "#88CCEE",  # Cyan
    "#44AA99",  # Teal
    "#117733",  # Green
    "#999933",  # Olive
    "#DDCC77",  # Sand
    "#CC6677",  # Rose
    "#882255",  # Wine
    "#AA4499",  # Purple
]

# Tableau 10 (widely used, colorblind-friendly)
TABLEAU10 = [
    "#4E79A7",  # Blue
    "#F28E2B",  # Orange
    "#E15759",  # Red
    "#76B7B2",  # Teal
    "#59A14F",  # Green
    "#EDC948",  # Yellow
    "#B07AA1",  # Purple
    "#FF9DA7",  # Pink
    "#9C755F",  # Brown
    "#BAB0AC",  # Gray
]

# Colorblind-safe diverging (for heatmaps)
COLORBLIND_DIVERGING = [
    "#313695",  # Dark blue
    "#4575B4",
    "#74ADD1",
    "#ABD9E9",
    "#E0F3F8",
    "#FFFFBF",  # Yellow (center)
    "#FEE090",
    "#FDAE61",
    "#F46D43",
    "#D73027",  # Dark red
]

# VAE/ML specific colors
VAE_COLORS = {
    "encoder_a": "#1976D2",  # Blue
    "encoder_b": "#7B1FA2",  # Purple
    "decoder_a": "#388E3C",  # Green
    "decoder_b": "#F57C00",  # Orange
    "latent": "#C2185B",  # Pink
    "reconstruction": "#00796B",  # Teal
    "kl_divergence": "#5D4037",  # Brown
}

# Protein/structure colors
STRUCTURE_COLORS = {
    "helix": "#E91E63",  # Pink
    "sheet": "#2196F3",  # Blue
    "coil": "#9E9E9E",  # Gray
    "turn": "#4CAF50",  # Green
    "glycan": "#FF9800",  # Orange
    "disulfide": "#FFEB3B",  # Yellow
}


# =============================================================================
# Colormap Functions
# =============================================================================


def get_risk_cmap(n_colors: int = 256) -> LinearSegmentedColormap:
    """Create diverging colormap from protective (blue) to high risk (red).

    Args:
        n_colors: Number of color levels

    Returns:
        LinearSegmentedColormap for risk visualization
    """
    colors = [
        SEMANTIC.risk_protective,
        SEMANTIC.risk_low,
        SEMANTIC.goldilocks_modified,
        SEMANTIC.risk_moderate,
        SEMANTIC.risk_high,
    ]
    return LinearSegmentedColormap.from_list("risk", colors, N=n_colors)


def get_safety_cmap(n_colors: int = 256) -> LinearSegmentedColormap:
    """Create colormap for safety metrics (green=safe, red=unsafe).

    Args:
        n_colors: Number of color levels

    Returns:
        LinearSegmentedColormap for safety visualization
    """
    colors = [SEMANTIC.unsafe, SEMANTIC.partial, SEMANTIC.safe]
    return LinearSegmentedColormap.from_list("safety", colors, N=n_colors)


def get_goldilocks_cmap(n_colors: int = 256) -> LinearSegmentedColormap:
    """Create colormap for Goldilocks zones (self -> modified-self -> foreign).

    Args:
        n_colors: Number of color levels

    Returns:
        LinearSegmentedColormap for Goldilocks zone visualization
    """
    colors = [
        SEMANTIC.goldilocks_self,
        SEMANTIC.goldilocks_modified,
        SEMANTIC.goldilocks_foreign,
    ]
    return LinearSegmentedColormap.from_list("goldilocks", colors, N=n_colors)


def get_diverging_cmap(
    low: str = "#313695",
    mid: str = "#FFFFBF",
    high: str = "#D73027",
    n_colors: int = 256,
) -> LinearSegmentedColormap:
    """Create custom diverging colormap.

    Args:
        low: Color for low values
        mid: Color for middle values
        high: Color for high values
        n_colors: Number of color levels

    Returns:
        LinearSegmentedColormap
    """
    return LinearSegmentedColormap.from_list("diverging", [low, mid, high], N=n_colors)


def get_sequential_cmap(
    colors: Sequence[str] | None = None,
    name: str = "sequential",
    n_colors: int = 256,
) -> LinearSegmentedColormap:
    """Create sequential colormap from list of colors.

    Args:
        colors: List of hex colors (defaults to viridis-like)
        name: Colormap name
        n_colors: Number of color levels

    Returns:
        LinearSegmentedColormap
    """
    if colors is None:
        # Default: scientific blue-green-yellow
        colors = ["#0D0887", "#5B02A3", "#9A179B", "#CB4678", "#EB7852", "#FDB32F", "#EFF821"]
    return LinearSegmentedColormap.from_list(name, colors, N=n_colors)


def get_categorical_cmap(
    palette: str = "tolvibrant",
    n_colors: int | None = None,
) -> ListedColormap:
    """Get categorical colormap by name.

    Args:
        palette: Palette name ('tolvibrant', 'tolmuted', 'tableau10')
        n_colors: Number of colors to use (cycles if exceeds palette size)

    Returns:
        ListedColormap for categorical data
    """
    palettes = {
        "tolvibrant": TOLVIBRANT,
        "tolmuted": TOLMUTED,
        "tableau10": TABLEAU10,
    }

    colors = palettes.get(palette.lower(), TOLVIBRANT)

    if n_colors is not None:
        # Cycle colors if needed
        colors = [colors[i % len(colors)] for i in range(n_colors)]

    return ListedColormap(colors, name=palette)


# =============================================================================
# Color Utility Functions
# =============================================================================


def lighten(color: str, factor: float = 0.3) -> str:
    """Lighten a color by mixing with white.

    Args:
        color: Hex color string
        factor: Lightening factor (0-1)

    Returns:
        Lightened hex color
    """
    rgba = to_rgba(color)
    lightened = [c + (1 - c) * factor for c in rgba[:3]]
    return f"#{int(lightened[0] * 255):02x}{int(lightened[1] * 255):02x}{int(lightened[2] * 255):02x}"


def darken(color: str, factor: float = 0.3) -> str:
    """Darken a color by mixing with black.

    Args:
        color: Hex color string
        factor: Darkening factor (0-1)

    Returns:
        Darkened hex color
    """
    rgba = to_rgba(color)
    darkened = [c * (1 - factor) for c in rgba[:3]]
    return f"#{int(darkened[0] * 255):02x}{int(darkened[1] * 255):02x}{int(darkened[2] * 255):02x}"


def with_alpha(color: str, alpha: float = 0.5) -> tuple[float, float, float, float]:
    """Add alpha channel to color.

    Args:
        color: Hex color string
        alpha: Alpha value (0-1)

    Returns:
        RGBA tuple
    """
    rgba = to_rgba(color)
    return (rgba[0], rgba[1], rgba[2], alpha)


def color_gradient(
    color1: str,
    color2: str,
    n_steps: int = 10,
) -> list[str]:
    """Create gradient between two colors.

    Args:
        color1: Starting hex color
        color2: Ending hex color
        n_steps: Number of gradient steps

    Returns:
        List of hex colors
    """
    rgba1 = np.array(to_rgba(color1))
    rgba2 = np.array(to_rgba(color2))

    gradient = []
    for i in range(n_steps):
        t = i / (n_steps - 1) if n_steps > 1 else 0
        rgba = rgba1 + t * (rgba2 - rgba1)
        hex_color = f"#{int(rgba[0] * 255):02x}{int(rgba[1] * 255):02x}{int(rgba[2] * 255):02x}"
        gradient.append(hex_color)

    return gradient


# =============================================================================
# Register Custom Colormaps with Matplotlib
# =============================================================================


def register_colormaps() -> None:
    """Register all custom colormaps with matplotlib."""
    cmaps = {
        "risk": get_risk_cmap(),
        "safety": get_safety_cmap(),
        "goldilocks": get_goldilocks_cmap(),
        "tolvibrant": get_categorical_cmap("tolvibrant"),
        "tolmuted": get_categorical_cmap("tolmuted"),
    }

    for name, cmap in cmaps.items():
        try:
            plt.colormaps.register(cmap, name=name)
        except ValueError:
            # Already registered
            pass


# Register on import
register_colormaps()
